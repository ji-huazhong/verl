# SPDX-License-Identifier: Apache-2.0
"""Opt-in real HF import/export and TP/EP collectives on the tiny fixture.

First export the fixture with test_qwen38_next_gpu.py. Run this file under
torchrun with RUN_QWEN38_PARALLEL_TESTS=1 and QWEN38_TINY_EXPORT_DIR set.
QWEN38_TEST_TP/EP default to WORLD_SIZE; PP/CP/ETP remain one.
This is not full-checkpoint, trainer, optimizer-resume or vision acceptance.
"""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_PARALLEL_TESTS") != "1", reason="explicit distributed GPU opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def parallel_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        pytest.skip("Need 8 GiB free headroom; never evict another job")
    torch.cuda.set_per_process_memory_fraction(min(0.04, 5 * 1024**3 / total))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.distributed.init_process_group(
        "nccl", timeout=timedelta(seconds=120), device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    )
    world = torch.distributed.get_world_size()
    tp = int(os.environ.get("QWEN38_TEST_TP", world))
    ep = int(os.environ.get("QWEN38_TEST_EP", world))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp, expert_model_parallel_size=ep, expert_tensor_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    yield tp, ep
    torch.cuda.synchronize()
    print(f"QWEN38_PARALLEL_PEAK_ALLOCATED_MIB={torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


def test_hf_import_roundtrip_and_parallel_forward(parallel_context):
    from megatron.bridge.peft.lora import LoRA
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    from megatron.core.transformer.module import Float16Module
    from safetensors.torch import load_file, save_file

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.model_forward import gptmodel_forward_model_engine
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge

    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    tp, ep = parallel_context
    bridge = AutoBridge.from_hf_pretrained(fixture / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = tp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = tp > 1
    provider.moe_router_load_balancing_type = "none"
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_permute_fusion = False
    provider.language_max_sequence_length = 256
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    model = provider.provide(pre_process=True, post_process=True).cuda()
    # Match the production mixed-precision boundary before loading/exporting
    # BF16 fixture weights. The bare VL provider leaves some vision parameters
    # FP32 even when the language configuration is BF16.
    mixed_model = Float16Module(provider, model)
    bridge.load_hf_weights([model])

    original = load_file(str(fixture / "model" / "model.safetensors"))
    exported = {}
    for item in bridge.export_hf_weights([model], cpu=True, show_progress=False):
        assert item.param_name not in exported
        exported[item.param_name] = item.weight
        torch.testing.assert_close(item.weight, original[item.param_name], rtol=0, atol=0)
    # Frozen PLE storage and hash metadata load independently, never through
    # the trainable parameter mapping. Every other input weight must roundtrip.
    expected = {name for name in original if ".ple.ple_embedding." not in name}
    assert exported.keys() == expected, (expected - exported.keys(), exported.keys() - expected)
    print(f"QWEN38_HF_ROUNDTRIP_TENSORS={len(exported)} TP={tp} EP={ep}")

    reference = load_file(str(fixture / "reference.safetensors"))
    ids = reference["input_ids"].cuda()
    tokens = ids.shape[-1]
    positions = torch.arange(tokens, device="cuda").reshape(1, -1).repeat(3, 1, 1)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=tokens, max_seqlen_kv=tokens
    )
    model.eval()
    inputs = dict(input_ids=ids, position_ids=positions, attention_mask=None, packed_seq_params=packed)

    def full_logits(module):
        local = module(**inputs)
        assert local.shape[-1] * tp == reference["base_logits"].shape[-1]
        return gather_from_tensor_model_parallel_region(local, group=parallel_state.get_tensor_model_parallel_group())

    with torch.no_grad():
        # The VL provider intentionally returns vocab-parallel logits. Gather
        # the vocabulary before applying log_softmax, as the TP1 reference is
        # a full-vocabulary distribution, not independent shard distributions.
        logits = full_logits(mixed_model)
    assert logits.shape == reference["base_logits"].shape
    assert bool(logits.isfinite().all())
    gap = (logits.float().cpu().log_softmax(-1) - reference["base_logits"].float().log_softmax(-1)).abs()
    print(f"QWEN38_PARALLEL_LOGPROB_GAP_MEAN={gap.mean().item():.8f}")
    print(f"QWEN38_PARALLEL_LOGPROB_GAP_MAX={gap.max().item():.8f}")
    assert gap.mean() < 0.005 and gap.max() < 0.05

    # Exercise verl's actual nested-input/VL packing entry, not just direct
    # calls with hand-built PackedSeqParams. Unequal lengths require TP padding.
    documents = [ids[0, :13], ids[0, 5:12]]

    def engine_logits(docs):
        nested = torch.nested.nested_tensor(docs, layout=torch.jagged)
        outputs = gptmodel_forward_model_engine(model, nested, multi_modal_inputs={}, vision_model=True, pad_token_id=0)
        return [
            gather_from_tensor_model_parallel_region(value, group=parallel_state.get_tensor_model_parallel_group())
            for value in outputs.unbind()
        ]

    with torch.no_grad():
        packed_outputs = engine_logits(documents)
        for document, packed_output in zip(documents, packed_outputs, strict=True):
            standalone = engine_logits([document])[0]
            assert packed_output.shape == standalone.shape == (document.numel(), 256)
            gap = (packed_output.float().log_softmax(-1) - standalone.float().log_softmax(-1)).abs()
            assert bool(gap.isfinite().all()) and gap.mean() < 0.005 and gap.max() < 0.05
    print(f"QWEN38_VERL_PACKED_FORWARD_PASSED TP={tp} EP={ep}")

    # Use real Megatron DDP gradient synchronization, including TP/SP handling.
    # The small AdamW step below is a component check, not the verl optimizer or
    # GRPO schedule. Configure recompute before PEFT installs its input hooks.
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    peft = LoRA(
        dim=int(reference.get("lora_rank", torch.tensor(4))),
        alpha=int(reference.get("lora_alpha", torch.tensor(8))),
        target_modules=[
            "language_model.decoder.layers.*.self_attention.in_proj",
            "language_model.decoder.layers.*.self_attention.out_proj",
            "language_model.decoder.layers.*.self_attention.linear_qkv",
            "language_model.decoder.layers.*.self_attention.linear_proj",
            "language_model.decoder.layers.*.mlp.*.linear_fc1",
            "language_model.decoder.layers.*.mlp.*.linear_fc2",
        ],
    )
    model = peft([model])[0]
    assert mixed_model.module is model
    peft.set_params_to_save([model])
    frozen = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not p.requires_grad}
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("adapter" in name for name, _ in trainable)
    with torch.no_grad():
        torch.testing.assert_close(full_logits(mixed_model), logits, rtol=0, atol=0)
    ddp = DistributedDataParallel(
        config=provider,
        ddp_config=DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
        module=model,
        pg_collection=provider._pg_collection,
    )
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-2)
    ddp.zero_grad_buffer()
    ddp.train()
    # Each TP rank owns only its vocabulary shard, matching vocab-parallel loss
    # ownership; finalize_model_grads synchronizes the replicated parameters.
    loss = ddp(**inputs).float().square().mean() / tp
    assert bool(loss.isfinite())
    loss.backward()
    finalize_model_grads([ddp], pg_collection=provider._pg_collection)
    assert all(bool(p.main_grad.isfinite().all()) for _, p in trainable)
    for family in (".self_attention.", ".mlp.experts.", ".mlp.shared_experts."):
        assert any(family in name and bool((p.main_grad != 0).any()) for name, p in trainable), family
    for _, p in trainable:
        p.grad = p.main_grad.to(p.dtype)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        updated = full_logits(mixed_model)
        assert bool(updated.isfinite().all()) and not torch.equal(updated, logits)
        with peft.disable_adapter([model]):
            torch.testing.assert_close(full_logits(mixed_model), logits, rtol=0, atol=0)
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p.cpu(), frozen[name], rtol=0, atol=0)
    adapters = list(bridge._model_bridge.stream_adapter_weights_megatron_to_hf([ddp], cpu=True, show_progress=False))
    tensors = {entry.param_name: entry.weight.detach().contiguous().clone() for entry in adapters}
    assert len(tensors) == len(adapters) == 78
    assert all(bool(value.isfinite().all()) for value in tensors.values())
    print(f"QWEN38_PARALLEL_DDP_LORA_UPDATED_AND_EXPORTED={len(tensors)} TP={tp} EP={ep}")
    output_dir = os.environ.get("QWEN38_PARALLEL_EXPORT_DIR")
    if output_dir and torch.distributed.get_rank() == 0:
        destination = Path(output_dir)
        (destination / "model").mkdir(parents=True, exist_ok=False)
        (destination / "adapter").mkdir()
        bridge.hf_pretrained.config.save_pretrained(destination / "model")
        save_file(original, str(destination / "model" / "model.safetensors"))
        save_file(tensors, str(destination / "raw_adapter.safetensors"))
        save_file(
            {
                "input_ids": ids.cpu(),
                "base_logits": logits.cpu(),
                "adapter_logits": updated.cpu(),
                "lora_rank": torch.tensor(peft.dim),
                "lora_alpha": torch.tensor(peft.alpha),
            },
            str(destination / "reference.safetensors"),
        )
    torch.distributed.barrier()
