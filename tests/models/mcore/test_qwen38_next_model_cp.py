# SPDX-License-Identifier: Apache-2.0
"""Separate-process CP1/CP2 full random-model packing/DDP/LoRA gate.

Run CP1 first, then CP2 with QWEN38_MODEL_CP_REFERENCE=<CP1 output directory>.
Both need RUN_QWEN38_MODEL_CP_TESTS=1, QWEN38_TINY_EXPORT_DIR=<original fixture>
and a fresh QWEN38_MODEL_CP_OUTPUT. Exported artifacts stay outside the repo.
Optional QWEN38_MODEL_TP=2 / QWEN38_MODEL_EP=2 uses two/four workers for
CP1/CP2, comparing each TP shard with its matching independent CP1 baseline.
The production provider guard is never bypassed or monkey-patched here.
"""

import os
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_MODEL_CP_TESTS") != "1", reason="explicit full tiny-model CP opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def model_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tensor_random
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    size = int(os.environ.get("WORLD_SIZE", "0"))
    tp = int(os.environ.get("QWEN38_MODEL_TP", "1"))
    ep = int(os.environ.get("QWEN38_MODEL_EP", str(tp)))
    assert tp in (1, 2) and ep == tp and size in (tp, 2 * tp)
    cp = size // tp
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.025, 3 * 1024**3 / total))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
    original_backward = tensor_random.CheckpointFunction.backward
    try:
        enough = torch.tensor(int(free >= 8 * 1024**3), device=device)
        torch.distributed.all_reduce(enough, op=torch.distributed.ReduceOp.MIN)
        if not enough.item():
            pytest.skip("Every GPU needs 8 GiB free; never evict another job")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=1,
            context_parallel_size=cp,
        )
        model_parallel_cuda_manual_seed(123)
        apply_patch_megatron_recomputation_backward()
        yield tp, ep, cp
    finally:
        tensor_random.CheckpointFunction.backward = staticmethod(original_backward)
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_complete_model_packing_ddp_lora_recompute_and_export(model_context):
    try:
        _run_model_case(*model_context)
    except Exception:
        traceback.print_exc()
        raise


def _run_model_case(tp_size, ep_size, cp_size):
    from megatron.bridge.peft.lora import LoRA
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    from megatron.core.transformer.module import Float16Module
    from safetensors.torch import load_file, save_file

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.model_forward import gptmodel_forward_model_engine
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.ops.ple import current_ple_batch

    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    output_dir = Path(os.environ["QWEN38_MODEL_CP_OUTPUT"])
    rank = torch.distributed.get_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    comparison_name = "comparison.safetensors" if tp_size == 1 else f"comparison-tp{tp_rank}.safetensors"
    assert not output_dir.exists(), "Use a fresh output directory; never overwrite prior evidence"
    baseline = None
    if cp_size > 1:
        baseline = load_file(str(Path(os.environ["QWEN38_MODEL_CP_REFERENCE"]) / comparison_name))
    bridge = AutoBridge.from_hf_pretrained(fixture / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    config = bridge.to_megatron_provider(load_weights=False)
    assert config.num_layers == 4 and config.hidden_size == 128
    config.context_parallel_size = cp_size
    config.sequence_parallel = tp_size > 1
    config.tensor_model_parallel_size = tp_size
    config.expert_model_parallel_size = ep_size
    config.expert_tensor_parallel_size = 1
    config.moe_router_load_balancing_type = "none"
    config.moe_token_dispatcher_type = "alltoall"
    config.moe_aux_loss_coeff = 0.0
    config.moe_permute_fusion = False
    config.params_dtype = torch.bfloat16
    config.bf16 = True
    config.gradient_accumulation_fusion = True
    # The scalar objective below is already normalized globally. Sum CP's
    # local gradient contributions once, not an additional CP average. This
    # gate does not substitute for the trainer's token-count/loss contract.
    config.calculate_per_token_loss = True
    config.recompute_granularity = "full"
    config.recompute_method = "uniform"
    config.recompute_num_layers = 1
    config.finalize()
    config._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    model = config.provide(pre_process=True, post_process=True).cuda()
    bridge.load_hf_weights([model])
    mixed = Float16Module(config, model).eval()
    docs_by_batch = []
    for batch_index, lengths in enumerate(([13, 7, 23], [19, 9])):
        docs = [
            (torch.arange(n, device="cuda") + 17 * doc + 11 * batch_index) % 200 + 3 for doc, n in enumerate(lengths)
        ]
        for ids in docs:
            ids[4::11] = config.qwen3_8_next_eos_token_id
        docs_by_batch.append(docs)

    def forward(module, docs):
        nested = torch.nested.nested_tensor(docs, layout=torch.jagged)
        output = gptmodel_forward_model_engine(module, nested, multi_modal_inputs={}, vision_model=True, pad_token_id=0)
        assert [v.shape[0] for v in output.unbind()] == [v.numel() for v in docs]
        assert output.shape[-1] * tp_size == 256
        # Gather autograd splits the vocabulary gradient back to its owner;
        # the full-vocabulary mean below is already normalized exactly once.
        return gather_from_tensor_model_parallel_region(output.values(), group=config._pg_collection.tp)

    recorded = {}

    def record_or_compare(name, value, *, gradient=False, exact=False):
        detached = value.detach().float().cpu()
        assert bool(detached.isfinite().all())
        if baseline is not None:
            expected = baseline[name]
            if exact:
                torch.testing.assert_close(detached, expected, rtol=0, atol=0)
            elif gradient:
                torch.testing.assert_close(detached, expected, rtol=0.03, atol=2e-3)
                if expected.norm() > 1e-10:
                    relative = (detached - expected).norm() / expected.norm()
                    assert relative < 0.03, (name, relative.item())
                else:
                    assert detached.abs().max() < 1e-7
            else:
                gap = (detached.log_softmax(-1) - expected.log_softmax(-1)).abs()
                print(
                    f"QWEN38_MODEL_CP TP={tp_size} EP={ep_size} CP={cp_size} NAME={name} "
                    f"RANK={rank} GAP_MEAN={gap.mean():.9f} GAP_MAX={gap.max():.9f}",
                    flush=True,
                )
                assert gap.mean() < 0.005 and gap.max() < 0.05
        recorded[name] = detached.contiguous()

    with torch.no_grad():
        for index, docs in enumerate(docs_by_batch):
            record_or_compare(f"base.{index}", forward(mixed, docs))
    peft = LoRA(
        dim=16,
        alpha=32,
        dropout=0.0,
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
    peft.set_params_to_save([model])
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("adapter" in name for name in trainable)
    frozen = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not p.requires_grad}
    with torch.no_grad():
        for index, (name, parameter) in enumerate(sorted(trainable.items())):
            values = torch.randn(parameter.shape, generator=torch.Generator().manual_seed(1000 + index)) * 0.02
            parameter.copy_(values.to(parameter.dtype).to(parameter.device))
            record_or_compare(f"initial.{name}", parameter, exact=True)
    ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
        module=model,
        pg_collection=config._pg_collection,
    )
    ddp.train()
    normal_grads = None
    for recompute in (False, True):
        config.recompute_granularity = "full" if recompute else None
        ddp.zero_grad_buffer()
        for index, docs in enumerate(docs_by_batch):
            output = forward(ddp, docs)
            record_or_compare(f"adapter.{int(recompute)}.{index}", output)
            (output.float().square().mean() / len(docs_by_batch)).backward()
            with pytest.raises(RuntimeError, match="no n-gram ids published"):
                current_ple_batch()
            assert all(not getattr(m, "_ple_recompute_fifo", []) for m in model.modules())
        finalize_model_grads([ddp], pg_collection=config._pg_collection)
        grads = {name: p.main_grad.detach().clone() for name, p in trainable.items()}
        for family in (".self_attention.", ".mlp.experts.", ".mlp.shared_experts."):
            assert any(family in name and bool((g != 0).any()) for name, g in grads.items()), family
        for name, value in grads.items():
            record_or_compare(f"grad.{int(recompute)}.{name}", value, gradient=True)
            if normal_grads is not None:
                torch.testing.assert_close(value, normal_grads[name], rtol=0.02, atol=2e-5)
        normal_grads = grads
        print(
            f"QWEN38_MODEL_CP_DDP_GRADS TP={tp_size} EP={ep_size} CP={cp_size} RANK={rank} "
            f"RECOMPUTE={recompute} TENSORS={len(grads)}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-2)
    for parameter in trainable.values():
        parameter.grad = parameter.main_grad.to(parameter.dtype)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        for index, docs in enumerate(docs_by_batch):
            updated = forward(mixed, docs)
            record_or_compare(f"updated.{index}", updated)
            assert not torch.equal(updated.float().cpu(), recorded[f"adapter.1.{index}"])
            with peft.disable_adapter([model]):
                torch.testing.assert_close(
                    forward(mixed, docs).float().cpu(), recorded[f"base.{index}"], rtol=0, atol=0
                )
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter.cpu(), frozen[name], rtol=0, atol=0)
    adapters = list(bridge._model_bridge.stream_adapter_weights_megatron_to_hf([ddp], cpu=True, show_progress=False))
    tensors = {item.param_name: item.weight.detach().cpu().contiguous().clone() for item in adapters}
    assert len(tensors) == len(adapters) == 78 and all(bool(t.isfinite().all()) for t in tensors.values())
    # Separate plain-text fixture for the existing independent vLLM reload gate.
    ids = torch.arange(3, 19, device="cuda", dtype=torch.long)
    with torch.no_grad():
        tuned_logits = forward(mixed, [ids])
        with peft.disable_adapter([model]):
            base_logits = forward(mixed, [ids])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "model").mkdir()
        (output_dir / "adapter").mkdir()
        bridge.hf_pretrained.config.save_pretrained(output_dir / "model")
        original = load_file(str(fixture / "model/model.safetensors"))
        save_file(original, str(output_dir / "model/model.safetensors"))
        save_file(tensors, str(output_dir / "raw_adapter.safetensors"))
        save_file(
            {
                "input_ids": ids[None].cpu(),
                "base_logits": base_logits[None].cpu(),
                "adapter_logits": tuned_logits[None].cpu(),
                "lora_rank": torch.tensor(16),
                "lora_alpha": torch.tensor(32),
            },
            str(output_dir / "reference.safetensors"),
        )
    torch.distributed.barrier()
    if cp_rank == 0:
        save_file(recorded, str(output_dir / comparison_name))
    torch.distributed.barrier()
    print(f"QWEN38_MODEL_CP_EXPORT_PASSED TP={tp_size} EP={ep_size} CP={cp_size} RANK={rank} TENSORS=78", flush=True)
