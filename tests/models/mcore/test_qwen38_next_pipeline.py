# SPDX-License-Identifier: Apache-2.0
"""Opt-in PP2 numerical gate with real dynamic P2P, not a construction-only check.

Run under two torchrun workers with RUN_QWEN38_PIPELINE_TESTS=1 and
QWEN38_TINY_EXPORT_DIR pointing to the original four-layer GPU export.
Set QWEN38_PIPELINE_VPP=2 to exercise interleaved model chunks as well.
VPP defaults to the production overlap-P2P path; synchronous VPP is guarded.
The full trainer smoke separately covers optimizer/checkpoint/adapter reload.
"""

import os
import traceback
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_PIPELINE_TESTS") != "1", reason="explicit two-GPU pipeline opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def pipeline_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tensor_random
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    assert int(os.environ.get("WORLD_SIZE", "0")) == 2, "PP2 gate requires two ranks"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.025, 3 * 1024**3 / total))
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
    try:
        headroom = torch.tensor(int(free >= 8 * 1024**3), device=device)
        torch.distributed.all_reduce(headroom, op=torch.distributed.ReduceOp.MIN)
        if not headroom.item():
            pytest.skip("Every GPU needs 8 GiB headroom; never evict another job")
        vpp = int(os.environ.get("QWEN38_PIPELINE_VPP", "1"))
        assert vpp in (1, 2)
        parallel_state.initialize_model_parallel(
            pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=vpp if vpp > 1 else None
        )
        model_parallel_cuda_manual_seed(123)
        original_backward = tensor_random.CheckpointFunction.backward
        apply_patch_megatron_recomputation_backward()
        try:
            yield vpp
        finally:
            tensor_random.CheckpointFunction.backward = staticmethod(original_backward)
    finally:
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_dynamic_hc_pipeline_matches_unpartitioned_logits(monkeypatch, pipeline_context):
    try:
        _run_pipeline_case(monkeypatch, pipeline_context)
    except Exception:
        # A peer may still be in NCCL when one rank raises. Print before the
        # fixture's CUDA teardown, which can otherwise hide the first failure.
        traceback.print_exc()
        raise


def _run_pipeline_case(monkeypatch, pipeline_context):
    from megatron.bridge.peft.lora import LoRA
    from megatron.core import parallel_state
    from megatron.core.enums import ModelType
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.module import Float16Module
    from safetensors.torch import load_file

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.ops.ple import current_ple_batch

    vpp = pipeline_context
    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    bridge = AutoBridge.from_hf_pretrained(fixture / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert provider.num_layers == 4 and provider.hidden_size == 128, "Never run this gate on the full model"
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 2
    provider.virtual_pipeline_model_parallel_size = vpp if vpp > 1 else None
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.variable_seq_lengths = True
    provider.batch_p2p_comm = os.environ.get("QWEN38_PIPELINE_BATCH_P2P", "0") == "1"
    provider.overlap_p2p_comm = os.environ.get("QWEN38_PIPELINE_OVERLAP", "1" if vpp > 1 else "0") == "1"
    provider.moe_router_load_balancing_type = "none"
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_permute_fusion = False
    provider.language_max_sequence_length = 256
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.gradient_accumulation_fusion = False
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    rank = parallel_state.get_pipeline_model_parallel_rank()
    models = [
        provider.provide(
            pre_process=rank == 0 and chunk == 0,
            post_process=rank == 1 and chunk == vpp - 1,
            vp_stage=chunk if vpp > 1 else None,
        ).cuda()
        for chunk in range(vpp)
    ]
    for chunk, model in enumerate(models):
        # Production get_model sets this wrapper attribute before scheduling.
        model.model_type = ModelType.encoder_or_decoder
        if vpp > 1:
            assert model.vp_stage == chunk
        expected_layer_ids = [rank + 2 * chunk + 1] if vpp > 1 else [2 * rank + 1, 2 * rank + 2]
        assert [layer.layer_number for layer in model.language_model.decoder.layers] == expected_layer_ids
    bridge.load_hf_weights(models)
    wrapped = [Float16Module(provider, model).eval() for model in models]

    shapes = []
    original = P2PCommunicator._communicate_shapes

    def record_shapes(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        shapes.extend(tuple(shape) for shape in result if any(shape))
        return result

    monkeypatch.setattr(P2PCommunicator, "_communicate_shapes", record_shapes)
    reference = load_file(str(fixture / "reference.safetensors"))
    lengths = [16, 13, 7, 11]
    assert reference["input_ids"].shape[-1] >= max(lengths)

    observed = []

    def make_inputs(batch_index, length, training):
        ids = reference["input_ids"][:, :length].cuda()
        if training:
            # Distinct token streams and packed boundaries expose stale PLE
            # contexts across warmup, 1F1B and pipeline cooldown.
            ids = (ids + 7 * batch_index) % 200 + 3
        positions = torch.arange(length, device="cuda").reshape(1, 1, -1).repeat(3, 1, 1)
        bounds = [0, length // 2, length] if training else [0, length]
        cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
        if training:
            positions = torch.cat(
                [torch.arange(hi - lo, device="cuda") for lo, hi in zip(bounds, bounds[1:], strict=False)]
            )
            positions = positions.reshape(1, 1, -1).repeat(3, 1, 1)
        packed = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=length, max_seqlen_kv=length
        )
        return dict(input_ids=ids, position_ids=positions, attention_mask=None, packed_seq_params=packed)

    # No synchronous host transfers inside an interleaved forward/loss call:
    # another communication stream may be waiting for this rank to progress.
    inputs = {
        training: [make_inputs(index, length, training) for index, length in enumerate(lengths)]
        for training in (False, True)
    }
    torch.cuda.synchronize()

    def forward_step(iterator, module):
        batch_index, _length = next(iterator)
        output = module(**inputs[module.training][batch_index])

        def collect(value, non_loss_data=False):
            if non_loss_data:
                return value.detach().float()
            observed.append(value.detach().float())
            loss = value.float().square().mean()
            return loss, {"loss": loss.detach()}

        return output, collect

    with torch.no_grad():
        outputs = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=[iter(enumerate(lengths)) for _ in models] if vpp > 1 else iter(enumerate(lengths)),
            model=wrapped,
            num_microbatches=len(lengths),
            seq_length=max(lengths),
            micro_batch_size=1,
            forward_only=True,
            collect_non_loss_data=True,
        )
    if rank == 1:
        assert Counter(shapes) == Counter({(length, 1, 256): vpp for length in lengths}), shapes
        assert len(outputs) == len(lengths)
        for length, output in zip(lengths, outputs, strict=True):
            expected = reference["base_logits"][:, :length].float()
            assert output.shape == expected.shape
            gap = (output.cpu().log_softmax(-1) - expected.log_softmax(-1)).abs()
            assert bool(gap.isfinite().all()) and gap.mean() < 0.005 and gap.max() < 0.05
            print(
                f"QWEN38_PP2_VPP={vpp} LENGTH={length} GAP_MEAN={gap.mean().item():.8f} GAP_MAX={gap.max().item():.8f}"
            )

    peft = LoRA(
        target_modules=[
            "language_model.decoder.layers.*.self_attention.in_proj",
            "language_model.decoder.layers.*.self_attention.out_proj",
            "language_model.decoder.layers.*.self_attention.linear_qkv",
            "language_model.decoder.layers.*.self_attention.linear_proj",
            "language_model.decoder.layers.*.mlp.*.linear_fc1",
            "language_model.decoder.layers.*.mlp.*.linear_fc2",
        ],
        dim=16,
        alpha=32,
        dropout=0.0,
    )
    models = peft(models)
    peft.set_params_to_save(models)
    trainable = {
        f"chunk{chunk}/{name}": p
        for chunk, model in enumerate(models)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    assert trainable and all("adapter" in name for name in trainable)
    for module in wrapped:
        module.train()
    with torch.no_grad():
        for name, param in trainable.items():
            if "linear_out" in name:
                param.normal_(std=0.01)

    baseline_grads, baseline_logits = None, None
    for recompute in (False, True):
        print(f"QWEN38_PP2_VPP={vpp} BACKWARD_BEGIN RANK={rank} RECOMPUTE={recompute}", flush=True)
        provider.recompute_granularity = "full" if recompute else None
        observed.clear()
        for model in models:
            model.zero_grad(set_to_none=True)
        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=[iter(enumerate(lengths)) for _ in models] if vpp > 1 else iter(enumerate(lengths)),
            model=wrapped,
            num_microbatches=len(lengths),
            seq_length=max(lengths),
            micro_batch_size=1,
            forward_only=False,
        )
        grads = {name: p.grad.detach().clone() for name, p in trainable.items() if p.grad is not None}
        assert grads and all(bool(grad.isfinite().all()) for grad in grads.values())
        assert any(bool((grad != 0).any()) for grad in grads.values())
        assert all(not getattr(module, "_ple_recompute_fifo", []) for model in models for module in model.modules())
        with pytest.raises(RuntimeError, match="no n-gram ids published"):
            current_ple_batch()
        if not recompute:
            baseline_grads, baseline_logits = grads, list(observed)
        else:
            assert grads.keys() == baseline_grads.keys()
            for name, grad in grads.items():
                torch.testing.assert_close(grad, baseline_grads[name], rtol=0.02, atol=2e-5, msg=name)
            assert len(observed) == len(baseline_logits)
            for actual, expected in zip(observed, baseline_logits, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"QWEN38_PP2_VPP={vpp} LORA_RECOMPUTE_PASSED RANK={rank} GRAD_TENSORS={len(grads)}")
