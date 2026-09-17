# SPDX-License-Identifier: Apache-2.0
"""Opt-in packed GDN CP2/CP4 numerical gate using Core's native all-to-all.

RUN_QWEN38_GDN_CP_TESTS=1 QWEN38_TINY_EXPORT_DIR=<tiny fixture>, then
torchrun --standalone --nproc-per-node=2 -m pytest -s -q <this file>.
This component test does not bypass or establish whole-model CP support.
"""

import copy
import os
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_GDN_CP_TESTS") != "1", reason="explicit multi-GPU GDN CP opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def cp_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tensor_random
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    size = int(os.environ.get("WORLD_SIZE", "0"))
    assert size in (2, 4)
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
        parallel_state.initialize_model_parallel(context_parallel_size=size)
        model_parallel_cuda_manual_seed(123)
        apply_patch_megatron_recomputation_backward()
        yield
    finally:
        tensor_random.CheckpointFunction.backward = staticmethod(original_backward)
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("long_document", [False, True])
def test_native_gdn_cp_matches_cp1_lora_and_recompute(long_document):
    try:
        _run_gdn_case(long_document)
    except Exception:
        traceback.print_exc()
        raise


def _run_gdn_case(long_document):
    from megatron.bridge.peft.lora import LoRA
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import checkpoint

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.ops.context_parallel import PackedContextParallelLayout
    from verl.models.mcore.qwen3_8_next.ops.gated_delta_net import Qwen38NextGatedDeltaNet

    groups = ProcessGroupCollection.use_mpu_process_groups()
    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    bridge = AutoBridge.from_hf_pretrained(fixture / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    config = bridge.to_megatron_provider(load_weights=False)
    assert config.num_layers == 4 and config.hidden_size == 128
    config.context_parallel_size = groups.cp.size()
    config.sequence_parallel = False
    config.params_dtype = torch.bfloat16
    config.bf16 = True
    config.gradient_accumulation_fusion = True
    config.finalize()
    spec = get_transformer_block_with_experimental_attention_variant_spec(config)
    submodules = copy.deepcopy(spec.layer_specs[0].submodules.self_attention.submodules)
    submodules.in_proj = TEColumnParallelLinear
    module = Qwen38NextGatedDeltaNet(config, submodules, layer_number=1, pg_collection=groups).cuda()
    ref_config = copy.copy(config)
    ref_config.context_parallel_size = 1
    ref_groups = copy.copy(groups)
    ref_groups.cp = parallel_state.get_tensor_model_parallel_group()
    reference = Qwen38NextGatedDeltaNet(ref_config, submodules, layer_number=1, pg_collection=ref_groups).cuda()
    peft = LoRA(target_modules=["in_proj", "out_proj"], dim=16, alpha=32, dropout=0.0)
    module, reference = peft([module])[0], peft([reference])[0]
    torch.manual_seed(47)
    trainable = {name: p for name, p in module.named_parameters() if p.requires_grad}
    assert len(trainable) == 4 and all("adapter" in name for name in trainable)
    with torch.no_grad():
        for name, parameter in trainable.items():
            if "linear_out" in name:
                parameter.normal_(std=0.03)
        for parameter in module.parameters():
            torch.distributed.broadcast(parameter, src=0, group=groups.cp)
        for method in ("named_parameters", "named_buffers"):
            actual, target = dict(getattr(module, method)()), dict(getattr(reference, method)())
            assert actual.keys() == target.keys()
            for name, value in actual.items():
                target[name].copy_(value)
    for component in (module, reference):
        component.train()
        for parameter in component.parameters():
            if parameter.requires_grad:
                parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
    lengths = [8, 24, 136 if long_document else 64]
    cu = torch.tensor([0, lengths[0], sum(lengths[:2]), sum(lengths)], dtype=torch.int32, device="cuda")
    layout = PackedContextParallelLayout(cu, groups.cp.size(), groups.cp.rank())
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths)
    )
    torch.manual_seed(61)
    states = torch.randn(layout.total, 1, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn_like(states) * 0.1
    reference_input = states.detach().clone().requires_grad_()
    expected, _ = reference(reference_input, attention_mask=None, packed_seq_params=packed)
    (expected.float() * weights.float()).sum().backward()
    expected_grads = {name: p.main_grad.clone() for name, p in reference.named_parameters() if p.requires_grad}
    for recompute in (False, True):
        module.zero_grad(set_to_none=True)
        for parameter in trainable.values():
            parameter.main_grad.zero_()
        local_input = layout.local(states).detach().requires_grad_()

        def forward(value):
            return module(value, attention_mask=None, packed_seq_params=packed)[0]

        actual = checkpoint(forward, False, local_input) if recompute else forward(local_input)
        gap = (actual.float() - layout.local(expected).float()).abs()
        print(
            f"QWEN38_GDN_CP CP={groups.cp.size()} RANK={groups.cp.rank()} LONG={long_document} "
            f"RECOMPUTE={recompute} GAP_MEAN={gap.mean().item():.9f} GAP_MAX={gap.max().item():.9f}",
            flush=True,
        )
        torch.testing.assert_close(actual, layout.local(expected), rtol=0.02, atol=5e-4)
        (actual.float() * layout.local(weights).float()).sum().backward()
        torch.testing.assert_close(local_input.grad, layout.local(reference_input.grad), rtol=0.03, atol=2e-3)
        expected_input = layout.local(reference_input.grad).float()
        assert (local_input.grad.float() - expected_input).norm() / expected_input.norm() < 0.02
        for name, parameter in trainable.items():
            gradient = parameter.main_grad
            assert bool(gradient.isfinite().all()) and bool((gradient != 0).any())
            torch.distributed.all_reduce(gradient, group=groups.cp)
            relative = (gradient - expected_grads[name]).norm() / expected_grads[name].norm()
            print(f"QWEN38_GDN_CP_GRAD NAME={name} REL_L2={relative.item():.9f}", flush=True)
            torch.testing.assert_close(gradient, expected_grads[name], rtol=0.03, atol=2e-3)
            assert relative < 0.02
    print(f"QWEN38_GDN_CP_LORA_PASSED CP={groups.cp.size()} RANK={groups.cp.rank()} LONG={long_document}", flush=True)
