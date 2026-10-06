# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Hybrid full-parameter, CPU optimizer and checkpoint gate using a tiny HF export.

RUN_QWEN38_PARALLEL_TESTS=1 QWEN38_TINY_EXPORT_DIR=<tiny-hf> torchrun
--standalone --nproc-per-node=4 -m pytest -s -q <this file>
QWEN38_TEST_TP/PP/EP/DP can select other topologies (world size is TP * PP * DP).
"""

import os
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_PARALLEL_TESTS") != "1", reason="explicit multi-GPU opt-in required"
)
TP = int(os.environ.get("QWEN38_TEST_TP", "2"))
PP = int(os.environ.get("QWEN38_TEST_PP", "2"))
EP = int(os.environ.get("QWEN38_TEST_EP", "2"))
DP = int(os.environ.get("QWEN38_TEST_DP", "1"))


@pytest.fixture(scope="module", autouse=True)
def distributed_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tensor_random

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    assert int(os.environ.get("WORLD_SIZE", "0")) == TP * PP * DP
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.10, 8 * 1024**3 / total))
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    original = tensor_random.CheckpointFunction.backward
    try:
        headroom = torch.tensor(int(free > 8 * 1024**3), device=device)
        torch.distributed.all_reduce(headroom, op=torch.distributed.ReduceOp.MIN)
        if not headroom.item():
            pytest.skip("Every GPU needs 8 GiB free; never evict another job")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=PP,
            expert_model_parallel_size=EP,
            expert_tensor_parallel_size=1,
        )
        assert parallel_state.get_data_parallel_world_size() == DP
        tensor_random.model_parallel_cuda_manual_seed(123)
        apply_patch_megatron_recomputation_backward()
        yield
    finally:
        tensor_random.CheckpointFunction.backward = staticmethod(original)
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_full_parameter_hybrid_import_recompute_update():
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.enums import ModelType
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    from megatron.core.transformer.module import Float16Module
    from safetensors.torch import load_file

    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing, save_dist_checkpointing

    checkpoint = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    bridge = AutoBridge.from_hf_pretrained(checkpoint, local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert provider.num_layers == 4 and provider.hidden_size == 128, "Never run this gate on the real model"
    provider.tensor_model_parallel_size = TP
    provider.pipeline_model_parallel_size = PP
    provider.expert_model_parallel_size = EP
    provider.expert_tensor_parallel_size = 1
    provider.qwen3_8_next_train_ple = os.environ.get("QWEN38_TEST_TRAIN_PLE", "1") == "1"
    provider.sequence_parallel = True
    provider.variable_seq_lengths = True
    provider.pipeline_dtype = torch.bfloat16
    provider.batch_p2p_comm = False
    provider.overlap_p2p_comm = False
    provider.moe_router_load_balancing_type = "none"
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_permute_fusion = False
    provider.language_max_sequence_length = 256
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.gradient_accumulation_fusion = os.environ.get("QWEN38_TEST_GRAD_FUSION", "0") == "1"
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    model = provider.provide(pre_process=pp_rank == 0, post_process=pp_rank == PP - 1).cuda()
    model.model_type = ModelType.encoder_or_decoder
    mixed = Float16Module(provider, model)
    bridge.load_hf_weights([model])

    original = load_file(str(checkpoint / "model.safetensors"))
    exported = set()
    for entry in bridge.export_hf_weights([model], cpu=True, show_progress=False):
        torch.testing.assert_close(entry.weight, original[entry.param_name], rtol=0, atol=0)
        exported.add(entry.param_name)
    immutable = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
    expected_export = {name for name in original if not name.endswith(immutable)}
    frozen_tables = {}
    if not provider.qwen3_8_next_train_ple:
        expected_export = {name for name in expected_export if "ngram_embedding.shard_" not in name}
        for module in model.modules():
            if hasattr(module, "table"):
                module.load_from_hf(str(checkpoint))
        frozen_tables = {
            name: (module.table, module.table.clone())
            for name, module in model.named_modules()
            if hasattr(module, "table")
        }
        assert not any("ngram_embedding.weight" in name for name, _ in model.named_parameters())
    assert exported == expected_export

    reference = load_file(str(checkpoint.parent / (checkpoint.name + "-reference.safetensors")))
    ids = reference["input_ids"].cuda()
    cu = torch.tensor([0, 8, 16], dtype=torch.int32, device="cuda")
    inputs = dict(
        input_ids=ids,
        position_ids=torch.arange(8, device="cuda").repeat(2).reshape(1, 1, 16).repeat(3, 1, 1),
        attention_mask=None,
        packed_seq_params=PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=8, max_seqlen_kv=8
        ),
    )
    observed = []

    def forward_step(iterator, module):
        output = module(**next(iterator))

        def loss_func(value, non_loss_data=False):
            if non_loss_data:
                return gather_from_tensor_model_parallel_region(value.detach())
            observed.append(value.detach().clone())
            loss = value.float().square().mean() / TP
            return loss, {"loss": loss.detach()}

        return output, loss_func

    def run(module, forward_only):
        return get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter([inputs] * 4),
            model=[module],
            num_microbatches=4,
            seq_length=16,
            micro_batch_size=1,
            forward_only=forward_only,
            collect_non_loss_data=forward_only,
        )

    mixed.eval()
    with torch.no_grad():
        results = run(mixed, True)
    if pp_rank == PP - 1:
        expected = reference["logits"].float().log_softmax(-1)
        for logits in results:
            gap = (logits.float().cpu().log_softmax(-1) - expected).abs()
            assert gap.mean() < 0.005 and gap.max() < 0.05
            print("QWEN38_HYBRID_LOGPROB_GAP", gap.mean().item(), gap.max().item(), flush=True)

    ddp = DistributedDataParallel(
        config=provider,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=True
        ),
        module=mixed,
        pg_collection=provider._pg_collection,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-2,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=True,
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=1.0,
            overlap_cpu_optimizer_d2h_h2d=True,
        ),
        [ddp],
        use_gloo_process_groups=False,
        pg_collection=provider._pg_collection,
    )
    provider.grad_scale_func = optimizer.scale_loss
    ddp.train()
    parameters = dict(model.named_parameters())
    assert all(p.requires_grad for p in parameters.values())
    baseline_grads = baseline_logits = None
    for recompute in (False, True):
        provider.recompute_granularity = "full" if recompute else None
        provider.recompute_method = "uniform" if recompute else None
        provider.recompute_num_layers = 1 if recompute else None
        optimizer.zero_grad()
        ddp.zero_grad_buffer()
        observed.clear()
        run(ddp, False)
        finalize_model_grads([ddp], pg_collection=provider._pg_collection)
        grads = {name: p.main_grad.detach().clone() for name, p in parameters.items()}
        assert all(bool(value.isfinite().all()) for value in grads.values())
        gdn_grads = {name: value for name, value in grads.items() if ".in_proj." in name}
        if gdn_grads:
            assert any(".qkvz.weight" in name for name in gdn_grads)
            assert any(".ba.weight" in name for name in gdn_grads)
            assert all(value.abs().sum() > 0 for value in gdn_grads.values())
        expert_grad = sum(value.abs().sum() for name, value in grads.items() if "mlp.experts" in name)
        torch.distributed.all_reduce(expert_grad, group=parallel_state.get_expert_model_parallel_group())
        assert expert_grad > 0
        if any("ngram_embedding.weight" in name for name in parameters):
            # Sparse lookups can leave individual TP shards untouched. In the
            # TP8 fixture the final shards even consist entirely of padding.
            ple_grad = sum(value.abs().sum() for name, value in grads.items() if "ngram_embedding.weight" in name)
            torch.distributed.all_reduce(ple_grad, group=parallel_state.get_tensor_model_parallel_group())
            assert ple_grad > 0
        assert all(not getattr(module, "_ple_recompute_fifo", []) for module in model.modules())
        if not recompute:
            baseline_grads, baseline_logits = grads, list(observed)
        else:
            for name, value in grads.items():
                torch.testing.assert_close(value, baseline_grads[name], rtol=0.02, atol=2e-5, msg=name)
            for value, baseline in zip(observed, baseline_logits, strict=True):
                torch.testing.assert_close(value, baseline, rtol=0, atol=0)
    before = {name: p.detach().clone() for name, p in parameters.items() if "ngram_embedding.weight" in name}
    successful, _, _ = optimizer.step()
    assert successful
    for name, value in before.items():
        changed = torch.tensor(int(not torch.equal(parameters[name], value)), device=value.device)
        torch.distributed.all_reduce(changed, group=parallel_state.get_tensor_model_parallel_group())
        assert changed > 0

    metadata = {
        "distrib_optim_sharding_type": "dp_reshardable",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
        "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True),
    }

    def sharded_state(is_loading=False):
        state = {"model": model.sharded_state_dict(metadata=metadata)}
        state["optimizer"] = optimizer.sharded_state_dict(state, is_loading=is_loading, metadata=metadata)
        return state

    resume_dirs = [None]
    if torch.distributed.get_rank() == 0:
        resume_dirs[0] = tempfile.mkdtemp(
            prefix=f"{checkpoint.name}-resume-tp{TP}-pp{PP}-ep{EP}-dp{DP}-", dir=checkpoint.parent
        )
    torch.distributed.broadcast_object_list(resume_dirs, src=0)
    resume_path = resume_dirs[0]
    # Process groups are runtime-only; content metadata must remain serializable.
    save_dist_checkpointing(
        sharded_state(), str(resume_path), content_metadata={k: v for k, v in metadata.items() if k != "dp_cp_group"}
    )
    after_first = {name: p.detach().cpu().clone() for name, p in parameters.items()}

    def inner_optimizer_snapshot():
        snapshot = {}
        for i, part in enumerate(getattr(optimizer, "chained_optimizers", [optimizer])):
            for j, inner in enumerate(part.optimizer.sub_optimizers):
                for k, group in enumerate(inner.param_groups):
                    for n, parameter in enumerate(group["params"]):
                        prefix = f"optimizer{i}/sub{j}/group{k}/param{n}"
                        snapshot[prefix] = parameter.detach().cpu().clone()
                        for key, value in inner.state[parameter].items():
                            if isinstance(value, torch.Tensor):
                                snapshot[f"{prefix}/{key}"] = value.detach().cpu().clone()
        return snapshot

    saved_optimizer = inner_optimizer_snapshot()

    def next_update():
        optimizer.zero_grad()
        ddp.zero_grad_buffer()
        run(ddp, False)
        finalize_model_grads([ddp], pg_collection=provider._pg_collection)
        assert optimizer.step()[0]

    next_update()
    continued = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    restored = load_dist_checkpointing(sharded_state(is_loading=True), str(resume_path))
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    for name, parameter in parameters.items():
        torch.testing.assert_close(parameter.cpu(), after_first[name], rtol=0, atol=0, msg=name)
    for name, value in inner_optimizer_snapshot().items():
        torch.testing.assert_close(
            value, saved_optimizer[name], rtol=0, atol=0, msg=lambda message, name=name: f"{name}: {message}"
        )
    next_update()
    for name, parameter in parameters.items():
        torch.testing.assert_close(
            parameter.cpu(), continued[name], rtol=0, atol=0, msg=lambda message, name=name: f"{name}: {message}"
        )
    for table, initial in frozen_tables.values():
        assert table.device.type == "cpu" and table.is_pinned() and not table.requires_grad
        torch.testing.assert_close(table, initial, rtol=0, atol=0)
    print(f"QWEN38_TP{TP}_PP{PP}_EP{EP}_DP{DP}_UPDATE_AND_RESUME_PASS", torch.distributed.get_rank(), flush=True)
