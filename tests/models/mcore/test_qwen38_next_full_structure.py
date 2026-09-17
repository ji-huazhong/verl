# SPDX-License-Identifier: Apache-2.0
"""Opt-in full checkpoint structure audit, NOT loading or numerical acceptance.

Use eight torchrun workers, RUN_QWEN38_FULL_STRUCTURE_TESTS=1 and
QWEN38_MODEL_PATH. QWEN38_STRUCTURE_HYBRID=1 selects TP2/PP2/EP2/CP2/VPP2;
otherwise TP8/EP8/PP1 is inspected. The real config is unchanged. A test-only
dispatch context redirects explicit device allocations missed by native meta
initialization (Core embeddings/GDN and Bridge vision); every parameter must
remain meta. NCCL/runtime allocation is still bounded and reported.
Only checkpoint headers are read. No tensor payload, forward or update runs.
Logical parameter and PLE bytes describe the constructed rank-local layout,
not measured runtime peaks or a promise that training fits in that memory.
"""

import json
import os
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _MetaAllocations(TorchDispatchMode):
    """Inspection only: change allocation device, never architecture or shapes.

    In particular this must not wrap Bridge loading, communication or forward.
    An unsupported value-dependent constructor still fails on meta tensors.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if "device" in kwargs:
            kwargs["device"] = torch.device("meta")
            if "pin_memory" in kwargs:
                kwargs["pin_memory"] = False
        return func(*args, **kwargs)


def test_inspection_redirects_explicit_factories_and_restores_device():
    with _MetaAllocations(), torch.device("meta"):
        assert torch.empty((8, 16), device="cpu", pin_memory=True).is_meta
        assert torch.ones(4, device="cpu").is_meta
        assert torch.empty((8, 16)).to(device="cpu").is_meta
    assert torch.empty(1).device.type == "cpu"


def _source_shape(task, headers):
    """Independent shape-only check of the supported mapping families.

    This intentionally does not claim to validate packing order or values.
    Unknown mapping families fail closed instead of being counted as covered.
    """
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param

    mapping = task.mapping
    kind = type(mapping).__name__
    names = [mapping.hf_param] if isinstance(mapping.hf_param, str) else list(mapping.hf_param.values())
    shapes = [tuple(headers[name]["shape"]) for name in names]
    if kind in ("FusedExpertMapping", "FusedGatedExpertMapping"):
        assert len(shapes) == 1 and len(shapes[0]) == 3
        expert = extract_expert_number_from_param(mapping.megatron_param)
        assert 0 <= expert < shapes[0][0]
        shape = shapes[0][1:]
    elif len(shapes) > 1:
        assert kind in ("QKVMapping", "GDNLinearMappingSeparate", "GatedMLPMapping"), kind
        assert len({shape[1:] for shape in shapes}) == 1, (names, shapes)
        shape = (sum(shape[0] for shape in shapes), *shapes[0][1:])
    else:
        shape = shapes[0]
    if getattr(mapping, "permute_dims", None) is not None:
        shape = tuple(shape[axis] for axis in mapping.permute_dims)

    if kind in (
        "ColumnParallelMapping",
        "QKVMapping",
        "ConcatenatedQKVMapping",
        "GDNLinearMappingSeparate",
        "GDNConv1dMapping",
        "GatedMLPMapping",
        "FusedGatedExpertMapping",
    ):
        parallel = "column"
    elif kind == "ReplicatedMapping":
        parallel = "replicated"
    elif kind == "RowParallelMapping":
        parallel = "row"
    elif isinstance(mapping, AutoMapping) and kind in (
        "AutoMapping",
        "FusedExpertMapping",
        "RMSNorm2ZeroCenteredRMSNormMapping",
    ):
        parallel = mapping._detect_parallelism_type(task.megatron_module)
    else:
        raise AssertionError(f"Unaudited mapping family: {kind}")
    shape = list(shape)
    if parallel != "replicated" and not (parallel == "row" and len(shape) == 1):
        axis = 0 if parallel == "column" else 1
        assert shape[axis] % mapping.tp_size == 0, (names, shape, mapping.tp_size)
        shape[axis] //= mapping.tp_size
    return tuple(shape), names


def test_shape_audit_rejects_missing_unknown_and_invalid_expert(monkeypatch):
    from megatron.bridge.models.conversion.param_mapping import (
        ColumnParallelMapping,
        FusedGatedExpertMapping,
        MegatronParamMapping,
        ReplicatedMapping,
    )

    monkeypatch.setattr(MegatronParamMapping, "tp_size", property(lambda self: 2))
    task = SimpleNamespace(mapping=ColumnParallelMapping("weight", "source"))
    assert _source_shape(task, {"source": {"shape": [16, 8]}})[0] == (8, 8)
    with pytest.raises(KeyError):
        _source_shape(task, {})
    with pytest.raises(AssertionError):
        _source_shape(task, {"source": {"shape": [15, 8]}})

    class Unsupported(ReplicatedMapping):
        pass

    task.mapping = Unsupported("weight", "source")
    with pytest.raises(AssertionError, match="Unaudited mapping family"):
        _source_shape(task, {"source": {"shape": [16, 8]}})
    task.mapping = FusedGatedExpertMapping("decoder.layers.0.mlp.experts.linear_fc1.weight3", "source")
    assert _source_shape(task, {"source": {"shape": [4, 16, 8]}})[0] == (8, 8)
    with pytest.raises(AssertionError):
        _source_shape(task, {"source": {"shape": [3, 16, 8]}})


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_FULL_STRUCTURE_TESTS") != "1",
    reason="explicit eight-GPU header-only full structure opt-in required",
)
def test_full_checkpoint_target_structure_and_shape_coverage():
    import struct

    from megatron.bridge.utils.common_utils import extract_expert_number_from_param
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.module import Float16Module

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.ops.ple import Qwen38NextFrozenNGramEmbedding

    assert int(os.environ.get("WORLD_SIZE", "0")) == 8
    checkpoint = Path(os.environ["QWEN38_MODEL_PATH"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(1024**3 / total)
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    try:
        ready = torch.tensor(int(free >= 4 * 1024**3), device=device)
        torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        if not ready.item():
            pytest.skip("Every GPU needs 4 GiB headroom; never evict another job")
        hybrid = os.environ.get("QWEN38_STRUCTURE_HYBRID") == "1"
        tp, pp, ep, cp, vp = (2, 2, 2, 2, 2) if hybrid else (8, 1, 8, 1, None)
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            context_parallel_size=cp,
            virtual_pipeline_model_parallel_size=vp,
            expert_tensor_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)
        bridge = AutoBridge.from_hf_pretrained(checkpoint, local_files_only=True)
        assert isinstance(bridge._model_bridge, Qwen38NextBridge)
        config = bridge.to_megatron_provider(load_weights=False)
        assert config.num_layers == 48, "The full audit must not silently inspect a reduced fixture"
        config.tensor_model_parallel_size = tp
        config.pipeline_model_parallel_size = pp
        config.expert_model_parallel_size = ep
        config.expert_tensor_parallel_size = 1
        config.context_parallel_size = cp
        config.virtual_pipeline_model_parallel_size = vp
        config.sequence_parallel = True
        config.variable_seq_lengths = True
        config.overlap_p2p_comm = bool(vp)
        config.batch_p2p_comm = not bool(vp)
        config.calculate_per_token_loss = True
        config.moe_router_load_balancing_type = "none"
        config.moe_permute_fusion = False
        config.params_dtype = torch.bfloat16
        config.bf16 = True
        config.init_model_with_meta_device = True
        config.perform_initialization = False
        config.finalize()
        config._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        with _MetaAllocations(), torch.device("meta"):
            models = [
                config.provide(
                    pre_process=pp_rank == 0 and chunk == 0,
                    post_process=pp_rank == pp - 1 and chunk == (vp or 1) - 1,
                    vp_stage=chunk if vp else None,
                )
                for chunk in range(vp or 1)
            ]
            wrappers = [Float16Module(config, model) for model in models]
        assert len(wrappers) == len(models)

        headers = {}
        weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        for filename in sorted(set(weight_map.values())):
            with (checkpoint / filename).open("rb") as file:
                header_size = struct.unpack("<Q", file.read(8))[0]
                assert 0 < header_size < 64 * 1024**2
                header = json.loads(file.read(header_size))
            for name, metadata in header.items():
                if name == "__metadata__":
                    continue
                assert name not in headers and weight_map[name] == filename
                headers[name] = metadata
        assert set(headers) == set(weight_map)
        tasks = bridge._model_bridge.build_conversion_tasks(bridge.hf_pretrained, models)
        local_tasks = [task for task in tasks if task is not None and task.param_weight is not None]
        expected = {(chunk, name) for chunk, model in enumerate(models) for name, _ in model.named_parameters()}
        actual = {(task.vp_stage, task.param_name) for task in local_tasks}
        errors = []
        if actual != expected:
            errors.append(f"Missing targets: {sorted(expected - actual)[:10]}; extra: {sorted(actual - expected)[:10]}")
        used = set()
        expert_rows = {}
        mapping_counts = Counter()
        for task in local_tasks:
            try:
                shape, names = _source_shape(task, headers)
                assert shape == tuple(task.param_weight.shape), (task.global_param_name, shape, task.param_weight.shape)
                assert all(headers[name]["dtype"] == "BF16" for name in names), names
                assert task.param_weight.dtype == torch.bfloat16, task.global_param_name
                used.update(names)
                mapping_counts[type(task.mapping).__name__] += 1
                if type(task.mapping).__name__ in ("FusedExpertMapping", "FusedGatedExpertMapping"):
                    expert_rows.setdefault(names[0], set()).add(
                        extract_expert_number_from_param(task.mapping.megatron_param)
                    )
            except (AssertionError, KeyError, ValueError) as error:
                errors.append(f"{task.global_param_name}: {error}")
        nonmeta = [
            (name, parameter.numel() * parameter.element_size())
            for model in models
            for name, parameter in model.named_parameters()
            if not parameter.is_meta
        ]
        real_bytes = sum(size for _, size in nonmeta)
        if nonmeta:
            errors.insert(0, f"Unexpected real allocation: {nonmeta[:10]}, total bytes: {real_bytes}")
        ple_host_bytes = 0
        ple_tables = 0
        for model in models:
            for module in model.modules():
                if isinstance(module, Qwen38NextFrozenNGramEmbedding):
                    assert module.table.is_meta and not module._loaded
                    ple_host_bytes += module.table.numel() * module.table.element_size()
                    ple_tables += 1
        base_parameter_bytes = sum(task.param_weight.numel() * task.param_weight.element_size() for task in local_tasks)
        layers = [layer.layer_number for model in models for layer in model.language_model.decoder.layers]
        results = [None] * 8
        if errors:
            print(
                "QWEN38_STRUCTURE_ERRORS " + json.dumps({"rank": torch.distributed.get_rank(), "errors": errors}),
                flush=True,
            )
        torch.distributed.all_gather_object(
            results,
            {
                "sources": sorted(used),
                "layers": layers,
                "errors": errors[:10],
                "expert_rows": {name: sorted(rows) for name, rows in expert_rows.items()},
            },
        )
        assert not any(result["errors"] for result in results), [result["errors"] for result in results]
        assert set().union(*(set(result["layers"]) for result in results)) == set(range(1, 49))
        covered = set().union(*(set(result["sources"]) for result in results))
        excluded = {name for name in weight_map if name.startswith("mtp.") or ".ple.ple_embedding." in name}
        assert covered == set(weight_map) - excluded, sorted(set(weight_map) - excluded - covered)[:20]
        all_experts = {}
        for result in results:
            for name, rows in result["expert_rows"].items():
                all_experts.setdefault(name, set()).update(rows)
        for name, rows in all_experts.items():
            assert rows == set(range(headers[name]["shape"][0])), (name, len(rows))
        print(
            "QWEN38_FULL_STRUCTURE "
            + json.dumps(
                dict(
                    rank=torch.distributed.get_rank(),
                    hybrid=hybrid,
                    layers=layers,
                    targets=len(local_tasks),
                    source_tensors=len(covered),
                    explicitly_excluded=len(excluded),
                    mapping_counts=dict(mapping_counts),
                    fused_expert_sources=len(all_experts),
                    fused_expert_rows=sum(len(rows) for rows in all_experts.values()),
                    logical_base_parameter_bytes=base_parameter_bytes,
                    logical_ple_host_bytes=ple_host_bytes,
                    local_ple_tables=ple_tables,
                    free_cuda_bytes_before_audit=free,
                    real_parameter_bytes=real_bytes,
                    peak_cuda_mib=torch.cuda.max_memory_allocated() / 1024**2,
                    payload_loaded=False,
                    training_steps=0,
                )
            ),
            flush=True,
        )
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
