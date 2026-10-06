# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Record arithmetic settings and compare repeated, fixed-token forwards."""

import hashlib
import json
import os
from pathlib import Path


def record_megatron_runtime(directory, provider, model, gdn_layers=()):
    """Read settings without changing the precision or determinism policy."""
    import torch
    from megatron.core import parallel_state

    root = Path(__file__).resolve().parents[3]
    sources = [
        Path(__file__),
        Path(__file__).with_name("check_logprobs.py"),
        Path(__file__).with_name("backend_trace.py"),
    ]
    sources.extend(sorted((root / "verl/models/mcore/qwen3_8_next").rglob("*.py")))
    cuda = torch.backends.cuda.matmul
    payload = {
        "rank": torch.distributed.get_rank(),
        "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
        "pp_rank": parallel_state.get_pipeline_model_parallel_rank(),
        "dp_rank": parallel_state.get_data_parallel_rank(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": cuda.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "allow_bf16_reduced_precision_reduction": cuda.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction": cuda.allow_fp16_reduced_precision_reduction,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "training_modules": [name for name, module in model.named_modules() if module.training],
        "environment": {
            name: os.environ.get(name)
            for name in (
                "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
            )
        },
        "config": {
            name: str(getattr(provider, name, None))
            for name in (
                "moe_router_dtype",
                "moe_router_load_balancing_type",
                "moe_token_dispatcher_type",
                "moe_permute_fusion",
                "moe_grouped_gemm",
                "params_dtype",
                "pipeline_dtype",
                "sequence_parallel",
                "attention_dropout",
                "hidden_dropout",
                "deterministic_mode",
            )
        },
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank-{payload['rank']:05d}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    if gdn_layers:
        records = {}
        for layer in model.module.language_model.decoder.layers:
            index = layer.layer_number - 1
            if index not in gdn_layers:
                continue
            attention = layer.self_attention
            if type(attention).__name__ != "Qwen38NextGatedDeltaNet":
                raise ValueError("Detailed GDN weight capture requested for a non-GDN layer")
            state = {name: parameter.detach().cpu().contiguous() for name, parameter in attention.named_parameters()}
            filename = f"gdn-layer-{index:02d}-rank-{payload['rank']:05d}.pt"
            torch.save(state, directory / filename)
            records[str(index)] = {
                "file": filename,
                "parameters": {
                    name: {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    }
                    for name, tensor in state.items()
                },
            }
        (directory / f"gdn-weights-rank-{payload['rank']:05d}.json").write_text(json.dumps(records, indent=2) + "\n")
    if payload["rank"] == 0:
        # This HC is replicated across TP; copying its four small tensors does
        # not require collectives or serialize the PLE/expert checkpoint.
        first = model.module.language_model.decoder.layers[0].self_attention_hyper_connection
        if first.layer_number != 1:
            raise ValueError("Expected the first global HC on rank zero")
        state = {
            name: getattr(first, name).detach().cpu().clone()
            for name in ("hc_norm_weight", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
        }
        torch.save(
            dict(state=state, hc_count=first.n, hidden_size=first.hidden_size, norm_eps=first.norm_eps),
            directory / "first-hc-megatron.pt",
        )


def compare_repeated_logprobs(reference, candidate):
    """Keep all-token and prefix checks separate, including first changed index."""
    import numpy as np

    for result in (reference, candidate):
        if result.get("complete") is not True:
            raise ValueError("Repetition control requires complete results")
    for field in ("config_sha256", "prompt_sha256", "backend", "dtype", "parallelism"):
        if reference[field] != candidate[field]:
            raise ValueError(f"Repetition control has different {field}")

    def metrics(delta):
        if not delta.size or not np.isfinite(delta).all():
            raise ValueError("Empty or non-finite repetition result")
        changed = np.flatnonzero(delta)
        return dict(
            tokens=int(delta.size),
            mean_abs=float(delta.mean()),
            max_abs=float(delta.max()),
            changed=int(changed.size),
            first_changed_index=int(changed[0]) if changed.size else None,
        )

    records, differences = [], []
    for left, right in zip(reference["records"], candidate["records"], strict=True):
        if left["id"] != right["id"] or left["input_ids"] != right["input_ids"]:
            raise ValueError("Repetition control has different token sequences")
        if len(left["logprobs"]) != len(right["logprobs"]) or len(left["logprobs"]) != len(left["input_ids"]) - 1:
            raise ValueError("Repetition control has incomplete token logprobs")
        delta = np.abs(np.asarray(left["logprobs"], dtype=np.float64) - np.asarray(right["logprobs"], dtype=np.float64))
        records.append(
            dict(id=left["id"], all=metrics(delta), prefix31=metrics(delta[:31]), prefix2048=metrics(delta[:2048]))
        )
        differences.append(delta)
    return dict(all=metrics(np.concatenate(differences)), records=records)


def save_megatron_control(directory, phases):
    directory = Path(directory)
    results = {name: json.loads((directory / (name + ".json")).read_text()) for name in phases}
    baseline = "baseline1"
    report = {
        "phases": phases,
        "reference": baseline,
        "same_process_same_loaded_weights": True,
        "full_model_acceptance": False,
        "comparisons": {
            name: compare_repeated_logprobs(results[baseline], result)
            for name, result in results.items()
            if name != baseline
        },
        "traced_repeat": compare_repeated_logprobs(results["traced0"], results["traced1"]),
    }
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "QWEN38_MEGATRON_REPETITION_CONTROL",
        json.dumps({name: item["all"] for name, item in report["comparisons"].items()}),
        flush=True,
    )
