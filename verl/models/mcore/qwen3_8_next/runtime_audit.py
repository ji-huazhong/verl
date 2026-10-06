# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Opt-in production boundary evidence; no changes to forward arithmetic."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from verl.utils.device import get_torch_device

AUDIT_ENV = "VERL_QWEN38_PRODUCTION_AUDIT_DIR"
RUNTIME_ENV = "VERL_QWEN38_RUNTIME_METADATA_DIR"


def write_immutable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def tensor_sha256(value, chunk_bytes=32 * 1024 * 1024):
    """Hash logical row-major bytes without making a full GPU/CPU clone."""
    if value.layout != torch.strided or value.is_quantized:
        raise ValueError("Production weight audit expects dense unquantized tensors")
    if chunk_bytes < value.element_size():
        raise ValueError("Hash chunk must fit one tensor element")
    digest = hashlib.sha256()

    def visit(tensor):
        if tensor.numel() * tensor.element_size() <= chunk_bytes or tensor.ndim == 0:
            block = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(block))
        elif tensor.shape[0] == 1:
            visit(tensor[0])
        else:
            row_bytes = tensor[0].numel() * tensor.element_size()
            count = max(1, chunk_bytes // row_bytes)
            for start in range(0, tensor.shape[0], count):
                piece = tensor[start : start + count]
                visit(piece[0] if count == 1 else piece)

    visit(value)
    return digest.hexdigest()


def model_fingerprints(model, chunk_bytes=32 * 1024 * 1024):
    values = {}
    for kind, tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in tensors:
            values[kind + ":" + name] = dict(
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                device_type=tensor.device.type,
                bytes=tensor.numel() * tensor.element_size(),
                sha256=tensor_sha256(tensor, chunk_bytes),
            )
    return values


def compare_fingerprints(before, after):
    return dict(
        missing=sorted(before.keys() - after.keys()),
        added=sorted(after.keys() - before.keys()),
        changed={
            name: {
                key: dict(before=before[name][key], after=after[name][key])
                for key in ("shape", "dtype", "sha256")
                if before[name][key] != after[name][key]
            }
            for name in before.keys() & after.keys()
            if any(before[name][key] != after[name][key] for key in ("shape", "dtype", "sha256"))
        },
    )


def model_runtime_metadata(model):
    """Read actual MoE sharding and the precision flags used by the model."""
    moe, hc, conv = [], [], []
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen4ExpSparseMoeBlock":
            experts = module.experts
            config = experts.moe_config
            parallel = config.moe_parallel_config
            method = getattr(experts, "routed_experts", experts).quant_method
            moe.append(
                dict(
                    name=name,
                    tp=parallel.tp_size,
                    ep=parallel.ep_size,
                    ep_rank=parallel.ep_rank,
                    enable_ep=parallel.use_ep,
                    local_experts=config.num_local_experts,
                    all2all_backend=parallel.all2all_backend,
                    backend=str(getattr(method, "unquantized_backend", None)),
                )
            )
        elif type(module).__name__ == "GatedResidual":
            hc.append(bool(module.use_fp32))
        elif type(module).__name__ in ("QwenGatedDeltaNetAttention", "Qwen4ExpGatedDeltaNetAttention"):
            conv.append(str(module.conv1d.weight.dtype))
    return dict(model_type=type(model).__name__, moe=moe, hc_fp32=hc, gdn_conv_dtypes=conv)


def audit_worker_runtime(worker):
    directory = os.environ.get(RUNTIME_ENV)
    if not directory:
        return
    metadata = model_runtime_metadata(worker.get_model())
    replica = int(os.environ["VERL_REPLICA_RANK"])
    rank = torch.distributed.get_rank()
    expected_ep = int(os.environ["VERL_QWEN38_EXPECT_MOE_EP_SIZE"])
    checks = dict(
        layer_count=len(metadata["moe"]) == 48,
        expert_parallel=all(row["ep"] == expected_ep for row in metadata["moe"]),
        expert_tensor_parallel=all(row["tp"] == (1 if expected_ep > 1 else 8) for row in metadata["moe"]),
        local_experts=all(row["local_experts"] == 512 // expected_ep for row in metadata["moe"]),
        expert_parallel_enabled=all(row["enable_ep"] == (expected_ep > 1) for row in metadata["moe"]),
        triton_backend=all("TRITON" in row["backend"].upper() for row in metadata["moe"]),
        hc_fp32=len(metadata["hc_fp32"]) == 97 and all(metadata["hc_fp32"]),
        conv_fp32=len(metadata["gdn_conv_dtypes"]) == 36
        and all(dtype == "torch.float32" for dtype in metadata["gdn_conv_dtypes"]),
    )
    write_immutable(
        Path(directory) / f"replica-{replica:03d}" / f"rank-{rank:03d}.json",
        dict(complete=True, replica=replica, rank=rank, expected_ep=expected_ep, checks=checks, metadata=metadata),
    )
    if not all(checks.values()):
        raise RuntimeError(f"Qwen production runtime differs from requested precision/topology: {checks}")


def audit_worker_weights(worker, stage):
    directory = os.environ.get(AUDIT_ENV)
    if not directory:
        return
    if stage not in ("hf-loaded", "first-refit"):
        raise ValueError(stage)
    rank = torch.distributed.get_rank()
    replica = int(os.environ["VERL_REPLICA_RANK"])
    model = worker.get_model()
    start = time.monotonic()
    get_torch_device().synchronize()
    fingerprints = model_fingerprints(model)
    comparison = None
    if stage == "hf-loaded":
        assert not hasattr(worker, "_qwen38_initial_fingerprints")
        worker._qwen38_initial_fingerprints = fingerprints
    else:
        comparison = compare_fingerprints(worker._qwen38_initial_fingerprints, fingerprints)
    payload = dict(
        complete=True,
        stage=stage,
        replica=replica,
        rank=rank,
        model_type=type(model).__name__,
        seconds=time.monotonic() - start,
        tensors=fingerprints,
        comparison=comparison,
        scope="All named parameters and buffers, including full host PLE tables. Logical tensor bytes; "
        "buffers may include mutable runtime state and must be interpreted separately from parameters.",
    )
    write_immutable(Path(directory) / f"replica-{replica:03d}" / f"rank-{rank:03d}" / f"{stage}.json", payload)
    print("QWEN38_PRODUCTION_WEIGHT_AUDIT", stage, replica, rank, payload["seconds"], flush=True)


def audit_server_response(*, replica, request_id, prompt_ids, final_res, sampling_params, global_steps):
    directory = os.environ.get("VERL_QWEN38_RESPONSE_AUDIT_DIR") or os.environ.get(AUDIT_ENV)
    if not directory:
        return
    answer = final_res.outputs[0]
    response = list(answer.token_ids)
    logs = (
        None
        if answer.logprobs is None
        else [entry[token].logprob for token, entry in zip(response, answer.logprobs, strict=True)]
    )
    fingerprint = hashlib.sha256(json.dumps([list(prompt_ids), response], separators=(",", ":")).encode()).hexdigest()
    payload = dict(
        complete=True,
        replica=replica,
        request_id=request_id,
        global_steps=global_steps,
        input_ids=list(prompt_ids),
        engine_prompt_ids=final_res.prompt_token_ids,
        response_ids=response,
        logprobs=logs,
        tokens_sha256=fingerprint,
        cached_tokens=getattr(final_res, "num_cached_tokens", None),
        sampling={
            name: getattr(sampling_params, name, None)
            for name in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "max_tokens",
                "min_tokens",
                "ignore_eos",
                "seed",
                "logprobs",
                "allowed_token_ids",
                "logit_bias",
                "bad_words",
                "stop_token_ids",
            )
        },
    )
    name = hashlib.sha256(request_id.encode()).hexdigest()
    write_immutable(Path(directory) / f"replica-{replica:03d}" / "responses" / f"{name}.json", payload)
