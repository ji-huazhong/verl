# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Opt-in capture of actual rollout tokens, routes and pre-update actor scores."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch


def select_capture_indices(data, count=16):
    """Keep length coverage and high-error cases; this is not an unbiased estimate."""
    if count < 1:
        raise ValueError("Capture count must be positive")
    rows = []
    for i, (prompt, response, actor, rollout, mask) in enumerate(
        zip(
            data["prompts"].unbind(),
            data["responses"].unbind(),
            data["old_log_probs"].unbind(),
            data["rollout_log_probs"].unbind(),
            data["response_mask"].unbind(),
            strict=True,
        )
    ):
        if len(response) != len(actor) or len(actor) != len(rollout) or len(mask) != len(response):
            raise ValueError("Capture response, score and mask coordinates differ")
        mask = mask.bool()
        if not mask.any():
            continue
        delta = (actor.float() - rollout.float()).abs()[mask]
        if not delta.isfinite().all():
            raise ValueError("Nonfinite scored logprobs in production capture")
        rows.append((i, len(prompt) + len(response), delta.mean().item()))
    if not rows:
        raise ValueError("Capture requires a scored response")
    count = min(count, len(rows))
    by_length = sorted(rows, key=lambda row: (row[1], row[0]))
    coverage = min((count + 1) // 2, len(rows))
    selected = {by_length[round(i * (len(rows) - 1) / max(1, coverage - 1))][0] for i in range(coverage)}
    for index, _, _ in sorted(rows, key=lambda row: (-row[2], row[0])):
        if len(selected) == count:
            break
        selected.add(index)
    return sorted(selected)


def capture_logprob_batch(root, step, keys, data, indices, routes, *, metadata):
    """Write immutable replay inputs; never alter tensors stored in TransferQueue."""
    import numpy as np

    if step < 1 or len(keys) != data.batch_size[0]:
        raise ValueError("Invalid production capture identity")
    if not indices or len(set(indices)) != len(indices) or len(routes) != len(indices):
        raise ValueError("Capture indices and selected route rows differ")
    destination = Path(root) / f"step-{step:06d}"
    if destination.exists():
        raise FileExistsError(destination)
    samples, selected, whole_batch = [], [], []
    for i, (actor, rollout, mask) in enumerate(
        zip(
            data["old_log_probs"].unbind(),
            data["rollout_log_probs"].unbind(),
            data["response_mask"].unbind(),
            strict=True,
        )
    ):
        delta = (actor.float() - rollout.float()).abs()[mask.bool()]
        whole_batch.append(
            dict(
                batch_index=i,
                scored_tokens=len(delta),
                mean_abs=delta.mean().item() if len(delta) else 0.0,
                max_abs=delta.max().item() if len(delta) else 0.0,
            )
        )
    with tempfile.TemporaryDirectory(prefix="verl-logprob-capture-") as scratch:
        local = Path(scratch)
        (local / "generation").mkdir()
        for ordinal, (index, route) in enumerate(zip(indices, routes, strict=True)):
            prompt = data["prompts"][index].detach().cpu().tolist()
            response = data["responses"][index].detach().cpu().tolist()
            ids = prompt + response
            if not prompt or not response or route.ndim != 3 or len(route) not in (len(ids), len(ids) - 1):
                raise ValueError("Production route rows do not match causal token coordinates")
            route = route[: len(ids) - 1].detach().cpu().contiguous()
            if route.dtype not in (torch.int16, torch.int32, torch.int64):
                raise TypeError("Captured routes must retain integer expert indices")
            name = f"production-step-{step:06d}-{ordinal:03d}"
            filename = f"routes-{ordinal:03d}.npy"
            np.save(local / "generation" / filename, route.numpy(), allow_pickle=False)
            samples.append(
                dict(
                    id=name,
                    production_key=str(keys[index]),
                    batch_index=index,
                    prompt_ids=prompt,
                    input_ids=ids,
                    prompt_length=len(prompt),
                    response_length=len(response),
                    response_mask=data["response_mask"][index].detach().cpu().bool().tolist(),
                    generation_logprobs=data["rollout_log_probs"][index].detach().cpu().float().tolist(),
                    production_actor_logprobs=data["old_log_probs"][index].detach().cpu().float().tolist(),
                    generation_routes=filename,
                )
            )
            selected.append(dict(id=name, prompt_ids=prompt, production_key=str(keys[index])))
        write = lambda path, value: path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        write(local / "samples.json", samples)
        write(local / "selected-prompts.json", selected)
        manifest = dict(
            complete=True,
            stage="pre-policy-update old-logprob scoring",
            step=step,
            batch_size=len(keys),
            captured_samples=len(samples),
            indices=indices,
            selection="Length quantiles plus largest response-mean errors; selected subset is biased.",
            batch_response_mean_abs=sum(row["mean_abs"] for row in whole_batch) / len(whole_batch),
            batch_token_mean_abs=sum(row["mean_abs"] * row["scored_tokens"] for row in whole_batch)
            / sum(row["scored_tokens"] for row in whole_batch),
            per_response=whole_batch,
            metadata=metadata,
            files={
                str(path.relative_to(local)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in local.rglob("*")
                if path.is_file()
            },
        )
        write(local / "capture.json", manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Publish a completion marker last; incomplete copies must never be replayed.
        shutil.copytree(local, destination)
        write(destination / "capture-complete.json", dict(complete=True, captured_samples=len(samples)))
    return dict(
        path=str(destination), samples=len(samples), batch_response_mean_abs=manifest["batch_response_mean_abs"]
    )
