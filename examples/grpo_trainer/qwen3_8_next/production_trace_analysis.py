# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Compare real production activations after verifying tokens, shards and routes."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch


def merge_stages(records, input_ids):
    """Join SP pieces by causal position; reject conflicting replicated rows."""
    merged = defaultdict(dict)
    for record in records:
        for stage, pieces in record["stages"].items():
            for piece in pieces:
                positions, tokens, values = piece["positions"], piece["token_ids"], piece["value"]
                if values.ndim != 2 or len(positions) != len(tokens) or len(tokens) != len(values):
                    raise ValueError(f"Invalid activation coordinates at {stage}")
                for position, token, value in zip(positions, tokens, values, strict=True):
                    if not 0 <= position < len(input_ids) - 1 or input_ids[position] != token:
                        raise ValueError(f"Activation token identity differs at {stage}/{position}")
                    previous = merged[stage].get(position)
                    if previous is not None:
                        if previous.dtype != value.dtype or not torch.equal(previous, value):
                            raise ValueError(f"Conflicting duplicate activation at {stage}/{position}")
                    else:
                        merged[stage][position] = value
    return dict(merged)


def required_stages(tp_size):
    suffixes = ["attn_hc/mixed", "attention/output", "mlp_hc/mixed", "router/logits", "mlp/output"]
    stages = [f"layers/{layer:02d}/{suffix}" for layer in range(48) for suffix in suffixes]
    stages.append("final_mixer/output")
    stages.extend(
        f"layers/00/gdn/{part}/tp-{rank:02d}"
        for part in ["projection/qkvz", "projection/ba", "out_proj_input"]
        for rank in range(tp_size)
    )
    return stages


def require_positions(stages, names, positions):
    for name in names:
        if name not in stages:
            raise ValueError(f"Missing required stage {name}")
        absent = set(positions) - stages[name].keys()
        if absent:
            raise ValueError(f"Missing causal queries at {name}: {sorted(absent)[:8]}")


def compare_stages(actor, rollout, response, routes, query_limit, tp_size=8):
    prompt_length = len(response["prompt_ids"])
    positions = list(range(prompt_length - 1, prompt_length - 1 + min(query_limit, len(response["response_ids"]))))
    if not positions:
        raise ValueError("No response queries to compare")
    names = required_stages(tp_size)
    require_positions(
        actor, names + ["log_probs"] + [f"layers/{layer:02d}/router/selected" for layer in range(48)], positions
    )
    require_positions(rollout, names, positions)
    if routes.ndim != 3 or routes.shape[1] != 48 or len(routes) <= positions[-1]:
        raise ValueError("Actual rollout routes do not cover the traced positions/layers")
    routing_checks = 0
    mismatches = []
    for layer in range(48):
        for position in positions:
            actual = torch.where(actor[f"layers/{layer:02d}/router/selected"][position].bool())[0]
            expected = routes[position, layer].to(torch.int64).sort().values
            routing_checks += 1
            if not torch.equal(actual, expected):
                mismatches.append(
                    dict(layer=layer, query_position=position, actor=actual.tolist(), rollout=expected.tolist())
                )
    actor_scores = torch.tensor([actor["log_probs"][position].item() for position in positions], dtype=torch.float64)
    rollout_scores = torch.tensor(response["logprobs"][: len(positions)], dtype=torch.float64)
    if not actor_scores.isfinite().all() or not rollout_scores.isfinite().all():
        raise ValueError("Nonfinite production scores")
    delta = (actor_scores - rollout_scores).abs()
    worst_index = int(delta.argmax())
    worst_position = positions[worst_index]
    metrics = {}
    for name in names:
        left = torch.stack([actor[name][position] for position in positions])
        right = torch.stack([rollout[name][position] for position in positions])
        if left.shape != right.shape:
            raise ValueError(f"Activation feature shapes differ at {name}: {left.shape}, {right.shape}")
        if not left.isfinite().all() or not right.isfinite().all():
            raise ValueError(f"Nonfinite production activation at {name}")
        difference = right.double() - left.double()
        reference = left.double()
        per_query = difference.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-30)
        metrics[name] = dict(
            relative_l2=(difference.norm() / reference.norm().clamp_min(1e-30)).item(),
            query_relative_l2_median=per_query.median().item(),
            mean_abs=difference.abs().mean().item(),
            max_abs=difference.abs().max().item(),
            differing_elements=int(torch.count_nonzero(difference)),
            actor_dtype=str(left.dtype),
            rollout_dtype=str(right.dtype),
            worst_logprob_query_relative_l2=per_query[worst_index].item(),
            worst_logprob_query_max_abs=difference[worst_index].abs().max().item(),
        )
    # The first 241 entries are execution-order layer boundaries; detailed GDN
    # shard records appear afterward and refine the first layer independently.
    boundaries = names[: 48 * 5 + 1]
    first = next((name for name in boundaries if metrics[name]["worst_logprob_query_max_abs"] != 0), None)
    return dict(
        complete=True,
        request_id=response["request_id"],
        compared_queries=len(positions),
        coordinate_validation_passed=True,
        all_required_stages_present=True,
        route_checks=routing_checks,
        route_mismatches=mismatches,
        route_control_passed=not mismatches,
        mean_abs_logprob=delta.mean().item(),
        max_abs_logprob=delta.max().item(),
        worst_response_index=worst_index,
        worst_query_position=worst_position,
        worst_actor_logprob=actor_scores[worst_index].item(),
        worst_rollout_logprob=rollout_scores[worst_index].item(),
        first_nonzero_boundary_at_worst_query=first,
        stages=metrics,
        scope=(
            "Original production forwards for one actual response. Initial query predicts the first response token. "
            "Layer errors include propagated upstream error; "
            "a nonzero boundary alone does not prove a faulty operator."
        ),
        full_precision_accepted=False,
    )


def load_verified(manifest):
    metadata = json.loads(manifest.read_text())
    path = manifest.with_name("activations.pt")
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != metadata["sha256"]:
            raise ValueError(f"Incomplete or corrupt activation artifact {manifest}")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not metadata["complete"] or any(record["metadata"][key] != metadata[key] for key in record["metadata"]):
        raise ValueError("Activation metadata differs from its completion marker")
    return record


def rollout_manifests_for_response(root, response):
    # V2 InputProcessor.assign_request_id appends an eight-hex internal suffix.
    # Match the complete external ID and reject reuse/ambiguous internal IDs.
    pattern = re.compile(re.escape(response["request_id"]) + r"(?:-[0-9a-f]{8})?")
    matches, identities = [], set()
    for path in sorted((root / "vllm" / f"replica-{response['replica']:03d}").glob("rank-*/*/complete.json")):
        metadata = json.loads(path.read_text())
        if not pattern.fullmatch(metadata["identity"]):
            continue
        if metadata["prompt_ids"] != response["prompt_ids"]:
            raise ValueError("Internal rollout request has a different prompt")
        identities.add(metadata["identity"])
        matches.append(path)
    if len(identities) != 1:
        raise ValueError("Missing or ambiguous internal rollout request identity")
    ranks = [json.loads(path.read_text())["tp_rank"] for path in matches]
    sizes = {json.loads(path.read_text())["tp_size"] for path in matches}
    if len(sizes) != 1 or sorted(ranks) != list(range(next(iter(sizes)))):
        raise ValueError("Missing or duplicated rollout TP feature shards")
    return matches


def analyze_request(root, response_manifest, query_limit=512):
    response = json.loads(response_manifest.read_text())
    if not response["complete"] or response["input_ids"] != response["prompt_ids"] + response["response_ids"]:
        raise ValueError("Invalid production response identity")
    route_path = response_manifest.with_name("routes.pt")
    with route_path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != response["routes_sha256"]:
            raise ValueError("Incomplete or corrupt production routes")
    routes = torch.load(route_path, map_location="cpu", weights_only=True)
    rollout_manifests = rollout_manifests_for_response(root, response)
    actor_manifests, seen_rank = [], set()
    for path in sorted((root / "megatron").glob("rank-*/forward-*/*/complete.json")):
        metadata = json.loads(path.read_text())
        if metadata["input_ids"] != response["input_ids"]:
            continue
        rank = path.parents[2].name
        if rank in seen_rank:
            raise ValueError("Multiple identical actor trajectories on one rank; request identity is ambiguous")
        seen_rank.add(rank)
        actor_manifests.append(path)
    if not actor_manifests:
        raise ValueError("Missing matching original actor trajectory")
    actor = merge_stages((load_verified(path) for path in actor_manifests), response["input_ids"])
    rollout = merge_stages((load_verified(path) for path in rollout_manifests), response["input_ids"])
    tp_sizes = {json.loads(path.read_text())["tp_size"] for path in rollout_manifests}
    if len(tp_sizes) != 1 or len(rollout_manifests) != next(iter(tp_sizes)):
        raise ValueError("Missing rollout TP feature shards")
    result = compare_stages(actor, rollout, response, routes, query_limit, next(iter(tp_sizes)))
    result["actor_artifacts"] = [str(path) for path in actor_manifests]
    result["rollout_artifacts"] = [str(path) for path in rollout_manifests]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-limit", type=int, default=512)
    args = parser.parse_args()
    result = analyze_request(args.root, args.response, args.query_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ["stages", "actor_artifacts", "rollout_artifacts", "route_mismatches"]
            }
        )
    )


if __name__ == "__main__":
    main()
