#!/usr/bin/env python3
"""Summarize console logs from the fixed-rollout H20 CE benchmark.

Throughput is sum(tokens) / sum(seconds), not the mean of per-step rates.
Replay step throughput excludes real generation and is not end-to-end RL TPS.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"\bstep:(\d+) - ")
TIMINGS = ("old_log_prob", "update_actor", "update_weights", "step")
MEMORY = ("allocated", "reserved")
NUMERICS = (
    "actor/entropy",
    "actor/mtp_losses/mtp_1_loss",
    "actor/pg_loss",
    "actor/grad_norm",
    "rollout_corr/training_log_ppl",
    "critic/rewards/mean",
    "critic/advantages/max",
    "critic/advantages/min",
)


def read_steps(path: Path, expected_steps: int) -> list[dict[str, float]]:
    steps = {}
    for line in path.read_text().splitlines():
        line = ANSI.sub("", line)
        match = STEP.search(line)
        if match is None or "perf/total_num_tokens:" not in line:
            continue
        row = {"step": int(match[1])}
        for field in line[match.end() :].split(" - "):
            key, _, value = field.partition(":")
            try:
                row[key] = float(value)
            except ValueError:
                continue  # numpy-formatted diagnostic counters are not used
        if row["step"] in steps:
            raise ValueError(f"{path}: duplicate metric row for step {row['step']}")
        steps[row["step"]] = row
    if sorted(steps) != list(range(1, expected_steps + 1)):
        raise ValueError(f"{path}: expected steps 1..{expected_steps}, found {sorted(steps)}")
    rows = [steps[step] for step in sorted(steps)]
    required = (
        "perf/total_num_tokens",
        *(f"timing_s/{name}" for name in TIMINGS),
        *(f"actor/perf/max_memory_{name}_gb" for name in MEMORY),
        *NUMERICS,
    )
    for row in rows:
        for key in required:
            if key not in row or not math.isfinite(row[key]):
                raise ValueError(f"{path}: missing/non-finite {key} at step {row['step']}")
        if row["perf/total_num_tokens"] <= 0 or any(row[f"timing_s/{name}"] <= 0 for name in TIMINGS):
            raise ValueError(f"{path}: non-positive token count or timing at step {row['step']}")
    return rows


def summarize(rows: list[dict[str, float]], gpus: int) -> dict:
    tokens = sum(row["perf/total_num_tokens"] for row in rows)
    result = {
        "steps": [row["step"] for row in rows],
        "total_tokens": tokens,
        "timings": {},
        "nonzero_advantage_steps": sum(
            row["critic/advantages/max"] != 0 or row["critic/advantages/min"] != 0 for row in rows
        ),
    }
    for name in TIMINGS:
        seconds = [row[f"timing_s/{name}"] for row in rows]
        result["timings"][name] = {
            "total_s": sum(seconds),
            "mean_s": statistics.mean(seconds),
            "median_s": statistics.median(seconds),
            "min_s": min(seconds),
            "max_s": max(seconds),
            "tokens_per_s_cluster": tokens / sum(seconds),
            "tokens_per_s_per_gpu": tokens / sum(seconds) / gpus,
        }
    for name in MEMORY:
        result[f"lifetime_max_memory_{name}_gib"] = max(row[f"actor/perf/max_memory_{name}_gb"] for row in rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unfused", type=Path)
    parser.add_argument("fused", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.gpus < 1 or not 0 <= args.warmup < args.steps:
        parser.error("require gpus > 0 and 0 <= warmup < steps")
    unfused = read_steps(args.unfused, args.steps)
    fused = read_steps(args.fused, args.steps)
    for left, right in zip(unfused, fused, strict=True):
        if left["perf/total_num_tokens"] != right["perf/total_num_tokens"]:
            raise ValueError(f"token count differs at step {left['step']}; check rollout cache")
    output = {
        "note": "Fixed-rollout replay: step TPS is not online end-to-end RL throughput.",
        "gpus": args.gpus,
        "warmup_steps": args.warmup,
        "unfused": {"source": str(args.unfused), "steps": unfused},
        "fused": {"source": str(args.fused), "steps": fused},
        "comparison": {},
    }
    for window, start in (("all", 0), ("warm", args.warmup)):
        left = summarize(unfused[start:], args.gpus)
        right = summarize(fused[start:], args.gpus)
        output["unfused"][window] = left
        output["fused"][window] = right
        output["comparison"][window] = {
            name: {
                "time_reduction_pct": 100 * (1 - right["timings"][name]["total_s"] / left["timings"][name]["total_s"]),
                "throughput_increase_pct": 100
                * (left["timings"][name]["total_s"] / right["timings"][name]["total_s"] - 1),
            }
            for name in TIMINGS
        }
    output["max_abs_metric_differences"] = {
        key: max(abs(left[key] - right[key]) for left, right in zip(unfused, fused, strict=True)) for key in NUMERICS
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
