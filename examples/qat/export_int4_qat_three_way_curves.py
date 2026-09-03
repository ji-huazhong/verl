#!/usr/bin/env python3
"""Validate and plot a matched BF16 / PTQ-INT4 / QAT-INT4 RL experiment.

The three inputs are TensorBoard directories.  A resumed run may contain
multiple event files and replay a few training steps; the newest event by wall
time is retained for each (metric, step), so the resulting curve represents the
latest successful continuation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = {
    "reward": ("critic/rewards/mean", "Reward mean"),
    "aime_acc": ("val-core/math_dapo/acc/mean@1", "AIME accuracy@1"),
    "logprob_abs_diff": (
        "rollout_corr/log_ppl_abs_diff",
        "Mean absolute rollout/train logprob gap",
    ),
}
RUNS = ("BF16", "PTQ-INT4", "QAT-INT4")
COLORS = {"BF16": "#1f77b4", "PTQ-INT4": "#ff7f0e", "QAT-INT4": "#d62728"}


@dataclass(frozen=True)
class Point:
    step: int
    wall_time: float
    value: float


def event_files(tensorboard_dir: Path) -> list[Path]:
    files = sorted(tensorboard_dir.rglob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files under {tensorboard_dir}")
    return files


def read_metric(tensorboard_dir: Path, tag: str) -> list[Point]:
    """Read every event file and retain the latest write for each step."""
    latest_by_step: dict[int, Point] = {}
    for event_file in event_files(tensorboard_dir):
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for scalar in accumulator.Scalars(tag):
            point = Point(step=scalar.step, wall_time=scalar.wall_time, value=scalar.value)
            previous = latest_by_step.get(point.step)
            if previous is None or point.wall_time >= previous.wall_time:
                latest_by_step[point.step] = point
    return [latest_by_step[step] for step in sorted(latest_by_step)]


def collect_run(tensorboard_dir: Path) -> dict[str, list[Point]]:
    return {metric: read_metric(tensorboard_dir, tag) for metric, (tag, _) in METRICS.items()}


def validation_errors(curves: dict[str, dict[str, list[Point]]], total_steps: int) -> list[str]:
    errors: list[str] = []
    for run_name, run_curves in curves.items():
        for metric_name, points in run_curves.items():
            if not points:
                errors.append(f"{run_name}: missing {METRICS[metric_name][0]}")
                continue
            final_step = points[-1].step
            if final_step < total_steps:
                errors.append(f"{run_name}: {metric_name} ends at step {final_step}, expected {total_steps}")
    return errors


def write_outputs(
    output_dir: Path,
    input_dirs: dict[str, Path],
    curves: dict[str, dict[str, list[Point]]],
    errors: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_dirs": {name: str(path) for name, path in input_dirs.items()},
        "metric_tags": {name: tag for name, (tag, _) in METRICS.items()},
        "validation_errors": errors,
        "curves": {
            run_name: {
                metric_name: [asdict(point) for point in points]
                for metric_name, points in run_curves.items()
            }
            for run_name, run_curves in curves.items()
        },
    }
    with (output_dir / "three_way_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    with (output_dir / "three_way_metric_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "metric", "tag", "step", "wall_time", "value"))
        writer.writeheader()
        for run_name, run_curves in curves.items():
            for metric_name, points in run_curves.items():
                for point in points:
                    writer.writerow(
                        {
                            "run": run_name,
                            "metric": metric_name,
                            "tag": METRICS[metric_name][0],
                            "step": point.step,
                            "wall_time": point.wall_time,
                            "value": point.value,
                        }
                    )


def plot_curves(output_dir: Path, curves: dict[str, dict[str, list[Point]]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=len(METRICS), ncols=1, figsize=(10, 11), sharex=True, layout="constrained")
    for axis, (metric_name, (_, metric_label)) in zip(axes, METRICS.items(), strict=True):
        for run_name in RUNS:
            points = curves[run_name][metric_name]
            if points:
                axis.plot(
                    [point.step for point in points],
                    [point.value for point in points],
                    label=run_name,
                    color=COLORS[run_name],
                    linewidth=1.8,
                )
        axis.set_ylabel(metric_label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Training step")
    fig.suptitle("Qwen3-30B-A3B: BF16 / PTQ-INT4 / QAT-INT4 RL")
    fig.savefig(output_dir / "three_way_curves.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-dir", type=Path, required=True, help="BF16 TensorBoard directory")
    parser.add_argument("--ptq-int4-dir", type=Path, required=True, help="PTQ-INT4 TensorBoard directory")
    parser.add_argument("--qat-int4-dir", type=Path, required=True, help="QAT-INT4 TensorBoard directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--allow-incomplete", action="store_true", help="Write partial data instead of failing validation")
    parser.add_argument("--no-plot", action="store_true", help="Only write JSON and CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs = {
        "BF16": args.bf16_dir,
        "PTQ-INT4": args.ptq_int4_dir,
        "QAT-INT4": args.qat_int4_dir,
    }
    curves = {run_name: collect_run(path) for run_name, path in input_dirs.items()}
    errors = validation_errors(curves, args.total_steps)
    write_outputs(args.output_dir, input_dirs, curves, errors)
    if not args.no_plot:
        plot_curves(args.output_dir, curves)
    if errors:
        print("Three-way curve validation failed:", *errors, sep="\n- ", file=sys.stderr)
        return 0 if args.allow_incomplete else 1
    print(f"Validated three {args.total_steps}-step curves in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
