#!/usr/bin/env python3
"""Summarize a trusted PyTorch CUDA allocator snapshot.

The allocator trace is a ring buffer.  When it is full, a forward replay no
longer knows the live allocations at the first retained event.  This tool
anchors the replay at the blocks stored in the snapshot and walks the trace
backwards, so allocated bytes remain absolute throughout the retained window.

Only open snapshots produced by a trusted process: the input is a pickle.
"""

from __future__ import annotations

import argparse
import collections
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GIB = 1024**3


@dataclass
class Allocation:
    address: int
    size: int
    frames: list[dict[str, Any]]
    origin: str
    matches: frozenset[str] = field(default_factory=frozenset)


def _site(frames: list[dict[str, Any]]) -> str:
    for frame in frames:
        filename = frame.get("filename", "")
        if "/workspace/verl" in filename or "/workspace/Megatron" in filename:
            return f"{filename}:{frame.get('line', 0)}::{frame.get('name', '?')}"
    for frame in frames:
        filename = frame.get("filename", "")
        if filename not in {"", "??"} and not filename.endswith(".cpp"):
            return f"{filename}:{frame.get('line', 0)}::{frame.get('name', '?')}"
    return "<native-or-unknown>"


def _end_allocations(snapshot: dict[str, Any]) -> dict[int, Allocation]:
    result: dict[int, Allocation] = {}
    for segment in snapshot["segments"]:
        for block in segment["blocks"]:
            if block["state"] != "active_allocated":
                continue
            result[block["address"]] = Allocation(
                address=block["address"],
                size=block["size"],
                frames=block.get("frames", []),
                origin="snapshot-end",
            )
    return result


def _pair_frees(trace: list[dict[str, Any]]) -> dict[int, Allocation]:
    """Associate each free_requested event with its preceding allocation."""
    current: dict[int, Allocation] = {}
    result: dict[int, Allocation] = {}
    for index, event in enumerate(trace):
        action = event["action"]
        address = event.get("addr")
        if action == "alloc":
            current[address] = Allocation(
                address=address,
                size=event["size"],
                frames=event.get("frames", []),
                origin=f"trace-alloc-{index}",
            )
        elif action == "free_requested":
            allocation = current.pop(address, None)
            if allocation is None:
                allocation = Allocation(
                    address=address,
                    size=event["size"],
                    frames=[],
                    origin="allocated-before-retained-trace",
                )
            result[index] = allocation
    return result


def _tag_matches(allocations: list[Allocation], patterns: list[str]) -> None:
    lowered = {pattern: pattern.lower() for pattern in patterns}
    for allocation in allocations:
        stack = "\n".join(
            f"{frame.get('filename', '')}::{frame.get('name', '')}" for frame in allocation.frames
        ).lower()
        allocation.matches = frozenset(pattern for pattern, value in lowered.items() if value in stack)


def analyze(path: Path, label: str, top: int, matches: list[str], max_entries: int | None) -> None:
    with path.open("rb") as stream:
        snapshot = pickle.load(stream)  # noqa: S301 - snapshots are trusted local artifacts

    traces = snapshot["device_traces"]
    if len(traces) != 1:
        raise ValueError(f"expected exactly one device trace, got {len(traces)}")
    trace = traces[0]
    if not trace:
        raise ValueError("snapshot contains no allocator events")

    live = _end_allocations(snapshot)
    paired_frees = _pair_frees(trace)
    unique_allocations = {id(item): item for item in [*live.values(), *paired_frees.values()]}
    _tag_matches(list(unique_allocations.values()), matches)
    allocated = sum(item.size for item in live.values())
    end_allocated = allocated
    max_allocated = allocated
    min_allocated = allocated
    max_index = len(trace)
    max_live = dict(live)
    matched_live = {pattern: sum(item.size for item in live.values() if pattern in item.matches) for pattern in matches}
    matched_max = dict(matched_live)

    for index in range(len(trace) - 1, -1, -1):
        event = trace[index]
        action = event["action"]
        address = event.get("addr")
        if action == "alloc":
            allocation = live.pop(address, None)
            allocation_size = allocation.size if allocation is not None else event["size"]
            allocated -= allocation_size
            if allocation is not None:
                for pattern in allocation.matches:
                    matched_live[pattern] -= allocation_size
        elif action == "free_requested":
            allocation = paired_frees[index]
            live[address] = allocation
            allocated += allocation.size
            for pattern in allocation.matches:
                matched_live[pattern] += allocation.size

        for pattern in matches:
            matched_max[pattern] = max(matched_max[pattern], matched_live[pattern])

        min_allocated = min(min_allocated, allocated)
        if allocated > max_allocated:
            max_allocated = allocated
            max_index = index
            max_live = dict(live)

    initial_allocated = allocated

    action_counts = collections.Counter(event["action"] for event in trace)
    duration_seconds = (trace[-1]["time_us"] - trace[0]["time_us"]) / 1e6
    end_reserved = sum(segment["total_size"] for segment in snapshot["segments"])
    unknown_bytes = sum(
        allocation.size for allocation in max_live.values() if allocation.origin == "allocated-before-retained-trace"
    )

    grouped: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for allocation in max_live.values():
        item = grouped[_site(allocation.frames)]
        item[0] += allocation.size
        item[1] += 1

    print(f"[{label}] {path}")
    cap_reached = "yes" if max_entries is not None and len(trace) == max_entries else "no"
    print(
        f"events={len(trace)} span={duration_seconds:.3f}s actions={dict(action_counts)} "
        f"trace_cap_reached={cap_reached if max_entries is not None else 'unknown'}"
    )
    print(
        f"retained_window_allocated_gib: min={min_allocated / GIB:.6f} "
        f"max={max_allocated / GIB:.6f} end={end_allocated / GIB:.6f}; "
        f"end_reserved_gib={end_reserved / GIB:.6f}; max_state_before_event={max_index}"
    )
    print(f"peak_live_allocations={len(max_live)} unknown_pre_trace_gib={unknown_bytes / GIB:.6f}")
    print(f"top_{top}_peak_allocation_sites:")
    for site, (size, count) in sorted(grouped.items(), key=lambda item: item[1][0], reverse=True)[:top]:
        print(f"  {size / GIB:10.6f} GiB  {count:6d} allocs  {site}")
    print(f"top_{top}_individual_peak_allocations:")
    for allocation in sorted(max_live.values(), key=lambda item: item.size, reverse=True)[:top]:
        print(
            f"  {allocation.size / GIB:10.6f} GiB  addr={allocation.address} "
            f"origin={allocation.origin}  {_site(allocation.frames)}"
        )

    allocations = [
        Allocation(
            address=event["addr"],
            size=event["size"],
            frames=event.get("frames", []),
            origin=f"trace-alloc-{index}",
        )
        for index, event in enumerate(trace)
        if event["action"] == "alloc"
    ]
    _tag_matches(allocations, matches)
    allocations_by_index = {
        int(allocation.origin.removeprefix("trace-alloc-")): allocation for allocation in allocations
    }
    forward_allocated = initial_allocated
    matched_global_max = {pattern: 0 for pattern in matches}
    for index, event in enumerate(trace):
        if event["action"] == "alloc":
            allocation = allocations_by_index[index]
            forward_allocated += allocation.size
            for pattern in allocation.matches:
                matched_global_max[pattern] = max(matched_global_max[pattern], forward_allocated)
        elif event["action"] == "free_requested":
            allocation = paired_frees[index]
            for pattern in allocation.matches:
                matched_global_max[pattern] = max(matched_global_max[pattern], forward_allocated)
            forward_allocated -= allocation.size

    for pattern in matches:
        selected = [allocation for allocation in allocations if pattern in allocation.matches]
        total = sum(allocation.size for allocation in selected)
        maximum = max((allocation.size for allocation in selected), default=0)
        print(
            f"match={pattern!r}: alloc_events={len(selected)} churn_gib={total / GIB:.6f} "
            f"largest_alloc_gib={maximum / GIB:.6f} max_live_gib={matched_max[pattern] / GIB:.6f} "
            f"global_allocated_at_matched_event_max_gib={matched_global_max[pattern] / GIB:.6f}"
        )
        grouped_matches: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
        for allocation in selected:
            item = grouped_matches[_site(allocation.frames)]
            item[0] += allocation.size
            item[1] += 1
            item[2] = max(item[2], allocation.size)
        for site, (size, count, largest) in sorted(grouped_matches.items(), key=lambda item: item[1][0], reverse=True)[
            :top
        ]:
            print(f"  {size / GIB:10.6f} GiB churn  {largest / GIB:10.6f} GiB largest  {count:6d} allocs  {site}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", nargs="+", type=Path)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--match", action="append", default=[])
    parser.add_argument("--max-entries", type=int)
    args = parser.parse_args()
    for index, path in enumerate(args.snapshot):
        label = args.label[index] if index < len(args.label) else path.stem
        analyze(path, label, args.top, args.match, args.max_entries)


if __name__ == "__main__":
    main()
