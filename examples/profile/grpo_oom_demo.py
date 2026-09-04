# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare tiny GRPO inputs and validate artifacts from the intentional CUDA OOM demo."""

import argparse
import pickle
import re
from pathlib import Path


def prepare(run_dir: Path) -> None:
    """Check CUDA prerequisites and write synthetic arithmetic data in the GSM8K schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError("This example requires one node with 8 visible CUDA GPUs (e.g. 8 x H20 141 GB)")
    if not hasattr(torch._C, "_cuda_attach_out_of_memory_observer"):
        raise RuntimeError("This PyTorch build does not provide the CUDA OOM observer")
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError("This example requires PyTorch's native CUDA caching allocator")
    for index in range(8):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}, {props.total_memory / 1024**3:.1f} GiB", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    # This is generated test data, not the actual GSM8K dataset. Reuse its rule-based
    # reward route to avoid downloading a dataset or starting a reward model.
    rows = [
        {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": f"What is {i} + 7? Give the final answer as #### <number>."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(i + 7)},
            "extra_info": {"index": i},
        }
        for i in range(64)
    ]
    pq.write_table(pa.Table.from_pylist(rows), run_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows[:8]), run_dir / "test.parquet")


def read_snapshot(path: Path) -> list[dict]:
    """Validate a locally generated snapshot and return its flattened allocator trace."""
    # Pickle is executable input: only load trusted artifacts produced by this run.
    with path.open("rb") as stream:
        snapshot = pickle.load(stream)
    if not isinstance(snapshot, dict) or "segments" not in snapshot or "device_traces" not in snapshot:
        raise ValueError(f"Not a CUDA allocator snapshot: {path}")
    events = [event for trace in snapshot["device_traces"] for event in trace]
    if not events or not any(event.get("frames") for event in events):
        raise ValueError(f"Missing allocation history/stack frames: {path}")
    return events


def check(run_dir: Path, trainer_exit_code: int) -> None:
    """Require the intentional failure, both snapshot types, and OOM diagnostic logs."""
    if trainer_exit_code == 0:
        raise ValueError("Training unexpectedly succeeded; the intentional OOM was not exercised")
    log = (run_dir / "train.log").read_text(errors="replace")
    for marker in (
        "GRPO_OOM_DEMO: allocator OOM observed",
        "GRPO_OOM_DEMO: intentional stop after allocator probe",
        "[torch_memory] CUDA OOM on device",
        "[torch_memory] Python stack at OOM:",
        "[torch_memory] allocator memory at OOM:",
    ):
        if marker not in log:
            raise ValueError(f"Missing log marker: {marker}")
    request = re.search(r"GRPO_OOM_DEMO: rank=0 requesting=(\d+) bytes at loss_call=3", log)
    if request is None:
        raise ValueError("Missing intentional allocation size in train.log")
    requested = int(request.group(1))
    snapshots = run_dir / "snapshots"
    normal = sorted(snapshots.glob("steps1-2/torch_memory_rank0_pid*.pickle"))
    oom = sorted(snapshots.glob("oom_*/torch_memory_oom_rank0_pid*.pickle"))
    if not normal or not oom:
        raise ValueError("Expected both steps1-2/torch_memory_*.pickle and oom_*/torch_memory_oom_*.pickle")
    if list(snapshots.glob("step[12]/torch_memory_rank0_pid*.pickle")):
        raise ValueError("Unexpected per-step dump: the two-step memory history window was not applied")
    for path in normal:
        read_snapshot(path)
    normal_processes = {path.name.removeprefix("torch_memory_") for path in normal}
    for path in oom:
        if path.name.removeprefix("torch_memory_oom_") not in normal_processes:
            continue
        events = read_snapshot(path)
        if any(event.get("action") == "oom" and event.get("size", 0) >= requested for event in events):
            print(f"PASS: two-step history + real allocator OOM + diagnostic logs verified in {run_dir}")
            print(f"Window snapshot: {normal[0]}\nOOM snapshot: {path}")
            return
    raise ValueError("No matching actor-process OOM snapshot contains the intentional oversized allocation")


def main() -> None:
    """Dispatch preparation or validation without importing GPU dependencies for the checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep_parser = commands.add_parser("prepare")
    prep_parser.add_argument("--run-dir", type=Path, required=True)
    check_parser = commands.add_parser("check", help="Only use with trusted, locally generated pickle files")
    check_parser.add_argument("--run-dir", type=Path, required=True)
    check_parser.add_argument("--trainer-exit-code", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run_dir)
    else:
        check(args.run_dir, args.trainer_exit_code)


if __name__ == "__main__":
    main()
