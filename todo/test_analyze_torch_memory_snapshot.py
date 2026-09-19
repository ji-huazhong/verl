"""Synthetic trace checks; no CUDA runtime is required."""

import contextlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path

from analyze_torch_memory_snapshot import GIB, analyze


class PhaseMemoryTest(unittest.TestCase):
    def test_event_interval_includes_worker_but_excludes_later_free(self):
        events = [
            ("alloc", 1, 8, "initialize"),
            ("free_requested", 1, 8, "initialize"),
            ("alloc", 2, 2, "train_batch"),
            ("alloc", 3, 4, "backward_worker"),
            ("free_requested", 3, 4, "backward_worker"),
            ("alloc", 4, 1, "train_batch"),
            ("alloc", 5, 12, "outside_training"),
            # Original allocation stack contains train_batch, this event does not.
            ("free_requested", 2, 2, "cleanup"),
        ]
        trace = [
            {
                "action": action,
                "addr": address,
                "size": size * GIB,
                "time_us": index * 1000,
                "frames": [{"filename": "/workspace/verl/test.py", "name": name}],
            }
            for index, (action, address, size, name) in enumerate(events)
        ]
        snapshot = {
            "device_traces": [trace],
            "segments": [
                {
                    "total_size": 13 * GIB,
                    "blocks": [
                        {"state": "active_allocated", "address": 4, "size": GIB, "frames": trace[5]["frames"]},
                        {"state": "active_allocated", "address": 5, "size": 12 * GIB, "frames": trace[6]["frames"]},
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trusted.pickle"
            with path.open("wb") as stream:
                pickle.dump(snapshot, stream)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                analyze(path, "test", 2, ["::train_batch"], 100, ["::train_batch", "::missing"])
        result = output.getvalue()
        self.assertIn("enclosing_event_interval=[2,5] max_allocated_gib=6.000000", result)
        self.assertIn("global_allocated_at_matched_event_max_gib=15.000000", result)
        self.assertIn("phase_stack_match='::missing': no matching event stacks", result)


if __name__ == "__main__":
    unittest.main()
