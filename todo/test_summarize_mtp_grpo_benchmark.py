"""Standard-library checks for the temporary benchmark report tool."""

import tempfile
import unittest
from pathlib import Path

from summarize_mtp_grpo_benchmark import MEMORY, NUMERICS, TIMINGS, read_steps, summarize


def row(step, tokens, seconds):
    return {
        "step": step,
        "perf/total_num_tokens": tokens,
        **{f"timing_s/{name}": seconds for name in TIMINGS},
        **{f"actor/perf/max_memory_{name}_gb": 40.0 for name in MEMORY},
        **dict.fromkeys(NUMERICS, 0.0),
    }


def line(metrics):
    return "\x1b[36m(TaskRunner pid=1)\x1b[0m " + " - ".join(f"{k}:{v}" for k, v in metrics.items())


class BenchmarkSummaryTest(unittest.TestCase):
    def test_rate_is_ratio_of_sums(self):
        result = summarize([row(1, 100, 1), row(2, 100, 9)], 2)
        self.assertEqual(result["timings"]["step"]["tokens_per_s_cluster"], 20)
        self.assertEqual(result["timings"]["step"]["tokens_per_s_per_gpu"], 10)

    def test_parse_console_and_require_complete_finite_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            first, second = row(1, 100, 1), row(2, 100, 9)
            path.write_text("config noise\n" + line(first) + "\n" + line(second))
            self.assertEqual(read_steps(path, 2), [first, second])
            with self.assertRaisesRegex(ValueError, "expected steps"):
                read_steps(path, 3)
            path.write_text(line(first) + "\n" + line(first))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_steps(path, 2)
            first["actor/grad_norm"] = float("nan")
            path.write_text(line(first))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_steps(path, 1)

    def test_nonzero_advantages_and_lifetime_peak(self):
        first, second = row(1, 100, 1), row(2, 100, 9)
        second["critic/advantages/max"] = 0.7
        second["actor/perf/max_memory_allocated_gb"] = 42
        result = summarize([first, second], 8)
        self.assertEqual(result["nonzero_advantage_steps"], 1)
        self.assertEqual(result["lifetime_max_memory_allocated_gib"], 42)


if __name__ == "__main__":
    unittest.main()
