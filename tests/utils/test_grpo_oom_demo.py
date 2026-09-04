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

"""CPU-only tests for the intentional OOM example's artifact acceptance criteria."""

import contextlib
import importlib.util
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from examples.profile.grpo_oom_demo import check


class TestGrpoOomDemoChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_dir = Path(self.temp_dir.name)
        self.normal = self.run_dir / "snapshots/steps1-2/torch_memory_rank0_pid123.pickle"
        self.oom = self.run_dir / "snapshots/oom_test/torch_memory_oom_rank0_pid123.pickle"
        self.write_snapshot(self.normal, "alloc", 64)
        self.write_snapshot(self.oom, "oom", 1024)
        self.log = (
            "GRPO_OOM_DEMO: rank=0 requesting=1024 bytes at loss_call=3\n"
            "GRPO_OOM_DEMO: allocator OOM observed\n"
            "GRPO_OOM_DEMO: intentional stop after allocator probe\n"
            "[torch_memory] CUDA OOM on device 0\n"
            "[torch_memory] Python stack at OOM:\n"
            "[torch_memory] allocator memory at OOM:\n"
        )
        (self.run_dir / "train.log").write_text(self.log)

    def write_snapshot(self, path, action, size, frames=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {"action": action, "size": size, "frames": [{"name": "test"}] if frames else []}
        with path.open("wb") as stream:
            pickle.dump({"segments": [], "device_traces": [[event]]}, stream)

    def test_expected_oom_passes(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            check(self.run_dir, 1)
        self.assertIn("PASS", output.getvalue())

    def test_successful_training_is_not_a_pass(self):
        with self.assertRaisesRegex(ValueError, "unexpectedly succeeded"):
            check(self.run_dir, 0)

    def test_unrelated_failure_is_not_a_pass(self):
        (self.run_dir / "train.log").write_text("Model download failed")
        with self.assertRaisesRegex(ValueError, "Missing log marker"):
            check(self.run_dir, 1)

    def test_both_snapshot_types_are_required(self):
        for path in (self.normal, self.oom):
            with self.subTest(path=path):
                contents = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(ValueError, "Expected both"):
                    check(self.run_dir, 1)
                path.write_bytes(contents)

    def test_oom_must_come_from_same_actor_process(self):
        self.oom.rename(self.oom.with_name("torch_memory_oom_rank0_pid456.pickle"))
        with self.assertRaisesRegex(ValueError, "No matching actor-process"):
            check(self.run_dir, 1)

    def test_real_oversized_oom_event_is_required(self):
        for action, size in (("alloc", 1024), ("oom", 512)):
            with self.subTest(action=action, size=size):
                self.write_snapshot(self.oom, action, size)
                with self.assertRaisesRegex(ValueError, "No matching actor-process"):
                    check(self.run_dir, 1)

    def test_history_frames_are_required(self):
        self.write_snapshot(self.normal, "alloc", 64, frames=False)
        with self.assertRaisesRegex(ValueError, "Missing allocation history/stack frames"):
            check(self.run_dir, 1)

    def test_intermediate_step_dump_is_rejected(self):
        self.write_snapshot(self.run_dir / "snapshots/step1/torch_memory_rank0_pid123.pickle", "alloc", 64)
        with self.assertRaisesRegex(ValueError, "Unexpected per-step dump"):
            check(self.run_dir, 1)

    def test_both_diagnostic_logs_are_required(self):
        for marker in ("Python stack at OOM:", "allocator memory at OOM:"):
            with self.subTest(marker=marker):
                (self.run_dir / "train.log").write_text(self.log.replace(marker, "missing"))
                with self.assertRaisesRegex(ValueError, "Missing log marker"):
                    check(self.run_dir, 1)


class TestGrpoOomLoss(unittest.TestCase):
    """Mock CUDA only at the boundary; never allocate device memory in these tests."""

    def setUp(self):
        self.torch = MagicMock()
        self.torch.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        self.dist = self.torch.distributed
        self.dist.is_initialized.return_value = True
        self.dist.get_world_size.return_value = 8
        self.dist.get_rank.return_value = 0
        self.algos = MagicMock()
        self.algos.register_policy_loss.side_effect = lambda name: lambda function: function
        module_path = Path(__file__).resolve().parents[2] / "examples/profile/grpo_oom_loss.py"
        spec = importlib.util.spec_from_file_location("grpo_oom_loss_test", module_path)
        self.loss = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {"torch": self.torch, "torch.distributed": self.dist, "verl.trainer.ppo.core_algos": self.algos},
        ):
            spec.loader.exec_module(self.loss)
        self.config = SimpleNamespace(
            ppo_mini_batch_size=8,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            rollout_n=2,
            use_dynamic_bsz=False,
            engine=SimpleNamespace(ulysses_sequence_parallel_size=1),
        )

    def test_third_loss_call_injects_after_two_vanilla_updates(self):
        log_prob = MagicMock()
        with patch.object(self.loss, "trigger_cuda_oom", side_effect=RuntimeError("intentional stop")) as inject:
            for _ in range(2):
                self.loss.compute_policy_loss_with_oom(log_prob, self.config, old_log_prob="old")
            inject.assert_not_called()
            self.assertEqual(self.algos.compute_policy_loss_vanilla.call_count, 2)
            with self.assertRaisesRegex(RuntimeError, "intentional stop"):
                self.loss.compute_policy_loss_with_oom(log_prob, self.config)
            inject.assert_called_once_with(log_prob.device)

    def test_dynamic_batching_is_rejected(self):
        self.config.use_dynamic_bsz = True
        with self.assertRaisesRegex(ValueError, "fixed 8-GPU batching"):
            self.loss.compute_policy_loss_with_oom(MagicMock(), self.config)

    def test_rank_zero_requests_more_than_total_capacity_before_broadcast(self):
        total = 141 * 1024**3
        self.torch.cuda.get_device_properties.return_value.total_memory = total
        observed = self.torch.zeros.return_value
        observed.item.return_value = 1
        order = []

        def allocate(*args, **kwargs):
            order.append("allocate")
            raise self.torch.OutOfMemoryError("allocator OOM")

        self.torch.empty.side_effect = allocate
        observed.fill_.side_effect = lambda value: order.append("observer returned")
        self.dist.broadcast.side_effect = lambda *args, **kwargs: order.append("broadcast")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "intentional stop"):
            self.loss.trigger_cuda_oom("cuda:0")
        self.torch.empty.assert_called_once_with(total + 1024**3, dtype=self.torch.uint8, device="cuda:0")
        self.assertEqual(order, ["allocate", "observer returned", "broadcast"])

    def test_other_ranks_wait_without_allocating(self):
        self.dist.get_rank.return_value = 1
        self.torch.zeros.return_value.item.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "intentional stop"):
            self.loss.trigger_cuda_oom("cuda:0")
        self.torch.empty.assert_not_called()
        self.dist.broadcast.assert_called_once_with(self.torch.zeros.return_value, src=0)

    def test_unexpected_allocation_error_is_not_an_expected_oom(self):
        self.torch.cuda.get_device_properties.return_value.total_memory = 141 * 1024**3
        self.torch.empty.side_effect = ValueError("not an OOM")
        self.torch.zeros.return_value.item.return_value = 0
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "did not produce"):
            self.loss.trigger_cuda_oom("cuda:0")
        self.dist.broadcast.assert_called_once()


if __name__ == "__main__":
    unittest.main()
