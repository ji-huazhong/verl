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

import unittest
from unittest.mock import MagicMock, patch

from verl.utils.profiler.config import ProfilerConfig, TorchMemoryToolConfig
from verl.utils.profiler.torch_memory_profile import TorchMemoryProfiler


class TestTorchMemoryProfiler(unittest.TestCase):
    def _config(self) -> ProfilerConfig:
        return ProfilerConfig(enable=True, ranks=[0], save_path="/tmp/profiles")

    def test_dump_on_oom_registers_observer_and_dumps_without_sync(self):
        tool_config = TorchMemoryToolConfig(dump_on_oom=True)
        attach_observer = MagicMock()

        with (
            patch("verl.utils.profiler.torch_memory_profile.enable_memory_visualize"),
            patch("verl.utils.profiler.torch_memory_profile.torch.cuda.is_available", return_value=True),
            patch(
                "verl.utils.profiler.torch_memory_profile.torch._C._cuda_attach_out_of_memory_observer",
                attach_observer,
                create=True,
            ),
            patch.object(TorchMemoryProfiler, "_memory_history_enabled", False),
            patch.object(TorchMemoryProfiler, "_oom_observer_attached", False),
        ):
            profiler = TorchMemoryProfiler(rank=0, config=self._config(), tool_config=tool_config)

            attach_observer.assert_called_once()
            observer = attach_observer.call_args.args[0]
            with (
                patch("verl.utils.profiler.torch_memory_profile.traceback.format_stack", return_value=["stack"]),
                patch(
                    "verl.utils.profiler.torch_memory_profile.get_memory_info", return_value={"allocated": 1234}
                ) as memory_info,
                patch.object(profiler.sampler, "dump_memory_snapshot") as dump_snapshot,
            ):
                observer(0, 4096, 1234, 5678)

            dump_snapshot.assert_called_once()
            memory_info.assert_called_once()
            kwargs = dump_snapshot.call_args.kwargs
            self.assertEqual(kwargs["out_dir"], "/tmp/profiles")
            self.assertEqual(kwargs["tag"], "torch_memory_oom")
            self.assertTrue(kwargs["sub_dir"].startswith("oom_"))
            self.assertFalse(kwargs["synchronize"])

    def test_dump_on_oom_can_be_disabled(self):
        with (
            patch("verl.utils.profiler.torch_memory_profile.enable_memory_visualize"),
            patch(
                "verl.utils.profiler.torch_memory_profile.torch._C._cuda_attach_out_of_memory_observer",
                create=True,
            ) as attach_observer,
            patch.object(TorchMemoryProfiler, "_memory_history_enabled", False),
            patch.object(TorchMemoryProfiler, "_oom_observer_attached", False),
        ):
            TorchMemoryProfiler(rank=0, config=self._config(), tool_config=TorchMemoryToolConfig(dump_on_oom=False))

        attach_observer.assert_not_called()

    def test_snapshot_window_keeps_history_until_the_configured_step_count(self):
        tool_config = TorchMemoryToolConfig(dump_on_oom=False, memory_snapshot_num_steps=2)
        with (
            patch("verl.utils.profiler.torch_memory_profile.enable_memory_visualize"),
            patch("verl.utils.profiler.torch_memory_profile.clear_memory_history") as clear_memory_history,
            patch.object(TorchMemoryProfiler, "_memory_history_enabled", False),
        ):
            profiler = TorchMemoryProfiler(rank=0, config=self._config(), tool_config=tool_config)
            with patch.object(profiler.sampler, "dump_memory_snapshot") as dump_snapshot:
                profiler.start(profile_step=4)
                profiler.stop()
                dump_snapshot.assert_not_called()
                clear_memory_history.assert_not_called()

                profiler.start(profile_step=5)
                profiler.stop()

            dump_snapshot.assert_called_once_with(
                out_dir="/tmp/profiles", tag="torch_memory", sub_dir="steps4-5"
            )
            clear_memory_history.assert_called_once_with(trace_alloc_max_entries=100_000, stack_depth=32)
