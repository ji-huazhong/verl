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

"""Example-only policy loss: trigger a real allocator OOM on the third actor update.

Loaded through model.external_lib; never use this loss for production training.
The companion launcher fixes batching to one loss call per rank per GRPO step.
"""

import torch
import torch.distributed as dist

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, register_policy_loss

_loss_calls = 0


def trigger_cuda_oom(device: torch.device) -> None:
    """Probe rank 0's allocator, then stop all ranks after its callback has returned."""
    # Allocate the collective buffer before the probe. No other rank enters backward
    # (or exits and causes Ray to tear down rank 0) while the snapshot is being written.
    observed = torch.zeros(1, dtype=torch.int32, device=device)
    if dist.get_rank() == 0:
        requested = torch.cuda.get_device_properties(device).total_memory + 1024**3
        print(f"GRPO_OOM_DEMO: rank=0 requesting={requested} bytes at loss_call=3", flush=True)
        try:
            # An allocation larger than the *total* device capacity fails even on a
            # 141 GB H20. Raising OutOfMemoryError manually would NOT call the observer.
            allocation = torch.empty(requested, dtype=torch.uint8, device=device)
            del allocation
        except torch.OutOfMemoryError:
            observed.fill_(1)
            print("GRPO_OOM_DEMO: allocator OOM observed", flush=True)
        except Exception as exc:
            print(f"GRPO_OOM_DEMO: unexpected allocation error: {exc}", flush=True)

    dist.broadcast(observed, src=0)
    if observed.item() != 1:
        raise RuntimeError("GRPO_OOM_DEMO: probe did not produce the expected allocator OOM")
    raise RuntimeError("GRPO_OOM_DEMO: intentional stop after allocator probe")


@register_policy_loss("grpo_oom_demo")
def compute_policy_loss_with_oom(log_prob: torch.Tensor, config=None, **kwargs):
    """Run vanilla PPO loss twice, then inject an OOM with the launcher's fixed batching."""
    global _loss_calls
    if (
        not dist.is_initialized()
        or dist.get_world_size() != 8
        or config is None
        or config.ppo_mini_batch_size != 8
        or config.ppo_micro_batch_size_per_gpu != 2
        or config.ppo_epochs != 1
        or config.rollout_n != 2
        or config.use_dynamic_bsz
        or config.engine.ulysses_sequence_parallel_size != 1
    ):
        raise ValueError("Use grpo_oom_demo only with the companion launcher's fixed 8-GPU batching")
    _loss_calls += 1
    if _loss_calls == 3:
        trigger_cuda_oom(log_prob.device)
    return compute_policy_loss_vanilla(log_prob=log_prob, config=config, **kwargs)
