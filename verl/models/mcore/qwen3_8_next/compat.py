# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Compatibility fixes for the explicitly pinned Megatron Core runtime."""

from functools import wraps
from importlib.metadata import version


def install_cpu_optimizer_resume_fix():
    """Restore Core 0.19.2's CPU optimizer after loading DP-sharded state.

    Its DP loader passes a LocalNonpersistentObject `step` from the running
    optimizer into the state setter, overwriting the checkpoint's step from
    param_groups. It also replaces the master/moment tensors without syncing
    them back to HybridDeviceOptimizer's CPU sub-optimizers. The model weights
    restore correctly, but the next update differs from uninterrupted training.
    """
    if version("megatron-core").split("+")[0] != "0.19.2":
        return
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    original_load = DistributedOptimizer.load_parameter_state_from_dp_reshardable
    if getattr(original_load, "_qwen38_cpu_resume_fix", False):
        return
    original_set = DistributedOptimizer._set_main_param_and_optimizer_states

    @wraps(original_set)
    def set_parameter_state(self, model_param, tensors):
        if isinstance(self.optimizer, HybridDeviceOptimizer):
            # Adam's scalar step is restored through param_groups, never the
            # local, nonpersistent placeholder in the tensor shard template.
            tensors = {key: value for key, value in tensors.items() if key != "step"}
        return original_set(self, model_param, tensors)

    @wraps(original_load)
    def load_parameter_state(self, state_dict):
        result = original_load(self, state_dict)
        if isinstance(self.optimizer, HybridDeviceOptimizer):
            self.optimizer._sync_hdo_state_to_sub_optimizers()
        return result

    load_parameter_state._qwen38_cpu_resume_fix = True
    DistributedOptimizer._set_main_param_and_optimizer_states = set_parameter_state
    DistributedOptimizer.load_parameter_state_from_dp_reshardable = load_parameter_state
