# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging

import torch
from torch.distributed._tensor import Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init

from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import FSDPModule, fsdp_version

logger = logging.getLogger(__file__)


@torch.no_grad()
def offload_veomni_model_to_cpu(model, empty_cache: bool = True):
    """
    Offload VeOmni model to CPU with Expert Parallelism (EP) support.

    VeOmni supports both FSDP1 and FSDP2 for model parallelism. When EP is enabled,
    expert parameters are sharded across EP ranks and stored as local tensors.
    This function ensures both regular FSDP parameters and EP parameters are
    properly offloaded to CPU.

    Args:
        model: VeOmni model wrapped with FSDP1 or FSDP2
        empty_cache: Whether to empty CUDA cache after offloading

    Note:
        - EP parameters are already local tensors (not DTensors) after EP sharding
        - FSDP1: Uses flat_param handles for offloading
        - FSDP2: Uses simple model.cpu() call
        - The model may have _fqn2spec_info attribute storing EP parameter mappings
    """
    # Check if EP is enabled
    try:
        from veomni.distributed.parallel_state import get_parallel_state

        parallel_state = get_parallel_state()
        ep_enabled = parallel_state.ep_enabled if parallel_state is not None else False
    except (ImportError, AttributeError, RuntimeError):
        # If parallel_state is not available or not initialized, assume EP is disabled
        ep_enabled = False

    # Determine FSDP version and offload accordingly
    fsdp_ver = fsdp_version(model)
    
    if fsdp_ver == 1:
        # FSDP1: Handle flat_param offloading
        assert isinstance(model, FSDP), "Model must be FSDP1 wrapped"
        # Lazy init FSDP model if needed
        _lazy_init(model, model)
        assert model._is_root, "Only support root model offloading to CPU"
        
        for handle in model._all_handles:
            if handle._offload_params:
                continue
            flat_param = handle.flat_param
            assert (
                flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
                and id(flat_param.data) != id(flat_param._local_shard)
                and flat_param.data.size() == flat_param._local_shard.size()
            )
            handle.flat_param_to(torch.device("cpu"), non_blocking=True)
            # The following still keeps id(._local_shard) != id(.data)
            flat_param._local_shard = flat_param.data
            assert id(flat_param._local_shard) != id(flat_param.data)
    elif fsdp_ver == 2:
        # FSDP2: Simple CPU offload
        model.cpu()
    else:
        # Fallback: try simple CPU offload
        logger.warning(f"Unknown FSDP version {fsdp_ver}, attempting simple model.cpu()")
        model.cpu()

    # Verify EP parameters are offloaded if EP is enabled
    if ep_enabled:
        ep_fqn2spec_info = getattr(model, "_fqn2spec_info", None)
        if ep_fqn2spec_info is not None:
            ep_param_count = 0
            ep_offload_issues = []
            
            # Use _fqn2spec_info to identify EP parameters (those with Shard placement)
            for fqn, spec_info in ep_fqn2spec_info.items():
                if isinstance(spec_info.placement, Shard):
                    # This is an EP parameter, verify it's offloaded
                    try:
                        # Get the parameter by FQN
                        param = dict(model.named_parameters())[fqn]
                        ep_param_count += 1
                        
                        # Verify EP parameter is on CPU
                        if param.device.type != "cpu":
                            ep_offload_issues.append(
                                f"EP parameter {fqn} is still on {param.device.type}, expected CPU"
                            )
                        
                        # Verify gradient is also offloaded if it exists
                        if param.grad is not None and param.grad.device.type != "cpu":
                            ep_offload_issues.append(
                                f"EP parameter {fqn} gradient is still on {param.grad.device.type}, expected CPU"
                            )
                    except KeyError:
                        # Parameter not found, might be wrapped by FSDP
                        # This is okay, FSDP will handle it
                        pass
            
            if ep_param_count > 0:
                logger.debug(
                    f"Verified offload of {ep_param_count} EP parameters to CPU "
                    f"(FSDP version: {fsdp_ver})"
                )
                if ep_offload_issues:
                    for issue in ep_offload_issues:
                        logger.warning(issue)
        else:
            logger.debug("EP is enabled but _fqn2spec_info not found on model")

    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_veomni_model_to_gpu(model, device_id=None):
    """
    Load VeOmni model from CPU to GPU with Expert Parallelism (EP) support.

    VeOmni supports both FSDP1 and FSDP2 for model parallelism. When EP is enabled,
    expert parameters are sharded across EP ranks and stored as local tensors.
    This function ensures both regular FSDP parameters and EP parameters are
    properly loaded to GPU.

    Args:
        model: VeOmni model wrapped with FSDP1 or FSDP2
        device_id: Target device ID (e.g., 0 for cuda:0). If None, uses get_device_id()

    Note:
        - EP parameters are already local tensors (not DTensors) after EP sharding
        - FSDP1: Uses flat_param handles for loading
        - FSDP2: Uses simple model.to(device) call
        - The model may have _fqn2spec_info attribute storing EP parameter mappings
    """
    if device_id is None:
        device_id = get_device_id()
    
    device = torch.device(f"{get_device_name()}:{device_id}")

    # Check if EP is enabled
    try:
        from veomni.distributed.parallel_state import get_parallel_state

        parallel_state = get_parallel_state()
        ep_enabled = parallel_state.ep_enabled if parallel_state is not None else False
    except (ImportError, AttributeError, RuntimeError):
        # If parallel_state is not available or not initialized, assume EP is disabled
        ep_enabled = False

    # Determine FSDP version and load accordingly
    fsdp_ver = fsdp_version(model)
    
    if fsdp_ver == 1:
        # FSDP1: Handle flat_param loading
        assert isinstance(model, FSDP), "Model must be FSDP1 wrapped"
        # Lazy init FSDP model if needed
        _lazy_init(model, model)
        assert model._is_root, "Only support root model loading to GPU"
        
        for handle in model._all_handles:
            if handle._offload_params:
                continue
            flat_param = handle.flat_param
            handle.flat_param_to(device, non_blocking=True)
            # The following still keeps id(._local_shard) != id(.data)
            flat_param._local_shard = flat_param.data
    elif fsdp_ver == 2:
        # FSDP2: Simple GPU load
        model.to(device)
    else:
        # Fallback: try simple GPU load
        logger.warning(f"Unknown FSDP version {fsdp_ver}, attempting simple model.to({device})")
        model.to(device)

    # Verify EP parameters are loaded if EP is enabled
    if ep_enabled:
        ep_fqn2spec_info = getattr(model, "_fqn2spec_info", None)
        if ep_fqn2spec_info is not None:
            ep_param_count = 0
            ep_load_issues = []
            
            # Use _fqn2spec_info to identify EP parameters (those with Shard placement)
            for fqn, spec_info in ep_fqn2spec_info.items():
                if isinstance(spec_info.placement, Shard):
                    # This is an EP parameter, verify it's loaded
                    try:
                        # Get the parameter by FQN
                        param = dict(model.named_parameters())[fqn]
                        ep_param_count += 1
                        
                        # Verify EP parameter is on GPU
                        if param.device.type != device.type:
                            ep_load_issues.append(
                                f"EP parameter {fqn} is on {param.device.type}, expected {device.type}"
                            )
                        
                        # Verify gradient is also on GPU if it exists
                        if param.grad is not None and param.grad.device.type != device.type:
                            ep_load_issues.append(
                                f"EP parameter {fqn} gradient is on {param.grad.device.type}, expected {device.type}"
                            )
                    except KeyError:
                        # Parameter not found, might be wrapped by FSDP
                        # This is okay, FSDP will handle it
                        pass
            
            if ep_param_count > 0:
                logger.debug(
                    f"Verified load of {ep_param_count} EP parameters to {device} "
                    f"(FSDP version: {fsdp_ver})"
                )
                if ep_load_issues:
                    for issue in ep_load_issues:
                        logger.warning(issue)
        else:
            logger.debug("EP is enabled but _fqn2spec_info not found on model")


@torch.no_grad()
def offload_veomni_optimizer(optimizer, empty_cache: bool = True):
    """
    Offload VeOmni optimizer to CPU with Expert Parallelism (EP) support.

    VeOmni supports both FSDP1 and FSDP2. When EP+FSDP2 is enabled, the optimizer
    is a MultiOptimizer containing separate optimizers for EP and non-EP parameters.
    This function handles both standard optimizers and MultiOptimizer instances.

    Args:
        optimizer: VeOmni optimizer (standard optimizer or MultiOptimizer)
        empty_cache: Whether to empty CUDA cache after offloading

    Note:
        - For EP+FSDP2: Optimizer is a MultiOptimizer with "ep" and "non_ep" sub-optimizers
        - For EP+FSDP1 or non-EP: Optimizer is a standard torch.optim.Optimizer
        - All optimizer states (momentum, variance, etc.) are offloaded to CPU
    """
    if optimizer is None or not hasattr(optimizer, "state"):
        return

    # Check if this is a MultiOptimizer (used for EP+FSDP2)
    is_multi_optimizer = getattr(optimizer, "_is_multi_optimizer", False)
    
    if is_multi_optimizer:
        # MultiOptimizer: offload each sub-optimizer
        optimizers_dict = getattr(optimizer, "optimizers_dict", {})
        logger.debug(f"Offloading MultiOptimizer with {len(optimizers_dict)} sub-optimizers")
        
        for opt_name, sub_optimizer in optimizers_dict.items():
            logger.debug(f"Offloading sub-optimizer '{opt_name}' to CPU")
            _offload_optimizer_states(sub_optimizer)
    else:
        # Standard optimizer: offload directly
        _offload_optimizer_states(optimizer)

    if empty_cache:
        get_torch_device().empty_cache()


def _offload_optimizer_states(optimizer):
    """
    Helper function to offload optimizer states to CPU.
    
    Args:
        optimizer: Standard torch.optim.Optimizer instance
    """
    if not optimizer.state:
        return
    
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_veomni_optimizer(optimizer, device_id=None):
    """
    Load VeOmni optimizer from CPU to GPU with Expert Parallelism (EP) support.

    VeOmni supports both FSDP1 and FSDP2. When EP+FSDP2 is enabled, the optimizer
    is a MultiOptimizer containing separate optimizers for EP and non-EP parameters.
    This function handles both standard optimizers and MultiOptimizer instances.

    Args:
        optimizer: VeOmni optimizer (standard optimizer or MultiOptimizer)
        device_id: Target device ID (e.g., 0 for cuda:0). If None, uses get_device_id()

    Note:
        - For EP+FSDP2: Optimizer is a MultiOptimizer with "ep" and "non_ep" sub-optimizers
        - For EP+FSDP1 or non-EP: Optimizer is a standard torch.optim.Optimizer
        - All optimizer states (momentum, variance, etc.) are loaded to GPU
    """
    from verl.utils.device import get_device_id, get_device_name

    if optimizer is None or not hasattr(optimizer, "state"):
        return

    if device_id is None:
        device_id = get_device_id()
    
    device = torch.device(f"{get_device_name()}:{device_id}")

    # Check if this is a MultiOptimizer (used for EP+FSDP2)
    is_multi_optimizer = getattr(optimizer, "_is_multi_optimizer", False)
    
    if is_multi_optimizer:
        # MultiOptimizer: load each sub-optimizer
        optimizers_dict = getattr(optimizer, "optimizers_dict", {})
        logger.debug(f"Loading MultiOptimizer with {len(optimizers_dict)} sub-optimizers to {device}")
        
        for opt_name, sub_optimizer in optimizers_dict.items():
            logger.debug(f"Loading sub-optimizer '{opt_name}' to {device}")
            _load_optimizer_states(sub_optimizer, device)
    else:
        # Standard optimizer: load directly
        _load_optimizer_states(optimizer, device)


def _load_optimizer_states(optimizer, device):
    """
    Helper function to load optimizer states from CPU to device.
    
    Args:
        optimizer: Standard torch.optim.Optimizer instance
        device: Target device (torch.device)
    """
    if not optimizer.state:
        return
    
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

