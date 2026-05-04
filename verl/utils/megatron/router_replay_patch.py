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
"""
Router Replay Patch (R3 only - No VPP Support)

This module provides router replay functionality for MoE models in RL training.
R3 mode: Rollout records routing decisions → Training replays them.

NOTE: This version does NOT support Virtual Pipeline Parallel (VPP).
      If you need VPP support, please use the original version from git history.
"""

import torch
from megatron.core.transformer.moe.moe_utils import (
    apply_router_token_dropping,
    compute_routing_scores_for_aux_loss,
    group_limited_topk,
)
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


class RouterReplay:
    """
    A class to manage the replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control replaying.

    R3 Mode Usage (No VPP Support):
    1. Rollout phase: Records routing decisions (handled by inference engine)
    2. Training phase: set_target_indices() with recorded decisions
    3. Forward pass: Replays routing decisions if target_topk_idx is set
    """

    router_instances = []

    @staticmethod
    def set_replay_data(all_layers_topk_indices: list):
        """将所有moe层的topk索引分发给各自的RouterReplay实例
        all_layers_topk_indices: 张量列表，每个张量包含特定层的topk索引。顺序必须与TopKRouter实例化顺序匹配。
        """
        if len(all_layers_topk_indices) != len(RouterReplay.router_instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match the number of router instances ({len(RouterReplay.router_instances)})."
            )
        for i, router_instance in enumerate(RouterReplay.router_instances):
            router_instance.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def clear_global_indices():
        """将所有路由实例的target topk idx清空"""
        for router in RouterReplay.router_instances:
            router.clear_indices()

    def __init__(self):
        """初始化给定层的RouterReplay实例."""
        self.target_topk_idx = None
        RouterReplay.router_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor):
        """为当前层设置目标top-k索引."""
        self.target_topk_idx = topk_indices

    def clear_indices(self):
        """将当前层的target topk idx清空"""
        self.target_topk_idx = None


def _patched_topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    score_function: str,
    expert_bias: torch.Tensor,
    fused: bool,
    router_replay: RouterReplay,
    scaling_factor: float,
):
    """
    Patched version of topk_routing_with_score_function that supports router replay.
    """
    num_tokens, num_experts = logits.shape

    def _compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if router_replay is None or router_replay.target_topk_idx is None:
            return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

        top_indices = router_replay.target_topk_idx
        top_indices = top_indices.to(scores.device)
        probs = scores.gather(1, top_indices)
        return probs, top_indices

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    if torch.are_deterministic_algorithms_enabled():
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_((rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False)
        routing_map = routing_map.bool()
    else:
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


def patched_routing(self, logits: torch.Tensor, *args, **kwargs):
    """Top-k routing function

    Args:
        logits (torch.Tensor): Logits tensor after gating.

    Returns:
        probs (torch.Tensor): The probabilities of token to experts assignment.
        routing_map (torch.Tensor): The mapping of token to experts assignment,
            with shape [num_tokens, num_experts].
    """
    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)

    logits = self.apply_z_loss(logits)

    if self.routing_type == "sinkhorn":
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    else:
        probs, routing_map = _patched_topk_routing_with_score_function(
            logits=logits,
            topk=self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=self.config.moe_router_fusion,
            router_replay=getattr(self, "router_replay", None),
        )

    if self.config.moe_expert_capacity_factor is not None:
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
        routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
            logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
        )
        probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        probs = self._apply_seq_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz)
        probs = self._apply_global_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)

    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    return probs, routing_map


def apply_router_replay_patch():
    """
    Applies the monkey patch for MoE Router Replay functionality.
    This patch dynamically adds the 'enable_routing_replay' attribute to TransformerConfig
    and modifies the TopKRouter to support replaying of routing decisions.
    
    NOTE: This version does NOT support Virtual Pipeline Parallel (VPP).
    """
    print("Applying Router Replay Patch (No VPP Support)...")
    
    # Check if VPP is enabled
    if hasattr(TransformerConfig, "virtual_pipeline_model_parallel_size"):
        # This check will be performed at runtime when config is available
        pass
    
    RouterReplay.router_instances.clear()
    # Step 1: Patch TransformerConfig to include the feature flag
    if not hasattr(TransformerConfig, "enable_routing_replay"):
        # Add class attribute with default value
        TransformerConfig.enable_routing_replay = False

        original_tf_config_init = TransformerConfig.__init__

        # Define new __init__ method that safely handles enable_routing_replay parameter
        def patched_tf_config_init(self, *args, **kwargs):
            # Simple solution: remove the unknown parameter before calling original constructor
            enable_routing_replay = kwargs.pop("enable_routing_replay", TransformerConfig.enable_routing_replay)

            # Call original constructor with remaining kwargs
            original_tf_config_init(self, *args, **kwargs)

            # Set the instance attribute
            self.enable_routing_replay = enable_routing_replay
            
            # Check for VPP and raise error if enabled
            if hasattr(self, "virtual_pipeline_model_parallel_size") and self.virtual_pipeline_model_parallel_size is not None:
                raise ValueError(
                    "Virtual Pipeline Parallel (VPP) is enabled, but this version of router replay "
                    "does not support VPP. Please use the original version from git history or disable VPP."
                )

        # Apply the patch
        TransformerConfig.__init__ = patched_tf_config_init

     # Step 2: Patch TopKRouter only once to ensure idempotency.
    if hasattr(TopKRouter, "_router_replay_patched"):
        return

    original_init = TopKRouter.__init__

    # Step 3: Define the new __init__ method
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.router_replay = None
        if self.config.enable_routing_replay:
            self.router_replay = RouterReplay()

    # Step 4: Patch MoEAlltoAllTokenDispatcher.preprocess to handle router replay
    # When router replay is enabled, duplicate indices in top_indices can cause
    # routing_map.sum() < num_tokens * topk, leading to split size mismatch in alltoall.
    if MoEAlltoAllTokenDispatcher is not None and not hasattr(MoEAlltoAllTokenDispatcher, "_preprocess_patched"):
        original_preprocess = MoEAlltoAllTokenDispatcher.preprocess

        def patched_preprocess(self, routing_map):
            result = original_preprocess(self, routing_map)

            # Fix num_out_tokens when router replay is enabled
            if (
                getattr(self.config, "enable_routing_replay", False)
                and not self.drop_and_pad
                and self.config.moe_expert_capacity_factor is None
                and not (
                    getattr(self.config, "moe_router_padding_for_quantization", None)
                    or getattr(self.config, "moe_router_padding_for_fp8", None)
                )
            ):
                # With router replay, duplicate indices can reduce the actual routed
                # token count, so derive it from the routing map instead.
                self.num_out_tokens = int(routing_map.sum().item())

            return result

        MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
        MoEAlltoAllTokenDispatcher._preprocess_patched = True

    # Step 5: Apply the patches
    TopKRouter.__init__ = patched_init
    TopKRouter.routing = patched_routing
    TopKRouter._router_replay_patched = True
