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
Router Replay Utilities (R3 only - No VPP Support)

Utilities for handling router replay functionality in Megatron models.
R3 mode: Rollout records routing decisions → Training replays them.

NOTE: This version does NOT support Virtual Pipeline Parallel (VPP).
      If you need VPP support, please use the original version from git history.
"""

import warnings
from typing import Optional

import torch

try:
    from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    pass

from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from verl.models.mcore.util import (
    preprocess_packed_seqs,
    preprocess_thd_no_padding,
)
from verl.utils.device import get_device_name
from verl.utils.megatron.router_replay_patch import RouterReplay


def get_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """当前 GPU 流水线阶段，需要构建多少层 Transformer 解码器层。"""
    if pp_rank is None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    # 不均匀切分（自定义首/尾层数），中间阶段均分剩余层数
    if config.num_layers_in_first_pipeline_stage is not None or config.num_layers_in_last_pipeline_stage is not None:
        assert not (config.account_for_embedding_in_pipeline_split or config.account_for_loss_in_pipeline_split), (
            " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        )
        layers_to_distribute = config.num_layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        if pipeline_stages_left > 0:
            assert layers_to_distribute % pipeline_stages_left == 0, (
                "With uneven pipelineing the left over layers must be divisible by left over stages"
            ) # 中间阶段必须被整除
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        if is_first_pp_stage and config.num_layers_in_first_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if is_last_pp_stage and config.num_layers_in_last_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else: # 均匀切分，所有pp阶段层数完全一样，支持嵌入层/损失层独立
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1 # 独立嵌入层，记入总层数

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1 # 独立损失层，计入总层数

        assert num_layers % config.pipeline_model_parallel_size == 0, (
            "num_layers should be divisible by pipeline_model_parallel_size"
        ) # 总层数必须能被pp阶段数整除
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    # 如果开启了vpp（把一个物理pp阶段再拆分）
    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
        assert num_layers_per_pipeline_rank % vp_size == 0, (
            f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        ) # m每个物理pp阶段的层数，必须能被虚拟阶段数整除
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        num_layers_to_build = num_layers_per_pipeline_rank

    # 如果embedding和loss单独占一层，则扣除一层（他们不是transformer layer)
    if config.account_for_embedding_in_pipeline_split:
        if is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

    if config.account_for_loss_in_pipeline_split:
        if is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


def is_moe_layer(tf_config, layer_idx):
    moe_layer_freq = getattr(tf_config, "moe_layer_freq", None) # moe层数显的频率/位置

    if isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0 # 如果是整数，按照每n层一个moe判断
    elif isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1 # 如果是列表，按照指定哪些层是moe判断
    else:
        raise ValueError(f"Unsupported moe_layer_freq type: {type(moe_layer_freq)}")


def get_moe_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Count the number of MoE layers assigned to the current rank."""
    # 1. 先算出：当前阶段一共要建多少层（普通层+moe层总和）
    total_layers = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    # 2. 算出：这些层在【全局模型】中的起始偏移（从第几层开始）
    # layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    # 3. 生成当前阶段负责的【全局层编号】
    local_global_indices = range(layer_offset, layer_offset + total_layers)
    # 4. 统计当前阶段负责的【全局层编号】中，有多少层是moe层
    num_moe_layers = sum(1 for idx in local_global_indices if is_moe_layer(config, idx))

    return num_moe_layers


def get_current_rank_layer_info(tf_config, vp_rank=None):
    """返回当前进程(GPU)负责的模型层信息——从第基层开始，到第几层结束，一共几层"""
    if vp_rank is None:
        vp_rank = 0 # 如果没有vpp stage，则vp_rank=0
    num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_rank) # 当前rank构建多少层
    offset = get_transformer_layer_offset(tf_config, vp_stage=vp_rank) # 当前层在全局模型的偏移

    local = {}
    local["start"] = offset # 起始层
    local["end"] = offset + num_layers_to_build # 结束层（不包含）
    local["count"] = num_layers_to_build # 总层数
    return local


def set_router_replay_data(layers_topk_idx, attention_mask, tf_config, vp_rank=None):
    """
    把路由决策数据分发给序列并行rank，并设置到本地RouterReplay实例中。
    作用：让训练阶段严格复用rollout阶段的路由结果。
    layers_topk_idx: 形状 [bs, max_seq_len, 总层数, topk]，存好的路由结果
    在forward_backward_batch的forward_step方法中调用，把路由结果设置到每个MoERouter的RouterReplay实例中，router.set_target_indices
    """
    with torch.no_grad():
        # cp
        if layers_topk_idx.is_nested:
            layers_topk_idx_rmpad, _, _ = preprocess_thd_no_padding(layers_topk_idx, pre_process=True)
        else: # 普通张量：根据 attention_mask 去掉序列 padding——适配CP
            layers_topk_idx_rmpad, _ = preprocess_packed_seqs(layers_topk_idx, attention_mask, pre_process=True)
        layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous()

        # 序列并行（SP/TP）切分：把【序列维度】切分给不同的TP/GPU，每个GPU只保留自己负责的那一段序列
        # CP 和 SP 是叠加、同时生效的，是「先 CP → 再 SP」的双层序列切分
        layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
            layers_topk_idx_rmpad.to(get_device_name()).squeeze(dim=0)
        ).unsqueeze(dim=0)

        # 调整维度 + 流水线并行（PP）切分
        # 调整维度：把层维度放到最前面，方便按层取数据
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(dim=0)
        # 获取当前GPU/VP阶段负责哪几层
        local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
        offset, end = local_rank_info["start"], local_rank_info["end"]
        # 获取当前GPU对应的所有MoE路由器实例
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        # 判断数据格式：是否按【全局层】索引
        index_by_layer = len(layers_topk_idx_reshape) == tf_config.num_layers
        # 计算：在全局所有MoE层中，当前GPU负责的MoE层起始索引
        moe_idx = sum(1 for i in range(offset) if is_moe_layer(tf_config, i))

        router_offset = 0
        for layer_idx in range(offset, end): # 遍历本地层 → 给每个MoE RouterReplay设置routed_indices
            if not is_moe_layer(tf_config, layer_idx):
                continue
            router = router_instances_list[router_offset] # 拿到当前层的router replay
            idx = layer_idx if index_by_layer else moe_idx
            router.set_target_indices(layers_topk_idx_reshape[idx].to(torch.int64)) # 把routed_indices设置给router replay
            router_offset += 1
            moe_idx += 1


class RouterReplayHelper:
    """帮当前GPU找到自己负责的那几个MoE路由器实例，并判断现在是否要重放前向
    
    Simplified version without VPP support.
    """
    
    @staticmethod
    def get_micro_batch_router_list(tf_config):
        """
        Return the list of RouterReplay instances corresponding to the current PP rank.
        
        Simplified version without VPP support - directly returns all router instances
        for the current PP rank without VPP offset calculation.

        Args:
            tf_config: Configuration object used to compute layer assignments.
            
        Returns:
            list: A list of RouterReplay.router_instances for the current PP rank.
        """
        num_layers_to_build = get_moe_num_layers_to_build(tf_config)
        router_instances_list = RouterReplay.router_instances[:num_layers_to_build]
        return router_instances_list

    @staticmethod
    def if_router_replay(tf_config) -> bool:
        """Return True if target_topk_idx is set for the local router instances."""
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config)
        return (
            router_instances_list and router_instances_list[0].target_topk_idx is not None
        )
