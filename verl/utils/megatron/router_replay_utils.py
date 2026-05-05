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

from typing import Optional

import torch

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


def get_num_layers_to_build(config: TransformerConfig, pp_rank: Optional[int] = None) -> int:
    """当前 GPU 流水线阶段，需要构建多少层 Transformer 解码器层。
    
    Simplified version without VPP support.
    """
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
            )
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        if is_first_pp_stage and config.num_layers_in_first_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if is_last_pp_stage and config.num_layers_in_last_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert num_layers % config.pipeline_model_parallel_size == 0, (
            "num_layers should be divisible by pipeline_model_parallel_size"
        )
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    num_layers_to_build = num_layers_per_pipeline_rank

    # 如果embedding和loss单独占一层，则扣除一层（他们不是transformer layer)
    if config.account_for_embedding_in_pipeline_split:
        if is_first_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the first pipeline stage"

    if config.account_for_loss_in_pipeline_split:
        if is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the last pipeline stage"

    return num_layers_to_build


def is_moe_layer(tf_config, layer_idx):
    moe_layer_freq = getattr(tf_config, "moe_layer_freq", None) # moe层数显的频率/位置

    if isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0 # 如果是整数，按照每n层一个moe判断
    elif isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1 # 如果是列表，按照指定哪些层是moe判断
    else:
        raise ValueError(f"Unsupported moe_layer_freq type: {type(moe_layer_freq)}")


def get_moe_num_layers_to_build(config: TransformerConfig, pp_rank: Optional[int] = None) -> int:
    """Count the number of MoE layers assigned to the current rank.
    
    Simplified version without VPP support.
    """
    total_layers = get_num_layers_to_build(config, pp_rank=pp_rank)
    layer_offset = get_transformer_layer_offset(config)
    local_global_indices = range(layer_offset, layer_offset + total_layers)
    num_moe_layers = sum(1 for idx in local_global_indices if is_moe_layer(config, idx))

    return num_moe_layers


def get_current_rank_layer_info(tf_config):
    """返回当前进程(GPU)负责的模型层信息——从第基层开始，到第几层结束，一共几层
    
    Simplified version without VPP support.
    """
    num_layers_to_build = get_num_layers_to_build(tf_config)
    offset = get_transformer_layer_offset(tf_config)

    local = {}
    local["start"] = offset
    local["end"] = offset + num_layers_to_build
    local["count"] = num_layers_to_build
    return local


def set_router_replay_data(layers_topk_idx, attention_mask, tf_config):
    """
    把路由决策数据分发给序列并行rank，并设置到本地RouterReplay实例中。
    作用：让训练阶段严格复用rollout阶段的路由结果。
    layers_topk_idx: 形状 [bs, max_seq_len, 总层数, topk]，存好的路由结果
    在forward_backward_batch的forward_step方法中调用，把路由结果设置到每个MoERouter的RouterReplay实例中，router.set_target_indices
    
    Simplified version without VPP support.
    """
    with torch.no_grad():
        # cp
        if layers_topk_idx.is_nested:
            layers_topk_idx_rmpad, _, _ = preprocess_thd_no_padding(layers_topk_idx, pre_process=True)
        else:
            layers_topk_idx_rmpad, _ = preprocess_packed_seqs(layers_topk_idx, attention_mask, pre_process=True)
        layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous() # [1, 440, 4, 8]
        # print(f">>>{os.getpid()}debug lcx ")

        # 序列并行（SP/TP）切分，在sequence维度上进一步切分，因为经过cp预处理（packing，起始bsz维始终为1了）
        layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
            layers_topk_idx_rmpad.to(get_device_name()).squeeze(dim=0)
        ).unsqueeze(dim=0) # [1, 220, 4, 8]

        # 调整维度 + 流水线并行（PP）切分
        # bsz seq_len num-layers topk -> bsz layers seq-len topk->layer, seq-len topk【去掉bsz维，并把layer维放到第一维】
        # [4, 220, 8]
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(dim=0)
        # 获取当前GPU负责哪几层
        local_rank_info = get_current_rank_layer_info(tf_config)
        offset, end = local_rank_info["start"], local_rank_info["end"]
        # 获取当前GPU对应的所有MoE路由器实例
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config)
        # 判断数据格式：是否按【全局层】索引
        index_by_layer = len(layers_topk_idx_reshape) == tf_config.num_layers
        # 计算：在全局所有MoE层中，当前GPU负责的MoE层起始索引
        moe_idx = sum(1 for i in range(offset) if is_moe_layer(tf_config, i))

        router_offset = 0
        for layer_idx in range(offset, end):
            if not is_moe_layer(tf_config, layer_idx):
                continue
            router = router_instances_list[router_offset]
            idx = layer_idx if index_by_layer else moe_idx
            router.set_target_indices(layers_topk_idx_reshape[idx].to(torch.int64))
            router_offset += 1
            moe_idx += 1
