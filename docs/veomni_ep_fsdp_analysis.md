# VeOmni EP+FSDP2 实现分析 - 以 Qwen3-VL-MoE 为例

## 概述

本文档深入分析 VeOmni 框架中 Expert Parallelism (EP) 和 FSDP2 的实现机制，以 Qwen3-VL-MoE 模型为例。

## 1. 架构概览

### 1.1 并行策略组合

VeOmni 支持多种并行策略的组合：
- **Expert Parallelism (EP)**: 在专家维度（dim-0）上切分 MoE 层
- **FSDP2**: 在隐藏维度（dim-1）上切分专家参数，在标准维度（dim-0）上切分非专家参数
- **Tensor Parallelism (TP)**: 可选
- **Pipeline Parallelism (PP)**: 可选
- **Sequence Parallelism (SP)**: 可选

### 1.2 Qwen3-VL-MoE 模型特点

- **专家数量**: 128 个专家
- **隐藏维度**: 2048
- **中间维度**: 768 (MoE)
- **每 token 激活专家数**: 8
- **特殊结构**: `gate_up_proj` 将 gate 和 up projection 合并为单个张量

## 2. EP 实现详解

### 2.1 Parallel Plan 定义

**文件**: `VeOmni/veomni/models/transformers/qwen3_vl_moe/parallel_plan.py`

```python
def get_parallel_plan():
    ep_plan = {
        "model.language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.language_model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    parallel_plan = ParallelPlan(ep_plan=ep_plan)
    return parallel_plan
```

**关键点**:
- 使用通配符模式 `*` 匹配所有层
- `Shard(0)` 表示在专家维度（第0维）上切分
- 只对专家参数应用 EP，非专家参数保持 Replicate

### 2.2 EP 应用流程

**文件**: `VeOmni/veomni/distributed/parallel_plan.py`

#### Step 1: 识别专家参数

```python
def apply(self, model: nn.Module, ep_fsdp_mesh: DeviceMesh):
    ep_mesh = ep_fsdp_mesh["ep"]
    fqn2spec_info = {}
    
    for fqn, param in model.named_parameters():
        for fqn_pattern, shard in self.ep_plan.items():
            if check_fqn_match(fqn_pattern, fqn):
                # 匹配到专家参数，进行 EP 切分
                assert param.size(shard.dim) % ep_size == 0
                ep_placement = ep_replicate[:-1] + [shard]
                # ... 创建 DTensor 并切分
```

#### Step 2: 张量切分

对于 Qwen3-VL-MoE，假设 EP size = 8：

- **原始形状**: 
  - `gate_up_proj`: [128, 768, 2048]
  - `down_proj`: [128, 2048, 768]

- **EP 切分后 (每个 rank)**:
  - `gate_up_proj`: [16, 768, 2048] (128/8 = 16)
  - `down_proj`: [16, 2048, 768]

#### Step 3: 转换为本地张量

```python
dtensor = dtensor.redistribute(device_mesh=ep_mesh, placements=ep_placement)
local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
```

**重要**: EP 切分后立即转换为本地张量，**丢弃 DTensor 信息**。这是因为：
1. 专家计算需要手动控制 EP ranks 之间的通信
2. 避免在 FSDP 过程中意外 gather 专家参数
3. 使用 fused MoE 实现来管理 EP 通信

### 2.3 EP 通信机制

在 MoE forward 过程中，EP 通过 All-to-All 通信分发 token：

**文件**: `VeOmni/veomni/ops/fused_moe/group_gemm.py`

```python
def group_gemm_fused_moe_forward(...):
    if get_parallel_state().ep_enabled:
        # Step 1: 预处理，计算每个本地专家需要处理的 token 数量
        input_splits, output_splits, ... = preprocess(
            expert_mask=expert_mask,
            num_experts=num_experts,
            ep_group=get_parallel_state().ep_group,
        )
        
        # Step 2: All-to-All 分发 token 到对应的 EP rank
        permute_tokens, routing_map, ... = token_pre_all2all(
            hidden_states=hidden_states,
            expert_mask=expert_mask,
            input_splits=input_splits,
            output_splits=output_splits,
            ep_group=get_parallel_state().ep_group,
        )
        
        # Step 3: 本地专家计算
        final_permute_tokens = EPGroupGemm.apply(...)
        
        # Step 4: All-to-All 收集结果
        final_hidden_states = token_post_all2all(...)
```

**通信模式**:
```
Rank 0: tokens for experts [0-15]  -> All-to-All -> 计算结果 -> All-to-All -> 返回
Rank 1: tokens for experts [16-31] -> All-to-All -> 计算结果 -> All-to-All -> 返回
...
Rank 7: tokens for experts [112-127] -> All-to-All -> 计算结果 -> All-to-All -> 返回
```

## 3. FSDP2 实现详解

### 3.1 设备网格构建

**文件**: `VeOmni/veomni/distributed/torch_parallelize.py`

对于 EP+FSDP2 组合，设备网格是二维的：

```python
# 假设: world_size=16, ep_size=8, fsdp_size=16
ep_fsdp_device_mesh = DeviceMesh(
    device_type="cuda",
    mesh=[[0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15]],  # 2 EP groups
    mesh_dim_names=["ep", "ep_fsdp"]
)

# 对于专家模块: 使用 ep_fsdp_mesh = [0,8], [1,9], ... (跨 EP groups)
# 对于非专家模块: 使用 fsdp_mesh = [0,1,2,...,15] (全局 FSDP)
```

### 3.2 分层 Sharding 策略

#### 专家模块的 FSDP2 Sharding

```python
# 专家模块: 在 dim-1 (hidden dim) 上 shard
def _experts_shard_placement_fn(param):
    return Shard(1)  # 沿隐藏维度切分

expert_fsdp_kwargs = {
    "mesh": ep_fsdp_mesh,  # 跨 EP groups 的 FSDP mesh
    "shard_placement_fn": _experts_shard_placement_fn,
}
fully_shard(experts_mod, **expert_fsdp_kwargs)
```

**示例** (EP=8, FSDP2=2):
- EP 切分后: `gate_up_proj` [16, 768, 2048]
- FSDP2 切分后: `gate_up_proj` [16, 768, 1024] (2048/2 = 1024)

#### 非专家模块的 FSDP2 Sharding

```python
# 非专家模块: 标准 dim-0 sharding
fsdp_kwargs = {"mesh": parallel_state.fsdp_mesh}
fully_shard(layer_mod, **fsdp_kwargs)
```

### 3.3 自底向上的 Sharding 顺序

```python
for layer_fqn, layer_mod, experts_mod in layer_pairs:
    if parallel_state.ep_enabled and experts_mod is not None:
        # 1. 先 shard 专家模块 (使用 dim-1 sharding)
        fully_shard(experts_mod, **expert_fsdp_kwargs)
        
        # 2. 设置梯度聚合因子 (EP ranks 之间平均梯度)
        experts_mod.set_gradient_divide_factor(parallel_state.ep_gradient_divide_factor)
    
    # 3. 再 shard 整个 decoder layer (使用标准 dim-0 sharding)
    fully_shard(layer_mod, **fsdp_kwargs)

# 4. 最后 shard root model
fully_shard(model, **fsdp_kwargs)
```

### 3.4 手动 Prefetching 配置

由于嵌套 sharding，需要手动配置 prefetching：

```python
# Forward prefetch: 当前层 prefetch 下一层的 attention 和 experts
for current_block, next_block in zip(blocks, next_blocks):
    if next_block is not None:
        prefetch_modules = next_block._fsdp_modules
        # 按顺序: experts, gate, attention
        current_block.set_modules_to_forward_prefetch(list(reversed(prefetch_modules)))

# Backward prefetch: 当前层 prefetch 上一层的 attention 和 experts
for current_block, prev_block in zip(rev_blocks, prev_blocks):
    if prev_block is not None:
        prefetch_modules = prev_block._fsdp_modules
        current_block.set_modules_to_backward_prefetch(list(reversed(prefetch_modules)))
```

**目的**: 让 attention 计算和 experts 的 AllGather 重叠，提高效率。

## 4. 完整流程示例

### 4.1 初始化流程

**文件**: `verl/verl/workers/engine/veomni/transformer_impl.py`

```python
def _build_model_optimizer(self):
    # Step 1: 加载基础模型
    module = build_foundation_model(
        config_path=self.model_config.hf_config_path,
        weights_path=self.model_config.path,
        ...
    )
    
    # Step 2: 应用并行策略
    module = build_parallelize_model(
        module,
        enable_full_shard=self.engine_config.enable_full_shard,
        enable_mixed_precision=self.engine_config.mixed_precision,
        basic_modules=module._no_split_modules + self.engine_config.basic_modules,
        ...
    )
    
    # Step 3: 构建优化器
    optimizer = self._build_optimizer(module)
    
    self.module = module
    self.optimizer = optimizer
```

### 4.2 并行化流程

**文件**: `VeOmni/veomni/distributed/torch_parallelize.py::parallelize_model_fsdp2`

```python
def parallelize_model_fsdp2(...):
    # Step 1: 应用 EP (切分专家张量)
    if parallel_state.ep_enabled:
        parallel_plan = model.get_parallel_plan()
        ep_fqn2spec_info = parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)
        experts_map = parallel_plan.get_fsdp_no_shard_info(model)
        # 结果: 专家参数从 [128, H, I] -> [16, H, I] (假设 EP=8)
    
    # Step 2: 提取 decoder layers 和对应的 experts modules
    layer_pairs = []
    for layer_fqn, layer_mod in decoder_blocks:
        experts_mod = extract_experts_from_layer(layer_mod, experts_map)
        layer_pairs.append((layer_fqn, layer_mod, experts_mod))
    
    # Step 3: 对每个 layer 应用 FSDP2
    for layer_fqn, layer_mod, experts_mod in layer_pairs:
        if experts_mod is not None:
            # 专家模块: dim-1 sharding, 使用 ep_fsdp_mesh
            fully_shard(experts_mod, **expert_fsdp_kwargs)
        # 整个 layer: dim-0 sharding, 使用 fsdp_mesh
        fully_shard(layer_mod, **fsdp_kwargs)
    
    # Step 4: Shard root model
    fully_shard(model, **fsdp_kwargs)
    
    # Step 5: 配置 prefetching
    configure_manual_prefetching(layer_pairs)
```

### 4.3 Forward 流程

```python
# 在 Qwen3VLMoeTextSparseMoeBlock.forward() 中
def forward(self, hidden_states, ...):
    # Step 1: Router 计算，选择 top-k 专家
    routing_weights, selected_experts = self.router(hidden_states)
    
    # Step 2: Fused MoE Forward (包含 EP 通信)
    if self.moe_implementation == "fused":
        final_hidden_states = group_gemm_fused_moe_forward(
            module=self,
            num_experts=self.num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=self.gate_up_proj[..., :self.intermediate_size, :],  # gate
            fc1_2_weight=self.gate_up_proj[..., self.intermediate_size:, :],    # up
            fc2_weight=self.down_proj,
        )
        # 内部流程:
        #   - All-to-All: 分发 token 到对应的 EP rank
        #   - 本地专家计算 (使用 EPGroupGemm)
        #   - All-to-All: 收集结果
```

## 5. 关键设计决策

### 5.1 为什么 EP 在 dim-0，FSDP2 在 dim-1？

- **EP dim-0**: 专家数量维度，自然切分点
- **FSDP2 dim-1**: 避免与 EP 冲突，允许更灵活的并行度组合
  - 如果都在 dim-0: `EP_size × FSDP2_size = num_experts` (限制性强)
  - 现在: `EP_size × FSDP2_size = world_size` (更灵活)

### 5.2 为什么 EP 参数转换为本地张量？

- 避免 FSDP 的自动 AllGather 干扰 EP 通信
- 使用 fused MoE 手动控制 All-to-All 通信
- 简化实现，避免复杂的 DTensor 嵌套

### 5.3 为什么需要手动 Prefetching？

- 嵌套 sharding 导致默认 prefetching 不高效
- 需要明确指定 attention 和 experts 的 prefetch 顺序
- 最大化计算和通信重叠

## 6. 配置示例

### 6.1 配置文件

```yaml
# VeOmni/configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml
train:
  data_parallel_mode: fsdp2
  expert_parallel_size: 8
  data_parallel_size: 16  # 自动计算: world_size / expert_parallel_size
  enable_full_shard: true
  mixed_precision: true
  moe_implementation: fused  # 必须使用 fused 以支持 EP
```

### 6.2 并行度计算

假设 16 GPUs:
- `expert_parallel_size = 8`: 8 个 EP ranks
- `data_parallel_size = 16`: 16 个 FSDP ranks (自动)
- 每个 EP group 有 2 个 FSDP ranks

**专家参数分布**:
- EP rank 0-7: 各自持有 16 个专家 (128/8)
- FSDP rank 0,8: 在 EP rank 0 的专家上，各自持有 1/2 的 hidden dim

## 7. 性能优化要点

1. **Fused MoE**: 使用 `moe_implementation="fused"` 启用优化的 MoE kernel
2. **Gradient Divide Factor**: EP ranks 之间正确聚合梯度
3. **Prefetching**: 手动配置以重叠计算和通信
4. **Mixed Precision**: 支持 bfloat16 训练
5. **Activation Offloading**: 可选，减少显存占用

## 8. 总结

VeOmni 的 EP+FSDP2 实现通过以下方式实现了高效的 MoE 模型训练：

1. **EP**: 在专家维度切分，使用 All-to-All 通信分发 token
2. **FSDP2**: 在隐藏维度切分专家参数，在标准维度切分非专家参数
3. **嵌套 Sharding**: 自底向上应用，确保正确的参数分布
4. **手动控制**: Prefetching 和通信都由框架精确控制

这种设计既保持了灵活性，又实现了高效的并行训练。

