支持 hydrid data parallelism

实现：Qwen3-8B, TP4CP4, max_tokens=32k


### 依赖

```
git clone https://gitcode.com/GitHub_Trending/me/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cd -

git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout f6688c61bcfe45243ee5eb34c6f013b1e06eca81
cd -

git clone https://gitcode.com/GitHub_Trending/vl/vllm.git
cd vllm
git checkout v0.14.1
cd -

git clone https://gitcode.com/gh_mirrors/vl/vllm-ascend.git
cd vllm-ascend
git checkout v0.14.0rc1
git submodule update --init --recursive
cd -
```

---
# 背景与动机

上下文并行（Context Parallelism，CP）通过将批次中的序列沿序列长度的维度进行切分，并将其分配至不同设备并行计算，从而有效解决了超长序列的训练问题。在模型训练中，为处理不同长度序列并防止内存溢出（OOM），通常会将多个序列打包成一个达到最大序列长度 max_packing_token_size的序列，并配置足够的 context_parallel_size。

然而，该方法存在明显的效率缺陷：无论序列长短，所有序列均需经历相同的划分与通信过程。当短序列与长序列一同被打包并进行CP处理时，短序列本不需拆分的部分也被强制划分，引入不必要的通信开销，造成计算资源浪费与训练效率下降。在训练数据包含大量短序列时，该问题尤为突出。

> 如果一个序列很短（例如 512 个 token），但 CP 组有 8 个进程，每个进程仍然只持有很小的一块，并且 Ring Attention 强制这 8 个进程之间进行通信，造成不必要的开销。

HDP 的核心思想：根据每个序列的实际长度，动态地将它分配到 CP 进程的一个子集上。短序列可能只在一个进程内完成，完全不产生跨进程通信；只有真正需要多个进程的长序列才被切分。这可以大幅减少 Ring Attention 中的通信量。

# 主函数：`generate_hdp_group_from_batch`
## 输入
- `attn_mask`：形状为 `[micro_batch_size, seq_len]` 的注意力掩码（1表示有效，0表示padding）
- 三个阈值参数（控制小鼠部分如何处理）
## 执行步骤
1. 计算每个序列的有效长度
  `seq_len_effective = attn_mask.sum(-1)`
2. 计算每个序列理想占用的进程数
  ```
  total_tokens = sum(seq_len_effective)
  tokens_per_rank = total_tokens // cp_size
  ranks_per_seq = seq_len_effective / tokens_per_rank
  ```
  例如：总 token 数 2048， cp大小4->每个进程处理 512 token。一个1024 token的序列需要 1024/512=2个进程
3. 先分配整数部分
  从 rank 0 开始，一次为每个序列分配 `floor(ranks_per_seq)`进程，并更新其实rank
4. 处理小数部分
  对于每个序列的小数部分 `frac`：
  - 如果 `frac < min(per_rank_overload_threshold * int_ranks, max_fraction_threshold)`，则不再增加进程（吸收少量超载）。
  - 如果`frac > fractional_roundup_threshold`，则增加一个rank（向上取整）
  - 否则，将该序列放入 `pairs`列表，留待后续统一打包
5. 打包剩余的小数序列
  - 如果没有空闲进程了 `start_rank >= cp_size`，调用 `_pack_into_existing_groups`将小数序列合并到已有的组中。
  - 否则，调用 `_pack_into_new_ranks`为新进程分配小数序列（贪心装箱，分数大的优先）。
6. 分配多余的进程
  - 如果还有未使用的进程，将它们依次添加到当前最小的组中，以平衡负载。
7. 负载均衡检查
  - 使用 `check_load_balance` 检查每个进程的实际负载，如果最大负载超过阈值（默认1.3），则回退到标准 CP（设置 `batch_hdp_group = None`)
8. 特殊回退
  如果所有序列都分配到同一组（即所有进程都在一个组里），则HDP没有意义，回退到标准CP

## 输出

一个列表 batch_hdp_group，其中 batch_hdp_group[i] 是一个列表，包含分配给第 i 个序列的所有 CP rank 编号。
例如 [[0,1], [2], [3]] 表示：
- 序列 0 使用 rank 0 和 rank 1

- 序列 1 只使用 rank 2

- 序列 2 只使用 rank 3

# 预处理：`preprocess_packed_seqs_hdp`
当 `batch_hdp_goup` 存在时，该函数为当前 CP rank 准备输入。
1. 确定当前rank参与的序列
  调用 `find_group` 得到：
  - `indices`：当前 rank 需要处理的序列下标列表。
  - `local_group`：这些序列所属的 HDP 组（即当前 rank 与哪些其他 rank 共同处理这些序列
    同时得到 `hdp_size = len(local_group)` 和 `hdp_rank = local_group.index(cp_rank)`
2. 对齐与填充。
  为了兼容张量并行 （TP）和两段式切分模式，将每个序列的填充后长度对齐到 `tp_size * hdp_size * 2`的倍数
3. 构建 packed 序列
  计算 `cu_seqlens_padded`（每个序列填充后的累计长度）和最大长度 `max_seqlen_in_batch`。
4. 重新排列 token
  对于当前 rank 负责的每个序列：
  - 将该序列的有效 token 按 `hdp_size` 和两段切分的方式拆分。
  - 每个 rank 会拿到前半段的一个切分和后半段的一个切分（可能为空）。
    最终得到一个形状为 `[1, total_padded_tokens // hdp_size, hidden]` 的packed tensor `input_ids_rmpad`
5. 返回 `packed_seq_params` 和 （可选）重排后的输入

# 4. 后处理
将 HDP 分散计算的输出重新组装会原始的 batch 形状
1. 全收集输出（如果 CP 大小>1）
  通过 `all_gather` 收集所有 cp rank的输出及其序列长度。
2. 逐序列恢复
  对每个序列 `i`
  - 获取它的 HDP 组 `batch_hdp_group[i]`。
  - 如果 `hdp_size == 1`：直接从对应 rank 的输出中复制连续的一段。
  - 如果 `hdp_size > `：从各个rank的输出中分别提取前半段和后半块，然后按
    `[rank0的前半,rank1的前半,...,rank0的后半,...rank1的后半,...]`的顺序交错拼回原始的 token 顺序。
3. 写回`output-new[i, attn_mask[i]]`
  利用原始 attn_mask将恢复的token放到正确的位置（支持非连续的valid token）。

# 关键设计考量
两段切分（two‑chunk splitting）
每个 rank 贡献一个序列的前半段（所有 rank 的前半段连续排列）和后半段（所有 rank 的后半段倒序排列），这种模式与 Ring Attention 的通信模式匹配良好。

对齐到 tp_size * hdp_size * 2
确保张量并行（TP）和 HDP 的分块大小对齐，简化内存布局。

回退机制
当 HDP 无法带来收益（所有 rank 在一个组）或导致严重负载不均时，自动回退到标准 CP，保证稳定性。

小数部分阈值

per_rank_overload_threshold：允许一个进程额外承载少量 token（默认 10% 的负载）。

fractional_roundup_threshold：当小数部分超过此值（默认 0.9）时，宁愿增加一个进程以避免过载。

max_fraction_threshold：绝对不允许小数部分超过此比例（默认 0.5），防止某个进程被过度超载。

# 使用场景
该代码通常集成在 Megatron‑Core / verl 框架的训练循环中：

在每一个 micro‑batch 开始前，调用 generate_hdp_group_from_batch 决定当前 batch 的 HDP 分组。

在 Transformer 层的前向传播中，使用 preprocess_packed_seqs_hdp 将输入转换为 packed 格式，并传入 PackedSeqParams。

在输出后，调用 postprocess_packed_seqs_hdp 将输出恢复为原始 batch 形状，以便计算 loss。

通过全局变量 batch_hdp_group（set_batch_hdp_group / get_batch_hdp_group）在各个函数之间传递分组信息。

# 总结
HDP 是一种针对变长序列的上下文并行优化策略。它根据每个序列的长度动态分配 CP 进程子集，避免短序列触发不必要的跨进程通信。通过贪心分配、阈值控制和负载均衡检查，在保证效率的同时维持了较好的负载均衡。该实现与 Megatron 的 packed sequence 机制紧密集成，并保留了回退到标准 CP 的能力。
