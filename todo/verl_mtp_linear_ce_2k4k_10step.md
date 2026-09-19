# H20：MTP 主头 Linear CE，2K/4K、10 步对照

日期：2026-09-20（Asia/Shanghai）。过程资料，最终 ready 后可移除。

结论：两组均完成 10 步。本配置没有观察到 Linear CE 加速；去掉首步后，
fused actor update 耗时高 2.06%，回放 TPS 低 1.86%，全程显存峰值不变。
该小幅差异不足以判定稳定退化，但不支持把此开关作为当前配置的必开优化。
全部 advantage 为 0，精度验收仍不充分。
独立单步 profile 确认 rank 0 训练区间约省 0.52 GiB、old-logprob 区间约省
0.96 GiB；这些局部收益被初始化峰值遮住，并不是“完全没有显存节省”。

## 实验设置

- 分支 `hz/feat/mtp-fused-linear-ce`，代码基于 `e6b6b1d1`。
- 8 × H20-3e；Qwen3.5-35B-A3B；DAPO-Math-17k；GRPO。
- `max_prompt_length=2048`、`max_response_length=4096`、`max_model_len=6144`。
- actor/log-prob token budget 与 vLLM batched-token budget 均为 6144。
- train batch 8、rollout n=2、micro-batch/GPU=1、TP2/EP8、rollout TP8。
- MTP 1 层、scale 0.1、detach encoder、full recompute、CPU optimizer offload。
- 运行时沿用[已有 H20 验证记录](verl_mtp_fused_linear_ce_validation.md)。
- 脚本：`todo/run_h20_qwen35_mtp_linear_ce_grpo.sh`，默认 10 步；
  `TOTAL_TRAINING_STEPS=1` 可用于独立显存 profiling。
- 新 cache：`/workspace/mtp-linear-ce-2k4k-cache-20260920`，不复用短序列结果。
- experiment：`qwen35_mtp_linear_ce_h20_8gpu_2k4k_20260920`。
- 已逐一核对 H20 与本地 `model_forward.py`、`model_forward_fused.py`、`mtp_patch.py`、
  `transformer_impl.py`、`megatron_utils.py` 和测试 shell 的 SHA256，全部一致。

## 比较口径

1. 先生成并缓存 10 步 rollout，再从同一初始 checkpoint 分别运行 unfused/fused。
2. 每组使用相同步骤的 cached tokens/rewards；不是只固定随机种子后重新生成。
3. 吞吐轮次不启用内存 profiler；显存快照另行运行。
4. 同时报告全部 10 步与去掉首步的 2–10 步，计算 `sum(tokens)/sum(seconds)`；
   不对单步 TPS 做算术平均。每卡 TPS 再除以 8。
5. old-logprob、actor update、weight sync 和整步分开报告。
   cache replay 整步 TPS **不代表包含真实生成的在线端到端 RL TPS**。
6. allocator lifetime watermark 包含初始化，不能自动视为训练区间峰值。
7. 多步 scalar 指标接近不等于参数梯度/更新逐元素等价，也不代表收敛验收。

汇总工具：

```bash
python todo/summarize_mtp_grpo_benchmark.py \
  /path/to/unfused.log /path/to/fused.log --steps 10 --warmup 1 --gpus 8
python -m unittest discover -s todo -p 'test_summarize_mtp_grpo_benchmark.py' -v
```

汇总器拒绝缺步、重复步、关键指标非有限值、A/B token 总数不一致。
相同 token 总数本身不能证明数据相同，仍需核对逐步 cache 命中记录。

### 可复现命令

在 H20 容器 `verl-cxli-qwen38` 内、源码目录
`/workspace/verl-mtp-fused-linear-ce-current` 下执行：

```bash
export PATH=/workspace/qwen38-bridge-te218-validation-venv/bin:$PATH
export PYTHONPATH=/workspace/verl-mtp-fused-linear-ce-current:/workspace/Megatron-Bridge-qwen3.8-flash-next/src:/workspace/Megatron-Bridge-qwen3.8-flash-next/3rdparty/Megatron-LM
export CUDNN_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64
export MODEL_PATH=/workspace/models/Qwen3.5-35B-A3B
export DATA_FILE=/workspace/data/DAPO-Math-17k/data/dapo-math-17k.parquet
export SKIP_DUMP_DIR=/workspace/mtp-linear-ce-2k4k-cache-20260920
export EXPERIMENT_NAME=qwen35_mtp_linear_ce_h20_8gpu_2k4k_20260920
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=4096 MAX_TOKEN_LEN_PER_GPU=6144
export TOTAL_TRAINING_STEPS=10

# Cache must contain all 10 steps before timing a replay. Use fresh output
# directories and distinct RAY_TMPDIR for each invocation; run sequentially.
FUSED_KERNELS=False OUTPUT_DIR=/workspace/mtp-linear-ce-2k4k-unfused10-v1 \
  RAY_TMPDIR=/tmp/mtp2k4k-u10v1 \
  bash todo/run_h20_qwen35_mtp_linear_ce_grpo.sh
FUSED_KERNELS=True OUTPUT_DIR=/workspace/mtp-linear-ce-2k4k-fused10-v1 \
  RAY_TMPDIR=/tmp/mtp2k4k-f10v1 \
  bash todo/run_h20_qwen35_mtp_linear_ce_grpo.sh
```

## 环境排障记录

- 启动前终止了 `codex-qwen3-full` 中占用 GPU 4–7 的 SFT launcher（容器 PID 9380），
  八卡归零后开始测试。
- 相同 SFT 又以容器 PID 10062 启动；已再次终止。第一次长序列 profile
  `/workspace/mtp-linear-ce-2k4k-unfused-profile-v1.log` 退出码 1：
  vLLM 启动时 GPU 4–7 只剩约 35 GiB，低于 utilization=0.30 所需约 42 GiB。
- 第三次 SFT 改在 `codex-qwen3-gpu-train-20260920` 容器启动，宿主 launcher PID 3975931，
  又占用 GPU 4–7。`/workspace/mtp-linear-ce-2k4k-cache10-v1.log` 退出码 1，
  vLLM 报 `No available memory for the cache blocks`。该尝试未产生有效训练步。
- 上述失败尝试不用于吞吐、精度或 Linear CE 显存收益结论。
- 宿主直接 TERM 第三组 launcher 因权限不足未生效；随后对已核实 entrypoint
  仅为该 SFT 的独立容器执行 `docker stop --time 30`。Docker 返回未收到 exit event，
  但随后新查询确认容器已为 `Exited (137)`，八卡均为 0 MiB；没有删除容器或数据。
  之后重启为 `/workspace/mtp-linear-ce-2k4k-cache10-v2.log`。

## 结果

### 真实生成基线（缓存准备）

`mtp-linear-ce-2k4k-cache10-v2` 完成 10/10 步，退出码 0。共 160 条轨迹、
679,812 个 prompt+response tokens。全部 response 为 4096 tokens；实际 prompt
每步均值 134.875–175.125，单条最大 303，并非每条 prompt 都达到 2048。

| 指标 | 全 10 步 | 去掉首步（2–10） |
|---|---:|---:|
| mean old-logprob | 6.081 s | 2.927 s |
| mean actor update | 16.122 s | 12.485 s |
| mean step（包含真实 rollout） | 65.170 s | 56.073 s |
| tokens/s/GPU（包含真实 rollout） | 130.392 | 151.470 |
| actor reported lifetime max allocated | 39.022610 GiB | 39.022610 GiB |
| actor reported lifetime max reserved | 52.048828 GiB | 52.048828 GiB |

所有 batch 的 reward 都是 -2，GRPO advantage 和主 pg_loss 都是 0；MTP/auxiliary
仍参与更新，但这组数据无法验收非零主 policy-gradient 的梯度/参数等价性。
抽查第 1 步缓存：16 条响应均没有 tokenizer EOS（248046），第 1 条结尾仍在数学推导中；
不能把长响应被截断后的统一惩罚当成模型答题精度指标。
此轮主要用于生成固定数据；不能将它的在线整步 TPS 直接与后续 cache replay TPS 比较。

### 固定数据 A/B

两组均完成 10/10 步并退出 0，均逐步命中 10 个 cache；各步 token 数一致。
fused 日志确认 `use_fused_kernels=True`，未出现 capability fallback。
完整逐步指标和汇总见 [机器可读结果](verl_mtp_linear_ce_2k4k_10step_metrics.json)。

主要口径为去掉首步的 2–10 步，共 611,522 tokens：

| 指标 | Unfused | Fused | Fused 相对变化 |
|---|---:|---:|---:|
| mean old-logprob | 2.557 s | 2.631 s | +2.87% 耗时 |
| mean actor update | 12.391 s | 12.646 s | +2.06% 耗时 |
| actor-only tokens/s/GPU | 685.446 | 671.641 | -2.01% 吞吐 |
| mean weight sync | 5.604 s | 5.781 s | +3.17% 耗时 |
| mean replay step | 21.617 s | 22.026 s | +1.89% 耗时 |
| replay tokens/s/GPU | 392.902 | 385.613 | -1.86% 吞吐 |
| actor lifetime max allocated | 39.022610 GiB | 39.022610 GiB | 0 |
| actor lifetime max reserved | 52.048828 GiB | 52.048828 GiB | 0 |
| 1 s 外部采样最高值（任一卡） | 86349 MiB | 86391 MiB | +42 MiB |

全 10 步（含首步）则分别为：actor update 平均 14.102/14.261 s，
replay step 平均 26.449/26.603 s，replay 吞吐 321.283/319.421 tokens/s/GPU。
首步 old-logprob 为 33.019/32.081 s、actor update 为 29.505/28.795 s，
不能代表稳态。

actor update 在第 2–10 步的范围为 11.732–13.121 s / 12.291–13.477 s，
两组存在重叠；weight sync 这种非 CE 阶段也有约 3% 波动。因此准确结论是：
**本次 10 步未观察到 Linear CE 的训练加速，观测吞吐略低；不能把约 2% 差异直接
归因为 kernel 的稳定退化。** 早先短序列单步的约 3% 改善也不能当成稳定加速承诺。

### 数值观测与限制

下表为 10 个同 batch 步骤上 scalar 指标的最大绝对差异；右列是相同 unfused
实现的真实生成轮与 replay 轮之间的参考差异，并非完整的重复实验噪声分布。

| 指标 | Fused vs unfused | Unfused 两轮参考差异 |
|---|---:|---:|
| entropy | 3.4970e-4 | 3.0014e-4 |
| MTP loss | 4.2278e-4 | 2.8196e-4 |
| grad norm | 2.2016e-4 | 1.5622e-4 |
| training log-PPL | 2.3836e-4 | 2.4706e-4 |
| pg_loss | 0 | 0 |

在尚未发生多步更新漂移的第 1 步，entropy 差 2.3246e-6、MTP loss 差 0、
grad norm 差 7.5996e-7、training log-PPL 差 1.6391e-6。

没有 NaN/Inf 或训练中断，但不宣称“精度无损”：advantage 全 0，没有覆盖非零主
policy-gradient；也未逐元素比较梯度/optimizer 更新后的参数，更没有做数学正确率
或多步收敛验收。当前观测只支持这组 shape 的运行与 scalar 数值对照。

### 独立显存快照

吞吐对照之外，另采第 1 步的 rank-0 allocator trace，分别报告全程 watermark、
训练区间峰值及 MTP 路径关联分配。两轮退出码均为 0，结束后八卡均为 0 MiB。
原始分析输出见 [allocator 分析记录](verl_mtp_linear_ce_2k4k_memory.txt)。

| 指标（rank 0，GiB） | Unfused | Fused | Fused - unfused |
|---|---:|---:|---:|
| 全程 max allocated | 39.022610 | 39.022610 | 0 |
| train_batch 区间 max allocated | 38.827215 | 38.305870 | -0.521345 |
| infer_batch 区间 max allocated | 37.867579 | 36.906889 | -0.960690 |
| 快照结束 allocated | 32.392360 | 32.392360 | 0 |
| 快照结束 reserved | 41.296875 | 40.398438 | -0.898437 |
| MTP 路径关联 max live | 5.072664 | 5.072664 | 0 |
| MTP BF16 logits 最大单次分配 | 1.003232 | 1.003232 | 0 |
| MTP CE FP32 最大单次分配 | 2.006464 | 2.006464 | 0 |

- 两个 trace 分别为 448,035 / 447,042 events，均低于 1,000,000 上限，
  反向回放回到 0 bytes，未截断。trace 时间包含 profiling 开销，不用于比较吞吐。
- 全程峰值均发生在初始化、event 16831 之前；31.334952 GiB param/grad buffer
  加 7.687500 GiB TE 初始化分配仍构成几乎全部峰值。
- `--phase-stack-match` 根据事件自身栈定位第一次到最后一次对应调用的包围区间，
  再统计整个区间的 absolute allocated；包含中间的 backward-worker 事件及调用间隙。
  本次各 trace 只有一个训练步。它不是 NVTX 精确边界，也不能外推成所有 rank/所有步的峰值。
- fused trace 实际出现 `linear_cross_entropy.py` 分配（197 次），unfused 为 0，
  证明开关进入了实际 fused 路径；该路径最大单次分配为 dWeight 的 0.473633 GiB。
- MTP 关联的 5.072664 GiB 包含 logits、CE 中间张量和同栈其他分配，不能承诺辅助
  融合可全部回收。当前实现没有融合辅助 CE，所以两组完全相同。
- **有局部显存收益，但没有降低当前完整运行峰值。** 也不能把初始化峰值不变解读为
  长序列下输出头优化永远无用：本批实际 prompt 最大仅 303，未覆盖 2048+4096
  的完整 6144-token 最坏形状；更长有效序列或更大 micro-batch 需要重新测量。

快照与日志均保留在 H20 `/workspace`（宿主 `/AII/ldata/cxli`）：

```text
mtp-linear-ce-2k4k-unfused-profile-v2-snapshot/step1/torch_memory_rank0_pid1062874.pickle
mtp-linear-ce-2k4k-fused-profile-v1-snapshot/step1/torch_memory_rank0_pid1092909.pickle
mtp-linear-ce-2k4k-cache10-v2.log
mtp-linear-ce-2k4k-unfused10-v1.log
mtp-linear-ce-2k4k-fused10-v1.log
mtp-linear-ce-2k4k-unfused10-v1.gpu.csv
mtp-linear-ce-2k4k-fused10-v1.gpu.csv
```

采集命令在上述环境变量基础上设置 `TOTAL_TRAINING_STEPS=1`，为两组分别使用
独立 OUTPUT_DIR/RAY_TMPDIR，并向 shell 追加：

```bash
global_profiler.tool=torch_memory \
global_profiler.steps='[1]' \
global_profiler.save_path=/workspace/mtp-linear-ce-2k4k-unfused-profile-v2-snapshot \
global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries=1000000 \
global_profiler.global_tool_config.torch_memory.stack_depth=32 \
global_profiler.global_tool_config.torch_memory.dump_on_oom=True \
actor_rollout_ref.actor.profiler.enable=True \
actor_rollout_ref.actor.profiler.ranks='[0]'
```

## 提交前检查与状态

- `bash -n`、默认参数 dry-run（2048/4096/6144、10 步 cache 列表）通过。
- 两个分析工具共 4 项标准库 unittest 通过，Ruff check/format、JSON 解析、
  `git diff --check` 通过。
- 10 步 A/B 和两个独立 profiling 轮次均完成。日志中存在 numba 可选依赖导入告警及
  Ray 退出阶段 DataLoader 清理告警；均不影响完整步数和退出码 0，未修改运行时依赖。
- 本次只更改复现/分析脚本及方案、验证记录，未更改 Linear CE/MTP 核心实现。
- 本配置下不建议为吞吐收益强制开启，也不继续实现辅助 CE；保留主头兼容开关，
  若实际生产长序列/更大 micro-batch 使训练临时分配主导整体峰值，再评审辅助头优化。
- 非零 GRPO 梯度、逐元素参数差异、完整并行矩阵及收敛仍未完成，不能标为生产 ready。
