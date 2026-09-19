# verl MTP 主头 Fused Linear CE：实施与验证记录

Last updated: 09/19/2026

> 过程文档：随当前特性分支暂存于 `todo/`，最终 ready 后可移除。
> 当前状态：实现完成，CPU 回归、H20 单卡/TP2 定向 GPU 验证，以及 Qwen3.5-35B-A3B + DAPO 数据的 8 卡 Megatron GRPO 单步对照均已通过。目标 PyTorch 2.11、PP/CP/Dynamic CP 和多步训练仍待验收。

## 1. 实施范围与决策

- 分支：`hz/feat/mtp-fused-linear-ce`。
- 基线：`a95cebd05594c7f556a02c1b45159a81d0d577f7`。
- 方案：[更新后的实施方案及历史草案](verl_mtp_fused_linear_cross_entropy.md)。
- 仅让主头已有 fused Linear CE 与 MTP 共存，复用 `use_fused_kernels`；未增加配置开关或 Triton kernel。
- 保留原生 MCore 与 legacy 辅助 CE 的梯度、mask、loss scaling 和日志语义；不实现辅助 CE 融合，不强制 detach 输出权重。
- 优先验证低侵入的主头兼容收益，再决定是否推进辅助头优化。8 卡真实模型 smoke 的稳态 actor update 快 0.93%，整步快 3.18%；未观察到有意义的全局显存峰值降低。因此保留本次主头兼容改造，但不继续投入辅助头 fused CE。

## 2. 实施经过与代码落点

1. 在 `verl/models/mcore/model_forward.py` 提取共享 labels/mask 对齐函数，供普通与 fused THD 路径使用。
2. 在 `verl/models/mcore/model_forward_fused.py` 分离主头 context labels 与控制 MTP 训练的 model labels；传递 packed positions、response mask、Dynamic CP group 及 loss normalization 元数据，补齐 PP 末端输入打包。
3. 在 `verl/models/mcore/mtp_patch.py` 保留完整辅助 loss 路径，在辅助 loss 注入 hidden 之后调用主头 output processor；无 hook 时仍走原 logits 路径。
4. 在 `verl/workers/engine/megatron/transformer_impl.py` 将 MTP 一刀切禁用改为能力检查，先检查本地所有 pipeline chunks，再安装 patch；不支持的配置明确回退。需要完整 logits 的 top-K distillation 显式报错。
5. 补充混合梯度兼容：原生辅助头可能直接累加 `main_grad`，主 fused CE 则返回 autograd 梯度。设置 MCore 的 `zero_out_wgrad=True`，避免主头梯度被 `grad_added_to_main_grad` 分支跳过，同时确保辅助分支 dummy wgrad 为零；覆盖 tied/untied 和两个 microbatch 的回归。
6. 新增 CPU 合约/梯度回归及 Megatron Engine gate 测试，更新 `docs/advance/mtp.md` 使用说明。
7. H20 的 MCore 0.20 暴露 `FullyShardedDataParallel` factory 而非 wrapper class；旧分支在模型 unwrap 和 optimizer training-hook 注册两处将它传给 `isinstance`/union，分别会报 `TypeError`。统一只注册真实 V1/V2 wrapper class，并增加类型与 training-hook 回归测试。MCore 0.20 的 wrapper-chain 测试夹具同时补齐新增的只读 config 字段。
8. Qwen3.5-35B-A3B 使用外层多模态 wrapper，但其 `language_model` 是原生 GPTModel，且 wrapper 通过 `**kwargs` 透传 output-processor hook。能力检查改为复用实际 patch 目标解析，支持这类 wrapper，同时继续拒绝不具备 GPTModel language-model contract 的封装。
9. MCore 0.20 的 MTP `_get_embeddings` 新增 `mtp_input_mask`，原 patch 会在首个真实训练步报未知参数。patch 现在按运行时签名透传/平移 mask，保留 sequence-parallel scatter；`detach_encoder=True` 则显式创建需求梯度的 detached leaf，保留 MTP 参数梯度且阻断 encoder 梯度。
10. 新增可复现 H20 GRPO 脚本 `todo/run_h20_qwen35_mtp_linear_ce_grpo.sh`，支持用 `FUSED_KERNELS=False/True` 和同一 rollout cache 做 A/B。

原始方案曾将辅助权重一概视为 detached；实现前已修正。legacy 与不同版本的原生 MCore 并不保证相同梯度需求，后续 dHidden-only 优化必须以原有语义为前提。

## 3. 本地验证环境

- 平台：macOS arm64，CUDA 不可用。
- 临时测试环境：Python 3.12.14、PyTorch 2.14.0、pytest 9.1.1，通过 uv 管理。
- Ruff：0.12.2，与 `.pre-commit-config.yaml` 一致。
- 仓库 `pyproject.toml` 目标 PyTorch 为 2.11.0。本次临时环境与目标版本不一致，目标版本验证待补。
- CPU 测试使用真实 PyTorch autograd 和真实 verl packing/forward/postprocess；MCore collectives、原生辅助处理、AutoScaler、DDP 行为及 Triton 运算以显式 stub/替身覆盖相关接口。
- 这些测试证明本地逻辑和模拟梯度合约，不证明真实 NCCL、MCore DDP、CUDA kernel 或端到端训练正确性。

## 4. H20 定向验证环境（2026-09-19）

- GPU：8 × NVIDIA H20-3e，单卡 143771 MiB；驱动 615.71.09。测试开始和结束时八卡均无其他显存占用。
- 容器：`docker.m.daocloud.io/vllm/vllm-openai:v0.27.1`，image id `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`。
- Python 3.12.13、PyTorch 2.13.0+cu130、Triton 3.7.1、pytest 9.1.1、Transformer Engine 2.18.0。
- 使用参考项目源码目录中的 Megatron-Core 0.20.0；该目录不是可解析 commit 的 Git checkout，因此没有记录 MCore SHA。
- 被测代码以 commit `893b998125e835f696c4f355777b35f103cda318` 为基础，包含本记录中的 MCore 0.20/Qwen3.5 兼容修正，同步到独立目录 `/workspace/verl-mtp-fused-linear-ce-current`。没有修改参考项目的 `/workspace/verl` 脏工作区。
- 真实模型：`/workspace/models/Qwen3.5-35B-A3B`，Qwen3.5 MoE/VLM wrapper，hidden size 2048、vocab 248320、40 层、256 experts/top-8、1 层 MTP。
- 数据：`/workspace/data/DAPO-Math-17k/data/dapo-math-17k.parquet`。该路径名称为 DAPO-Math-17k，但实际 parquet 在运行时报告 `1,791,700` 行；脚本关闭全库 overlong 预过滤，只消费本次 batch。
- 8 卡开始前 GPU 4–7 有一组无关 Qwen SFT 作业。按用户指令在定位容器后只结束其 launcher，确认 8 卡均为 0 MiB 再启动对照；所有对照结束后 8 卡再次均为 0 MiB。
- 当前仓库声明的目标 PyTorch 是 2.11.0；H20 结果验证了另一套较新的运行时，不能替代目标版本 CI。

## 5. 已执行命令与结果

以下命令均在 verl 仓库根目录执行。临时环境路径是本机执行记录，不是其他机器的安装要求。

### CPU 回归

```bash
/private/tmp/verl-mtp-ce-tests/bin/python -m pytest -q \
  tests/models/test_mtp_fused_main_ce_on_cpu.py --tb=short
```

提交前复验结果：`44 passed in 1.82s`，退出码 0。

覆盖：普通/fused 打包一致性、nested/padded response mask、zigzag/contiguous、FP8 padding、native/legacy 辅助梯度、K=1/2、per-token normalization、tied/untied、零 mask、仅加载 MTP、主 hook 顺序、PP stages、context labels、positions、fallback 与 Dynamic CP 元数据。FP8 packing 与 Dynamic CP 的底层测试不代表完整 Engine 支持二者组合。

### 静态检查

```bash
mtp_checked_files=(
  verl/models/mcore/model_forward_fused.py
  verl/models/mcore/mtp_patch.py
  verl/utils/megatron_utils.py
  tests/models/test_model_forward_fused.py
  tests/models/test_mtp_fused_main_ce_on_cpu.py
  tests/special_distributed/test_megatron_dynamic_cp_features.py
  tests/utils/test_megatron_mtp_dcp.py
  todo/run_h20_mtp_fused_linear_ce_validation.py
)
uvx --from ruff==0.12.2 ruff check "${mtp_checked_files[@]}"
uvx --from ruff==0.12.2 ruff format --check "${mtp_checked_files[@]}"
.venv/bin/python -m compileall -q "${mtp_checked_files[@]}"
bash -n todo/run_h20_qwen35_mtp_linear_ce_grpo.sh
git diff --check
```

结果：Ruff `All checks passed!`；格式 `8 files already formatted`；语法编译、shell 语法和 diff 检查退出码均为 0。只检查本次相关文件，未声称全仓测试或全套 pre-commit 通过。

### H20 完整 MCore 合约回归

在 MCore 0.20 / TE 2.18 源码环境、没有 runtime monkeypatch 的最终命令中运行四个定向测试文件：

```bash
python -m pytest -q \
  tests/models/test_mtp_fused_main_ce_on_cpu.py \
  tests/models/test_model_forward_fused.py \
  tests/utils/test_megatron_mtp_dcp.py \
  tests/workers/test_megatron_mtp_fused_gate.py --tb=short
```

结果：`66 passed, 22 warnings in 13.29s`。warnings 为依赖弃用提示和未安装的非本任务可选 engine；没有跳过或 xfail。它覆盖真实 MCore wrapper/hook API，以及 0.20 `mtp_input_mask`/detach 梯度合约，但大部分用例本身仍是 CPU 合约测试。

### H20 单卡 native MTP + 实际 Triton

可复现脚本：`todo/run_h20_mtp_fused_linear_ce_validation.py mtp`。使用 BF16、384 tokens、hidden 512、vocab 32768、MCore 0.20 原生 `process_mtp_loss` 和真实 verl Triton Linear CE；比较非 fused 与 fused 主头的输出及包含辅助 loss 的梯度：

| 项目 | 最大绝对差 |
|---|---:|
| 主头 log-probability | 0.00012016296 |
| 主头 entropy | 0.0000057220459 |
| MTP 输入 hidden gradient | 0.0000019073486 |
| output weight gradient | 0.00000047683716 |

结果：`integration=PASS native_auxiliary=true actual_triton=true`。同时断言 fused 返回不含 logits，并启用混合 wgrad 的 `zero_out_wgrad` 契约。该测试采用合成模型并关闭指标日志，不等同于真实 DDP 训练。

### H20 Linear CE 数值、性能和显存

- 非整除词表回归：`tests/utils/test_linear_cross_entropy.py::test_lce_non_divisible_vocab_padding`，`1 passed in 7.43s`。
- 单卡 case 0：BF16、1937 tokens、hidden 3584、vocab 152064，3 次数值前反向全部通过。
- TP2 case 0：两个 H20 rank、NCCL、同一 shape，3 次数值前反向全部通过；命令为 `torch.distributed.run --nproc-per-node=2 todo/run_h20_mtp_fused_linear_ce_validation.py tp2`。

| 场景 | 实现 | Forward | Backward | Forward peak allocated | Backward peak allocated |
|---|---|---:|---:|---:|---:|
| 单卡 | PyTorch reference | 79.43 ms | 174.71 ms | 6595.28 MiB | 11091.29 MiB |
| 单卡 | Linear CE kernel | 17.53 ms | 52.63 ms | 1120.68 MiB | 2219.30 MiB |
| TP2（每 rank 本地词表 152064） | PyTorch TP reference | 92.21 ms | 188.42 ms | 8954.05 MiB | 18987.07 MiB |
| TP2（每 rank 本地词表 152064） | Linear CE kernel | 17.71 ms | 52.30 ms | 1121.20 MiB | 2219.30 MiB |

峰值是现有测试在输入/权重已分配后重置统计，再读取 `max_memory_allocated` 的结果，因此数值包含当时仍存活的输入和参数，不是单个算子的净新增分配。时延为同一进程内预热后两次均值，仅用于该合成 shape 的方向性对照，不能外推为完整训练吞吐。

仓库综合 benchmark 中另一条 `FusedLinearForPPO + torch.compile` 对照在 PyTorch 2.13/Triton 3.7 上触发 Dynamo `SymNodeVariable.value` 内部错误。它不是本特性使用的 `linear_cross_entropy` 路径；定向测试绕开该对照并单独记录，未把失败改记为通过。

### H20 8 卡 Qwen3.5-35B-A3B GRPO 对照

可复现入口：`todo/run_h20_qwen35_mtp_linear_ce_grpo.sh`。关键配置：

- 8 × H20-3e，TP=2、EP=8、PP=1、CP=1、SP/remove-padding/THD 开启；
- GRPO，train batch=8、rollout n=2、prompt 上限 512、response 上限 256、actor micro-batch=1；
- `mtp.enable_train=True`、`detach_encoder=True`、MTP loss scale=0.1；
- BF16，full recompute/uniform/1 layer，MCore precision-aware CPU optimizer offload；
- rollout 使用 vLLM TP=8、memory utilization=0.30。先生成 8 prompts × 2 responses，再对 unfused/fused 重放同一份 `tq_batch.pt`。

固定 batch 共 6850 tokens，16 条 response 都达到 256 tokens。这次样本的 DAPO reward 全为 -2，因此 group-relative advantages 和 policy-gradient loss 为 0。它仍完整执行 actor old-logprob、MTP auxiliary loss、backward、optimizer step 和权重同步，但不能单独证明非零主 policy-gradient 的端到端等价；该缺口由上述 native-MTP + Triton 非零 hidden/output-weight 梯度对照补充。

| 指标 | Unfused replay | Fused 首次 | Fused warm replay | Warm 相对 unfused |
|---|---:|---:|---:|---:|
| actor entropy | 0.230080724 | 0.230087012 | 0.230087012 | abs diff 0.000006288 |
| MTP loss | 0.300186664 | 0.300186664 | 0.300186664 | 完全一致 |
| actor grad norm | 0.210171551 | 0.210171551 | 0.210171551 | 完全一致 |
| training log-PPL | 0.238395661 | 0.238270476 | 0.238270476 | abs diff 0.000125185 |
| old-logprob | 30.891 s | 32.560 s | 29.107 s | -5.78% |
| actor update | 25.834 s | 39.655 s | 25.593 s | -0.93% |
| weight update/sync | 5.102 s | 5.168 s | 5.188 s | +1.69% |
| 整步（cache replay） | 62.965 s | 78.570 s | 60.966 s | -3.18% |
| 吞吐 | 13.599 tokens/s | 10.898 tokens/s | 14.045 tokens/s | +3.28% |
| actor max allocated | 39.023 GiB | 39.023 GiB | 39.023 GiB | 0 |
| actor max reserved | 52.049 GiB | 52.049 GiB | 52.049 GiB | 0 |
| 2 s `nvidia-smi` 最高样本 | 85249 MiB | 79115 MiB | 85043 MiB | -206 MiB（-0.24%） |

fused 首次的 actor update 包含 Triton 首次编译，因此比 unfused 慢 53.5%；缓存命中后恢复到与 unfused 接近并小幅更快。这说明生产作业应预热/持久化 Triton cache，不应用首个 step 代表稳态吞吐。

显存方面，PyTorch actor 峰值完全相同，2 秒外部采样仅差 206 MiB，且 fused 首次采样明显漏掉短暂峰值。本 smoke 只融合主头，MTP auxiliary logits 仍物化；在当前 512+256 短序列、recompute 和 CPU optimizer offload 组合下，主头 logits 不是全局峰值的决定项。因此结论是“未观察到有意义的全局显存降低”，不将 206 MiB 解读为可稳定复现的算子净收益。上述合成 shape 的算子级显存收益仍成立，但不能直接外推到本完整训练峰值。

所有三次 replay 均记录 `use_fused_kernels` 的期望值，fused 无 capability fallback warning，并以退出码 0 完成 100% 的训练步。Ray 关闭期间有 DataLoader worker 被清理的 atexit 告警，发生在最终 metrics 与 100% progress 之后，未改变进程退出码或该步结果。

### 排障过程与 4 卡试跑

- vLLM utilization=0.35 时在 4 卡 actor/rollout colocate 环境无法预留要求的显存，将可复现脚本默认值降为 0.30。
- DAPO reward 的 overlong log 配置在该运行时不能缺省，脚本显式设置 `log=False`。
- 首个真实 MTP forward 暴露 MCore 0.20 `mtp_input_mask` 签名和 `detach_encoder` leaf 问题，修正后由 66 项合约测试和 8 卡完整步共同验证。
- 4 卡关闭 CPU optimizer offload 时，forward/backward 已完成，但 Adam 状态首次懒加载在约 129.6 GiB PyTorch allocated 处 OOM；这是 optimizer 峰值，不是 Linear CE 失败。恢复 Qwen3.5 官方 precision-aware CPU optimizer offload 后，4 卡单步可完成；最终 A/B 改为用户指定的 8 卡。

## 6. 尚未执行的验证

### 目标版本与完整训练栈

在仓库要求的 PyTorch 2.11.0 + 对应 Megatron/verl 环境执行：

```bash
pytest -q tests/models/test_mtp_fused_main_ce_on_cpu.py
pytest -q tests/models/test_model_forward_fused.py \
  tests/utils/test_megatron_mtp_dcp.py \
  tests/workers/test_megatron_mtp_fused_gate.py
```

H20 已在 PyTorch 2.13 + MCore 0.20 上运行上述四个文件和真实模型 Megatron Engine 单步对照；仓库目标 PyTorch 2.11.0 + 对应 MCore/TE 的同组测试仍待执行。单步对照也不能代替多步收敛、实际长序列峰值或完整并行矩阵的生产验收。

### 目标 GPU 对照与最终 ready 标准

- 已完成 TP2+SP、EP8、`enable_train=true`、`detach_encoder=true` 的 8 卡真实模型单步对照。尚需覆盖 `enable_train=false`、TP1、PP2、CP2/THD 及部署所需的 Dynamic CP，并在 Engine 支持的配置内单独验证 fallback。
- 运行非零 group-relative advantage 的固定 batch，比较主 policy-gradient 参数梯度及 optimizer step 后参数差异；当前 DAPO smoke 的 policy-gradient 为 0。
- 记录主 log-probability/entropy、各层 MTP loss、各参数梯度和 optimizer step 后参数差异。预先给定与 dtype 相适应的绝对/相对容差，不只比较 loss 或 grad norm。
- 特别检查真实 MCore DDP 下主头/辅助头混合梯度，及 pipeline/tied weight 的同步结果。
- 预热后同步 CUDA，重置峰值计数并测量相同区间；记录 allocated/reserved 峰值、step time、tokens/s、通信及输出头相关分配。保存模型、依赖版本、硬件、序列长度、batch 和并行配置。
- 主头不再物化完整 logits，但辅助头仍会物化；现有 split-N backward 也保留局部 dLogits buffer。不能按 `(1+K)` 直接估算实际节省，更不能据此宣称加速。
- 完成上述数值、梯度和并行矩阵验证并经人工 review 后，才能判定生产 ready。当前实测不支持为“进一步降全局峰值”立即实现 MTP auxiliary fused CE；保留关闭开关的部署选择，若长序列生产 profile 确认 auxiliary logits 是峰值主因，再重新评审辅助头优化。

## 7. 发现但未纳入本次修改的问题

短序列 FP8 packing 边界：两条 8-token 输入、TP=2、CP=2、FP8 hybrid、zigzag 下，原有非 fused `preprocess_thd_engine` 已因尾部总长度补齐触发 shape 错误（expanded 252 vs existing 16）。这是本次测试发现的独立基线问题，未修改；对齐回归改用基线可运行的 256/512-token 长度，仍覆盖 padding。

完整 Engine 已拒绝 Dynamic CP + FP8 组合，本次没有放开这个限制。

## 8. 回退与文档管理

- 关闭 `actor_rollout_ref.model.use_fused_kernels` 即回到既有主头 logits 路径，不需要变更 checkpoint 或 MTP 训练配置。
- `todo/verl_mtp_fused_linear_cross_entropy.md`、本文件、`todo/run_h20_mtp_fused_linear_ce_validation.py` 和 `todo/run_h20_qwen35_mtp_linear_ce_grpo.sh` 是分支内暂存的方案/过程资料；按用户要求保留，最终 ready 后由用户决定移除。
- `docs/advance/mtp.md` 是功能使用说明，应在移除过程资料后保留。
- 本阶段尚未创建 PR，也未完成上游 PR 所需的查重和人工验收流程。
