# verl MTP 主头 Fused Linear CE：实施与验证记录

Last updated: 09/04/2026

> 过程文档：随当前特性分支暂存于 `todo/`，最终 ready 后可移除。
> 当前状态：实现完成，本地 CPU 回归通过；完整依赖栈、分布式及 GPU 验收未完成。

## 1. 实施范围与决策

- 分支：`hz/feat/mtp-fused-linear-ce`。
- 基线：`a95cebd05594c7f556a02c1b45159a81d0d577f7`。
- 方案：[更新后的实施方案及历史草案](verl_mtp_fused_linear_cross_entropy.md)。
- 仅让主头已有 fused Linear CE 与 MTP 共存，复用 `use_fused_kernels`；未增加配置开关或 Triton kernel。
- 保留原生 MCore 与 legacy 辅助 CE 的梯度、mask、loss scaling 和日志语义；不实现辅助 CE 融合，不强制 detach 输出权重。
- 优先验证低侵入的主头兼容收益，再决定是否推进辅助头优化。当前没有实测显存或吞吐收益。

## 2. 实施经过与代码落点

1. 在 `verl/models/mcore/model_forward.py` 提取共享 labels/mask 对齐函数，供普通与 fused THD 路径使用。
2. 在 `verl/models/mcore/model_forward_fused.py` 分离主头 context labels 与控制 MTP 训练的 model labels；传递 packed positions、response mask、Dynamic CP group 及 loss normalization 元数据，补齐 PP 末端输入打包。
3. 在 `verl/models/mcore/mtp_patch.py` 保留完整辅助 loss 路径，在辅助 loss 注入 hidden 之后调用主头 output processor；无 hook 时仍走原 logits 路径。
4. 在 `verl/workers/engine/megatron/transformer_impl.py` 将 MTP 一刀切禁用改为能力检查，先检查本地所有 pipeline chunks，再安装 patch；不支持的配置明确回退。需要完整 logits 的 top-K distillation 显式报错。
5. 补充混合梯度兼容：原生辅助头可能直接累加 `main_grad`，主 fused CE 则返回 autograd 梯度。设置 MCore 的 `zero_out_wgrad=True`，避免主头梯度被 `grad_added_to_main_grad` 分支跳过，同时确保辅助分支 dummy wgrad 为零；覆盖 tied/untied 和两个 microbatch 的回归。
6. 新增 CPU 合约/梯度回归及 Megatron Engine gate 测试，更新 `docs/advance/mtp.md` 使用说明。

原始方案曾将辅助权重一概视为 detached；实现前已修正。legacy 与不同版本的原生 MCore 并不保证相同梯度需求，后续 dHidden-only 优化必须以原有语义为前提。

## 3. 本地验证环境

- 平台：macOS arm64，CUDA 不可用。
- 临时测试环境：Python 3.12.14、PyTorch 2.14.0、pytest 9.1.1，通过 uv 管理。
- Ruff：0.12.2，与 `.pre-commit-config.yaml` 一致。
- 仓库 `pyproject.toml` 目标 PyTorch 为 2.11.0。本次临时环境与目标版本不一致，目标版本验证待补。
- CPU 测试使用真实 PyTorch autograd 和真实 verl packing/forward/postprocess；MCore collectives、原生辅助处理、AutoScaler、DDP 行为及 Triton 运算以显式 stub/替身覆盖相关接口。
- 这些测试证明本地逻辑和模拟梯度合约，不证明真实 NCCL、MCore DDP、CUDA kernel 或端到端训练正确性。

## 4. 已执行命令与结果

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
  verl/models/mcore/model_forward.py
  verl/models/mcore/model_forward_fused.py
  verl/models/mcore/mtp_patch.py
  verl/workers/engine/megatron/transformer_impl.py
  tests/models/test_model_forward_fused.py
  tests/models/test_mtp_fused_main_ce_on_cpu.py
  tests/workers/test_megatron_mtp_fused_gate.py
)
/private/tmp/verl-mtp-lint/bin/ruff check "${mtp_checked_files[@]}"
/private/tmp/verl-mtp-lint/bin/ruff format --check "${mtp_checked_files[@]}"
.venv/bin/python -m compileall -q "${mtp_checked_files[@]}"
git diff --check
```

结果：Ruff `All checks passed!`；格式 `7 files already formatted`；语法编译和 diff 检查退出码均为 0。只检查本次相关文件，未声称全仓测试或全套 pre-commit 通过。

## 5. 尚未执行的验证

### 完整依赖栈回归

在仓库要求的 PyTorch 2.11.0 + 对应 Megatron/verl 环境执行：

```bash
pytest -q tests/models/test_mtp_fused_main_ce_on_cpu.py
pytest -q tests/models/test_model_forward_fused.py \
  tests/utils/test_megatron_mtp_dcp.py \
  tests/workers/test_megatron_mtp_fused_gate.py
```

后面三个文件本地未运行，包括新增 Engine gate 测试；本地只做了涉及改动文件的静态检查。上述测试也不能代替真实 GPU 分布式训练对照。

### 目标 GPU 对照与最终 ready 标准

- 固定 checkpoint、seed、输入 batch、有效 token 数、dtype 和并行配置，分别运行 `use_fused_kernels=false/true`；`mtp.enable=true`，覆盖 `enable_train=false/true`。
- TP1、TP2+SP、PP2、CP2/THD，以及部署所需的 Dynamic CP；覆盖 tied/untied、`detach_encoder`、per-token normalization 和多个 microbatch。只在 Engine 支持的配置内测试，单独验证 fallback。
- 记录主 log-probability/entropy、各层 MTP loss、各参数梯度和 optimizer step 后参数差异。预先给定与 dtype 相适应的绝对/相对容差，不只比较 loss 或 grad norm。
- 特别检查真实 MCore DDP 下主头/辅助头混合梯度，及 pipeline/tied weight 的同步结果。
- 预热后同步 CUDA，重置峰值计数并测量相同区间；记录 allocated/reserved 峰值、step time、tokens/s、通信及输出头相关分配。保存模型、依赖版本、硬件、序列长度、batch 和并行配置。
- 主头不再物化完整 logits，但辅助头仍会物化；现有 split-N backward 也保留局部 dLogits buffer。不能按 `(1+K)` 直接估算实际节省，更不能据此宣称加速。
- 完成上述数值、梯度和性能验证并经人工 review 后，才能判定生产 ready。若收益不足或出现吞吐回退，保留关闭开关的部署选择，再决定是否需要辅助头优化。

## 6. 发现但未纳入本次修改的问题

短序列 FP8 packing 边界：两条 8-token 输入、TP=2、CP=2、FP8 hybrid、zigzag 下，原有非 fused `preprocess_thd_engine` 已因尾部总长度补齐触发 shape 错误（expanded 252 vs existing 16）。这是本次测试发现的独立基线问题，未修改；对齐回归改用基线可运行的 256/512-token 长度，仍覆盖 padding。

完整 Engine 已拒绝 Dynamic CP + FP8 组合，本次没有放开这个限制。

## 7. 回退与文档管理

- 关闭 `actor_rollout_ref.model.use_fused_kernels` 即回到既有主头 logits 路径，不需要变更 checkpoint 或 MTP 训练配置。
- `todo/verl_mtp_fused_linear_cross_entropy.md` 与本文件是分支内暂存的方案/过程资料；按用户要求提交，最终 ready 后由用户决定移除。
- `docs/advance/mtp.md` 是功能使用说明，应在移除过程资料后保留。
- 本阶段尚未创建 PR，也未完成上游 PR 所需的查重和人工验收流程。
