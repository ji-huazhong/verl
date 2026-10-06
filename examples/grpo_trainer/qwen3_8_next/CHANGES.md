# Qwen3.8-Flash-Next 修改清单

分支：`hz/feat/qwen-next`；基线：`3084c2617c9f26defba704359cdd89c9cca0060e`。

## 交付范围

- 模型实现、Bridge out-of-tree 注册、并行权重映射、checkpoint 兼容与数值策略在 verl。
- vLLM 必需补丁、Torch 2.10 ABI 回移与 Bridge Python 3.13 包元数据补丁也保存在本仓库；运行时需要相应构建，不能用 stock wheel 替代。详见 [UPSTREAM_PATCHES.md](UPSTREAM_PATCHES.md)。
- 默认只冻结 PLE 大表，其余可微策略参数和 PLE 小层按 recipe 训练；MTP 关闭。硬 top-k indexer 不新增 auxiliary loss。
- 本次整理没有改变已跑完 100 步的模型数值算法。修复旧 GDN trace 测试夹具、诊断同步改用 verl device API，规范公共脚本名，补齐依赖构建材料、版权与最终报告。
- 历史大 JSON、早期 verl 自身的实验 patch 和旧 README 原样备份到本地 `logs/qwen38-next/development-evidence-20261006/`；公共目录保留可复现诊断代码和精简验证记录。原始训练日志仍在本地忽略目录，不纳入 Git。

## 与外部项目的边界

| 项目 | 是否修改其源码 / 行为 | 实际交付 |
| --- | --- | --- |
| Megatron-Bridge | 仅包元数据；注册采用公共 API | Python 3.13 patch + verl provider/mapping |
| Megatron-Core | 版本限定 CPU optimizer 类方法 shim；GDN 由本地子类覆盖 | `compat.py` + GDN 模块及测试，无私有 Core wheel |
| vLLM | HC/GDN 精度、runtime pool、Torch 2.10/UVA；QSA extension 改变运行时排序 | 源码补丁、版本/哈希约束、构建工具、worker extension |
| Transformer Engine | 本次无源代码改动 | 继续使用 TE，改变 Qwen 投影分组；依赖 ABI 匹配的既有环境 |
| Transformers / FLA | 本次无安装源码修改 | 原生配置；固定版本调用。Q/K norm 在 verl 本地实现 |
| Nebula | 内部提交、打包和日志流程是平台集成 | 不搬入公共库；100 步结束的日志/cleanup 失败单独记录 |

## 逐文件修改

| 文件 | 类别 | 作用 |
| --- | --- | --- |
| `examples/grpo_trainer/qwen3_8_next/.gitattributes` | 启动 / 依赖配置 | 补丁文件的必要空白上下文行属性 |
| `examples/grpo_trainer/qwen3_8_next/CHANGES.md` | 报告 / 验证记录 | 交付文档、100 步聚合指标或依赖补丁清单 |
| `examples/grpo_trainer/qwen3_8_next/README.md` | 报告 / 验证记录 | 交付文档、100 步聚合指标或依赖补丁清单 |
| `examples/grpo_trainer/qwen3_8_next/UPSTREAM_PATCHES.md` | 报告 / 验证记录 | 交付文档、100 步聚合指标或依赖补丁清单 |
| `examples/grpo_trainer/qwen3_8_next/activation_trace.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/backend_trace.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/check_logprobs.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/cuda_constraints.txt` | 启动 / 依赖配置 | 固定 ABI 的独立依赖、约束或 override |
| `examples/grpo_trainer/qwen3_8_next/dependency_build/build_vllm.py` | 外部依赖补丁 / 构建 | 独立 Linux venv、ABI/源码哈希检查与源码构建 |
| `examples/grpo_trainer/qwen3_8_next/dependency_build/package_hc_fp32_wheel.py` | 外部依赖补丁 / 构建 | 复用已验证 CUDA binary 的 Python/Triton wheel 打包和 RECORD 更新 |
| `examples/grpo_trainer/qwen3_8_next/dependency_build/patch_cuda_view_torch210.py` | 外部依赖补丁 / 构建 | UVA from_blob keepalive / deleter 的旧 ABI 兼容 |
| `examples/grpo_trainer/qwen3_8_next/dependency_build/patch_torch210.py` | 外部依赖补丁 / 构建 | 目标 Torch 2.10 build profile |
| `examples/grpo_trainer/qwen3_8_next/dependency_build/prepare_sources.py` | 外部依赖补丁 / 构建 | 精确基线校验、临时树预检查和有序应用依赖补丁 |
| `examples/grpo_trainer/qwen3_8_next/excludes.txt` | 启动 / 依赖配置 | 固定 ABI 的独立依赖、约束或 override |
| `examples/grpo_trainer/qwen3_8_next/gdn_beta_ab.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/metrics-100steps.json` | 报告 / 验证记录 | 交付文档、100 步聚合指标或依赖补丁清单 |
| `examples/grpo_trainer/qwen3_8_next/native_decode_replay_smoke.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/packing_ab.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/patches/bridge-python313.patch` | 外部依赖补丁 / 构建 | 外部源码补丁；适用条件和应用顺序见 UPSTREAM_PATCHES.md |
| `examples/grpo_trainer/qwen3_8_next/patches/manifest.json` | 外部依赖补丁 / 构建 | 交付文档、100 步聚合指标或依赖补丁清单 |
| `examples/grpo_trainer/qwen3_8_next/patches/torch210-compatibility.patch` | 外部依赖补丁 / 构建 | 外部源码补丁；适用条件和应用顺序见 UPSTREAM_PATCHES.md |
| `examples/grpo_trainer/qwen3_8_next/patches/torch210-uva-ownership.patch` | 外部依赖补丁 / 构建 | 外部源码补丁；适用条件和应用顺序见 UPSTREAM_PATCHES.md |
| `examples/grpo_trainer/qwen3_8_next/patches/vllm-gdn-conv-fp32.patch` | 外部依赖补丁 / 构建 | 外部源码补丁；适用条件和应用顺序见 UPSTREAM_PATCHES.md |
| `examples/grpo_trainer/qwen3_8_next/patches/vllm-hc-fp32.patch` | 外部依赖补丁 / 构建 | 外部源码补丁；适用条件和应用顺序见 UPSTREAM_PATCHES.md |
| `examples/grpo_trainer/qwen3_8_next/patches/vllm-model-state-sleep.patch` | 外部依赖补丁 / 构建 | 外部源码补丁；适用条件和应用顺序见 UPSTREAM_PATCHES.md |
| `examples/grpo_trainer/qwen3_8_next/ple-state-overrides.txt` | 启动 / 依赖配置 | 固定 ABI 的独立依赖、约束或 override |
| `examples/grpo_trainer/qwen3_8_next/ple_runtime_analysis.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/ple_runtime_trace.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/ple_sleep2_ab.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/ple_sleep2_analysis.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/precision-overrides.txt` | 启动 / 依赖配置 | 固定 ABI 的独立依赖、约束或 override |
| `examples/grpo_trainer/qwen3_8_next/precision_probe.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/prepare_dapo17k.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/prepare_logprob_prompts.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/production_decode_replay.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/production_trace_analysis.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/qsa_order.py` | 可复现诊断工具 | 所选 block 的统一顺序与 probe |
| `examples/grpo_trainer/qwen3_8_next/qsa_trace.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/real_prompt_trace.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/repetition.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/report.html` | 报告 / 验证记录 | 交付文档、100 步聚合指标或依赖补丁清单 |
| `examples/grpo_trainer/qwen3_8_next/requirements.txt` | 启动 / 依赖配置 | 固定 ABI 的独立依赖、约束或 override |
| `examples/grpo_trainer/qwen3_8_next/run_qwen3_8_next_dapo17k_megatron.sh` | 启动 / 依赖配置 | 公共训练入口、运行环境检查与日志路径 |
| `examples/grpo_trainer/qwen3_8_next/run_qwen3_8_next_gsm8k_megatron.sh` | 启动 / 依赖配置 | 公共训练入口、运行环境检查与日志路径 |
| `examples/grpo_trainer/qwen3_8_next/validate_vllm_model_state_sleep.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `examples/grpo_trainer/qwen3_8_next/validation-summary.json` | 报告 / 验证记录 | 交付文档、100 步聚合指标或依赖补丁清单 |
| `examples/grpo_trainer/qwen3_8_next/vocab_mask_ab.py` | 可复现诊断工具 | 有界、可复现的数值诊断 / 分析入口 |
| `tests/models/mcore/test_qwen38_next_backend_trace_on_cpu.py` | 测试 / 仓库检查 | 回归验证：backend trace on cpu |
| `tests/models/mcore/test_qwen38_next_beta_probe_on_cpu.py` | 测试 / 仓库检查 | 回归验证：beta probe on cpu |
| `tests/models/mcore/test_qwen38_next_decode_replay_on_cpu.py` | 测试 / 仓库检查 | 回归验证：decode replay on cpu |
| `tests/models/mcore/test_qwen38_next_full_parameter_gpu.py` | 测试 / 仓库检查 | 回归验证：full parameter gpu |
| `tests/models/mcore/test_qwen38_next_full_parameter_parallel.py` | 测试 / 仓库检查 | 回归验证：full parameter parallel |
| `tests/models/mcore/test_qwen38_next_gdn_trace_on_cpu.py` | 测试 / 仓库检查 | 回归验证：gdn trace on cpu |
| `tests/models/mcore/test_qwen38_next_packing_probe_on_cpu.py` | 测试 / 仓库检查 | 回归验证：packing probe on cpu |
| `tests/models/mcore/test_qwen38_next_ple_distributed.py` | 测试 / 仓库检查 | 回归验证：ple distributed |
| `tests/models/mcore/test_qwen38_next_ple_mapping_on_cpu.py` | 测试 / 仓库检查 | 回归验证：ple mapping on cpu |
| `tests/models/mcore/test_qwen38_next_ple_runtime_analysis_on_cpu.py` | 测试 / 仓库检查 | 回归验证：ple runtime analysis on cpu |
| `tests/models/mcore/test_qwen38_next_ple_sleep2_on_cpu.py` | 测试 / 仓库检查 | 回归验证：ple sleep2 on cpu |
| `tests/models/mcore/test_qwen38_next_precision_probe_on_cpu.py` | 测试 / 仓库检查 | 回归验证：precision probe on cpu |
| `tests/models/mcore/test_qwen38_next_production_trace_analysis_on_cpu.py` | 测试 / 仓库检查 | 回归验证：production trace analysis on cpu |
| `tests/models/mcore/test_qwen38_next_production_trace_on_cpu.py` | 测试 / 仓库检查 | 回归验证：production trace on cpu |
| `tests/models/mcore/test_qwen38_next_qk_norm_gpu.py` | 测试 / 仓库检查 | 回归验证：qk norm gpu |
| `tests/models/mcore/test_qwen38_next_qsa_order_on_cpu.py` | 测试 / 仓库检查 | 回归验证：qsa order on cpu |
| `tests/models/mcore/test_qwen38_next_qsa_trace_on_cpu.py` | 测试 / 仓库检查 | 回归验证：qsa trace on cpu |
| `tests/models/mcore/test_qwen38_next_real_prompt_trace_on_cpu.py` | 测试 / 仓库检查 | 回归验证：real prompt trace on cpu |
| `tests/models/mcore/test_qwen38_next_repetition_on_cpu.py` | 测试 / 仓库检查 | 回归验证：repetition on cpu |
| `tests/models/mcore/test_qwen38_next_runtime_audit_on_cpu.py` | 测试 / 仓库检查 | 回归验证：runtime audit on cpu |
| `tests/models/mcore/test_qwen38_next_vocab_mask_on_cpu.py` | 测试 / 仓库检查 | 回归验证：vocab mask on cpu |
| `tests/special_sanity/check_license.py` | 测试 / 仓库检查 | 允许准确的 2026 Individual Contributor 版权年份 |
| `tests/trainer/ppo/v1/test_logprob_context_metrics_on_cpu.py` | 测试 / 仓库检查 | 回归验证：logprob context metrics on cpu |
| `tests/utils/debug/test_logprob_capture.py` | 测试 / 仓库检查 | 回归验证：logprob capture |
| `tests/utils/debug/test_metrics.py` | 测试 / 仓库检查 | 回归验证：metrics |
| `tests/utils/test_padding_on_cpu.py` | 测试 / 仓库检查 | 回归验证：padding on cpu |
| `verl/models/mcore/patch.py` | verl 公共集成 | recompute backward 保持 checkpointing 标志 |
| `verl/models/mcore/qwen3_8_next/THIRD_PARTY.md` | 模型适配 / 兼容层 | 保留 Miles / Bridge 的来源、许可和本地修改范围 |
| `verl/models/mcore/qwen3_8_next/__init__.py` | 模型适配 / 兼容层 | 显式 Python 包边界 |
| `verl/models/mcore/qwen3_8_next/bridge.py` | 模型适配 / 兼容层 | 外部注册与 HF ↔ Megatron 映射，覆盖检查和跨 rank 权重导出 |
| `verl/models/mcore/qwen3_8_next/compat.py` | 模型适配 / 兼容层 | Core 0.19.2 CPU optimizer resume 的版本限定兼容修复 |
| `verl/models/mcore/qwen3_8_next/config.py` | 模型适配 / 兼容层 | 模型配置转换、并行组合与不支持功能的提前校验 |
| `verl/models/mcore/qwen3_8_next/gdn_mapping.py` | 模型适配 / 兼容层 | qkvz / ba 分组 TP 权重映射 |
| `verl/models/mcore/qwen3_8_next/hyper_connection.py` | 模型适配 / 兼容层 | HC 多残差流组合、分配与头部收缩 |
| `verl/models/mcore/qwen3_8_next/layer.py` | 模型适配 / 兼容层 | 把 HC 和 PLE 接入 transformer layer / checkpoint 流程 |
| `verl/models/mcore/qwen3_8_next/ops/__init__.py` | 模型适配 / 兼容层 | 显式 Python 包边界 |
| `verl/models/mcore/qwen3_8_next/ops/attention.py` | 模型适配 / 兼容层 | QSA attention、mRoPE 与 packed CP |
| `verl/models/mcore/qwen3_8_next/ops/context_parallel.py` | 模型适配 / 兼容层 | QSA packed CP 的投影/KV交换 |
| `verl/models/mcore/qwen3_8_next/ops/gated_delta_net.py` | 模型适配 / 兼容层 | FP32 Q/K 归一化与 gated norm；默认 FLA，FlashQLA 可选 |
| `verl/models/mcore/qwen3_8_next/ops/gdn_projection.py` | 模型适配 / 兼容层 | 两组 TE 输入投影，训练梯度及 shard state |
| `verl/models/mcore/qwen3_8_next/ops/kernel/__init__.py` | 模型适配 / 兼容层 | HC / PLE / QSA 的数值 kernel（保留上游来源） |
| `verl/models/mcore/qwen3_8_next/ops/kernel/hc_triton.py` | 模型适配 / 兼容层 | HC / PLE / QSA 的数值 kernel（保留上游来源） |
| `verl/models/mcore/qwen3_8_next/ops/kernel/ple_gather.py` | 模型适配 / 兼容层 | HC / PLE / QSA 的数值 kernel（保留上游来源） |
| `verl/models/mcore/qwen3_8_next/ops/kernel/ple_triton.py` | 模型适配 / 兼容层 | HC / PLE / QSA 的数值 kernel（保留上游来源） |
| `verl/models/mcore/qwen3_8_next/ops/kernel/qsa_block_sparse_attn.py` | 模型适配 / 兼容层 | HC / PLE / QSA 的数值 kernel（保留上游来源） |
| `verl/models/mcore/qwen3_8_next/ops/kernel/qsa_sparse_attn.py` | 模型适配 / 兼容层 | HC / PLE / QSA 的数值 kernel（保留上游来源） |
| `verl/models/mcore/qwen3_8_next/ops/ple.py` | 模型适配 / 兼容层 | host-frozen / trainable lookup table、读取小层及上下文钩子 |
| `verl/models/mcore/qwen3_8_next/ops/ple_context_parallel.py` | 模型适配 / 兼容层 | packed CP 的 PLE 上下文与梯度通信 |
| `verl/models/mcore/qwen3_8_next/ops/qsa_indexer.py` | 模型适配 / 兼容层 | QSA 候选选择与块索引 |
| `verl/models/mcore/qwen3_8_next/ops/sequence.py` | 模型适配 / 兼容层 | packed 文档边界与 token 索引辅助 |
| `verl/models/mcore/qwen3_8_next/param_mapping.py` | 模型适配 / 兼容层 | PLE 表按 HF shard 延迟转换 / 导出，避免全表 GPU 复制 |
| `verl/models/mcore/qwen3_8_next/production_trace.py` | 诊断运行时 | 有界、可复现的数值诊断 / 分析入口 |
| `verl/models/mcore/qwen3_8_next/provider.py` | 模型适配 / 兼容层 | HC/GDN/QSA/PLE layer spec、packed PLE context 与模型构造 |
| `verl/models/mcore/qwen3_8_next/qsa_order.py` | 模型适配 / 兼容层 | 所选 block 的统一顺序与 probe |
| `verl/models/mcore/qwen3_8_next/runtime_audit.py` | 诊断运行时 | 有界、可复现的数值诊断 / 分析入口 |
| `verl/models/mcore/qwen3_8_next/vllm_worker_extension.py` | 模型适配 / 兼容层 | 显式选择的 QSA 顺序 extension；保留 router FP32 候选 |
| `verl/models/mcore/registry.py` | verl 公共集成 | 登记 Qwen4Exp VLM 架构 |
| `verl/trainer/config/qwen3_8_next_dapo17k.yaml` | 训练配置 | GSM8K / DAPO / debug / 128 GPU Hydra profile |
| `verl/trainer/config/qwen3_8_next_dapo17k_128gpu.yaml` | 训练配置 | GSM8K / DAPO / debug / 128 GPU Hydra profile |
| `verl/trainer/config/qwen3_8_next_dapo17k_debug.yaml` | 训练配置 | GSM8K / DAPO / debug / 128 GPU Hydra profile |
| `verl/trainer/config/qwen3_8_next_gsm8k.yaml` | 训练配置 | GSM8K / DAPO / debug / 128 GPU Hydra profile |
| `verl/trainer/ppo/v1/trainer_base.py` | verl 公共集成 | 可选真实 response / route 捕获和上下文分桶读数 |
| `verl/utils/debug/logprob_capture.py` | verl 公共集成 | 保持真实 token、route、mask 的有限采样归档 |
| `verl/utils/debug/metrics.py` | verl 公共集成 | sample/token mean、最大及 nonfinite logprob 差和上下文分桶 |
| `verl/utils/import_utils.py` | verl 公共集成 | 外部模型插件 rollout 配置校验入口 |
| `verl/utils/tensordict_utils.py` | verl 公共集成 | nested MRoPE jagged 轴的显式构造 |
| `verl/utils/transferqueue_utils.py` | verl 公共集成 | TQ 接收后规范化 3D position_ids |
| `verl/workers/engine/megatron/transformer_impl.py` | verl 公共集成 | HF 初始加载与 dist save 解耦、lazy export 禁用 autograd、可选 trace |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | verl 公共集成 | 插件配置校验和可选真实 response 审计 |
| `verl/workers/utils/padding.py` | verl 公共集成 | 等长 MRoPE row 也保持正确 jagged 轴 |
