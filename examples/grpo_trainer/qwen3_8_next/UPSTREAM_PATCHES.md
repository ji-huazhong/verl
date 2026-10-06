# 外部依赖修改与上游交付边界

本次模型适配的源代码在 verl；外部依赖修改也以补丁、应用工具、构建工具和测试入口保存在 verl。**这不代表运行时只需 stock vLLM / stock Bridge**：Python 3.13 / Torch 2.10 / CUDA 13.1 环境仍须使用本目录描述的依赖构建。这里区分正确性缺陷、数值策略和特定环境兼容，方便拆分上游反馈。

没有修改已安装 Bridge 的模型注册表，也没有修改 Transformers 注册表。`external_lib` 导入触发 Bridge 公共注册 API；provider、映射、HC/PLE/QSA/GDN 适配全部在 `verl/models/mcore/qwen3_8_next/`。

## 修改清单和作用范围

| ID | 归属 / 类型 | 触发与修复 | 在 verl 中的交付 | 作用范围 / 验证 |
| --- | --- | --- | --- | --- |
| V1 | vLLM / 正确性 bug | 模型状态在 weights 内存池内分配；level-2 sleep 后 weights 丢弃，而 refit 不恢复 PLE context offsets 等运行状态。改为 sleep-mode 下独立 runtime pool。 | `patches/vllm-model-state-sleep.patch` | 通用 `init_model_state` 路径，不仅 Qwen。无 sleep 时仍用 `nullcontext`；需上游补充其他模型 / runner 回归。已有 6 项真实 CuMem 生命周期测试、TP8 全模型固定调度恢复实验。 |
| V2 | vLLM / 数值对齐优化 | HC norm、down、SiLU、up、residual 中间值过早写回 BF16。保留 FP32 后在模型边界转回 BF16。 | `patches/vllm-hc-fp32.patch` | Qwen4Exp 专属、默认关闭的 `VLLM_QWEN4_EXP_HC_FP32`；verl recipe 开启。含 HC 算子测试。不能描述成所有 BF16 实现都错误。 |
| V3 | vLLM / 数值对齐优化 | GDN causal convolution 的乘法中间舍入与训练侧不同。Qwen 子类在 convolution 路径保留 FP32 产品。 | `patches/vllm-gdn-conv-fp32.patch` | `VLLM_QWEN4_EXP_GDN_CONV_FP32` 默认关闭；包含 prefill / decode 卷积测试。不改通用 GDN 默认行为。 |
| V4 | vLLM / Torch 2.10 构建兼容 | 主线 stable API 目标和依赖需要较新 Torch；本环境固定 2.10。使用上游 `use_existing_torch.py`，调整 build target 和依赖。Torch 2.10 stable `from_blob` 不能持有 deleter，UVA ownership 边界改用匹配该 ABI 的 ATen。 | `dependency_build/patch_torch210.py`、`dependency_build/patch_cuda_view_torch210.py` 和对应审计 patch | 编译产物仅面向 Python 3.13 / Torch 2.10 / cu131 / Linux x86_64 / SM90；不能宣传为通用 stable ABI wheel。保留 CPU tensor keepalive 与 cudaFreeHost，不能简单丢弃 deleter。不是 Qwen 数值 bug。 |
| B1 | Megatron-Bridge / 包元数据兼容 | v0.6.2 声明 Python `<3.13`，改为 `<3.14`，本地版本号 `0.6.2+nebula1`。 | `patches/bridge-python313.patch` | 仅 `pyproject.toml` / `package_info.py`；**无 Bridge 运行算法 bug patch**。元数据放宽不等于所有 Bridge 功能已在 3.13 验证。 |
| M1 | Megatron-Core / 正确性 bug | CPU hybrid optimizer 的 DP reshardable restore 把本地 nonpersistent `step` 覆盖到 checkpoint 状态，并未把恢复后的 master/moment tensor 同步回 CPU 子优化器。过滤 placeholder step 并重新同步。 | `verl/models/mcore/qwen3_8_next/compat.py` | 仅 Core 0.19.2，且实际变更针对 `HybridDeviceOptimizer`。导入插件时替换 `DistributedOptimizer` 的两个类方法，是进程级 patch；同进程其他该类实例也受版本 / optimizer 条件约束。已测恢复后的状态及下一次 update。 |
| M2 | Megatron / verl 模型实现 / 数值策略 | 融合 qkvzba 的 TE GEMM 形状改变低精度归约；拆为 qkvz、ba 两组继续使用 TE，配套映射与 shard checkpoint。固定 FP32 Q/K rsqrt；gated norm FP32。 | `ops/gdn_projection.py`、`gdn_mapping.py`、`ops/gated_delta_net.py` | 在 verl 的 Qwen 子类实现，未改 Core 或 TE 源码。PyTorch 的 BF16 reduced-reduction 开关不能控制 TE。固定 shape 对照、反向和恢复测试支撑，不据此宣称 TE 通用 bug。 |
| V5 | vLLM / verl worker extension / 数值策略 | 对 QSA 已选 block 排成统一顺序，使两侧 accumulation 顺序可比；不改变 top-k membership。 | `vllm_worker_extension.py`、`qsa_order.py` | 由配置显式选择的 worker extension 安装到 Qwen4Exp 函数，属于运行时修改外部行为；没有隐藏的 site-packages 编辑。现有 R3 配置同时回放 MoE 路由。 |

外部源码参考：[Megatron-Bridge #6123](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6123)、[Megatron-LM #7393](https://github.com/NVIDIA/Megatron-LM/pull/7393)、[Miles #2777](https://github.com/radixark/miles/pull/2777)、[SkyRL #2216](https://github.com/NovaSky-AI/SkyRL/pull/2216)。归属与许可证见 `THIRD_PARTY.md`。本清单没有声称这些 PR 的当前合并状态。

## 补丁应用顺序和构建

以下路径相对于仓库根；先克隆到独立构建目录，不修改原 Notebook 环境。只检查补丁可应用性时省略 `--apply`；工具校验精确 HEAD、要求干净工作树，先在临时源码树验证全部修改，再应用到指定 checkout。

```bash
PROFILE="$PWD/examples/grpo_trainer/qwen3_8_next"
BUILD_ROOT=/tmp/qwen38-build
mkdir -p "$BUILD_ROOT"
git clone https://github.com/vllm-project/vllm.git "$BUILD_ROOT/vllm"
git -C "$BUILD_ROOT/vllm" checkout 6e517b15c1833cf72a7f557ee32524d98682e617
python "$PROFILE/dependency_build/prepare_sources.py" vllm "$BUILD_ROOT/vllm" --apply

git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "$BUILD_ROOT/bridge"
git -C "$BUILD_ROOT/bridge" checkout c0e164ed2aedac4ad1c877780e2564a19d5d54ec
python "$PROFILE/dependency_build/prepare_sources.py" bridge "$BUILD_ROOT/bridge" --apply
```

vLLM 顺序：Torch 2.10 依赖 / CMake profile → UVA ownership → HC FP32 → GDN convolution FP32 → model-state sleep。`torch210-compatibility.patch` 和 `torch210-uva-ownership.patch` 是原始构建审计快照，应用工具使用对应脚本；**不要再重复应用审计快照**。早期名为 `megatron-gdn-*.patch` 的实验 diff 只修改 verl 子类，已合入适配源码，因此移入本地实验归档，不当作 Core 安装补丁。

用 uv 创建独立 Python 3.13 venv，安装目标 Torch 2.10.0/cu131 和 native build requirements；已有 Notebook Torch 可通过独立、带 system-site-packages 的 venv 读取，但所有安装均指向 venv。准备与上述 vLLM commit 的 CMake 固定 revision 一致的第三方源码，离线放入 vendor 目录；`manifest.json` 至少记录 `vllm_commit`，实际构建归档还应保存每个 vendor revision / SHA。不要让 CMake 在离线节点上隐式取不同版本。

```bash
# CC/CXX/CUDAHOSTCXX 由目标构建镜像提供；已验证环境使用 GCC 13。
python "$PROFILE/dependency_build/build_vllm.py" --build-root "$BUILD_ROOT" \
  --source "$BUILD_ROOT/vllm" --python "$BUILD_ROOT/build-env/bin/python" \
  --vendor "$BUILD_ROOT/vendor"
(cd "$BUILD_ROOT/bridge" && NO_VCS_VERSION=1 uv build \
  --python "$BUILD_ROOT/build-env/bin/python" --wheel --no-build-isolation \
  --out-dir "$BUILD_ROOT/dist")
```

`build_vllm.py` 要求 Linux、指定 ABI、独立 venv、构建目录中的源码，以及 prepare 阶段的源码哈希。新构建仍需执行 native operator / HC / GDN / CuMem 测试和全模型 smoke；**本次整理验证的是补丁应用，不是重新编译并重跑全部 CUDA 测试**。

历史发布流程先编译 native wheel，再以 `dependency_build/package_hc_fp32_wheel.py` 注入 Python/Triton 变更并重写 wheel RECORD，且严格核验基础 wheel 和每个修改文件哈希。最终 wheel 的 11 个 CUDA binary 与基础 wheel 一致。完整重建时直接从所有补丁应用后的源码构建即可；ZIP 时间戳等会影响 wheel SHA，不能要求新包字节哈希相同。

| 构件 | 已验证版本 / SHA-256 |
| --- | --- |
| vLLM base | `6e517b15c1833cf72a7f557ee32524d98682e617` |
| 最终 vLLM wheel | `0.30.1.dev0+g6e517b15.torch210.cu131.hcgdnfp32.plestate` · `ef4d559b743df5e0ae02251cdc500601cd839f69a366badd998e68950e8a279d` |
| Bridge wheel | `0.6.2+nebula1` · `1f0814b7bb68fd43f40af89664d99eb3ffdfd3535404b8880d815a08e8008c61` |
| 主要运行依赖 | Core 0.19.2、Transformers 5.16.1、FLA 0.5.2、FA 2.8.3、Ray 2.48.0 |

TE 是既有目标镜像提供的 ABI 匹配依赖，本次没有新增 TE 源码 patch。历史其他任务的 TE CP patch 不属于本次 CP1 长跑新增变更。FlashQLA 候选需要额外依赖，其构建不属于默认 100 步验证组合。

## 给上游的拆分建议与证据

这是一份待人工审阅的拆分清单，未提交 issue 或 PR。正式提交前要针对当时上游 HEAD 查重、确认是否仍受影响，按目标社区要求补充最小复现与 CI；不能把整个历史补丁组合一次性推成“精度 bug”。

1. **vLLM model-state sleep correctness**：优先单独评审 V1。最小复现为非 weight 状态在 weights pool 内分配，sleep(2) / wake 后 refit 仅恢复权重。补丁内包含 `test_cumem.py` 与 `test_sleep_mode_backend.py` 回归；`validate_vllm_model_state_sleep.py` 提供真实 GPU 生命周期验证。固定调度的 TP8 实验：32 requests、每阶段 4,096 token logprobs、6 阶段逐位一致，所有 rank 的 offsets / model tensors 保留。该结果是同一 vLLM 的生命周期等价性，不是 Megatron/vLLM 逐位一致。
2. **Megatron-Core CPU optimizer resume**：单独评审 M1。重点复现 Adam step、master/moment state 和下一次 update，而不只比较加载后的模型参数。当前实现是 verl shim；上游应修到 `load_parameter_state_from_dp_reshardable` / `_set_main_param_and_optimizer_states` 合适位置，并保留 GPU-only optimizer 路径行为。回归入口在 `test_qwen38_next_full_parameter_parallel.py`，8 rank TP2/PP2/EP2/DP2 曾全部通过。
3. **vLLM HC / GDN numerical precision**：V2、V3 分别提交数值证据，保留开关，并增加速度、显存、不同 batch / prefill / decode 的对照。它们是有收益和成本的精度策略，不能仅凭跨框架差异判定上游算法错误。
4. **Bridge Python 3.13 packaging**：B1 走兼容性支持，附 Python 3.13 导入、转换、训练证据；本地 `+nebula1` 版本号通常不应原样进入上游。
5. **Torch 2.10 ABI 回移**：V4 服务固定内部环境；先确认 vLLM 是否愿意维护该旧 ABI，再决定反馈形式。UVA 所有权测试具有通用价值，但不能用去掉 deleter 的方式“修编译”。

用户提到的 vLLM #59945 是另一项 head_dim fallback 问题。2026-10-04 的本地审计表明本 checkpoint 显式设置 `head_dim=256`，不触发该 fallback；它不是本次误差根因，未为本实验新增此补丁。此结论只针对当时固定源码和 checkpoint 条件。
