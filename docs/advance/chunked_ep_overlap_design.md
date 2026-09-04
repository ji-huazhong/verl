# Full recompute 下的 Chunked EP Overlap：实现方案

状态：首版代码和 CPU/Gloo 数值测试已落地，CUDA/TE 与完整 PPO 验证待执行。
实际使用范围、验证命令和限制见 [使用说明](chunked_ep_overlap.md)，
VeOmni 的调研见 [独立报告](veomni_ep_overlap_investigation.md)。

基线：本地 verl `main`，commit `3af734b6145bc84903ed33f228ecab5e6c93cb00`。
开发分支：`hz/feat/chunked-ep-overlap`。
独立 worktree：`/Users/jhz/rl/verl-chunked-ep-overlap`。
原工作目录的 MTP 分支及未提交修改予以保留。

## 1. 结论

在 verl 内增加一个可选的、MoE 层内的 token chunk 调度器，继续使用 Megatron-Core 的模型、router、expert 参数、优化器及 full recompute。
用户入口使用独立的 `chunked_ep_overlap.enabled`，默认关闭。
启用时要求原生 `overlap_moe_expert_parallel_comm=False`；两者同时启用会明确报错。
保持 Core 原有开关及其校验、调度语义，不自动改写用户的原生 overlap 配置，也不移除禁止 full recompute 的上游断言。

第一阶段实现完整的 forward、recompute forward、backward 分块通信与计算重叠；第二阶段再研究 delayed wgrad 和 recompute/backward 融合。
后者不是第一阶段的正确性前提，也不能把博客包含这些优化的性能数字作为第一阶段承诺。

优先实现原生 `alltoall` / NCCL，避免新增强制依赖；后续在相同调度接口下接入 `flex/deepep`。
这是一项实现选择，不代表 alltoall 必然比 DeepEP 更快。基线和优化版本必须使用相同通信后端比较。

## 2. 调研依据

### 2.1 版本与现有冲突

本地 main 的 `pyproject.toml` 将 Megatron-Core 固定为 `core_v0.18.0`，Megatron-Bridge 为 `0.5.2`，legacy mbridge 为 `641a5a0`。
本次检查的 Core tag 对应 commit `ba7b5ebce12af60627a80985792a1449ce45f46c`。

Core 的 `TransformerConfig.__post_init__` 明确禁止同时设置 full recompute 和 `overlap_moe_expert_parallel_comm`，同时要求 recompute method/num_layers 为空。
该 overlap 开关进入 `combined_1f1b`，将不同 microbatch 的前向与反向细分后交错调度；并非仅切换 all-to-all 为异步。
因此移除断言不能构成实现。

另一个现有选项 `overlap_dispatch_backward_with_experts_wgrad` 针对 dispatch backward 与 expert wgrad 的重叠，不提供这里需要的 token chunk 前向和重计算调度。

### 2.2 博客与公开 mlite 源码

博客明确区分 chunk overlap 和进一步的 recompute/backward 融合。其单层 MoE proxy 不包含 attention、optimizer 或完整 PPO step。
博客的 forward trace 还指出：两个 dispatch 均可能发生在 expert 计算窗口前，可见收益来自前一块 combine 与后一块 GEMM 的重叠。因此验收不预设 dispatch 一定被隐藏。

检查了本地 `origin/dev` 的 mlite dispatcher 与 Qwen3.5 模型，并联网读取了公开 dev 分支相同文件。
dispatcher 有 DeepEP `submit/finish` 接口，但所检查的 Qwen3.5 forward 仍是整块 dispatch → experts → combine；autograd 通信封装内还会插入 stream wait。
这些代码可以参考通信接口和梯度语义，不能直接视为博客完整 chunk 训练调度器。本文不据此断言整个公开仓库不存在其他实验实现。

### 2.3 verl 的集成位置

- `verl/workers/config/engine.py`：为 `McoreEngineConfig` 增加 typed `chunked_ep_overlap` 配置。
- `verl/trainer/config/engine/megatron.yaml`：声明独立配置入口，默认关闭。
- `verl/workers/engine/megatron/transformer_impl.py`：检查开关冲突和最终配置、模型结构，向两个 bridge 传递安装回调。
- `verl/utils/megatron_utils.py::make_megatron_module`：已有 provider pre-wrap hook 和 legacy mbridge post-creation callback，可在 DDP 包装前安装。
- `verl/models/mcore/patch.py::apply_patch_megatron_recomputation_backward`：有手动 `untyped_storage().resize_(0)`，需要审计跨 stream 生命周期。
- `verl/utils/megatron/router_replay_patch.py`：保留现有 router replay 接口与调用粒度。

## 3. 核心设计

### 3.1 在 router 之后切分

一次 MoE 调用执行一次完整 router，再同时切分 hidden states、routing map 和 routing probabilities。
维持原输入的序列/批次形状直到 routing 完成，之后按展平 token 维切分。

这样保留 sequence aux loss、global aux loss、expert bias 更新、padding mask 和 R2/R3 replay 的原有粒度。
全重算时 router 的调用次数和副作用与未启用 chunk 的基线一致，而不是每个 chunk 重做一次路由。
反向需要同时返回 hidden-state gradient 和 routing-probability gradient，不能只验证 expert 权重或丢失 router 梯度。

shared expert 先继续完整执行，并在最后与 routed output 相加；首版要求关闭原生 `moe_shared_expert_overlap`，防止复用其内部状态与流依赖。

### 3.2 独立的 invocation / chunk 状态

现有 dispatcher 会在实例上保存 hidden shape、split sizes、permutation map、probabilities、DeepEP handle 等。
不能让多个 in-flight chunk 共用这些可变字段，也不能让新 microbatch 覆盖旧 microbatch backward 的状态。

每次调用创建一个 invocation state，每个 chunk 独立持有 dispatcher 状态、输入输出引用、局部 autograd 图和完成 event。
只复用无调用状态的配置、进程组和 expert 参数。不得 shallow-copy 正在使用的 dispatcher 或共享其 comm-manager 可变字段。

安装采用实例级行为适配，保留原 MoELayer 及其参数对象、名称、state_dict 和权重导出路径，不全局替换所有 MoELayer.forward。
actor/ref 可以各自启用。资源采用惰性初始化，以免 pre-wrap 阶段设备尚未就绪。

### 3.3 前向调度

两个 chunk 的目标依赖如下，D 为 dispatch，E 为 expert 计算，C 为 combine：

```text
每块内部：D0 → E0 → C0        D1 → E1 → C1
通信流序：D0 → D1 → C0 → C1
计算流序：E0 → E1
```

每个计算节点只等待自己输入的 event；不能使用一个全局 event 将所有 chunk 再次串起来。
例如 E1 等待 D1，而不是等待 C0，因此 C0 有机会与 E1 重叠。
输入生产、跨流消费、返回调用方都有明确 event；同一 communicator 的 collective 提交顺序在所有 rank 一致。

alltoall 的 token-count 交换和 D2H split metadata 尽量在计算窗口前准备，可合并各 chunk 的计数交换；不能在每个 GEMM 前引入 device-wide synchronize。
优先复用原生 permute/unpermute 和 expert 计算语义，异步通信封装负责 submit/wait 分离与反向的逆向 split。
使用固定的小窗口限制通信临时 buffer 数量；chunk 数增加不意味着预先保存所有临时 tensor。

### 3.4 显式 backward 调度

只写 Python chunk 循环或指定 CUDA stream，不能保证 autograd 会按照期望的跨 chunk 顺序执行。
采用 routed-MoE 区域的自定义 autograd 边界，在重计算产生梯度图时记录可单独执行的局部节点。
参考 Core `ScheduleNode` 的分阶段思想，但不复用其单 event 串联和主动释放输入 storage 的策略，也不依赖整个 combined_1f1b schedule。

```text
每块内部：combine-bwd → expert-bwd → dispatch-bwd
通信流序：CB0 → CB1 → DB0 → DB1
计算流序：EB0 → EB1
```

expert-bwd 首版包含 dgrad 和 wgrad，复用 TE / Core autograd 计算；DB0 可以和 EB1 重叠。
调度器显式消费每个 chunk 的输出梯度，最后拼回 hidden/probability 梯度，交还外层 router 与 transformer backward。
所有通信逆运算、概率梯度、参数梯度均需独立验证。

**参数梯度与 DDP：** 局部 backward 可能多次触发同一 expert 参数的梯度累积 hook。
首版显式要求 `override_ddp_config.overlap_grad_reduce=False`，完成所有 chunk 梯度累积后由原训练调度统一同步。
这不禁用 distributed optimizer。仍需验证 TE `main_grad`、普通 `.grad`、多 microbatch 和 loss scaling 的累积语义。
遇到用户配置 `overlap_grad_reduce=True` 时给出明确配置错误，不静默覆盖。
后续支持 bucket ready 在整层完成后只报告一次，再恢复梯度归约 overlap。

### 3.5 Full recompute 的生命周期

- 原 forward（外层 checkpoint 的 no_grad 阶段）：执行分块流水线，调用结束即释放本次临时状态，不保存内部 autograd 图。
- recompute forward：重新创建本次 chunk 状态并构建所需局部梯度图。
- backward：按显式调度消费这些图和通信状态；不能额外把整个 MoE forward 再运行一次。
- 返回外层 checkpoint 前，所有侧流对输入及梯度的访问都必须完成依赖衔接，并正确记录 allocator stream 使用。

首版保留 Core `recompute.py::checkpointed_forward` 的 uniform / block 语义，包括 uniform 跨多个层。
`full` 是重算粒度；`block` 是否覆盖全部层仍由 `recompute_num_layers` 决定。
对没有进入 checkpoint 的 block 剩余层，也需要正确支持常规有梯度 forward。

重点审计 verl 当前 checkpoint backward 的 storage resize：不能释放仍被 chunk graph、view 或通信持有的存储。
优先在新调度器内保证生命周期闭合；若现有补丁仍不安全，只为启用该能力的 checkpoint 上下文提供范围明确的延迟释放策略。
不全局删除既有内存回收行为。

### 3.6 不均衡 token 数与回退

EP rank 可以有不同 token 数，不能各自依据本地长度决定调用不同数量的 collective。
以配置的固定 chunk 数为基础；短输入回退决策在通信组内达成一致，尽量复用 microbatch metadata，避免逐层额外同步。
需要覆盖尾块、零发送/零接收 expert 和 rank 间长度不同的场景。
如果某通信后端不支持空 chunk，则整个通信组共同回退原路径，不能单 rank 跳过。

`num_chunks=1`、组内一致的短序列回退以及禁用配置都调用原生 MoE 路径。
回退仅用于输入规模；不兼容配置应在启动时明确报错。

## 4. 配置草案与支持范围

新字段属于 verl engine，不透传成未知的 Core TransformerConfig 参数。
既检查用户 overrides，也检查模型/provider 合并后的实际配置。启用时必须验证 chunk 调度器确已安装到目标 MoE 层；模型结构不支持应明确报错。

```yaml
actor_rollout_ref:
  actor:
    megatron:
      tensor_model_parallel_size: 1
      expert_tensor_parallel_size: 1
      expert_model_parallel_size: 8
      chunked_ep_overlap:
        enabled: true
        num_chunks: 2
        min_tokens_per_chunk: 4096
      override_ddp_config:
        overlap_grad_reduce: false
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
        overlap_moe_expert_parallel_comm: false
        overlap_dispatch_backward_with_experts_wgrad: false
        delay_wgrad_compute: false
        moe_token_dispatcher_type: alltoall
        moe_shared_expert_overlap: false
```

上面的 2 chunks / 4096 tokens 是待 benchmark 的起点，全局默认 `enabled: false`。
启用后执行本方案的兼容性校验；禁用后保持原有行为，原生 overlap 继续由 Core 校验。

首版验收范围：CUDA、BF16、Core 0.18 的标准 MoELayer + TE grouped experts、EP>1、TP=ETP=1、dropless routing、普通 Megatron DDP/distributed optimizer。
先通过 PP=1，然后验证 PP=2、静态 CP>1、THD/BSHD、R2/R3；只有验证通过的组合进入支持列表。
forward-only 的 actor/ref 路径也要验证，因为 PPO 既计算 log-prob 又执行训练。

以下暂不宣称支持，启用时应给出具体原因：TP/ETP>1、Dynamic CP、MTP、FP8/FP4/QAT、MoE CUDA Graph、Megatron-FSDP、参数/activation 细粒度 offload、capacity padding/drop、latent/custom MoE、delay-wgrad 组合和未经验证的 PEFT expert 包装。
它们会影响通信域、buffer/参数生命周期或梯度调度。通过功能探测和配置校验限制新路径，保持禁用时的原行为。

## 5. 实施拆分

1. **配置与接入**：新增 typed chunk 配置、YAML 与生成配置；给 `make_megatron_module` 增加明确的 callback 接口，分别接入两个 bridge；完成模型结构和配置校验。
2. **原生分块数值路径**：建立 invocation/chunk 状态，完整 router 后切分；先用串行 chunk 确认 output、input/router/expert/shared-expert gradients、梯度累积与 full recompute 等价。
3. **前后向异步流水线**：实现独立的 NCCL submit/wait、CUDA events、自定义 autograd 调度、组内一致的 chunk policy、buffer 回收和 NVTX 标记。阶段结束必须实测 backward overlap，不能只交付 no_grad forward 优化。
4. **verl 集成验证与调优**：接入 Qwen3/Qwen3.5，跑 PP/CP/replay、checkpoint reload、权重导出和 PPO actor update；benchmark 确定推荐 chunk 数和回退阈值。
5. **后续优化**：DeepEP adapter、DDP grad-reduce overlap、TP/ETP 支持，再研究 delayed wgrad 与 recompute/backward fusion。

首版文件划分（通信与调度合并为一个模块）：

```text
verl/utils/megatron/chunked_ep_overlap.py       # 校验、安装、独立 chunk 状态、NCCL、autograd
tests/utils/test_chunked_ep_overlap_on_cpu.py   # 两进程 Gloo 数值和重算测试
tests/workers/config/test_engine_config_on_cpu.py
tests/trainer/test_constants_ppo_on_cpu.py
tests/special_distributed/test_chunked_ep_overlap.py
benchmarks/megatron/benchmark_chunked_ep_overlap.py
docs/advance/chunked_ep_overlap.md             # 经验证的用法和限制
```

保留单层接口边界，避免复制整个 MoELayer / TransformerLayer / PPO worker 实现。
具体模块划分可在实现中按代码量收敛。

## 6. 验证与验收

### 正确性

- 三组 paired baseline：原生不分块、串行分块、异步分块；相同参数、输入、RNG、路由、通信后端和并行配置。
- 覆盖独立开关的启用、禁用和原生 overlap 冲突，以及两个 bridge 的配置构建路径；验证禁用时行为不变、chunk full 不进入 combined_1f1b、安装失败明确报错。
- 比较 output、loss、hidden gradient、router gradient、每个 expert 的 gradient、shared-expert gradient，以及 optimizer step 后的参数；BF16 设置合理误差预算，不要求因 GEMM 形状变化产生的结果逐位一致。
- 分别测试 no recompute、full/uniform 的 1 层与多层、full/block、多个 microbatch 累积和连续多个训练 step。
- 覆盖不均衡路由、空 expert、空 split、尾块、packed 输入、padding、不同 EP rank token 数。
- EP=2/4，并至少包含 expert-data-parallel size>1 的组合，验证实际梯度归约与 distributed optimizer，而非只验证单个 EP 组。
- PP=2、静态 CP=2、router replay R2/R3、forward-only；state_dict keys/shape、保存恢复和 HF 权重导出保持一致。
- CPU/Gloo 测试验证配置、chunk policy 及分布式 autograd 数值；不能替代 CUDA/TE/NCCL 测试。

### 性能与内存

- 单层 MoE：local tokens 4K/8K/16K/32K/64K，chunks 1/2/4；同时计入 forward、recompute 和 backward。
- 训练集成：同一 Qwen3.5 actor microbatch 和 token budget，对比 actor update 时间及峰值 allocated/reserved memory，另报端到端 PPO 时间。
- 多次 warmup 后多次采样，使用最慢 rank 的 step 时间，记录 GPU、互联、软件版本、并行配置、microbatch 和 recompute 参数。
- Nsight Systems / NVTX 必须看到不同 chunk 的 A2A 与 expert GEMM 有真实时间交集；单有 `async_op=True` 或多个 stream 不算通过。
- 连续多 step 检查峰值/驻留显存及状态容器无增长，定位 storage resize、异步 use-after-free 和 collective deadlock。
- 与相同 `overlap_grad_reduce=False` 的原生基线比较，以隔离 chunk 本身收益；另与用户原训练配置比较，确认总收益是否覆盖关闭 grad-reduce overlap 的代价。

预期是长序列有更多 overlap 窗口，但不能预先保证吞吐增幅或显存减半。
首版普通重计算仍要保留其 backward 所需 activation；仅限制通信临时 buffer 不足以复现博客所有显存收益。
当前为 Mac 本地调研，尚无 CUDA 多卡验证结果。

## 7. 后续 fusion 为什么单独做

博客更深的优化会让 backward 直接消费重算的 expert 中间结果，减少 recompute 的部分 fc2/combine，并把 delayed wgrad 放在合适位置。
在通用 Core full checkpoint 中，某层输出可能仍需用于 residual、后续层、同一 uniform checkpoint 中的下一层或 MTP；不能直接在 `is_recompute` 分支删掉这些运算。
只有明确重构 checkpoint 边界和数据依赖后才能消除它们。

因此后续 fusion 需要独立设计和数值测试；首版先保持完整 forward 语义，实现兼容 full recompute 的 chunk overlap。

## 8. 来源

- [用户提供的博客，§2.4](https://iseekyan.github.io/zh/posts/qwen35-long-sequence-moe-rl/#24-%E9%80%9A%E4%BF%A1%E7%BA%BFchunked-ep-overlap)
- [Core 0.18 TransformerConfig 的 overlap 校验](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/megatron/core/transformer/transformer_config.py#L2399)
- [Core combined_1f1b](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/megatron/core/pipeline_parallel/combined_1f1b.py)
- [Core MoELayer](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/megatron/core/transformer/moe/moe_layer.py)
- [Core token dispatcher](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/megatron/core/transformer/moe/token_dispatcher.py)
- [Core DeepEP autograd 封装](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/megatron/core/transformer/moe/fused_a2a.py)
- [Core recompute](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/megatron/core/recompute.py)
- [Core DDP backward hook](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/megatron/core/distributed/distributed_data_parallel.py#L449)
- [所检查的 mlite dispatcher](https://github.com/NVIDIA/Megatron-LM/blob/dev/experimental/lite/megatron/lite/primitive/modules/dispatcher.py)
- [所检查的 mlite Qwen3.5 模型](https://github.com/NVIDIA/Megatron-LM/blob/dev/experimental/lite/megatron/lite/model/qwen3_5/lite/model.py)
