# VeOmni EP all-to-all overlap 调研

检查日期：2026-09-04。基线为本地 verl main `3af734b6`。

## 结论

本地 verl 的锁文件固定 `veomni==0.1.11`。该版本支持普通 expert parallel all-to-all，
但检查到的 MoE 路径没有实现 EP all-to-all 与 expert 计算重叠，也没有 chunked EP overlap 配置入口。
`moe_implementation=fused` 选择 fused/grouped expert kernel，不代表通信与计算 overlap。

这不表示 gradient checkpointing 与普通 EP 不兼容：verl 将
`model.enable_gradient_checkpointing` 和 `engine.enable_reentrant` 传给 VeOmni 的模型并行化入口。
但没有可通过配置启用的「全重计算 + EP overlap」调度器。

## 证据

1. `pyproject.toml` 的 veomni-sft extra 和 `uv.lock` 固定版本为 0.1.11。
   本次直接解包检查锁文件中的 PyPI wheel，没有安装或修改 VeOmni。
2. `veomni/ops/kernels/moe/group_gemm.py::group_gemm_fused_moe_forward` 的 EP 分支顺序为：
   `preprocess → token_pre_all2all → EPGroupGemm/EPMergedFc1GroupGemm → tokens_post_all2all`。
3. `veomni/distributed/moe/moe_layer.py` 的前后两个通信函数均调用 `all_to_all`。
   它们使用相同输入块完成 dispatch、expert GEMM 和 combine，没有跨 chunk 或 microbatch 的调度。
4. `veomni/distributed/moe/comm.py` 定义了 `all_to_all_async`，但对整个 0.1.11 包的搜索
   只找到其定义，没有 MoE 调用点。存在异步原语不能证明执行路径有 overlap。
5. 0.1.11 包内未找到 DeepEP 实现或依赖调用。verl 的 VeOmni engine 配置也没有
   Megatron 风格的 `overlap_moe_expert_parallel_comm` 或 chunk 配置。
6. 另外联网检查了 VeOmni 当前公开 main 的 `distributed/moe/moe_layer.py` 和 `comm.py`。
   新的 `dispatch_to_ep_class` 仍按 `token_pre_all2all → ep_class.apply → tokens_post_all2all` 调用；
   通信仍使用普通 `all_to_all`。这验证了该公开路径，没有据此声称排除所有实验分支或私有实现。

因此当前不能通过打开 `fused` 或普通 gradient checkpointing 得到所需 overlap。
若扩展到 VeOmni，需要另外实现 MoE 层内调度，并验证 FSDP 参数 all-gather/reshard、
梯度归约和 checkpoint 与分块反向的组合，不能直接套用本次 Megatron DDP 的参数生命周期假设。

## 来源

- [VeOmni 0.1.11 发布包](https://pypi.org/project/veomni/0.1.11/)
- [检查的 main MoE 实现](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/distributed/moe/moe_layer.py)
- [检查的 main 通信原语](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/distributed/moe/comm.py)
- verl：`verl/workers/engine/veomni/transformer_impl.py::_build_model_optimizer`、
  `verl/workers/config/engine.py::VeOmniEngineConfig`、`verl/trainer/config/engine/veomni.yaml`。
