# SPDX-License-Identifier: Apache-2.0
"""Flash-Next configuration translation and explicit implementation boundaries."""


def apply_flash_next_config(provider, text_config, checkpoint):
    """Translate architecture fields without substituting Qwen3.5 defaults."""
    provider.num_residual_streams = text_config.hc_count
    for name in (
        "hc_lowrank",
        "ple_embed_dim",
        "ngram_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "split_ngram_parts",
        "ple_conv_kernel_size",
        "indexer_budget",
        "indexer_compress_ratio",
        "indexer_n_heads",
        "indexer_head_dim",
        "indexer_kv_heads",
        "eos_token_id",
    ):
        value = getattr(text_config, name, None)
        if value is None:
            raise ValueError(f"Flash-Next checkpoint is missing required field {name}")
        setattr(provider, f"qwen3_8_next_{name}", value)
    provider.qwen3_8_next_ple_conv_dilation = getattr(text_config, "ple_conv_dilation", None) or text_config.ngram_size
    ple_ids = [int(i) - 1 for i in text_config.ple_layer_ids]
    if len(set(ple_ids)) != len(ple_ids) or any(i < 0 or i >= provider.num_layers for i in ple_ids):
        raise ValueError("PLE layer ids must be unique, 1-based, and within the decoder")
    provider.qwen3_8_next_ple_layer_ids = sorted(ple_ids)
    layer_types = list(text_config.layer_types)
    if len(layer_types) != provider.num_layers or not set(layer_types) <= {"linear_attention", "full_attention"}:
        raise ValueError("Flash-Next requires an explicit GDN/QSA type for every layer")
    provider.qwen3_8_next_layer_types = layer_types
    # Core's list form uses 1 for GDN, 0 for standard attention.
    provider.linear_attention_freq = [int(t == "linear_attention") for t in layer_types]
    provider.qwen3_8_next_hf_checkpoint = str(checkpoint)
    if getattr(text_config, "output_gate_type", "sigmoid") != "sigmoid":
        raise ValueError("Only sigmoid attention output gates have been implemented")
    provider.qwen3_8_next_output_gate_type = "sigmoid"
    # GRPO uses the main autoregressive policy, not the auxiliary MTP heads.
    provider.mtp_num_layers = None


def validate_runtime(config):
    """Fail before allocating a model for unsupported execution contracts.

    PP uses Core's dynamic P2P shape exchange for the wide HC residual stream.
    Fixed-shape PP and unvalidated combinations of CP with other model-parallel
    axes remain guarded. Packed CP requires the per-token gradient contract.
    """
    if (getattr(config, "pipeline_model_parallel_size", 1) or 1) > 1 and not getattr(
        config, "variable_seq_lengths", False
    ):
        raise NotImplementedError("Flash-Next PP requires variable_seq_lengths=True for HC P2P shapes")
    cp_size = getattr(config, "context_parallel_size", 1) or 1
    if cp_size != 1:
        if cp_size != 2:
            raise NotImplementedError("Flash-Next model validation currently covers context_parallel_size=1 or 2")
        if any(
            (getattr(config, name, 1) or 1) != 1
            for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "expert_model_parallel_size")
        ):
            raise NotImplementedError("Flash-Next CP2 combined TP/PP/EP execution still requires validation")
        if not getattr(config, "calculate_per_token_loss", False):
            raise ValueError("Flash-Next CP requires calculate_per_token_loss=True")
        for name in ("linear_num_key_heads", "linear_num_value_heads"):
            count = getattr(config, name, 0)
            if count < cp_size or count % cp_size:
                raise ValueError(f"Flash-Next CP requires {name} to be divisible by context_parallel_size")
    if (
        getattr(config, "virtual_pipeline_model_parallel_size", None)
        and (getattr(config, "pipeline_model_parallel_size", 1) or 1) < 2
    ):
        raise NotImplementedError("Flash-Next interleaved pipeline requires at least two physical stages")
    if getattr(config, "virtual_pipeline_model_parallel_size", None) and not getattr(config, "overlap_p2p_comm", False):
        raise NotImplementedError(
            "Flash-Next VPP requires overlap_p2p_comm=True; synchronous dynamic P2P backward is not validated"
        )
    if getattr(config, "mtp_num_layers", None):
        raise NotImplementedError("Flash-Next MTP is not part of the current policy implementation")
    if getattr(config, "cuda_graph_impl", "none") not in (None, "none"):
        raise NotImplementedError("Flash-Next HC/PLE CUDA graph capture is not validated")
    if getattr(config, "recompute_granularity", None) == "selective":
        raise NotImplementedError("Use full-layer recompute; selective QSA recompute loses its selection context")
    if getattr(config, "fp8", None) or getattr(config, "fp4", None):
        raise NotImplementedError("Flash-Next HC/PLE mixed-precision recompute is not implemented")
    if getattr(config, "cpu_offloading", False) or getattr(config, "fine_grained_activation_offloading", False):
        raise NotImplementedError("Flash-Next HC activation offloading is not implemented")
    if getattr(config, "num_residual_streams", 0) < 2:
        raise ValueError("Flash-Next requires multiple residual streams")
    if getattr(config, "qwen3_8_next_indexer_kv_heads", None) != 1:
        raise ValueError("QSA kernels require one indexer KV head")


def validate_rollout(hf_config, rollout_config):
    """The actor exporter omits immutable PLE storage; rollout must load it."""
    if getattr(hf_config, "model_type", None) != "qwen4_exp":
        return
    if not getattr(hf_config.text_config, "ple_layer_ids", None):
        return
    engine_options = getattr(rollout_config, "engine_kwargs", {}) or {}
    vllm_options = engine_options.get("vllm", {}) or {}
    formats = [getattr(rollout_config, "load_format", None)]
    formats.extend(vllm_options[key] for key in ("load_format", "load-format") if key in vllm_options)
    if any(value not in ("auto", "safetensors") for value in formats):
        raise ValueError(
            "Flash-Next's frozen PLE table is not exported by actor weight sync. "
            "Use rollout.load_format=auto or safetensors with the original complete "
            "base checkpoint; dummy loading and engine_kwargs overrides are unsafe."
        )
