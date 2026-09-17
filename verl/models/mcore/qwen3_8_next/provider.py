# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Megatron-Bridge provider, retaining the vision encoder and PEFT path."""

import copy
from dataclasses import dataclass

from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLMoEModelProvider
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.identity_op import IdentityOp

from .config import validate_runtime
from .hyper_connection import Qwen38NextHCHeadContraction
from .layer import Qwen38NextTransformerLayer
from .ops.attention import Qwen38NextAttention
from .ops.gated_delta_net import Qwen38NextGatedDeltaNet
from .ops.ple import (
    Qwen38NextFrozenNGramEmbedding,
    build_ngram_contexts_packed,
    clear_ple_batch,
    publish_ple_batch,
)


def build_flash_next_spec(config, vp_stage=None, pp_rank=None):
    validate_runtime(config)
    spec = get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=vp_stage, pp_rank=pp_rank)
    spec = copy.deepcopy(spec)
    for layer_spec in spec.layer_specs:
        layer_spec.module = Qwen38NextTransformerLayer
        sub = layer_spec.submodules
        sub.input_layernorm = IdentityOp
        sub.pre_mlp_layernorm = IdentityOp
        attention = sub.self_attention
        if hasattr(attention.submodules, "linear_qkv"):
            attention.module = Qwen38NextAttention
            attention.submodules.linear_qkv = TEColumnParallelLinear
        else:
            attention.module = Qwen38NextGatedDeltaNet
            attention.submodules.in_proj = TEColumnParallelLinear
        # Remove aliases for layernorms that do not exist in Flash-Next.
        sub.sharded_state_dict_keys_map = {}
    # Core 0.18 has no HC-head slot. Its final norm builder is the contraction
    # slot here; it adds no extra normalization or trainable norm weight.
    spec.layer_norm = Qwen38NextHCHeadContraction
    return spec


def install_ple_context_hooks(model):
    embeddings = [m for m in model.modules() if isinstance(m, Qwen38NextFrozenNGramEmbedding)]
    if not embeddings:
        return
    if len(embeddings) != 1:
        raise NotImplementedError("Multiple PLE layers need independently keyed n-gram contexts")
    embedding = embeddings[0]

    def pre_hook(_module, args, kwargs):
        clear_ple_batch()
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise ValueError("PLE requires the original token ids, including vision placeholder ids")
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise NotImplementedError("Flash-Next currently requires a single packed token stream per microbatch")
        packed = kwargs.get("packed_seq_params")
        cu = getattr(packed, "cu_seqlens_q", None)
        contexts = build_ngram_contexts_packed(ids.reshape(-1), cu, embedding.ngram_size, embedding.eos_token_id)
        publish_ple_batch(embedding.compute_ngram_ids(contexts), cu)

    def post_hook(_module, _args, _output):
        clear_ple_batch()

    # The VL wrapper removes padding and packs token ids before it calls the
    # language model. Hook there, not at the padded outer multimodal boundary.
    language_model = getattr(model, "language_model", model)
    language_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    language_model.register_forward_hook(post_hook, always_call=True)


@dataclass
class Qwen38NextModelProvider(Qwen35VLMoEModelProvider):
    num_residual_streams: int = 4

    def build_language_spec(self, vp_stage=None, pp_rank=None):
        return build_flash_next_spec(self, vp_stage, pp_rank)

    def build_mtp_spec(self, vp_stage=None):
        validate_runtime(self)
        return None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        validate_runtime(self)
        model = super().provide(pre_process, post_process, vp_stage)
        # Bridge passes vp_stage to the language decoder but drops it on the
        # outer VL wrapper. Core's schedule, PEFT and verl inspect this root
        # attribute to distinguish virtual chunks on the same physical rank.
        model.vp_stage = vp_stage
        install_ple_context_hooks(model)
        return model
