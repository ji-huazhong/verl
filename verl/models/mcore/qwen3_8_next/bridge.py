# SPDX-License-Identifier: Apache-2.0
"""Opt-in registration: ``actor_rollout_ref.model.external_lib=...bridge``.

Use the real Qwen4Exp config from vLLM, not a Qwen3.5 model-type alias.
"""

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ColumnParallelMapping, ReplicatedMapping
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge, _apply_qwen35_moe_config
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import _get_vision_mappings
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpTextConfig

from .config import apply_flash_next_config, validate_rollout
from .provider import Qwen38NextModelProvider

for config_cls in (Qwen4ExpConfig, Qwen4ExpTextConfig):
    if config_cls.model_type not in CONFIG_MAPPING:
        AutoConfig.register(config_cls.model_type, config_cls)


def validate_verl_rollout(model_config, rollout_config):
    """Model-plugin contract consumed by the rollout server before allocation."""
    validate_rollout(model_config.hf_config, rollout_config)


@MegatronModelBridge.register_bridge(
    source="Qwen4ExpForConditionalGeneration",
    target=Qwen3VLModel,
    provider=Qwen38NextModelProvider,
    model_type="qwen4_exp",
)
class Qwen38NextBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        config = hf_pretrained.config
        text = config.text_config
        kwargs = self.hf_config_to_provider_kwargs(text)
        kwargs["vision_config"] = config.vision_config
        provider = Qwen38NextModelProvider(**kwargs)
        _apply_qwen35_moe_config(provider, text)
        apply_flash_next_config(provider, text, hf_pretrained.model_name_or_path)
        provider.share_embeddings_and_output_weights = config.tie_word_embeddings
        provider.hf_text_config = text
        provider.head_dim = text.head_dim
        provider.bos_token_id = text.bos_token_id
        provider.eos_token_id = text.eos_token_id
        for name in ("vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id"):
            setattr(provider, name, getattr(config, name))
        rope = text.rope_parameters
        provider.position_embedding_type = "mrope"
        provider.mrope_section = list(rope["mrope_section"])
        provider.rotary_base = rope["rope_theta"]
        provider.rotary_percent = rope["partial_rotary_factor"]
        return provider

    def mapping_registry(self):
        mp, hp = "language_model.", "model.language_model."
        mappings = Qwen35MoEBridge._get_moe_lm_mappings(hf_prefix=hp, megatron_prefix=mp)
        # HC owns normalization; these parameters are absent from this model.
        absent_norms = (
            "decoder.final_layernorm.weight",
            "pre_mlp_layernorm.weight",
            "linear_qkv.layer_norm_weight",
            "in_proj.layer_norm_weight",
        )
        mappings = [m for m in mappings if not m.megatron_param.endswith(absent_norms)]
        # These head-wise vectors remain column-sharded in the GDN subclass.
        # AutoMapping dispatches by concrete class name, not inheritance; encode
        # their actual layout explicitly instead of registering a global alias.
        mappings = [
            ColumnParallelMapping(m.megatron_param, m.hf_param)
            if m.megatron_param.endswith(("self_attention.A_log", "self_attention.dt_bias"))
            else m
            for m in mappings
        ]

        def replicated(mcore, hf):
            mappings.append(ReplicatedMapping(megatron_param=mp + mcore, hf_param=hp + hf))

        hc_fields = {
            "hc_norm_weight": "hc_norm.weight",
            "input_mix_weight_down": "input_mix_weight_down.weight",
            "input_mix_weight_up": "input_mix_weight_up.weight",
            "block_inject_weight": "block_inject_weight.weight",
        }
        for mcore_name, hf_name in (
            ("self_attention_hyper_connection", "attn_hyper_connection"),
            ("mlp_hyper_connection", "mlp_hyper_connection"),
        ):
            for mfield, hfield in hc_fields.items():
                replicated(f"decoder.layers.*.{mcore_name}.{mfield}", f"layers.*.{hf_name}.{hfield}")
        for mfield, hfield in hc_fields.items():
            if mfield != "block_inject_weight":
                replicated(f"decoder.final_layernorm.{mfield}", f"hyper_connection_mixer.{hfield}")
        for mfield, hfield in {
            "index_qk_proj.weight": "index_qk_proj.weight",
            "q_layernorm": "q_layernorm.weight",
            "k_layernorm": "k_layernorm.weight",
        }.items():
            replicated(f"decoder.layers.*.self_attention.indexer.{mfield}", f"layers.*.self_attn.indexer.{hfield}")
        for mfield, hfield in {
            "key_proj.weight": "key_proj.weight",
            "value_proj.weight": "value_proj.weight",
            "norm_key": "norm_key.weight",
            "norm_query": "norm_query.weight",
            "norm_conv": "norm_conv.weight",
            "conv1d_weight": "conv1d.weight",
        }.items():
            replicated(f"decoder.layers.*.self_attention_hyper_connection.ple.{mfield}", f"layers.*.ple.{hfield}")
        # Frozen PLE tables + hash metadata load directly from the same source
        # checkpoint on both actor and rollout. They are not trainable/resharded
        # model parameters. Rollout must load the base checkpoint, never dummy.
        mappings.extend(_get_vision_mappings())
        return MegatronMappingRegistry(*mappings)
