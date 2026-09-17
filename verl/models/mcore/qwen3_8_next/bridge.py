# SPDX-License-Identifier: Apache-2.0
"""Opt-in registration: ``actor_rollout_ref.model.external_lib=...bridge``.

Use the real Qwen4Exp config from vLLM, not a Qwen3.5 model-type alias.
"""

from itertools import chain

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ColumnParallelMapping, ReplicatedMapping
from megatron.bridge.models.conversion.utils import persistent_buffers, unwrap_model
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
    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        models = unwrap_model(megatron_model if isinstance(megatron_model, list) else [megatron_model])
        tasks = super().build_conversion_tasks(hf_pretrained, models)
        self._validate_base_weight_plan(hf_pretrained, models, tasks)
        return tasks

    def stream_weights_hf_to_megatron(self, hf_pretrained, megatron_model, conversion_tasks=None):
        # A caller-supplied plan must not bypass the checks used by ordinary
        # loading. Validation reads keys only, before fetching any weight data.
        if getattr(getattr(hf_pretrained, "state", None), "source", None) is None:
            raise ValueError("Flash-Next base weight import requires a source catalog")
        if conversion_tasks is not None:
            models = unwrap_model(megatron_model if isinstance(megatron_model, list) else [megatron_model])
            self._validate_base_weight_plan(hf_pretrained, models, conversion_tasks)
        return super().stream_weights_hf_to_megatron(hf_pretrained, megatron_model, conversion_tasks)

    def _validate_base_weight_plan(self, hf_pretrained, models, tasks):
        """Reject silent skips in native Bridge planning, without reading payload.

        Native PP placeholders are valid; missing local owners are not. This is
        a coverage guard, not a transactional load or a tensor checksum. Actual
        shape conversion/copy checks remain in the native loader.
        """
        config = models[0].config
        tied = self._share_embeddings_and_output_weights(config)
        expected = {}
        for stage, model in enumerate(models):
            for name, weight in chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in name or self._is_adapter_param_name(name):
                    continue
                name = self._unwrap_name(name)
                if tied and "output_layer" in name:
                    continue  # Native Bridge broadcasts shared embeddings separately.
                expected[stage, name] = weight

        planned = {}
        used_sources = set()
        for task in tasks:
            if task is None:
                raise ValueError("Flash-Next base weight plan contains an unmapped target")
            names = task.mapping.hf_param
            used_sources.update([names] if isinstance(names, str) else names.values())
            if task.param_weight is None:
                continue  # A task owned by another PP stage, not a local skip.
            key = (task.vp_stage, task.param_name)
            if key in planned:
                raise ValueError(f"Flash-Next base weight plan duplicates local target {key}")
            if key not in expected or task.param_weight is not expected[key]:
                raise ValueError(f"Flash-Next base weight plan has an invalid local owner for {key}")
            planned[key] = task.param_weight
        missing_targets = expected.keys() - planned.keys()
        if missing_targets:
            raise ValueError(f"Flash-Next base weight plan misses local targets: {sorted(missing_targets)[:8]}")

        state = getattr(hf_pretrained, "state", None)
        source = getattr(state, "source", None)
        if source is None:
            return  # Config-only export has no source catalog to validate.
        keys = set(source.get_all_keys())
        # These tensors are intentionally absent from ordinary conversion, but
        # the direct PLE reader still requires every configured shard and hash.
        ple_sources = set()
        for layer in config.qwen3_8_next_ple_layer_ids:
            prefix = f"model.language_model.layers.{layer}.ple.ple_embedding."
            ple_sources.update(
                prefix + name for name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
            )
            ple_sources.update(
                prefix + f"ngram_embedding.shard_{shard}.weight"
                for shard in range(config.qwen3_8_next_split_ngram_parts)
            )
        missing_sources = (used_sources | ple_sources) - keys
        unused_sources = keys - used_sources - ple_sources
        unused_sources = {name for name in unused_sources if not name.startswith("mtp.")}
        if missing_sources or unused_sources:
            raise ValueError(
                "Flash-Next base weight coverage failed: "
                f"missing sources={sorted(missing_sources)[:8]}; unused sources={sorted(unused_sources)[:8]}"
            )

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
