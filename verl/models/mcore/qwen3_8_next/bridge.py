# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Opt-in registration: ``actor_rollout_ref.model.external_lib=...bridge``.

Use the native Transformers Qwen4Exp configuration.
"""

from copy import copy
from itertools import chain

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ColumnParallelMapping, ReplicatedMapping
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed, persistent_buffers, unwrap_model
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge, _apply_qwen35_moe_config
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import _get_vision_mappings
from megatron.core import parallel_state
from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig  # noqa: F401

from .compat import install_cpu_optimizer_resume_fix
from .config import apply_flash_next_config, validate_rollout
from .gdn_mapping import Qwen38NextGDNProjectionMapping
from .param_mapping import PLENGramEmbeddingMapping, _PLEShardExport, ple_local_rows_from_shards
from .provider import Qwen38NextModelProvider

install_cpu_optimizer_resume_fix()


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
    @staticmethod
    def _cast_export_weight_dtype(weights, weight_dtype):
        # Bridge 0.6.2 casts dict values eagerly. Preserve the streaming table
        # view so full-parameter sync holds only one HF shard at a time.
        if isinstance(weights, _PLEShardExport):
            return weights.with_dtype(weight_dtype)
        return MegatronModelBridge._cast_export_weight_dtype(weights, weight_dtype)

    def build_conversion_tasks(self, hf_pretrained, megatron_model, **kwargs):
        models = unwrap_model(megatron_model if isinstance(megatron_model, list) else [megatron_model])
        # Forward Bridge's export options (including weight_dtype) so its
        # ordinary conversion stream and the full-parameter actor sync agree.
        tasks = super().build_conversion_tasks(hf_pretrained, models, **kwargs)
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
        # Native plans contain only this EP rank's experts. Validate the source
        # catalog against all expert owners rather than rejecting remote
        # experts as unused (unpacked HF checkpoints expose this distinction).
        if parallel_state.get_expert_model_parallel_world_size() > 1:
            group = parallel_state.get_expert_model_parallel_group()
            expert_sources = [None] * torch.distributed.get_world_size(group)
            torch.distributed.all_gather_object(expert_sources, used_sources, group=group)
            used_sources = set().union(*expert_sources)
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
        common_text = copy(text)
        # Bridge's Qwen3.5 translator only recognizes full/linear attention.
        # Reuse its common dimensions, then restore the explicit QSA layout.
        common_text.layer_types = ["full_attention" if t == "qwen_sparse_attention" else t for t in text.layer_types]
        _apply_qwen35_moe_config(provider, common_text)
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
        mappings = Qwen35MoEBridge._get_moe_lm_mappings(
            hf_prefix=hp,
            megatron_prefix=mp,
            experts_packed=moe_experts_stored_packed(self.hf_pretrained, hp + "layers."),
        )
        # HC owns normalization; these parameters are absent from this model.
        absent_norms = (
            "decoder.final_layernorm.weight",
            "pre_mlp_layernorm.weight",
            "linear_qkv.layer_norm_weight",
            "in_proj.layer_norm_weight",
        )
        mappings = [m for m in mappings if not m.megatron_param.endswith(absent_norms)]
        mappings = [m for m in mappings if not m.megatron_param.endswith("self_attention.in_proj.weight")]
        for part, sources in (("qkvz", ("qkv", "z")), ("ba", ("b", "a"))):
            mappings.append(
                Qwen38NextGDNProjectionMapping(
                    mp + f"decoder.layers.*.self_attention.in_proj.{part}.weight",
                    {source: hp + f"layers.*.linear_attn.in_proj_{source}.weight" for source in sources},
                )
            )
        # GDN output norm uses ordinary gamma in HF, vLLM and this provider.
        # Qwen3.5's +/-1 conversion is lossy for small BF16 training updates.
        mappings = [
            ReplicatedMapping(m.megatron_param, m.hf_param)
            if m.megatron_param.endswith("self_attention.out_norm.weight")
            else m
            for m in mappings
        ]
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
        hf_config = self.hf_pretrained.config if hasattr(self.hf_pretrained, "config") else self.hf_pretrained
        mappings.append(
            PLENGramEmbeddingMapping(
                megatron_param=(
                    mp + "decoder.layers.*.self_attention_hyper_connection.ple.ple_embedding.ngram_embedding.weight"
                ),
                hf_param=hp + "layers.*.ple.ple_embedding.ngram_embedding.shard_{}.weight",
                num_shards=hf_config.text_config.split_ngram_parts,
            )
        )
        # Immutable hash metadata is read from the original checkpoint. In the
        # frozen fallback there is no ngram_embedding parameter to match above.
        mappings.extend(_get_vision_mappings())
        return MegatronMappingRegistry(*mappings)

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        """Load only the PLE checkpoint shards owned by this tensor rank."""
        if isinstance(hf_param, dict) and hf_param and all(k.startswith("shard_") for k in hf_param):
            names = [hf_param[f"shard_{i}"] for i in range(len(hf_param))]
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            first = hf_state_dict[names[0]]
            rows = first.shape[0]
            # The embedding constructor validates uniform HF shard heights.
            # Retain just shard zero plus the slices owned by this rank.
            local = ple_local_rows_from_shards(
                lambda i: first if i == 0 else hf_state_dict[names[i]],
                len(names),
                rows * len(names),
                rows,
                tp_rank,
                tp_size,
            )
            return {"local_rows": local}
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
