# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM integration for compressed-tensors integer INT4 WNA16 RL reload."""

import logging
import os
from collections.abc import Iterable, Iterator
from types import MethodType
from typing import Any

import torch

logger = logging.getLogger(__name__)

_FUSED_GATE_UP_SUFFIXES = (".mlp.experts.gate_up_proj.weight_packed", ".mlp.experts.gate_up_proj.weight_scale")
_FUSED_DOWN_SUFFIXES = (".mlp.experts.down_proj.weight_packed", ".mlp.experts.down_proj.weight_scale")
_WNA16_DERIVED_RELOAD_TENSORS = frozenset(
    {
        "w13_weight_shape",
        "w2_weight_shape",
        "w13_weight_g_idx",
        "w2_weight_g_idx",
        "w13_g_idx_sort_indices",
        "w2_g_idx_sort_indices",
    }
)


def _field(value: Any, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def is_int4_wna16_quant_config(quant_config: Any) -> bool:
    """Feature-detect compressed-tensors integer W4A16 without class coupling."""
    if quant_config is None or _field(quant_config, "quant_format") != "pack-quantized":
        return False

    scheme_map = _field(quant_config, "target_scheme_map", {}) or {}
    for scheme in scheme_map.values():
        weights = _field(scheme, "weights")
        inputs = _field(scheme, "input_activations")
        if weights is None:
            continue
        if (
            _field(weights, "num_bits") == 4
            and _enum_value(_field(weights, "type")) == "int"
            and _enum_value(_field(weights, "strategy")) in {"group", "channel"}
            and inputs is None
        ):
            return True
    return False


def expand_qwen3_5_fused_int4_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Expand compact fused Qwen3.5 INT4 tensors for vLLM's expert loader.

    Megatron-Bridge exports one 3D tensor per projection. vLLM 0.24 supports
    that fused alias only for unquantized weights, not suffixes such as
    ``weight_packed`` and ``weight_scale``. Expansion happens inside the vLLM
    process, after IPC, so the wire format stays compact.
    """
    for name, tensor in weights:
        gate_up_suffix = next((suffix for suffix in _FUSED_GATE_UP_SUFFIXES if name.endswith(suffix)), None)
        if gate_up_suffix is not None:
            if tensor.ndim != 3 or tensor.shape[1] % 2 != 0:
                raise ValueError(f"Invalid fused Qwen3.5 gate_up INT4 tensor {name}: shape={tuple(tensor.shape)}")
            gate, up = tensor.chunk(2, dim=1)
            leaf = gate_up_suffix.rsplit(".", 1)[-1]
            prefix = name[: -len(gate_up_suffix)]
            for expert_id, (gate_expert, up_expert) in enumerate(zip(gate.unbind(0), up.unbind(0), strict=True)):
                yield f"{prefix}.mlp.experts.{expert_id}.gate_proj.{leaf}", gate_expert
                yield f"{prefix}.mlp.experts.{expert_id}.up_proj.{leaf}", up_expert
            continue

        down_suffix = next((suffix for suffix in _FUSED_DOWN_SUFFIXES if name.endswith(suffix)), None)
        if down_suffix is not None:
            if tensor.ndim != 3:
                raise ValueError(f"Invalid fused Qwen3.5 down INT4 tensor {name}: shape={tuple(tensor.shape)}")
            leaf = down_suffix.rsplit(".", 1)[-1]
            prefix = name[: -len(down_suffix)]
            for expert_id, expert in enumerate(tensor.unbind(0)):
                yield f"{prefix}.mlp.experts.{expert_id}.down_proj.{leaf}", expert
            continue

        yield name, tensor


def patch_qwen3_5_fused_int4_loader(model: torch.nn.Module) -> bool:
    """Patch Qwen3.5's instance loader to accept compact fused INT4 updates."""
    if type(model).__name__ not in {"Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"}:
        return False
    if hasattr(model, "_verl_int4_original_load_weights"):
        return True

    model._verl_int4_original_load_weights = model.load_weights

    def _load_int4_weights(self, weights):
        return self._verl_int4_original_load_weights(expand_qwen3_5_fused_int4_weights(weights))

    model.load_weights = MethodType(_load_int4_weights, model)
    logger.info("Patched %s for compact fused integer INT4 expert updates", type(model).__name__)
    return True


def configure_int4_layerwise_reload(skip_tensors: set[str] | None = None) -> None:
    """Keep WNA16 metadata/indices resident across online weight reloads.

    These tensors are derived from static model geometry. Group/sort indices
    are absent from the actor's online stream, while per-expert shape tensors
    may be redundantly streamed many times into one fused parameter. None of
    them should gate reload completion. Counting them as reloadable prevents
    vLLM from finalizing each ``RoutedExperts`` layer until the end of the full
    model sync, which retains every layer's temporary packed weights on device.

    This must run before vLLM constructs the model and records its layerwise
    reload metadata. ``skip_tensors`` is injectable for dependency-free tests.
    """
    if skip_tensors is None:
        try:
            from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS
        except ImportError as exc:
            raise RuntimeError("Integer INT4 RL reload requires vLLM's layerwise reload API (vLLM 0.24+).") from exc

        skip_tensors = SKIP_TENSORS
    skip_tensors.update(_WNA16_DERIVED_RELOAD_TENSORS)


def configure_int4_vllm_backend(force_generic: bool | None = None, moe_module: Any = None) -> bool:
    """Optionally force vLLM's generic WNA16 MoE method for diagnosis/fallback.

    The default remains Marlin when vLLM reports it supported. Setting
    ``VERL_INT4_QAT_FORCE_GENERIC_WNA16=1`` disables only the Marlin method
    selector inside the dedicated rollout worker process. This makes it
    possible to distinguish checkpoint/reload errors from Marlin repack/kernel
    errors without changing the trainer/export format.
    """
    if force_generic is None:
        force_generic = os.environ.get("VERL_INT4_QAT_FORCE_GENERIC_WNA16", "0") == "1"
    if not force_generic:
        return False

    if moe_module is None:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
            compressed_tensors_moe as moe_module,
        )

    moe_module.check_moe_marlin_supports_layer = lambda *_args, **_kwargs: False
    logger.warning("Integer INT4 QAT: forcing generic vLLM WNA16 MoE backend instead of Marlin")
    return True


def prepare_int4_for_weight_reload(model: torch.nn.Module) -> None:
    """Restore checkpoint-layout tensors and wrap loaders before bucketed sync."""
    try:
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload
    except ImportError as exc:
        raise RuntimeError("Integer INT4 RL reload requires vLLM's layerwise reload API (vLLM 0.24+).") from exc

    initialize_layerwise_reload(model)


def finalize_int4_weight_reload(model: torch.nn.Module, model_config: Any) -> None:
    """Repack WNA16 weights and copy them into stable kernel tensor storage."""
    try:
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload
    except ImportError as exc:
        raise RuntimeError("Integer INT4 RL reload requires vLLM's layerwise reload API (vLLM 0.24+).") from exc

    finalize_layerwise_reload(model, model_config)
    if os.environ.get("VERL_INT4_QAT_RELOAD_DIAGNOSTICS", "0") == "1":
        scale_tensors = [
            (name, parameter.detach())
            for name, parameter in model.named_parameters()
            if name.endswith(("w13_weight_scale", "w2_weight_scale"))
        ]
        invalid_by_name = []
        scale_min = float("inf")
        scale_max = float("-inf")
        for name, scale in scale_tensors:
            invalid = (~torch.isfinite(scale)) | (scale <= 0)
            invalid_count = int(invalid.sum().item())
            if invalid_count:
                invalid_by_name.append((name, invalid_count, scale.numel()))
            finite_positive = scale[~invalid]
            if finite_positive.numel() > 0:
                scale_min = min(scale_min, float(finite_positive.min().item()))
                scale_max = max(scale_max, float(finite_positive.max().item()))
        logger.warning(
            "Integer INT4 reload diagnostics: scale_tensors=%d invalid=%d min=%g max=%g invalid_by_name=%s",
            len(scale_tensors),
            sum(item[1] for item in invalid_by_name),
            scale_min,
            scale_max,
            invalid_by_name,
        )


__all__ = [
    "configure_int4_layerwise_reload",
    "configure_int4_vllm_backend",
    "expand_qwen3_5_fused_int4_weights",
    "finalize_int4_weight_reload",
    "is_int4_wna16_quant_config",
    "patch_qwen3_5_fused_int4_loader",
    "prepare_int4_for_weight_reload",
]
