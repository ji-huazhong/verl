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
from collections.abc import Iterable, Iterator
from types import MethodType
from typing import Any

import torch

logger = logging.getLogger(__name__)

_FUSED_GATE_UP_SUFFIXES = (".mlp.experts.gate_up_proj.weight_packed", ".mlp.experts.gate_up_proj.weight_scale")
_FUSED_DOWN_SUFFIXES = (".mlp.experts.down_proj.weight_packed", ".mlp.experts.down_proj.weight_scale")


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


__all__ = [
    "expand_qwen3_5_fused_int4_weights",
    "finalize_int4_weight_reload",
    "is_int4_wna16_quant_config",
    "patch_qwen3_5_fused_int4_loader",
    "prepare_int4_for_weight_reload",
]
