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

"""Integer INT4 W4A16 utilities shared by Megatron QAT and vLLM export.

The checkpoint contract is symmetric uint4b8 (signed levels biased by eight),
group-wise along the last weight dimension, with eight nibbles packed into one
INT32. Activations and master weights remain BF16/FP16.
"""

import logging
import re
from collections.abc import Iterable, Iterator
from types import MethodType
from typing import Any

import torch

logger = logging.getLogger(__name__)

INT4_QMIN = -7
INT4_QMAX = 7
INT4_BIAS = 8
INT4_PACK_FACTOR = 8
INT4_SCALE_EPS = 1e-5

_INDIVIDUAL_EXPERT_WEIGHT_RE = re.compile(r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$")
_FUSED_EXPERT_SUFFIXES = (
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
)
_MCORE_EXPERT_SUFFIXES = (
    "mlp.experts.linear_fc1",
    "mlp.experts.linear_fc2",
)
_MCORE_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _resolve_scale_dtype(scale_dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(scale_dtype, torch.dtype):
        return scale_dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return dtype_map[scale_dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported INT4 scale dtype: {scale_dtype!r}") from exc


def _validate_weight(weight: torch.Tensor, group_size: int) -> None:
    if weight.ndim < 2:
        raise ValueError(f"INT4 weight must have at least two dimensions, got shape={tuple(weight.shape)}")
    if group_size <= 0:
        raise ValueError(f"INT4 group_size must be positive, got {group_size}")
    if weight.shape[-1] % group_size != 0:
        raise ValueError(f"INT4 input dimension ({weight.shape[-1]}) must be divisible by group_size ({group_size})")
    if weight.shape[-1] % INT4_PACK_FACTOR != 0:
        raise ValueError(
            f"INT4 input dimension ({weight.shape[-1]}) must be divisible by pack factor {INT4_PACK_FACTOR}"
        )


@torch.no_grad()
def quantize_int4_levels(
    weight: torch.Tensor,
    group_size: int = 128,
    scale_dtype: str | torch.dtype = "bfloat16",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed INT4 levels and stored per-group scales.

    Amax and division are calculated in FP32. The scale is rounded to its
    checkpoint dtype before levels are calculated, so trainer fake-quant and
    exporter dequantization share exactly the same representable grid.
    """
    _validate_weight(weight, group_size)
    stored_dtype = _resolve_scale_dtype(scale_dtype)
    grouped = weight.float().reshape(*weight.shape[:-1], weight.shape[-1] // group_size, group_size)
    scale_fp32 = grouped.abs().amax(dim=-1).div(INT4_QMAX).clamp_min(INT4_SCALE_EPS)
    scale = scale_fp32.to(stored_dtype)
    levels = torch.round(grouped / scale.float().unsqueeze(-1)).clamp(INT4_QMIN, INT4_QMAX)
    return levels.to(torch.int8).flatten(start_dim=-2), scale


def dequantize_int4_levels(
    levels: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize signed levels using the stored scale tensor."""
    if levels.shape[-1] % group_size != 0:
        raise ValueError("INT4 level tensor is incompatible with group_size")
    grouped = levels.float().reshape(*levels.shape[:-1], levels.shape[-1] // group_size, group_size)
    return (grouped * scale.float().unsqueeze(-1)).flatten(start_dim=-2).to(output_dtype)


class _Int4FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, group_size: int, scale_dtype: str) -> torch.Tensor:
        if weight.is_cuda and weight.is_contiguous():
            from verl.utils.qat.int4_triton import fake_quant_int4_cuda

            output = fake_quant_int4_cuda(weight, group_size, scale_dtype)
        else:
            levels, scale = quantize_int4_levels(weight, group_size, scale_dtype)
            output = dequantize_int4_levels(levels, scale, group_size, weight.dtype)
        if hasattr(weight, "main_grad"):
            output.main_grad = weight.main_grad
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


def fake_quant_int4_ste(
    weight: torch.Tensor,
    group_size: int = 128,
    scale_dtype: str = "bfloat16",
) -> torch.Tensor:
    """Apply integer INT4 QDQ in forward and identity STE in backward."""
    _validate_weight(weight, group_size)
    _resolve_scale_dtype(scale_dtype)
    return _Int4FakeQuantSTE.apply(weight, group_size, scale_dtype)


def pack_int4_levels(levels: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 levels along the last dimension into GPTQ-order INT32."""
    if levels.shape[-1] % INT4_PACK_FACTOR != 0:
        raise ValueError(f"INT4 level dimension must be divisible by {INT4_PACK_FACTOR}")
    unsigned = (levels.to(torch.int32) + INT4_BIAS).bitwise_and(0xF)
    chunks = unsigned.reshape(*unsigned.shape[:-1], unsigned.shape[-1] // INT4_PACK_FACTOR, INT4_PACK_FACTOR)
    packed = torch.zeros(chunks.shape[:-1], dtype=torch.int32, device=levels.device)
    for index in range(INT4_PACK_FACTOR):
        packed.bitwise_or_(chunks[..., index] << (index * 4))
    return packed


def unpack_int4_levels(packed: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ-order uint4b8 INT32 values back to signed INT8 levels."""
    values = [((packed >> (index * 4)) & 0xF) - INT4_BIAS for index in range(INT4_PACK_FACTOR)]
    return torch.stack(values, dim=-1).flatten(start_dim=-2).to(torch.int8)


@torch.no_grad()
def quantize_int4_weight(
    weight: torch.Tensor,
    group_size: int = 128,
    scale_dtype: str | torch.dtype = "bfloat16",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and pack a 2D or fused 3D expert weight."""
    _validate_weight(weight, group_size)
    if weight.is_cuda and weight.is_contiguous():
        from verl.utils.qat.int4_triton import quantize_pack_int4_cuda

        dtype_name = "bfloat16" if _resolve_scale_dtype(scale_dtype) == torch.bfloat16 else "float16"
        return quantize_pack_int4_cuda(weight, group_size, dtype_name)
    levels, scale = quantize_int4_levels(weight, group_size, scale_dtype)
    return pack_int4_levels(levels), scale


def _fused_expert_base_name(name: str) -> str | None:
    candidate = name.removesuffix(".weight")
    return candidate if candidate.endswith(_FUSED_EXPERT_SUFFIXES) else None


def is_routed_expert_weight(name: str, weight: torch.Tensor) -> bool:
    """Return whether an HF-format tensor is a supported routed-expert weight."""
    if _fused_expert_base_name(name) is not None:
        return weight.ndim == 3
    return weight.ndim == 2 and _INDIVIDUAL_EXPERT_WEIGHT_RE.search(name) is not None


def order_mbridge_tasks_by_layer(tasks: Iterable[Any]) -> list[Any]:
    """Return a stable layer-major order for Megatron Bridge export tasks.

    Megatron Bridge normally follows checkpoint/mapping order, which may emit
    one projection for every transformer layer before returning to the next
    projection. vLLM's layerwise reload then has to retain one incomplete
    ``RoutedExperts`` buffer per layer. Grouping tasks by the numeric layer in
    ``global_param_name`` lets vLLM finalize each parent layer immediately.

    Non-layer tasks stay stable and are emitted first. The key depends only on
    the global task directory, so every model-parallel rank keeps the same
    collective order.
    """
    indexed_tasks = list(enumerate(tasks))

    def _key(indexed_task: tuple[int, Any]) -> tuple[int, int, int]:
        index, task = indexed_task
        match = _MCORE_LAYER_RE.search(str(getattr(task, "global_param_name", "")))
        if match is None:
            return (0, 0, index)
        return (1, int(match.group(1)), index)

    return [task for _, task in sorted(indexed_tasks, key=_key)]


class Int4WeightExporter:
    """Stream BF16 HF weights as compressed-tensors integer INT4 tensors."""

    def __init__(
        self,
        group_size: int = 128,
        scale_dtype: str = "bfloat16",
        scope: str = "routed_experts",
        require_match: bool = True,
    ) -> None:
        if scope != "routed_experts":
            raise ValueError("Integer INT4 exporter currently supports only scope='routed_experts'.")
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.scope = scope
        self.require_match = require_match

    def process_weights_iterator(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        quantized = 0
        for name, weight in weights:
            if not is_routed_expert_weight(name, weight):
                yield name, weight
                continue

            base_name = name.removesuffix(".weight")
            packed, scale = quantize_int4_weight(weight, self.group_size, self.scale_dtype)
            yield f"{base_name}.weight_packed", packed
            yield f"{base_name}.weight_scale", scale

            # Qwen3.5's fused expert loader in vLLM 0.24 does not dispatch a
            # fused weight_shape suffix. The parameter is metadata-only and is
            # not read by WNA16 post-processing, so online sync intentionally
            # leaves its dummy-initialized value in place. Individual expert
            # layouts retain the standard compressed-tensors tensor.
            if weight.ndim == 2:
                shape = torch.tensor(weight.shape, dtype=torch.int64, device=weight.device)
                yield f"{base_name}.weight_shape", shape
            quantized += 1

        if self.require_match and quantized == 0:
            raise RuntimeError(
                "Integer INT4 QAT did not find routed expert weights in the Megatron-to-HF export stream."
            )
        logger.info("Integer INT4 exporter quantized %d routed expert tensors", quantized)


def _is_mcore_routed_expert_grouped_linear(name: str, module: torch.nn.Module) -> bool:
    return name.endswith(_MCORE_EXPERT_SUFFIXES) and callable(getattr(module, "_get_weight_tensors", None))


def apply_int4_qat_to_modules(modules: list[torch.nn.Module], qat_config: Any) -> list[torch.nn.Module]:
    """Patch instantiated Megatron TE GroupedLinear routed experts for INT4 QAT."""
    group_size = int(getattr(qat_config, "group_size", 128))
    scale_dtype = str(getattr(qat_config, "scale_dtype", "bfloat16"))
    patched = 0

    for model in modules:
        for name, module in model.named_modules():
            if not _is_mcore_routed_expert_grouped_linear(name, module):
                continue
            if hasattr(module, "_verl_int4_original_get_weight_tensors"):
                continue

            module._verl_int4_original_get_weight_tensors = module._get_weight_tensors

            def _qat_get_weight_tensors(self):
                weights = self._verl_int4_original_get_weight_tensors()
                return [fake_quant_int4_ste(weight, group_size, scale_dtype) for weight in weights]

            module._get_weight_tensors = MethodType(_qat_get_weight_tensors, module)
            module._verl_int4_qat_group_size = group_size
            module._verl_int4_qat_scale_dtype = scale_dtype
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "Integer INT4 QAT requires routed experts implemented by Megatron TE GroupedLinear "
            "(mlp.experts.linear_fc1/linear_fc2); no compatible modules were found."
        )
    logger.info("Enabled integer INT4 fake quant on %d Megatron routed-expert GroupedLinear modules", patched)
    return modules


__all__ = [
    "INT4_BIAS",
    "INT4_PACK_FACTOR",
    "INT4_QMAX",
    "INT4_QMIN",
    "Int4WeightExporter",
    "apply_int4_qat_to_modules",
    "dequantize_int4_levels",
    "fake_quant_int4_ste",
    "is_routed_expert_weight",
    "order_mbridge_tasks_by_layer",
    "pack_int4_levels",
    "quantize_int4_levels",
    "quantize_int4_weight",
    "unpack_int4_levels",
]
