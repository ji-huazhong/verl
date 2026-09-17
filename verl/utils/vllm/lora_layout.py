# SPDX-License-Identifier: Apache-2.0
"""Normalize canonical stacked expert LoRA exports for vLLM's 2D MoE loader.

Input contract: A=[experts, rank, in], B=[experts, out, rank]. Packed
projections concatenate B's output rows in packed_modules_mapping order.
This is NOT the transposed/flattened PEFT ParamWrapper checkpoint format.
There are no model-type checks; unsupported or ambiguous shapes fail closed.
"""

from collections.abc import Mapping, Sequence

import torch


def expand_stacked_expert_lora(
    tensors: dict[str, torch.Tensor],
    packed_modules_mapping: Mapping[str, Sequence[str]],
    *,
    supported_expert_modules: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Expose per-expert factor views without merging adapters into base weights.

    Ordinary 2D adapters pass through unchanged. Each stacked factor pair must
    address a projection immediately below ``experts``. This deliberately does
    not guess the meaning of other 3D adapter tensors or split unequal slices.
    """
    if not any(tensor.ndim == 3 for tensor in tensors.values()):
        return tensors
    supported = None if supported_expert_modules is None else set(supported_expert_modules)
    stacked = {}
    result = {}
    for name, tensor in tensors.items():
        if tensor.ndim != 3:
            result[name] = tensor
            continue
        base, sep, suffix = name.rpartition(".lora_")
        parent, dot, projection = base.rpartition(".")
        if not sep or suffix not in ("A.weight", "B.weight") or not dot or parent.split(".")[-1] != "experts":
            raise ValueError(f"Unsupported stacked LoRA target: {name}")
        stacked.setdefault((parent, projection), {})[suffix[0]] = tensor

    def add(name, value):
        if name in result:
            raise ValueError(f"Overlapping stacked and per-expert LoRA target: {name}")
        result[name] = value.contiguous()

    for (parent, projection), pair in stacked.items():
        if pair.keys() != {"A", "B"}:
            raise ValueError(f"Incomplete stacked LoRA A/B pair: {parent}.{projection}")
        a, b = pair["A"], pair["B"]
        if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[2] or min(*a.shape, *b.shape) == 0:
            raise ValueError(f"Invalid canonical expert LoRA shapes: A={tuple(a.shape)}, B={tuple(b.shape)}")
        parts = packed_modules_mapping.get(projection, [projection])
        if (
            isinstance(parts, str)
            or not parts
            or len(parts) != len(set(parts))
            or any(not part or "." in part for part in parts)
        ):
            raise ValueError(f"Invalid packed projection mapping for {projection}")
        if b.shape[1] % len(parts):
            raise ValueError(f"Packed projection {projection} cannot be split into equal output slices")
        chunks = b.chunk(len(parts), dim=1)
        for expert in range(a.shape[0]):
            for part, b_chunk in zip(parts, chunks, strict=True):
                if supported is not None and f"experts.{expert}.{part}" not in supported:
                    raise ValueError(
                        f"The active vLLM loader does not advertise the 2D expert target experts.{expert}.{part}"
                    )
                prefix = f"{parent}.{expert}.{part}"
                add(f"{prefix}.lora_A.weight", a[expert])
                add(f"{prefix}.lora_B.weight", b_chunk[expert])
    return result
