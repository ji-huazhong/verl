# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""HF conversion for the separately trained QKVZ and beta/alpha projections."""

import torch
from megatron.bridge.models.conversion.param_mapping import ColumnParallelMapping, MegatronParamMapping

from .ops.gdn_projection import gdn_projection_sections


class Qwen38NextGDNProjectionMapping(MegatronParamMapping):
    def __init__(self, megatron_param, hf_param):
        if set(hf_param) not in ({"qkv", "z"}, {"b", "a"}):
            raise ValueError("Expected either QKV/Z or beta/alpha HF projections")
        super().__init__(megatron_param, hf_param)
        self.qkvz = "qkv" in hf_param
        self._tp_mapping = ColumnParallelMapping(megatron_param, megatron_param)

    def hf_to_megatron(self, hf_weights, megatron_module):
        merged = None
        if self.tp_rank == 0:
            sections = gdn_projection_sections(self._get_config(megatron_module))[0 if self.qkvz else 1]
            if self.qkvz:
                parts = [*hf_weights["qkv"].split(sections[:3], dim=0), hf_weights["z"]]
            else:
                parts = [hf_weights["b"], hf_weights["a"]]
            for part, size in zip(parts, sections, strict=True):
                if part.shape[0] != size or size % self.tp_size:
                    raise ValueError("HF GDN projection shape or TP partition is invalid")
            # Rank-major packing: [Q_r, K_r, V_r, Z_r] or [B_r, A_r].
            merged = torch.cat([p.reshape(self.tp_size, -1, p.shape[-1]) for p in parts], dim=1)
            merged = merged.flatten(0, 1)
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        sections = None
        if megatron_module is not None:
            sections = gdn_projection_sections(self._get_config(megatron_module))[0 if self.qkvz else 1]
        sections = self.broadcast_obj_from_pp_rank(sections)
        packed = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed:
            return {}
        weight = next(iter(packed.values()))
        rank_major = weight.reshape(self.tp_size, -1, weight.shape[-1])
        parts = [p.flatten(0, 1) for p in rank_major.split([s // self.tp_size for s in sections], dim=1)]
        if self.qkvz:
            return {self.hf_param["qkv"]: torch.cat(parts[:3], dim=0), self.hf_param["z"]: parts[3]}
        return {self.hf_param["b"]: parts[0], self.hf_param["a"]: parts[1]}

    def resolve(self, captures):
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param)
