# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Separate the wide GDN QKVZ projection from its narrow beta/alpha gates.

The TP8 Flash-Next projection has 2060 output channels. Merging the 2048
QKVZ channels and 12 gate channels selects a less accurate BF16 GEMM on
the validated SM90/cu131 runtime. Keep native TE autograd and collectives
for both projections, with the same component ordering as vLLM.
"""

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.ssm.utils import _split_tensor_factory
from megatron.core.transformer.module import MegatronModule


def gdn_projection_sections(config):
    key = config.linear_num_key_heads * config.linear_key_head_dim
    value = config.linear_num_value_heads * config.linear_value_head_dim
    return (key, key, value, value), (config.linear_num_value_heads,) * 2


class Qwen38NextGDNInputProjection(MegatronModule):
    def __init__(self, input_size, output_size, *, config, tp_group, name=None, **kwargs):
        super().__init__(config)
        if kwargs.get("bias", False) or kwargs.get("gather_output", False):
            raise ValueError("Flash-Next GDN requires bias-free, column-sharded input projections")
        if config.tp_comm_overlap:
            raise NotImplementedError("Split GDN projections do not support shared TE userbuffer overlap")
        self.tp_group = tp_group
        self.sections = gdn_projection_sections(config)
        if sum(map(sum, self.sections)) != output_size:
            raise ValueError("GDN projection dimensions disagree with the model configuration")
        for part, sections in zip(("qkvz", "ba"), self.sections, strict=True):
            if any(size % tp_group.size() for size in sections):
                raise ValueError("Each GDN projection component must be divisible by TP")
            setattr(
                self,
                part,
                TEColumnParallelLinear(
                    input_size,
                    sum(sections),
                    config=config,
                    tp_group=tp_group,
                    name=f"{name}.{part}" if name else None,
                    **kwargs,
                ),
            )

    def forward(self, hidden_states):
        qkvz, qkvz_bias = self.qkvz(hidden_states)
        ba, ba_bias = self.ba(hidden_states)
        assert qkvz_bias is None and ba_bias is None
        return torch.cat((qkvz, ba), dim=-1), None

    def backward_dw(self):
        self.qkvz.backward_dw()
        self.ba.backward_dw()

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        result = {}
        group = tp_group if tp_group is not None else self.tp_group
        if group is not self.tp_group:
            raise ValueError("GDN projection checkpoint TP group must match the module")
        for part, sections, names in zip(
            ("qkvz", "ba"), self.sections, (("query", "key", "value", "z"), ("beta", "alpha")), strict=True
        ):
            state = getattr(self, part).sharded_state_dict(f"{prefix}{part}.", sharded_offsets, metadata)
            key = f"{prefix}{part}.weight"
            state[key] = _split_tensor_factory(state[key], [s // group.size() for s in sections], list(names), 0)
            result.update(state)
        return result
