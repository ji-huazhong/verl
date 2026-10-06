# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
# Q/K packing adapted from NVIDIA Megatron Core common.py:
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.
"""Flash-Next separates the GDN output gate from convolution/MLP activation."""

from copy import copy

import torch
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.utils import _split_tensor_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint


@torch.compile(dynamic=True, fullgraph=True)
def fp32_qk_l2norm(x, eps=1e-6):
    """Normalize Q/K in FP32 and preserve FP32 intermediates for autograd."""
    values = x.float()
    inverse_norm = torch.rsqrt(values.square().sum(-1, keepdim=True) + eps)
    return (values * inverse_norm).to(x.dtype)


class Qwen38NextGDNOutputNorm(TENorm):
    """Keep the HF/vLLM norm's ordinary gamma representation during training.

    Converting a BF16 zero-centered gamma by adding/subtracting one loses
    small optimizer updates on every actor-to-rollout weight transfer.
    Other Flash-Next norms still use their native zero-centered convention.
    """

    def __new__(cls, config, *args, **kwargs):
        norm_config = copy(config)
        norm_config.layernorm_zero_centered_gamma = False
        return TENorm(norm_config, *args, **kwargs)


class Qwen38NextGatedDeltaNet(GatedDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.config, "qwen3_8_next_gdn_backend", "fla") == "flashqla":
            from importlib.metadata import version

            if version("flash-qla") != "0.1.2+7c7dfe1.nebula1":
                raise RuntimeError("FlashQLA candidate requires the pinned, separately validated Miles source wheel")
            from flash_qla import chunk_gated_delta_rule

            self.gated_delta_rule = chunk_gated_delta_rule

    def _prepare_input_for_gated_delta_rule(self, qkv, gate, batch, seq_len, *gate_feats):
        # Packing follows pinned Core _GDNBase; only Q/K normalization changes.
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        if self.use_qk_l2norm:
            query_key = fp32_qk_l2norm(query_key.contiguous())
        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)
        repeat_factor = self.num_value_heads // self.num_key_heads
        if repeat_factor > 1:
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)
        return (
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            *(tensor.contiguous() for tensor in gate_feats),
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        # Core assumes one in_proj.weight. The split module owns its two
        # factories; retain native GDN sharding for the remaining parameters.
        if tp_group is not None and tp_group is not self.tp_group:
            raise ValueError("Flash-Next GDN checkpoint TP group must match the module")
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        result = MegatronModule.sharded_state_dict(self, prefix, sharded_offsets, metadata)
        direct = {"A_log": self.A_log, "dt_bias": self.dt_bias, "conv1d.weight": self.conv1d.weight}
        if self.conv_bias:
            direct["conv1d.bias"] = self.conv1d.bias
        result.update(
            make_sharded_tensors_for_checkpoint(
                direct,
                prefix,
                {name: 0 for name in direct},
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        for name in ("conv1d.weight", "conv1d.bias"):
            if name in direct:
                result[prefix + name] = _split_tensor_factory(
                    result[prefix + name],
                    [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
                    ["query", "key", "value"],
                    0,
                )
        return result

    def _apply_gated_norm(self, x, gate):
        # Core 0.18 uses config.activation_func (SiLU) for both the convolution
        # and output gate. Flash-Next's checkpoint explicitly requests sigmoid
        # for the latter; changing activation_func would also corrupt conv/MLP.
        if self.config.qwen3_8_next_output_gate_type != "sigmoid":
            raise ValueError("Flash-Next GDN requires its declared sigmoid output gate")
        # Preserve the BF16 parameter and its sequence-parallel metadata, but
        # avoid rounding the normalized activation before the output gate.
        # Casting TENorm's input alone still returns BF16 in the pinned TE build.
        rows = x.reshape(-1, x.shape[-1]).float()
        inverse_rms = torch.rsqrt(rows.square().mean(-1, keepdim=True) + self.config.layernorm_epsilon)
        normed = rows * inverse_rms * self.out_norm.weight.float()
        output_gate = gate.reshape(-1, gate.shape[-1]).float().sigmoid()
        return (normed * output_gate).to(x.dtype)
