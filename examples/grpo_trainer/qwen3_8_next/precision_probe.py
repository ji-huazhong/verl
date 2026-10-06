# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Explicit native-HF precision ablations; never installed in a training model.

Layer formulas follow Transformers 5.16's modeling_qwen4_exp.py. The HC FP32
variant mirrors the Megatron plugin's intermediate and residual cast boundaries.
Only the requested component's arithmetic changes; checkpoint values are retained.
"""

from types import MethodType

import torch
import torch.nn.functional as F


def _replace_forward(module, forward):
    """Keep Accelerate's device/offload wrapper around the diagnostic formula."""
    bound = MethodType(forward, module)
    if hasattr(module, "_hf_hook"):
        if not hasattr(module, "_old_forward"):
            raise RuntimeError("Accelerate hook has no wrapped forward")
        module._old_forward = bound
    else:
        module.forward = bound


def _norm_fp32(self, x):
    return self._norm(x.float()) * (1.0 + self.weight.float())


def _hc_fp32(self, hyper_input):
    normed = self.hc_norm(hyper_input)
    down = F.silu(self.input_mix_weight_down(normed) / self.hc_count)
    gate = self.input_mix_weight_up(down).sigmoid().unflatten(-1, (self.hc_count, self.hidden_size))
    mixed = (gate * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2).to(hyper_input.dtype)
    if self.block_inject_weight is None:
        return mixed
    injection = 2 * (self.block_inject_weight(normed) / self.hc_count).sigmoid()
    return mixed, hyper_input, injection


def _combine(residual, block_output, injection):
    increment = block_output.float().unsqueeze(-2) * injection.float().unsqueeze(-1)
    return (residual.float() + increment.flatten(-2)).to(residual.dtype)


def _layer_hc_fp32(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    conv_mask=None,
    past_key_values=None,
    ple_input_ids=None,
    **kwargs,
):
    if past_key_values is not None:
        raise ValueError("Precision ablations support uncached fixed-token forwards only")
    if self.ple is not None:
        hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, None, conv_mask=conv_mask)
    mixed, residual, injection = self.attn_hyper_connection(hidden_states)
    if self.layer_type == "linear_attention":
        block = self.linear_attn(mixed, cache_params=None, attention_mask=conv_mask, **kwargs)
    else:
        block, _ = self.self_attn(
            mixed, position_embeddings, attention_mask=attention_mask, past_key_values=None, **kwargs
        )
    hidden_states = _combine(residual, block, injection)
    mixed, residual, injection = self.mlp_hyper_connection(hidden_states)
    return _combine(residual, self.mlp(mixed), injection)


def _router_fp32(self, hidden_states):
    inputs = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(inputs.float(), self.weight)
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    scores, indices = probabilities.topk(self.top_k, dim=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(dim=-1, keepdim=True)
    # Retain the native expert-input score dtype, isolating router logits and
    # expert selection from expert accumulation/weighting precision changes.
    return logits, scores.to(hidden_states.dtype), indices


def apply_precision_probe(model, mode):
    if mode == "native":
        return
    if mode not in ("hc_fp32", "router_fp32", "hc_router_fp32"):
        raise ValueError(f"Unknown precision ablation: {mode}")
    if model.training:
        raise ValueError("Precision ablations require an evaluation model")
    language = model.model.language_model
    if mode in ("hc_fp32", "hc_router_fp32"):
        modules = [language.hyper_connection_mixer]
        for layer in language.layers:
            modules.extend((layer.attn_hyper_connection, layer.mlp_hyper_connection))
            _replace_forward(layer, _layer_hc_fp32)
        for module in modules:
            module.float()
            _replace_forward(module.hc_norm, _norm_fp32)
            _replace_forward(module, _hc_fp32)
    if mode in ("router_fp32", "hc_router_fp32"):
        for layer in language.layers:
            layer.mlp.gate.float()
            _replace_forward(layer.mlp.gate, _router_fp32)
