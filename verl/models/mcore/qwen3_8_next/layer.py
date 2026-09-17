# SPDX-License-Identifier: Apache-2.0
"""HC transformer layer for stock Megatron-Core 0.18, without global patches."""

from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor

from .hyper_connection import Qwen38NextHyperConnection, Qwen38NextPLEHyperConnection


class Qwen38NextTransformerLayer(TransformerLayer):
    def __init__(self, config, submodules, **kwargs):
        super().__init__(config, submodules, **kwargs)
        hc_cls = (
            Qwen38NextPLEHyperConnection
            if self.layer_number - 1 in config.qwen3_8_next_ple_layer_ids
            else Qwen38NextHyperConnection
        )
        self.self_attention_hyper_connection = hc_cls(config, self.layer_number)
        self.mlp_hyper_connection = Qwen38NextHyperConnection(config, self.layer_number)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        padding_mask=None,
    ):
        if context is not None or context_mask is not None:
            raise ValueError("Flash-Next has no cross-attention sublayer")
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("Use vLLM for incremental inference; this layer is the training policy")
        if self.layer_number == 1:
            if hidden_states.shape[-1] != self.config.hidden_size:
                raise ValueError("The first HC layer expects the unexpanded embedding")
            hidden_states = hidden_states.repeat(1, 1, self.config.num_residual_streams)
        if hidden_states.shape[-1] != self.config.hidden_size * self.config.num_residual_streams:
            raise ValueError("Invalid HC residual-stream width")

        hc = self.self_attention_hyper_connection
        mixed, h_res, h_post, residual = hc(hidden_states)
        attention_output = self.self_attention(
            mixed,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        hidden_states = hc.fused_h_res_h_post_bda(
            h_res, residual, h_post, attention_output, self.hidden_dropout, self.training, False
        )
        hc = self.mlp_hyper_connection
        mixed, h_res, h_post, residual = hc(hidden_states)
        output = hc.fused_h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            self.mlp(mixed, padding_mask=padding_mask),
            self.hidden_dropout,
            self.training,
            False,
        )
        return make_viewless_tensor(output, requires_grad=output.requires_grad, keep_graph=True), context
