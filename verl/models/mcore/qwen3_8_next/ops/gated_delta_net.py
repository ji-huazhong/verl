# SPDX-License-Identifier: Apache-2.0
"""Flash-Next separates the GDN output gate from convolution/MLP activation."""

from megatron.core.ssm.gated_delta_net import GatedDeltaNet


class Qwen38NextGatedDeltaNet(GatedDeltaNet):
    def _apply_gated_norm(self, x, gate):
        # Core 0.18 uses config.activation_func (SiLU) for both the convolution
        # and output gate. Flash-Next's checkpoint explicitly requests sigmoid
        # for the latter; changing activation_func would also corrupt conv/MLP.
        if self.config.qwen3_8_next_output_gate_type != "sigmoid":
            raise ValueError("Flash-Next GDN requires its declared sigmoid output gate")
        normed = self.out_norm(x.reshape(-1, x.shape[-1]))
        output_gate = gate.reshape(-1, gate.shape[-1]).float().sigmoid()
        return (normed * output_gate).to(x.dtype)
