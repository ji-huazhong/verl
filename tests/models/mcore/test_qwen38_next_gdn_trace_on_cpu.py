# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Individual Contributor
import pytest
import torch

from examples.grpo_trainer.qwen3_8_next.backend_trace import (
    BackendPrefixTrace,
    compare_backend_traces,
    load_backend_trace,
)


def test_comparison_orders_other_rank_inputs_before_attention_output(tmp_path):
    for variant in ("left", "right"):
        for rank in range(2):
            trace = BackendPrefixTrace(tmp_path / variant, "megatron", rank, rank, 2)
            trace.start(dict(id="short", input_ids=list(range(198))), 200, "config")
            value = torch.zeros(200, 1, 8)
            if variant == "right" and rank == 1:
                value[0] = 0.25
            trace.capture_gdn_partition("layers/20/gdn/prepared/k", value)
            if rank == 0:
                output = torch.zeros(32, 8) if variant == "left" else torch.ones(32, 8)
                trace._store_window("layers/20/attention/output", output, 0)
            trace.finish()
    result = compare_backend_traces(tmp_path / "left", tmp_path / "right", "short")
    assert result["first_nonzero_stage"] == "layers/20/gdn/prepared/k/tp-01"


def test_feature_shards_preserve_tokens_heads_and_rank(tmp_path):
    prompt = dict(id="short", input_ids=list(range(198)))
    for rank in range(2):
        trace = BackendPrefixTrace(tmp_path, "megatron", rank, rank, 2)
        trace.start(prompt, 200, "config")
        value = torch.arange(200 * 2 * 4).reshape(1, 200, 2, 4) + rank
        trace.capture_gdn_partition("layers/20/gdn/core", value, batch_first=True)
        trace.capture_gdn_partition("layers/20/gdn/norm", value.reshape(400, 4), flatten_head_rows=True)
        with pytest.raises(ValueError, match="Incomplete"):
            trace.capture_gdn_partition("invalid", value[:, :100], batch_first=True)
        trace.finish()
    _, values = load_backend_trace(tmp_path, "short")
    for rank in range(2):
        expected = torch.arange(200 * 8).reshape(200, 8)[:32] + rank
        assert torch.equal(values[f"layers/20/gdn/core/tp-{rank:02d}"], expected)
        assert torch.equal(values[f"layers/20/gdn/norm/tp-{rank:02d}"], expected)


def test_detailed_gdn_hooks_are_passive_and_restorable(tmp_path):
    class SplitProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkvz = torch.nn.Identity()
            self.ba = torch.nn.Identity()

        def forward(self, value):
            return torch.cat((self.qkvz(value[..., :6]), self.ba(value[..., 6:])), dim=-1)

    class GDN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = SplitProjection()
            self.out_proj = torch.nn.Identity()

        def _prepare_input_for_gated_delta_rule(self, value):
            return (value.reshape(1, 200, 2, 4),) * 6

        def _compute_g_and_beta(self, value):
            return value[..., 0], value[..., 1]

        def gated_delta_rule(self, value):
            return value * 2, None

        def _apply_gated_norm(self, value):
            return value.reshape(400, 4) + 1

        def forward(self, value):
            projected = self.in_proj(value)
            prepared = self._prepare_input_for_gated_delta_rule(projected.transpose(0, 1))
            self._compute_g_and_beta(prepared[0])
            core = self.gated_delta_rule(prepared[0])[0]
            return self.out_proj(self._apply_gated_norm(core).reshape(200, 1, 8))

    module = GDN()
    value = torch.randn(200, 1, 8)
    expected = module(value)
    trace = BackendPrefixTrace(tmp_path, "megatron", 0, 0, 1)
    trace.attach_gdn_detail(module, "layers/20")
    trace.start(dict(id="short", input_ids=list(range(198))), 200, "config")
    assert torch.equal(module(value), expected)
    trace.finish()
    trace.close()
    assert torch.equal(module(value), expected)
    _, snapshots = load_backend_trace(tmp_path, "short")
    assert torch.equal(snapshots["layers/20/gdn/out_proj_input/tp-00"], expected[:32, 0])

    for part, expected_part in (("qkvz", value[:32, 0, :6]), ("ba", value[:32, 0, 6:])):
        assert torch.equal(snapshots[f"layers/20/gdn/projection/{part}/tp-00"], expected_part)
        assert not getattr(module.in_proj, part)._forward_hooks
