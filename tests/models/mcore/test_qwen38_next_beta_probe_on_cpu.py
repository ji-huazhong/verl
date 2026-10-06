# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Verify decay invariance, unchanged inputs and restoration of beta probes."""

import pytest
import torch

from examples.grpo_trainer.qwen3_8_next.gdn_beta_ab import GdnBetaPrecisionProbe


class Qwen38NextGatedDeltaNet(torch.nn.Module):
    @staticmethod
    def _compute_g_and_beta(a_log, dt_bias, alpha, beta):
        return -a_log.float().exp() * torch.nn.functional.softplus(alpha.float() + dt_bias), beta.sigmoid()


def test_candidate_changes_only_beta_precision_and_restores_original_methods():
    model = torch.nn.ModuleList([Qwen38NextGatedDeltaNet() for _ in range(36)])
    arguments = [torch.randn(1, 12, 6).bfloat16() for _ in range(4)]
    saved = [value.clone() for value in arguments]
    methods = [module._compute_g_and_beta for module in model]
    native = methods[0](*arguments)
    probe = GdnBetaPrecisionProbe(model)
    try:
        probe.select("traced0")
        for module in model:
            result = module._compute_g_and_beta(*arguments)
            assert torch.equal(result[0], native[0])
            assert result[1].dtype == torch.float32
            assert torch.equal(result[1], arguments[-1].float().sigmoid())
        with pytest.raises(ValueError, match="Unknown beta"):
            probe.select("unknown")
    finally:
        probe.close()
    assert [module._compute_g_and_beta for module in model] == methods
    assert torch.equal(model[0]._compute_g_and_beta(*arguments)[1], native[1])
    assert all(torch.equal(before, after) for before, after in zip(saved, arguments, strict=True))


def test_probe_rejects_partial_model_before_changing_methods():
    model = torch.nn.ModuleList([Qwen38NextGatedDeltaNet()])
    original = model[0]._compute_g_and_beta
    with pytest.raises(ValueError, match="36"):
        GdnBetaPrecisionProbe(model)
    assert model[0]._compute_g_and_beta is original
