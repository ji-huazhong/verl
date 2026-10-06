# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Keep real Accelerate pre/post hooks active during diagnostic forward changes."""

import pytest
import torch

accelerate_hooks = pytest.importorskip("accelerate.hooks")

from examples.grpo_trainer.qwen3_8_next.precision_probe import _replace_forward  # noqa: E402


class Scale(torch.nn.Module):
    def forward(self, value):
        return value * 2


class TransferSentinel(accelerate_hooks.ModelHook):
    def __init__(self):
        self.calls = []

    def pre_forward(self, module, value):
        self.calls.append("pre")
        return (value * 3,), {}

    def post_forward(self, module, output):
        self.calls.append("post")
        return output + 5


def replacement(self, value):
    return value * 7


def test_precision_probe_preserves_accelerate_transfer_hooks():
    module = Scale()
    hook = TransferSentinel()
    accelerate_hooks.add_hook_to_module(module, hook)
    _replace_forward(module, replacement)
    assert module(torch.tensor(1.0)).item() == 26.0
    assert hook.calls == ["pre", "post"]
    accelerate_hooks.remove_hook_from_module(module)
    assert module(torch.tensor(1.0)).item() == 7.0


def test_precision_probe_without_accelerate_dispatch():
    module = Scale()
    _replace_forward(module, replacement)
    assert module(torch.tensor(1.0)).item() == 7.0
