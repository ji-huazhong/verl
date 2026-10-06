# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Individual Contributor
"""The order-only probe preserves native selection, padding and caller buffers."""

from pathlib import Path

import torch

from verl.utils.import_utils import load_module

# Load the pure tensor helper directly: importing the mcore package eagerly
# initializes the optional Megatron registry, which is unavailable on CPU CI.
qsa_order = load_module(str(Path(__file__).resolve().parents[3] / "verl/models/mcore/qwen3_8_next/qsa_order.py"))
QsaBlockOrderProbe = qsa_order.QsaBlockOrderProbe
canonical_block_order = qsa_order.canonical_block_order


def test_canonical_order_keeps_short_context_padding_at_end():
    selected = torch.tensor([[5, -1, 0, 3, -1], [-1, -1, -1, -1, -1]], dtype=torch.int32)
    original = selected.clone()
    result = canonical_block_order(selected)
    assert torch.equal(selected, original)
    assert torch.equal(result, torch.tensor([[0, 3, 5, -1, -1], [-1, -1, -1, -1, -1]], dtype=torch.int32))
    assert torch.equal(selected.sort(dim=-1).values, result.sort(dim=-1).values)


def test_canonical_order_preserves_duplicate_multiplicity_and_strided_input():
    storage = torch.tensor([[4, 99, 2, 99, 4, 99, -1, 99]], dtype=torch.int64)
    selected = storage[:, ::2]
    result = canonical_block_order(selected)
    assert torch.equal(result, torch.tensor([[2, 4, 4, -1]], dtype=torch.int64))
    assert torch.equal(selected.sort(dim=-1).values, result.sort(dim=-1).values)
    assert torch.equal(storage[:, 1::2], torch.full((1, 4), 99, dtype=torch.int64))


def test_probe_calls_native_once_with_original_arguments_then_updates_buffer():
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    visible = torch.tensor([3], dtype=torch.int32)
    workspace = torch.empty(4, dtype=torch.uint8)
    output = torch.empty((1, 3), dtype=torch.int32)
    calls = []

    def native(a, b, budget, ratio, c, d):
        assert a is logits and b is visible and c is output and d is workspace
        calls.append((budget, ratio))
        c.copy_(torch.tensor([[2, 0, 1]], dtype=torch.int32))

    probe = QsaBlockOrderProbe(native)
    assert probe(logits, visible, 12, 4, output, workspace) is None
    assert calls == [(12, 4)] and probe.calls == 1
    assert torch.equal(output, torch.tensor([[0, 1, 2]], dtype=torch.int32))
    assert torch.equal(logits, torch.tensor([[3.0, 2.0, 1.0]]))
