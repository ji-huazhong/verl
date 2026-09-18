# SPDX-License-Identifier: Apache-2.0
"""CPU/metadata contracts only, not full-checkpoint numerical acceptance."""

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from tests.models.mcore.qwen38_full_validation import (
    check_recompute_gradient,
    full_checkpoint_config,
    logprob_differences,
    tensor_sha256,
)


def test_full_probe_callback_does_not_synchronize_with_host():
    # Static guard only: the full hybrid GPU run is the runtime acceptance gate.
    # An interleaved loss callback must not wait for pending P2P receives before
    # it has returned the loss required to produce the peer's backward send.
    source = Path(__file__).with_name("test_qwen38_next_full_numerical.py").read_text()
    callbacks = [
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == "collect"
    ]
    assert len(callbacks) == 1
    forbidden = {"cpu", "numpy", "item", "tolist", "synchronize"}
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden
        for node in ast.walk(callbacks[0])
    ), "Do not materialize probe logits on the host inside the pipeline loss callback"


def test_recompute_gradient_does_not_accept_erased_small_gradient():
    expected = torch.full((4,), 1e-8)
    actual = torch.zeros_like(expected)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-5)
    with pytest.raises(AssertionError, match="relative L2"):
        check_recompute_gradient(actual, expected)


def test_recompute_gradient_handles_zero_and_matching_gradients():
    assert check_recompute_gradient(torch.zeros(3), torch.zeros(3)) == 0
    assert check_recompute_gradient(torch.ones(3), torch.ones(3)) == 0
    with pytest.raises(AssertionError, match="relative L2"):
        check_recompute_gradient(torch.full((3,), 1e-20), torch.zeros(3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_bounded_hash_matches_canonical_bytes(dtype):
    base = torch.arange(120).to(dtype).reshape(4, 5, 6)
    for tensor in (base, base.transpose(0, 2), base[:, ::2], base[0, 0, 0], base[:0]):
        expected = hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        actual = tensor_sha256(tensor, chunk_bytes=32)
        assert actual == {"dtype": str(dtype), "shape": list(tensor.shape), "sha256": expected}


def test_hash_rejects_nonmaterialized_tensor_and_invalid_budget():
    with pytest.raises(ValueError, match="materialized"):
        tensor_sha256(torch.empty(3, device="meta"))
    with pytest.raises(ValueError, match="one element"):
        tensor_sha256(torch.ones(3), chunk_bytes=1)


def test_hash_bounds_each_host_transfer(monkeypatch):
    sizes = []
    original = torch.Tensor.cpu

    def record(tensor, *args, **kwargs):
        sizes.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record)
    tensor_sha256(torch.arange(120).reshape(4, 5, 6).transpose(0, 2), chunk_bytes=32)
    assert sizes and max(sizes) <= 32


def test_frozen_vllm_hash_includes_ple_metadata_but_not_lora_or_workspace():
    from tests.models.mcore.test_qwen38_next_full_vllm import frozen_vllm_hashes

    model = torch.nn.Module()
    model.ple = torch.nn.Module()
    model.ple.ngram_embedding = torch.nn.Embedding(4, 2)
    model.ple.register_buffer("ngram_heads_offsets", torch.tensor([0, 2]))
    model.ple.register_buffer("padded_buffer", torch.zeros(3))
    model.lora_A = torch.nn.Parameter(torch.zeros(2, 2))
    original = frozen_vllm_hashes(model)
    assert set(original) == {"ple.ngram_embedding.weight", "ple.ngram_heads_offsets"}
    with torch.no_grad():
        model.lora_A.add_(1)
        model.ple.padded_buffer.add_(1)
    assert frozen_vllm_hashes(model) == original
    model.ple.ngram_heads_offsets.add_(1)
    assert frozen_vllm_hashes(model) != original


def test_token_report_does_not_cancel_opposite_errors():
    expected = torch.zeros(2, 4)
    actual = expected.clone()
    actual[0, 1], actual[1, 2] = 0.25, -0.25
    result = logprob_differences(actual, expected, torch.tensor([1, 2]))
    assert result["sequence_mean_abs_diff"] == 0
    assert result["selected_token_mean"] == result["selected_token_max"] == 0.25
    assert result["all_vocab_mean"] == 0.0625
    assert result["max_position"] == 0 and result["max_vocab_id"] == 1


def test_token_report_rejects_invalid_comparisons():
    base = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="shapes"):
        logprob_differences(base, base[:1], torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="One target"):
        logprob_differences(base, base, torch.tensor([0]))
    with pytest.raises(ValueError, match="Nonfinite"):
        logprob_differences(base * float("nan"), base, torch.tensor([0, 1]))


def test_full_checkpoint_guard_rejects_reduced_models(tmp_path):
    config = {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 8}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="48-layer"):
        full_checkpoint_config(tmp_path)


def test_real_full_cases_use_original_processor():
    from tests.models.mcore.qwen38_full_validation import make_full_cases

    checkpoint = os.environ.get("QWEN38_MODEL_PATH")
    if not checkpoint:
        pytest.skip("Set the real checkpoint path for processor/metadata validation")
    full_checkpoint_config(checkpoint)
    cases = make_full_cases(checkpoint)
    assert [case["name"] for case in cases] == ["text_short", "text_long", "image", "changed_image"]
    assert cases[0]["multimodal"] == cases[1]["multimodal"] == {}
    assert len(cases[0]["input_ids"]) != len(cases[1]["input_ids"])
