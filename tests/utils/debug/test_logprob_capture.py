# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.utils.debug.logprob_capture import capture_logprob_batch, select_capture_indices


def make_batch():
    def jagged(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    return TensorDict(
        {
            "prompts": jagged([torch.tensor([10, 11]), torch.tensor([20]), torch.tensor([30, 31, 32, 33])]),
            "responses": jagged([torch.tensor([12, 13]), torch.tensor([21, 22, 23]), torch.tensor([34])]),
            "old_log_probs": jagged(
                [torch.tensor([-1.1, -2.0]), torch.tensor([-1.0, -8.0, -2.0]), torch.tensor([-1.4])]
            ),
            "rollout_log_probs": jagged(
                [torch.tensor([-1.0, -2.0]), torch.tensor([-1.0, -2.0, -2.0]), torch.tensor([-1.0])]
            ),
            "response_mask": jagged([torch.tensor([1, 1]), torch.tensor([1, 0, 1]), torch.tensor([1])]),
        },
        batch_size=[3],
    )


def test_capture_preserves_exact_tokens_and_causal_routes_without_mutation(tmp_path):
    batch = make_batch()
    before = {name: [row.clone() for row in value.unbind()] for name, value in batch.items()}
    # Include last native placeholder in one route input, as production TQ does.
    routes = [torch.arange(4 * 2 * 2).view(4, 2, 2).short(), torch.arange(4 * 2 * 2).view(4, 2, 2).short()]
    result = capture_logprob_batch(tmp_path, 1, ["a", "b", "c"], batch, [0, 2], routes, metadata={"fixture": True})
    path = tmp_path / "step-000001"
    samples = json.loads((path / "samples.json").read_text())
    assert samples[0]["input_ids"] == [10, 11, 12, 13]
    assert samples[1]["input_ids"] == [30, 31, 32, 33, 34]
    assert np.array_equal(np.load(path / "generation/routes-000.npy"), routes[0][:3].numpy())
    assert np.array_equal(np.load(path / "generation/routes-001.npy"), routes[1].numpy())
    assert samples[0]["generation_logprobs"] == [-1.0, -2.0]
    assert samples[0]["production_actor_logprobs"] == pytest.approx([-1.1, -2.0])
    assert result["batch_response_mean_abs"] == pytest.approx((0.05 + 0.0 + 0.4) / 3)
    manifest = json.loads((path / "capture.json").read_text())
    assert manifest["batch_token_mean_abs"] == pytest.approx(0.5 / 5)
    assert all(
        hashlib.sha256((path / name).read_bytes()).hexdigest() == digest for name, digest in manifest["files"].items()
    )
    assert (path / "capture-complete.json").exists()
    for name, parts in before.items():
        assert all(torch.equal(left, right) for left, right in zip(parts, batch[name].unbind(), strict=True))
    with pytest.raises(FileExistsError):
        capture_logprob_batch(tmp_path, 1, ["a", "b", "c"], batch, [0, 2], routes, metadata={})


def test_selection_uses_masked_errors_and_keeps_lengths():
    batch = make_batch()
    assert select_capture_indices(batch, 2) == [0, 2]
    batch["response_mask"].unbind()[1].zero_()
    assert select_capture_indices(batch, 16) == [0, 2]
    with pytest.raises(ValueError, match="positive"):
        select_capture_indices(batch, 0)


def test_capture_rejects_bad_route_lengths_before_publishing(tmp_path):
    with pytest.raises(ValueError, match="causal token"):
        capture_logprob_batch(
            tmp_path, 1, ["a", "b", "c"], make_batch(), [0], [torch.zeros(2, 2, 2, dtype=torch.int16)], metadata={}
        )
    assert not (tmp_path / "step-000001/capture-complete.json").exists()
