# Copyright 2026 Individual Contributor
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.v1 import trainer_base


class _StubTrainer(trainer_base.PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _Batch:
    keys = ["first", "second"]
    partition_id = "train"

    def __init__(self):
        self.extra_info = {}

    def __len__(self):
        return len(self.keys)


def _jagged(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


@pytest.mark.parametrize("context_buckets", [False, True])
@pytest.mark.parametrize("capture_enabled", [False, True])
def test_v1_logprob_context_uses_jagged_lengths_and_keeps_actor_outputs(
    monkeypatch, context_buckets, capture_enabled, tmp_path
):
    """Exercise selective TQ reads, causal output slicing and diagnostic padding together."""
    if context_buckets:
        monkeypatch.setenv("VERL_LOGPROB_CONTEXT_BUCKETS", "2048:2051")
    else:
        monkeypatch.delenv("VERL_LOGPROB_CONTEXT_BUCKETS", raising=False)
    if capture_enabled:
        monkeypatch.setenv("VERL_LOGPROB_CAPTURE_DIR", str(tmp_path / "capture"))
    else:
        monkeypatch.delenv("VERL_LOGPROB_CAPTURE_DIR", raising=False)
    batch = _Batch()
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.global_steps = 1
    trainer.config = OmegaConf.create(
        {
            "algorithm": {},
            "trainer": {"experiment_name": "fixture"},
            "actor_rollout_ref": {
                "rollout": {"calculate_log_probs": True, "temperature": 1.0},
                "model": {"path": "fixture"},
                "actor": {"loss_agg_mode": "token-mean", "loss_scale_factor": None},
            },
        }
    )
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.compute_log_prob.return_value = batch
    log_probs = [torch.zeros(2052), torch.zeros(2054)]
    log_probs[0][-5:-1] = torch.tensor([1.0, 20.0, 3.0, 4.0])
    log_probs[1][-3:-1] = torch.tensor([5.0, 6.0])
    data = TensorDict(
        {
            "prompts": _jagged([torch.zeros(2048, dtype=torch.long), torch.zeros(2052, dtype=torch.long)]),
            "responses": _jagged([torch.zeros(4, dtype=torch.long), torch.zeros(2, dtype=torch.long)]),
            "response_mask": _jagged([torch.tensor([1, 0, 1, 1]), torch.tensor([1, 1])]),
            "rollout_log_probs": _jagged([torch.zeros(4), torch.zeros(2)]),
            "log_probs": _jagged(log_probs),
            "entropy": _jagged([torch.ones(2052), torch.ones(2054)]),
        },
        batch_size=[2],
    )
    route_data = TensorDict(
        {
            "routed_experts": _jagged(
                [torch.zeros(2052, 2, 2, dtype=torch.int16), torch.zeros(2054, 2, 2, dtype=torch.int16)]
            )
        },
        batch_size=[2],
    )

    def fetch(**kwargs):
        if kwargs["select_fields"] == ["routed_experts"]:
            assert kwargs["keys"] == batch.keys
            return route_data.clone()
        return data.select(*kwargs["select_fields"]).clone()

    get = MagicMock(side_effect=fetch)
    put = MagicMock(return_value=batch)
    monkeypatch.setattr(trainer_base, "tq", SimpleNamespace(kv_batch_get=get, kv_batch_put=put))
    metrics = {}
    assert trainer._compute_old_log_prob(batch, metrics) is batch

    selected_fields = get.call_args_list[0].kwargs["select_fields"]
    assert ("prompts" in selected_fields) is (context_buckets or capture_enabled)
    assert (tmp_path / "capture/step-000001/capture-complete.json").exists() is capture_enabled
    assert get.call_count == (2 if capture_enabled else 1)
    written = put.call_args.kwargs["fields"]
    assert set(written.keys()) == {"old_log_probs", "entropy"}
    assert torch.equal(written["old_log_probs"].unbind()[0], torch.tensor([1.0, 20.0, 3.0, 4.0]))
    assert torch.equal(written["old_log_probs"].unbind()[1], torch.tensor([5.0, 6.0]))
    assert metrics["training/train_rollout_logprob_abs_diff"] == pytest.approx(49 / 12)
    assert metrics["training/train_rollout_logprob_token_abs_diff"] == pytest.approx(19 / 5)
    assert metrics["actor/entropy"] == 1.0
    if context_buckets:
        assert metrics["training/train_rollout_logprob_ctx_le2048_tokens"] == 1
        assert metrics["training/train_rollout_logprob_ctx_le2048_abs_diff"] == 1.0
        assert metrics["training/train_rollout_logprob_ctx_gt2048_le2051_tokens"] == 2
        assert metrics["training/train_rollout_logprob_ctx_gt2048_le2051_abs_diff"] == 3.5
        assert metrics["training/train_rollout_logprob_ctx_gt2051_tokens"] == 2
        assert metrics["training/train_rollout_logprob_ctx_gt2051_abs_diff"] == 5.5
    else:
        assert not any("train_rollout_logprob_ctx_" in key for key in metrics)
