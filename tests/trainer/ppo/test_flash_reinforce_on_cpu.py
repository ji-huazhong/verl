# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 verl contributors
#
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

"""FlashREINFORCE math, gradients, actor integration, and configuration on CPU."""

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import (
    compute_flash_reinforce_outcome_advantage,
    compute_policy_loss_flash_reinforce,
)
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.utils import tensordict_utils as tu
from verl.utils.config import validate_config
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.losses import ppo_loss


def _config(threshold=0.01, **kwargs):
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        use_dynamic_bsz=True,
        loss_agg_mode="seq-mean-token-mean",
        policy_loss=PolicyLossConfig(loss_mode="flash_reinforce", flash_reinforce_kl_threshold=threshold),
        **kwargs,
    )


def test_batch_baseline_ignores_groups_lengths_and_empty_rows():
    # An outcome may be attached to a non-policy slot (e.g. a final observation).
    rewards = torch.tensor([[0.0, 0.0, 3.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [9.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    batch = DataProto.from_dict({"token_level_rewards": rewards, "response_mask": mask})
    result = compute_advantage(batch, "flash_reinforce")
    expected = torch.tensor([[2.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(result.batch["advantages"], expected)
    torch.testing.assert_close(result.batch["returns"], rewards.sum(-1, keepdim=True) * mask)


@pytest.mark.parametrize("reward", [-2.0, 0.0, 1.0])
def test_equal_outcomes_have_zero_advantage(reward):
    rewards = torch.tensor([[reward, 0.0], [0.0, reward]], requires_grad=True)
    advantages, _ = compute_flash_reinforce_outcome_advantage(rewards, torch.ones_like(rewards))
    assert not advantages.requires_grad
    torch.testing.assert_close(advantages, torch.zeros_like(rewards))


def test_sequence_gate_and_original_denominators_match_analytic_gradient():
    # Sequence 0: KL(0.5||0.6)/2 < .015; tokenwise gating would drop its first token.
    # Sequence 1: KL(0.5||0.9)/2 > .015; its matching second token must also be dropped.
    # Sequence 2: shorter, but has the same sample weight.
    p = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    q = torch.tensor([[0.6, 0.5], [0.9, 0.5], [0.5, 0.5]])
    mask = torch.tensor([[1, 1], [1, 1], [1, 0]])
    logp = p.log().requires_grad_()
    logq = q.log().requires_grad_()
    advantages = torch.tensor([[2.0, 2.0], [-1.0, -1.0], [-1.0, 0.0]], requires_grad=True)
    loss, metrics = compute_policy_loss_flash_reinforce(logp, logq, advantages, mask, config=_config(0.015))
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(-(2.0 * 1.1 - 1.0) / 3))
    torch.testing.assert_close(logq.grad, torch.tensor([[-0.4, -1 / 3], [0.0, 0.0], [1 / 3, 0.0]]))
    assert logp.grad is None and advantages.grad is None
    assert metrics["actor/flash_reinforce_reject_frac"] == pytest.approx(1 / 3)
    bernoulli_kl = torch.distributions.kl_divergence(
        torch.distributions.Bernoulli(probs=p.double()), torch.distributions.Bernoulli(probs=q.double())
    )
    expected_kl = ((bernoulli_kl * mask).sum(-1) / mask.sum(-1)).mean()
    assert metrics["actor/flash_reinforce_kl"] == pytest.approx(expected_kl.item(), abs=1e-7)


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("micro_size", [1, 2])
def test_global_loss_and_gradient_are_invariant_to_partition(dp_size, micro_size):
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    behavior = torch.full((4, 3), -0.7)
    learner = torch.tensor([[-0.69, -0.7, -0.7], [-0.7, 0.0, 0.0], [-0.1, -0.1, 0.0], [-0.7, -0.7, -0.7]])
    advantages = torch.tensor([1.0, -1.0, 2.0, -2.0])[:, None] * mask
    whole_logp = learner.clone().requires_grad_()
    whole, _ = compute_policy_loss_flash_reinforce(behavior, whole_logp, advantages, mask, config=_config())
    whole.backward()

    sharded_logp = learner.clone().requires_grad_()
    config = _config(global_batch_info={"global_batch_size": 4, "dp_size": dp_size})
    # Sum accumulated microbatch losses and model DDP's mean gradient reduction.
    accumulated = sum(
        compute_policy_loss_flash_reinforce(
            behavior[i : i + micro_size],
            sharded_logp[i : i + micro_size],
            advantages[i : i + micro_size],
            mask[i : i + micro_size],
            config=config,
        )[0]
        / dp_size
        for i in range(0, 4, micro_size)
    )
    accumulated.backward()
    torch.testing.assert_close(accumulated, whole)
    torch.testing.assert_close(sharded_logp.grad, whole_logp.grad)


def test_masked_values_boundaries_and_empty_rows_are_safe():
    behavior = torch.tensor([[0.0, -1e-8, float("nan")], [float("nan")] * 3])
    learner = behavior.clone().requires_grad_()
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    advantages = torch.tensor([[1.0, 1.0, float("nan")], [float("nan")] * 3])
    loss, metrics = compute_policy_loss_flash_reinforce(behavior, learner, advantages, mask, config=_config(0.0))
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(learner.grad, torch.tensor([[-0.5, -0.5, 0.0], [0.0, 0.0, 0.0]]))
    assert metrics["actor/flash_reinforce_kl"] == 0.0

    empty = torch.full((2, 3), float("nan"), requires_grad=True)
    loss, metrics = compute_policy_loss_flash_reinforce(empty, empty, empty, torch.zeros_like(empty), config=_config())
    loss.backward()
    assert loss.item() == 0.0
    assert all(value == 0.0 for value in metrics.values())
    torch.testing.assert_close(empty.grad, torch.zeros_like(empty))


def test_all_rejected_sequences_have_zero_loss_and_gradient():
    learner = torch.full((2, 3), -0.1, requires_grad=True)
    loss, metrics = compute_policy_loss_flash_reinforce(
        torch.full_like(learner, -1.0), learner, torch.ones_like(learner), torch.ones_like(learner), config=_config()
    )
    loss.backward()
    assert loss.item() == 0.0
    assert metrics["actor/flash_reinforce_reject_frac"] == 1.0
    torch.testing.assert_close(learner.grad, torch.zeros_like(learner))


def test_ratio_clamp_has_zero_gradient_outside_interval():
    # Tiny action probabilities keep the KL gate open despite extreme log ratios.
    behavior = torch.tensor([[-100.0, -40.0, -60.0, -60.0]])
    learner = torch.tensor([[-40.0, -100.0, -59.0, -61.0]], requires_grad=True)
    loss, _ = compute_policy_loss_flash_reinforce(
        behavior, learner, torch.ones_like(learner), torch.ones_like(learner), config=_config()
    )
    loss.backward()
    expected = -torch.tensor([[0.0, 0.0, torch.e, 1 / torch.e]]) / 4
    torch.testing.assert_close(learner.grad, expected)


def test_actor_adapter_uses_rollout_probabilities_and_global_normalization():
    behavior = torch.full((2, 2), -0.7)
    learner = torch.full((2, 2), -0.69, requires_grad=True)
    mask = torch.tensor([[1, 1], [1, 0]])
    advantages = torch.tensor([[1.0, 1.0], [-1.0, 0.0]])
    data = TensorDict(
        {
            "prompts": torch.zeros(2, 1, dtype=torch.long),
            "responses": torch.zeros(2, 2, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "response_mask": mask,
            "old_log_probs": torch.full((2, 2), -10.0),
            "rollout_log_probs": behavior,
            "advantages": advantages,
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=2, global_batch_size=8, batch_num_tokens=12)
    # One prompt token per row: the final model output is not a response prediction.
    packed = torch.cat([torch.cat([row, row.new_zeros(1)]) for row in learner])
    actual, _ = ppo_loss(_config(), {"log_probs": packed}, data)
    actual_grad = torch.autograd.grad(actual, learner)[0]
    expected, _ = compute_policy_loss_flash_reinforce(
        behavior,
        learner,
        advantages,
        mask,
        config=_config(global_batch_info={"dp_size": 2, "global_batch_size": 8}),
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, learner)[0])
    # Recomputed anchors alone must never silently replace behavior probabilities.
    del data["rollout_log_probs"]
    with pytest.raises(ValueError, match="requires rollout_log_probs"):
        ppo_loss(_config(), {"log_probs": packed}, data)


def test_external_is_is_rejected():
    x = torch.full((1, 2), -1.0)
    with pytest.raises(ValueError, match="disable external"):
        compute_policy_loss_flash_reinforce(x, x, x, torch.ones_like(x), config=_config(), rollout_is_weights=x)


@pytest.mark.parametrize("mini_batch_size,epochs", [(4, 1), (2, 1), (4, 2)])
def test_worker_requires_one_optimizer_step(mini_batch_size, epochs):
    actor = SimpleNamespace(
        engine=SimpleNamespace(get_data_parallel_size=lambda: 2), train_mini_batch=Mock(return_value=None)
    )
    worker = SimpleNamespace(config=SimpleNamespace(actor=_config()), actor=actor)
    data = TensorDict({"response_mask": torch.ones(2, 3)}, batch_size=[2])
    tu.assign_non_tensor(data, mini_batch_size=mini_batch_size, epochs=epochs)
    update_actor = inspect.unwrap(ActorRolloutRefWorker.update_actor)
    if mini_batch_size == 4 and epochs == 1:
        update_actor(worker, data)
        actor.train_mini_batch.assert_called_once_with(data=data)
    else:
        with pytest.raises(ValueError, match="one full-batch optimizer step"):
            update_actor(worker, data)
        actor.train_mini_batch.assert_not_called()


@pytest.fixture
def trainer_config():
    config_dir = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=flash_reinforce",
                "actor_rollout_ref.actor.policy_loss.loss_mode=flash_reinforce",
                "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "data.train_batch_size=8",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
                "critic.enable=false",
            ],
        )


def test_valid_trainer_config(trainer_config):
    validate_config(trainer_config, use_reference_policy=False, use_critic=False)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("algorithm.adv_estimator", "reinforce_plus_plus", "requires both"),
        ("actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla", "requires both"),
        ("actor_rollout_ref.rollout.calculate_log_probs", False, "calculate_log_probs"),
        ("actor_rollout_ref.rollout.n", 2, "rollout.n=1"),
        ("actor_rollout_ref.actor.ppo_epochs", 2, "ppo_epochs=1"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", 4, "one-pass"),
        ("actor_rollout_ref.actor.loss_agg_mode", "token-mean", "seq-mean-token-mean"),
        ("algorithm.rollout_correction.rollout_is", "token", "disable external"),
        ("algorithm.rollout_correction.rollout_rs", "sequence_k1", "disable external"),
        ("algorithm.rollout_correction.bypass_mode", True, "disable external"),
        ("algorithm.use_kl_in_reward", True, "no critic"),
        ("actor_rollout_ref.actor.use_kl_loss", True, "no critic"),
    ],
)
def test_invalid_trainer_config(trainer_config, key, value, match):
    OmegaConf.update(trainer_config, key, value)
    # Hydra wraps dataclass validation errors in InstantiationException.
    with pytest.raises((ValueError, InstantiationException), match=match):
        validate_config(trainer_config, use_reference_policy=False, use_critic=False)


@pytest.mark.parametrize("threshold", [-0.001, float("nan"), float("inf")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _config(threshold)
