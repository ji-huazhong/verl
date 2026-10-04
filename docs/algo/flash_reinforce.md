# FlashREINFORCE

Last updated: 10/05/2026.

[FlashREINFORCE](https://yifanzhang-pro.github.io/FlashREINFORCE/FlashREINFORCE.pdf)
is a critic-free, single-rollout algorithm. This implementation adds its base
advantage estimator and policy objective to the existing verl training path.

## Objective

For a fresh batch of $B$ complete trajectories, compute outcome rewards and a
batch baseline **before** splitting into workers or microbatches:

$$R_i = \sum_t r_{i,t}, \qquad A_i = R_i - \frac{1}{B}\sum_j R_j.$$

There is no prompt-group baseline or standard-deviation normalization. Each row
must represent one complete trajectory, with one rollout per prompt. Observation
and padding tokens are excluded from the policy-token mask; entirely masked rows
are excluded from the baseline.

Let $p_{i,t}=\mu_i(a_{i,t}|h_{i,t})$ be the stored rollout probability and
$q_{i,t}=\pi_\theta(a_{i,t}|h_{i,t})$ the learner probability. The loss uses
`rollout_log_probs` as its denominator, even if recomputed `old_log_probs` are also
present. The sequence gate is

$$d_{i,t}=p_{i,t}\log\frac{p_{i,t}}{q_{i,t}}+
(1-p_{i,t})\log\frac{1-p_{i,t}}{1-q_{i,t}},\qquad
m_i=\mathbf{1}\left[\frac{1}{T_i}\sum_{t\in\mathcal A_i}d_{i,t}\leq\delta\right].$$

Here $\mathcal A_i$ contains policy tokens and $T_i=|\mathcal A_i|$. This sampled-action
Bernoulli KL is a proxy, not an estimate of the full-vocabulary KL. The objective is

$$\mathcal L=-\frac{1}{B}\sum_i\frac{m_i A_i}{T_i}
\sum_{t\in\mathcal A_i}\exp\left(\operatorname{clamp}
(\log q_{i,t}-\log p_{i,t},-30,30)\right).$$

As in Appendix A, log ratios are evaluated in float32 and differentiated directly.
Advantages, behavior probabilities, and gates are detached. Outside the clamp
interval the policy-loss gradient is zero. This guard is distinct from PPO ratio
clipping. A rejected sequence contributes zero while **the original $B$ and $T_i$
remain unchanged**. Existing `agg_loss` normalization handles microbatch gradient
accumulation and data parallelism.

## Core configuration

Merge these settings into an existing trainer configuration, retaining its model,
data, backend, and resource settings:

```yaml
algorithm:
  adv_estimator: flash_reinforce
  use_kl_in_reward: false
  rollout_correction:
    rollout_is: null
    rollout_rs: null
    bypass_mode: false

actor_rollout_ref:
  rollout:
    n: 1
    calculate_log_probs: true
  actor:
    policy_loss:
      loss_mode: flash_reinforce
      flash_reinforce_kl_threshold: 0.001
      flash_reinforce_neg_topq: 1.0
    loss_agg_mode: seq-mean-token-mean
    ppo_mini_batch_size: ${data.train_batch_size}
    ppo_epochs: 1
    use_kl_loss: false
    entropy_coeff: 0

critic:
  enable: false
```

Each fresh batch receives one full-batch optimizer step and is then discarded.
Microbatch accumulation is supported; repeated epochs and multiple optimizer
minibatches are rejected. External IS/rejection correction must be disabled because
the loss already applies token IS and sequence admission. Configure the advantage
estimator and policy loss together. The threshold is task dependent; `0.001` is a
default, not a claim to reproduce all paper experiments.

The actor reports `actor/flash_reinforce_reject_frac` and
`actor/flash_reinforce_kl`. These are local sequence-mean diagnostics, aggregated
with the existing policy metric pipeline; with uneven microbatches they are not
exact global sequence means.

This change does not introduce an asynchronous scheduler or experiment recipe.
For optional negative-token filtering (Appendix C), set
`policy_loss.flash_reinforce_neg_topq` in `[0, 1)` and `actor.calculate_entropy=true`.
For binary rewards, trajectories with outcome `R <= 0` retain exactly
`ceil(q * T_i)` highest-entropy policy tokens (ties follow token order). Successful
trajectories retain all tokens. This uses the uncentered `returns` from the
estimator; the original sequence gate and normalization masks are preserved.
Multi-turn trajectories
with masked observations within a single row are supported; splitting one rollout
across multiple training rows requires additional trajectory-level aggregation.

## Reference implementations

The [labs-molt implementation](https://github.com/NVIDIA-NeMo/labs-molt/blob/d651b7e97e19a70056f9412286f5eb3f9f5379d0/molt/models/loss.py)
composes its policy objective and IS gate; its
[advantage estimator](https://github.com/NVIDIA-NeMo/labs-molt/blob/d651b7e97e19a70056f9412286f5eb3f9f5379d0/molt/trainer/algorithm/advantage.py)
centers rewards without whitening. The public implementation uses detached IS
weights, so its numerical behavior outside the ratio clamp differs from the
paper's Appendix A implementation followed here.
[slime's policy loss](https://github.com/THUDM/slime/blob/8c17b676cb57af1d17ee4402e91e9209af84b60b/slime/backends/megatron_utils/loss.py)
also separates rejection from the original per-rollout reduction denominators.
