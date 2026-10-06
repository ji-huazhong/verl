# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging
import os

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def calculate_logprob_context_metrics(data, logprob_diff, response_mask, boundaries):
    """Bucket sampled-token error by the logical context that predicted it.

    A response token is scored by the preceding query. Use the full attention
    mask for context length, including tokens excluded by a response loss mask.
    Left prompt padding and right response padding never count as context.
    """
    if not boundaries or any(value < 1 for value in boundaries) or sorted(set(boundaries)) != list(boundaries):
        raise ValueError("Logprob context boundaries must be positive and strictly increasing")
    attention = data.batch.get("attention_mask")
    response_width = data.batch["responses"].shape[1]
    if attention is None or attention.shape[1] <= response_width:
        return {}
    attention = attention.to(device=logprob_diff.device).bool()
    context = attention[:, :-response_width].sum(-1, keepdim=True)
    context = context + attention[:, -response_width:].cumsum(-1) - 1
    metrics = {}
    lower = None
    for upper in [*boundaries, None]:
        selected = response_mask & attention[:, -response_width:]
        if lower is not None:
            selected &= context > lower
        if upper is not None:
            selected &= context <= upper
        label = f"le{upper}" if lower is None else (f"gt{lower}" if upper is None else f"gt{lower}_le{upper}")
        prefix = f"training/train_rollout_logprob_ctx_{label}"
        counts = selected.sum(-1)
        samples = counts > 0
        values = logprob_diff[selected]
        metrics[prefix + "_tokens"] = values.numel()
        metrics[prefix + "_samples"] = samples.sum().item()
        if values.numel():
            response_sums = torch.where(selected, logprob_diff, 0.0).sum(-1)
            metrics[prefix + "_abs_diff"] = (response_sums[samples] / counts[samples]).mean().item()
            metrics[prefix + "_token_abs_diff"] = values.mean().item()
            metrics[prefix + "_max_abs_diff"] = values.max().item()
            metrics[prefix + "_nonfinite_tokens"] = (~values.isfinite()).sum().item()
        lower = upper
    return metrics


def calculate_debug_metrics(data: DataProto) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()

    # check if there are any valid tokens before computing metrics
    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
            "training/train_rollout_logprob_abs_diff": float("nan"),
            "training/train_rollout_logprob_token_abs_diff": float("nan"),
            "training/train_rollout_logprob_max_abs_diff": float("nan"),
            "training/train_rollout_logprob_nonfinite_tokens": 0,
        }

    # Independent actor scoring before the PPO update, on sampled response tokens.
    # Average within each response, then over the batch, matching Miles' sample
    # weighting. Keep this separate from the historical exp(logprob) metrics.
    logprob_diff = torch.where(
        response_mask_bool, (actor_old_log_probs.float() - rollout_old_log_probs.float()).abs(), 0.0
    )
    response_counts = response_mask_bool.sum(dim=-1).clamp_min(1)
    valid_logprob_diff = logprob_diff[response_mask_bool]
    logprob_metrics = {
        "training/train_rollout_logprob_abs_diff": (logprob_diff.sum(dim=-1) / response_counts).mean().item(),
        "training/train_rollout_logprob_token_abs_diff": valid_logprob_diff.mean().item(),
        "training/train_rollout_logprob_max_abs_diff": valid_logprob_diff.max().item(),
        "training/train_rollout_logprob_nonfinite_tokens": (~valid_logprob_diff.isfinite()).sum().item(),
    }
    context_boundaries = os.environ.get("VERL_LOGPROB_CONTEXT_BUCKETS")
    if context_boundaries:
        logprob_metrics.update(
            calculate_logprob_context_metrics(
                data, logprob_diff, response_mask_bool, [int(value) for value in context_boundaries.split(":")]
            )
        )
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    return {
        **logprob_metrics,
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }
