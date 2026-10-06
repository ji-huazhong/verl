# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Check the installed V2 replay kernel and forced-token logprobs before loading weights."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def run(output):
    import torch
    import vllm
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
    from vllm.v1.worker.gpu.sample.trace_replay import TraceReplayState

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    request = SimpleNamespace(
        max_num_reqs=2,
        max_model_len=32,
        device=torch.device("cuda"),
        total_len=SimpleNamespace(gpu=i32([2, 4])),
        prompt_len=SimpleNamespace(gpu=i32([2, 4])),
    )
    state = TraceReplayState(request)
    trajectories = [[5, 3], [2, 4]]
    for slot, tokens in enumerate(trajectories):
        state.add_request(slot, SamplingParams(trace_decode_token_ids=tokens, max_tokens=len(tokens)))
    state.apply_staged_writes()
    logits = torch.tensor([[1.0, 0.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]] * 2, device="cuda")
    original = logits.clone()
    scores = []
    for step, order in enumerate(([1, 0], [0, 1])):
        request.total_len.gpu.copy_(request.prompt_len.gpu + step)
        sampled = logits.argmax(dim=-1)
        state.apply_trace(sampled, i32(order))
        assert sampled.tolist() == [trajectories[slot][step] for slot in order]
        result = compute_topk_scores(logits, 0, sampled)
        expected = logits.log_softmax(dim=-1).gather(-1, sampled[:, None])
        torch.testing.assert_close(result.logprobs, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(logits, original, rtol=0, atol=0)
        assert (result.logprobs < -1).all(), "Forced token probabilities must retain the real distribution"
        scores.extend(result.logprobs.flatten().tolist())
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()
    sampled = logits.argmax(dim=-1)
    state.apply_trace(sampled, i32([1, 0]))
    assert sampled.tolist() == [0, 3], "A reused request slot retained the previous trace"
    report = dict(
        complete=True,
        vllm_version=vllm.__version__,
        logits_unchanged=True,
        reordered_slots=True,
        cleared_slot=True,
        forced_token_logprobs=scores,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("QWEN38_REAL_PROMPT_DECODE_SMOKE", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
