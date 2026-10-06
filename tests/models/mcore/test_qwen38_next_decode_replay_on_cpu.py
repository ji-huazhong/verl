# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Select fixed production-token probes without including unscored positions."""

from examples.grpo_trainer.qwen3_8_next.production_decode_replay import select_probe_samples


def test_selection_uses_scored_short_responses_and_centers_worst_query():
    def sample(name, errors, length=32, mask=None):
        return dict(
            id=name,
            input_ids=list(range(length)),
            prompt_length=length - len(errors),
            response_length=len(errors),
            generation_logprobs=[-10] * len(errors),
            production_actor_logprobs=[x - 10 for x in errors],
            response_mask=[1] * len(errors) if mask is None else mask,
        )

    samples = [
        sample("a", [0] * 20 + [7]),
        sample("b", [6, 0]),
        sample("control", [0.01, 0.02]),
        sample("middle", [1, 2]),
        sample("long", [9, 9], length=2048),
        sample("masked", [9, 9], mask=[0, 1]),
    ]
    probes = select_probe_samples(samples)
    assert [s["id"] for s in probes] == ["a", "b", "control"]
    assert probes[0]["trace_token_start"] == 22
    assert "trace_token_start" not in samples[0]
