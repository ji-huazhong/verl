# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from examples.grpo_trainer.qwen3_8_next.packing_ab import (
    PHASES,
    assess_packing,
    mrope_positions,
    unroll_document_scores,
)


def test_mrope_resets_at_each_distinct_document():
    ids = torch.nested.as_nested_tensor([torch.tensor([11, 12, 13]), torch.tensor([21, 22])], layout=torch.jagged)
    positions = mrope_positions(ids)
    assert positions.dim() == 3 and positions.size(1) == 4
    assert positions.offsets() is ids.offsets()
    assert positions[0].tolist() == [[0, 1, 2]] * 4
    assert positions[1].tolist() == [[0, 1]] * 4


def test_scoring_drops_each_document_last_query_without_crossing_boundary():
    assert unroll_document_scores([-1, -2, -99, -3, -88], [3, 2]) == [[-1, -2], [-3]]
    with pytest.raises(ValueError, match="boundaries"):
        unroll_document_scores([-1, -2, -3], [3, 2])


def test_assessment_rejects_drift_and_counts_each_response_equally(tmp_path):
    output, reference = tmp_path / "out", tmp_path / "ref"
    (output / "packing-control").mkdir(parents=True)
    (reference / "vllm-control").mkdir(parents=True)
    samples = [
        dict(id="a", prompt_length=2, response_length=1, generation_logprobs=[-1.0]),
        dict(id="b", prompt_length=1, response_length=2, generation_logprobs=[-2.0, -3.0]),
    ]
    (output / "samples.json").write_text(json.dumps(samples))
    (reference / "vllm-control/report.json").write_text(
        json.dumps(dict(repeat_max_abs={"x": 0}, config_sha256="fixture"))
    )
    base = {
        "a": dict(log_probs=[-10.0, -1.0], fp32_log_probs=[-10.0, -1.0]),
        "b": dict(log_probs=[-2.0, -3.0], fp32_log_probs=[-2.0, -3.0]),
    }
    for phase in PHASES:
        value = json.loads(json.dumps(base))
        if phase.startswith("packed"):
            value["a"]["log_probs"][-1] += 0.25
        (output / "packing-control" / f"{phase}.json").write_text(json.dumps(value))
    audit = dict(
        config_sha256="fixture",
        parameter_versions_unchanged=True,
        layouts=[],
        partitions=[[0, 1]],
        phases={phase: (48 if phase.startswith("packed") else 96) for phase in PHASES},
    )
    for rank in range(8):
        (output / f"packing-audit-rank-{rank:02d}.json").write_text(json.dumps(audit))
    assess_packing(output, reference)
    report = json.loads((output / "report.json").read_text())
    assert report["summaries"]["packed_vs_single"]["response_mean_abs"] == 0.125
    assert report["summaries"]["packed_vs_single"]["token_mean_abs"] == pytest.approx(0.25 / 3)
    # Replay a production capture with a masked token carrying an intentionally huge difference.
    samples[0]["production_actor_logprobs"] = [-1.0]
    samples[0]["response_mask"] = [True]
    samples[1]["production_actor_logprobs"] = [100.0, -3.0]
    samples[1]["response_mask"] = [False, True]
    (output / "samples.json").write_text(json.dumps(samples))
    (output / "production-capture.json").write_text(
        json.dumps(dict(complete=True, step=1, metadata=dict(before_policy_update=True)))
    )
    assess_packing(output)
    report = json.loads((output / "report.json").read_text())
    assert report["production_capture_replayed"] and not report["rollout_repeat_controls_available"]
    assert report["summaries"]["production_actor_vs_generation"]["max_abs"] == 0
    assert report["summaries"]["packed0_vs_production_actor"]["response_mean_abs"] == 0.125
    assert report["summaries"]["packed0_vs_production_actor"]["token_mean_abs"] == 0.125
    samples[1]["prompt_length"] = 2051
    (output / "samples.json").write_text(json.dumps(samples))
    for phase in PHASES:
        path = output / "packing-control" / f"{phase}.json"
        result = json.loads(path.read_text())
        result["b"] = {key: [0.0] * 2050 + values for key, values in result["b"].items()}
        path.write_text(json.dumps(result))
    assess_packing(output)
    report = json.loads((output / "report.json").read_text())
    assert report["summaries"]["production_actor_vs_generation_ctx_gt2051"]["scored_tokens"] == 1
    assert report["summaries"]["production_actor_vs_generation_ctx_gt2051"]["max_abs"] == 0
    assert report["summaries"]["packed0_vs_production_actor_ctx_le2051"]["response_mean_abs"] == 0.25
    (output / "packing-control/manual1.json").write_text("{}")
    with pytest.raises(AssertionError, match="changed after packing"):
        assess_packing(output, reference)
