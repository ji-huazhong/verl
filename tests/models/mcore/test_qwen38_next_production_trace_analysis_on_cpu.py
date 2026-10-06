# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Do not confuse a routing/coordinate mismatch with an arithmetic difference."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[3] / "examples/grpo_trainer/qwen3_8_next/production_trace_analysis.py"
_spec = importlib.util.spec_from_file_location("production_trace_analysis", _path)
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def fixture():
    positions = [1, 2]
    actor = {
        name: {position: torch.tensor([1.0, 2.0]) for position in positions} for name in analysis.required_stages(2)
    }
    rollout = copy.deepcopy(actor)
    actor["log_probs"] = {1: torch.tensor([-1.0]), 2: torch.tensor([-4.0])}
    for layer in range(48):
        actor[f"layers/{layer:02d}/router/selected"] = {
            position: torch.tensor([False, True, True]) for position in positions
        }
    response = dict(
        request_id="r", prompt_ids=[10, 20], input_ids=[10, 20, 30, 40], response_ids=[30, 40], logprobs=[-1.0, -2.0]
    )
    routes = torch.tensor([2, 1]).expand(3, 48, 2).clone()
    return actor, rollout, response, routes


def test_first_divergence_and_worst_token_use_causal_query_not_response_index():
    actor, rollout, response, routes = fixture()
    rollout["layers/03/attention/output"][2][0] = 2.0
    result = analysis.compare_stages(actor, rollout, response, routes, 512, 2)
    assert result["worst_response_index"] == 1
    assert result["worst_query_position"] == 2
    assert result["max_abs_logprob"] == 2.0
    assert result["first_nonzero_boundary_at_worst_query"] == "layers/03/attention/output"
    assert result["route_control_passed"] and result["route_checks"] == 96


def test_route_disagreement_is_reported_even_when_activations_match():
    actor, rollout, response, routes = fixture()
    routes[2, 3] = torch.tensor([0, 1])
    result = analysis.compare_stages(actor, rollout, response, routes, 512, 2)
    assert not result["route_control_passed"]
    assert result["route_mismatches"] == [dict(layer=3, query_position=2, actor=[1, 2], rollout=[0, 1])]


def test_incomplete_tp_or_query_coverage_is_not_a_success():
    actor, rollout, response, routes = fixture()
    del rollout["layers/00/gdn/projection/ba/tp-01"][2]
    with pytest.raises(ValueError, match="Missing causal queries"):
        analysis.compare_stages(actor, rollout, response, routes, 512, 2)


def test_sp_merge_verifies_token_identity_and_conflicting_duplicates():
    record = dict(stages={"layer": [dict(positions=[1], token_ids=[20], value=torch.tensor([[1.0, 2.0]]))]})
    assert analysis.merge_stages([record, record], [10, 20, 30])["layer"][1].tolist() == [1.0, 2.0]
    wrong = copy.deepcopy(record)
    wrong["stages"]["layer"][0]["value"][0, 0] = 2.0
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        analysis.merge_stages([record, wrong], [10, 20, 30])
    wrong["stages"]["layer"][0]["token_ids"] = [99]
    with pytest.raises(ValueError, match="token identity"):
        analysis.merge_stages([wrong], [10, 20, 30])


def test_native_v2_internal_id_maps_to_external_response_without_prefix_collisions(tmp_path):
    response = dict(request_id="request", replica=1, prompt_ids=[10, 20])
    for rank in range(2):
        path = tmp_path / "vllm" / "replica-001" / f"rank-{rank:03d}" / "hash" / "complete.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(identity="request-0123abcd", prompt_ids=[10, 20], tp_rank=rank, tp_size=2)))
    assert len(analysis.rollout_manifests_for_response(tmp_path, response)) == 2
    path = tmp_path / "vllm" / "replica-001" / "rank-000" / "second" / "complete.json"
    path.parent.mkdir()
    path.write_text(json.dumps(dict(identity="request-more-0123abcd", prompt_ids=[10, 20], tp_rank=0, tp_size=2)))
    assert len(analysis.rollout_manifests_for_response(tmp_path, response)) == 2
    path.write_text(json.dumps(dict(identity="request-1234abcd", prompt_ids=[10, 20], tp_rank=0, tp_size=2)))
    with pytest.raises(ValueError, match="ambiguous internal"):
        analysis.rollout_manifests_for_response(tmp_path, response)
