# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from examples.grpo_trainer.qwen3_8_next.backend_trace import BackendPrefixTrace
from examples.grpo_trainer.qwen3_8_next.real_prompt_trace import (
    assess_trace_repeats,
    make_trace_views,
    response_logprobs,
    reuse_completed_vllm_reference,
    reuse_real_samples,
    write_json,
)


def test_response_scores_include_first_generated_token_and_exclude_prompt():
    sample = dict(prompt_length=3, response_length=2)
    assert response_logprobs(sample, [-90, -80, -1, -2]) == [-1, -2]
    with pytest.raises(ValueError, match="coordinates"):
        response_logprobs(sample, [-90, -80, -1])


def test_trace_views_preserve_real_tokens_and_add_budget_window_only_when_reached():
    samples = [
        dict(id="short", input_ids=[7] * 10, prompt_length=4, response_length=6),
        dict(id="long", input_ids=[8] * 2100, prompt_length=100, response_length=2000),
    ]
    views = make_trace_views(samples)
    assert [row["trace_token_start"] for row in views] == [0, 3, 0, 99, 2032]
    assert len({row["id"] for row in views}) == len(views)
    for row in views:
        original = next(sample for sample in samples if sample["id"] == row["sample_id"])
        assert row["input_ids"] is original["input_ids"]


def test_reuse_preserves_real_sample_bytes_and_route_coordinates(tmp_path):
    import numpy as np

    source, output = tmp_path / "source", tmp_path / "output"
    (source / "generation").mkdir(parents=True)
    output.mkdir()
    selected = [dict(id="real", prompt_ids=[1, 2])]
    samples = [
        dict(
            selected[0],
            input_ids=[1, 2, 3, 4],
            prompt_length=2,
            response_length=2,
            generation_logprobs=[-1.0, -2.0],
            generation_routes="routes-000.npy",
        )
    ]
    for name, data in [("selected-prompts.json", selected), ("samples.json", samples)]:
        (source / name).write_text(json.dumps(data) + "\n")
    np.save(source / "generation/routes-000.npy", np.ones((3, 48, 10), dtype=np.int16))
    assert reuse_real_samples(source, output, 1) == (selected, samples)
    for name in ["samples.json", "selected-prompts.json", "generation/routes-000.npy"]:
        assert (output / name).read_bytes() == (source / name).read_bytes()
    assert json.loads((output / "sample-reuse.json").read_text())["regenerated"] is False
    write_json(source / "status.json", dict(complete=True))
    write_json(source / "report.json", dict(complete=True, controls_valid=True))
    write_json(source / "fixed-prompts.json", make_trace_views(samples))
    reused = tmp_path / "completed-reference"
    reused.mkdir()
    assert reuse_completed_vllm_reference(source, reused, 1) == (selected, samples)
    assert (reused / "fixed-prompts.json").read_bytes() == (source / "fixed-prompts.json").read_bytes()
    write_json(source / "report.json", dict(complete=True, controls_valid=False))
    with pytest.raises(ValueError, match="valid completed forward controls"):
        reuse_completed_vllm_reference(source, tmp_path / "failed-reference", 1)
    samples[0]["prompt_length"] = 1
    (source / "samples.json").write_text(json.dumps(samples))
    with pytest.raises(ValueError, match="response coordinates"):
        reuse_real_samples(source, tmp_path / "invalid", 1)


@pytest.mark.parametrize("external_reference", [False, True])
@pytest.mark.parametrize("changed_query,repeatable", [(4, True), (3, False)])
def test_activation_control_excludes_unsampled_query_and_detects_hidden_drift(
    tmp_path, changed_query, repeatable, external_reference
):
    views = make_trace_views([dict(id="sample", input_ids=[1, 2, 3, 4, 5], prompt_length=3, response_length=2)])
    (tmp_path / "fixed-prompts.json").write_text(json.dumps(views))
    reference = tmp_path / "external-vllm" if external_reference else tmp_path
    for backend, directory in [
        ("megatron", tmp_path / "megatron-control/activations"),
        ("vllm", reference / "vllm-control"),
    ]:
        for phase in ["traced0", "traced1"]:
            trace = BackendPrefixTrace(directory / phase, backend, 0, 0, 1)
            for view in views:
                trace.start(view, 5, "config")
                values = torch.ones(5, 2)
                if phase == "traced1" and backend == "vllm":
                    values[changed_query] += 1
                for stage in ["attn_hc/input", "attn_hc/mixed", "attention/output", "mlp/output"]:
                    trace.capture("layers/00/" + stage, values)
                trace.finish()
    report = assess_trace_repeats(tmp_path, reference)
    assert report["all_windows_repeatable"] is repeatable
    assert [row["compared_queries"] for row in report["windows"]] == [4, 2]
    for row in report["windows"]:
        assert row["megatron"]["max_abs"] == 0
        assert row["vllm"]["max_abs"] == (0 if repeatable else 1)


def test_reuse_production_capture_checks_hashes_step_and_original_scores(tmp_path):
    from tensordict import TensorDict

    from examples.grpo_trainer.qwen3_8_next.real_prompt_trace import reuse_production_capture
    from verl.utils.debug.logprob_capture import capture_logprob_batch

    def nested(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    data = TensorDict(
        {
            "prompts": nested([torch.tensor([1, 2])]),
            "responses": nested([torch.tensor([3, 4])]),
            "old_log_probs": nested([torch.tensor([-1.1, -2.1])]),
            "rollout_log_probs": nested([torch.tensor([-1.0, -2.0])]),
            "response_mask": nested([torch.tensor([1, 0])]),
        },
        batch_size=[1],
    )
    route = torch.arange(10, dtype=torch.int16).expand(4, 48, 10)
    capture_logprob_batch(
        tmp_path / "source",
        1,
        ["original-production-key"],
        data,
        [0],
        [route],
        metadata=dict(before_policy_update=True, temperature=1.0, model_path="/model/pinned-revision"),
    )
    source = tmp_path / "source/step-000001"
    output = tmp_path / "replay"
    output.mkdir()
    _, samples = reuse_production_capture(source, output, 1)
    assert samples[0]["input_ids"] == [1, 2, 3, 4]
    assert samples[0]["response_mask"] == [True, False]
    assert samples[0]["production_actor_logprobs"] == pytest.approx([-1.1, -2.1])
    assert (output / "samples.json").read_bytes() == (source / "samples.json").read_bytes()
    marker = source / "capture-complete.json"
    marker.unlink()
    with pytest.raises(FileNotFoundError):
        reuse_production_capture(source, tmp_path / "incomplete", 1)
    marker.write_text(json.dumps(dict(complete=True, captured_samples=1)))
    (source / "samples.json").write_text("[]")
    with pytest.raises(ValueError, match="hash"):
        reuse_production_capture(source, tmp_path / "corrupt", 1)
