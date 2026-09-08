# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from verl.utils.qat.int4 import fake_quant_int4_ste
from verl.utils.qat.int4_profile import current_int4_qat_profile, int4_qat_profile


def test_disabled_profile_does_not_touch_cuda(monkeypatch):
    monkeypatch.delenv("VERL_INT4_QAT_TRAIN_PROFILE", raising=False)

    def unexpected(*args, **kwargs):
        raise AssertionError("disabled profiler touched CUDA")

    monkeypatch.setattr(torch.cuda, "Event", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    weight = torch.randn(2, 128, requires_grad=True)
    with int4_qat_profile("train", 1, enabled=True) as profile:
        assert profile is None
        fake_quant_int4_ste(weight).sum().backward()
    assert current_int4_qat_profile() is None
    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_profile_distinguishes_schedule_forward_and_recompute(monkeypatch, capsys):
    monkeypatch.setenv("VERL_INT4_QAT_TRAIN_PROFILE", "1")
    weight = torch.randn(2, 128, requires_grad=True)
    with int4_qat_profile("train", 2, enabled=True) as profile:
        forward = profile.wrap_forward_step(lambda: fake_quant_int4_ste(weight))
        forward().sum().backward()
        forward()
        fake_quant_int4_ste(weight)
    record = json.loads(capsys.readouterr().err.split("INT4_QAT_TRAIN_PROFILE ", 1)[1])
    counts = {group["phase"]: group["calls"] for group in record["groups"]}
    assert counts == {"forward": 2, "backward_or_recompute": 1}
    assert sum(group["output_bytes"] for group in record["groups"]) == 3 * weight.numel() * weight.element_size()
    assert all(group["sampled_calls"] == 0 for group in record["groups"])
    assert record["complete"] is True
    assert record["num_microbatches"] == 2
    assert current_int4_qat_profile() is None
    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_profile_restores_context_after_exception(monkeypatch, capsys):
    monkeypatch.setenv("VERL_INT4_QAT_TRAIN_PROFILE", "1")
    with pytest.raises(RuntimeError, match="intentional"):
        with int4_qat_profile("logprob", 1, enabled=True):
            fake_quant_int4_ste(torch.randn(2, 128))
            raise RuntimeError("intentional")
    assert current_int4_qat_profile() is None
    record = json.loads(capsys.readouterr().err.split("INT4_QAT_TRAIN_PROFILE ", 1)[1])
    assert record["complete"] is False
    assert record["groups"][0]["phase"] == "forward"


def test_profile_schedule_limit_disables_later_schedules(monkeypatch, capsys):
    monkeypatch.setenv("VERL_INT4_QAT_TRAIN_PROFILE", "1")
    monkeypatch.setenv("VERL_INT4_QAT_PROFILE_MAX_SCHEDULES", "2")
    monkeypatch.setattr("verl.utils.qat.int4_profile._profiled_schedules", 0)
    for index in range(4):
        with int4_qat_profile("logprob", 1, enabled=True) as profile:
            assert (profile is None) == (index >= 2)
            fake_quant_int4_ste(torch.randn(2, 128))
    assert capsys.readouterr().err.count("INT4_QAT_TRAIN_PROFILE ") == 2
    assert current_int4_qat_profile() is None


def test_profile_counts_calls_from_autograd_style_worker_threads(monkeypatch):
    monkeypatch.setenv("VERL_INT4_QAT_TRAIN_PROFILE", "1")
    weight = torch.randn(2, 128)
    with int4_qat_profile("train", 1, enabled=True) as profile:
        profile.wrap_forward_step(lambda: fake_quant_int4_ste(weight))()
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: fake_quant_int4_ste(weight), range(10)))
    counts = {group["phase"]: group["calls"] for group in profile.report(True)["groups"]}
    assert counts == {"forward": 1, "backward_or_recompute": 10}
    assert current_int4_qat_profile() is None


def test_profile_rejects_overlapping_schedules(monkeypatch):
    monkeypatch.setenv("VERL_INT4_QAT_TRAIN_PROFILE", "1")
    with int4_qat_profile("train", 1, enabled=True) as outer:
        with pytest.raises(RuntimeError, match="one active Megatron schedule"):
            with int4_qat_profile("train", 1, enabled=True):
                pass
        assert current_int4_qat_profile() is outer
    assert current_int4_qat_profile() is None


def test_opt_in_profile_survives_framework_logger_filters(monkeypatch, capsys):
    import logging

    monkeypatch.setenv("VERL_INT4_QAT_TRAIN_PROFILE", "1")
    monkeypatch.setattr(logging.root.manager, "disable", logging.CRITICAL)
    with int4_qat_profile("logprob", 1, enabled=True):
        fake_quant_int4_ste(torch.randn(2, 128))
    assert "INT4_QAT_TRAIN_PROFILE " in capsys.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autograd worker thread")
def test_cuda_profile_counts_reentrant_checkpoint_backward(monkeypatch):
    from torch.utils.checkpoint import checkpoint

    monkeypatch.setenv("VERL_INT4_QAT_TRAIN_PROFILE", "1")
    weight = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with int4_qat_profile("train", 1, enabled=True) as profile:
        forward = profile.wrap_forward_step(lambda: checkpoint(fake_quant_int4_ste, weight, use_reentrant=True))
        forward().sum().backward()
    counts = {group["phase"]: group["calls"] for group in profile.report(True)["groups"]}
    assert counts == {"forward": 1, "backward_or_recompute": 1}
    assert torch.equal(weight.grad, torch.ones_like(weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA events and Triton")
def test_cuda_profile_bounds_samples_and_preserves_results(monkeypatch):
    monkeypatch.setenv("VERL_INT4_QAT_TRAIN_PROFILE", "1")
    monkeypatch.setenv("VERL_INT4_QAT_PROFILE_SAMPLE_EVERY", "2")
    monkeypatch.setenv("VERL_INT4_QAT_PROFILE_MAX_SAMPLES", "3")
    weight = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
    expected = fake_quant_int4_ste(weight)
    with int4_qat_profile("logprob", 1, enabled=True) as profile:
        for _ in range(10):
            assert torch.equal(fake_quant_int4_ste(weight), expected)
    record = profile.report(complete=True)
    assert record["groups"][0]["calls"] == 10
    assert record["groups"][0]["sampled_calls"] == 3
    assert record["groups"][0]["sampled_cuda_span_ms"] >= 0
