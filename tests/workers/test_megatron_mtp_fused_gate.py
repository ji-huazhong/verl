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
"""Engine gate tests for the provisioned Megatron test environment (no CUDA work)."""

from types import SimpleNamespace

import pytest

from verl.models.mcore import model_forward_fused as mff
from verl.workers.engine.megatron.transformer_impl import MegatronEngine


@pytest.mark.parametrize(
    ("enabled", "remove_padding", "value_model", "mtp", "reject_last", "expected"),
    [
        (False, True, False, True, False, False),
        (True, False, False, True, False, False),
        (True, True, True, True, False, False),
        (True, True, False, False, True, True),
        (True, True, False, True, False, True),
        (True, True, False, True, True, False),
    ],
)
def test_all_chunks_are_checked_before_patching(
    monkeypatch, enabled, remove_padding, value_model, mtp, reject_last, expected
):
    chunks = [object(), object()]
    engine = SimpleNamespace(
        engine_config=SimpleNamespace(use_fused_kernels=enabled, use_remove_padding=remove_padding),
        model_config=SimpleNamespace(mtp=SimpleNamespace(enable=mtp)),
        is_value_model=value_model,
        module=chunks,
    )
    events = []

    def reason(model):
        events.append(("check", model))
        return "legacy forward" if reject_last and model is chunks[-1] else None

    monkeypatch.setattr(mff, "mtp_fused_forward_unavailable_reason", reason)
    monkeypatch.setattr(mff, "patch_fused_forward", lambda model: events.append(("patch", model)))
    MegatronEngine._maybe_enable_fused_kernels(engine)
    assert engine.engine_config.use_fused_kernels is expected
    patches = [(action, model) for action, model in events if action == "patch"]
    assert patches == ([("patch", model) for model in chunks] if expected else [])
    if expected and mtp:
        assert events == [("check", model) for model in chunks] + patches
