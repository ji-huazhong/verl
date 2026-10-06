# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Prove audit hashes cover every logical byte and boundary records are exact."""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "qwen38_runtime_audit", Path(__file__).resolve().parents[3] / "verl/models/mcore/qwen3_8_next/runtime_audit.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
audit_server_response = _module.audit_server_response
compare_fingerprints = _module.compare_fingerprints
model_fingerprints = _module.model_fingerprints
tensor_sha256 = _module.tensor_sha256


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
def test_chunk_hash_covers_transposed_strided_and_empty_tensors(dtype):
    value = torch.arange(6 * 7 * 8).reshape(6, 7, 8).to(dtype)
    for tensor in (value, value.transpose(0, 2), value[:, ::2, ::2], value[:0], value[:1], value[0, 0, 0]):
        expected = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        assert tensor_sha256(tensor, chunk_bytes=16) == hashlib.sha256(expected).hexdigest()


def test_hash_audit_detects_interior_parameter_change_and_separates_buffers():
    model = torch.nn.Linear(17, 9, bias=False)
    model.register_buffer("state", torch.zeros(4, dtype=torch.int64))
    initial = model_fingerprints(model, chunk_bytes=16)
    assert compare_fingerprints(initial, model_fingerprints(model)) == dict(missing=[], added=[], changed={})
    with torch.no_grad():
        model.weight[4, 8] += 1
        model.state[2] = 3
    differences = compare_fingerprints(initial, model_fingerprints(model, chunk_bytes=16))
    assert set(differences["changed"]) == {"parameter:weight", "buffer:state"}


def test_server_audit_is_opt_in_exact_and_immutable(tmp_path, monkeypatch):
    monkeypatch.delenv("VERL_QWEN38_PRODUCTION_AUDIT_DIR", raising=False)
    audit_server_response(
        replica=0, request_id="id", prompt_ids=None, final_res=None, sampling_params=None, global_steps=0
    )
    monkeypatch.setenv("VERL_QWEN38_PRODUCTION_AUDIT_DIR", str(tmp_path))
    result = SimpleNamespace(
        prompt_token_ids=[2, 3],
        num_cached_tokens=0,
        outputs=[
            SimpleNamespace(
                token_ids=[4, 5], logprobs=[{4: SimpleNamespace(logprob=-1.5)}, {5: SimpleNamespace(logprob=-2.0)}]
            )
        ],
    )
    params = SimpleNamespace(temperature=1.0, top_k=-1, top_p=1.0, logprobs=0)
    kwargs = dict(
        replica=2,
        request_id="test-exact-boundary",
        prompt_ids=[2, 3],
        final_res=result,
        sampling_params=params,
        global_steps=0,
    )
    audit_server_response(**kwargs)
    path = next(tmp_path.rglob("*.json"))
    saved = path.read_bytes()
    row = json.loads(saved)
    assert row["input_ids"] == row["engine_prompt_ids"] == [2, 3]
    assert row["response_ids"] == [4, 5] and row["logprobs"] == [-1.5, -2.0]
    assert row["sampling"]["temperature"] == 1.0
    with pytest.raises(FileExistsError):
        audit_server_response(**kwargs)
    assert path.read_bytes() == saved


def test_actual_extension_audits_after_native_load_and_only_first_refit(monkeypatch):
    events = []

    class Base:
        def monkey_patch_model(self, vocab_size, banned_token_ids):
            events.append(("native-mask", vocab_size, banned_token_ids))

        def update_weights_from_ipc(self, **kwargs):
            events.append(("native-refit", kwargs))

    stub = ModuleType("qwen38_audit_test.runtime_audit")
    stub.audit_worker_weights = lambda worker, stage: events.append(("audit", stage))
    stub.audit_worker_runtime = lambda worker: None
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    trace_stub = ModuleType("qwen38_audit_test.production_trace")
    trace_stub.install_vllm_production_trace = lambda worker: None
    monkeypatch.setitem(sys.modules, trace_stub.__name__, trace_stub)
    path = Path(__file__).resolve().parents[3] / "verl/models/mcore/qwen3_8_next/vllm_worker_extension.py"
    definition = next(
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", "") == "Qwen38QsaCanonicalOrderWorkerExtension"
    )
    scope = dict(
        __name__="qwen38_audit_test.extension",
        __package__="qwen38_audit_test",
        vLLMColocateWorkerExtension=Base,
        apply_qsa_canonical_order=lambda: None,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), scope)
    worker = scope[definition.name]()
    worker.monkey_patch_model(248077, [248056, 248057])
    worker.update_weights_from_ipc(use_shm=True)
    worker.update_weights_from_ipc(use_shm=False)
    assert events == [
        ("native-mask", 248077, [248056, 248057]),
        ("audit", "hf-loaded"),
        ("native-refit", dict(peft_config=None, base_sync_done=False, use_shm=True)),
        ("audit", "first-refit"),
        ("native-refit", dict(peft_config=None, base_sync_done=False, use_shm=False)),
    ]
