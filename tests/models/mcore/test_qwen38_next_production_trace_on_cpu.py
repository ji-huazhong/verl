# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Check causal coordinates across reordered requests, THD padding and SP."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_path = Path(__file__).resolve().parents[3] / "verl/models/mcore/qwen3_8_next/production_trace.py"
_spec = importlib.util.spec_from_file_location("qwen38_production_trace", _path)
trace_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trace_module)


def test_selected_causal_rows_keep_request_identity_and_own_storage(tmp_path):
    trace = trace_module.ProductionTrace({"prompts": [[1, 2], [3, 4]], "response_queries": 2}, tmp_path, "vllm", 0, 8)
    trace.register("b", [3, 4])
    trace.register("a", [1, 2])
    trace.total_rows = 6
    trace.select_rows("b", 0, [0, 1, 2], [3, 4, 8])
    trace.select_rows("a", 3, [0, 1, 2], [1, 2, 9])
    values = torch.arange(12).reshape(6, 2).to(torch.bfloat16)
    before = values.clone()
    trace.capture("layer", values)
    assert torch.equal(values, before)
    values.fill_(100)
    trace.flush()
    assert trace.bytes == 0
    records = [torch.load(p, weights_only=True) for p in tmp_path.rglob("activations.pt")]
    records = {r["metadata"]["identity"]: r["stages"]["layer"][0] for r in records}
    assert records["b"]["positions"] == [1, 2]
    assert records["b"]["token_ids"] == [4, 8]
    assert torch.equal(records["b"]["value"], before[1:3])
    assert torch.equal(records["a"]["value"], before[4:6])


@pytest.mark.parametrize("rank", [0, 1])
def test_sequence_parallel_selects_only_owned_physical_rows(tmp_path, rank):
    trace = trace_module.ProductionTrace({"prompts": [[1, 2]], "response_queries": 4}, tmp_path, "megatron", rank, 2)
    trace.register("doc", [1, 2])
    trace.total_rows = 8
    trace.select_rows("doc", 2, [0, 1, 2, 3], [1, 2, 9, 8])
    full = torch.arange(16).reshape(8, 2)
    trace.capture("layer", full[rank * 4 : (rank + 1) * 4])
    piece = trace.records["doc"]["stages"]["layer"][0]
    assert piece["positions"] == ([1] if rank == 0 else [2, 3])
    assert torch.equal(piece["value"], full[[3]] if rank == 0 else full[[4, 5]])


def test_disabled_install_never_touches_worker(monkeypatch):
    monkeypatch.delenv(trace_module.PLAN_ENV, raising=False)
    trace_module.install_vllm_production_trace(None)
    forward = object()
    assert trace_module.wrap_megatron_production_forward(forward, None, None, enabled=False) is forward


@pytest.mark.parametrize("full_prefill", [False, True])
def test_vllm_adapter_uses_prepared_request_order_and_native_result(tmp_path, monkeypatch, full_prefill):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"prompts": [[1, 2], [3, 4]], "response_queries": 2}))
    monkeypatch.setenv(trace_module.PLAN_ENV, str(plan))
    monkeypatch.setenv(trace_module.OUTPUT_ENV, str(tmp_path / "output"))
    monkeypatch.setenv("VERL_REPLICA_RANK", "0")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    distributed = ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: 0
    distributed.get_tensor_model_parallel_world_size = lambda: 8
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setattr(trace_module.ProductionTrace, "attach_vllm", lambda self, model: None)
    native_result = object()

    class Runner:
        def prepare_inputs(self):
            return SimpleNamespace(
                req_ids=["b", "a"],
                query_start_loc_np=[0, 1, 2],
                num_tokens_after_padding=2,
                positions=torch.tensor([1, 2]),
                input_ids=torch.tensor([4, 9]),
            )

        def execute_model(self, scheduler):
            self.prepare_inputs()
            worker._qwen38_production_trace.capture("layer", torch.tensor([[10.0], [20.0]]))
            return native_result

    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True),
        model_runner=Runner(),
        get_model=lambda: None,
        sleep=lambda level: level,
    )
    original_prepare, original_execute, original_sleep = (
        worker.model_runner.prepare_inputs,
        worker.model_runner.execute_model,
        worker.sleep,
    )
    trace_module.install_vllm_production_trace(worker)
    requests = [
        SimpleNamespace(req_id=name, prompt_token_ids=ids)
        for name, ids in [("a", [1, 2, 9] if full_prefill else [1, 2]), ("b", [3, 4])]
    ]
    assert (
        worker.model_runner.execute_model(SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=requests))
        is native_result
    )
    assert worker.sleep(1) == 1
    records = [torch.load(p, weights_only=True) for p in (tmp_path / "output").rglob("activations.pt")]
    rows = {r["metadata"]["identity"]: r["stages"]["layer"][0] for r in records}
    assert rows["a"]["positions"] == [2] and rows["a"]["token_ids"] == [9]
    assert rows["b"]["positions"] == [1] and rows["b"]["token_ids"] == [4]
    assert rows["a"]["value"].item() == 20.0
    worker._qwen38_production_trace.close()
    assert worker.model_runner.prepare_inputs == original_prepare
    assert worker.model_runner.execute_model == original_execute
    assert worker.sleep == original_sleep


def test_megatron_adapter_excludes_per_document_padding_and_restores_hooks(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"prompts": [[1, 2], [3, 4]], "response_queries": 4}))
    monkeypatch.setenv(trace_module.PLAN_ENV, str(plan))
    monkeypatch.setenv(trace_module.OUTPUT_ENV, str(tmp_path / "output"))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 2,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", core)

    class Language(torch.nn.Module):
        def forward(self, input_ids, packed_seq_params):
            return torch.arange(8, dtype=torch.float32).reshape(8, 1)

    language = Language()
    model = SimpleNamespace(language_model=language, training=False)
    monkeypatch.setattr(
        trace_module.ProductionTrace,
        "attach_megatron",
        lambda self, model: self.output_hook("layer", model.language_model),
    )
    logical = [torch.tensor([1, 2, 9]), torch.tensor([3, 4, 8, 7])]
    inputs = SimpleNamespace(unbind=lambda: logical)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 4, 8]))
    physical = torch.tensor([[1, 2, 9, 0, 3, 4, 8, 7]])

    scores = {"log_probs": -torch.arange(8, dtype=torch.float32).reshape(1, 8)}

    def forward(logits_processor=None):
        result = language(input_ids=physical, packed_seq_params=packed)
        if logits_processor is not None:
            assert logits_processor() is scores
        return result

    native = forward()
    wrapped = trace_module.wrap_megatron_production_forward(forward, model, inputs, enabled=True)
    assert torch.equal(wrapped(logits_processor=lambda: scores), native)
    assert not language._forward_hooks and not language._forward_pre_hooks
    records = [torch.load(p, weights_only=True) for p in (tmp_path / "output").rglob("activations.pt")]
    by_prompt = {tuple(r["metadata"]["prompt_ids"]): r for r in records}
    first = by_prompt[(1, 2)]
    assert first["metadata"]["input_ids"] == [1, 2, 9]
    assert first["stages"]["layer"][0]["positions"] == [1]
    assert first["stages"]["log_probs"][0]["value"].item() == -1.0
    assert by_prompt[(3, 4)]["stages"]["layer"][0]["positions"] == [1, 2]


def test_invalid_prompt_coordinates_fail_closed(tmp_path):
    trace = trace_module.ProductionTrace({"prompts": [[1, 2]], "response_queries": 2}, tmp_path, "vllm", 0, 8)
    trace.register("a", [1, 2])
    with pytest.raises(ValueError, match="differs"):
        trace.select_rows("a", 0, [1], [99])


def test_duplicate_query_capture_is_rejected(tmp_path):
    trace = trace_module.ProductionTrace({"prompts": [[1, 2]], "response_queries": 2}, tmp_path, "vllm", 0, 8)
    trace.register("a", [1, 2])
    trace.total_rows = 1
    trace.select_rows("a", 0, [1], [2])
    trace.capture("layer", torch.ones(1, 2))
    trace.capture("layer", torch.ones(1, 2))
    with pytest.raises(ValueError, match="Duplicate"):
        trace.flush()


def test_ple_trace_brackets_native_inplace_convolution_without_changing_results(tmp_path):
    class Embedding(torch.nn.Module):
        def compute_ngram_ids(self, ids):
            return ids[:, None] + torch.tensor([10, 20])

        def forward(self, ids):
            return self.compute_ngram_ids(ids).float()

    class Projection(torch.nn.Module):
        output_sizes = (2, 2)

        def forward(self, embeddings):
            return torch.cat([embeddings, embeddings * 2], dim=-1), None

    class Ple(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ple_embedding, self.kv_proj = Embedding(), Projection()

        def _short_conv(self, inputs, residual, outer):
            residual.add_(inputs + outer)

        def forward(self, hidden, ids):
            kv, _ = self.kv_proj(self.ple_embedding(ids))
            key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
            self._short_conv(key, value, hidden)
            return value

    ple = Ple()
    inputs, ids = torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([2, 9])
    expected = ple(inputs, ids)
    original = ple._short_conv
    trace = trace_module.ProductionTrace(
        {"prompts": [[1, 2]], "response_queries": 2, "ple_substages": True}, tmp_path, "vllm", 0, 8
    )
    trace.register("request", [1, 2])
    trace.total_rows = 2
    trace.select_rows("request", 0, [1, 2], ids.tolist())
    trace.attach_ple("layers/01", ple)
    result = ple(inputs, ids)
    assert torch.equal(result, expected)
    stages = trace.records["request"]["stages"]
    tensor = lambda key: stages["layers/01/ple/" + key][0]["value"]
    assert torch.equal(tensor("input"), inputs)
    assert torch.equal(tensor("ngram_ids"), ids[:, None] + torch.tensor([10, 20]))
    assert torch.equal(tensor("key"), tensor("conv_input"))
    assert torch.equal(tensor("value"), tensor("gated"))
    assert torch.equal(tensor("output"), tensor("gated") + tensor("conv_input") + inputs)
    trace.close()
    assert ple._short_conv == original
    assert not ple._forward_hooks and not ple._forward_pre_hooks


@pytest.mark.parametrize("dim_first", [False, True])
def test_ple_cache_capture_tracks_reordered_slots_before_mutation(tmp_path, monkeypatch, dim_first):
    trace = trace_module.ProductionTrace(
        {"prompts": [[1, 2], [3, 4]], "response_queries": 2, "ple_cache_queries": 1},
        tmp_path,
        "vllm",
        0,
        8,
    )
    trace.register("b", [3, 4])
    trace.register("a", [1, 2])
    trace.select_rows("b", 0, [2], [8])
    trace.select_rows("a", 1, [2], [9])
    trace.total_rows = 2
    metadata = SimpleNamespace(
        num_spec_decodes=0,
        num_decode_tokens=2,
        state_indices_tensor=torch.tensor([2, 0]),
        has_initial_states_d=torch.tensor([True, True]),
    )
    forward = ModuleType("vllm.forward_context")
    forward.get_forward_context = lambda: SimpleNamespace(attn_metadata={"ple": metadata})
    utils = ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
    utils.is_conv_state_dim_first = lambda: dim_first
    monkeypatch.setitem(sys.modules, forward.__name__, forward)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    values = torch.arange(24).reshape(3, 2, 4)
    native = values.clone() if dim_first else values.transpose(-1, -2).contiguous()
    ple = SimpleNamespace(prefix="ple", kv_cache=[native], conv_state_len=3)
    trace.capture_ple_cache("ple/", ple)
    native.zero_()
    for identity, slot in [("a", 0), ("b", 2)]:
        stages = trace.records[identity]["stages"]
        assert torch.equal(stages["ple/cache_before"][0]["value"], values[slot, :, -3:].reshape(1, -1))
        assert stages["ple/cache_slot_and_valid"][0]["value"].tolist() == [[slot, 1]]


def test_selected_response_preserves_exact_routes_scores_and_request_id(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"prompts": [[1, 2]], "response_queries": 2}))
    monkeypatch.setenv(trace_module.PLAN_ENV, str(plan))
    monkeypatch.setenv(trace_module.OUTPUT_ENV, str(tmp_path / "output"))
    routes = torch.arange(12).reshape(3, 2, 2)
    result = SimpleNamespace(
        num_cached_tokens=0,
        outputs=[
            SimpleNamespace(
                token_ids=[3, 4],
                routed_experts=routes,
                logprobs=[{3: SimpleNamespace(logprob=-1.25)}, {4: SimpleNamespace(logprob=-2.5)}],
            )
        ],
    )
    trace_module.audit_selected_response(replica=3, request_id="request-a", prompt_ids=[1, 2], final_res=result)
    path = next((tmp_path / "output").rglob("complete.json"))
    record = json.loads(path.read_text())
    assert record["request_id"] == "request-a" and record["input_ids"] == [1, 2, 3, 4]
    assert record["logprobs"] == [-1.25, -2.5]
    assert torch.equal(torch.load(path.with_name("routes.pt"), weights_only=True), routes)
