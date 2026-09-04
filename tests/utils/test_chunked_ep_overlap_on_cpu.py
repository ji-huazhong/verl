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
"""Run the real chunk scheduler/autograd/collectives without Ray, TE or CUDA."""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint


def _load_runtime():
    path = Path(__file__).resolve().parents[2] / "verl/utils/megatron/chunked_ep_overlap.py"
    spec = importlib.util.spec_from_file_location("_verl_chunked_ep_test_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime_module = _load_runtime()


def _engine(**overrides):
    values = dict(
        chunked_ep_overlap=SimpleNamespace(enabled=True, num_chunks=2, min_tokens_per_chunk=0),
        tensor_model_parallel_size=1,
        expert_model_parallel_size=2,
        dtype="bfloat16",
        override_transformer_config={},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size"),
        ({"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size"),
        ({"expert_model_parallel_size": 1}, "expert_model_parallel_size"),
        ({"dynamic_context_parallel": True}, "dynamic_context_parallel"),
        ({"use_megatron_fsdp": True}, "use_megatron_fsdp"),
        ({"override_ddp_config": {"overlap_grad_reduce": True}}, "overlap_grad_reduce"),
        ({"override_transformer_config": {"overlap_moe_expert_parallel_comm": True}}, "overlap_moe"),
        ({"override_transformer_config": {"moe_shared_expert_overlap": True}}, "shared_expert"),
        ({"override_transformer_config": {"moe_token_dispatcher_type": "flex"}}, "alltoall"),
        ({"override_transformer_config": {"moe_expert_capacity_factor": 1.0}}, "dropless"),
        ({"override_transformer_config": {"fp8": "hybrid"}}, "fp8"),
        ({"override_transformer_config": {"cuda_graph_impl": "local"}}, "CUDA graphs"),
    ],
)
def test_unsupported_config(overrides, match):
    with pytest.raises(ValueError, match=match):
        runtime_module.validate_chunked_ep_config(_engine(**overrides))


def test_effective_config_and_disabled_path():
    # Validate model defaults even when they were absent in user overrides.
    with pytest.raises(ValueError, match="shared_expert"):
        runtime_module.validate_chunked_ep_config(
            _engine(), transformer_config=SimpleNamespace(moe_shared_expert_overlap=True)
        )
    runtime_module.validate_chunked_ep_config(SimpleNamespace(chunked_ep_overlap={"enabled": False}))
    model = torch.nn.Linear(2, 2)
    assert runtime_module.install_chunked_ep_overlap(model, _engine(chunked_ep_overlap={"enabled": False})) is model
    runtime_module.validate_chunked_ep_config(_engine(override_transformer_config={"recompute_granularity": "full"}))


@pytest.mark.parametrize("tokens,chunks,expected", [(7, 3, [3, 2, 2]), (2, 4, [1, 1, 0, 0]), (0, 2, [0, 0])])
def test_chunk_partition(tokens, chunks, expected):
    assert runtime_module.chunk_sizes(tokens, chunks) == expected


class _Experts(torch.nn.Module):
    def __init__(self, w1, w2, ignore_probs=False):
        super().__init__()
        self.w1 = torch.nn.Parameter(w1.clone())
        self.w2 = torch.nn.Parameter(w2.clone())
        self.ignore_probs = ignore_probs

    def forward(self, hidden, counts, probs):
        outputs = []
        offset = 0
        for e, count in enumerate(counts.tolist()):
            x = hidden[offset : offset + count]
            activation = torch.tanh(x @ self.w1[e])
            if not self.ignore_probs:
                activation = activation * probs[offset : offset + count, None]
            outputs.append(activation @ self.w2[e])
            offset += count
        return torch.cat(outputs), None


def _routing(x, gate, topk, empty_experts):
    probs = (x @ gate).softmax(-1)
    # Use fixed routes to exercise zero-receive ranks independently of weights.
    indices = torch.arange(topk).expand(x.shape[0], topk) if empty_experts else probs.topk(topk).indices
    routes = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, indices, True)
    routes[-1] = False  # padding with zero dispatched tokens
    return probs, routes


def _dense(x, gate, w1, w2, topk, empty_experts, ignore_probs):
    probs, routes = _routing(x, gate, topk, empty_experts)
    result = x * 0
    for e in range(w1.shape[0]):
        p = routes[:, e] if ignore_probs else probs[:, e] * routes[:, e]
        result = result + (torch.tanh(x @ w1[e]) * p[:, None]) @ w2[e]
    return result + probs.sum() * 0


def _distributed_worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=90)
    )
    try:
        # All cases run real alltoall, including a whole rank with zero receives,
        # missing probability gradients, and multiple live microbatch graphs.
        cases = [(2, 1, False, False), (4, 2, False, True), (2, 1, True, True)]
        for num_chunks, topk, empty_experts, recompute in cases:
            for ignore_probs in (False, True):
                torch.manual_seed(19)
                w1 = torch.randn(4, 5, 7, dtype=torch.float64) / 4
                w2 = torch.randn(4, 7, 5, dtype=torch.float64) / 4
                ref_w1 = w1.clone().requires_grad_()
                ref_w2 = w2.clone().requires_grad_()
                ref_gate = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
                gate = ref_gate.detach().clone().requires_grad_()
                experts = _Experts(w1[rank * 2 : rank * 2 + 2], w2[rank * 2 : rank * 2 + 2], ignore_probs)
                runtime = runtime_module._Runtime(experts, dist.group.WORLD, num_chunks, 0)
                reference_inputs, inputs, targets, outputs, reference_outputs = [], [], [], [], []

                def run(x, gate=gate, topk=topk, empty_experts=empty_experts, runtime=runtime):
                    probs, routes = _routing(x, gate, topk, empty_experts)
                    return runtime(x, probs, routes)

                for microbatch in range(2):
                    torch.manual_seed(77 + rank + microbatch * 9)
                    x = torch.randn(7 + rank * 4 + microbatch, 5, dtype=torch.float64, requires_grad=True)
                    ref_x = x.detach().clone().requires_grad_()
                    assert runtime.should_chunk(x)
                    out = checkpoint(run, x, use_reentrant=True) if recompute else run(x)
                    ref_out = _dense(ref_x, ref_gate, ref_w1, ref_w2, topk, empty_experts, ignore_probs)
                    torch.testing.assert_close(out, ref_out, atol=1e-12, rtol=1e-12)
                    with torch.no_grad():
                        torch.testing.assert_close(run(x), ref_out, atol=1e-12, rtol=1e-12)
                    inputs.append(x)
                    reference_inputs.append(ref_x)
                    outputs.append(out)
                    reference_outputs.append(ref_out)
                    targets.append(torch.randn_like(out))

                sum((o * t).sum() for o, t in zip(outputs, targets, strict=False)).backward()
                sum((o * t).sum() for o, t in zip(reference_outputs, targets, strict=False)).backward()
                for x, ref_x in zip(inputs, reference_inputs, strict=False):
                    torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-11, rtol=1e-11)
                torch.testing.assert_close(gate.grad, ref_gate.grad, atol=1e-11, rtol=1e-11)
                # Each expert receives contributions from *both* source ranks.
                for param, reference in ((experts.w1, ref_w1), (experts.w2, ref_w2)):
                    dist.all_reduce(reference.grad)
                    expected = reference.grad[rank * 2 : rank * 2 + 2]
                    torch.testing.assert_close(param.grad, expected, atol=1e-11, rtol=1e-11)
                    with torch.no_grad():
                        updated = param - 0.03 * param.grad
                        ref_updated = reference[rank * 2 : rank * 2 + 2] - 0.03 * expected
                        torch.testing.assert_close(updated, ref_updated, atol=1e-11, rtol=1e-11)

        # A short rank must make every EP rank choose the same native fallback.
        runtime.min_tokens_per_chunk = 8
        assert not runtime.should_chunk(torch.zeros(1 if rank == 0 else 128, 5))
        runtime.num_chunks = 1
        assert not runtime.should_chunk(torch.zeros(128, 5))

        # Frozen inputs/router must not suppress expert parameter training.
        runtime.num_chunks = 2
        experts.zero_grad(set_to_none=True)
        x = torch.ones(4, 5, dtype=torch.float64)
        probs, routes = _routing(x, gate.detach(), 2, False)
        runtime(x, probs, routes).sum().backward()
        assert experts.w1.grad is not None and experts.w2.grad is not None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_distributed_output_and_gradients(tmp_path):
    mp.spawn(_distributed_worker, args=(2, str(tmp_path / "rendezvous")), nprocs=2, join=True)
