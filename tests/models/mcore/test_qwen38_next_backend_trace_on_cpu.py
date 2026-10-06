# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Validate token coordinates and original-forward behavior of diagnostic hooks."""

from types import SimpleNamespace

import pytest
import torch

from examples.grpo_trainer.qwen3_8_next.backend_trace import BackendPrefixTrace, load_backend_trace


def test_detailed_vllm_gdn_hooks_preserve_native_calls_across_prefill_chunks(tmp_path):
    class Projection(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = width

        def forward(self, x):
            return x[:, : self.width], None

    def prepare(*, conv_output, a, b):
        q, k, v = conv_output.split([2, 2, 4], dim=-1)
        return q.reshape(-1, 1, 2), k.reshape(-1, 1, 2), v.reshape(-1, 2, 2), a.float(), b.float()

    ops = SimpleNamespace(causal_conv1d_fn=lambda x: x * 2, fused_post_conv_prep=prepare)
    original_conv, original_prepare = ops.causal_conv1d_fn, ops.fused_post_conv_prep

    class Core(torch.nn.Module):
        def forward(self, *, q, v):
            return q.repeat_interleave(2, dim=2) + v, None

    class Norm(torch.nn.Module):
        def forward(self, x, z):
            return x + z

    class GDN(torch.nn.Module):
        gqa_interleaved_layout = enable_fused_gdn_spec_decode = False
        key_dim, value_dim, tp_size, num_k_heads, num_v_heads = 2, 4, 1, 1, 2

        def __init__(self):
            super().__init__()
            self.in_proj_qkvz, self.in_proj_ba = Projection(12), Projection(4)
            self.chunk_gated_delta_rule, self.norm, self.out_proj = Core(), Norm(), Projection(4)

        def forward(self, x):
            qkvz, _ = self.in_proj_qkvz(x)
            ba, _ = self.in_proj_ba(x)
            b, a = ba.chunk(2, dim=-1)
            conv = ops.causal_conv1d_fn(qkvz[:, :8].T).T
            q, k, v, g, beta = ops.fused_post_conv_prep(conv_output=conv, a=a, b=b)
            core, _ = self.chunk_gated_delta_rule(q=q.unsqueeze(0), v=v.unsqueeze(0))
            normed = self.norm(core[0], qkvz[:, 8:].reshape(-1, 2, 2))
            return self.out_proj(normed.flatten(1))[0]

    gdn = GDN()
    original_forward = gdn.forward
    full = torch.arange(480 * 12).reshape(480, 12).float()
    expected = gdn(full).clone()
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 1, token_start=390, tokens=32)

    class Model(torch.nn.Module):
        def forward(self, input_ids, positions):
            return gdn(full[positions])

    model = Model()
    trace.attach_vllm_positions(model)
    trace.attach_vllm_gdn_detail(gdn, "layers/00", ops)
    trace.start(dict(id="real", input_ids=list(range(480))), 480, "config")
    for start, end in [(0, 400), (400, 480)]:
        result = model(torch.arange(start, end), torch.arange(start, end))
        torch.testing.assert_close(result, expected[start:end], rtol=0, atol=0)
        result.zero_()
        # Calls outside the selected layer must not create duplicate snapshots.
        ops.causal_conv1d_fn(torch.ones(8, end - start))
    trace.finish()
    trace.close()
    assert gdn.forward == original_forward
    assert ops.causal_conv1d_fn is original_conv and ops.fused_post_conv_prep is original_prepare
    _, values = load_backend_trace(tmp_path, "real")
    prefix = "layers/00/gdn/"
    torch.testing.assert_close(values[prefix + "projection/qkvz/tp-00"], full[390:422], rtol=0, atol=0)
    torch.testing.assert_close(values[prefix + "conv/tp-00"], full[390:422, :8] * 2, rtol=0, atol=0)
    q = (full[390:422, :2] * 2).repeat_interleave(2, dim=0).reshape(32, 4)
    torch.testing.assert_close(values[prefix + "prepared/q/tp-00"], q, rtol=0, atol=0)
    torch.testing.assert_close(values[prefix + "norm_gate/tp-00"], expected[390:422], rtol=0, atol=0)
    torch.testing.assert_close(gdn(full), expected, rtol=0, atol=0)


def test_per_prompt_response_window_does_not_leak_to_next_prompt(tmp_path):
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 1, tokens=3, token_start=1)
    full = torch.arange(24).reshape(8, 3)
    for prompt in [
        dict(id="response", input_ids=list(range(8)), trace_token_start=4),
        dict(id="default", input_ids=list(range(8))),
    ]:
        trace.start(prompt, 8, "config")
        trace.capture("layers/00/attention/output", full)
        trace.finish()
    first, values = load_backend_trace(tmp_path, "response")
    assert first["token_start"] == 4
    torch.testing.assert_close(values["layers/00/attention/output"], full[4:7])
    second, values = load_backend_trace(tmp_path, "default")
    assert second["token_start"] == 1
    torch.testing.assert_close(values["layers/00/attention/output"], full[1:4])


def test_sequence_parallel_prefix_crosses_rank_boundary(tmp_path):
    prompt = dict(id="short", input_ids=list(range(198)))
    full = torch.arange(200 * 4).reshape(200, 1, 4).to(torch.bfloat16)
    for rank in range(8):
        trace = BackendPrefixTrace(tmp_path, "megatron", rank, rank, 8)
        trace.start(prompt, 200, "config")
        trace.capture("layers/00/attn_hc/input", full[rank * 25 : (rank + 1) * 25])
        trace.capture("layers/01/ple/output", full[:, 0])
        trace.finish()
    metadata, values = load_backend_trace(tmp_path, "short")
    assert metadata["input_ids"] == prompt["input_ids"]
    torch.testing.assert_close(values["layers/00/attn_hc/input"], full[:32, 0], rtol=0, atol=0)
    torch.testing.assert_close(values["layers/01/ple/output"], full[:32, 0], rtol=0, atol=0)
    # Losing TP rank one must fail, rather than silently compare only 25 tokens.
    (tmp_path / "rank-00001/index.json").unlink()
    with pytest.raises(ValueError, match="Incomplete prefix"):
        load_backend_trace(tmp_path, "short")


def test_vllm_delayed_residual_capture_does_not_change_forward(tmp_path):
    class Mixer(torch.nn.Module):
        def mix(self, x):
            return x, x * 2, None

        def combine_and_mix(self, x, block, injection):
            updated = x + block * injection
            return updated, updated * 2, None

    mixer = Mixer()
    x, block, injection = torch.randn(198, 4), torch.randn(198, 4), torch.randn(198, 1)
    expected = mixer.combine_and_mix(x, block, injection)
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 8)
    trace._vllm_hc(mixer, "layers/00/mlp_hc/input", "layers/00/mlp_hc/mixed")
    trace.start(dict(id="short", input_ids=list(range(198))), 198, "config")
    output = mixer.combine_and_mix(x, block, injection)
    assert output[2] is None
    torch.testing.assert_close(output[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(output[1], expected[1], rtol=0, atol=0)
    output[0].zero_()  # Subsequent in-place consumers must not corrupt snapshots.
    trace.finish()
    trace.close()
    _, values = load_backend_trace(tmp_path, "short")
    torch.testing.assert_close(values["layers/00/mlp_hc/input"], expected[0][:32], rtol=0, atol=0)
    torch.testing.assert_close(mixer.combine_and_mix(x, block, injection)[1], expected[1], rtol=0, atol=0)


def test_backend_trace_rejects_chunked_or_repeated_forward(tmp_path):
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 8)
    trace.start(dict(id="short", input_ids=list(range(198))), 198, "config")
    with pytest.raises(ValueError, match="Unknown token coordinates"):
        trace.capture("layers/00/attn_hc/input", torch.zeros(64, 4))
    trace.capture("layers/00/attn_hc/input", torch.zeros(198, 4))
    with pytest.raises(RuntimeError, match="Multiple forwards"):
        trace.capture("layers/00/attn_hc/input", torch.zeros(198, 4))


def test_vllm_absolute_positions_join_window_across_hybrid_prefill_chunks(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, input_ids, positions, *, marker):
            assert marker is sentinel
            value = full[positions[0]]
            trace.capture("layers/00/attn_hc/input", value)
            trace.active_qsa = "layers/03"
            trace._capture_qsa_scores(value[:, :2].clone(), torch.full((value.shape[0],), 2))
            return value

    sentinel = object()
    full = torch.arange(480 * 4).reshape(480, 4)
    model = Model()
    original = model.forward
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 8, tokens=32, token_start=390, trace_qsa=True)
    trace.attach_vllm_positions(model)
    trace.start(dict(id="real-480", input_ids=list(range(480))), 480, "config")
    for start, end in [(0, 400), (400, 480)]:
        positions = torch.arange(start, end).expand(3, -1)
        output = model(torch.arange(start, end), positions, marker=sentinel)
        torch.testing.assert_close(output, full[start:end], rtol=0, atol=0)
        output.zero_()  # Trace chunks must not alias their downstream consumers.
    trace.finish()
    trace.close()
    assert model.forward == original
    metadata, values = load_backend_trace(tmp_path, "real-480")
    assert metadata["forward_chunks"] == [dict(start=0, tokens=400), dict(start=400, tokens=80)]
    torch.testing.assert_close(values["layers/00/attn_hc/input"], full[390:422], rtol=0, atol=0)
    torch.testing.assert_close(values["layers/03/qsa/scores/tp-00"], full[390:422, :2], rtol=0, atol=0)


@pytest.mark.parametrize("invalid_positions", [torch.arange(399, 480), torch.arange(401, 480), torch.arange(400, 481)])
def test_vllm_position_trace_rejects_overlap_missing_and_decode_queries(tmp_path, invalid_positions):
    class Model(torch.nn.Module):
        def forward(self, input_ids, positions):
            return input_ids

    model = Model()
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 1)
    trace.attach_vllm_positions(model)
    trace.start(dict(id="real-480", input_ids=list(range(480))), 480, "config")
    model(torch.arange(400), torch.arange(400))
    with pytest.raises(ValueError, match="token coordinates"):
        model(invalid_positions, invalid_positions)
    with pytest.raises(ValueError, match="Incomplete vLLM prefill"):
        trace.finish()
    trace.close()


def test_vllm_position_trace_rejects_wrong_tokens_and_noncontiguous_positions(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, input_ids, positions):
            return input_ids

    model = Model()
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 1)
    trace.attach_vllm_positions(model)
    trace.start(dict(id="real", input_ids=[1, 2, 3]), 3, "config")
    with pytest.raises(ValueError, match="token IDs"):
        model(torch.tensor([1, 9]), torch.arange(2))
    with pytest.raises(ValueError, match="Noncontiguous"):
        model(torch.tensor([1, 3]), torch.tensor([0, 2]))
    trace.close()


def test_vllm_decode_trace_requires_exact_unsampled_tail_coverage(tmp_path):
    class Layer(torch.nn.Module):
        def forward(self, input_ids):
            return input_ids[:, None].float()

    layer = Layer()

    class Model(torch.nn.Module):
        def forward(self, input_ids, positions):
            return layer(input_ids)

    model = Model()
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 1, tokens=3, token_start=2)
    trace.attach_vllm_positions(model)
    trace.output_hook("layers/00/attention/output", layer)
    prompt = dict(id="decode", input_ids=list(range(7)), trace_query_length=6)
    trace.start(prompt, 7, "config")
    model(torch.arange(3), torch.arange(3))
    for pos in (3, 4):
        model(torch.tensor([pos]), torch.tensor([pos]))
    with pytest.raises(ValueError, match="Incomplete"):
        trace.finish()
    model(torch.tensor([5]), torch.tensor([5]))
    with pytest.raises(ValueError, match="token coordinates"):
        model(torch.tensor([6]), torch.tensor([6]))
    trace.finish()
    trace.close()
    metadata, values = load_backend_trace(tmp_path, "decode")
    assert metadata["trace_query_length"] == 6 and len(metadata["input_ids"]) == 7
    assert metadata["forward_chunks"] == [dict(start=0, tokens=3)] + [dict(start=i, tokens=1) for i in (3, 4, 5)]
    torch.testing.assert_close(values["layers/00/attention/output"], torch.arange(2, 5)[:, None].float())


@pytest.mark.parametrize("query_length", [0, 8, True, 3.0, 4])
def test_vllm_decode_trace_rejects_invalid_or_uncovered_window(tmp_path, query_length):
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 1, tokens=3, token_start=2)
    with pytest.raises(ValueError, match="query coverage"):
        trace.start(dict(id="invalid", input_ids=list(range(7)), trace_query_length=query_length), 7, "config")


def test_window_reconstructs_global_coordinates_across_sequence_shards(tmp_path):
    prompt = dict(id="long", input_ids=list(range(2304)))
    full = torch.arange(2304 * 3).reshape(2304, 3)
    for rank in range(8):
        trace = BackendPrefixTrace(tmp_path, "megatron", rank, rank, 8, tokens=96, token_start=2000)
        trace.start(prompt, 2304, "config")
        trace.capture("layers/00/attn_hc/input", full[rank * 288 : (rank + 1) * 288])
        trace.finish()
    metadata, values = load_backend_trace(tmp_path, "long")
    assert metadata["token_start"] == 2000
    torch.testing.assert_close(values["layers/00/attn_hc/input"], full[2000:2096], rtol=0, atol=0)
    (tmp_path / "rank-00006/index.json").unlink()
    with pytest.raises(ValueError, match="Missing or overlapping"):
        load_backend_trace(tmp_path, "long")


def test_window_beyond_prompt_fails_before_forward(tmp_path):
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 8, token_start=2048)
    with pytest.raises(ValueError, match="beyond the prompt"):
        trace.start(dict(id="short", input_ids=list(range(198))), 198, "config")


def test_vllm_selections_preserve_each_tensor_parallel_rank(tmp_path):
    prompt = dict(id="long", input_ids=list(range(2304)))
    for rank in range(2):
        trace = BackendPrefixTrace(tmp_path, "vllm", rank, rank, 2, tokens=96, token_start=2016)
        trace.start(prompt, 2304, "config")
        trace.capture("layers/03/qsa/selection", torch.full((2304, 5), rank), all_tp=True)
        trace.finish()
    _, values = load_backend_trace(tmp_path, "long")
    assert len(values) == 2
    assert (values["layers/03/qsa/selection/tp-00"] == 0).all()
    assert (values["layers/03/qsa/selection/tp-01"] == 1).all()


def test_megatron_qsa_replicated_selections_preserve_each_rank_and_absolute_window(tmp_path):
    directory = tmp_path / "megatron-qsa"
    prompt = {"id": "long", "input_ids": [1] * 2304}
    for rank in range(2):
        trace = BackendPrefixTrace(directory, "megatron", rank, rank, 2, tokens=96, token_start=2016)
        trace.start(prompt, 2304, "config")
        trace.capture("layers/03/qsa/selection", torch.full((2304, 5), rank), all_tp=True)
        trace.finish()
    metadata, values = load_backend_trace(directory, "long")
    assert metadata["token_start"] == 2016
    assert values["layers/03/qsa/selection/tp-00"].shape == (96, 5)
    assert (values["layers/03/qsa/selection/tp-00"] == 0).all()
    assert (values["layers/03/qsa/selection/tp-01"] == 1).all()


def test_qsa_scores_ignore_only_uninitialized_workspace_outside_valid_lengths(tmp_path):
    trace = BackendPrefixTrace(tmp_path, "vllm", 0, 0, 8, tokens=2, token_start=4, trace_qsa=True)
    trace.start(dict(id="short", input_ids=list(range(8))), 8, "config")
    trace.active_qsa = "layers/03"
    logits = torch.full((8, 4), float("nan"))
    logits[4, :2] = torch.tensor([1.0, 2.0])
    logits[5, :3] = torch.tensor([3.0, 4.0, 5.0])
    visible = torch.tensor([0, 0, 0, 0, 2, 3, 0, 0], dtype=torch.int32)
    trace._capture_qsa_scores(logits, visible)
    trace.finish()
    _, values = load_backend_trace(tmp_path, "short")
    torch.testing.assert_close(
        values["layers/03/qsa/scores/tp-00"], torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 5.0, 0.0]])
    )
    assert torch.isnan(logits[4, 2:]).all()  # The live kernel workspace is untouched.
