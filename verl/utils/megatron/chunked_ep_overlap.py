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
"""MoE-local EP overlap, independent of Megatron's combined-1F1B schedule.

The router remains outside the custom autograd boundary. Inside it, each chunk
owns three local graphs and two communication graphs. Backward visits these
graphs explicitly, including zero probability gradients, so every EP rank issues
the same collectives. Expert parameter gradients accumulate through their usual
autograd hooks; overlapping DDP gradient reduction is therefore not supported.

Only TP=ETP=1 and dropless alltoall are implemented. The unfused primitives and
stream abstraction also run on CPU/Gloo for numerical tests; the public model
installer requires CUDA. No saved tensor's storage is resized by this module.
"""

import logging
import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from types import MethodType

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def validate_chunked_ep_config(engine_config, model_config=None, transformer_config=None):
    """Reject unsupported combinations without modifying the caller's config."""
    options = _get(engine_config, "chunked_ep_overlap")
    if not _get(options, "enabled", False):
        return
    for name, minimum in (("num_chunks", 1), ("min_tokens_per_chunk", 0)):
        value = _get(options, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"chunked_ep_overlap.{name} must be an integer >= {minimum}")
    required = {"tensor_model_parallel_size": 1, "dtype": "bfloat16"}
    for name, expected in required.items():
        if _get(engine_config, name) != expected:
            raise ValueError(f"chunked_ep_overlap requires {name}={expected}")
    if _get(engine_config, "expert_tensor_parallel_size") not in (None, 1):
        raise ValueError("chunked_ep_overlap requires expert_tensor_parallel_size=1")
    if _get(engine_config, "expert_model_parallel_size", 1) <= 1:
        raise ValueError("chunked_ep_overlap requires expert_model_parallel_size>1")
    for name in ("dynamic_context_parallel", "use_megatron_fsdp"):
        if _get(engine_config, name, False):
            raise ValueError(f"chunked_ep_overlap does not support {name}")
    if _get(engine_config, "virtual_pipeline_model_parallel_size") not in (None, 1):
        raise ValueError("chunked_ep_overlap does not yet support virtual pipeline parallelism")
    if _get(_get(engine_config, "override_ddp_config"), "overlap_grad_reduce", False):
        raise ValueError("chunked_ep_overlap requires override_ddp_config.overlap_grad_reduce=False")
    if _get(_get(engine_config, "qat"), "enable", False):
        raise ValueError("chunked_ep_overlap does not support QAT")
    if model_config is not None:
        if _get(_get(model_config, "mtp"), "enable", False):
            raise ValueError("chunked_ep_overlap does not yet support MTP")
        if _get(_get(model_config, "lora"), "rank", 0) or _get(model_config, "lora_rank", 0):
            raise ValueError("chunked_ep_overlap does not yet support LoRA")

    configs = [_get(engine_config, "override_transformer_config", {})]
    if transformer_config is not None:
        configs.append(transformer_config)
    for config in configs:
        for name in (
            "overlap_moe_expert_parallel_comm",
            "overlap_dispatch_backward_with_experts_wgrad",
            "delay_wgrad_compute",
            "moe_shared_expert_overlap",
            "moe_pad_expert_input_to_capacity",
            "moe_router_padding_for_quantization",
            "moe_latent_size",
            "fp8",
            "fp4",
            "cpu_offloading",
            "mtp_num_layers",
        ):
            if _get(config, name, False):
                raise ValueError(f"chunked_ep_overlap is incompatible with {name}")
        if _get(config, "moe_expert_capacity_factor") is not None:
            raise ValueError("chunked_ep_overlap requires dropless routing (moe_expert_capacity_factor=None)")
        if _get(config, "moe_token_dispatcher_type", "alltoall") != "alltoall":
            raise ValueError("chunked_ep_overlap requires moe_token_dispatcher_type=alltoall")
        if _get(config, "cuda_graph_impl", "none") != "none":
            raise ValueError("chunked_ep_overlap does not support CUDA graphs")
        if _get(config, "recompute_granularity") not in (None, "full"):
            raise ValueError("chunked_ep_overlap supports full recompute or no recompute")
        if _get(config, "fine_grained_activation_offloading", False) or _get(config, "offload_modules", []):
            raise ValueError("chunked_ep_overlap does not support activation offloading")
    if transformer_config is not None:
        if _get(transformer_config, "tensor_model_parallel_size", 1) != 1 or _get(
            transformer_config, "expert_tensor_parallel_size", 1
        ) not in (None, 1):
            raise ValueError("chunked_ep_overlap requires effective TP=ETP=1")


def chunk_sizes(num_tokens, num_chunks):
    """Return exactly num_chunks balanced sizes, including empty tail chunks."""
    quotient, remainder = divmod(num_tokens, num_chunks)
    return [quotient + (i < remainder) for i in range(num_chunks)]


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, input_splits, output_splits, group):
        ctx.input_splits, ctx.output_splits, ctx.group = input_splits, output_splits, group
        output = tensor.new_empty((sum(output_splits), *tensor.shape[1:]))
        work = dist.all_to_all_single(
            output,
            tensor.contiguous(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
            async_op=True,
        )
        # On NCCL this orders the *current communication stream*. The compute
        # stream only waits on the individual stage event at its consumption point.
        work.wait()
        return output

    @staticmethod
    def backward(ctx, grad):
        return _AllToAll.apply(grad, ctx.output_splits, ctx.input_splits, ctx.group), None, None, None


class _Streams:
    def __init__(self, device, comm=None):
        self.cuda = device.type == "cuda"
        self.compute = torch.cuda.current_stream(device) if self.cuda else None
        self.comm = (comm if comm is not None else torch.cuda.Stream(device=device)) if self.cuda else None

    def context(self, stream):
        return torch.cuda.stream(stream) if self.cuda else nullcontext()

    def record(self, stream):
        return stream.record_event() if self.cuda else None

    def wait(self, stream, events):
        if self.cuda:
            for event in events:
                if event is not None:
                    stream.wait_event(event)

    def use(self, tensors, stream):
        if self.cuda:
            for tensor in tensors:
                if tensor is not None:
                    tensor.record_stream(stream)

    def range(self, name):
        return torch.cuda.nvtx.range(name) if self.cuda else torch.autograd.profiler.record_function(name)


class _Stage:
    """An invocation-local graph with explicit forward and backward dependencies."""

    def __init__(self, function, streams, stream, track_grad, name):
        self.function, self.streams, self.stream = function, streams, stream
        self.track_grad, self.name = track_grad, name
        self.inputs = self.outputs = None
        self.event = None

    def forward(self, *inputs, events=()):
        self.streams.wait(self.stream, events)
        with self.streams.context(self.stream), torch.set_grad_enabled(self.track_grad):
            with self.streams.range(f"chunked_ep/{self.name}/forward"):
                self.streams.use(inputs, self.stream)
                # All stage inputs are floating tensors. Always enable their
                # gradients so even disconnected/empty experts participate in A2A.
                leaves = tuple(t.detach().requires_grad_(self.track_grad) for t in inputs)
                result = self.function(*leaves)
                outputs = result if isinstance(result, tuple) else (result,)
                self.event = self.streams.record(self.stream)
        if self.track_grad:
            self.inputs, self.outputs = leaves, outputs
        return outputs

    def backward(self, *grads, events=()):
        self.streams.wait(self.stream, events)
        with self.streams.context(self.stream):
            with self.streams.range(f"chunked_ep/{self.name}/backward"):
                self.streams.use(grads, self.stream)
                pairs = [(out, grad) for out, grad in zip(self.outputs, grads, strict=True) if out.requires_grad]
                if pairs:
                    torch.autograd.backward([p[0] for p in pairs], [p[1] for p in pairs])
                result = tuple(t.grad if t.grad is not None else torch.zeros_like(t) for t in self.inputs)
                self.event = self.streams.record(self.stream)
        self.inputs = self.outputs = None
        self.function = None
        return result


class _Dispatcher:
    """Dropless TP=ETP=1 layout; communication metadata is prepared in one batch."""

    def __init__(self, routing_map, counts, rank, group, fused):
        self.routing_map, self.group, self.fused = routing_map, group, fused
        ep_size, num_experts = counts.shape
        local_experts = num_experts // ep_size
        local_counts = counts[rank]
        received = counts[:, rank * local_experts : (rank + 1) * local_experts]
        self.input_splits = local_counts.reshape(ep_size, local_experts).sum(1).tolist()
        self.output_splits = received.sum(1).tolist()
        self.tokens_per_expert = received.sum(0).contiguous()
        self.received_splits = received.reshape(-1).tolist()
        self.expert_splits = received.T.reshape(-1).tolist()
        self.sort_order = [r * local_experts + e for e in range(local_experts) for r in range(ep_size)]
        self.restore_order = [e * ep_size + r for r in range(ep_size) for e in range(local_experts)]
        self.num_out_tokens = sum(self.input_splits)
        if fused:
            from megatron.core.transformer.moe.moe_utils import permute, sort_chunks_by_idxs, unpermute

            self.permute, self.sort, self.unpermute = permute, sort_chunks_by_idxs, unpermute
            device = routing_map.device
            self.received_splits_tensor = torch.tensor(self.received_splits, device=device)
            self.expert_splits_tensor = torch.tensor(self.expert_splits, device=device)
            self.sort_order_tensor = torch.tensor(self.sort_order, device=device)
            self.restore_order_tensor = torch.tensor(self.restore_order, device=device)

    def preprocess(self, hidden, probs):
        self.hidden_shape = hidden.shape
        if self.fused:
            hidden, probs, self.indices, *_ = self.permute(
                hidden, self.routing_map, probs=probs, num_out_tokens=self.num_out_tokens, fused=True
            )
            return hidden, probs
        experts, self.indices = self.routing_map.T.nonzero(as_tuple=True)
        return hidden.index_select(0, self.indices), probs[self.indices, experts]

    def dispatch(self, hidden, probs):
        return (
            _AllToAll.apply(hidden, self.input_splits, self.output_splits, self.group),
            _AllToAll.apply(probs, self.input_splits, self.output_splits, self.group),
        )

    def compute(self, experts, hidden, probs):
        if self.fused:
            hidden, probs = self.sort(
                hidden, self.received_splits_tensor, self.sort_order_tensor, probs=probs, fused=True
            )
        else:
            hidden = self._sort(hidden, self.received_splits, self.sort_order)
            probs = self._sort(probs, self.received_splits, self.sort_order)
        output, bias = experts(hidden, self.tokens_per_expert, probs)
        if bias is not None:
            raise RuntimeError("chunked_ep_overlap expects experts to return no separate bias")
        if self.fused:
            return self.sort(output, self.expert_splits_tensor, self.restore_order_tensor, fused=True)[0]
        return self._sort(output, self.expert_splits, self.restore_order)

    @staticmethod
    def _sort(tensor, sizes, order):
        pieces = tensor.split(sizes, dim=0)
        return torch.cat([pieces[i] for i in order], dim=0)

    def combine(self, output):
        return _AllToAll.apply(output, self.output_splits, self.input_splits, self.group)

    def postprocess(self, output):
        if self.fused:
            return self.unpermute(
                output, self.indices, restore_shape=self.hidden_shape, routing_map=self.routing_map, fused=True
            )
        return output.new_zeros(self.hidden_shape).index_add_(0, self.indices, output)


@dataclass
class _Chunk:
    pre: _Stage
    dispatch: _Stage
    compute: _Stage
    combine: _Stage
    post: _Stage


class _Invocation:
    def __init__(self, experts, routing_maps, counts, group, fused, streams):
        self.experts, self.routing_maps, self.counts = experts, routing_maps, counts
        self.group, self.fused, self.streams = group, fused, streams
        self.sizes = [r.shape[0] for r in routing_maps]
        self.chunks = []

    def forward(self, hidden, probs, track_grad):
        streams = self.streams
        ready = streams.record(streams.compute)
        rank = dist.get_rank(self.group)
        prepared = []
        for i, (h, p, routing) in enumerate(
            zip(hidden.split(self.sizes), probs.split(self.sizes), self.routing_maps, strict=False)
        ):
            dispatcher = _Dispatcher(routing, self.counts[:, i], rank, self.group, self.fused)

            def stage(function, name, stream=streams.compute, index=i):
                return _Stage(function, streams, stream, track_grad, f"chunk{index}/{name}")

            chunk = _Chunk(
                stage(dispatcher.preprocess, "permute"),
                stage(dispatcher.dispatch, "dispatch", streams.comm),
                stage(partial(dispatcher.compute, self.experts), "experts"),
                stage(dispatcher.combine, "combine", streams.comm),
                stage(dispatcher.postprocess, "unpermute"),
            )
            self.chunks.append(chunk)
            prepared.append(chunk.pre.forward(h, p, events=(ready,)))

        # Two outstanding dispatches are enough for lookahead. Keep the same
        # submission order on every rank, independent of local token counts.
        dispatched = {}

        def dispatch(index):
            chunk = self.chunks[index]
            dispatched[index] = chunk.dispatch.forward(*prepared[index], events=(chunk.pre.event,))
            prepared[index] = None

        dispatch(0)
        combined = []
        for i, chunk in enumerate(self.chunks):
            if i + 1 < len(self.chunks):
                dispatch(i + 1)
            output = chunk.compute.forward(*dispatched.pop(i), events=(chunk.dispatch.event,))
            combined.append(chunk.combine.forward(*output, events=(chunk.compute.event,)))

        outputs = [
            chunk.post.forward(*out, events=(chunk.combine.event,))[0]
            for chunk, out in zip(self.chunks, combined, strict=True)
        ]
        return torch.cat(outputs, dim=0)

    def backward(self, grad):
        streams = self.streams
        ready = streams.record(streams.compute)
        post_grads = [
            chunk.post.backward(g, events=(ready,))
            for chunk, g in zip(self.chunks, grad.split(self.sizes), strict=True)
        ]
        received = {}

        def combine_backward(index):
            chunk = self.chunks[index]
            received[index] = chunk.combine.backward(*post_grads[index], events=(chunk.post.event,))
            post_grads[index] = None

        combine_backward(0)
        returned = []
        for i, chunk in enumerate(self.chunks):
            if i + 1 < len(self.chunks):
                combine_backward(i + 1)
            grads = chunk.compute.backward(*received.pop(i), events=(chunk.combine.event,))
            returned.append(chunk.dispatch.backward(*grads, events=(chunk.compute.event,)))
        inputs = [
            chunk.pre.backward(*grads, events=(chunk.dispatch.event,))
            for chunk, grads in zip(self.chunks, returned, strict=True)
        ]
        self.chunks.clear()
        return torch.cat([g[0] for g in inputs]), torch.cat([g[1] for g in inputs])


class _ChunkedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, probs, grad_anchor, invocation):
        ctx.invocation = invocation
        return invocation.forward(hidden, probs, track_grad=True)

    @staticmethod
    def backward(ctx, grad):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("chunked_ep_overlap requires loss.backward(), like Megatron full recompute")
        hidden_grad, prob_grad = ctx.invocation.backward(grad)
        ctx.invocation = None
        return hidden_grad, prob_grad, None, None


class _Runtime:
    """One module's immutable settings and reusable communication stream."""

    def __init__(self, experts, group, num_chunks, min_tokens_per_chunk, fused=False):
        self.experts, self.group = experts, group
        self.num_chunks, self.min_tokens_per_chunk, self.fused = num_chunks, min_tokens_per_chunk, fused
        self.comm = None
        self.device = None

    def should_chunk(self, hidden):
        if self.num_chunks == 1:
            return False
        # A rank-local decision could hang alltoall on imbalanced RL batches.
        minimum = torch.tensor(hidden.numel() // hidden.shape[-1], dtype=torch.int64, device=hidden.device)
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=self.group)
        return minimum.item() >= self.num_chunks * max(1, self.min_tokens_per_chunk)

    def __call__(self, hidden, probs, routing_map):
        sizes = chunk_sizes(hidden.shape[0], self.num_chunks)
        routing_maps = routing_map.split(sizes)
        # One metadata collective / host synchronization for all chunks, before
        # the expert window. Padding masks can make counts smaller than N*topk.
        local_counts = torch.stack([r.sum(0, dtype=torch.int64) for r in routing_maps])
        counts = [torch.empty_like(local_counts) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(counts, local_counts, group=self.group)
        counts = torch.stack(counts).cpu()
        if self.device != hidden.device:
            self.device = hidden.device
            self.comm = torch.cuda.Stream(device=hidden.device) if hidden.is_cuda else None
        streams = _Streams(hidden.device, self.comm)
        invocation = _Invocation(self.experts, routing_maps, counts, self.group, self.fused, streams)
        if torch.is_grad_enabled():
            # Ensure expert training works even when both router and input are
            # frozen. Do not expose expert leaves twice to the outer graph.
            anchor = hidden.new_empty((), requires_grad=True)
            return _ChunkedMoE.apply(hidden, probs, anchor, invocation)
        return invocation.forward(hidden, probs, track_grad=False)


def _forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
    if intermediate_tensors is not None:
        raise ValueError("chunked_ep_overlap does not support the combined-1F1B intermediate tensor interface")
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        raise ValueError("chunked_ep_overlap requires CUDA BF16 hidden states")
    runtime = self._verl_chunked_ep_runtime
    if not runtime.should_chunk(hidden_states):
        return self._verl_chunked_ep_original_forward(hidden_states, padding_mask=padding_mask)
    if padding_mask is not None:
        padding_mask = padding_mask.transpose(0, 1).bool()
    shared = self.shared_experts_compute(hidden_states)
    probs, routing_map = self.route(hidden_states, padding_mask)
    output = runtime(hidden_states.reshape(-1, hidden_states.shape[-1]), probs, routing_map)
    output = output.view_as(hidden_states)
    if shared is not None:
        output = output + shared
    return output, None


def install_chunked_ep_overlap(model, engine_config):
    """Install on standard MoELayer instances before either bridge wraps DDP."""
    options = _get(engine_config, "chunked_ep_overlap")
    if not _get(options, "enabled", False):
        return model

    from megatron.core.package_info import __version__
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from packaging.version import Version

    if Version(__version__).release[:2] != (0, 18):
        raise ValueError("chunked_ep_overlap currently supports Megatron-Core 0.18.x")
    if not torch.cuda.is_available():
        raise ValueError("chunked_ep_overlap requires CUDA")
    if _get(options, "num_chunks") > 1 and os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        raise ValueError("chunked_ep_overlap requires CUDA_DEVICE_MAX_CONNECTIONS>1 before CUDA initialization")
    layers = [m for m in model.modules() if isinstance(m, MoELayer)]
    if not layers:
        raise ValueError("chunked_ep_overlap found no standard MoELayer on this pipeline stage")
    for layer in layers:
        validate_chunked_ep_config(engine_config, transformer_config=layer.config)
        if type(layer) is not MoELayer or type(layer.token_dispatcher) is not MoEAlltoAllTokenDispatcher:
            raise ValueError("chunked_ep_overlap requires standard MoELayer / alltoall dispatcher")
        if type(layer.experts) is not TEGroupedMLP or getattr(layer.experts, "_with_fused_impl", False):
            raise ValueError("chunked_ep_overlap requires standard TEGroupedMLP experts without fused implementation")
        if layer.is_mtp_layer or getattr(layer, "_inference_token_dispatcher", None) is not None:
            raise ValueError("chunked_ep_overlap does not support MTP or inference-specific dispatchers")
        if layer.experts.activation_recompute:
            raise ValueError("chunked_ep_overlap does not support nested expert activation recompute")
    for layer in layers:
        if hasattr(layer, "_verl_chunked_ep_runtime"):
            continue
        layer._verl_chunked_ep_original_forward = layer.forward
        layer._verl_chunked_ep_runtime = _Runtime(
            layer.experts,
            layer.ep_group,
            _get(options, "num_chunks"),
            _get(options, "min_tokens_per_chunk"),
            fused=layer.config.moe_permute_fusion,
        )
        layer.forward = MethodType(_forward, layer)
    logger.info("Installed chunked EP overlap on %d MoE layers (%d chunks)", len(layers), _get(options, "num_chunks"))
    return model
