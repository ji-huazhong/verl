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

"""Opt-in, bounded CUDA-event sampling of training-side INT4 fake quantization."""

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

from verl.utils.device import get_torch_device

# Autograd worker threads do not inherit Python ContextVars. The Megatron
# schedule is process-wide; reject overlapping profiled schedules explicitly.
_active_profile = None
_schedule_lock = threading.Lock()
_inside_forward: ContextVar = ContextVar("int4_qat_inside_forward", default=False)
_profiled_schedules = 0


def current_int4_qat_profile():
    """Return the active schedule's profiler, including on autograd threads."""
    return _active_profile


class Int4QATProfile:
    """Count all QDQ calls while sampling a bounded number of stream spans.

    Event spans include stream scheduling and host launch gaps, not just kernel
    execution. Sampled milliseconds are never extrapolated to a full-stage
    GPU time. Profiling synchronizes sampled events only at the stage boundary.
    """

    def __init__(self, stage: str, num_microbatches: int):
        self.stage = stage
        self.num_microbatches = num_microbatches
        self.sample_every = int(os.environ.get("VERL_INT4_QAT_PROFILE_SAMPLE_EVERY", "1024"))
        self.max_samples = int(os.environ.get("VERL_INT4_QAT_PROFILE_MAX_SAMPLES", "128"))
        if self.sample_every < 1 or self.max_samples < 0:
            raise ValueError("INT4 QAT profiling requires sample_every >= 1 and max_samples >= 0")
        self.groups = {}
        self.memory_start = {}
        self.device_api = get_torch_device()
        self.lock = threading.Lock()
        self.started = time.perf_counter()

    def wrap_forward_step(self, forward_step):
        """Distinguish schedule forward callbacks from backward/recompute QDQ."""

        @wraps(forward_step)
        def wrapped(*args, **kwargs):
            token = _inside_forward.set(True)
            try:
                return forward_step(*args, **kwargs)
            finally:
                _inside_forward.reset(token)

        return wrapped

    @contextmanager
    def measure(self, weight: torch.Tensor, group_size: int):
        """Measure one QDQ invocation without retaining its input or output."""
        phase = "forward" if _inside_forward.get() else "backward_or_recompute"
        if self.stage == "logprob":
            phase = "forward"
        key = (phase, tuple(weight.shape), str(weight.dtype), str(weight.device), group_size)
        sample = None
        with self.lock:
            if key not in self.groups:
                self.groups[key] = {"calls": 0, "numel": 0, "output_bytes": 0, "host_seconds": 0.0, "events": []}
            group = self.groups[key]
            group["calls"] += 1
            group["numel"] += weight.numel()
            group["output_bytes"] += weight.numel() * weight.element_size()
            if weight.is_cuda:
                device = weight.device
                if device not in self.memory_start:
                    self.memory_start[device] = self.device_api.memory_allocated(device)
                if (group["calls"] - 1) % self.sample_every == 0 and len(group["events"]) < self.max_samples:
                    # Reserve the sample before yielding so concurrent autograd
                    # calls cannot exceed the per-group cap.
                    stream = self.device_api.current_stream(device)
                    sample = (
                        self.device_api.Event(enable_timing=True),
                        self.device_api.Event(enable_timing=True),
                        stream,
                    )
                    sample[0].record(stream)
                    group["events"].append(sample[:2])
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            if sample is not None:
                sample[1].record(sample[2])
            with self.lock:
                group["host_seconds"] += elapsed

    def report(self, complete: bool) -> dict:
        """Resolve sampled events and return JSON-safe counters and memory stats."""
        host_seconds = time.perf_counter() - self.started
        groups = []
        for key, group in self.groups.items():
            for _, end in group["events"]:
                end.synchronize()
            spans = [begin.elapsed_time(end) for begin, end in group["events"]]
            phase, shape, dtype, device, group_size = key
            groups.append(
                {
                    "phase": phase,
                    "shape": shape,
                    "dtype": dtype,
                    "device": device,
                    "group_size": group_size,
                    **{k: v for k, v in group.items() if k != "events"},
                    "sampled_calls": len(spans),
                    "sampled_cuda_span_ms": sum(spans),
                }
            )
        return {
            "stage": self.stage,
            "complete": complete,
            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            "num_microbatches": self.num_microbatches,
            "host_schedule_seconds": host_seconds,
            "sample_every": self.sample_every,
            "max_samples_per_group": self.max_samples,
            "groups": groups,
            "memory": {
                str(device): {
                    "allocated_before_bytes": before,
                    "allocated_after_bytes": self.device_api.memory_allocated(device),
                    "reserved_bytes": self.device_api.memory_reserved(device),
                    "process_peak_allocated_bytes": self.device_api.max_memory_allocated(device),
                }
                for device, before in self.memory_start.items()
            },
        }


@contextmanager
def int4_qat_profile(stage: str, num_microbatches: int, *, enabled: bool):
    """Enable diagnostic counters only for explicitly requested INT4 schedules."""
    global _active_profile, _profiled_schedules

    if not enabled or os.environ.get("VERL_INT4_QAT_TRAIN_PROFILE", "0") != "1":
        yield None
        return
    max_schedules = int(os.environ.get("VERL_INT4_QAT_PROFILE_MAX_SCHEDULES", "0"))
    if max_schedules < 0:
        raise ValueError("VERL_INT4_QAT_PROFILE_MAX_SCHEDULES must be nonnegative")
    if not _schedule_lock.acquire(blocking=False):
        raise RuntimeError("INT4 QAT profiling requires one active Megatron schedule per process")
    try:
        if max_schedules and _profiled_schedules >= max_schedules:
            yield None
            return
        profile = Int4QATProfile(stage, num_microbatches)
        _profiled_schedules += 1
        _active_profile = profile
        complete = False
        try:
            yield profile
            complete = True
        finally:
            _active_profile = None
            # Explicit opt-in diagnostics must survive framework logger filters.
            # Ray console deduplication may still merge ranks; disable it or use
            # original per-worker stderr files when auditing every rank.
            print("INT4_QAT_TRAIN_PROFILE " + json.dumps(profile.report(complete)), file=sys.stderr, flush=True)
    finally:
        _schedule_lock.release()
