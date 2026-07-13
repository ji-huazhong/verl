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
"""Reloadable NCCL process groups for communicator memory offload.

The module replaces ``torch.distributed.new_group`` before Megatron model
parallel initialization. NCCL subgroups are returned through a stable proxy.
The real groups can then be destroyed during rollout and recreated, in their
original collective order, before training resumes.

The default process group, Gloo groups, and one-rank groups are intentionally
left untouched.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ProcessGroupStat:
    """Timing and status for one process-group destroy or reload operation."""

    name: str
    duration_ms: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class SuspendResult:
    """Aggregate result of destroying reloadable NCCL process groups."""

    success: bool = False
    skipped_reason: Optional[str] = None
    freed_mb: float = 0.0
    total_ms: float = 0.0
    groups: list[ProcessGroupStat] = field(default_factory=list)


@dataclass
class ResumeResult:
    """Aggregate result of recreating reloadable NCCL process groups."""

    success: bool = False
    skipped_reason: Optional[str] = None
    reclaimed_mb: float = 0.0
    total_ms: float = 0.0
    groups: list[ProcessGroupStat] = field(default_factory=list)


@dataclass
class _GroupRecord:
    """One collective ``new_group`` call that must be replayed on reload."""

    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    proxy: Optional[ReloadableProcessGroup] = None


@dataclass
class _PatchState:
    pid: int
    originals: dict[str, Any]
    records: list[_GroupRecord] = field(default_factory=list)
    suspended: bool = False
    sealed: bool = False


_state: Optional[_PatchState] = None


def _require_inner(group: ReloadableProcessGroup) -> dist.ProcessGroup:
    inner = group.inner_group
    if inner is None:
        raise RuntimeError("ReloadableProcessGroup is suspended; call resume_nccl_process_groups() first")
    return inner


def _unwrap(value: Any) -> Any:
    if isinstance(value, ReloadableProcessGroup):
        return _require_inner(value)
    return value


def _call_arg(args: tuple[Any, ...], kwargs: dict[str, Any], position: int, name: str, default: Any = None) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return default


def _backend_uses_nccl(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    backend = _call_arg(args, kwargs, 2, "backend")
    if backend is None:
        try:
            backend = dist.get_backend()
        except Exception:
            return False
    return "nccl" in str(backend).lower()


def _group_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    ranks = _call_arg(args, kwargs, 0, "ranks")
    if ranks is None:
        return dist.get_world_size()
    return len(ranks)


def _freeze_new_group_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Shallow-copy a new_group call and freeze its mutable ranks sequence."""
    frozen_args = list(args)
    frozen_kwargs = dict(kwargs)
    ranks = _call_arg(args, kwargs, 0, "ranks")
    if ranks is not None:
        ranks = tuple(ranks)
        if "ranks" in frozen_kwargs:
            frozen_kwargs["ranks"] = ranks
        elif frozen_args:
            frozen_args[0] = ranks
        else:
            frozen_kwargs["ranks"] = ranks
    return tuple(frozen_args), frozen_kwargs


def _group_name(args: tuple[Any, ...], kwargs: dict[str, Any], index: int) -> str:
    group_desc = _call_arg(args, kwargs, 5, "group_desc")
    if group_desc:
        return str(group_desc)
    ranks = _call_arg(args, kwargs, 0, "ranks")
    size_label = "world" if ranks is None else str(len(ranks))
    return f"nccl_group_{index}[size={size_label}]"


class ReloadableProcessGroup(dist.ProcessGroup):
    """Stable ProcessGroup identity whose underlying NCCL group can change."""

    def __init__(self, inner_group: dist.ProcessGroup):
        super().__init__(rank=inner_group.rank(), size=inner_group.size())
        self.inner_group: Optional[dist.ProcessGroup] = inner_group

    def __getattr__(self, name: str) -> Any:
        return getattr(_require_inner(self), name)

    def _forward(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return getattr(_require_inner(self), method)(*args, **kwargs)

    def rank(self) -> int:
        return self._forward("rank")

    def size(self) -> int:
        return self._forward("size")

    def name(self) -> str:
        return self._forward("name")

    def shutdown(self) -> None:
        if self.inner_group is not None:
            self.inner_group.shutdown()

    def abort(self) -> None:
        if self.inner_group is not None:
            self.inner_group.abort()

    def barrier(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("barrier", *args, **kwargs)

    def monitored_barrier(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("monitored_barrier", *args, **kwargs)

    def broadcast(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("broadcast", *args, **kwargs)

    def allreduce(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("allreduce", *args, **kwargs)

    def allreduce_coalesced(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("allreduce_coalesced", *args, **kwargs)

    def reduce(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("reduce", *args, **kwargs)

    def allgather(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("allgather", *args, **kwargs)

    def _allgather_base(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_allgather_base", *args, **kwargs)

    def allgather_coalesced(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("allgather_coalesced", *args, **kwargs)

    def allgather_into_tensor_coalesced(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("allgather_into_tensor_coalesced", *args, **kwargs)

    def gather(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("gather", *args, **kwargs)

    def scatter(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("scatter", *args, **kwargs)

    def reduce_scatter(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("reduce_scatter", *args, **kwargs)

    def _reduce_scatter_base(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_reduce_scatter_base", *args, **kwargs)

    def reduce_scatter_tensor_coalesced(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("reduce_scatter_tensor_coalesced", *args, **kwargs)

    def alltoall_base(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("alltoall_base", *args, **kwargs)

    def alltoall(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("alltoall", *args, **kwargs)

    def send(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("send", *args, **kwargs)

    def recv(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("recv", *args, **kwargs)

    def recv_anysource(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("recv_anysource", *args, **kwargs)

    def _start_coalescing(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_start_coalescing", *args, **kwargs)

    def _end_coalescing(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_end_coalescing", *args, **kwargs)

    def _get_backend_name(self) -> str:
        return self._forward("_get_backend_name")

    def _get_backend(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_get_backend", *args, **kwargs)

    def _set_default_backend(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_set_default_backend", *args, **kwargs)

    def _wait_for_pending_works(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_wait_for_pending_works", *args, **kwargs)

    def _get_sequence_number_for_group(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_get_sequence_number_for_group", *args, **kwargs)

    def _register_on_completion_hook(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_register_on_completion_hook", *args, **kwargs)

    def _enable_collectives_timing(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_enable_collectives_timing", *args, **kwargs)

    def _has_hooks(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_has_hooks", *args, **kwargs)

    def _set_group_name(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_set_group_name", *args, **kwargs)

    def _set_group_desc(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("_set_group_desc", *args, **kwargs)

    def boxed(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward("boxed", *args, **kwargs)

    @property
    def bound_device_id(self) -> Any:
        return _require_inner(self).bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, device: Any) -> None:
        _require_inner(self).bound_device_id = device

    @property
    def group_name(self) -> str:
        return _require_inner(self).group_name

    @property
    def group_desc(self) -> str:
        return _require_inner(self).group_desc


def _wrap_function(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        unwrapped_args = tuple(_unwrap(arg) for arg in args)
        unwrapped_kwargs = {key: _unwrap(value) for key, value in kwargs.items()}
        return func(*unwrapped_args, **unwrapped_kwargs)

    return wrapped


def _restore_patch(state: _PatchState) -> None:
    for name, original in state.originals.items():
        if name.startswith("P2POp."):
            setattr(dist.P2POp, name.split(".", 1)[1], original)
        elif name.startswith("c10d."):
            import torch.distributed.distributed_c10d as c10d

            setattr(c10d, name.split(".", 1)[1], original)
        else:
            setattr(dist, name, original)


def install_reloadable_process_groups() -> bool:
    """Patch torch.distributed before Megatron creates its model-parallel groups.

    Returns ``True`` when the patch is installed and ``False`` when the current
    process had already installed it.
    """
    global _state

    pid = os.getpid()
    if _state is not None and _state.pid == pid:
        return False
    if _state is not None:
        # A fork inherited Python module mutations but not a valid distributed
        # runtime. Restore the inherited functions before creating child state.
        _restore_patch(_state)

    originals: dict[str, Any] = {
        "new_group": dist.new_group,
        "destroy_process_group": dist.destroy_process_group,
    }
    state = _PatchState(pid=pid, originals=originals)
    _state = state

    original_new_group = originals["new_group"]
    original_destroy_process_group = originals["destroy_process_group"]

    def new_group(*args: Any, **kwargs: Any) -> Any:
        if state.suspended and _backend_uses_nccl(args, kwargs):
            raise RuntimeError("Cannot create a process group while reloadable NCCL groups are suspended")

        group = original_new_group(*args, **kwargs)
        if state.sealed:
            return group
        if not _backend_uses_nccl(args, kwargs) or _group_size(args, kwargs) <= 1:
            return group

        frozen_args, frozen_kwargs = _freeze_new_group_call(args, kwargs)
        record = _GroupRecord(
            name=_group_name(frozen_args, frozen_kwargs, len(state.records)),
            args=frozen_args,
            kwargs=frozen_kwargs,
        )
        # Non-members receive GroupMember.NON_GROUP_MEMBER. They still record
        # the call so every default-group rank replays new_group in the same order.
        if isinstance(group, dist.ProcessGroup):
            record.proxy = ReloadableProcessGroup(group)
            group = record.proxy
        state.records.append(record)
        return group

    def destroy_process_group(group: Optional[dist.ProcessGroup] = None) -> None:
        if isinstance(group, ReloadableProcessGroup):
            inner = _require_inner(group)
            original_destroy_process_group(inner)
            group.inner_group = None
            return
        original_destroy_process_group(group)

    dist.new_group = new_group
    dist.destroy_process_group = destroy_process_group

    # Some PyTorch and Megatron helpers call distributed_c10d functions
    # directly instead of using the torch.distributed re-exports.
    import torch.distributed.distributed_c10d as c10d

    state.originals["c10d.new_group"] = c10d.new_group
    state.originals["c10d.destroy_process_group"] = c10d.destroy_process_group
    c10d.new_group = new_group
    c10d.destroy_process_group = destroy_process_group

    function_names = (
        "get_rank",
        "get_world_size",
        "get_backend",
        "get_global_rank",
        "get_group_rank",
        "get_process_group_ranks",
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "all_gather_object",
        "all_to_all",
        "all_to_all_single",
        "broadcast",
        "broadcast_object_list",
        "reduce",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "scatter",
        "gather",
        "barrier",
        "monitored_barrier",
        "send",
        "recv",
        "isend",
        "irecv",
        "_coalescing_manager",
    )
    for name in function_names:
        func = getattr(dist, name, None)
        if func is None:
            continue
        state.originals[name] = func
        setattr(dist, name, _wrap_function(func))

        c10d_func = getattr(c10d, name, None)
        if c10d_func is not None:
            state.originals[f"c10d.{name}"] = c10d_func
            setattr(c10d, name, _wrap_function(c10d_func))

    old_isend = state.originals.get("isend")
    old_irecv = state.originals.get("irecv")

    def wrap_p2pop(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            def convert(value: Any) -> Any:
                value = _unwrap(value)
                if value is dist.isend:
                    return old_isend
                if value is dist.irecv:
                    return old_irecv
                return value

            return func(*(convert(arg) for arg in args), **{key: convert(value) for key, value in kwargs.items()})

        return wrapped

    state.originals["P2POp.__new__"] = dist.P2POp.__new__
    state.originals["P2POp.__init__"] = dist.P2POp.__init__
    dist.P2POp.__new__ = wrap_p2pop(dist.P2POp.__new__)
    dist.P2POp.__init__ = wrap_p2pop(dist.P2POp.__init__)

    logger.info("Installed reloadable NCCL process-group support in pid %d", pid)
    return True


def is_installed() -> bool:
    return _state is not None and _state.pid == os.getpid()


def seal_reloadable_process_groups() -> None:
    """Stop registering groups after Megatron parallel-state initialization."""
    state = _state
    if state is not None and state.pid == os.getpid():
        state.sealed = True
        logger.info("Sealed %d reloadable NCCL process-group records", len(state.records))


def _gpu_used_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def suspend_nccl_process_groups() -> SuspendResult:
    """Destroy all active reloadable NCCL groups in collective creation order."""
    state = _state
    if state is None or state.pid != os.getpid():
        return SuspendResult(skipped_reason="not_installed")
    if state.suspended:
        return SuspendResult(skipped_reason="already_suspended")
    if not state.records:
        return SuspendResult(skipped_reason="no_process_groups")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    used_before = _gpu_used_mb()
    started = time.perf_counter()
    stats: list[ProcessGroupStat] = []
    failures: list[str] = []

    for record in state.records:
        if record.proxy is None:
            continue
        inner = record.proxy.inner_group
        if inner is None:
            continue
        group_started = time.perf_counter()
        try:
            state.originals["destroy_process_group"](inner)
            record.proxy.inner_group = None
            stats.append(ProcessGroupStat(name=record.name, duration_ms=(time.perf_counter() - group_started) * 1000))
        except Exception as exc:
            failures.append(f"{record.name}: {exc}")
            stats.append(
                ProcessGroupStat(
                    name=record.name,
                    duration_ms=(time.perf_counter() - group_started) * 1000,
                    success=False,
                    error=str(exc),
                )
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - started) * 1000
    freed_mb = used_before - _gpu_used_mb()
    if failures:
        raise RuntimeError("Failed to destroy reloadable NCCL process groups: " + "; ".join(failures))

    state.suspended = True
    logger.info("Destroyed %d reloadable NCCL groups in %.0f ms, freed %.0f MB", len(stats), total_ms, freed_mb)
    return SuspendResult(success=True, freed_mb=freed_mb, total_ms=total_ms, groups=stats)


def resume_nccl_process_groups() -> ResumeResult:
    """Recreate all tracked NCCL groups by replaying their original calls."""
    state = _state
    if state is None or state.pid != os.getpid():
        return ResumeResult(skipped_reason="not_installed")
    if not state.suspended:
        return ResumeResult(skipped_reason="not_suspended")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    used_before = _gpu_used_mb()
    started = time.perf_counter()
    stats: list[ProcessGroupStat] = []
    failures: list[str] = []
    original_new_group = state.originals["new_group"]

    for record in state.records:
        group_started = time.perf_counter()
        try:
            group = original_new_group(*record.args, **record.kwargs)
            if record.proxy is not None:
                if not isinstance(group, dist.ProcessGroup):
                    raise RuntimeError("rank was a group member before suspend but is not a member after reload")
                record.proxy.inner_group = group
                stats.append(
                    ProcessGroupStat(name=record.name, duration_ms=(time.perf_counter() - group_started) * 1000)
                )
            elif isinstance(group, dist.ProcessGroup):
                # Membership cannot legitimately change while the default world
                # group is alive. Destroy the unexpected group before failing.
                state.originals["destroy_process_group"](group)
                raise RuntimeError("rank was a non-member before suspend but became a member after reload")
        except Exception as exc:
            failures.append(f"{record.name}: {exc}")
            stats.append(
                ProcessGroupStat(
                    name=record.name,
                    duration_ms=(time.perf_counter() - group_started) * 1000,
                    success=False,
                    error=str(exc),
                )
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - started) * 1000
    reclaimed_mb = _gpu_used_mb() - used_before
    if failures:
        raise RuntimeError("Failed to reload NCCL process groups: " + "; ".join(failures))

    state.suspended = False
    logger.info("Reloaded %d NCCL groups in %.0f ms, reclaimed %.0f MB", len(stats), total_ms, reclaimed_mb)
    return ResumeResult(success=True, reclaimed_mb=reclaimed_mb, total_ms=total_ms, groups=stats)


def log_aggregate_summary(action: str, results: Any, *, size_attr: str, size_verb: str) -> None:
    """Aggregate one suspend/resume result per training rank into one log line."""
    valid = [result for result in (results or []) if result is not None]
    if not valid:
        return
    skipped = [result for result in valid if result.skipped_reason]
    if skipped:
        reasons = sorted({result.skipped_reason for result in skipped})
        logger.info("NCCL process-group %s skipped on %d/%d ranks: %s", action, len(skipped), len(valid), reasons)
    actual = [result for result in valid if not result.skipped_reason]
    if not actual:
        return
    sizes = [getattr(result, size_attr) for result in actual]
    durations = [result.total_ms for result in actual]
    logger.info(
        "NCCL process-group %s: %d/%d ranks succeeded, %s %.0f MB avg (range %.0f-%.0f), %.0f ms avg (range %.0f-%.0f)",
        action,
        sum(1 for result in actual if result.success),
        len(actual),
        size_verb,
        sum(sizes) / len(sizes),
        min(sizes),
        max(sizes),
        sum(durations) / len(durations),
        min(durations),
        max(durations),
    )


def _reset_for_testing() -> None:
    """Restore torch.distributed globals. Intended only for isolated unit tests."""
    global _state
    if _state is not None:
        _restore_patch(_state)
    _state = None
