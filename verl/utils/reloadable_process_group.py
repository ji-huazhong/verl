# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Reloadable NCCL process groups for Megatron training.

Enables destroying and rebuilding all NCCL communicators between training steps so
colocated inference workers can use the freed GPU memory during rollout, then
restoring them when training resumes.

Usage:
    # At engine init (before mpu.initialize_model_parallel):
    monkey_patch_torch_dist()

    # To put training to sleep (e.g. before rollout):
    destroy_process_groups()

    # To wake up for training / weight export:
    reload_process_groups()
"""

import gc
import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from verl.utils.device import get_torch_device

logger = logging.getLogger(__name__)

# Maps pid -> original dist.new_group before patching
_old_new_group_dict: dict[int, object] = {}

# Pids that have nccl_offload enabled.  Once any engine in the process sets
# nccl_offload=True, ALL engines in the same process must respect the
# destroy/reload lifecycle because they share the same Megatron process groups
# (mpu.is_initialized() guard prevents duplicate initialization).
_nccl_offload_pids: set[int] = set()

# Free memory threshold (GB) below which we flush cache before NCCL calls
_LOW_MEM_THRESHOLD_GB = 3.0


def is_nccl_offload_enabled() -> bool:
    """Return True if nccl_offload is active in the current process.

    This is a process-level flag: once *any* engine in the process enables
    nccl_offload (by calling :func:`monkey_patch_torch_dist`), every engine
    sharing the same Megatron process groups must participate in the
    destroy/reload cycle.
    """
    return os.getpid() in _nccl_offload_pids


def monkey_patch_torch_dist() -> None:
    """Intercept ``dist.new_group`` so every created NCCL group is wrapped in a
    :class:`ReloadableProcessGroup`.  Also wraps all collective / P2P functions so
    they transparently unwrap the wrapper before calling the real PyTorch op.

    The function is idempotent: calling it multiple times in the same process is safe.
    """
    pid = os.getpid()
    _nccl_offload_pids.add(pid)  # mark process as nccl_offload enabled

    if pid in _old_new_group_dict:
        # Already patched in this process; verify consistency.
        assert dist.old_new_group == _old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed (pid=%d)", pid)

    old_new_group = dist.new_group
    _old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        group = old_new_group(*args, **kwargs)

        # Leave gloo groups unwrapped – they are used for CPU barriers and do
        # not consume significant GPU memory.
        backend = kwargs.get("backend")
        if backend is None and len(args) >= 3:
            backend = args[2]
        backend = backend or "hccl"

        is_gloo = backend == "gloo"
        if is_gloo:
            return group

        # Determine the rank list for later reconstruction.
        if len(args) >= 1 and args[0] is not None:
            ranks = list(args[0])
        elif kwargs.get("ranks") is not None:
            ranks = list(kwargs["ranks"])
        else:
            ranks = list(range(dist.get_world_size()))

        # Single-rank groups don't need NCCL communication; skip wrapping.
        if len(ranks) <= 1:
            return group

        return ReloadableProcessGroup(group, ranks, backend=backend)

    dist.new_group = new_group

    # ------------------------------------------------------------------
    # Helper factories
    # ------------------------------------------------------------------

    def _query_wrapper(func):
        """Wrap query functions (get_rank, get_world_size, …) without memory check.

        Only unwrap a ReloadableProcessGroup when its inner group is alive.  If
        the group has been destroyed (inner == None) we keep the wrapper object
        so that callers like dist.get_world_size() delegate to our size() /
        rank() methods, which return the stored values rather than falling back
        to the global world size.
        """

        def _unwrap(arg):
            if isinstance(arg, ReloadableProcessGroup) and arg.group is not None:
                return arg.group
            return arg

        def wrapped(*args, **kwargs):
            func_name = getattr(func, "__name__", "")

            # When the wrapped group is in destroyed state (inner group == None),
            # some query APIs (e.g. get_rank/get_group_rank) cannot accept the
            # wrapper object because c10d checks a global "registered group" map.
            # In that case, answer from cached metadata directly.
            maybe_group = kwargs.get("group", args[0] if len(args) >= 1 else None)
            if isinstance(maybe_group, ReloadableProcessGroup) and maybe_group.group is None:
                ranks = maybe_group.group_info["ranks"]
                if func_name == "get_rank":
                    return maybe_group.rank()
                if func_name == "get_world_size":
                    return maybe_group.size()
                if func_name == "get_backend":
                    return maybe_group.group_info.get("backend", "nccl")
                if func_name == "get_process_group_ranks":
                    return list(ranks)
                if func_name == "get_group_rank":
                    global_rank = kwargs.get("global_rank", args[1] if len(args) >= 2 else None)
                    if global_rank in ranks:
                        return ranks.index(global_rank)
                    raise ValueError(f"Global rank {global_rank} is not part of this process group")
                if func_name == "get_global_rank":
                    group_rank = kwargs.get("group_rank", args[1] if len(args) >= 2 else None)
                    if group_rank is None or group_rank < 0 or group_rank >= len(ranks):
                        raise ValueError(f"Group rank {group_rank} is out of range [0, {len(ranks)})")
                    return ranks[group_rank]

            args = tuple(_unwrap(a) for a in args)
            kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return wrapped

    def _comm_wrapper(func):
        """Wrap collective communication functions with pre-call memory check."""

        def _unwrap(arg):
            if isinstance(arg, ReloadableProcessGroup):
                if arg.group is None:
                    raise RuntimeError(
                        f"Attempted to use a destroyed ReloadableProcessGroup in "
                        f"{func.__name__}. Call reload_process_groups() first."
                    )
                return arg.group
            return arg

        def wrapped(*args, **kwargs):
            args = tuple(_unwrap(a) for a in args)
            kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
            with _wrap_low_level_call():
                return func(*args, **kwargs)

        return wrapped

    # Query functions
    dist.get_rank = _query_wrapper(dist.get_rank)
    dist.get_world_size = _query_wrapper(dist.get_world_size)
    dist.get_backend = _query_wrapper(dist.get_backend)
    dist.get_global_rank = _query_wrapper(dist.get_global_rank)
    dist.get_group_rank = _query_wrapper(dist.get_group_rank)
    dist.get_process_group_ranks = _query_wrapper(dist.get_process_group_ranks)

    # Collective / point-to-point functions
    dist.all_reduce = _comm_wrapper(dist.all_reduce)
    dist.all_gather = _comm_wrapper(dist.all_gather)
    dist.all_gather_into_tensor = _comm_wrapper(dist.all_gather_into_tensor)
    dist.all_gather_object = _comm_wrapper(dist.all_gather_object)
    dist.all_to_all = _comm_wrapper(dist.all_to_all)
    dist.all_to_all_single = _comm_wrapper(dist.all_to_all_single)
    dist.broadcast = _comm_wrapper(dist.broadcast)
    dist.broadcast_object_list = _comm_wrapper(dist.broadcast_object_list)
    dist.reduce = _comm_wrapper(dist.reduce)
    dist.reduce_scatter = _comm_wrapper(dist.reduce_scatter)
    dist.reduce_scatter_tensor = _comm_wrapper(dist.reduce_scatter_tensor)
    dist.scatter = _comm_wrapper(dist.scatter)
    dist.gather = _comm_wrapper(dist.gather)
    dist.barrier = _comm_wrapper(dist.barrier)
    dist.send = _comm_wrapper(dist.send)
    dist.recv = _comm_wrapper(dist.recv)
    dist._coalescing_manager = _comm_wrapper(dist._coalescing_manager)

    # isend / irecv need special handling because P2POp stores the function
    # reference itself; we must map the patched back to the original.
    _old_isend = dist.isend
    _old_irecv = dist.irecv
    dist.isend = _comm_wrapper(dist.isend)
    dist.irecv = _comm_wrapper(dist.irecv)

    def _p2pop_wrapper(func):
        def wrapped(*args, **kwargs):
            def _convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                if arg is dist.isend:
                    return _old_isend
                if arg is dist.irecv:
                    return _old_irecv
                return arg

            args = tuple(_convert(a) for a in args)
            kwargs = {k: _convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return wrapped

    dist.P2POp.__new__ = _p2pop_wrapper(dist.P2POp.__new__)
    dist.P2POp.__init__ = _p2pop_wrapper(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    """A thin wrapper around a real :class:`torch.distributed.ProcessGroup` that
    can be destroyed (setting the inner group to ``None``) and later reconstructed
    from the saved rank list.

    All instances for the current process are registered in the class-level
    ``GROUPS`` dict so :func:`destroy_process_groups` and
    :func:`reload_process_groups` can operate on them in bulk.
    """

    GROUPS: dict[int, list["ReloadableProcessGroup"]] = {}

    def __init__(self, group: torch.distributed.ProcessGroup, ranks: list[int], backend: str = "nccl") -> None:
        _rank = dist.get_rank(group)
        _size = dist.get_world_size(group)
        super().__init__(rank=_rank, size=_size)
        self._stored_rank = _rank
        self._stored_size = _size
        self.group = group
        self.group_info = {
            "ranks": ranks,
            "backend": backend,
        }

        pid = os.getpid()
        ReloadableProcessGroup.GROUPS.setdefault(pid, []).append(self)

    # Forward unknown attribute access to the inner group so callers using
    # the raw ProcessGroup API continue to work.
    def __getattr__(self, name: str):
        return getattr(self.group, name)

    # ------------------------------------------------------------------
    # Bulk lifecycle operations
    # ------------------------------------------------------------------

    @staticmethod
    def destroy_process_groups() -> None:
        """Destroy all registered NCCL groups for the current process.

        Idempotent: groups that are already ``None`` are skipped.
        """
        pid = os.getpid()
        for rpg in ReloadableProcessGroup.GROUPS.get(pid, []):
            if rpg.group is None:
                continue
            try:
                dist.destroy_process_group(rpg.group)
            except ValueError as exc:
                logger.warning(
                    "Process group already invalid/destroyed; skipping cleanup. Exception: %s",
                    exc,
                    exc_info=True,
                )
            del rpg.group
            rpg.group = None

    @staticmethod
    def reload_process_groups() -> None:
        """Rebuild all registered NCCL groups for the current process.

        Idempotent: groups that are already alive are skipped.
        """
        pid = os.getpid()
        groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        logger.info("Reloading %d process groups in pid %d", len(groups), pid)
        old_new_group = _old_new_group_dict.get(pid)
        if old_new_group is None:
            raise RuntimeError(
                "reload_process_groups called but monkey_patch_torch_dist() was never applied in this process."
            )
        for rpg in groups:
            if rpg.group is not None:
                continue
            rpg.group = old_new_group(ranks=rpg.group_info["ranks"], backend="nccl")

    # ------------------------------------------------------------------
    # ProcessGroup interface forwarding
    # ------------------------------------------------------------------

    def rank(self) -> int:
        if self.group is None:
            return self._stored_rank
        return self.group.rank()

    def size(self) -> int:
        if self.group is None:
            return self._stored_size
        return self.group.size()

    def name(self) -> str:
        return self.group.name()

    def shutdown(self) -> None:
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method: str, *args, **kwargs):
        """Forward a communication method with pre-call memory guard."""
        inner = self.group
        if inner is None:
            raise RuntimeError(
                f"ReloadableProcessGroup: inner process group is None when calling '{method}'. "
                "Call reload_process_groups() first."
            )
        with _wrap_low_level_call():
            return getattr(inner, method)(*args, **kwargs)

    def _fwd_query(self, method: str, *args, **kwargs):
        """Forward a non-communication (query) method without memory guard."""
        inner = self.group
        if inner is None:
            raise RuntimeError(
                f"ReloadableProcessGroup: inner process group is None when calling '{method}'. "
                "Call reload_process_groups() first."
            )
        return getattr(inner, method)(*args, **kwargs)

    # Collective operations
    def barrier(self, *a, **kw):
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, *a, **kw):
        return self._fwd("allgather_into_tensor_coalesced", *a, **kw)

    def gather(self, *a, **kw):
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, *a, **kw):
        return self._fwd("reduce_scatter_tensor_coalesced", *a, **kw)

    def alltoall_base(self, *a, **kw):
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        return self._fwd("recv_anysource", *a, **kw)

    def _start_coalescing(self, *a, **kw):
        return self._fwd_query("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd_query("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd_query("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd_query("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def destroy_process_groups() -> None:
    """Destroy all reloadable NCCL process groups in the current process."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups() -> None:
    """Rebuild all reloadable NCCL process groups in the current process."""
    ReloadableProcessGroup.reload_process_groups()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@contextmanager
def _wrap_low_level_call():
    """Context manager that guards every low-level NCCL call.

    - Proactively flushes GPU cache if free memory is below the threshold.
    - Annotates exceptions with GPU memory state for easier post-mortem debugging.
    """
    try:
        device = get_torch_device()
        if device.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes < _LOW_MEM_THRESHOLD_GB * 1024**3:
                    gc.collect()
                    device.empty_cache()
            except Exception:
                pass
        yield
    except Exception as exc:
        try:
            device = get_torch_device()
            if device.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                mem_note = (
                    f"GPU memory at NCCL error: free={free_bytes / 1024**3:.2f} GB, "
                    f"total={total_bytes / 1024**3:.2f} GB, "
                    f"allocated={device.memory_allocated() / 1024**3:.2f} GB"
                )
                exc.add_note(mem_note)
        except Exception:
            pass
        raise
