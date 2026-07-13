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

from __future__ import annotations

from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from verl.utils import reloadable_process_group as rpg


class _FakeProcessGroup(dist.ProcessGroup):
    def __init__(self, rank: int, size: int, generation: int):
        super().__init__(rank=rank, size=size)
        self._rank = rank
        self._size = size
        self.generation = generation

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size

    def name(self) -> str:
        return "fake_nccl"


@pytest.fixture
def fake_distributed(monkeypatch):
    rpg._reset_for_testing()
    current_rank = 0
    created = []
    destroyed = []
    rank_queries = []

    def fake_new_group(
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None,
        device_id=None,
    ):
        ranks = tuple(range(4)) if ranks is None else tuple(ranks)
        call = {
            "ranks": ranks,
            "timeout": timeout,
            "backend": backend,
            "pg_options": pg_options,
            "use_local_synchronization": use_local_synchronization,
            "group_desc": group_desc,
            "device_id": device_id,
        }
        created.append(call)
        if current_rank not in ranks:
            return dist.GroupMember.NON_GROUP_MEMBER
        return _FakeProcessGroup(rank=ranks.index(current_rank), size=len(ranks), generation=len(created))

    def fake_destroy_process_group(group=None):
        destroyed.append(group)

    def fake_get_rank(group=None):
        rank_queries.append(group)
        return 0 if group is None else group.rank()

    monkeypatch.setattr(dist, "new_group", fake_new_group)
    monkeypatch.setattr(dist, "destroy_process_group", fake_destroy_process_group)
    monkeypatch.setattr(dist, "get_backend", lambda group=None: "nccl")
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4 if group is None else group.size())
    monkeypatch.setattr(dist, "get_rank", fake_get_rank)
    monkeypatch.setattr(c10d, "get_backend", lambda group=None: "nccl")
    monkeypatch.setattr(c10d, "get_world_size", lambda group=None: 4 if group is None else group.size())
    monkeypatch.setattr(c10d, "get_rank", fake_get_rank)

    rpg.install_reloadable_process_groups()
    yield created, destroyed, rank_queries
    rpg._reset_for_testing()


def test_destroy_and_reload_preserve_proxy_and_group_spec(fake_distributed):
    created, destroyed, rank_queries = fake_distributed
    timeout = timedelta(minutes=3)

    group = dist.new_group(
        ranks=[0, 1],
        timeout=timeout,
        backend="nccl",
        use_local_synchronization=True,
        group_desc="tensor_parallel",
    )
    assert isinstance(group, rpg.ReloadableProcessGroup)
    original_inner = group.inner_group
    assert dist.get_rank(group) == 0
    assert rank_queries[-1] is original_inner
    assert c10d.get_rank(group) == 0

    suspend_result = rpg.suspend_nccl_process_groups()
    assert suspend_result.success
    assert destroyed == [original_inner]
    assert group.inner_group is None
    with pytest.raises(RuntimeError, match="suspended"):
        dist.get_rank(group)
    with pytest.raises(RuntimeError, match="suspended"):
        c10d.get_rank(group)

    resume_result = rpg.resume_nccl_process_groups()
    assert resume_result.success
    assert group.inner_group is not None
    assert group.inner_group is not original_inner
    assert created[-1]["ranks"] == (0, 1)
    assert created[-1]["timeout"] == timeout
    assert created[-1]["backend"] == "nccl"
    assert created[-1]["use_local_synchronization"] is True
    assert created[-1]["group_desc"] == "tensor_parallel"
    assert dist.get_rank(group) == 0


def test_gloo_and_single_rank_groups_are_not_wrapped(fake_distributed):
    gloo_group = dist.new_group(ranks=[0, 1], backend="gloo")
    single_rank_group = dist.new_group(ranks=[0], backend="nccl")

    assert isinstance(gloo_group, _FakeProcessGroup)
    assert isinstance(single_rank_group, _FakeProcessGroup)

    result = rpg.suspend_nccl_process_groups()
    assert not result.success
    assert result.skipped_reason == "no_process_groups"


def test_non_member_replays_collective_new_group_call(fake_distributed):
    created, destroyed, _ = fake_distributed
    member_group = dist.new_group(ranks=[0, 1], backend="nccl", group_desc="member_group")
    non_member = dist.new_group(ranks=[2, 3], backend="nccl", group_desc="non_member_group")

    assert isinstance(member_group, rpg.ReloadableProcessGroup)
    assert non_member == dist.GroupMember.NON_GROUP_MEMBER

    rpg.suspend_nccl_process_groups()
    rpg.resume_nccl_process_groups()

    assert destroyed
    assert [call["group_desc"] for call in created] == [
        "member_group",
        "non_member_group",
        "member_group",
        "non_member_group",
    ]


def test_install_and_lifecycle_calls_are_idempotent(fake_distributed):
    dist.new_group(ranks=[0, 1], backend="nccl")

    assert not rpg.install_reloadable_process_groups()
    assert rpg.suspend_nccl_process_groups().success
    assert rpg.suspend_nccl_process_groups().skipped_reason == "already_suspended"
    assert rpg.resume_nccl_process_groups().success
    assert rpg.resume_nccl_process_groups().skipped_reason == "not_suspended"


def test_seal_excludes_runtime_process_groups(fake_distributed):
    created, destroyed, _ = fake_distributed
    tracked = dist.new_group(ranks=[0, 1], backend="nccl", group_desc="megatron_group")
    rpg.seal_reloadable_process_groups()
    transient = dist.new_group(ranks=[0, 1], backend="nccl", group_desc="checkpoint_group")

    assert isinstance(tracked, rpg.ReloadableProcessGroup)
    assert isinstance(transient, _FakeProcessGroup)

    rpg.suspend_nccl_process_groups()
    rpg.resume_nccl_process_groups()

    assert len(destroyed) == 1
    assert destroyed[0] is not transient
    assert [call["group_desc"] for call in created] == [
        "megatron_group",
        "checkpoint_group",
        "megatron_group",
    ]
