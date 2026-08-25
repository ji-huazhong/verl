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

import pytest
import torch

from verl.utils.offload import DiskOffloadStore


def _new_store(tmp_path, *, cleanup_on_exit=False):
    return DiskOffloadStore(
        str(tmp_path),
        rank=3,
        chunk_size_mb=1,
        cleanup_on_exit=cleanup_on_exit,
    )


def test_disk_store_round_trip_across_repeated_writes(tmp_path):
    store = _new_store(tmp_path)
    first = torch.arange(300_000, dtype=torch.float32)
    second = torch.arange(257, dtype=torch.int64)
    expected_first = first.clone()
    expected_second = second.clone()

    store.write_tensors("param", [("first", first), ("second", second)])
    first.zero_()
    second.zero_()
    store.read_tensors("param", [("first", first), ("second", second)])
    torch.testing.assert_close(first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second, expected_second, rtol=0, atol=0)

    first.add_(1)
    second.add_(2)
    expected_first = first.clone()
    expected_second = second.clone()
    store.write_tensors("param", [("first", first), ("second", second)])
    first.zero_()
    second.zero_()
    store.read_tensors("param", [("first", first), ("second", second)])
    torch.testing.assert_close(first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second, expected_second, rtol=0, atol=0)


def test_disk_store_rejects_layout_changes(tmp_path):
    store = _new_store(tmp_path)
    store.write_tensors("optimizer", [("moment", torch.ones(8, dtype=torch.float32))])

    with pytest.raises(ValueError, match="layout changed"):
        store.write_tensors("optimizer", [("moment", torch.ones(9, dtype=torch.float32))])


def test_disk_store_does_not_publish_failed_write(tmp_path, monkeypatch):
    store = _new_store(tmp_path)
    tensor = torch.ones(8, dtype=torch.float32)
    store.write_tensors("grad", [("grad", tensor)])

    def fail_write(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(store, "_write_tensors_pipelined", fail_write)
    with pytest.raises(OSError, match="write failed"):
        store.write_tensors("grad", [("grad", tensor)])

    with pytest.raises(RuntimeError, match="No complete grad"):
        store.read_tensors("grad", [("grad", tensor)])


def test_disk_store_keeps_components_independent(tmp_path):
    store = _new_store(tmp_path)
    param = torch.tensor([1.0])
    optimizer = torch.tensor([2.0])
    store.write_tensors("param", [("state", param)])
    store.write_tensors("optimizer", [("state", optimizer)])

    param.zero_()
    optimizer.zero_()
    store.read_tensors("param", [("state", param)])
    store.read_tensors("optimizer", [("state", optimizer)])

    assert param.item() == 1.0
    assert optimizer.item() == 2.0


def test_disk_store_only_cleans_its_owned_directory(tmp_path):
    store = _new_store(tmp_path, cleanup_on_exit=True)
    sibling = store.root.parent / "store_sibling"
    sibling.mkdir(parents=True)

    root = store.root
    store.close()

    assert not root.exists()
    assert sibling.exists()


def test_disk_store_isolates_store_instances(tmp_path):
    first = _new_store(tmp_path)
    second = DiskOffloadStore(
        str(tmp_path),
        rank=3,
        chunk_size_mb=1,
        cleanup_on_exit=False,
    )

    first.write_tensors("param", [("weight", torch.tensor([1.0]))])
    second.write_tensors("param", [("weight", torch.tensor([2.0]))])
    first_target = torch.zeros(1)
    second_target = torch.zeros(1)
    first.read_tensors("param", [("weight", first_target)])
    second.read_tensors("param", [("weight", second_target)])

    assert first.root != second.root
    assert first.root.name.startswith("store_3_")
    assert second.root.name.startswith("store_3_")
    assert first_target.item() == 1.0
    assert second_target.item() == 2.0
