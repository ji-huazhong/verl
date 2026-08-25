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

import json

import pytest
import torch

from verl.utils.offload import DiskOffloadStore


def _new_store(tmp_path, *, cleanup_on_exit=False):
    return DiskOffloadStore(
        str(tmp_path),
        rank=3,
        chunk_size_mb=1,
        cleanup_on_exit=cleanup_on_exit,
        job_id="test-job",
    )


def test_disk_store_round_trip_and_layout_reuse(tmp_path):
    store = _new_store(tmp_path)
    first = torch.arange(300_000, dtype=torch.float32)
    second = torch.arange(257, dtype=torch.int64)
    expected_first = first.clone()
    expected_second = second.clone()

    store.write_tensors("param", [("first", first), ("second", second)])
    first_metadata = store.metadata("param", "first")
    second_metadata = store.metadata("param", "second")
    assert first_metadata.device_type == "cpu"
    state_path = store.root / "param" / "state.bin"
    initial_file_size = state_path.stat().st_size

    first.zero_()
    second.zero_()
    store.read_tensors("param", [("first", first), ("second", second)])
    torch.testing.assert_close(first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second, expected_second, rtol=0, atol=0)

    first.add_(1)
    store.write_tensors("param", [("first", first), ("second", second)])
    assert store.metadata("param", "first").offset == first_metadata.offset
    assert store.metadata("param", "second").offset == second_metadata.offset
    assert state_path.stat().st_size == initial_file_size


def test_disk_store_rejects_layout_changes(tmp_path):
    store = _new_store(tmp_path)
    store.write_tensors("optimizer", [("moment", torch.ones(8, dtype=torch.float32))])

    with pytest.raises(ValueError, match="layout changed"):
        store.write_tensors("optimizer", [("moment", torch.ones(9, dtype=torch.float32))])


def test_disk_store_rejects_uncommitted_generation(tmp_path):
    store = _new_store(tmp_path)
    tensor = torch.ones(8, dtype=torch.float32)
    store.write_tensors("grad", [("grad", tensor)])
    store.invalidate("grad")

    with pytest.raises(RuntimeError, match="No committed grad"):
        store.read_tensors("grad", [("grad", tensor)])


def test_disk_store_uses_one_data_file_per_component(tmp_path):
    store = _new_store(tmp_path)
    store.write_tensors("param", [(f"tensor-{index}", torch.ones(4)) for index in range(10)])

    component_files = sorted(path.name for path in (store.root / "param").iterdir())
    assert component_files == ["generation", "manifest.json", "state.bin"]
    manifest = json.loads((store.root / "param" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 10


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
        job_id="test-job",
    )

    first.write_tensors("param", [("weight", torch.tensor([1.0]))])
    second.write_tensors("param", [("weight", torch.tensor([2.0]))])
    first_target = torch.zeros(1)
    second_target = torch.zeros(1)
    first.read_tensors("param", [("weight", first_target)])
    second.read_tensors("param", [("weight", second_target)])

    assert first.root != second.root
    assert first_target.item() == 1.0
    assert second_target.item() == 2.0
