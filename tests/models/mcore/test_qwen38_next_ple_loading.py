# SPDX-License-Identifier: Apache-2.0
"""CPU gates for the frozen table's direct checkpoint reader, not model parity."""

import builtins
import json
import struct
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("via_flag", [False, True])
def test_meta_table_does_not_allocate_host_storage_or_read_payload(tmp_path, monkeypatch, via_flag):
    from torch.utils._python_dispatch import TorchDispatchMode

    from verl.models.mcore.qwen3_8_next.ops import ple

    path, _, _ = make_checkpoint(tmp_path)
    config = SimpleNamespace(
        qwen3_8_next_ngram_size=2,
        qwen3_8_next_heads_per_ngram=1,
        qwen3_8_next_ple_embed_dim=3,
        qwen3_8_next_split_ngram_parts=2,
        qwen3_8_next_hf_checkpoint=str(tmp_path),
        init_model_with_meta_device=via_flag,
    )
    allocations = []

    class AllocationGuard(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten.empty.memory_format:
                allocations.append(kwargs.copy())
                assert not kwargs.get("pin_memory", False), "Meta inspection must never pin host storage"
            return func(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Meta inspection read tensor payload or launched a PLE kernel")

    load_metadata = ple.Qwen38NextFrozenNGramEmbedding.load_metadata_from_hf
    monkeypatch.setattr(ple.Qwen38NextFrozenNGramEmbedding, "load_metadata_from_hf", forbidden)
    monkeypatch.setattr(ple, "gather_ple_rows", forbidden)
    with AllocationGuard(), nullcontext() if via_flag else torch.device("meta"):
        model = ple.Qwen38NextFrozenNGramEmbedding(config, layer_number=2)
    assert allocations and path.exists()
    assert model.table.is_meta and model.table.shape == (8, 3)
    assert all(buffer.is_meta for buffer in model.buffers())
    assert not model._loaded and not list(model.parameters()) and not model.state_dict()
    with pytest.raises(RuntimeError, match="inspection-only"):
        load_metadata(model, str(tmp_path))
    with pytest.raises(RuntimeError, match="inspection-only"):
        model.load_from_hf(str(tmp_path))
    with pytest.raises(RuntimeError, match="inspection-only"):
        model(torch.tensor([[0]]))
    # to_empty only touches registered buffers, not the deliberately unregistered
    # frozen table. It must not accidentally make this instance runnable.
    model.to_empty(device="cpu")
    assert model.table.is_meta
    with pytest.raises(RuntimeError, match="inspection-only"):
        model(torch.tensor([[0]]))


def make_checkpoint(tmp_path, *, dtype=torch.bfloat16):
    from safetensors.torch import save_file

    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    tensors = {
        f"{prefix}.shard_{shard}.weight": torch.arange(12, dtype=dtype).reshape(4, 3) + shard * 20 for shard in range(2)
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, path.name)})
    )
    state = SimpleNamespace(
        hf_layer_index=1,
        shard_ids=[0, 1],
        rows_per_shard=4,
        table=torch.full((8, 3), float("nan"), dtype=torch.bfloat16),
        _loaded=False,
    )
    return path, state, torch.cat(list(tensors.values()))


def test_table_reader_completes_short_reads(tmp_path, monkeypatch):
    from verl.models.mcore.qwen3_8_next.ops import ple

    path, state, expected = make_checkpoint(tmp_path)
    calls = []

    class ShortReader:
        def __init__(self, file):
            self.file = file

        def __getattr__(self, name):
            return getattr(self.file, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def readinto(self, destination):
            calls.append(len(destination))
            return self.file.readinto(destination[:3])

    def short_open(filename, *args, **kwargs):
        file = builtins.open(filename, *args, **kwargs)
        return ShortReader(file) if str(filename) == str(path) else file

    monkeypatch.setattr(ple, "open", short_open, raising=False)
    ple.Qwen38NextFrozenNGramEmbedding.load_from_hf(state, str(tmp_path))
    assert state._loaded
    assert len(calls) == 16  # two 24-byte shards, three bytes per read
    torch.testing.assert_close(state.table, expected, rtol=0, atol=0)


@pytest.mark.parametrize("was_loaded", [False, True])
@pytest.mark.parametrize("fault", ["truncated", "wrong_dtype", "wrong_byte_range", "negative_offset"])
def test_table_reader_rejects_invalid_checkpoint(tmp_path, fault, was_loaded):
    from verl.models.mcore.qwen3_8_next.ops import ple

    path, state, _ = make_checkpoint(tmp_path, dtype=torch.float16 if fault == "wrong_dtype" else torch.bfloat16)
    state._loaded = was_loaded
    if fault == "truncated":
        with path.open("r+b") as file:
            file.truncate(path.stat().st_size - 2)
    elif fault in ("wrong_byte_range", "negative_offset"):
        data = path.read_bytes()
        header_size = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + header_size])
        offsets = next(iter(header.values()))["data_offsets"]
        if fault == "wrong_byte_range":
            offsets[1] -= 2
        else:
            offsets[:] = [-1, 23]
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + header_size :])
    with pytest.raises((EOFError, ValueError), match="PLE"):
        ple.Qwen38NextFrozenNGramEmbedding.load_from_hf(state, str(tmp_path))
    assert not state._loaded, "An incomplete or reinterpreted table must never become usable"
