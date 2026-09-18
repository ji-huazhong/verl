# SPDX-License-Identifier: Apache-2.0
"""Payload and non-overwrite tests for the explicit reduced-layer fixture."""

import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.models.mcore.prepare_qwen38_next_layer_subset import keep_tensor, prepare, reduce_config


@pytest.mark.parametrize("existing", ["output", "ray", "missing_data"])
def test_subset_launcher_preserves_existing_evidence(tmp_path, existing):
    output, ray = tmp_path / "output", tmp_path / "ray"
    if existing != "missing_data":
        target = output if existing == "output" else ray
        target.mkdir()
        (target / "keep.txt").write_text("previous evidence")
    script = Path(__file__).resolve().parents[3] / "examples/tuning/lora/run_qwen38_flash_next_12layer_smoke.sh"
    result = subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "MODEL_PATH": str(tmp_path / "not_loaded"),
            "TRAIN_FILE": str(tmp_path / "missing.parquet"),
            "VAL_FILE": str(tmp_path / "missing.parquet"),
            "QWEN38_SUBSET_OUTPUT": str(output),
            "QWEN38_RAY_TEMP": str(ray),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == (4 if existing == "missing_data" else 3)
    if existing != "missing_data":
        assert (target / "keep.txt").read_text() == "previous evidence"


def config():
    return {
        "model_type": "qwen4_exp",
        "text_config": {
            "num_hidden_layers": 48,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "ple_layer_ids": [2],
            "hidden_size": 2560,
        },
    }


def test_prefix_changes_only_depth_and_preserves_ple():
    source = config()
    source["text_config"]["layer_types"] *= 12
    subset = reduce_config(source, 12)
    assert source["text_config"]["num_hidden_layers"] == 48
    assert subset["text_config"]["num_hidden_layers"] == 12
    assert subset["text_config"]["layer_types"].count("full_attention") == 3
    assert subset["text_config"]["ple_layer_ids"] == [2]
    assert subset["text_config"]["hidden_size"] == 2560
    for invalid in (0, 10, 48, 52):
        with pytest.raises(ValueError):
            reduce_config(source, invalid)
    source["text_config"]["ple_layer_ids"] = [20]
    with pytest.raises(ValueError, match="drop PLE"):
        reduce_config(source, 12)


def test_prefix_selection_keeps_vision_and_global_state():
    assert keep_tensor("model.visual.blocks.20.weight", 12)
    assert keep_tensor("model.language_model.embed_tokens.weight", 12)
    assert keep_tensor("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight", 12)
    assert keep_tensor("model.language_model.layers.11.mlp.weight", 12)
    assert not keep_tensor("model.language_model.layers.12.mlp.weight", 12)
    assert not keep_tensor("mtp.layers.0.weight", 12)


def test_materialized_payload_round_trip_and_source_unchanged(tmp_path, monkeypatch):
    from tests.models.mcore import prepare_qwen38_next_layer_subset as builder

    monkeypatch.setattr(builder.shutil, "disk_usage", lambda path: SimpleNamespace(free=2**50))
    source, output = tmp_path / "source", tmp_path / "subset"
    source.mkdir()
    source_config = config()
    source_config["text_config"]["layer_types"] *= 12
    (source / "config.json").write_text(json.dumps(source_config))
    names = [f"model.language_model.layers.{i}.mlp.weight" for i in range(48)]
    names += ["model.language_model.layers.1.ple.ple_embedding.table", "model.visual.blocks.20.weight", "mtp.weight"]
    header = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [i * 4, (i + 1) * 4]} for i, name in enumerate(names)
    }
    encoded = json.dumps(header).encode()
    payload = b"".join(struct.pack("<f", float(i)) for i in range(len(names)))
    original = struct.pack("<Q", len(encoded)) + encoded + payload
    (source / "model.safetensors").write_bytes(original)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "model.safetensors")})
    )
    prepare(source, output, 12)
    assert (source / "model.safetensors").read_bytes() == original
    assert json.loads((source / "config.json").read_text()) == source_config
    copied = (output / "model.safetensors").read_bytes()
    header_len = struct.unpack("<Q", copied[:8])[0]
    result_header = json.loads(copied[8 : 8 + header_len])
    result_payload = copied[8 + header_len :]
    expected_names = {name for name in names if keep_tensor(name, 12)}
    assert set(result_header) - {"__metadata__"} == expected_names
    for name in expected_names:
        start, end = result_header[name]["data_offsets"]
        before_start, before_end = header[name]["data_offsets"]
        assert result_payload[start:end] == payload[before_start:before_end]
    manifest = json.loads((output / "subset_manifest.json").read_text())
    assert manifest["complete"] and manifest["layers"] == 12 and manifest["source_layers"] == 48
    assert manifest["shards"]["model.safetensors"]["payload_sha256"] == hashlib.sha256(result_payload).hexdigest()
    with pytest.raises(FileExistsError):
        prepare(source, output, 12)
    assert (output / "model.safetensors").read_bytes() == copied
