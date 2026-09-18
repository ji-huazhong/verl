# SPDX-License-Identifier: Apache-2.0
"""Materialize a real-weight decoder-prefix fixture; NEVER an accuracy model.

Keeps the original width, vocabulary, experts, vision and complete frozen PLE.
Only decoder layers after the prefix and unused MTP tensors are removed. Copies
payload bytes without loading tensors into RAM; source files remain untouched.
The completion manifest contains no source paths or private environment data.
"""

import argparse
import hashlib
import json
import re
import shutil
import struct
from collections import defaultdict
from pathlib import Path

DECODER_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def keep_tensor(name, layers):
    if name.startswith("mtp."):
        return False
    match = DECODER_LAYER.match(name)
    return match is None or int(match[1]) < layers


def reduce_config(config, layers):
    config = json.loads(json.dumps(config))
    text = config["text_config"]
    original = text["num_hidden_layers"]
    if config["model_type"] != "qwen4_exp" or not 0 < layers < original:
        raise ValueError("Expected a strict decoder prefix of a qwen4_exp checkpoint")
    if layers % 4 or len(text["layer_types"]) != original:
        raise ValueError("Keep complete four-layer GDN/QSA cycles")
    if any(layer > layers for layer in text["ple_layer_ids"]):
        raise ValueError("This prefix would drop PLE; the test must retain all frozen tables")
    text["num_hidden_layers"] = layers
    text["layer_types"] = text["layer_types"][:layers]
    text["mtp_num_hidden_layers"] = 0
    if isinstance(text.get("mtp"), dict):
        text["mtp"]["num_hidden_layers"] = 0
        text["mtp"]["layer_types"] = []
    return config


def copy_shard(source, destination, header, payload_start, names):
    """Return payload byte count/hash after an independent read-back check."""
    output_header = {"__metadata__": header.get("__metadata__", {"format": "pt"})}
    offset = 0
    for name in names:
        info = header[name]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        output_header[name] = {**info, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(output_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    checksum = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        dst.write(struct.pack("<Q", len(encoded)))
        dst.write(encoded)
        for name in names:
            start, end = header[name]["data_offsets"]
            src.seek(payload_start + start)
            remaining = end - start
            while remaining:
                chunk = src.read(min(16 * 1024**2, remaining))
                if not chunk:
                    raise EOFError("Truncated source tensor payload")
                dst.write(chunk)
                checksum.update(chunk)
                remaining -= len(chunk)
    verified = hashlib.sha256()
    with destination.open("rb") as copied:
        copied.seek(8 + len(encoded))
        while chunk := copied.read(16 * 1024**2):
            verified.update(chunk)
    if checksum.digest() != verified.digest():
        raise ValueError("Copied checkpoint payload checksum mismatch")
    return offset, checksum.hexdigest()


def prepare(source, output, layers):
    if output.exists():
        raise FileExistsError("Use a fresh output directory; never overwrite checkpoints")
    raw_config = (source / "config.json").read_bytes()
    config = reduce_config(json.loads(raw_config), layers)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    selected = {name: shard for name, shard in index["weight_map"].items() if keep_tensor(name, layers)}
    present = {int(DECODER_LAYER.match(name)[1]) for name in selected if DECODER_LAYER.match(name)}
    if present != set(range(layers)):
        raise ValueError("The source checkpoint does not cover every retained decoder layer")
    shards = defaultdict(list)
    for name, shard in selected.items():
        if Path(shard).name != shard:
            raise ValueError("Expected flat local safetensors shard names")
        shards[shard].append(name)
    headers, payload_bytes, ple_bytes = {}, 0, 0
    for shard, names in shards.items():
        with (source / shard).open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if length > 16 * 1024**2:
                raise ValueError("Unexpectedly large checkpoint header")
            header = json.loads(f.read(length))
        headers[shard] = header, 8 + length
        for name in names:
            start, end = header[name]["data_offsets"]
            if not 0 <= start <= end <= (source / shard).stat().st_size - 8 - length:
                raise ValueError("Invalid source tensor bounds")
            payload_bytes += end - start
            if ".ple_embedding." in name:
                ple_bytes += end - start
    if shutil.disk_usage(output.parent).free < payload_bytes + 50 * 1024**3:
        raise ValueError("Need payload size plus 50 GiB free; no automatic cleanup")
    output.mkdir()
    checksums = {}
    for number, (shard, names) in enumerate(sorted(shards.items()), 1):
        header, payload_start = headers[shard]
        size, checksum = copy_shard(source / shard, output / shard, header, payload_start, sorted(names))
        checksums[shard] = {"payload_bytes": size, "payload_sha256": checksum}
        print(f"QWEN38_LAYER_SUBSET shard={number}/{len(shards)} payload_bytes={size} verified", flush=True)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
    ):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": payload_bytes}, "weight_map": selected}, indent=2) + "\n"
    )
    manifest = {
        "kind": "real_checkpoint_decoder_prefix_integration_only",
        "source_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "source_layers": json.loads(raw_config)["text_config"]["num_hidden_layers"],
        "layers": layers,
        "payload_bytes": payload_bytes,
        "ple_bytes": ple_bytes,
        "tensors": len(selected),
        "shards": checksums,
        "complete": True,
    }
    (output / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"QWEN38_LAYER_SUBSET_COMPLETE layers={layers} tensors={len(selected)} bytes={payload_bytes}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=12)
    args = parser.parse_args()
    prepare(args.source, args.output, args.layers)
