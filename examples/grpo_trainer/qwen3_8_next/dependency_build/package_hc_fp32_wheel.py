# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Package the HC Python/Triton change with the existing, verified CUDA binaries."""

import argparse
import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

BASE_VERSION = "0.30.1.dev0+g6e517b15.torch210.cu131"
BASE_SHA256 = "81f9134a5f37707ae1bce29750d448b2d2423c74db6a0a565d1a110d876980aa"
HC_GDN_SHA256 = "a435362f3c0a42d7054a058986446721c34d4ad50b0e9b284fd79f00919a3e91"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-wheel", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("hcfp32", "hcgdnfp32", "hcgdnfp32.plestate"), default="hcfp32")
    args = parser.parse_args()
    version = BASE_VERSION + "." + args.variant
    base_version, base_sha256 = BASE_VERSION, BASE_SHA256
    if args.variant == "hcgdnfp32.plestate":
        base_version, base_sha256 = BASE_VERSION + ".hcgdnfp32", HC_GDN_SHA256
    assert digest(args.base_wheel) == base_sha256
    manifest = json.loads((args.sources / "manifest.json").read_text())
    assert manifest["base_commit"] == "6e517b15c1833cf72a7f557ee32524d98682e617"
    name = f"vllm-{version}-cp313-cp313-linux_x86_64.whl"
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / name
    if target.exists():
        raise FileExistsError(target)
    old_info, new_info = f"vllm-{base_version}.dist-info/", f"vllm-{version}.dist-info/"
    records, applied = [], set()
    with (
        zipfile.ZipFile(args.base_wheel) as source,
        zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output,
    ):

        def add(path, data):
            output.writestr(path, data)
            h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append((path, "sha256=" + h, str(len(data))))

        assert old_info + "METADATA" in source.namelist()
        for item in source.infolist():
            if item.filename.endswith("/RECORD"):
                continue
            data = source.read(item)
            filename = item.filename.replace(old_info, new_info, 1)
            if item.filename in manifest["files"]:
                entry = manifest["files"][item.filename]
                assert hashlib.sha256(data).hexdigest() == entry["original_sha256"]
                data = (args.sources / item.filename).read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"]
                applied.add(item.filename)
            if filename == new_info + "METADATA":
                text = data.decode()
                assert f"\nVersion: {base_version}\n" in text
                data = text.replace(f"\nVersion: {base_version}\n", f"\nVersion: {version}\n").encode()
            if filename == "vllm/_version.py":
                data = (
                    f"__version__ = version = {version!r}\n"
                    f"__version_tuple__ = version_tuple = (0, 30, 1, 'dev0', {version.split('+')[1]!r})\n"
                    "__commit_id__ = commit_id = 'g6e517b15'\n"
                ).encode()
            add(filename, data)
        for filename, entry in manifest["files"].items():
            if filename in applied:
                continue
            assert entry["original_sha256"] is None
            data = (args.sources / filename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            add(filename, data)
        provenance = {
            "hcfp32": "hc_fp32",
            "hcgdnfp32": "hc_gdn_fp32",
            "hcgdnfp32.plestate": "hc_gdn_fp32_ple_state",
        }[args.variant]
        add(f"vllm/qwen38_{provenance}_provenance.json", json.dumps(manifest, indent=2).encode())
        record = new_info + "RECORD"
        content = io.StringIO()
        csv.writer(content, lineterminator="\n").writerows([*records, (record, "", "")])
        output.writestr(record, content.getvalue())
    evidence = dict(
        version=version,
        wheel=str(target),
        sha256=digest(target),
        bytes=target.stat().st_size,
        base_wheel_sha256=base_sha256,
        source_manifest=manifest,
        cuda_binaries="Unchanged; Python/Triton-only precision variant",
        full_model_validated=False,
    )
    target.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("HC_FP32_WHEEL", json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
