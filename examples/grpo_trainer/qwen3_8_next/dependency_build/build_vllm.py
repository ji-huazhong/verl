# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned vLLM profile inside an existing isolated Linux environment."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

VERSION = "0.30.1.dev0+g6e517b15.torch210.cu131.hcgdnfp32.plestate"
VENDORS = {
    "VLLM_CUTLASS_SRC_DIR": "cutlass",
    "VLLM_FLASH_ATTN_SRC_DIR": "vllm-flash-attn",
    "FLASH_MLA_SRC_DIR": "flashmla",
    "FLASH_KDA_SRC_DIR": "flashkda",
    "DEEPGEMM_SRC_DIR": "deepgemm",
    "DEEPSELECT_SRC_DIR": "deepselect",
    "QUTLASS_SRC_DIR": "qutlass",
    "TRITON_KERNELS_SRC_DIR": "triton/python/triton_kernels/triton_kernels",
    "FMHA_SM100_SRC_DIR": "fmha-sm100",
    "TML_FA4_SRC_DIR": "tml-fa4",
}


def main():
    """Validate ABI/source prerequisites and build without changing the base image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Checkout patched by prepare_sources.py")
    parser.add_argument("--python", type=Path, required=True, help="Python in an isolated venv under build-root")
    parser.add_argument("--vendor", type=Path, required=True, help="Pinned external sources with manifest.json")
    args = parser.parse_args()
    root, source, vendor = (p.resolve() for p in (args.build_root, args.source, args.vendor))
    # Do not resolve the interpreter symlink: its venv path selects isolation.
    python = args.python.absolute()
    if not python.is_relative_to(root) or not (python.parent.parent / "pyvenv.cfg").exists():
        raise SystemExit("Python must belong to a dedicated venv inside --build-root")
    if not source.is_relative_to(root):
        raise SystemExit("Build from a dedicated source checkout inside --build-root")
    provenance = json.loads((source.parent / (source.name + "-qwen38-prepared.json")).read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != provenance["base"] or head != "6e517b15c1833cf72a7f557ee32524d98682e617":
        raise SystemExit("Prepared source revision changed")
    for name, expected in provenance["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Prepared source changed: {name}")
    manifest = json.loads((vendor / "manifest.json").read_text())
    if manifest["vllm_commit"] != "6e517b15c1833cf72a7f557ee32524d98682e617":
        raise SystemExit("Vendor manifest does not match the pinned vLLM source")
    subprocess.run(
        [
            str(python),
            "-c",
            "import sys, torch; assert sys.platform == 'linux'; "
            "assert sys.version_info[:2] == (3,13); assert torch.__version__.split('+')[0] == '2.10.0'; "
            "assert torch.version.cuda == '13.1'; print('BUILD_ABI',sys.version,torch.__version__,torch.version.cuda)",
        ],
        check=True,
    )
    env = os.environ.copy()
    for key, directory in VENDORS.items():
        path = vendor / directory
        if not path.is_dir():
            raise SystemExit(f"Missing vendor source: {path}")
        env[key] = str(path)
    env.update(
        VLLM_VERSION_OVERRIDE=VERSION,
        TORCH_CUDA_ARCH_LIST="9.0",
        VLLM_REQUIRE_RUST_FRONTEND="0",
        VLLM_USE_RUST_FRONTEND="0",
        MAX_JOBS=env.get("MAX_JOBS", "8"),
        NVCC_THREADS=env.get("NVCC_THREADS", "2"),
    )
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "build", "--python", str(python), "--wheel", "--no-build-isolation", "--out-dir", str(root / "dist")],
        cwd=source,
        env=env,
        check=True,
    )
    with (root / "build-environment.txt").open("w") as output:
        subprocess.run(["uv", "pip", "freeze", "--python", str(python)], stdout=output, check=True)


if __name__ == "__main__":
    main()
