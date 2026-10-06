# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Check or apply the pinned Qwen3.8 dependency patches in a clean checkout."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASES = {
    "vllm": "6e517b15c1833cf72a7f557ee32524d98682e617",
    "bridge": "c0e164ed2aedac4ad1c877780e2564a19d5d54ec",
}
SERIES = {
    "vllm": ["vllm-hc-fp32.patch", "vllm-gdn-conv-fp32.patch", "vllm-model-state-sleep.patch"],
    "bridge": ["bridge-python313.patch"],
}


def run(*args, cwd):
    """Run a source operation and fail on the first error."""
    subprocess.run(args, cwd=cwd, check=True)


def apply(project, source):
    """Apply compatibility scripts before the ordered model patches."""
    if project == "vllm":
        for name in ("patch_torch210.py", "patch_cuda_view_torch210.py"):
            run(sys.executable, str(ROOT / "dependency_build" / name), str(source), cwd=source)
    for name in SERIES[project]:
        patch = str(ROOT / "patches" / name)
        run("git", "apply", "--check", patch, cwd=source)
        run("git", "apply", patch, cwd=source)


def main():
    """Validate in a disposable tree before optionally changing the checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=BASES)
    parser.add_argument("source", type=Path)
    parser.add_argument("--apply", action="store_true", help="Modify the specified clean checkout after validation")
    args = parser.parse_args()
    source = args.source.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != BASES[args.project]:
        raise SystemExit(f"Expected {BASES[args.project]}, found {head}; rebase/review patches for a newer revision")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source):
        raise SystemExit("Use a clean, dedicated upstream checkout")
    content = subprocess.check_output(["git", "archive", "HEAD"], cwd=source)
    with tempfile.TemporaryDirectory(prefix="qwen38-patch-check-") as directory:
        tree = Path(directory) / "source"
        tree.mkdir()
        with tarfile.open(fileobj=io.BytesIO(content)) as archive:
            archive.extractall(tree, filter="data")
        # git apply must not discover a repository above this temporary tree.
        run("git", "init", "--quiet", str(tree), cwd=tree)
        apply(args.project, tree)
    if args.apply:
        apply(args.project, source)
        changed = (
            subprocess.check_output(
                ["git", "ls-files", "--modified", "--others", "--exclude-standard", "-z"], cwd=source
            )
            .decode()
            .split("\0")
        )
        provenance = {
            "base": head,
            "files": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in changed
                if name and (source / name).is_file()
            },
        }
        (source.parent / (source.name + "-qwen38-prepared.json")).write_text(json.dumps(provenance, indent=2) + "\n")
    records = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*(ROOT / "patches" / name for name in SERIES[args.project])]
    }
    if args.project == "vllm":
        for name in ("patch_torch210.py", "patch_cuda_view_torch210.py"):
            path = ROOT / "dependency_build" / name
            records[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"project": args.project, "base": head, "applied": args.apply, "sha256": records}, indent=2))


if __name__ == "__main__":
    main()
