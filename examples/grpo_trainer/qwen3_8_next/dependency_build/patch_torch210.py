# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Apply the explicit Torch 2.10 build profile to the isolated main checkout.

This changes the requested ABI and package metadata. Compilation and CUDA
operator/model tests must still pass before the resulting wheel is published.
"""

import difflib
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
original = {p: p.read_text() for p in [root / "pyproject.toml", *root.glob("requirements/**/*.txt")]}
subprocess.run([sys.executable, str(root / "tools/use_existing_torch.py"), "--prefix"], cwd=root, check=True)
cuda = root / "requirements/cuda.txt"
source = re.sub(r"(?m)^torchcodec\s*[>=!~].*$", "", cuda.read_text())
source += (
    "\n# Nebula Torch 2.10 ABI profile\ntorch==2.10.0\ntorchvision==0.25.0\ntorchaudio==2.10.0\ntorchcodec==0.10.0\n"
)
cuda.write_text(source)
targets = [root / "CMakeLists.txt", *root.glob("cmake/**/*.cmake")]
for path in targets:
    source = path.read_text()
    if "0x020B000000000000ULL" in source:
        original[path] = source
        path.write_text(source.replace("0x020B000000000000ULL", "0x020A000000000000ULL"))
patch = "".join(
    "".join(
        difflib.unified_diff(
            before.splitlines(True),
            path.read_text().splitlines(True),
            fromfile="a/" + str(path.relative_to(root)),
            tofile="b/" + str(path.relative_to(root)),
        )
    )
    for path, before in original.items()
    if before != path.read_text()
)
(root.parent / "torch210-compatibility.patch").write_text(patch)
print("Recorded profile patch:", root.parent / "torch210-compatibility.patch")
