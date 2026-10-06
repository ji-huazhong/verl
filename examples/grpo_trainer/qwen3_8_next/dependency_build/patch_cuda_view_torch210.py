# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Keep UVA storage ownership on Torch 2.10, which lacks stable from_blob deleters.

This one translation unit uses the exact Torch 2.10 C++ ABI. The resulting
wheel is deliberately specific to Torch 2.10, not portable across Torch ABIs.
"""

import difflib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
path = root / "csrc/libtorch_stable/cuda_view.cu"
before = path.read_text()
marker = "// QWEN38_TORCH210_UVA_OWNERSHIP"
if marker in before:
    raise SystemExit("Torch 2.10 UVA ownership patch already applied")
if before.count("torch::stable::from_blob(") != 2:
    raise SystemExit("Unexpected vLLM cuda_view source; review before patching")
prefix = """// QWEN38_TORCH210_UVA_OWNERSHIP
// This translation unit is compiled against the exact Torch 2.10 C++ ABI.
// Its stable from_blob overload cannot retain a deleter, so use ATen only
// at this ownership boundary. Never discard the original keepalive lambda.
#undef TORCH_TARGET_VERSION
#include <ATen/ops/from_blob.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
"""
helper = """
namespace {
torch::stable::Tensor torch210_from_blob_with_deleter(
    void* data, torch::headeronly::IntHeaderOnlyArrayRef sizes,
    torch::headeronly::IntHeaderOnlyArrayRef strides,
    torch::stable::Device device, torch::headeronly::ScalarType dtype,
    std::function<void(void*)> deleter) {
  auto result = at::from_blob(
      data, at::IntArrayRef(sizes.data(), sizes.size()),
      at::IntArrayRef(strides.data(), strides.size()), std::move(deleter),
      at::TensorOptions()
          .device(c10::Device(c10::DeviceType::CUDA, device.index()))
          .dtype(static_cast<at::ScalarType>(dtype)));
  return torch::stable::Tensor(
      torch::aot_inductor::new_tensor_handle(std::move(result)));
}
}  // namespace

"""
after = prefix + before.replace("// This function assumes", helper + "// This function assumes", 1)
after = after.replace("torch::stable::from_blob(", "torch210_from_blob_with_deleter(")
path.write_text(after)
patch = "".join(
    difflib.unified_diff(
        before.splitlines(True),
        after.splitlines(True),
        fromfile="a/csrc/libtorch_stable/cuda_view.cu",
        tofile="b/csrc/libtorch_stable/cuda_view.cu",
    )
)
(root.parent / "torch210-uva-ownership.patch").write_text(patch)
print("Applied Torch 2.10 UVA ownership compatibility")
