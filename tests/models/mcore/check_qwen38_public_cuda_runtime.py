# SPDX-License-Identifier: Apache-2.0
"""Small public-runtime ABI/collective check; no model or private dependency.

Uses only eight transient workers and small tensors. Passing this does not
validate Megatron, LoRA, the actual checkpoint, or training numerics.
"""

import datetime
import importlib.metadata
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, rendezvous):
    import vllm._custom_ops as ops

    torch.cuda.set_device(rank)
    assert torch.cuda.mem_get_info(rank)[0] >= 2 * 1024**3
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=8,
        timeout=datetime.timedelta(seconds=120),
        device_id=torch.device("cuda", rank),
    )
    try:
        tensor = torch.full((1024 * 1024,), rank + 1.0, device=f"cuda:{rank}")
        dist.all_reduce(tensor)
        assert torch.all(tensor == 36).item(), "NCCL reduction mismatch"
        x = torch.randn(16, 4096, dtype=torch.bfloat16, device=f"cuda:{rank}")
        weight, result = torch.ones(4096, dtype=x.dtype, device=x.device), torch.empty_like(x)
        ops.rms_norm(result, x, weight, 1e-6)
        reference = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        torch.testing.assert_close(result, reference, atol=0.015625, rtol=0.0078125)
        product = x[:8, :64].float() @ x[:8, :64].float().T
        assert torch.isfinite(product).all().item()
        torch.cuda.synchronize()
        print(f"PUBLIC_RUNTIME_GPU_GATE rank={rank} nccl=exact rms=passed matmul=finite", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    for name in ("torch", "vllm", "nvidia-nccl-cu13", "nvidia-cutlass-dsl"):
        print(name, importlib.metadata.version(name), flush=True)
    assert torch.__version__.split("+")[0] == "2.13.0"
    assert torch.version.cuda == "13.0"
    assert importlib.metadata.version("vllm") == "0.29.0"
    assert importlib.metadata.version("nvidia-nccl-cu13") == "2.29.7"
    assert torch.cuda.nccl.version() == (2, 29, 7)
    for name in (
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-core",
        "nvidia-cutlass-dsl-libs-cu13",
    ):
        assert importlib.metadata.version(name) == "4.6.2", f"Inconsistent public runtime: {name}"
    assert torch.cuda.device_count() == 8
    with tempfile.TemporaryDirectory(prefix="q38-public-runtime-") as tmp:
        mp.spawn(worker, args=((Path(tmp) / "nccl-rendezvous").as_uri(),), nprocs=8, join=True)
    print("PUBLIC_RUNTIME_GATE_PASSED: not a model/training acceptance", flush=True)
