# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Bounded snapshots of runner-owned tensors omitted by model.named_buffers()."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RunnerTensorSnapshot:
    """Preserve direct ModelState tensor attributes without traversing the model.

    vLLM V2 constructs ModelState inside the weights allocation pool, but native
    level-2 sleep only saves registered model buffers. Small runner constants
    can therefore be discarded together with weights. Do not walk nn.Modules,
    KV caches, or arbitrary object graphs and accidentally back up the model.
    This compatibility boundary is not a guarantee for every third-party runner.
    """

    state: Any
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]]
    nbytes: int

    @classmethod
    def capture(cls, state: Any, max_bytes: int = 64 << 20):
        tensors = (
            {}
            if state is None
            else {name: tensor for name, tensor in vars(state).items() if isinstance(tensor, torch.Tensor)}
        )
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if nbytes > max_bytes:
            raise RuntimeError(f"Runner state needs {nbytes} bytes, exceeding the {max_bytes}-byte sleep backup limit")
        if any(isinstance(tensor, torch.nn.Parameter) for tensor in tensors.values()):
            raise RuntimeError("Runner sleep snapshot must not contain model parameters")
        return cls(
            state,
            {name: (tensor, tensor.detach().to(device="cpu", copy=True)) for name, tensor in tensors.items()},
            nbytes,
        )

    @torch.no_grad()
    def restore(self, state: Any):
        if state is not self.state:
            raise RuntimeError("Runner ModelState was replaced during level-2 reload")
        # Validate everything before copying any data back into live state.
        for name, (target, saved) in self.tensors.items():
            if getattr(state, name, None) is not target or target.shape != saved.shape or target.dtype != saved.dtype:
                raise RuntimeError(f"Runner tensor changed during level-2 reload: {name}")
        for target, saved in self.tensors.values():
            target.copy_(saved, non_blocking=False)
