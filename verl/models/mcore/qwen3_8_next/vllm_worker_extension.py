# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Qwen precision candidates for colocated GRPO experiments."""

from functools import wraps

import torch

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


def apply_router_fp32_output():
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpSparseMoeBlock

    original = Qwen4ExpSparseMoeBlock.__init__
    if getattr(original, "_verl_router_fp32_output", False):
        return

    @wraps(original)
    def initialize(module, *args, **kwargs):
        original(module, *args, **kwargs)
        gate = module.gate
        if type(gate).__name__ != "GateLinear" or gate.weight.dtype != torch.bfloat16:
            raise RuntimeError("Qwen router candidate requires the pinned BF16 GateLinear implementation")
        # Preserve BF16 input/weight storage and the native dispatch policy:
        # eligible small batches use ll_bf16, otherwise cuBLAS FP32 output.
        gate.set_out_dtype(torch.float32)
        if not gate.allow_cublas_router_gemm:
            raise RuntimeError("Qwen router candidate lacks its cuBLAS FP32-output fallback")

    initialize._verl_router_fp32_output = True
    Qwen4ExpSparseMoeBlock.__init__ = initialize


class Qwen38RouterFP32WorkerExtension(vLLMColocateWorkerExtension):
    def __new__(cls, **kwargs):
        apply_router_fp32_output()
        return super().__new__(cls, **kwargs)

    def qwen38_router_precision_status(self):
        gates = [m.gate for m in self.get_model().modules() if type(m).__name__ == "Qwen4ExpSparseMoeBlock"]
        if not gates or any(
            gate.weight.dtype != torch.bfloat16 or gate.out_dtype != torch.float32 or not gate.allow_cublas_router_gemm
            for gate in gates
        ):
            raise RuntimeError("Qwen router FP32-output candidate is not active on every local MoE layer")
        return {"mode": "output_fp32", "gates": len(gates), "weight_dtype": "torch.bfloat16"}


def apply_qsa_canonical_order():
    """Preserve native block membership and visit it in ascending block order."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    from verl.models.mcore.qwen3_8_next.qsa_order import QsaBlockOrderProbe

    if not isinstance(qsa_indexer._topk, QsaBlockOrderProbe):
        qsa_indexer._topk = QsaBlockOrderProbe(qsa_indexer._topk)


class Qwen38QsaCanonicalOrderWorkerExtension(vLLMColocateWorkerExtension):
    """Install ordering before model construction; preserve colocated refit RPCs."""

    def __new__(cls, **kwargs):
        apply_qsa_canonical_order()
        return super().__new__(cls, **kwargs)

    def monkey_patch_model(self, vocab_size, banned_token_ids=None):
        from .production_trace import install_vllm_production_trace
        from .runtime_audit import audit_worker_runtime, audit_worker_weights

        super().monkey_patch_model(vocab_size, banned_token_ids)
        audit_worker_weights(self, "hf-loaded")
        audit_worker_runtime(self)
        install_vllm_production_trace(self)

    def update_weights_from_ipc(self, peft_config=None, base_sync_done=False, use_shm=False):
        from .runtime_audit import audit_worker_weights

        super().update_weights_from_ipc(peft_config=peft_config, base_sync_done=base_sync_done, use_shm=use_shm)
        count = getattr(self, "_qwen38_audited_refits", 0)
        if count == 0:
            audit_worker_weights(self, "first-refit")
        self._qwen38_audited_refits = count + 1

    def qwen38_qsa_precision_status(self):
        from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

        from verl.models.mcore.qwen3_8_next.qsa_order import QsaBlockOrderProbe

        layers = sum(type(module).__name__ == "QSAIndexer" for module in self.get_model().modules())
        probe = qsa_indexer._topk
        if not layers or not isinstance(probe, QsaBlockOrderProbe):
            raise RuntimeError("Canonical QSA ordering is not active on the Qwen rollout worker")
        return dict(policy="block-id", qsa_layers=layers, canonical_calls=probe.calls, membership_changed=False)
