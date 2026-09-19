# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Training-loop integration with Megatron-Bridge PEFT gradient ownership."""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def bridge_peft_grad_finalizer(model: list[Any], default_finalize: Callable) -> Callable:
    """Delegate EP-replicated adapter gradients to Bridge's step-level finalizer.

    New Bridge adapters can accumulate fused weight gradients into ``main_grad``.
    Their per-microbatch autograd fallback cannot synchronize that buffer. Match
    Bridge's native trainer by removing those fallback hooks once, then reducing
    after the ordinary DP gradient finalization. This is independent of model type.
    """
    from megatron.bridge.peft import utils as peft_utils

    enable = getattr(peft_utils, "enable_expert_parallel_grad_sync_in_finalize", None)
    finalize = getattr(peft_utils, "finalize_model_grads_with_expert_adapter_sync", None)
    if enable is None and finalize is None:
        # Older Bridge releases do not expose this step-level integration.
        return default_finalize
    if not callable(enable) or not callable(finalize):
        raise RuntimeError("Megatron-Bridge exposes an incomplete expert-adapter gradient finalization API")
    count = enable(model)
    logger.info("Megatron-Bridge owns step-level EP gradient sync for %s adapter parameters", count)
    return finalize
