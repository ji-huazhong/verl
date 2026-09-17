# SPDX-License-Identifier: Apache-2.0
"""Opt-in regression for the isolated Megatron-Bridge VL allocation patch.

This CPU test executes the installed VL constructor with allocation probes.
It must fail on the unpatched affected constructor. Run against the isolated
patched dependency with RUN_QWEN38_BRIDGE_PATCH_TESTS=1, never patch shared
site-packages implicitly. GPU memory and final-weight parity need separate A/B.
"""

import os
import weakref
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_BRIDGE_PATCH_TESTS") != "1", reason="explicit isolated Bridge patch opt-in required"
)


def test_temporary_decoder_released_before_vl_allocation(monkeypatch):
    import torch
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import text_model

    old_decoder = []
    original_module_order = []
    replacement = torch.nn.Linear(4, 4)
    config = SimpleNamespace(kv_channels=4, rotary_interleaved=False, mrope_section=[1, 1, 0])
    groups = SimpleNamespace(cp=None)

    def initialize_base(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = kwargs["config"]
        self.pg_collection = kwargs["pg_collection"]
        self.pre_process = kwargs["pre_process"]
        self.post_process = kwargs["post_process"]
        self.embedding = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.Linear(4, 4)
        self.output_layer = torch.nn.Linear(4, 4)
        old_decoder.append(weakref.ref(self.decoder))
        original_module_order.extend(self._modules)

    def initialize_vl_block(**kwargs):
        assert len(old_decoder) == 1
        assert old_decoder[0]() is None, "Temporary GPT decoder is still live while the VL decoder is allocated"
        assert kwargs["config"] is config and kwargs["pg_collection"] is groups
        return replacement

    monkeypatch.setattr(text_model.GPTModel, "__init__", initialize_base)
    monkeypatch.setattr(text_model, "Qwen3VLMultimodalRotaryEmbedding", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(text_model, "Qwen3VLTransformerBlock", initialize_vl_block)
    model = text_model.Qwen3VLGPTModel(
        config=config,
        transformer_layer_spec=None,
        vocab_size=8,
        max_sequence_length=8,
        pg_collection=groups,
    )
    assert model.decoder is replacement
    assert old_decoder[0]() is None
    assert [name for name in model._modules if name in original_module_order] == original_module_order
