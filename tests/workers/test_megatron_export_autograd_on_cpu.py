# SPDX-License-Identifier: Apache-2.0
"""Export must not attach Bridge collectives to subsequent training graphs."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine.megatron import transformer_impl as impl


def make_engine(monkeypatch, path, *, fail=False):
    engine = object.__new__(impl.MegatronEngine)
    weight = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    events = []

    def stream():
        try:
            events.append(("iteration", torch.is_grad_enabled()))
            yield "leaf", weight
            events.append(("iteration", torch.is_grad_enabled()))
            yield "view", weight.T
            if fail:
                raise RuntimeError("export failed")
            yield "computed", weight * 2
        finally:
            events.append(("cleanup", torch.is_grad_enabled()))

    def export(*args, **kwargs):
        events.append(("factory", torch.is_grad_enabled()))
        return stream()

    engine.module = [object()]
    engine.peft_cls = object() if path == "adapter" else None
    engine.model_config = SimpleNamespace(lora={"merge": False})
    engine.vanilla_bridge = path == "vanilla"
    engine._qat_enabled = False
    engine._hf_export_tasks = []
    engine.bridge = SimpleNamespace(
        export_weights=export,
        export_adapter_weights=export,
        export_hf_weights=export,
    )
    monkeypatch.setattr(impl, "load_megatron_model_to_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(impl, "build_peft_config_for_vllm", lambda config: "peft")
    return engine, weight, events


@pytest.mark.parametrize("path", ["adapter", "base", "vanilla"])
def test_export_is_lazy_detached_and_restores_training_grad_mode(monkeypatch, path):
    engine, weight, events = make_engine(monkeypatch, path)
    with torch.enable_grad():
        iterator, config = engine.get_per_tensor_param(base_sync_done=True)
        assert events == [("factory", False)]
        assert torch.is_grad_enabled()
        exported = []
        for name, tensor in iterator:
            assert torch.is_grad_enabled(), "export context leaked across yield"
            assert not tensor.requires_grad and tensor.grad_fn is None
            exported.append((name, tensor))
        assert all(not enabled for _, enabled in events)
        assert config == ("peft" if path == "adapter" else None)
        assert [name for name, _ in exported] == ["leaf", "view", "computed"]
        assert exported[0][1].data_ptr() == weight.data_ptr()
        assert exported[1][1].stride() == weight.T.stride()
        for (_, actual), expected in zip(exported, [weight, weight.T, weight * 2], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert weight.requires_grad and weight.is_leaf
        weight.square().sum().backward()
        torch.testing.assert_close(weight.grad, 2 * weight, rtol=0, atol=0)


def test_export_failure_restores_grad_mode(monkeypatch):
    engine, _, events = make_engine(monkeypatch, "adapter", fail=True)
    with torch.enable_grad():
        iterator, _ = engine.get_per_tensor_param(base_sync_done=True)
        with pytest.raises(RuntimeError, match="export failed"):
            list(iterator)
        assert torch.is_grad_enabled()
        assert ("cleanup", False) in events


def test_async_consumer_does_not_disable_unrelated_task_gradients(monkeypatch):
    engine, _, events = make_engine(monkeypatch, "adapter")

    async def run():
        iterator, _ = engine.get_per_tensor_param(base_sync_done=True)
        interleaved = []

        async def consumer():
            for _, weight in iterator:
                assert not weight.requires_grad
                await asyncio.sleep(0)
                assert torch.is_grad_enabled()

        async def other_task():
            for _ in range(3):
                await asyncio.sleep(0)
                parameter = torch.nn.Parameter(torch.ones(1))
                (parameter * 3).sum().backward()
                interleaved.append(parameter.grad.item())

        await asyncio.gather(consumer(), other_task())
        assert interleaved == [3.0] * 3

    with torch.enable_grad():
        asyncio.run(run())
    assert all(not enabled for _, enabled in events)


def test_detached_quantization_payload_preserves_identity_and_metadata():
    packed = torch.arange(6, dtype=torch.int32)
    packed.export_metadata = {"group_size": 128}
    scale = torch.ones(2, dtype=torch.bfloat16)
    with torch.no_grad():
        exported = list(impl._iter_detached_export_weights([("packed", packed), ("scale", scale)]))
        assert not torch.is_grad_enabled()
    assert exported[0][1] is packed
    assert exported[1][1] is scale
    assert exported[0][1].export_metadata == {"group_size": 128}


@pytest.mark.parametrize("path", ["adapter", "base", "vanilla"])
def test_hf_checkpoint_export_does_not_record_autograd(path, tmp_path):
    from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager

    manager = object.__new__(MegatronCheckpointManager)
    manager.vanilla_bridge = path == "vanilla"
    manager.peft_cls = object() if path == "adapter" else None
    manager.model = [object()]
    manager.rank = 0
    manager.checkpoint_config = SimpleNamespace(strict=True, mbridge_config={})
    parameter = torch.nn.Parameter(torch.arange(3.0))
    observed = []

    def save(models, weights_path, *args, **kwargs):
        observed.append(torch.is_grad_enabled())
        # Stand in for export work executed synchronously by the actual Bridge.
        assert not (parameter * 2).requires_grad

    manager.bridge = SimpleNamespace(save_weights=save, save_hf_adapter=save, save_hf_weights=save)
    with torch.enable_grad():
        manager._save_model_as_hf_via_bridge(str(tmp_path))
        assert torch.is_grad_enabled()
        parameter.square().sum().backward()
    assert observed == [False]
    torch.testing.assert_close(parameter.grad, 2 * parameter, rtol=0, atol=0)
