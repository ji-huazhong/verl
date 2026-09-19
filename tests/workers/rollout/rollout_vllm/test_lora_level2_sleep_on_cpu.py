# SPDX-License-Identifier: Apache-2.0
"""Ordering/fail-closed gates; actual CUDA reload parity has a separate test."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout import vllm_async_server


class Engine:
    def __init__(self):
        self.calls = []
        self.fail_reload = False

    async def sleep(self, level):
        self.calls.append(("sleep", level))

    async def wake_up(self, tags):
        self.calls.append(("wake", tags))

    async def collective_rpc(self, method):
        self.calls.append(("rpc", method))
        if self.fail_reload and method == "reload_lora_base_weights":
            raise RuntimeError("injected reload failure")

    async def reset_encoder_cache(self):
        self.calls.append(("encoder",))

    async def reset_prefix_cache(self, **kwargs):
        self.calls.append(("prefix",))


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda **kwargs: False)
    result = object.__new__(vllm_async_server.vLLMHttpServer)
    result.node_rank = 0
    result.rollout_mode = RolloutMode.HYBRID
    result.config = RolloutConfig(name="vllm", lora_sleep_level=2, load_format="safetensors", enforce_eager=True)
    result.model_config = SimpleNamespace(lora_rank=16, lora={"merge": False})
    result.engine = Engine()
    result._lora_base_reload_pending = False
    return result


def test_sleep_reload_adapter_kv_order_and_no_duplicate_reload(server):
    server._validate_lora_sleep_config()

    async def run():
        for _ in range(2):
            await server.sleep()
            assert server._lora_base_reload_pending
            await server.wake_up(tags=["weights"])
            assert not server._lora_base_reload_pending
            server.engine.calls.append(("adapter_sync",))
            await server.wake_up(tags=["kv_cache"])

    asyncio.run(run())
    assert (
        server.engine.calls
        == [
            ("rpc", "prepare_lora_level2_sleep"),
            ("sleep", 2),
            ("encoder",),
            ("wake", ["weights"]),
            ("rpc", "reload_lora_base_weights"),
            ("prefix",),
            ("adapter_sync",),
            ("wake", ["kv_cache"]),
            ("prefix",),
        ]
        * 2
    )


def test_reload_failure_and_wrong_wake_order_remain_blocked(server):
    async def run():
        await server.sleep()
        with pytest.raises(RuntimeError, match="weights reload"):
            await server.wake_up(tags=["kv_cache"])
        server.engine.fail_reload = True
        with pytest.raises(RuntimeError, match="injected reload failure"):
            await server.wake_up(tags=["weights"])
        assert server._lora_base_reload_pending
        with pytest.raises(RuntimeError, match="not been restored"):
            await server.resume_kv_cache()
        assert ("prefix",) not in server.engine.calls

    asyncio.run(run())


def test_release_kv_restores_base_before_sync(server):
    asyncio.run(server.release_kv_cache())
    assert server.engine.calls == [
        ("rpc", "prepare_lora_level2_sleep"),
        ("sleep", 2),
        ("encoder",),
        ("wake", ["weights"]),
        ("rpc", "reload_lora_base_weights"),
    ]
    assert not server._lora_base_reload_pending


def test_generate_rejected_until_base_reload_succeeds(server):
    async def run():
        await server.sleep()
        with pytest.raises(RuntimeError, match="Cannot generate"):
            await server.generate([1, 2], {}, "pending-base")

    asyncio.run(run())


@pytest.mark.parametrize("level,expected", [(1, 1), (2, 2)])
def test_lora_level_is_explicit_opt_in(server, level, expected):
    server.config = replace(server.config, lora_sleep_level=level)
    assert server._resolve_sleep_level() == expected


@pytest.mark.parametrize("invalid", [0, 3, True, "2", None])
def test_invalid_level_rejected(invalid):
    with pytest.raises(ValueError, match="must be 1 or 2"):
        RolloutConfig(name="vllm", lora_sleep_level=invalid)


@pytest.mark.parametrize("kwargs", [{"name": "sglang"}, {"free_cache_engine": False}, {"enable_sleep_mode": False}])
def test_disabled_sleep_rejected(kwargs):
    with pytest.raises(ValueError, match="requires vLLM"):
        RolloutConfig(**{"name": "vllm", "lora_sleep_level": 2, **kwargs})


@pytest.mark.parametrize("case", ["dp", "backend", "dummy", "quant", "mtp", "npu", "merged", "colocated", "graphs"])
def test_unsupported_paths_fail_before_engine_launch(server, monkeypatch, case):
    if case == "dp":
        server.config = replace(server.config, data_parallel_size=2)
    elif case == "backend":
        server.config = replace(
            server.config, checkpoint_engine=replace(server.config.checkpoint_engine, backend="nccl")
        )
    elif case == "dummy":
        server.config = replace(server.config, load_format="dummy")
    elif case == "quant":
        server.config = replace(server.config, quantization="fp8")
    elif case == "mtp":
        server.config = replace(server.config, mtp=replace(server.config.mtp, enable=True, enable_rollout=True))
    elif case == "npu":
        monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda **kwargs: True)
    elif case == "merged":
        server.model_config.lora["merge"] = True
    elif case == "graphs":
        server.config = replace(server.config, enforce_eager=False)
    else:
        server.rollout_mode = RolloutMode.COLOCATED
    with pytest.raises(ValueError):
        server._validate_lora_sleep_config()
    assert not server.engine.calls
