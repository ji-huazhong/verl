# SPDX-License-Identifier: Apache-2.0
"""Opt-in one-GPU level-2 sleep/reload gate using the exported random fixture.

Checks logical base parameter/buffer hashes (including frozen PLE), base and
adapter logprobs, and replacement with a changed adapter over repeated cycles.
Not a full-checkpoint, distributed or end-to-end GRPO acceptance test.
"""

import asyncio
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_LEVEL2_TESTS") != "1", reason="explicit one-GPU level-2 sleep opt-in required"
)


def test_level2_base_ple_and_replaced_adapter_parity():
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams

    from verl.utils.megatron_peft_utils import build_peft_config_for_vllm
    from verl.utils.vllm import TensorLoRARequest
    from verl.workers.config import RolloutConfig
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    reference = load_file(str(fixture / "reference.safetensors"))
    weights = load_file(str(fixture / "raw_adapter.safetensors"))
    rank, alpha = int(reference["lora_rank"]), int(reference["lora_alpha"])
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 10 * 1024**3:
        pytest.skip("Need 10 GiB free; never evict other jobs")
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / total)
    llm = LLM(
        model=str(fixture / "model"),
        skip_tokenizer_init=True,
        language_model_only=True,
        dtype="bfloat16",
        load_format="safetensors",
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.025,
        kv_cache_memory_bytes=128 * 1024**2,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=2,
        max_logprobs=256,
        enable_prefix_caching=False,
        enable_sleep_mode=True,
        enable_lora=True,
        max_lora_rank=max(8, rank),
        max_loras=1,
        max_cpu_loras=1,
        lora_dtype="bfloat16",
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        seed=123,
    )

    class Engine:
        async def sleep(self, level):
            if os.environ.get("QWEN38_LEVEL2_TRACE") == "1":
                print("LEVEL2_RUNNER_STATE_BEFORE", runner_state())
            llm.sleep(level=level)

        async def wake_up(self, tags):
            llm.wake_up(tags=tags)
            if "weights" in tags and os.environ.get("QWEN38_LEVEL2_TRACE") == "1":
                print("LEVEL2_RUNNER_STATE_AFTER", runner_state())

        async def collective_rpc(self, method):
            llm.collective_rpc(method)

        async def reset_encoder_cache(self):
            llm.collective_rpc(lambda worker: worker.model_runner.reset_encoder_cache())

        async def reset_prefix_cache(self, **kwargs):
            llm.reset_prefix_cache()

    def runner_state():
        def capture(worker):
            state = getattr(worker.model_runner, "model_state", None)
            return (
                {
                    name: value.detach().cpu().tolist()
                    for name, value in vars(state).items()
                    if isinstance(value, torch.Tensor) and value.numel() <= 8
                }
                if state is not None
                else {}
            )

        return llm.collective_rpc(capture)

    server = object.__new__(vLLMHttpServer)
    server.node_rank = 0
    server.rollout_mode = RolloutMode.HYBRID
    server.config = RolloutConfig(name="vllm", lora_sleep_level=2, load_format="safetensors", enforce_eager=True)
    server.model_config = SimpleNamespace(lora_rank=rank, lora={"merge": False})
    server.engine = Engine()
    server._lora_base_reload_pending = False
    server._validate_lora_sleep_config()
    ids = reference["input_ids"][0].tolist()
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=256, detokenize=False)

    def infer(adapter=None):
        output = llm.generate([{"prompt_token_ids": ids}], params, lora_request=adapter, use_tqdm=False)[0]
        result = torch.tensor([[p[token].logprob for token in range(256)] for p in output.prompt_logprobs[1:]])
        assert bool(result.isfinite().all())
        return result

    def hashes():
        def capture(model):
            result = {}
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                if "lora" in name.lower():
                    continue
                # Native HC pads its fused output to 16 rows. Those unused rows
                # are never loaded from the checkpoint, so raw storage hashes
                # are not a meaningful weight-correctness test for this tensor.
                marker = ".input_mix_weight_down_block_inject."
                if marker in name and name.endswith("weight"):
                    hc = model.get_submodule(name.split(marker)[0])
                    logical_rows = hc.lora_rank + hc.hc_count
                    assert tensor.shape[0] == logical_rows + hc.pad_size
                    tensor = tensor[:logical_rows]
                result[name] = hashlib.sha256(
                    tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest()
            return result

        return llm.apply_model(capture)[0]

    def request(tensors):
        return TensorLoRARequest(
            lora_name="level2_gate",
            lora_int_id=1,
            lora_path=str(fixture / "adapter"),
            peft_config=build_peft_config_for_vllm({"rank": rank, "alpha": alpha}),
            lora_tensors=tensors,
        )

    def install_traces(model):
        model._level2_traces = {}
        for name, module in model.named_modules():
            if not name:
                continue

            def capture_inputs(_module, args, kwargs, key=name):
                for index, value in enumerate(args):
                    if isinstance(value, torch.Tensor):
                        model._level2_traces[f"{key}:input:{index}"] = value.detach().cpu().clone()
                for label, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        model._level2_traces[f"{key}:input:{label}"] = value.detach().cpu().clone()

            def capture(_module, _args, output, key=name):
                value = output[0] if isinstance(output, tuple) and output else output
                if isinstance(value, torch.Tensor):
                    model._level2_traces[key] = value.detach().cpu().clone()

            module.register_forward_hook(capture)
            module.register_forward_pre_hook(capture_inputs, with_kwargs=True)

    try:
        if os.environ.get("QWEN38_LEVEL2_TRACE") == "1":
            llm.apply_model(install_traces)
        base = infer()
        base_traces = dict(llm.apply_model(lambda model: getattr(model, "_level2_traces", {}))[0])
        baseline_hashes = hashes()
        assert any("ple_embedding" in name for name in baseline_hashes)
        adapter = request(weights)
        tuned = infer(adapter)
        assert not torch.equal(tuned, base), "Adapter must have a measurable effect"
        changed = {name: tensor * 0.5 if "lora_B" in name else tensor.clone() for name, tensor in weights.items()}
        llm.llm_engine.remove_lora(1)
        updated = infer(request(changed))
        assert not torch.equal(updated, tuned), "Replacement adapter must change the output"
        torch.testing.assert_close(infer(), base, rtol=0, atol=0)
        for cycle, tensors, expected in ((1, weights, tuned), (2, changed, updated)):
            # Deliberately corrupt a base tensor: a no-op reload cannot pass.
            llm.apply_model(lambda model: model.language_model.lm_head.weight.data.zero_())
            asyncio.run(server.sleep())
            assert server._lora_base_reload_pending
            asyncio.run(server.wake_up(tags=["weights"]))
            assert hashes() == baseline_hashes, "Base/PLE/buffer hashes changed across level-2 reload"
            # Use the same native tensor adapter loader as production IPC completion.
            adapter = request(tensors)
            llm.collective_rpc("add_lora", args=(adapter,))
            asyncio.run(server.wake_up(tags=["kv_cache"]))
            restored = infer()
            if base_traces:
                traces = llm.apply_model(lambda model: model._level2_traces)[0]
                for name, expected_trace in base_traces.items():
                    actual_trace = traces[name]
                    if not torch.equal(actual_trace, expected_trace):
                        gap = (actual_trace.float() - expected_trace.float()).abs().max()
                        print(f"QWEN38_LEVEL2_TRACE first_difference={name} max_abs={gap.item()}")
                        break
            torch.testing.assert_close(restored, base, rtol=0, atol=0)
            torch.testing.assert_close(infer(adapter), expected, rtol=0, atol=0)
            print(f"QWEN38_LEVEL2 cycle={cycle} base_hashes={len(baseline_hashes)} base_gap=0 adapter_gap=0")
    finally:
        llm.llm_engine.engine_core.shutdown()
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
