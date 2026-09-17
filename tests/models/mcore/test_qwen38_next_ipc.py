# SPDX-License-Identifier: Apache-2.0
"""Opt-in Ray/CUDA IPC hot-reload gate for a Megatron-exported random fixture.

First export the GPU fixture with QWEN38_TINY_TP_CAPACITY=8 and
QWEN38_TINY_LORA_RANK=16, so the genuine BF16 adapter spans multiple 1 MiB
buckets. Then run this test with exactly one visible CUDA device:
RUN_QWEN38_IPC_TESTS=1 QWEN38_TINY_EXPORT_DIR=<fixture> \
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m pytest -s -q <this file>

The producer uses the production ServerAdapter; a real Ray actor delegates RPCs
to a real vLLM worker with the production colocate extension. This does not run
the HTTP server, live Megatron optimizer, full checkpoint, or GRPO trainer.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_IPC_TESTS") != "1", reason="explicit Ray/IPC GPU opt-in required"
)


def test_base_and_multibucket_adapter_hot_reload(monkeypatch):
    import ray
    from safetensors.torch import load_file

    from verl.utils.megatron_peft_utils import build_peft_config_for_vllm
    from verl.workers.config import RolloutConfig
    from verl.workers.config.rollout import CheckpointEngineConfig
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
    assert len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 1
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Select one GPU explicitly; never attach to a shared Ray cluster"
    assert not ray.is_initialized()
    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    reference = load_file(str(fixture / "reference.safetensors"))
    adapter_weights = load_file(str(fixture / "raw_adapter.safetensors"))
    adapter_bytes = sum(t.nbytes for t in adapter_weights.values())
    assert adapter_bytes > 1 << 20, "Export a rank-16 capacity-8 fixture to actually test multiple buckets"
    assert all(t.dtype == torch.bfloat16 and t.nbytes <= 1 << 20 for t in adapter_weights.values())
    ids = reference["input_ids"][0].tolist()
    rank, alpha = int(reference["lora_rank"]), int(reference["lora_alpha"])
    peft_config = build_peft_config_for_vllm({"rank": rank, "alpha": alpha})
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 10 * 1024**3:
        pytest.skip("Need 10 GiB free headroom; never evict another job")
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / total)
    device_uuid = str(torch.cuda.get_device_properties(0).uuid)

    @ray.remote(num_cpus=2, num_gpus=1, max_restarts=0)
    class Receiver:
        def __init__(self, fixture_path, lora_rank):
            from vllm import LLM

            os.environ["VERL_RAY_JOB_ID"] = ray.get_runtime_context().get_job_id()
            os.environ["VERL_REPLICA_RANK"] = "0"
            torch.cuda.set_device(0)
            free, total = torch.cuda.mem_get_info()
            assert free >= 8 * 1024**3, "Headroom changed; do not start the engine"
            torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / total)
            self.llm = LLM(
                model=str(Path(fixture_path) / "model"),
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
                enable_lora=True,
                max_lora_rank=lora_rank,
                max_loras=1,
                max_cpu_loras=1,
                lora_dtype="bfloat16",
                worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
                seed=123,
            )
            self.steps = []
            self.cache_resets = 0

        def identity(self):
            return str(torch.cuda.get_device_properties(0).uuid)

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            return self.llm.collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs)

        def clear_kv_cache(self):
            result = self.llm.reset_prefix_cache()
            assert result is not False
            self.cache_resets += 1

        def set_global_steps(self, step):
            self.steps.append(step)

        def status(self):
            return self.steps, self.cache_resets

        def base_hashes(self):
            def capture(model):
                import hashlib

                return {
                    name: (
                        str(t.dtype),
                        tuple(t.shape),
                        hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                    )
                    for name, t in model.named_parameters()
                    if "lora" not in name.lower()
                }

            return self.llm.apply_model(capture)[0]

        def perturb_base(self):
            def zero_head(model):
                with torch.no_grad():
                    model.language_model.lm_head.weight.zero_()

            self.llm.apply_model(zero_head)

        def logprobs(self, input_ids, use_adapter=False):
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest

            from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH

            request = LoRARequest(VLLM_LORA_NAME, VLLM_LORA_INT_ID, VLLM_LORA_PATH) if use_adapter else None
            output = self.llm.generate(
                [{"prompt_token_ids": input_ids}],
                SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=256, detokenize=False),
                lora_request=request,
                use_tqdm=False,
            )[0]
            return torch.tensor(
                [[position[token].logprob for token in range(256)] for position in output.prompt_logprobs[1:]]
            )

        def close(self):
            self.llm.llm_engine.engine_core.shutdown()
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "1")
    # A short, unique path avoids Ray's Unix socket path-length limit. Only this
    # test's local cluster is started/stopped; never call a global `ray stop`.
    ray_temp = tempfile.mkdtemp(prefix="q38i-")
    receiver = None
    ready = False
    try:
        ray.init(address="local", num_cpus=4, num_gpus=1, include_dashboard=False, _temp_dir=ray_temp)
        receiver = Receiver.remote(str(fixture), rank)
        assert ray.get(receiver.identity.remote(), timeout=180) == device_uuid
        ready = True
        rollout = ServerAdapter(
            RolloutConfig(
                name="vllm",
                tensor_model_parallel_size=1,
                checkpoint_engine=CheckpointEngineConfig(update_weights_bucket_megabytes=1),
            ),
            model_config=None,
            device_mesh=None,
        )
        rollout.server_handle = receiver
        assert not rollout.use_shm, "This gate must exercise real CUDA IPC, not shared-memory fallback"

        def infer(use_adapter=False):
            return ray.get(receiver.logprobs.remote(ids, use_adapter), timeout=90)

        def parity(label, actual):
            expected = reference[f"{label}_logits"][0, :-1].float().log_softmax(-1)
            gap = (actual - expected).abs()
            print(f"QWEN38_IPC_{label.upper()}_GAP_MEAN={gap.mean().item():.8f}")
            print(f"QWEN38_IPC_{label.upper()}_GAP_MAX={gap.max().item():.8f}")
            assert bool(actual.isfinite().all()) and gap.mean() < 0.005 and gap.max() < 0.05

        def sync(weights, step, base_sync_done):
            async def run():
                await rollout.update_weights(
                    ((name, tensor.cuda()) for name, tensor in weights.items()),
                    global_steps=step,
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                )

            asyncio.run(run())

        base = infer()
        parity("base", base)
        base_hashes = ray.get(receiver.base_hashes.remote(), timeout=30)
        assert base_hashes
        ray.get(receiver.perturb_base.remote(), timeout=30)
        assert not torch.equal(infer(), base), "The negative control did not affect the model"
        # Frozen PLE tables/hash metadata stay resident and are not streamed by
        # the actor exporter. Vision is explicitly outside this text-only gate.
        base_weights = {
            name: tensor
            for name, tensor in load_file(str(fixture / "model" / "model.safetensors")).items()
            if not name.startswith("model.visual.") and ".ple_embedding." not in name
        }
        sync(base_weights, 0, False)
        torch.testing.assert_close(infer(), base, rtol=0, atol=0)
        assert ray.get(receiver.base_hashes.remote(), timeout=30) == base_hashes
        sync(adapter_weights, 1, True)
        tuned = infer(True)
        assert not torch.equal(tuned, base), "LoRA was ignored"
        parity("adapter", tuned)
        torch.testing.assert_close(infer(), base, rtol=0, atol=0)
        # A genuinely different update catches stale adapter caching. Restore
        # the trained adapter afterwards to test repeated bucket-buffer reuse.
        zero_b = {name: torch.zeros_like(t) if "lora_B" in name else t for name, t in adapter_weights.items()}
        assert any("lora_B" in name and bool(t.count_nonzero()) for name, t in adapter_weights.items())
        sync(zero_b, 2, True)
        torch.testing.assert_close(infer(True), base, rtol=0, atol=0)
        sync(adapter_weights, 3, True)
        torch.testing.assert_close(infer(True), tuned, rtol=0, atol=0)
        torch.testing.assert_close(infer(), base, rtol=0, atol=0)
        assert ray.get(receiver.status.remote(), timeout=30) == ([0, 1, 2, 3], 4)
        assert ray.get(receiver.base_hashes.remote(), timeout=30) == base_hashes
        print(
            f"QWEN38_IPC_RELOAD_PASSED adapter_tensors={len(adapter_weights)} adapter_bytes={adapter_bytes} updates=4"
        )
    finally:
        try:
            if ready:
                ray.get(receiver.close.remote(), timeout=30)
        finally:
            if receiver is not None:
                ray.kill(receiver)
            ray.shutdown()
