# SPDX-License-Identifier: Apache-2.0
"""Opt-in native vLLM image/LoRA gate for a random, four-layer fixture.

Export first with test_qwen38_next_gpu.py, then run in a separate process:
RUN_QWEN38_VLLM_VISION_TESTS=1 QWEN38_TINY_EXPORT_DIR=<fixture> \
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m pytest -s -q <this file>

Uses real image processors and vision towers, but a tiny WordLevel tokenizer.
This is not full-checkpoint, real-tokenizer image GRPO, or video acceptance.
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_VLLM_VISION_TESTS") != "1", reason="explicit vLLM vision GPU opt-in required"
)


def test_native_image_adapter_reload_and_cached_decode():
    from PIL import Image
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams

    from verl.utils.megatron_peft_utils import build_peft_config_for_vllm
    from verl.utils.vllm import TensorLoRARequest, VLLMHijack

    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    reference = load_file(str(fixture / "reference.safetensors"))
    ids = reference["vision_input_ids"][0].tolist()
    assert ids[3:9] == [250, 252, 252, 252, 252, 251]
    # Feed one image placeholder, not four already-expanded patch tokens. The
    # native multimodal processor must expand it back to the reference IDs.
    raw_ids = ids[:5] + ids[8:]
    rgb = reference["vision_rgb"].numpy()
    rank, alpha = int(reference["lora_rank"]), int(reference["lora_alpha"])
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        pytest.skip("Need 8 GiB free headroom; never evict another job")
    torch.cuda.set_per_process_memory_fraction(min(0.035, 4 * 1024**3 / total))
    llm = LLM(
        model=str(fixture / "model"),
        skip_tokenizer_init=False,
        language_model_only=False,
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
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={"min_pixels": 4096, "max_pixels": 4096},
        mm_processor_cache_gb=0,
        enable_lora=True,
        max_lora_rank=max(8, rank),
        max_loras=1,
        max_cpu_loras=1,
        lora_dtype="bfloat16",
        seed=123,
    )
    try:
        VLLMHijack.hijack()
        params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=256, detokenize=False)

        def install_traces(model):
            model._test_vision_traces = {}

            def capture_vision(_module, args, output):
                model._test_vision_traces["pixels"] = args[0].detach().cpu().clone()
                value = output[0] if isinstance(output, tuple) else output
                model._test_vision_traces["embeddings"] = value.detach().cpu().clone()

            # The compiled language wrapper bypasses nn.Module pre-hooks.
            # Observe the outer VL forward without changing any arguments,
            # output, model math, or native multimodal processing.
            original_forward = model.forward

            def capture_forward(*args, **kwargs):
                positions = kwargs["positions"] if "positions" in kwargs else args[1]
                model._test_vision_traces["positions"] = positions.detach().cpu().clone()
                return original_forward(*args, **kwargs)

            model.visual.register_forward_hook(capture_vision)
            model.forward = capture_forward

        def base_hashes(model):
            import hashlib

            return {
                name: (
                    str(tensor.dtype),
                    tuple(tensor.shape),
                    hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                )
                for name, tensor in model.named_parameters()
                if "lora" not in name.lower()
            }

        def prompt(changed=False, continuation=()):
            return {
                "prompt_token_ids": raw_ids + list(continuation),
                "multi_modal_data": {"image": Image.fromarray(255 - rgb if changed else rgb)},
            }

        def logprobs(adapter=None, changed=False):
            output = llm.generate([prompt(changed)], params, lora_request=adapter, use_tqdm=False)[0]
            assert output.prompt_token_ids == ids, "Native processor expanded the wrong image tokens"
            return torch.tensor(
                [[position[token].logprob for token in range(256)] for position in output.prompt_logprobs[1:]]
            )

        def parity(label, actual):
            expected = reference[f"vision_{label}_logits"][0, :-1].float().log_softmax(-1)
            gap = (actual - expected).abs()
            print(f"QWEN38_VISION_{label.upper()}_GAP_MEAN={gap.mean().item():.8f}")
            print(f"QWEN38_VISION_{label.upper()}_GAP_MAX={gap.max().item():.8f}")
            assert bool(actual.isfinite().all()) and gap.mean() < 0.005 and gap.max() < 0.05

        llm.apply_model(install_traces)
        original_hashes = llm.apply_model(base_hashes)[0]
        assert any(name.startswith("visual.") for name in original_hashes), "Vision weights were not loaded"
        base = logprobs()
        traces = llm.apply_model(lambda model: model._test_vision_traces)[0]
        expected_pixels = reference["vision_pixel_values"].to(traces["pixels"].dtype)
        torch.testing.assert_close(traces["pixels"], expected_pixels, rtol=0, atol=0)
        expected_positions = reference["vision_position_ids"].squeeze(1)
        torch.testing.assert_close(traces["positions"][:, : len(ids)], expected_positions, rtol=0, atol=0)
        actual_embeddings, expected_embeddings = traces["embeddings"].float(), reference["vision_embeddings"].float()
        assert actual_embeddings.shape == expected_embeddings.shape == (4, 128)
        relative_l2 = (actual_embeddings - expected_embeddings).norm() / expected_embeddings.norm().clamp_min(1e-12)
        print(f"QWEN38_VISION_EMBEDDING_REL_L2={relative_l2.item():.8f}")
        assert actual_embeddings.isfinite().all() and relative_l2 < 0.01
        parity("base", base)
        changed_base = logprobs(changed=True)
        assert not torch.equal(changed_base, base), "Image was ignored or a stale embedding was reused"
        parity("changed_base", changed_base)
        adapter = TensorLoRARequest(
            lora_name="flash_next_tiny_image",
            lora_int_id=1,
            lora_path=str(fixture / "adapter"),
            peft_config=build_peft_config_for_vllm({"rank": rank, "alpha": alpha}),
            lora_tensors=load_file(str(fixture / "raw_adapter.safetensors")),
        )
        tuned = logprobs(adapter)
        parity("adapter", tuned)
        assert not torch.equal(base, tuned), "Adapter was ignored"
        changed_tuned = logprobs(adapter, changed=True)
        parity("changed_adapter", changed_tuned)
        assert not torch.equal(tuned, changed_tuned)
        torch.testing.assert_close(logprobs(), base, rtol=0, atol=0)
        assert llm.llm_engine.remove_lora(1)
        torch.testing.assert_close(logprobs(adapter), tuned, rtol=0, atol=0)
        assert llm.apply_model(base_hashes)[0] == original_hashes, "Adapter reload mutated frozen base/vision weights"
        print(f"QWEN38_VISION_BASE_HASH_UNCHANGED tensors={len(original_hashes)}")

        # Cache correctness on the same six generated tokens, not a claim of
        # cross-engine parity for new continuations or identical greedy ties.
        for label, request in (("base", None), ("adapter", adapter)):
            sampled = llm.generate(
                [prompt()],
                SamplingParams(temperature=0, max_tokens=6, logprobs=256, ignore_eos=True, detokenize=False),
                lora_request=request,
                use_tqdm=False,
            )[0].outputs[0]
            assert len(sampled.token_ids) == len(sampled.logprobs) == 6
            replay = llm.generate(
                [prompt(continuation=sampled.token_ids)], params, lora_request=request, use_tqdm=False
            )[0]
            assert replay.prompt_token_ids == ids + list(sampled.token_ids)
            cached = torch.tensor([[position[token].logprob for token in range(256)] for position in sampled.logprobs])
            teacher = torch.tensor(
                [[position[token].logprob for token in range(256)] for position in replay.prompt_logprobs[len(ids) :]]
            )
            gap = (cached - teacher).abs()
            print(f"QWEN38_VISION_{label.upper()}_DECODE_PREFILL_GAP_MEAN={gap.mean().item():.8f}")
            print(f"QWEN38_VISION_{label.upper()}_DECODE_PREFILL_GAP_MAX={gap.max().item():.8f}")
            assert bool(gap.isfinite().all()) and gap.mean() < 0.005 and gap.max() < 0.05
        assert llm.apply_model(base_hashes)[0] == original_hashes
    finally:
        llm.llm_engine.engine_core.shutdown()
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
