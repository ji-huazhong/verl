# SPDX-License-Identifier: Apache-2.0
"""Independent full48 TP8/EP8 vLLM teacher-forced/reload numerical gate.

Run after the full numerical Megatron job has released all GPUs. Requires
RUN_QWEN38_FULL_VLLM=1, QWEN38_MODEL_PATH, QWEN38_FULL_NUMERICAL_OUTPUT.
Private artifacts retain per-token and full-vocabulary differences. This test
does not replace the separate production HTTP/GRPO/save/resume run.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from tests.models.mcore.qwen38_full_validation import full_checkpoint_config, logprob_differences, tensor_sha256

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_FULL_VLLM") != "1", reason="explicit full48 eight-GPU vLLM opt-in required"
)


def frozen_vllm_hashes(model):
    """Logical frozen parameters/PLE hash buffers, not mutable KV/workspaces."""
    hashes = {
        name: tensor_sha256(parameter) for name, parameter in model.named_parameters() if "lora" not in name.lower()
    }
    for name, buffer in model.named_buffers():
        if "ple" in name and name.rsplit(".", 1)[-1] in (
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
        ):
            hashes[name] = tensor_sha256(buffer)
    return hashes


def test_full_checkpoint_text_image_adapter_disable_and_reload():
    from PIL import Image
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams

    from verl.utils.megatron_peft_utils import build_peft_config_for_vllm
    from verl.utils.vllm import TensorLoRARequest

    source = Path(os.environ["QWEN38_MODEL_PATH"])
    output = Path(os.environ["QWEN38_FULL_NUMERICAL_OUTPUT"])
    config = full_checkpoint_config(source)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["layers"] == 48 and manifest["vocabulary"] == 248320
    for field, filename in (("config_sha256", "config.json"), ("index_sha256", "model.safetensors.index.json")):
        assert manifest[field] == hashlib.sha256((source / filename).read_bytes()).hexdigest()
    assert not (output / "vllm-results.json").exists(), "Never overwrite prior results"
    assert torch.cuda.device_count() == 8
    for device in range(8):
        assert torch.cuda.mem_get_info(device)[0] >= 100 * 1024**3, "Wait for idle GPUs; never evict other jobs"
    reference = load_file(str(output / "reference.safetensors"))
    weights = load_file(str(output / "raw_adapter.safetensors"))
    assert len(weights) == 936
    vocab = config["text_config"]["vocab_size"]
    os.environ.setdefault("VERL_USE_EXTERNAL_MODULES", "verl.models.mcore.qwen3_8_next.bridge")
    llm = LLM(
        model=str(source),
        dtype="bfloat16",
        load_format="safetensors",
        enforce_eager=True,
        tensor_parallel_size=8,
        data_parallel_size=1,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
        fully_sharded_loras=False,
        distributed_executor_backend="mp",
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=512 * 1024**2,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        max_logprobs=vocab,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={"min_pixels": 4096, "max_pixels": 4096},
        mm_processor_cache_gb=0,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        max_cpu_loras=1,
        lora_dtype="bfloat16",
        seed=123,
    )
    report, errors = {}, []
    try:
        request = TensorLoRARequest(
            lora_name="full48_numerical_probe",
            lora_int_id=1,
            lora_path=str(output / "adapter"),
            peft_config=build_peft_config_for_vllm({"rank": 16, "alpha": 32}),
            lora_tensors=weights,
        )
        params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=vocab, detokenize=False)

        def evaluate(case, adapter=None):
            ids = reference[f"{case}.input_ids"].tolist()
            prompt = {"prompt_token_ids": reference[f"{case}.raw_input_ids"].tolist()}
            if f"{case}.rgb" in reference:
                prompt["multi_modal_data"] = {"image": Image.fromarray(reference[f"{case}.rgb"].numpy())}
            result = llm.generate([prompt], params, lora_request=adapter, use_tqdm=False)[0]
            assert result.prompt_token_ids == ids, "Native processor/tokenizer input mismatch"
            assert len(result.prompt_logprobs) == len(ids)
            # Full vocabulary is intentional here, not top-k plus an inferred tail.
            rows = result.prompt_logprobs[1:]
            assert all(len(row) == vocab for row in rows)
            return torch.tensor([[row[token].logprob for token in range(vocab)] for row in rows])

        def compare(case, state, actual):
            metrics = logprob_differences(actual, reference[f"{case}.{state}"], reference[f"{case}.input_ids"][1:])
            report[f"{case}.{state}"] = metrics
            (output / "vllm-results.json").write_text(json.dumps(report, indent=2))
            print(
                f"FULL48_VLLM case={case} state={state} mean={metrics['all_vocab_mean']:.9g} "
                f"max={metrics['all_vocab_max']:.9g} selected_max={metrics['selected_token_max']:.9g}",
                flush=True,
            )
            # Keep the established gate; do not increase tolerances after seeing results.
            if metrics["all_vocab_mean"] >= 0.005 or metrics["all_vocab_max"] >= 0.05:
                errors.append(f"cross-engine full-vocabulary parity: {case}.{state}")

        # Native image/text forwards before hashing allow legitimate lazy setup.
        base = {case: evaluate(case) for case in manifest["cases"]}
        original_hashes = llm.apply_model(frozen_vllm_hashes)
        assert len(original_hashes) == 8
        for rank_hash in original_hashes:
            assert any(name.startswith("visual.") for name in rank_hash), "Vision weights were not loaded"
            assert any(name.endswith("ngram_embedding.weight") for name in rank_hash), "PLE table not covered"
            assert any(name.endswith("ngram_heads_offsets") for name in rank_hash), "PLE hash metadata not covered"
        tuned = {}
        for case in manifest["cases"]:
            compare(case, "base", base[case])
            tuned[case] = evaluate(case, request)
            compare(case, "adapter", tuned[case])
            if torch.equal(base[case], tuned[case]):
                errors.append(f"adapter ignored: {case}")
            torch.testing.assert_close(evaluate(case), base[case], rtol=0, atol=0)
        assert not torch.equal(base["image"], base["changed_image"]), "Image was ignored"
        assert not torch.equal(tuned["image"], tuned["changed_image"]), "Adapted model ignored image"
        assert llm.llm_engine.remove_lora(1)
        for case in manifest["cases"]:
            torch.testing.assert_close(evaluate(case, request), tuned[case], rtol=0, atol=0)
        assert llm.apply_model(frozen_vllm_hashes) == original_hashes, "LoRA reload changed frozen weights"
        (output / "vllm-frozen-hashes.json").write_text(json.dumps(original_hashes, sort_keys=True))
        assert not errors, errors
        print("QWEN38_FULL48_VLLM_NUMERICAL_PASS TP8 EP8 DP1", flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()
