# SPDX-License-Identifier: Apache-2.0
"""Opt-in vLLM text/reload gate for the random fixture exported by the GPU test.

Run in a separate process, after Megatron has released its process groups:
RUN_QWEN38_VLLM_TESTS=1 QWEN38_TINY_EXPORT_DIR=<fixture> \
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m pytest -s -q <this file>

This does not establish full-checkpoint, multimodal, or GRPO correctness.
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_VLLM_TESTS") != "1", reason="explicit vLLM GPU opt-in required"
)


def test_vllm_base_adapter_disable_and_reload():
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams

    from verl.utils.megatron_peft_utils import build_peft_config_for_vllm
    from verl.utils.vllm import TensorLoRARequest, VLLMHijack

    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "Memory-bounded test must use the same process"
    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    reference = load_file(str(fixture / "reference.safetensors"))
    ids = reference["input_ids"][0].tolist()
    assert reference["base_logits"].shape == (1, len(ids), 256)
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        pytest.skip("Need 8 GiB free headroom; never evict another job")
    torch.cuda.set_per_process_memory_fraction(min(0.035, 4 * 1024**3 / total))
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
        enable_lora=True,
        max_lora_rank=8,
        max_loras=1,
        max_cpu_loras=1,
        lora_dtype="bfloat16",
        seed=123,
    )
    VLLMHijack.hijack()
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=256, detokenize=False)

    def install_traces(model):
        model._test_traces = {}

        def save(name, value):
            model._test_traces[name] = value.detach().reshape(-1, value.shape[-1]).cpu().contiguous().clone()

        def wrap_hc(hc, prefix):
            for method in ("mix", "combine_and_mix"):
                original = getattr(hc, method)

                def wrapped(*args, _original=original, **kwargs):
                    output = _original(*args, **kwargs)
                    save(f"{prefix}.input", output[1])
                    save(f"{prefix}.residual", output[0])
                    return output

                setattr(hc, method, wrapped)

        def hook(name):
            def capture(_module, _args, output):
                save(name, output[0] if isinstance(output, tuple) else output)

            return capture

        for i, layer in enumerate(model.language_model.model.layers):
            wrap_hc(layer.attn_hyper_connection, f"trace.{i}.self_attention")
            wrap_hc(layer.mlp_hyper_connection, f"trace.{i}.mlp")
            attn = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
            attn.register_forward_hook(hook(f"trace.{i}.self_attention.output"))
            layer.mlp.register_forward_hook(hook(f"trace.{i}.mlp.output"))
            if layer.ple is not None:
                layer.ple.register_forward_hook(hook(f"trace.{i}.ple"))
        wrap_hc(model.language_model.model.hyper_connection_mixer, "trace.final")

    if any(name.startswith("trace.") for name in reference):
        llm.apply_model(install_traces)

    def logprobs(adapter=None):
        output = llm.generate([{"prompt_token_ids": ids}], params, lora_request=adapter, use_tqdm=False)[0]
        return torch.tensor(
            [[position[token].logprob for token in range(256)] for position in output.prompt_logprobs[1:]]
        )

    def assert_parity(name, actual):
        expected = reference[f"{name}_logits"][0, :-1].float().log_softmax(-1)
        gap = (actual - expected).abs()
        print(f"QWEN38_TINY_{name.upper()}_LOGPROB_GAP_MEAN={gap.mean().item():.8f}")
        print(f"QWEN38_TINY_{name.upper()}_LOGPROB_GAP_MAX={gap.max().item():.8f}")
        assert bool(actual.isfinite().all())
        assert gap.mean() < 0.005 and gap.max() < 0.05, "Tiny cross-engine parity gate failed"

    base = logprobs()
    if any(name.startswith("trace.") for name in reference):
        trace = llm.apply_model(lambda model: model._test_traces)[0]
        trace["trace.contraction"] = trace.pop("trace.final.input")
        for name, expected in reference.items():
            if name.startswith("trace."):
                actual = trace[name][: len(ids)]
                assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
                gap = (actual.float() - expected.float()).abs()
                print(f"QWEN38_TRACE {name} mean={gap.mean():.8f} max={gap.max():.8f}")
    assert_parity("base", base)
    # Exercise the actual verl tensor loading contract, not a disk-only PEFT
    # adapter or a base checkpoint with adapter weights silently merged in.
    adapter = TensorLoRARequest(
        lora_name="flash_next_tiny",
        lora_int_id=1,
        lora_path=str(fixture / "adapter"),
        peft_config=build_peft_config_for_vllm({"rank": 4, "alpha": 8}),
        lora_tensors=load_file(str(fixture / "raw_adapter.safetensors")),
    )
    tuned = logprobs(adapter)
    assert not torch.equal(base, tuned), "Adapter was silently ignored"
    torch.testing.assert_close(logprobs(), base, rtol=0, atol=0)
    assert llm.llm_engine.remove_lora(1)
    torch.testing.assert_close(logprobs(adapter), tuned, rtol=0, atol=0)
    assert_parity("adapter", tuned)
