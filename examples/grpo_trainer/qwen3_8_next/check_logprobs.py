# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Compare fixed token sequences across Megatron and vLLM, without sampling.

Input is a JSON list of {"id": str, "input_ids": list[int]} records. Run the
Megatron phase with torchrun, then vLLM in a separate process after releasing
the training model. Both phases use the same complete HF checkpoint.
"""

import argparse
import hashlib
import json
import os
from copy import copy
from pathlib import Path


def load_prompts(path, max_length):
    prompts = json.loads(path.read_text())
    ids = set()
    for prompt in prompts:
        name, tokens = prompt["id"], prompt["input_ids"]
        if name in ids or not isinstance(name, str):
            raise ValueError("Prompt ids must be unique strings")
        ids.add(name)
        if not 2 <= len(tokens) <= max_length or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError(f"Invalid tokens or length for prompt {name}")
    if not prompts:
        raise ValueError("At least one fixed token sequence is required")
    return prompts


def fixed_token_logprobs(logits, labels, chunk_size=128):
    """Bound the FP32 softmax workspace by token rows, preserving every label."""
    import torch

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    return torch.cat(
        [
            -torch.nn.functional.cross_entropy(
                flat_logits[start : start + chunk_size].float(),
                flat_labels[start : start + chunk_size],
                reduction="none",
            )
            for start in range(0, flat_labels.numel(), chunk_size)
        ]
    )


def save_result(args, prompts, logprobs, peak_memory, *, complete=True):
    from importlib.metadata import version

    payload = {
        "backend": args.backend,
        "complete": complete,
        "model": str(args.model.resolve()),
        "config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "torch": version("torch"),
        "backend_version": version(
            {"megatron": "megatron-core", "vllm": "vllm", "transformers": "transformers"}[args.backend]
        ),
        "dtype": args.reference_dtype if args.backend == "transformers" else "bfloat16",
        "megatron_frozen_ple_table": getattr(args, "freeze_ple", False) if args.backend == "megatron" else None,
        "reference_implementation": getattr(args, "reference_implementation", None),
        "repeat_differences": getattr(args, "repeat_differences", []),
        "transformers": version("transformers"),
        "parallelism": {
            "tp": args.tp,
            "pp": args.pp if args.backend == "megatron" else 1,
            "ep": args.ep if args.backend == "megatron" else 1,
        },
        "peak_allocated_bytes": peak_memory,
        "records": [dict(prompt, logprobs=values) for prompt, values in zip(prompts, logprobs, strict=True)],
    }
    destination = args.output if complete else args.output.with_suffix(".partial.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def megatron_logprobs(args, prompts, *, phase_hook=None, prompt_hook=None, trace_phases=None, loaded_model_probe=None):
    """Evaluate fixed inputs; an optional diagnostic hook runs before each phase."""
    import torch
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state
    from megatron.core.enums import ModelType
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.module import Float16Module

    import verl.models.mcore.qwen3_8_next.bridge  # noqa: F401

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", device_id=device)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(123)
    trace = None
    try:
        bridge = AutoBridge.from_hf_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.tensor_model_parallel_size = args.tp
        provider.pipeline_model_parallel_size = args.pp
        provider.expert_model_parallel_size = args.ep
        provider.expert_tensor_parallel_size = 1
        provider.sequence_parallel = args.tp > 1
        provider.variable_seq_lengths = True
        provider.pipeline_dtype = torch.bfloat16
        provider.params_dtype = torch.bfloat16
        provider.bf16 = True
        provider.batch_p2p_comm = False
        provider.overlap_p2p_comm = False
        provider.moe_router_load_balancing_type = "none"
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        if args.freeze_ple:
            provider.qwen3_8_next_train_ple = False
        provider.language_max_sequence_length = args.max_length
        provider.finalize()
        provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        model = provider.provide(pre_process=pp_rank == 0, post_process=pp_rank == args.pp - 1).cuda()
        model.model_type = ModelType.encoder_or_decoder
        bridge.load_hf_weights([model])
        mixed = Float16Module(provider, model).eval()
        if loaded_model_probe is not None:
            return loaded_model_probe(model, mixed, provider)
        control = args.megatron_trace_control_dir
        phases = ("baseline0", "baseline1", "traced0", "traced1", "baseline2") if control else ("single",)
        if control:
            from examples.grpo_trainer.qwen3_8_next.repetition import record_megatron_runtime

            if torch.distributed.get_rank() == 0:
                control.mkdir(parents=True, exist_ok=False)
            torch.distributed.barrier()
            record_megatron_runtime(control / "runtime", provider, mixed, gdn_layers=args.trace_gdn_layers)
        writer = (
            pp_rank == args.pp - 1
            and parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_data_parallel_rank() == 0
        )
        for phase in phases:
            if phase_hook is not None:
                phase_hook(phase, model)
            trace_phase = phase in trace_phases if trace_phases is not None else phase.startswith("traced")
            directory = control / "activations" / phase if control and trace_phase else None
            directory = directory if control else args.backend_trace_dir
            if directory:
                from examples.grpo_trainer.qwen3_8_next.backend_trace import BackendPrefixTrace

                trace = BackendPrefixTrace(
                    directory,
                    "megatron",
                    torch.distributed.get_rank(),
                    parallel_state.get_tensor_model_parallel_rank(),
                    args.tp,
                    enabled=parallel_state.get_data_parallel_rank() == 0,
                    tokens=args.trace_tokens,
                    token_start=getattr(args, "trace_start", 0),
                    trace_qsa=getattr(args, "trace_qsa", False),
                )
                trace.attach_megatron(model, gdn_layers=args.trace_gdn_layers)
            results = []
            for prompt in prompts:
                length = len(prompt["input_ids"])
                padded = (length + args.tp - 1) // args.tp * args.tp
                tokens = prompt["input_ids"] + [provider.qwen3_8_next_eos_token_id] * (padded - length)
                capture = trace is not None and getattr(args, "trace_start", 0) < length
                if capture:
                    trace.start(prompt, padded, hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest())
                ids = torch.tensor([tokens], device=device)
                cu = torch.tensor([0, padded], dtype=torch.int32, device=device)
                inputs = dict(
                    input_ids=ids,
                    position_ids=torch.arange(padded, device=device).reshape(1, 1, -1).expand(3, 1, -1),
                    attention_mask=None,
                    packed_seq_params=PackedSeqParams(
                        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=padded, max_seqlen_kv=padded
                    ),
                )
                if prompt_hook is not None:
                    prompt_hook(phase, prompt, model, ids)

                def forward_step(iterator, module, ids=ids, length=length):
                    logits = module(**next(iterator))

                    def collect(logits, non_loss_data=False):
                        assert non_loss_data
                        values = -vocab_parallel_cross_entropy(logits[:, :-1].float().contiguous(), ids[:, 1:])
                        return values[0, : length - 1].float().cpu().tolist()

                    return logits, collect

                with torch.no_grad():
                    output = get_forward_backward_func()(
                        forward_step_func=forward_step,
                        data_iterator=iter([inputs]),
                        model=[mixed],
                        num_microbatches=1,
                        seq_length=padded,
                        micro_batch_size=1,
                        forward_only=True,
                        collect_non_loss_data=True,
                    )
                if capture:
                    trace.finish()
                if pp_rank == args.pp - 1:
                    results.append(output[0])
            if trace:
                trace.close()
                trace = None
            peak = torch.tensor(torch.cuda.max_memory_allocated(), device=device, dtype=torch.int64)
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            if writer:
                if control:
                    phase_args = copy(args)
                    phase_args.output = control / (phase + ".json")
                    save_result(phase_args, prompts, results, peak.item())
                    print("QWEN38_MEGATRON_CONTROL_PHASE", phase, flush=True)
                if not control or phase == "baseline1":
                    save_result(args, prompts, results, peak.item())
            torch.distributed.barrier()
        if control and writer:
            from examples.grpo_trainer.qwen3_8_next.repetition import save_megatron_control

            save_megatron_control(control, phases)
        torch.distributed.barrier()
    finally:
        if trace:
            trace.close()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def vllm_logprobs(args, prompts):
    import torch
    from vllm import LLM, SamplingParams

    trace_options = {}
    if args.backend_trace_dir:
        trace_options["worker_extension_cls"] = (
            "examples.grpo_trainer.qwen3_8_next.backend_trace.BackendTraceWorkerExtension"
        )
    engine = LLM(
        model=str(args.model),
        skip_tokenizer_init=True,
        language_model_only=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        load_format="safetensors",
        enforce_eager=True,
        max_model_len=args.max_length,
        max_num_batched_tokens=args.max_length,
        max_num_seqs=1,
        enable_prefix_caching=False,
        engram_config={"cpu_offload": True, "dp_shared_memory": False},
        gpu_memory_utilization=args.gpu_memory_utilization,
        **trace_options,
    )
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1, detokenize=False)
    results = []
    if args.backend_trace_dir:
        engine.collective_rpc(
            "qwen38_trace",
            kwargs=dict(action="install", directory=str(args.backend_trace_dir), tokens=args.trace_tokens),
        )
    try:
        for prompt in prompts:
            tokens = prompt["input_ids"]
            if args.backend_trace_dir:
                engine.collective_rpc(
                    "qwen38_trace",
                    kwargs=dict(
                        action="start",
                        prompt=prompt,
                        config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                    ),
                )
            output = engine.generate([{"prompt_token_ids": tokens}], params, use_tqdm=False)[0]
            if args.backend_trace_dir:
                engine.collective_rpc("qwen38_trace", kwargs=dict(action="finish"))
            assert output.prompt_logprobs is not None
            results.append([output.prompt_logprobs[i][token].logprob for i, token in enumerate(tokens) if i])
    finally:
        if args.backend_trace_dir:
            engine.collective_rpc("qwen38_trace", kwargs=dict(action="close"))
    save_result(args, prompts, results, None)  # Allocations belong to vLLM worker processes.
    del engine
    torch.cuda.empty_cache()


def use_torch_reference_functions():
    """Bypass HF's installed-package fallback as well as optional hub kernels."""
    import inspect

    from transformers.models.qwen4_exp import modeling_qwen4_exp

    implementations = {}
    for name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        # use_kernels=False only disables hub replacements. HF's functional
        # decorators still prefer installed FLA/causal-conv1d over Torch.
        function = inspect.unwrap(getattr(modeling_qwen4_exp, name))
        if inspect.getsourcefile(function) != modeling_qwen4_exp.__file__:
            raise RuntimeError(f"Cannot establish a native Torch reference for {name}")
        setattr(modeling_qwen4_exp, name, function)
        implementations[name] = f"{function.__module__}.{function.__name__}"
    return implementations


def transformers_logprobs(args, prompts):
    """Independent eager reference; native HF keeps the oversized PLE table on CPU."""
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    import torch
    from transformers import Qwen4ExpForConditionalGeneration

    # MDL ranks are node metadata, not a Transformers tensor-parallel launch.
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(name, None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    implementations = use_torch_reference_functions()
    dtype = getattr(torch, args.reference_dtype)
    memory = {i: "72GiB" for i in range(torch.cuda.device_count())}
    memory["cpu"] = "1200GiB"
    model, loading = Qwen4ExpForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        device_map="auto",
        max_memory=memory,
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
        raise ValueError(f"Transformers checkpoint coverage failed: {loading}")
    model.eval()
    if args.reference_ablation != "native":
        from precision_probe import apply_precision_probe

        apply_precision_probe(model, args.reference_ablation)
    parameter_dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    args.reference_implementation = {
        "functions": implementations,
        "attention": "eager",
        "experts": "eager",
        "parameter_dtypes": parameter_dtypes,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "precision_ablation": args.reference_ablation,
    }
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("The independent reference requires TF32 to be disabled")
    if dtype == torch.float32 and parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"The FP32 reference has mixed parameter dtypes: {parameter_dtypes}")
    print("QWEN38_REFERENCE_IMPLEMENTATION", json.dumps(args.reference_implementation), flush=True)
    print("QWEN38_TRANSFORMERS_LOADED", args.reference_dtype, json.dumps(model.hf_device_map), flush=True)
    input_device = model.get_input_embeddings().weight.device
    results = []
    trace = None
    args.repeat_differences = []
    if args.activation_trace_dir:
        from activation_trace import PrefixActivationTrace

        trace = PrefixActivationTrace(
            model, args.activation_trace_dir, tokens=args.trace_tokens, precision_variant=args.reference_ablation
        )
    try:
        for prompt in prompts:
            ids = torch.tensor([prompt["input_ids"]], device=input_device)
            first = None
            for repeat in range(args.repeat_count):
                torch.manual_seed(123)
                torch.cuda.manual_seed_all(123)
                if trace:
                    trace.start(prompt, repeat)
                with torch.inference_mode():
                    logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
                    if trace:
                        trace.finish(hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest())
                    labels = ids[:, 1:].to(logits.device)
                    values = fixed_token_logprobs(logits, labels)
                    if not torch.isfinite(values).all():
                        raise ValueError(f"Non-finite Transformers logprobs for {prompt['id']}")
                    current = values.cpu()
                    if first is None:
                        first = current
                        results.append(first.tolist())
                    else:
                        delta = (first - current).abs()
                        args.repeat_differences.append(
                            dict(
                                id=prompt["id"], repeat=repeat, mean_abs=delta.mean().item(), max_abs=delta.max().item()
                            )
                        )
                print(
                    "QWEN38_TRANSFORMERS_PROMPT", args.reference_dtype, prompt["id"], len(current), repeat, flush=True
                )
                del logits, values
            peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
            save_result(args, prompts[: len(results)], results, peak, complete=False)
    finally:
        if trace:
            trace.close()
    peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
    save_result(args, prompts, results, peak)


def compare(args):
    import numpy as np

    expected, actual = (json.loads(path.read_text()) for path in (args.reference, args.candidate))
    if not expected.get("complete", True) or not actual.get("complete", True):
        raise ValueError("Cannot accept an incomplete reference run")
    for field in ("config_sha256", "prompt_sha256"):
        if expected[field] != actual[field]:
            raise ValueError(f"Cannot compare different {field}")
    differences = []
    for left, right in zip(expected["records"], actual["records"], strict=True):
        assert left["id"] == right["id"] and left["input_ids"] == right["input_ids"]
        assert len(left["logprobs"]) == len(right["logprobs"]) == len(left["input_ids"]) - 1
        differences.extend(np.abs(np.array(left["logprobs"]) - np.array(right["logprobs"])))
    gaps = np.asarray(differences)
    assert np.isfinite(gaps).all(), "Non-finite logprobs"
    report = dict(tokens=len(gaps), mean=float(gaps.mean()), p99=float(np.quantile(gaps, 0.99)), max=float(gaps.max()))
    report["passed"] = report["mean"] < args.mean_tolerance and report["max"] < args.max_tolerance
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("Cross-backend logprob gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("megatron", "vllm", "transformers", "compare"), required=True)
    parser.add_argument("--reference-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--reference-ablation", choices=("native", "hc_fp32", "router_fp32", "hc_router_fp32"), default="native"
    )
    parser.add_argument("--activation-trace-dir", type=Path, help="private native-HF prefix activation directory")
    parser.add_argument("--backend-trace-dir", type=Path, help="private Megatron/vLLM prefix activation directory")
    parser.add_argument(
        "--megatron-trace-control-dir",
        type=Path,
        help="five passes on one loaded Megatron model; retain baseline and traced repetitions",
    )
    parser.add_argument("--trace-tokens", type=int, default=32)
    parser.add_argument("--trace-start", type=int, default=0)
    parser.add_argument("--trace-qsa", action="store_true", help="record actual QSA scores and selected token indices")
    parser.add_argument("--trace-gdn-layers", type=int, nargs="+", default=[])
    parser.add_argument("--repeat-count", type=int, default=1, help="fixed-RNG repetitions of each native-HF prompt")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--freeze-ple", action="store_true", help="use Megatron's frozen host-resident PLE table")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--mean-tolerance", type=float, default=0.01)
    parser.add_argument("--max-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if args.trace_gdn_layers and (
        args.backend != "megatron" or not args.megatron_trace_control_dir or any(x < 0 for x in args.trace_gdn_layers)
    ):
        parser.error("--trace-gdn-layers requires Megatron repetition control and nonnegative layer indexes")
    if args.trace_tokens < 1 or args.repeat_count < 1 or args.trace_start < 0:
        parser.error("trace-tokens and repeat-count must be positive; trace-start must be nonnegative")
    if args.freeze_ple and args.backend != "megatron":
        parser.error("--freeze-ple only applies to the Megatron backend")
    if args.megatron_trace_control_dir and (args.backend != "megatron" or args.backend_trace_dir):
        parser.error("Megatron repetition control cannot combine with --backend-trace-dir or another backend")
    if args.backend_trace_dir and args.backend not in ("megatron", "vllm"):
        parser.error("backend tracing requires Megatron or vLLM")
    if (
        args.activation_trace_dir or args.repeat_count != 1 or args.reference_ablation != "native"
    ) and args.backend != "transformers":
        parser.error("activation tracing/repetition currently requires the Transformers backend")
    if args.backend == "compare":
        if args.reference is None or args.candidate is None:
            parser.error("compare requires --reference and --candidate")
        compare(args)
        return
    if args.model is None or args.prompts is None:
        parser.error("model execution requires --model and --prompts")
    config = json.loads((args.model / "config.json").read_text())
    context_limit = config.get("text_config", config).get("max_position_embeddings")
    if context_limit is not None and args.max_length > context_limit:
        parser.error(
            f"--max-length={args.max_length} exceeds the model's declared context {context_limit}; "
            "the numerical gate must not extrapolate beyond its RoPE cache"
        )
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ["VLLM_USE_RUST_FRONTEND"] = "0"
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    prompts = load_prompts(args.prompts, args.max_length - 1)
    {"megatron": megatron_logprobs, "vllm": vllm_logprobs, "transformers": transformers_logprobs}[args.backend](
        args, prompts
    )


if __name__ == "__main__":
    main()
