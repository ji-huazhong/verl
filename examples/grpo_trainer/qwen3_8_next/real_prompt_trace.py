# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Real GSM8K generation, controlled prefill traces and Megatron route replay.

This is an eight-GPU forward diagnosis, not the TP8/PP4 GRPO acceptance run.
All artifacts are written to a fresh private diagnostic directory.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

PHASES = ("baseline0", "baseline1", "traced0", "traced1", "baseline2")


def response_logprobs(prompt, values):
    """Slice next-token logprobs at the first response prediction, not its input."""
    prompt_length = prompt["prompt_length"]
    response_length = prompt["response_length"]
    if prompt_length < 1 or response_length < 1 or len(values) != prompt_length + response_length - 1:
        raise ValueError("Response logprob coordinates do not match the complete token sequence")
    return values[prompt_length - 1 :]


def make_trace_views(samples):
    views = []
    for sample in samples:
        for label, start in (("prefix", 0), ("response", sample["prompt_length"] - 1)):
            views.append(dict(sample, id=sample["id"] + "/" + label, sample_id=sample["id"], trace_token_start=start))
        # Capture the sparse-budget transition only when a real response reaches it.
        if sample["prompt_length"] <= 2052 < len(sample["input_ids"]):
            views.append(
                dict(sample, id=sample["id"] + "/qsa-boundary", sample_id=sample["id"], trace_token_start=2032)
            )
    return views


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def assess_trace_repeats(output, vllm_reference=None):
    """Require activation repeats, not only an unchanged final sampled logprob."""
    from examples.grpo_trainer.qwen3_8_next.backend_trace import compare_backend_traces

    views = json.loads((output / "fixed-prompts.json").read_text())
    rows = []
    for view in views:
        start = view["trace_token_start"]
        # The last input token predicts an unsampled extra token. Exclude that query.
        count = min(32, len(view["input_ids"]) - 1 - start)
        row = dict(id=view["id"], token_start=start, compared_queries=count)
        for name, directory in [
            ("megatron", output / "megatron-control/activations"),
            ("vllm", (vllm_reference or output) / "vllm-control"),
        ]:
            report = compare_backend_traces(directory / "traced0", directory / "traced1", view["id"], max_tokens=count)
            row[name] = dict(
                first_nonzero_stage=report["first_nonzero_stage"],
                max_abs=max(stage["max_abs"] for stage in report["stages"].values()),
            )
        row["activation_repeatable"] = all(row[name]["first_nonzero_stage"] is None for name in ("megatron", "vllm"))
        rows.append(row)
    return dict(all_windows_repeatable=all(row["activation_repeatable"] for row in rows), windows=rows)


def reuse_real_samples(source, output, count):
    """Reuse saved stochastic responses and routes without sampling new tokens."""
    import numpy as np

    samples = json.loads((source / "samples.json").read_text())
    selected = json.loads((source / "selected-prompts.json").read_text())
    if len(samples) != count or [row["id"] for row in selected] != [row["id"] for row in samples]:
        raise ValueError("Saved real-prompt sample identities or count differ")
    if len({row["id"] for row in samples}) != count:
        raise ValueError("Duplicate saved real-prompt samples")
    records = []
    for sample in samples:
        ids, prompt_ids = sample["input_ids"], sample["prompt_ids"]
        if (
            sample["prompt_length"] != len(prompt_ids)
            or ids[: len(prompt_ids)] != prompt_ids
            or sample["response_length"] < 1
            or len(ids) != len(prompt_ids) + sample["response_length"]
            or len(sample["generation_logprobs"]) != sample["response_length"]
            or not np.isfinite(sample["generation_logprobs"]).all()
        ):
            raise ValueError("Saved response coordinates differ from its prompt")
        name = sample["generation_routes"]
        if Path(name).name != name:
            raise ValueError("Saved route file must be a basename")
        path = source / "generation" / name
        routes = np.load(path, allow_pickle=False)
        if routes.shape != (len(ids) - 1, 48, 10) or not np.issubdtype(routes.dtype, np.integer):
            raise ValueError("Saved generation route coordinates differ from the real tokens")
        records.append(dict(file="generation/" + name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (output / "generation").mkdir()
    for name in ["selected-prompts.json", "samples.json", *(row["file"] for row in records)]:
        shutil.copyfile(source / name, output / name)
    write_json(
        output / "sample-reuse.json",
        dict(
            source=str(source),
            samples=count,
            samples_sha256=hashlib.sha256((source / "samples.json").read_bytes()).hexdigest(),
            routes=records,
            regenerated=False,
        ),
    )
    return selected, samples


def reuse_completed_vllm_reference(source, output, count):
    """Reuse an immutable, validated vLLM control without reloading the model."""
    status = json.loads((source / "status.json").read_text())
    report = json.loads((source / "report.json").read_text())
    if not status["complete"] or not report["complete"] or not report["controls_valid"]:
        raise ValueError("The saved vLLM reference must have valid completed forward controls")
    selected, samples = reuse_real_samples(source, output, count)
    views = make_trace_views(samples)
    write_json(output / "fixed-prompts.json", views)
    if (output / "fixed-prompts.json").read_bytes() != (source / "fixed-prompts.json").read_bytes():
        raise ValueError("Reused vLLM trace windows differ from the real samples")
    write_json(
        output / "vllm-reference.json",
        dict(source=str(source), report_sha256=hashlib.sha256((source / "report.json").read_bytes()).hexdigest()),
    )
    return selected, samples


def reuse_production_capture(source, output, count):
    """Use a completed first-step production capture without resampling."""
    import numpy as np

    marker = json.loads((source / "capture-complete.json").read_text())
    manifest = json.loads((source / "capture.json").read_text())
    if (
        not marker["complete"]
        or not manifest["complete"]
        or manifest["step"] != 1
        or marker["captured_samples"] != count
        or manifest["captured_samples"] != count
        or not manifest["metadata"]["before_policy_update"]
        or manifest["metadata"]["temperature"] != 1.0
    ):
        raise ValueError("Replay requires a completed first-step pre-update production capture at temperature 1")
    for name, digest in manifest["files"].items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or hashlib.sha256((source / path).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Production capture file hash or path is invalid")
    raw_samples = json.loads((source / "samples.json").read_text())
    required = {"samples.json", "selected-prompts.json"}
    required.update("generation/" + sample["generation_routes"] for sample in raw_samples)
    if not required <= manifest["files"].keys():
        raise ValueError("Production capture manifest omits replay inputs")
    for sample in raw_samples:
        mask = np.asarray(sample["response_mask"], dtype=bool)
        actor = np.asarray(sample["production_actor_logprobs"])
        if mask.shape != (sample["response_length"],) or not mask.any() or actor.shape != mask.shape:
            raise ValueError("Production score/mask coordinates differ")
        if not np.isfinite(actor[mask]).all():
            raise ValueError("Production actor scores are nonfinite")
    selected, samples = reuse_real_samples(source, output, count)
    write_json(output / "fixed-prompts.json", make_trace_views(samples))
    write_json(output / "production-capture.json", manifest)
    write_json(
        output / "production-capture-source.json",
        dict(
            source=str(source),
            manifest_sha256=hashlib.sha256((source / "capture.json").read_bytes()).hexdigest(),
            source_model_path=manifest["metadata"]["model_path"],
            checkpoint_scope=(
                "First pre-update production step; replay loads the pinned HF base. "
                "Weight-refit equality is under test."
            ),
        ),
    )
    return selected, samples


def vllm_worker(output, model, dataset, count, seed, replay_samples=None):
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    config_bytes = (model / "config.json").read_bytes()
    config = json.loads(config_bytes)
    assert config["text_config"]["num_hidden_layers"] == 48
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    builder = create_continuous_token_builder(tokenizer, hf_model_type=config["model_type"])
    table = pq.read_table(dataset, columns=["prompt", "extra_info", "reward_model"]) if replay_samples is None else None
    indices = list(range(len(table))) if table is not None else []
    random.Random(seed).shuffle(indices)
    selected = []
    for index in indices:
        row = table.slice(index, 1).to_pylist()[0]
        ids = builder.build_initial_tokens(row["prompt"])
        if 1 <= len(ids) <= 1024:
            selected.append(
                dict(
                    id=f"gsm8k-train-{index}",
                    dataset_row=index,
                    messages=row["prompt"],
                    ground_truth=row["reward_model"]["ground_truth"],
                    prompt_ids=ids,
                )
            )
        if len(selected) == count:
            break
    if replay_samples is not None:
        selected, samples = reuse_real_samples(replay_samples, output, count)
    else:
        assert len(selected) == count
        samples = []
        write_json(output / "selected-prompts.json", selected)
    engine = LLM(
        model=str(model),
        skip_tokenizer_init=True,
        language_model_only=True,
        dtype="bfloat16",
        tensor_parallel_size=8,
        load_format="safetensors",
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        enable_prefix_caching=True,
        enable_return_routed_experts=True,
        engram_config={"cpu_offload": True, "dp_shared_memory": False},
        gpu_memory_utilization=0.7,
        worker_extension_cls="examples.grpo_trainer.qwen3_8_next.backend_trace.BackendTraceWorkerExtension",
    )
    order = engine.collective_rpc("qwen38_qsa_order", kwargs={"policy": "block-id"})
    assert len(order) == 8 and all(item["policy"] == "block-id" for item in order)
    generation = output / "generation"
    generation.mkdir(exist_ok=replay_samples is not None)
    try:
        for index, prompt in enumerate(selected if replay_samples is None else []):
            assert engine.reset_prefix_cache(reset_connector=True)
            result = engine.generate(
                [{"prompt_token_ids": prompt["prompt_ids"]}],
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=2048,
                    logprobs=0,
                    detokenize=False,
                    seed=seed + index,
                ),
                use_tqdm=False,
            )[0]
            assert result.num_cached_tokens == 0
            answer = result.outputs[0]
            ids = prompt["prompt_ids"] + list(answer.token_ids)
            routes = answer.routed_experts
            assert answer.token_ids and routes is not None and routes.shape == (len(ids) - 1, 48, 10)
            values = [value[token].logprob for value, token in zip(answer.logprobs, answer.token_ids, strict=True)]
            assert np.isfinite(values).all()
            filename = f"routes-{index:03d}.npy"
            np.save(generation / filename, routes.astype(np.int16))
            samples.append(
                dict(
                    prompt,
                    input_ids=ids,
                    prompt_length=len(prompt["prompt_ids"]),
                    response_length=len(answer.token_ids),
                    generation_logprobs=values,
                    generation_routes=filename,
                    response_text=tokenizer.decode(answer.token_ids),
                    finish_reason=answer.finish_reason,
                )
            )
            write_json(output / "samples.json", samples)
            print("QWEN38_REAL_PROMPT_GENERATED", prompt["id"], len(ids), len(answer.token_ids), flush=True)
        print("QWEN38_REAL_PROMPT_SAMPLES_READY", len(samples), "reused" if replay_samples else "generated", flush=True)
        views = make_trace_views(samples)
        write_json(output / "fixed-prompts.json", views)
        control = output / "vllm-control"
        control.mkdir()
        records, phase_values = {}, {}
        for phase in PHASES:
            traced = phase.startswith("traced")
            if traced:
                engine.collective_rpc(
                    "qwen38_trace",
                    kwargs=dict(
                        action="install", directory=str(control / phase), tokens=32, trace_qsa=True, gdn_layers=[0]
                    ),
                )
            rows, values = [], []
            try:
                for index, view in enumerate(views):
                    assert engine.reset_prefix_cache(reset_connector=True)
                    if traced:
                        engine.collective_rpc(
                            "qwen38_trace",
                            kwargs=dict(
                                action="start", prompt=view, config_sha256=hashlib.sha256(config_bytes).hexdigest()
                            ),
                        )
                    result = engine.generate(
                        [{"prompt_token_ids": view["input_ids"]}],
                        SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1, detokenize=False),
                        use_tqdm=False,
                    )[0]
                    if traced:
                        engine.collective_rpc("qwen38_trace", kwargs=dict(action="finish"))
                    assert result.num_cached_tokens == 0
                    ids = view["input_ids"]
                    logprobs = [result.prompt_logprobs[i][token].logprob for i, token in enumerate(ids) if i]
                    routes = result.outputs[0].routed_experts
                    assert routes is not None and routes.shape == (len(ids), 48, 10)
                    assert np.isfinite(logprobs).all()
                    filename = f"{phase}-routes-{index:03d}.npy"
                    np.save(control / filename, routes.astype(np.int16))
                    rows.append(dict(id=view["id"], routes=filename, logprobs=logprobs))
                    values.extend(logprobs)
                    print("QWEN38_REAL_PROMPT_VLLM_VIEW", phase, index + 1, len(views), view["id"], flush=True)
            finally:
                if traced:
                    engine.collective_rpc("qwen38_trace", kwargs={"action": "close"})
            records[phase], phase_values[phase] = rows, np.asarray(values)
            write_json(control / f"{phase}.json", rows)
            print("QWEN38_REAL_PROMPT_VLLM_PHASE", phase, len(views), flush=True)
        report = dict(
            complete=True,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            qsa_policy="block-id",
            runtime=engine.collective_rpc("qwen38_runtime_snapshot"),
            repeat_max_abs={
                phase: float(np.abs(value - phase_values["baseline1"]).max()) for phase, value in phase_values.items()
            },
        )
        route_changes = {}
        for phase, rows in records.items():
            route_changes[phase] = sum(
                np.any(
                    np.sort(np.load(control / row["routes"]), axis=-1)
                    != np.sort(np.load(control / reference["routes"]), axis=-1),
                    axis=-1,
                )
                .sum()
                .item()
                for row, reference in zip(rows, records["baseline1"], strict=True)
            )
        report["changed_route_sets_vs_baseline1"] = route_changes
        write_json(control / "report.json", report)
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)


def megatron_worker(output, model, vllm_reference=None, route_reference="prefill", beta_ab=False):
    import numpy as np
    import torch
    from megatron.core.transformer.moe.router import TopKRouter

    from examples.grpo_trainer.qwen3_8_next.check_logprobs import load_prompts, megatron_logprobs
    from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
    from verl.utils.megatron.router_replay_utils import (
        align_r3_router_replay_data,
        build_r3_replay_mask,
        iter_model_routers,
        set_model_router_replay_action,
        set_router_replay_data,
    )

    args = SimpleNamespace(
        backend="megatron",
        model=model,
        prompts=output / "fixed-prompts.json",
        output=output / "megatron.json",
        tp=8,
        pp=1,
        ep=8,
        freeze_ple=True,
        max_length=4096,
        megatron_trace_control_dir=output / "megatron-control",
        backend_trace_dir=None,
        trace_gdn_layers=[0],
        trace_tokens=32,
        trace_start=0,
        trace_qsa=True,
    )
    prompts = load_prompts(args.prompts, 4095)
    vllm_reference = vllm_reference or output
    reference = {row["id"]: row for row in json.loads((vllm_reference / "vllm-control/baseline1.json").read_text())}
    hooks, counts, phase_name = [], {}, [None]
    initial_versions = None
    beta_probe = None

    def select(phase, model):
        nonlocal initial_versions, beta_probe
        versions = {name: (id(p), p._version) for name, p in model.named_parameters()}
        if initial_versions is None:
            initial_versions = versions
            apply_router_replay_patch()

            def check(module, _inputs, result):
                replay = module.router_replay
                active = replay.target_replay_mask.bool()
                desired = torch.zeros_like(result[1]).scatter_(1, replay.target_topk_idx, 1)
                assert torch.equal(result[1][active], desired[active]), "Expert dispatch differs from replay target"
                counts[phase_name[0]] = counts.get(phase_name[0], 0) + 1

            routers = [module for module in model.modules() if isinstance(module, TopKRouter)]
            assert len(routers) == 48
            for module in routers:
                module.config.enable_routing_replay = True
                module.config.moe_enable_routing_replay = True
                module.router_replay = RouterReplay()
                hooks.append(module.register_forward_hook(check))
            if beta_ab:
                from examples.grpo_trainer.qwen3_8_next.gdn_beta_ab import GdnBetaPrecisionProbe

                beta_probe = GdnBetaPrecisionProbe(model)
        assert versions == initial_versions, "Forward diagnosis changed a parameter"
        phase_name[0] = phase
        if beta_probe is not None:
            beta_probe.select(phase)

    def replay(phase, prompt, model, ids):
        assert prompt["input_ids"] == ids[0, : len(prompt["input_ids"])].tolist()
        path = (
            output / "generation" / prompt["generation_routes"]
            if route_reference == "generation"
            else vllm_reference / "vllm-control" / reference[prompt["id"]]["routes"]
        )
        values = torch.from_numpy(np.load(path)).to(ids.device)
        length = len(prompt["input_ids"])
        assert values.shape == (length - (route_reference == "generation"), 48, 10)
        routes = torch.nested.as_nested_tensor([values], layout=torch.jagged)
        if route_reference == "generation":
            real_ids = torch.nested.as_nested_tensor([ids[0, :length]], layout=torch.jagged)
            routes = align_r3_router_replay_data(routes, real_ids)
            response_mask = torch.ones((1, prompt["response_length"]), device=ids.device, dtype=torch.bool)
            mask = build_r3_replay_mask(real_ids, response_mask)
        else:
            mask = torch.nested.as_nested_tensor(
                [torch.ones(len(values), device=ids.device, dtype=torch.bool)], layout=torch.jagged
            )
        for _, router in iter_model_routers(model):
            router.clear_indices()
        set_model_router_replay_action(model, RouterReplayAction.REPLAY_FORWARD)
        set_router_replay_data(routes, None, model.language_model.config, replay_mask=mask, model=model)

    try:
        megatron_logprobs(
            args,
            prompts,
            phase_hook=select,
            prompt_hook=replay,
            trace_phases=("baseline1", "traced0", "traced1", "baseline2") if beta_ab else None,
        )
        assert counts == {phase: len(prompts) * 48 for phase in PHASES}
    finally:
        if beta_probe is not None:
            beta_probe.close()
            write_json(
                output / f"beta-audit-rank-{int(os.environ['RANK']):02d}.json",
                dict(phases=beta_probe.phases, native_restored=True, production_changed=False),
            )
        for hook in hooks:
            hook.remove()
        write_json(
            output / f"replay-audit-rank-{int(os.environ['RANK']):02d}.json",
            dict(
                replay_reference=route_reference,
                replay_scope=(
                    "production R3: all causal prompt/response queries except the unsampled final row"
                    if route_reference == "generation"
                    else "all real prefill queries"
                ),
                verified_router_calls=counts,
                production_changed=False,
            ),
        )


def assess(output, vllm_reference=None, route_reference="prefill"):
    import numpy as np

    from examples.grpo_trainer.qwen3_8_next.backend_trace import compare_backend_traces

    samples = json.loads((output / "samples.json").read_text())
    vllm_reference = vllm_reference or output
    vllm = {row["id"]: row for row in json.loads((vllm_reference / "vllm-control/baseline1.json").read_text())}
    meg = json.loads((output / "megatron.json").read_text())
    mega = {row["id"]: row for row in meg["records"]}
    vcontrol = json.loads((vllm_reference / "vllm-control/report.json").read_text())
    mcontrol = json.loads((output / "megatron-control/comparison.json").read_text())
    controls_valid = all(value == 0 for value in vcontrol["repeat_max_abs"].values()) and all(
        value["all"]["max_abs"] == 0 for value in mcontrol["comparisons"].values()
    )
    controls_valid &= all(value == 0 for value in vcontrol["changed_route_sets_vs_baseline1"].values())
    activation_controls = assess_trace_repeats(output, vllm_reference)
    controls_valid &= activation_controls["all_windows_repeatable"]
    assert vcontrol["config_sha256"] == meg["config_sha256"]
    records, differences = (
        [],
        {name: [] for name in ["prefill_vs_megatron", "generation_vs_megatron", "generation_vs_prefill"]},
    )
    for sample in samples:
        name = sample["id"] + "/response"
        ref = np.asarray(response_logprobs(sample, vllm[name]["logprobs"]))
        actor = np.asarray(response_logprobs(sample, mega[name]["logprobs"]))
        rollout = np.asarray(sample["generation_logprobs"])
        assert ref.shape == actor.shape == rollout.shape
        row = dict(id=sample["id"], prompt_length=sample["prompt_length"], response_length=sample["response_length"])
        for label, left, right in [
            ("prefill_vs_megatron", ref, actor),
            ("generation_vs_megatron", rollout, actor),
            ("generation_vs_prefill", rollout, ref),
        ]:
            delta = np.abs(left - right)
            assert np.isfinite(delta).all()
            differences[label].append(delta)
            row[label] = dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()))
        original = np.load(output / "generation" / sample["generation_routes"])
        prefill = np.load(vllm_reference / "vllm-control" / vllm[name]["routes"])[:-1]
        changed = np.any(np.sort(original, axis=-1) != np.sort(prefill, axis=-1), axis=-1)
        row["generation_prefill_changed_route_sets"] = int(changed.sum())
        row["generation_prefill_changed_response_route_sets"] = int(changed[sample["prompt_length"] - 1 :].sum())
        records.append(row)
    summary = {
        name: dict(
            response_mean_abs=float(np.mean([x.mean() for x in values])),
            token_mean_abs=float(np.concatenate(values).mean()),
            max_abs=float(np.concatenate(values).max()),
            p99_abs=float(np.quantile(np.concatenate(values), 0.99)),
        )
        for name, values in differences.items()
    }
    report = dict(
        complete=True,
        full_model_acceptance=False,
        production_changed=False,
        megatron_route_reference=route_reference,
        vllm_reference=str(vllm_reference),
        controls_valid=bool(controls_valid),
        scope=(
            f"Real sampled GSM8K responses; TP8/PP1/EP8 diagnosis with {route_reference} expert choices. "
            + (
                "Generation/actor expert sets match on all scored causal queries; "
                "prefill comparisons use different routes. "
                if route_reference == "generation"
                else "Generation comparisons include decode/prefill and expert-set differences. "
            )
            + "production R3 replays all scored causal queries using generation routes. This is not TP8/PP4 acceptance."
        ),
        samples=len(samples),
        response_tokens=sum(x["response_length"] for x in samples),
        real_sequences_beyond_qsa_budget=sum(len(x["input_ids"]) > 2052 for x in samples),
        fixed_prompts_sha256=hashlib.sha256((output / "fixed-prompts.json").read_bytes()).hexdigest(),
        summaries=summary,
        records=records,
        vllm_repeat_controls=vcontrol["repeat_max_abs"],
        megatron_repeat_controls=mcontrol["comparisons"],
        activation_repeat_controls=activation_controls,
        diagnostics=[
            compare_backend_traces(
                output / "megatron-control/activations/traced0",
                vllm_reference / "vllm-control/traced0",
                name,
                max_tokens=min(32, len(mega[name]["input_ids"]) - 1 - mega[name]["trace_token_start"]),
            )
            for name in mega
        ],
    )
    write_json(output / "report.json", report)
    print(
        "QWEN38_REAL_PROMPT_TRACE_COMPLETE",
        json.dumps(
            {key: report[key] for key in ["complete", "controls_valid", "samples", "response_tokens", "summaries"]}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("vllm", "megatron", "assess"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--replay-samples", type=Path, help="reuse an existing real-prompt generation directory")
    parser.add_argument(
        "--vllm-reference", type=Path, help="completed immutable vLLM control used by Megatron and assess"
    )
    parser.add_argument("--route-reference", choices=("prefill", "generation"), default="prefill")
    parser.add_argument("--gdn-beta-ab", action="store_true", help="same-model native / FP32 beta / restored control")
    parser.add_argument("--packing-ab", action="store_true", help="same-model individual / production THD packing")
    parser.add_argument("--production-capture", type=Path, help="completed first-step production capture to replay")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.packing_ab or args.production_capture:
        if (
            args.gdn_beta_ab
            or args.route_reference != "generation"
            or bool(args.vllm_reference) == bool(args.production_capture)
        ):
            parser.error("Packing replay requires generation routes and exactly one completed reference/capture")
        from examples.grpo_trainer.qwen3_8_next.packing_ab import assess_packing, packing_worker

        if args.phase == "megatron":
            packing_worker(args.output, args.model)
        elif args.phase == "assess":
            assess_packing(args.output, args.vllm_reference)
        else:
            parser.error("Packing A/B reuses a completed vLLM reference")
    elif args.phase == "vllm":
        vllm_worker(args.output, args.model, args.dataset, args.count, args.seed, args.replay_samples)
    elif args.phase == "megatron":
        megatron_worker(args.output, args.model, args.vllm_reference, args.route_reference, args.gdn_beta_ab)
    elif args.gdn_beta_ab:
        from examples.grpo_trainer.qwen3_8_next.gdn_beta_ab import assess_beta_ab

        assess_beta_ab(args.output, args.vllm_reference or args.output, args.route_reference)
    else:
        assess(args.output, args.vllm_reference, args.route_reference)


if __name__ == "__main__":
    main()
