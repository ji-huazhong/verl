# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Compare individual and production THD forwards on identical real responses."""

import hashlib
import json
import os
from types import SimpleNamespace

PHASES = ("manual0", "engine_single", "packed0", "packed1", "manual1")


def unroll_document_scores(values, lengths):
    """Each document's last input predicts an unsampled token; drop it separately."""
    if len(values) != sum(lengths) or any(length < 2 for length in lengths):
        raise ValueError("Logprob rows do not match real document boundaries")
    offset, result = 0, []
    for length in lengths:
        result.append(values[offset : offset + length - 1])
        offset += length
    return result


def mrope_positions(ids):
    import torch

    values = torch.cat([torch.arange(len(row), device=row.device).expand(4, -1) for row in ids.unbind()], dim=1)
    return torch.nested.nested_tensor_from_jagged(values, offsets=ids.offsets(), jagged_dim=2)


def packing_worker(output, model_path):
    import numpy as np
    import torch
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
    from megatron.core.transformer.moe.router import TopKRouter
    from tensordict import TensorDict

    from examples.grpo_trainer.qwen3_8_next.check_logprobs import megatron_logprobs
    from examples.grpo_trainer.qwen3_8_next.real_prompt_trace import write_json
    from examples.grpo_trainer.qwen3_8_next.repetition import record_megatron_runtime
    from verl.models.mcore.model_forward import gptmodel_forward_model_engine
    from verl.models.mcore.util import accepts_packed_thd_vlm_inputs, preprocess_thd_engine
    from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
    from verl.utils.megatron.router_replay_utils import (
        align_r3_router_replay_data,
        build_r3_replay_mask,
        iter_model_routers,
        set_model_router_replay_action,
        set_router_replay_data,
    )
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
    from verl.utils.seqlen_balancing import rearrange_micro_batches

    samples = json.loads((output / "samples.json").read_text())
    args = SimpleNamespace(
        model=model_path,
        tp=8,
        pp=1,
        ep=8,
        freeze_ple=True,
        max_length=4096,
    )

    def probe(model, mixed, provider):
        assert accepts_packed_thd_vlm_inputs(mixed), "Production MRoPE THD path is unavailable"
        rank = torch.distributed.get_rank()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        directory = output / "packing-control"
        if rank == 0:
            directory.mkdir(exist_ok=False)
        torch.distributed.barrier()
        record_megatron_runtime(directory / "runtime", provider, mixed, gdn_layers=[0])
        ids = torch.nested.as_nested_tensor(
            [torch.tensor(row["input_ids"], device=device) for row in samples], layout=torch.jagged
        )
        _, partitions = rearrange_micro_batches(
            TensorDict({"input_ids": ids}, batch_size=[len(samples)]), max_token_len=4096, same_micro_num_in_dp=False
        )
        assert sorted(i for group in partitions for i in group) == list(range(len(samples)))
        assert len(partitions) > 1 and any(len(group) > 1 for group in partitions)
        native_versions = {name: (id(p), p._version) for name, p in model.named_parameters()}
        hooks, counts, layouts, logits_dtypes = [], {}, [], set()
        phase_name, active = [None], [None]
        apply_router_replay_patch()

        def check_router(module, _inputs, result):
            replay = module.router_replay
            mask = replay.target_replay_mask.bool()
            desired = torch.zeros_like(result[1]).scatter_(1, replay.target_topk_idx, 1)
            assert torch.equal(result[1][mask], desired[mask]), "Expert dispatch differs from generation routes"
            counts[phase_name[0]] = counts.get(phase_name[0], 0) + 1

        def check_layout(_module, _inputs, kwargs):
            group, manual = active[0]
            lengths = [len(samples[i]["input_ids"]) for i in group]
            padded = [(length + 7) // 8 * 8 for length in lengths]
            cu = [0]
            for length in padded:
                cu.append(cu[-1] + length)
            assert kwargs["packed_seq_params"].cu_seqlens_q.tolist() == cu
            tokens, positions = kwargs["input_ids"], kwargs["position_ids"]
            assert tokens.shape == (1, cu[-1]) and positions.shape == (3, 1, cu[-1])
            for i, length, start, stop in zip(group, lengths, cu[:-1], cu[1:], strict=True):
                assert tokens[0, start : start + length].tolist() == samples[i]["input_ids"]
                expected_positions = list(range(length)) + (
                    [*range(length, stop - start)] if manual else [0] * (stop - start - length)
                )
                assert all(row.tolist() == expected_positions for row in positions[:, 0, start:stop])
                pad_id = provider.qwen3_8_next_eos_token_id if manual else 0
                assert tokens[0, start + length : stop].tolist() == [pad_id] * (stop - start - length)
            layouts.append(
                dict(phase=phase_name[0], sample_ids=[samples[i]["id"] for i in group], cu=cu, lengths=lengths)
            )

        routers = [m for m in model.modules() if isinstance(m, TopKRouter)]
        assert len(routers) == 48
        for module in routers:
            module.config.enable_routing_replay = module.config.moe_enable_routing_replay = True
            module.router_replay = RouterReplay()
            hooks.append(module.register_forward_hook(check_router))
        hooks.append(model.register_forward_pre_hook(check_layout, with_kwargs=True))

        def logits_processor(logits, label):
            logits_dtypes.add(str(logits.dtype))
            # Production CE may consume logits. Independent clones isolate its arithmetic.
            native = vocab_parallel_log_probs_from_logits(logits.clone(), label)
            fp32 = -vocab_parallel_cross_entropy(logits.float().clone().contiguous(), label)
            return dict(log_probs=native, fp32_log_probs=fp32)

        try:
            for phase in PHASES:
                assert native_versions == {name: (id(p), p._version) for name, p in model.named_parameters()}
                phase_name[0] = phase
                groups = partitions if phase.startswith("packed") else [[i] for i in range(len(samples))]
                records = {}
                for group in groups:
                    manual = phase.startswith("manual")
                    active[0] = (group, manual)
                    batch_ids = torch.nested.as_nested_tensor([ids[i] for i in group], layout=torch.jagged)
                    routes = torch.nested.as_nested_tensor(
                        [
                            torch.from_numpy(
                                np.load(output / "generation" / samples[i]["generation_routes"], allow_pickle=False)
                            ).to(device)
                            for i in group
                        ],
                        layout=torch.jagged,
                    )
                    routes = align_r3_router_replay_data(routes, batch_ids)
                    response_mask = torch.nested.as_nested_tensor(
                        [torch.ones(samples[i]["response_length"], dtype=torch.bool, device=device) for i in group],
                        layout=torch.jagged,
                    )
                    mask = build_r3_replay_mask(batch_ids, response_mask)
                    for _, router in iter_model_routers(model):
                        router.clear_indices()
                    set_model_router_replay_action(model, RouterReplayAction.REPLAY_FORWARD)
                    set_router_replay_data(routes, None, model.language_model.config, replay_mask=mask, model=model)
                    _, packed, _ = preprocess_thd_engine(batch_ids)
                    total = int(packed.cu_seqlens_q[-1].item())

                    def forward_step(_iterator, module, manual=manual, group=group, total=total, batch_ids=batch_ids):
                        if manual:
                            assert len(group) == 1
                            row = samples[group[0]]["input_ids"]
                            padded_ids = torch.tensor(
                                [row + [provider.qwen3_8_next_eos_token_id] * (total - len(row))], device=device
                            )
                            cu = torch.tensor([0, total], device=device, dtype=torch.int32)
                            logits = module(
                                input_ids=padded_ids,
                                position_ids=torch.arange(total, device=device).reshape(1, 1, -1).expand(3, 1, -1),
                                attention_mask=None,
                                packed_seq_params=PackedSeqParams(
                                    qkv_format="thd",
                                    cu_seqlens_q=cu,
                                    cu_seqlens_kv=cu,
                                    max_seqlen_q=total,
                                    max_seqlen_kv=total,
                                ),
                            )
                            outputs = logits_processor(logits[:, :-1].contiguous(), padded_ids[:, 1:])
                        else:
                            outputs = gptmodel_forward_model_engine(
                                module,
                                batch_ids,
                                {},
                                logits_processor=logits_processor,
                                logits_processor_args={"label": batch_ids.clone()},
                                vision_model=True,
                                pad_token_id=provider.qwen3_8_next_eos_token_id,
                                position_ids=mrope_positions(batch_ids),
                                data_format="thd",
                            )

                        def collect(scores, non_loss_data=False):
                            assert non_loss_data
                            if manual:
                                return {
                                    key: [value[0, : len(row) - 1].float().cpu().tolist()]
                                    for key, value in scores.items()
                                }
                            lengths = [len(samples[i]["input_ids"]) for i in group]
                            return {
                                key: unroll_document_scores(value.values().float().cpu().tolist(), lengths)
                                for key, value in scores.items()
                            }

                        return outputs, collect

                    with torch.no_grad():
                        result = get_forward_backward_func()(
                            forward_step_func=forward_step,
                            data_iterator=iter([None]),
                            model=[mixed],
                            num_microbatches=1,
                            seq_length=total,
                            micro_batch_size=1,
                            forward_only=True,
                            collect_non_loss_data=True,
                        )[0]
                    for offset, i in enumerate(group):
                        records[samples[i]["id"]] = {key: values[offset] for key, values in result.items()}
                expected_calls = len(groups) * 48
                assert counts[phase] == expected_calls and len(records) == len(samples)
                if rank == 0:
                    write_json(directory / f"{phase}.json", records)
                    print("QWEN38_PACKING_PHASE", phase, flush=True)
                torch.distributed.barrier()
            assert native_versions == {name: (id(p), p._version) for name, p in model.named_parameters()}
        finally:
            for hook in hooks:
                hook.remove()
            write_json(
                output / f"packing-audit-rank-{rank:02d}.json",
                dict(
                    phases=counts,
                    layouts=layouts,
                    parameter_versions_unchanged=native_versions
                    == {name: (id(p), p._version) for name, p in model.named_parameters()},
                    partitions=partitions,
                    config_sha256=hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
                    logits_dtypes=sorted(logits_dtypes),
                    production_changed=False,
                ),
            )

    megatron_logprobs(args, samples, loaded_model_probe=probe)


def assess_packing(output, reference=None):
    import numpy as np

    from examples.grpo_trainer.qwen3_8_next.real_prompt_trace import response_logprobs, write_json

    samples = json.loads((output / "samples.json").read_text())
    results = {phase: json.loads((output / "packing-control" / f"{phase}.json").read_text()) for phase in PHASES}
    assert results["manual0"] == results["manual1"], "Individual forwards changed after packing"
    assert results["packed0"] == results["packed1"], "Packed forwards are not repeatable"
    audits = [json.loads((output / f"packing-audit-rank-{rank:02d}.json").read_text()) for rank in range(8)]
    assert all(audit["parameter_versions_unchanged"] for audit in audits)
    assert all(audit["layouts"] == audits[0]["layouts"] for audit in audits)
    assert all(audit["phases"] == audits[0]["phases"] for audit in audits)
    required_calls = {
        phase: 48 * (len(audits[0]["partitions"]) if phase.startswith("packed") else len(samples)) for phase in PHASES
    }
    assert audits[0]["phases"] == required_calls
    production = reference is None
    assert len({audit["config_sha256"] for audit in audits}) == 1
    if production:
        capture = json.loads((output / "production-capture.json").read_text())
        assert capture["complete"] and capture["step"] == 1 and capture["metadata"]["before_policy_update"]
    else:
        baseline = json.loads((reference / "vllm-control/report.json").read_text())
        assert all(audit["config_sha256"] == baseline["config_sha256"] for audit in audits)
        assert all(value == 0 for value in baseline["repeat_max_abs"].values())
    deltas, records = {}, []
    for sample in samples:
        name = sample["id"]
        values = {
            phase: {key: np.asarray(response_logprobs(sample, row)) for key, row in result[name].items()}
            for phase, result in results.items()
        }
        comparisons = [
            (f"{phase}_vs_generation", values[phase]["log_probs"], np.asarray(sample["generation_logprobs"]))
            for phase in ["manual0", "engine_single", "packed0"]
        ]
        comparisons += [
            ("engine_vs_manual", values["engine_single"]["log_probs"], values["manual0"]["log_probs"]),
            ("packed_vs_single", values["packed0"]["log_probs"], values["engine_single"]["log_probs"]),
        ]
        comparisons += [
            (f"{phase}_native_vs_fp32_ce", values[phase]["log_probs"], values[phase]["fp32_log_probs"])
            for phase in ["manual0", "engine_single", "packed0"]
        ]
        if production:
            actor = np.asarray(sample["production_actor_logprobs"])
            comparisons.append(("production_actor_vs_generation", actor, np.asarray(sample["generation_logprobs"])))
            comparisons += [
                (f"{phase}_vs_production_actor", values[phase]["log_probs"], actor)
                for phase in ["manual0", "engine_single", "packed0"]
            ]
        mask = np.asarray(sample.get("response_mask", [True] * sample["response_length"]), dtype=bool)
        assert mask.any() and mask.shape == (sample["response_length"],)
        contexts = sample["prompt_length"] + np.arange(sample["response_length"])
        row = dict(id=name, prompt_length=sample["prompt_length"], response_length=sample["response_length"])
        for label, left, right in comparisons:
            assert left.shape == right.shape
            delta = np.abs(left - right)[mask]
            assert np.isfinite(delta).all()
            deltas.setdefault(label, []).append(delta)
            row[label] = dict(
                mean_abs=float(delta.mean()),
                max_abs=float(delta.max()),
                max_error_context=int(contexts[mask][delta.argmax()]),
            )
            for suffix, selected in [("le2051", contexts[mask] <= 2051), ("gt2051", contexts[mask] > 2051)]:
                if selected.any():
                    deltas.setdefault(label + "_ctx_" + suffix, []).append(delta[selected])
        records.append(row)
    summaries = {
        name: dict(
            responses=len(values),
            scored_tokens=sum(len(value) for value in values),
            response_mean_abs=float(np.mean([v.mean() for v in values])),
            token_mean_abs=float(np.concatenate(values).mean()),
            max_abs=float(np.concatenate(values).max()),
            p99_abs=float(np.quantile(np.concatenate(values), 0.99)),
        )
        for name, values in deltas.items()
    }
    report = dict(
        complete=True,
        controls_valid=True,
        full_model_acceptance=False,
        production_changed=False,
        samples=len(samples),
        response_tokens=sum(s["response_length"] for s in samples),
        summaries=summaries,
        records=records,
        same_loaded_model=True,
        partitions=audits[0]["partitions"],
        production_capture_replayed=production,
        rollout_repeat_controls_available=not production,
        selection_scope="Length/high-error production subset; use the captured full-batch metric for acceptance."
        if production
        else "Fixed real diagnostic responses.",
        scope=(
            "TP8/PP1/EP8; unchanged actual generation routes and responses; legacy individual / "
            "production engine individual / production engine packed / repeat / restored individual. "
            "Does not execute PP4 or weight refit itself; production-capture replay compares against stored PP4 scores."
        ),
    )
    write_json(output / "report.json", report)
    print("QWEN38_REAL_PROMPT_PACKING_COMPLETE", json.dumps(summaries), flush=True)
