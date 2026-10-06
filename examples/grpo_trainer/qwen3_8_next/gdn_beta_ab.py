# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Same-model beta precision diagnostic on fixed real responses and routes."""

import json


class GdnBetaPrecisionProbe:
    """Change only sigmoid's output precision; preserve native decay computation."""

    def __init__(self, model):
        self.modules = [m for m in model.modules() if type(m).__name__ == "Qwen38NextGatedDeltaNet"]
        if len(self.modules) != 36:
            raise ValueError("Expected all 36 Flash-Next GDN modules")
        self.originals = [m._compute_g_and_beta for m in self.modules]
        self.phases = []

    def select(self, phase):
        if phase not in ("baseline0", "baseline1", "traced0", "traced1", "baseline2"):
            raise ValueError(f"Unknown beta comparison phase: {phase}")
        candidate = phase in ("traced0", "traced1")
        for module, original in zip(self.modules, self.originals, strict=True):

            def fp32_beta(a_log, dt_bias, alpha, beta, original=original):
                g, _ = original(a_log, dt_bias, alpha, beta)
                return g, beta.float().sigmoid()

            module._compute_g_and_beta = fp32_beta if candidate else original
        self.phases.append(dict(phase=phase, beta_dtype="float32" if candidate else "bfloat16", modules=36))

    def close(self):
        for module, original in zip(self.modules, self.originals, strict=True):
            module._compute_g_and_beta = original


def assess_beta_ab(output, reference, route_reference):
    import numpy as np

    from examples.grpo_trainer.qwen3_8_next.backend_trace import compare_backend_traces
    from examples.grpo_trainer.qwen3_8_next.real_prompt_trace import response_logprobs, write_json
    from examples.grpo_trainer.qwen3_8_next.repetition import compare_repeated_logprobs

    control = output / "megatron-control"
    results = {
        phase: json.loads((control / f"{phase}.json").read_text())
        for phase in ("baseline0", "baseline1", "traced0", "traced1", "baseline2")
    }
    checks = {
        "native_repeat": compare_repeated_logprobs(results["baseline0"], results["baseline1"]),
        "native_restored": compare_repeated_logprobs(results["baseline1"], results["baseline2"]),
        "candidate_repeat": compare_repeated_logprobs(results["traced0"], results["traced1"]),
    }
    assert all(check["all"]["max_abs"] == 0 for check in checks.values()), "Beta A/B logprobs are not repeatable"
    vcontrol = json.loads((reference / "vllm-control/report.json").read_text())
    assert all(value == 0 for value in vcontrol["repeat_max_abs"].values())
    assert all(value == 0 for value in vcontrol["changed_route_sets_vs_baseline1"].values())
    assert all(row["config_sha256"] == vcontrol["config_sha256"] for row in results.values())
    views = json.loads((output / "fixed-prompts.json").read_text())
    comparisons, activation_checks = [], []
    upstream_parts = [
        "projection/qkvz",
        "projection/ba",
        "conv",
        "prepared/q",
        "prepared/k",
        "prepared/v",
        "prepared/z",
        "prepared/b",
        "prepared/a",
        "g",
    ]
    upstream = {"layers/00/attn_hc/input", "layers/00/attn_hc/mixed", "layers/00/gdn/input"}
    upstream |= {f"layers/00/gdn/{part}/tp-{rank:02d}" for part in upstream_parts for rank in range(8)}
    for view in views:
        count = min(32, len(view["input_ids"]) - 1 - view["trace_token_start"])
        pair_checks = {}
        for name, left, right in [
            ("native_restored", "baseline1", "baseline2"),
            ("candidate_repeat", "traced0", "traced1"),
        ]:
            check = compare_backend_traces(
                control / "activations" / left, control / "activations" / right, view["id"], max_tokens=count
            )
            assert check["first_nonzero_stage"] is None, (view["id"], name)
            pair_checks[name] = True
        change = compare_backend_traces(
            control / "activations/baseline1", control / "activations/traced1", view["id"], max_tokens=count
        )
        assert upstream <= change["stages"].keys(), "Missing native/candidate upstream snapshots"
        assert all(change["stages"][name]["max_abs"] == 0 for name in upstream), "Beta A/B changed upstream values"
        for rank in range(8):
            beta = change["stages"][f"layers/00/gdn/beta/tp-{rank:02d}"]
            assert beta["reference_dtype"] == "torch.bfloat16" and beta["candidate_dtype"] == "torch.float32"
        activation_checks.append(dict(id=view["id"], upstream_bitwise_equal=True, **pair_checks))
        comparisons.append(change)

    samples = json.loads((output / "samples.json").read_text())
    native = {r["id"]: r for r in results["baseline1"]["records"]}
    candidate = {r["id"]: r for r in results["traced1"]["records"]}
    prefill = {r["id"]: r for r in json.loads((reference / "vllm-control/baseline1.json").read_text())}
    deltas, records = {}, []
    for sample in samples:
        name = sample["id"] + "/response"
        row = dict(id=sample["id"], response_length=sample["response_length"])
        targets = dict(
            generation=np.asarray(sample["generation_logprobs"]),
            prefill=np.asarray(response_logprobs(sample, prefill[name]["logprobs"])),
        )
        for variant, values in [("native", native), ("fp32_beta", candidate)]:
            actor = np.asarray(response_logprobs(sample, values[name]["logprobs"]))
            for backend, target in targets.items():
                assert target.shape == actor.shape
                delta = np.abs(target - actor)
                assert np.isfinite(delta).all()
                label = f"{variant}_vs_{backend}"
                row[label] = dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()))
                deltas.setdefault(label, []).append(delta)
        records.append(row)
    summaries = {
        name: dict(
            response_mean_abs=float(np.mean([d.mean() for d in values])),
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
        megatron_route_reference=route_reference,
        same_loaded_model=True,
        variant="fp32-beta-only",
        summaries=summaries,
        records=records,
        repeat_controls=checks,
        activation_controls=activation_checks,
        diagnostics=comparisons,
        scope=(
            "TP8/PP1/EP8 original real responses with fixed routes; native beta / FP32 beta / restored native. "
            "Native decay and every upstream layer0 snapshot must remain bitwise equal. "
            "Not packed TP8/PP4 acceptance."
        ),
    )
    write_json(output / "report.json", report)
    print("QWEN38_REAL_PROMPT_BETA_AB_COMPLETE", json.dumps(summaries), flush=True)
