# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Bounded native-HF prefix snapshots for finding the first numerical divergence.

Snapshots are private diagnostic artifacts, not a numerical acceptance gate.
The hook design follows the original-forward tracing in ji-huazhong/verl,
hz/feat/qwen3.8-flash-next, revision 8172ae153714a6a3046b619c0b3b7f098b9b6088.
"""

import hashlib
import json
from pathlib import Path

import torch


class PrefixActivationTrace:
    def __init__(self, model, directory, tokens=32, max_bytes=768 * 1024**2, precision_variant="native"):
        if tokens < 1:
            raise ValueError("Trace prefix length must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.tokens, self.max_bytes = tokens, max_bytes
        self.precision_variant = precision_variant
        self.current = None
        self.values, self.handles, self.records = {}, [], []
        self.bytes = 0
        language = model.model.language_model
        for index, layer in enumerate(language.layers):
            name = f"layers/{index:02d}"
            self._attach(name + "/input", layer, before=True)
            self._attach(name + "/output", layer)
            for site in ("attn", "mlp"):
                hc = getattr(layer, site + "_hyper_connection")
                self._attach(name + f"/{site}_hc/input", hc, before=True)
                self._attach(name + f"/{site}_hc/mixed", hc, item=0)
                self._attach(name + f"/{site}_hc/injection", hc, item=2)
                for child in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up"):
                    self._attach(name + f"/{site}_hc/{child}", getattr(hc, child))
            attention = getattr(layer, "linear_attn", None)
            if attention is None:
                self._attach(name + "/attention/output", layer.self_attn, item=0)
            else:
                self._attach(name + "/attention/output", attention)
                for child in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "norm", "out_proj"):
                    self._attach(name + f"/gdn/{child}", getattr(attention, child))
            self._attach(name + "/mlp/output", layer.mlp)
            self._attach(name + "/router/logits", layer.mlp.gate, item=0)
            self._attach(name + "/router/indices", layer.mlp.gate, item=2)
            if layer.ple is not None:
                self._attach(name + "/ple/output", layer.ple)
        self._attach("final_mixer/input", language.hyper_connection_mixer, before=True)
        self._attach("final_mixer/output", language.hyper_connection_mixer)

    def _attach(self, name, module, *, before=False, item=None):
        if before:

            def hook(_module, args, kwargs):
                self.capture(name, kwargs.get("hidden_states", args[0] if args else None))

            self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        else:

            def hook(_module, _args, output):
                self.capture(name, output[item] if item is not None else output)

            self.handles.append(module.register_forward_hook(hook))

    def start(self, prompt, repeat):
        if self.current is not None or self.values:
            raise RuntimeError("Previous activation trace was not drained")
        self.current = dict(prompt, repeat=repeat)

    def capture(self, name, value):
        if self.current is None:
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Activation is not a tensor: {name}")
        length = len(self.current["input_ids"])
        # The diagnostic deliberately uses one document per HF batch. GDN norm
        # flattens token/head axes; reconstruct those axes before slicing tokens.
        if value.ndim >= 3 and value.shape[:2] == (1, length):
            prefix = value[0, : self.tokens]
        elif value.ndim >= 2 and value.shape[0] % length == 0:
            prefix = value.reshape(length, value.shape[0] // length, *value.shape[1:])[: self.tokens]
        else:
            raise ValueError(f"Unknown token layout at {name}: {tuple(value.shape)}, length={length}")
        if name in self.values:
            raise RuntimeError(f"Duplicate original-forward activation: {name}")
        size = prefix.numel() * prefix.element_size()
        if self.bytes + size > self.max_bytes:
            raise RuntimeError("Prefix activation trace exceeded its memory budget")
        self.values[name] = prefix.detach().clone()
        self.bytes += size

    def finish(self, config_sha256):
        if self.current is None:
            raise RuntimeError("No activation trace is active")
        metadata = dict(
            self.current,
            config_sha256=config_sha256,
            prefix_tokens=self.tokens,
            precision_variant=self.precision_variant,
        )
        stem = hashlib.sha256(metadata["id"].encode()).hexdigest()[:16]
        filename = f"{stem}-repeat{metadata['repeat']}.pt"
        path = self.directory / filename
        if path.exists():
            raise FileExistsError(path)
        # Materialize only after the forward finishes. Hooks never synchronize
        # GPU execution or inspect tensor values, and never retain full sequences.
        values = {name: value.cpu() for name, value in self.values.items()}
        torch.save(dict(metadata=metadata, activations=values), path)
        self.records.append(dict(metadata, file=filename, stages=list(values), bytes=self.bytes))
        self.current = None
        self.values.clear()
        self.bytes = 0
        (self.directory / "index.json").write_text(json.dumps(self.records, indent=2) + "\n")

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()
        self.current = None


def activation_differences(reference, candidate):
    """Compare CPU snapshots with equal coordinates, allowing different dtypes."""
    if list(reference) != list(candidate):
        raise ValueError("Activation stages/order differ")
    result = {}
    for name, expected in reference.items():
        actual = candidate[name]
        if expected.shape != actual.shape or expected.device.type != "cpu" or actual.device.type != "cpu":
            raise ValueError(f"Activation coordinates differ: {name}")
        if not (torch.isfinite(expected).all() and torch.isfinite(actual).all()):
            raise ValueError(f"Non-finite activation: {name}")
        delta = actual.double() - expected.double()
        norm = expected.double().norm().item()
        report = dict(
            mean_abs=delta.abs().mean().item(),
            max_abs=delta.abs().max().item(),
            relative_l2=delta.norm().item() / norm if norm else None,
            changed_fraction=(delta != 0).double().mean().item(),
        )
        if name.endswith("/router/indices"):
            # Reordering equal-score experts does not change the selected set.
            left, right = expected.sort(dim=-1).values, actual.sort(dim=-1).values
            report["changed_expert_set_fraction"] = (left != right).any(dim=-1).double().mean().item()
        result[name] = report
    return result


def compare_trace_directories(directories):
    """Report fixed-input repeats, shared prefixes and matching cross-dtype cases."""
    indexes = {str(path): json.loads((Path(path) / "index.json").read_text()) for path in directories}
    comparisons = []

    def compare(left_dir, left, right_dir, right, kind):
        if left["config_sha256"] != right["config_sha256"]:
            raise ValueError("Cannot compare traces from different model configurations")
        count = min(left["prefix_tokens"], right["prefix_tokens"], len(left["input_ids"]), len(right["input_ids"]))
        if left["input_ids"][:count] != right["input_ids"][:count]:
            raise ValueError("Trace token coordinates differ")
        expected = torch.load(Path(left_dir) / left["file"], map_location="cpu", weights_only=True)["activations"]
        actual = torch.load(Path(right_dir) / right["file"], map_location="cpu", weights_only=True)["activations"]
        if kind == "cross_run":
            if left.get("precision_variant", "native") != right.get("precision_variant", "native"):
                kind = "precision_ablation"
            elif next(iter(expected.values())).dtype != next(iter(actual.values())).dtype:
                kind = "cross_dtype"
            else:
                kind = "independent_run"
        stages = activation_differences(
            {name: value[:count] for name, value in expected.items()},
            {name: value[:count] for name, value in actual.items()},
        )
        first = next((name for name, value in stages.items() if value["max_abs"] > 0), None)
        first_route = next(
            (name for name, value in stages.items() if value.get("changed_expert_set_fraction", 0) > 0), None
        )
        comparisons.append(
            dict(
                kind=kind,
                reference=dict(directory=left_dir, id=left["id"], repeat=left["repeat"]),
                candidate=dict(directory=right_dir, id=right["id"], repeat=right["repeat"]),
                compared_prefix_tokens=count,
                first_nonzero_stage=first,
                first_changed_expert_set=first_route,
                stages=stages,
            )
        )

    for directory, records in indexes.items():
        bases = [record for record in records if record["repeat"] == 0]
        for left in bases:
            for right in records:
                if right["id"] == left["id"] and right["repeat"] > 0:
                    if left["input_ids"] != right["input_ids"]:
                        raise ValueError("Repeat input changed")
                    compare(directory, left, directory, right, "fixed_input_repeat")
            for right in bases:
                if (
                    len(left["input_ids"]) < len(right["input_ids"])
                    and right["input_ids"][: len(left["input_ids"])] == left["input_ids"]
                ):
                    compare(directory, left, directory, right, "causal_prefix_extension")
    for index, (left_dir, left_records) in enumerate(indexes.items()):
        for right_dir, right_records in list(indexes.items())[index + 1 :]:
            for left in left_records:
                if left["repeat"] != 0:
                    continue
                for right in right_records:
                    if right["repeat"] == 0 and left["id"] == right["id"]:
                        if left["input_ids"] != right["input_ids"]:
                            raise ValueError("Cross-dtype input changed")
                        compare(left_dir, left, right_dir, right, "cross_run")
    return dict(acceptance_gate=False, comparisons=comparisons)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare_trace_directories(args.directories), indent=2) + "\n")
