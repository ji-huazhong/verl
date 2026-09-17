# SPDX-License-Identifier: Apache-2.0
"""Layout tests use unequal dimensions to expose transposition mistakes."""

import pytest
import torch

from verl.utils.vllm.lora_layout import expand_stacked_expert_lora


@pytest.mark.parametrize("experts,rank,hidden,intermediate", [(2, 3, 11, 7), (4, 2, 9, 5)])
def test_stacked_factors_preserve_each_expert_delta(experts, rank, hidden, intermediate):
    generator = torch.Generator().manual_seed(71)
    a = torch.randn(experts, rank, hidden, generator=generator)
    b = torch.randn(experts, 2 * intermediate, rank, generator=generator)
    down_a = torch.randn(experts, rank, intermediate, generator=generator)
    down_b = torch.randn(experts, hidden, rank, generator=generator)
    prefix = "arbitrary.layers.5.mlp.experts"
    tensors = {
        f"{prefix}.fused_input.lora_A.weight": a,
        f"{prefix}.fused_input.lora_B.weight": b,
        f"{prefix}.output.lora_A.weight": down_a,
        f"{prefix}.output.lora_B.weight": down_b,
        "attention.q_proj.lora_A.weight": torch.ones(3, 11),
    }
    result = expand_stacked_expert_lora(
        tensors,
        {"fused_input": ["gate", "up"]},
        supported_expert_modules=[f"experts.{e}.{part}" for e in range(experts) for part in ("gate", "up", "output")],
    )
    assert len(result) == experts * 6 + 1
    for expert in range(experts):
        for part, expected in (("gate", b[expert, :intermediate]), ("up", b[expert, intermediate:])):
            out = result[f"{prefix}.{expert}.{part}.lora_B.weight"]
            inp = result[f"{prefix}.{expert}.{part}.lora_A.weight"]
            torch.testing.assert_close(out @ inp, expected @ a[expert], rtol=0, atol=0)
        torch.testing.assert_close(
            result[f"{prefix}.{expert}.output.lora_B.weight"] @ result[f"{prefix}.{expert}.output.lora_A.weight"],
            down_b[expert] @ down_a[expert],
            rtol=0,
            atol=0,
        )
    assert result["attention.q_proj.lora_A.weight"] is tensors["attention.q_proj.lora_A.weight"]


def test_unstacked_export_is_unchanged():
    tensors = {"layers.0.mlp.experts.0.down_proj.lora_A.weight": torch.ones(3, 7)}
    assert expand_stacked_expert_lora(tensors, {}, supported_expert_modules=[]) is tensors


@pytest.mark.parametrize("targets", [[], ["experts.0.input"], ["experts.w1", "experts.w2", "experts.w3"]])
def test_loader_without_matching_per_expert_capability_fails(targets):
    tensors = {
        "model.layers.0.experts.input.lora_A.weight": torch.ones(2, 3, 11),
        "model.layers.0.experts.input.lora_B.weight": torch.ones(2, 7, 3),
    }
    with pytest.raises(ValueError, match="does not advertise the 2D expert target"):
        expand_stacked_expert_lora(tensors, {}, supported_expert_modules=targets)


@pytest.mark.parametrize("case", ["unpaired", "rank", "expert_count", "non_expert", "unequal_slices", "collision"])
def test_ambiguous_layout_fails(case):
    prefix = "model.layers.0.experts.input"
    tensors = {f"{prefix}.lora_A.weight": torch.ones(2, 3, 11), f"{prefix}.lora_B.weight": torch.ones(2, 14, 3)}
    if case == "unpaired":
        tensors.pop(f"{prefix}.lora_B.weight")
    elif case == "rank":
        tensors[f"{prefix}.lora_B.weight"] = torch.ones(2, 14, 4)
    elif case == "expert_count":
        tensors[f"{prefix}.lora_B.weight"] = torch.ones(3, 14, 3)
    elif case == "non_expert":
        tensors = {"attention.q_proj.lora_A.weight": torch.ones(2, 3, 11)}
    elif case == "unequal_slices":
        tensors[f"{prefix}.lora_B.weight"] = torch.ones(2, 13, 3)
    elif case == "collision":
        tensors["model.layers.0.experts.0.gate.lora_A.weight"] = torch.ones(3, 11)
    with pytest.raises(ValueError):
        expand_stacked_expert_lora(tensors, {"input": ["gate", "up"]})
