# SPDX-License-Identifier: Apache-2.0
"""CPU negative gates for production base-weight planning, not numerical parity."""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def weight_plan(monkeypatch):
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4, dtype=torch.float32)))
    model.register_buffer("persistent", torch.ones(2))
    model.register_buffer("transient", torch.zeros(2), persistent=False)
    model.layer = torch.nn.Module()
    model.layer.adapter = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(qwen3_8_next_ple_layer_ids=[1], qwen3_8_next_split_ngram_parts=2)
    prefix = "model.language_model.layers.1.ple.ple_embedding."
    ple = {
        prefix + name
        for name in (
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
            "ngram_embedding.shard_0.weight",
            "ngram_embedding.shard_1.weight",
        )
    }
    keys = {"source.weight", "source.buffer", "source.remote", "mtp.disabled.weight"} | ple

    class HeaderOnlyState:
        source = SimpleNamespace(get_all_keys=lambda: sorted(keys))

        def __getitem__(self, name):
            pytest.fail("Invalid weight plan reached a tensor payload read")

    hf = SimpleNamespace(state=HeaderOnlyState(), model_name_or_path="fixture")
    plan = [
        SimpleNamespace(
            vp_stage=0,
            param_name="weight",
            param_weight=model.weight,
            mapping=SimpleNamespace(hf_param="source.weight"),
        ),
        SimpleNamespace(
            vp_stage=0,
            param_name="persistent",
            param_weight=model.persistent,
            mapping=SimpleNamespace(hf_param="source.buffer"),
        ),
        SimpleNamespace(
            vp_stage=None,
            param_name="remote.weight",
            param_weight=None,
            mapping=SimpleNamespace(hf_param="source.remote"),
        ),
    ]
    monkeypatch.setattr(MegatronModelBridge, "build_conversion_tasks", lambda *args: plan)
    return SimpleNamespace(bridge=Qwen38NextBridge(), model=model, hf=hf, tasks=plan, keys=keys, ple=ple)


def test_complete_plan_keeps_remote_pp_tasks_and_excludes_only_explicit_state(weight_plan):
    case = weight_plan
    assert case.bridge.build_conversion_tasks(case.hf, [case.model]) is case.tasks
    # No source catalog exists when exporting an initialized/random model.
    assert case.bridge.build_conversion_tasks(SimpleNamespace(), [case.model]) is case.tasks
    case.keys.add("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_2.weight")
    with pytest.raises(ValueError, match="unused sources"):
        case.bridge.build_conversion_tasks(case.hf, [case.model])


@pytest.mark.parametrize("prebuilt", [False, True])
def test_stream_import_cannot_use_config_only_export_exception(weight_plan, prebuilt):
    case = weight_plan
    with pytest.raises(ValueError, match="requires a source catalog"):
        list(
            case.bridge.stream_weights_hf_to_megatron(SimpleNamespace(), [case.model], case.tasks if prebuilt else None)
        )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_target",
        "none_task",
        "duplicate_target",
        "wrong_owner",
        "unknown_source",
        "missing_weight",
        "missing_fused_input",
        "missing_ple_shard",
        "missing_ple_hash",
        "false_remote",
    ],
)
@pytest.mark.parametrize("entry", ["load", "explicit_stream"])
def test_invalid_plan_fails_before_reading_or_mutating_weights(weight_plan, fault, entry):
    case = weight_plan
    before = {name: value.clone() for name, value in case.model.state_dict().items()}
    if fault == "missing_target":
        case.tasks.pop(0)
    elif fault == "none_task":
        case.tasks[0] = None
    elif fault == "duplicate_target":
        case.tasks.append(case.tasks[0])
    elif fault == "wrong_owner":
        case.tasks[0].param_weight = case.model.weight.clone()
    elif fault == "unknown_source":
        case.keys.add("unrecognized.weight")
    elif fault == "missing_weight":
        case.keys.remove("source.weight")
    elif fault == "missing_fused_input":
        case.tasks[0].mapping.hf_param = {"gate": "source.weight", "up": "absent.up"}
    elif fault == "missing_ple_shard":
        case.keys.remove(next(name for name in case.ple if "shard_1" in name))
    elif fault == "missing_ple_hash":
        case.keys.remove(next(name for name in case.ple if name.endswith("layer_multipliers")))
    elif fault == "false_remote":
        case.tasks[0].vp_stage = None
        case.tasks[0].param_weight = None
    with pytest.raises(ValueError, match="Flash-Next base weight"):
        if entry == "load":
            case.bridge.load_weights_hf_to_megatron(case.hf, [case.model])
        else:
            list(case.bridge.stream_weights_hf_to_megatron(case.hf, [case.model], case.tasks))
    for name, value in case.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_interleaved_chunks_have_independent_local_owners(weight_plan):
    case = weight_plan
    second = torch.nn.Module()
    second.register_parameter("weight", torch.nn.Parameter(torch.ones(4)))
    case.tasks.append(
        SimpleNamespace(
            vp_stage=1,
            param_name="weight",
            param_weight=second.weight,
            mapping=SimpleNamespace(hf_param="source.remote"),
        )
    )
    assert case.bridge.build_conversion_tasks(case.hf, [case.model, second]) is case.tasks
    case.tasks[-1].vp_stage = 0
    with pytest.raises(ValueError, match="duplicates local target"):
        case.bridge.build_conversion_tasks(case.hf, [case.model, second])


def test_tied_output_uses_native_shared_embedding_path(weight_plan):
    case = weight_plan
    case.model.config.share_embeddings_and_output_weights = True
    case.model.output_layer = torch.nn.Linear(4, 4, bias=False)
    assert case.bridge.build_conversion_tasks(case.hf, [case.model]) is case.tasks
    case.model.config.share_embeddings_and_output_weights = False
    with pytest.raises(ValueError, match="misses local targets"):
        case.bridge.build_conversion_tasks(case.hf, [case.model])
