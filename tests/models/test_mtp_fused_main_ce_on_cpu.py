# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU integration contracts; MCore collectives and Triton are explicitly stubbed.

Load the real forward, packing and MTP patch modules without the model-provider
registry. PyTorch autograd checks the auxiliary-loss graph; these tests do not
claim CUDA-kernel correctness or distributed-collective coverage.
"""

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _torch_linear_ce(hidden, weight, labels, temperature, _reduction, _group):
    logits = hidden.reshape(-1, hidden.shape[-1]) @ weight.T / temperature
    logp = logits.log_softmax(-1)
    return logp.gather(-1, labels.reshape(-1, 1)).squeeze(-1), -(logp.exp() * logp).sum(-1)


@pytest.fixture
def modules(monkeypatch):
    def stub(name, **attrs):
        # All changes are fixture-scoped, including package-parent attributes.
        if name not in installed:
            module = ModuleType(name)
            module.__path__ = []
            module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            monkeypatch.setitem(sys.modules, name, module)
            installed[name] = module
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(stub(parent), attr, module)
        module = installed[name]
        module.__dict__.update(attrs)
        return module

    def load(name, relative_path):
        spec = importlib.util.spec_from_file_location(name, root / relative_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        parent, attr = name.rsplit(".", 1)
        setattr(stub(parent), attr, module)
        installed[name] = module
        spec.loader.exec_module(module)
        return module

    class GPTModel(torch.nn.Module):
        def forward(self, *, output_processor=None, output_processor_context=None, **kwargs):
            return output_processor, output_processor_context

    class AutoScaler(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, loss):
            ctx.save_for_backward(loss)
            return hidden

        @staticmethod
        def backward(ctx, grad):
            return grad, torch.ones_like(ctx.saved_tensors[0])

    def roll(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
        assert cp_group is None, "CPU roll stub is only valid for CP=1"
        result = tensor.roll(shifts, dims)
        result[..., -1] = 0
        return result, result.sum()

    installed = {}
    root = Path(__file__).parents[2] / "verl"
    state = stub(
        "megatron.core.parallel_state",
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: None,
        get_data_parallel_group=lambda **kw: None,
    )
    stub("megatron.core", __version__="0.18.0")
    stub("megatron.core.tensor_parallel")
    stub("megatron.core.tensor_parallel.mappings", gather_from_sequence_parallel_region=lambda h: h)
    stub("megatron.core.models.gpt.gpt_model", GPTModel=GPTModel)
    stub("megatron.core.packed_seq_params", PackedSeqParams=SimpleNamespace)
    stub("megatron.core.inference.contexts", BaseInferenceContext=object)
    stub("megatron.core.config_logger", has_config_logger_enabled=lambda c: False, log_config_to_disk=lambda *a: None)
    stub("megatron.core.utils", unwrap_model=lambda m: m, deprecate_inference_params=lambda a, b: a)
    stub(
        "megatron.core.transformer.multi_token_prediction",
        MTPLossAutoScaler=AutoScaler,
        MTPLossLoggingHelper=SimpleNamespace(save_loss_to_tracker=lambda *a, **kw: None),
        roll_tensor=roll,
    )
    stub("verl.utils.megatron_utils", unwrap_model=lambda m: m)
    stub("verl.utils.device", is_npu_available=False)
    stub("verl.utils.model", CausalLMOutputForPPO=SimpleNamespace)
    stub("verl.utils.kernel.linear_cross_entropy", linear_cross_entropy=_torch_linear_ce)
    stub("verl.workers.config", MtpConfig=SimpleNamespace)
    util = load("verl.models.mcore.util", "models/mcore/util.py")
    monkeypatch.setattr(util.logger, "warning_once", lambda *a, **kw: None, raising=False)
    mf = load("verl.models.mcore.model_forward", "models/mcore/model_forward.py")
    mff = load("verl.models.mcore.model_forward_fused", "models/mcore/model_forward_fused.py")
    mtp = load("verl.models.mcore.mtp_patch", "models/mcore/mtp_patch.py")
    return SimpleNamespace(mff=mff, mf=mf, mtp=mtp, util=util, GPTModel=GPTModel, state=state, scaler=AutoScaler)


class OutputLayer(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.bias = None
        self.calls = 0

    def forward(self, hidden, weight=None, **kwargs):
        self.calls += 1
        return hidden @ (self.weight if weight is None else weight).T, None


def _model(modules, *, k=1, per_token=False):
    model = modules.GPTModel()
    model.config = SimpleNamespace(
        mtp_num_layers=k,
        mtp_loss_scaling_factor=0.2,
        calculate_per_token_loss=per_token,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        use_mup=False,
        fp8=None,
    )
    model.pre_process = True
    model.post_process = True
    model.share_embeddings_and_output_weights = False
    model.output_layer = OutputLayer(torch.arange(28).reshape(7, 4).float() / 28)
    model.compute_language_model_loss = lambda labels, logits: torch.nn.functional.cross_entropy(
        logits.reshape(-1, 7), labels.reshape(-1), reduction="none"
    ).view_as(labels)
    return model


def _postprocess(modules, model, hidden, labels, mask, **kwargs):
    return modules.mtp._megatron_gptmodel_postprocess(
        model,
        hidden_states=hidden,
        input_ids=labels,
        position_ids=None,
        labels=labels,
        loss_mask=mask,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        **kwargs,
    )


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("fp8", [None, "hybrid"])
@pytest.mark.parametrize("padded_mask", [False, True])
def test_fused_and_normal_forward_pack_identical_mtp_inputs(modules, monkeypatch, layout, fp8, padded_mask):
    """Compare THD packing with a group stand-in, not full engine capability checks."""
    group = object()
    monkeypatch.setattr(modules.state, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        modules.state, "get_dynamic_data_context_parallel_groups", lambda group_size: group, raising=False
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(modules.mf, "postprocess_thd_engine", lambda output, *a, **kw: output)
    monkeypatch.setattr(modules.mff, "postprocess_thd_engine", lambda output, *a, **kw: output)
    # The stand-in is not a dataclass; opt in to the contiguous-layout contract.
    modules.util._PACKED_SEQ_PARAMS_HAS_CP_PARTITION_MODE = True
    # Keep the final sequence long enough for existing zigzag FP8 tail padding.
    # Tiny sequences already fail in the non-fused baseline when total padding
    # extends a CP chunk past its source; that independent utility issue is not
    # changed here. These lengths exercise a 256-token FP8 padded tail.
    ids = torch.nested.nested_tensor([torch.arange(256) % 6 + 1, torch.arange(512) % 6 + 1], layout=torch.jagged)
    if padded_mask:
        mask = torch.tensor([[1, 0, 1], [1, 0, 0]])
        response_attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    else:
        mask = torch.nested.nested_tensor([torch.tensor([1, 0, 1]), torch.tensor([1])], layout=torch.jagged)
        response_attention_mask = None
    captured = []
    padding_mask = torch.zeros(1, 512 if fp8 else 384, dtype=torch.bool)
    for fused in [False, True]:
        model = _model(modules)
        model.pre_process = False
        model.config.fp8 = fp8
        setattr(model, modules.mff._FUSED_FORWARD_MODE_ATTR, modules.mff._HOOK_MODE)

        def forward(**kw):
            captured.append(kw)
            n = kw["labels"].numel()
            return SimpleNamespace(log_probs=torch.zeros(n), entropy=torch.ones(n))

        model.forward = forward
        common = dict(
            mtp_enable_train=True,
            local_cp_size=2,
            router_padding_mask=padding_mask,
            mtp_loss_normalization_factor=1.25,
            cp_layout=layout,
        )
        if fused:
            modules.mff.fused_forward_model_engine()(
                model,
                ids,
                ids,
                {},
                1.0,
                False,
                0,
                loss_mask=mask,
                response_attention_mask=response_attention_mask,
                **common,
            )
        else:
            modules.mf.gptmodel_forward_model_engine(
                model,
                ids,
                {},
                logits_processor_args={
                    "label": ids,
                    "loss_mask": mask,
                    "response_attention_mask": response_attention_mask,
                },
                **common,
            )
    for key in ["input_ids", "labels", "loss_mask", "position_ids", "padding_mask"]:
        torch.testing.assert_close(captured[0][key], captured[1][key])
    for call in captured:
        assert call["packed_seq_params"].cp_group is group
        assert call["packed_seq_params"]._verl_mtp_loss_normalization_factor == 1.25
    assert captured[1]["output_processor_context"].labels is captured[1]["labels"]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("tied", [False, True])
def test_main_fusion_preserves_auxiliary_gradients(modules, monkeypatch, native, per_token, k, tied):
    # Legacy auxiliary heads detach their parameters; the native stand-in does
    # not. Both behaviors must be preserved without special-casing the hook.
    mtp = modules.mtp
    monkeypatch.setattr(mtp, "_HAS_PROCESS_MTP_LOSS", native)
    native_calls = []

    def native_process(hidden_states, output_layer, output_weight, labels, **kwargs):
        native_calls.append(kwargs)
        chunks = hidden_states.chunk(k + 1)
        main = chunks[0]
        for hidden in chunks[1:]:
            logits, _ = output_layer(hidden, weight=output_weight)
            main = modules.scaler.apply(main, logits.square().mean() * 0.2 / k)
        return main

    monkeypatch.setattr(mtp, "_process_mtp_loss", native_process, raising=False)
    monkeypatch.setattr(
        mtp,
        "_PROCESS_MTP_LOSS_PARAMS",
        {"hidden_states", "output_layer", "output_weight", "labels", "config", "cp_group", "packed_seq_params"},
    )
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0, 1.0]])
    base_hidden = torch.linspace(-1, 1, (k + 1) * 20).reshape((k + 1) * 5, 1, 4)
    results = []
    for fused in [False, True]:
        model = _model(modules, k=k, per_token=per_token)
        if tied:
            model.share_embeddings_and_output_weights = True
            model.shared_embedding_or_output_weight = lambda model=model: model.output_layer.weight
        hidden = base_hidden.clone().requires_grad_()
        kwargs = {}
        if fused:
            kwargs = dict(
                output_processor=modules.mff.fused_output_processor,
                output_processor_context=modules.mff.FusedOutputProcessorContext(0.7, labels),
            )
        output = _postprocess(modules, model, hidden, labels, mask.clone(), **kwargs)
        if fused:
            lp, entropy = output.log_probs, output.entropy
            assert output.logits is None
        else:
            logp = (output.reshape(-1, 7) / 0.7).log_softmax(-1)
            lp = logp.gather(-1, labels.reshape(-1, 1)).squeeze(-1)
            entropy = -(logp.exp() * logp).sum(-1)
        (-lp.mean() - 0.03 * entropy.mean()).backward()
        results.append((lp.detach(), hidden.grad, model.output_layer.weight.grad))
        assert model.output_layer.calls == k + (not fused)
        assert hidden.grad[5:].abs().sum() > 0  # AutoScaler still reaches auxiliary heads.
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected)
    assert len(native_calls) == (2 if native else 0)


@pytest.mark.parametrize("zero_mask", [False, True])
def test_training_executes_mtp_before_main_hook(modules, zero_mask):
    model = _model(modules)
    model.embedding = object()
    events = []

    class MTP(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            events.append("mtp")
            return torch.cat([hidden_states, hidden_states * 2])

    model.mtp = MTP()
    labels = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(3, 1, 4, requires_grad=True)
    mask = torch.zeros_like(labels) if zero_mask else torch.ones_like(labels)

    def processor(**kw):
        events.append("main")
        assert kw["hidden_states"].shape == hidden.shape
        assert kw["loss_mask"] is mask
        return modules.mff.fused_output_processor(**kw)

    output = _postprocess(
        modules,
        model,
        hidden,
        labels,
        mask,
        mtp_in_postprocess=True,
        output_processor=processor,
        output_processor_context=modules.mff.FusedOutputProcessorContext(1.0, labels),
    )
    (-output.log_probs.mean()).backward()
    assert events == ["mtp", "main"]
    assert model.output_layer.calls == 1  # Auxiliary output only.
    assert torch.isfinite(hidden.grad).all()


def test_load_only_hook_skips_mtp_and_main_output_layer(modules):
    model = _model(modules)
    model.mtp = lambda **kw: pytest.fail("load-only must not execute MTP")
    labels = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(3, 1, 4, requires_grad=True)
    output = _postprocess(
        modules,
        model,
        hidden,
        None,
        None,
        mtp_in_postprocess=True,
        output_processor=modules.mff.fused_output_processor,
        output_processor_context=modules.mff.FusedOutputProcessorContext(1.0, labels),
    )
    (-output.log_probs.mean()).backward()
    assert model.output_layer.calls == 0
    assert hidden.grad is not None and model.output_layer.weight.grad is not None


@pytest.mark.parametrize("tied", [False, True])
def test_mixed_main_grad_accumulation_does_not_drop_main_head_gradient(modules, monkeypatch, tied):
    """Model MCore's fused-wgrad dummy and DDP hook, including two microbatches."""

    class AccumulatingLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight):
            ctx.save_for_backward(hidden, weight)
            return hidden @ weight.T

        @staticmethod
        def backward(ctx, grad):
            hidden, weight = ctx.saved_tensors
            weight.main_grad.add_(grad.reshape(-1, 7).T @ hidden.reshape(-1, 4))
            weight.grad_added_to_main_grad = True
            dummy = (
                torch.zeros_like(weight) if getattr(weight, "zero_out_wgrad", False) else torch.full_like(weight, 99)
            )
            return grad @ weight, dummy

    def process(hidden_states, output_layer, output_weight, **kwargs):
        main, aux = hidden_states.chunk(2)
        logits, _ = output_layer(aux, weight=output_weight)
        return modules.scaler.apply(main, logits.square().mean() * 0.2)

    monkeypatch.setattr(modules.mtp, "_HAS_PROCESS_MTP_LOSS", True)
    monkeypatch.setattr(modules.mtp, "_process_mtp_loss", process, raising=False)
    monkeypatch.setattr(modules.mtp, "_PROCESS_MTP_LOSS_PARAMS", {"hidden_states", "output_layer", "output_weight"})
    labels = torch.tensor([[1, 2, 3]])
    base_hidden = torch.linspace(-1, 1, 24).reshape(6, 1, 4)
    results = []
    for fused in [False, True]:
        model = _model(modules)
        weight = model.output_layer.weight
        weight.main_grad = torch.zeros_like(weight)
        weight.grad_added_to_main_grad = False
        if tied:
            model.share_embeddings_and_output_weights = True
            model.shared_embedding_or_output_weight = lambda weight=weight: weight
            # MCore already sets this flag on shared embedding weights.
            weight.zero_out_wgrad = True

        def ddp_hook(param):
            if not param.grad_added_to_main_grad or getattr(param, "zero_out_wgrad", False):
                param.main_grad.add_(param.grad)
            param.grad = None

        handle = weight.register_post_accumulate_grad_hook(ddp_hook)
        model.output_layer.forward = lambda hidden, weight=None, default_weight=weight, **kw: (
            AccumulatingLinear.apply(hidden, default_weight if weight is None else weight),
            None,
        )
        grads = []
        for microbatch in range(2):
            hidden = (base_hidden + microbatch * 0.1).requires_grad_()
            kwargs = {}
            if fused:
                kwargs = dict(
                    output_processor=modules.mff.fused_output_processor,
                    output_processor_context=modules.mff.FusedOutputProcessorContext(1.0, labels),
                )
            output = _postprocess(modules, model, hidden, labels, torch.ones_like(labels), **kwargs)
            if fused:
                loss = -output.log_probs.mean()
            else:
                loss = torch.nn.functional.cross_entropy(output.reshape(-1, 7), labels.reshape(-1))
            if tied:
                loss = loss + torch.nn.functional.embedding(labels, weight).square().mean() * 0.01
            loss.backward()
            grads.append(hidden.grad)
        results.append((weight.main_grad, *grads))
        handle.remove()
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("train_mtp", [False, True])
@pytest.mark.parametrize("last_stage", [False, True])
def test_engine_labels_mask_positions_and_main_output_contract(modules, train_mtp, last_stage):
    model = _model(modules)
    model.pre_process = False  # PP last rank also needs actual input IDs for MTP.
    model.post_process = last_stage
    setattr(model, modules.mff._FUSED_FORWARD_MODE_ATTR, modules.mff._HOOK_MODE)
    captured = {}
    ids = torch.nested.nested_tensor([torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3, 4])], layout=torch.jagged)
    mask = torch.nested.nested_tensor([torch.tensor([1, 0]), torch.tensor([1])], layout=torch.jagged)

    def forward(**kwargs):
        captured.update(kwargs)
        n = kwargs["output_processor_context"].labels.numel()
        return SimpleNamespace(log_probs=torch.arange(n).float(), entropy=torch.ones(n))

    model.forward = forward
    output = modules.mff.fused_forward_model_engine()(
        model,
        ids,
        ids,
        {},
        0.7,
        True,
        0,
        mtp_enable_train=train_mtp,
        loss_mask=mask,
        mtp_loss_normalization_factor=1.25,
    )
    assert captured["packed_seq_params"]._verl_mtp_loss_normalization_factor == 1.25
    assert (captured["labels"] is not None) == (train_mtp and last_stage)
    assert ("loss_mask" in captured) == (train_mtp and last_stage)
    assert captured["output_processor_context"].labels.numel() == 7
    if train_mtp and last_stage:
        assert captured["position_ids"] is not None
        assert captured["input_ids"].tolist() == [[1, 2, 3, 4, 2, 3, 4]]
        # Each sample is shifted independently, with an invalid tail.
        assert captured["loss_mask"].tolist() == [[0, 1, 0, 0, 0, 1, 0]]
    if last_stage:
        assert set(output) == {"log_probs", "entropy"}
        assert output["log_probs"].numel() == 7


@pytest.mark.parametrize(
    "case", ["supported", "legacy", "vision", "mup", "fp8_output", "deferred", "tp_no_sp", "bias", "tp_sp"]
)
def test_capability_gate(modules, case):
    model = _model(modules)
    if case == "legacy":
        model.forward = lambda input_ids: input_ids
    elif case == "vision":
        model = SimpleNamespace(language_model=model)
    elif case == "mup":
        model.config.use_mup = True
    elif case == "fp8_output":
        model.config.fp8_output = True
    elif case == "deferred":
        model.config.defer_embedding_wgrad_compute = True
    elif case in ("tp_no_sp", "tp_sp"):
        model.config.tensor_model_parallel_size = 2
        model.config.sequence_parallel = case == "tp_sp"
    elif case == "bias":
        model.output_layer.bias = torch.nn.Parameter(torch.zeros(7))
    reason = modules.mff.mtp_fused_forward_unavailable_reason(model)
    assert (reason is None) == (case in ("supported", "tp_sp"))


def test_legacy_forward_cannot_silently_drop_mtp(modules):
    model = _model(modules)
    with pytest.raises(ValueError, match="native output-processor hook"):
        modules.mff.fused_forward_model_engine()(model, None, None, {}, 1.0, False, 0, mtp_enable_train=True)


def test_native_auxiliary_preserves_dynamic_cp_and_scaling(modules, monkeypatch):
    model = _model(modules, per_token=True)
    dynamic_group = object()
    packed = SimpleNamespace(cp_group=dynamic_group, _verl_mtp_loss_normalization_factor=1.25)
    seen = {}

    def process(**kwargs):
        seen.update(kwargs)
        return kwargs["hidden_states"][:3]

    monkeypatch.setattr(modules.mtp, "_HAS_PROCESS_MTP_LOSS", True)
    monkeypatch.setattr(modules.mtp, "_process_mtp_loss", process, raising=False)
    monkeypatch.setattr(
        modules.mtp, "_PROCESS_MTP_LOSS_PARAMS", {"hidden_states", "config", "cp_group", "packed_seq_params"}
    )
    labels = torch.tensor([[1, 2, 3]])
    _postprocess(
        modules,
        model,
        torch.randn(6, 1, 4),
        labels,
        torch.ones_like(labels),
        packed_seq_params=packed,
        output_processor=modules.mff.fused_output_processor,
        output_processor_context=modules.mff.FusedOutputProcessorContext(1.0, labels),
    )
    assert seen["cp_group"] is dynamic_group
    assert seen["packed_seq_params"] is packed
    assert seen["config"].mtp_loss_scaling_factor == pytest.approx(0.25)
    assert model.config.mtp_loss_scaling_factor == 0.2
    assert model.output_layer.calls == 0
