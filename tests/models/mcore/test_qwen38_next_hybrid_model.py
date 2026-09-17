# SPDX-License-Identifier: Apache-2.0
"""Eight-worker TP2/PP2/EP2/CP2/VPP2 numerical gate using Core's real schedule.

Requires RUN_QWEN38_HYBRID_MODEL_TESTS=1, QWEN38_TINY_EXPORT_DIR (original
four-layer fixture), QWEN38_HYBRID_REFERENCE (the independent TP2/EP2/CP2,
PP1 test_qwen38_next_model_cp output), and a fresh QWEN38_HYBRID_OUTPUT.
No production guard or communication operation is bypassed. The globally
normalized synthetic objective isolates pipeline parity; production GRPO's
token-count contract, optimizer resume and HTTP reload are separate gates.
QWEN38_MODEL_IMAGES=1 requires an image-enabled independent reference and
exercises mixed image/text documents without replacing the native vision tower.
"""

import os
import re
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_HYBRID_MODEL_TESTS") != "1", reason="explicit eight-GPU numerical opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def hybrid_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tensor_random
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    assert int(os.environ.get("WORLD_SIZE", "0")) == 8
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.025, 3 * 1024**3 / total))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
    original_backward = tensor_random.CheckpointFunction.backward
    try:
        enough = torch.tensor(int(free >= 8 * 1024**3), device=device)
        torch.distributed.all_reduce(enough, op=torch.distributed.ReduceOp.MIN)
        if not enough.item():
            pytest.skip("Every GPU needs 8 GiB free; never evict another job")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            context_parallel_size=2,
        )
        model_parallel_cuda_manual_seed(123)
        apply_patch_megatron_recomputation_backward()
        yield
    finally:
        tensor_random.CheckpointFunction.backward = staticmethod(original_backward)
        torch.cuda.synchronize()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_interleaved_hybrid_matches_independent_pp1():
    try:
        _run_hybrid_case()
    except Exception:
        traceback.print_exc()
        raise


def _run_hybrid_case():
    from megatron.bridge.peft.lora import LoRA
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    from megatron.core.transformer.module import Float16Module
    from safetensors.torch import load_file, save_file

    from tests.models.mcore.qwen38_vision_fixture import make_packed_fixture_batches
    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.model_forward import gptmodel_forward_model_engine
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.ops.ple import current_ple_batch
    from verl.models.mcore.util import preprocess_thd_engine

    fixture = Path(os.environ["QWEN38_TINY_EXPORT_DIR"])
    output_dir = Path(os.environ["QWEN38_HYBRID_OUTPUT"])
    assert not output_dir.exists(), "Never overwrite previous evidence"
    rank = torch.distributed.get_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    reference = load_file(str(Path(os.environ["QWEN38_HYBRID_REFERENCE"]) / f"comparison-tp{tp_rank}.safetensors"))
    images = os.environ.get("QWEN38_MODEL_IMAGES") == "1"
    assert bool(reference.get("fixture.images", torch.tensor(False))) == images, "Reference modality mismatch"
    bridge = AutoBridge.from_hf_pretrained(fixture / "model", local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    config = bridge.to_megatron_provider(load_weights=False)
    assert config.num_layers == 4 and config.hidden_size == 128
    config.tensor_model_parallel_size = 2
    config.pipeline_model_parallel_size = 2
    config.virtual_pipeline_model_parallel_size = 2
    config.expert_model_parallel_size = 2
    config.expert_tensor_parallel_size = 1
    config.context_parallel_size = 2
    config.sequence_parallel = True
    config.variable_seq_lengths = True
    config.overlap_p2p_comm = True
    config.batch_p2p_comm = False
    config.moe_router_load_balancing_type = "none"
    config.moe_token_dispatcher_type = "alltoall"
    config.moe_aux_loss_coeff = 0.0
    config.moe_permute_fusion = False
    config.params_dtype = torch.bfloat16
    config.bf16 = True
    config.gradient_accumulation_fusion = True
    config.calculate_per_token_loss = True
    config.recompute_granularity = "full"
    config.recompute_method = "uniform"
    config.recompute_num_layers = 1
    # Explicit gradient finalization below: the objective is already globally
    # normalized, so this numerical gate must not divide again by token count.
    config.finalize_model_grads_func = None
    config.finalize()
    config._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    models = [
        config.provide(
            pre_process=pp_rank == 0 and chunk == 0, post_process=pp_rank == 1 and chunk == 1, vp_stage=chunk
        ).cuda()
        for chunk in range(2)
    ]
    for chunk, model in enumerate(models):
        model.model_type = ModelType.encoder_or_decoder
        assert model.vp_stage == chunk
        assert [layer.layer_number for layer in model.language_model.decoder.layers] == [pp_rank + 2 * chunk + 1]
    bridge.load_hf_weights(models)
    wrapped = [Float16Module(config, model).eval() for model in models]

    docs_by_batch, mm_by_batch = make_packed_fixture_batches(config.qwen3_8_next_eos_token_id, images=images)
    docs_by_batch.append([torch.arange(3, 19, device="cuda")])
    mm_by_batch.append({})
    batches, local_token_counts = [], []
    for docs in docs_by_batch:
        batches.append(torch.nested.nested_tensor(docs, layout=torch.jagged))
        mask = torch.nested.nested_tensor([torch.ones_like(ids) for ids in docs], layout=torch.jagged)
        local_token_counts.append(preprocess_thd_engine(mask)[0].sum())
    torch.cuda.synchronize()

    observed = []
    errors, recorded = [], {"fixture.images": torch.tensor(images)}

    def checkpoint(label):
        valid = torch.tensor(int(not errors), device="cuda")
        torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
        assert valid.item(), f"{label}: {errors or 'another rank failed; inspect its assertion'}"

    def compare(name, value, *, gradient=False, exact=False):
        actual = value.detach().float().cpu().contiguous()
        recorded[name] = actual
        try:
            assert bool(actual.isfinite().all()), name
            expected = reference[name]
            if exact:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            elif gradient:
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=2e-3)
                relative = (actual - expected).norm() / expected.norm().clamp_min(1e-10)
                assert relative < 0.03, (name, relative.item())
                return relative.item(), (actual - expected).abs().max().item()
            else:
                gap = (actual.log_softmax(-1) - expected.log_softmax(-1)).abs()
                print(f"QWEN38_HYBRID RANK={rank} NAME={name} GAP_MEAN={gap.mean():.9f} GAP_MAX={gap.max():.9f}")
                assert gap.mean() < 0.005 and gap.max() < 0.05, name
        except (AssertionError, KeyError) as error:
            errors.append(f"{name}: {error}")
        return 0.0, 0.0

    def forward_step(iterator, module):
        index = next(iterator)
        output = gptmodel_forward_model_engine(
            module, batches[index], multi_modal_inputs=mm_by_batch[index], vision_model=True, pad_token_id=0
        )
        if output.is_nested:
            output = gather_from_tensor_model_parallel_region(output.values(), group=config._pg_collection.tp)

        def collect(value, non_loss_data=False):
            if non_loss_data:
                return value.detach().float()
            observed.append((index, value.detach().float()))
            # Three-return per-token Core loss contract avoids the legacy
            # callback's additional CP multiplier and microbatch division.
            loss = value.float().square().mean() / 4
            return loss, local_token_counts[index], {"loss": loss.detach()}

        return output, collect

    def schedule(modules, *, forward_only, indices=(0, 1, 0, 1)):
        return get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=[iter(indices) for _ in modules],
            model=modules,
            num_microbatches=len(indices),
            seq_length=64,
            micro_batch_size=1,
            forward_only=forward_only,
            collect_non_loss_data=forward_only,
        )

    with torch.no_grad():
        base_outputs = schedule(wrapped, forward_only=True)
    if pp_rank == 1:
        for index, value in zip((0, 1, 0, 1), base_outputs, strict=True):
            compare(f"base.{index}", value)
    checkpoint("base logits")

    peft = LoRA(
        dim=16,
        alpha=32,
        dropout=0.0,
        target_modules=[
            "language_model.decoder.layers.*.self_attention.in_proj",
            "language_model.decoder.layers.*.self_attention.out_proj",
            "language_model.decoder.layers.*.self_attention.linear_qkv",
            "language_model.decoder.layers.*.self_attention.linear_proj",
            "language_model.decoder.layers.*.mlp.*.linear_fc1",
            "language_model.decoder.layers.*.mlp.*.linear_fc2",
        ],
    )
    models = peft(models)
    peft.set_params_to_save(models)
    trainable, frozen = {}, {}
    for chunk, model in enumerate(models):
        for name, parameter in model.named_parameters():
            canonical = re.sub(r"(?<=decoder.layers.)0(?=\.)", str(pp_rank + 2 * chunk), name)
            if parameter.requires_grad:
                assert "adapter" in name and canonical not in trainable
                trainable[canonical] = parameter
            else:
                frozen[chunk, name] = parameter.detach().cpu().clone()
    keys = [None, None]
    torch.distributed.all_gather_object(keys, sorted(trainable), group=config._pg_collection.pp)
    combined = [name for subset in keys for name in subset]
    assert len(combined) == len(set(combined)) == 48
    assert set(combined) == {key.removeprefix("initial.") for key in reference if key.startswith("initial.")}
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(reference[f"initial.{name}"])
            compare(f"initial.{name}", parameter, exact=True)
    checkpoint("adapter initialization")
    ddp = [
        DistributedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
            module=model,
            pg_collection=config._pg_collection,
        )
        for model in models
    ]
    for module in ddp:
        module.train()
    normal_grads = None
    for recompute in (False, True):
        config.recompute_granularity = "full" if recompute else None
        observed.clear()
        for module in ddp:
            module.zero_grad_buffer()
        schedule(ddp, forward_only=False)
        finalize_model_grads(ddp, pg_collection=config._pg_collection)
        for index, output in observed:
            compare(f"adapter.{int(recompute)}.{index}", output)
        with pytest.raises(RuntimeError, match="no n-gram ids published"):
            current_ple_batch()
        assert all(not getattr(module, "_ple_recompute_fifo", []) for model in models for module in model.modules())
        grads = {name: parameter.main_grad.detach().clone() for name, parameter in trainable.items()}
        for family in (".self_attention.", ".mlp.experts.", ".mlp.shared_experts."):
            assert any(family in name and bool((value != 0).any()) for name, value in grads.items()), family
        deviations = []
        for name, value in grads.items():
            deviations.append(compare(f"grad.{int(recompute)}.{name}", value, gradient=True))
            if normal_grads is not None:
                try:
                    torch.testing.assert_close(value, normal_grads[name], rtol=0.02, atol=2e-5)
                except AssertionError as error:
                    errors.append(f"recompute {name}: {error}")
        checkpoint(f"gradients recompute={recompute}")
        normal_grads = grads
        print(
            f"QWEN38_HYBRID_GRADS RANK={rank} RECOMPUTE={recompute} TENSORS={len(grads)} "
            f"MAX_REL_L2={max(item[0] for item in deviations):.9g} MAX_ABS={max(item[1] for item in deviations):.9g}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-2)
    for parameter in trainable.values():
        parameter.grad = parameter.main_grad.to(parameter.dtype)
    optimizer.step()
    for module in wrapped:
        module.eval()
    with torch.no_grad():
        updated = schedule(wrapped, forward_only=True)
        with peft.disable_adapter(models):
            disabled = schedule(wrapped, forward_only=True)
    if pp_rank == 1:
        for index, value, base in zip((0, 1, 0, 1), updated, disabled, strict=True):
            compare(f"updated.{index}", value)
            assert not torch.equal(value.cpu(), recorded[f"adapter.1.{index}"])
            torch.testing.assert_close(base.cpu(), recorded[f"base.{index}"], rtol=0, atol=0)
    for chunk, model in enumerate(models):
        for name, parameter in model.named_parameters():
            if (chunk, name) in frozen:
                torch.testing.assert_close(parameter.cpu(), frozen[chunk, name], rtol=0, atol=0)
    checkpoint("optimizer and frozen base")
    adapters = list(bridge._model_bridge.stream_adapter_weights_megatron_to_hf(ddp, cpu=True, show_progress=False))
    tensors = {item.param_name: item.weight.detach().cpu().contiguous().clone() for item in adapters}
    assert len(tensors) == len(adapters) == 78 and all(bool(value.isfinite().all()) for value in tensors.values())
    with torch.no_grad():
        tuned_logits = schedule(wrapped, forward_only=True, indices=(2, 2, 2, 2))
        with peft.disable_adapter(models):
            base_logits = schedule(wrapped, forward_only=True, indices=(2, 2, 2, 2))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "model").mkdir()
        (output_dir / "adapter").mkdir()
        bridge.hf_pretrained.config.save_pretrained(output_dir / "model")
        save_file(load_file(str(fixture / "model/model.safetensors")), str(output_dir / "model/model.safetensors"))
        save_file(tensors, str(output_dir / "raw_adapter.safetensors"))
    torch.distributed.barrier()
    if cp_rank == 0:
        save_file(recorded, str(output_dir / f"comparison-tp{tp_rank}-pp{pp_rank}.safetensors"))
        if pp_rank == 1 and tp_rank == 0:
            save_file(
                {
                    "input_ids": torch.arange(3, 19)[None],
                    "base_logits": base_logits[0][None].cpu(),
                    "adapter_logits": tuned_logits[0][None].cpu(),
                    "lora_rank": torch.tensor(16),
                    "lora_alpha": torch.tensor(32),
                },
                str(output_dir / "reference.safetensors"),
            )
    torch.distributed.barrier()
    print(f"QWEN38_HYBRID_MODEL_PASSED RANK={rank} TP2 PP2 EP2 CP2 VPP2 IMAGES={images} TENSORS=78", flush=True)
