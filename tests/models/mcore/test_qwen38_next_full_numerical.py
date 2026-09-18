# SPDX-License-Identifier: Apache-2.0
"""Opt-in full48 hybrid LoRA, frozen-weight and recompute numerical probe.

CUDA_DEVICE_MAX_CONNECTIONS=1 RUN_QWEN38_FULL_NUMERICAL=1
QWEN38_MODEL_PATH=<original full checkpoint>
QWEN38_FULL_NUMERICAL_OUTPUT=<fresh private directory> torchrun --standalone
--nproc-per-node=8 -m pytest -s -q <this file>

This generates fixed text/image references for a separate vLLM process. Its
synthetic objective is a numerical probe, not GRPO or a learning benchmark.
The actual production trainer/save/resume gate is separate.
"""

import faulthandler
import hashlib
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from tests.models.mcore.qwen38_full_validation import (
    check_recompute_gradient,
    full_checkpoint_config,
    make_full_cases,
    tensor_sha256,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_FULL_NUMERICAL") != "1", reason="explicit full48 eight-GPU opt-in required"
)


@pytest.fixture(scope="module", autouse=True)
def full_context():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tensor_random
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    full_checkpoint_config(os.environ["QWEN38_MODEL_PATH"])
    assert int(os.environ.get("WORLD_SIZE", "0")) == 8
    output = Path(os.environ["QWEN38_FULL_NUMERICAL_OUTPUT"])
    assert not output.exists(), "Never overwrite previous numerical evidence"
    assert output.is_absolute() and output.parent.is_dir()
    assert shutil.disk_usage(output.parent).free >= 25 * 1024**3, (
        "Reserve space for private adapter/reference artifacts"
    )
    available_kib = int(
        next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
    )
    assert available_kib >= 1024**3, "Require 1 TiB available host memory"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    assert torch.cuda.mem_get_info()[0] >= 100 * 1024**3, "Need idle GPUs; never evict another job"
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=20), device_id=device)
    old_backward = tensor_random.CheckpointFunction.backward
    # Private stderr only. Preserve a Python stack if a stage stops producing
    # markers; this does not require ptrace or suppress NCCL's timeout handling.
    faulthandler.dump_traceback_later(300, repeat=True)
    try:
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
        faulthandler.cancel_dump_traceback_later()
        tensor_random.CheckpointFunction.backward = staticmethod(old_backward)
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_actual_full_model_lora_recompute_frozen_base_and_export():
    from megatron.bridge.peft.lora import LoRA
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    from megatron.core.transformer.module import Float16Module
    from safetensors.torch import save_file

    from verl.models.mcore.bridge import AutoBridge
    from verl.models.mcore.model_forward import gptmodel_forward_model_engine
    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.ops.ple import Qwen38NextFrozenNGramEmbedding, current_ple_batch
    from verl.models.mcore.util import preprocess_thd_engine
    from verl.workers.engine.megatron.transformer_impl import _iter_detached_export_weights

    source = Path(os.environ["QWEN38_MODEL_PATH"])
    output = Path(os.environ["QWEN38_FULL_NUMERICAL_OUTPUT"])
    rank = torch.distributed.get_rank()
    pp = parallel_state.get_pipeline_model_parallel_rank()
    tp = parallel_state.get_tensor_model_parallel_rank()
    cp = parallel_state.get_context_parallel_rank()
    bridge = AutoBridge.from_hf_pretrained(source, local_files_only=True)
    assert isinstance(bridge._model_bridge, Qwen38NextBridge)
    config = bridge.to_megatron_provider(load_weights=False)
    config.apply_overrides_and_finalize(
        dtype=torch.bfloat16,
        overrides=dict(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            context_parallel_size=2,
            sequence_parallel=True,
            variable_seq_lengths=True,
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_token_dispatcher_type="alltoall",
            gradient_accumulation_fusion=True,
            calculate_per_token_loss=True,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        ),
    )
    assert config.num_layers == 48 and config.hidden_size == 2560
    config._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    models = [
        config.provide(pre_process=pp == 0 and vp == 0, post_process=pp == 1 and vp == 1, vp_stage=vp).cuda()
        for vp in range(2)
    ]
    for vp, model in enumerate(models):
        model.model_type = ModelType.encoder_or_decoder
        assert [layer.layer_number for layer in model.language_model.decoder.layers] == list(
            range(12 * (pp + 2 * vp) + 1, 12 * (pp + 2 * vp) + 13)
        )
    bridge.load_hf_weights(models)
    wrapped = [Float16Module(config, model).eval() for model in models]
    cases = make_full_cases(source)
    batches = [torch.nested.nested_tensor([case["input_ids"].cuda()], layout=torch.jagged) for case in cases]
    multimodal = [{key: value.cuda() for key, value in case["multimodal"].items()} for case in cases]
    counts = [
        preprocess_thd_engine(
            torch.nested.nested_tensor([torch.ones_like(case["input_ids"], device="cuda")], layout=torch.jagged)
        )[0].sum()
        for case in cases
    ]
    observed, errors = {}, []

    def stage(label):
        print(f"FULL48_STAGE rank={rank} phase={label}", flush=True)

    def agree(label):
        valid = torch.tensor(int(not errors), device="cuda")
        torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
        assert valid.item(), (label, errors or "Another rank failed this check")

    def forward_step(iterator, module):
        index = next(iterator)
        value = gptmodel_forward_model_engine(
            module, batches[index], multi_modal_inputs=multimodal[index], vision_model=True, pad_token_id=0
        )
        if value.is_nested:
            value = gather_from_tensor_model_parallel_region(value.values(), group=config._pg_collection.tp)

        def collect(tensor, non_loss_data=False):
            if non_loss_data:
                return tensor.detach().float()
            # A blocking D2H copy here can wait on outstanding P2P work before
            # this loss returns and enables the peer's backward send. Keep the
            # small diagnostic snapshots on GPU until the schedule has ended.
            observed[index] = tensor.detach().float()
            loss = tensor.float().square().mean() / len(cases)
            return loss, counts[index], {"loss": loss.detach()}

        return value, collect

    def schedule(modules, forward_only):
        return get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=[iter(range(len(cases))) for _ in modules],
            model=modules,
            num_microbatches=len(cases),
            seq_length=128,
            micro_batch_size=1,
            forward_only=forward_only,
            collect_non_loss_data=forward_only,
        )

    with torch.no_grad():
        base = [value.cpu() for value in schedule(wrapped, True)]
    stage("base_forward_done")
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
    trainable = {
        f"chunk{vp}.{name}": p
        for vp, model in enumerate(models)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    assert trainable and all("adapter" in name for name in trainable)
    with torch.no_grad():
        initial = [value.cpu() for value in schedule(wrapped, True)]
    if pp == 1:
        assert len(base) == len(initial) == len(cases)
        for expected, actual in zip(base, initial, strict=True):
            if not torch.equal(expected, actual):
                errors.append("zero adapter changes base logits")
        if torch.equal(base[2], base[3]):
            errors.append("full language model ignores changed image pixels")
    agree("initial LoRA and native image")
    stage("zero_lora_and_image_done")

    def frozen_hashes():
        hashes = {}
        for vp, model in enumerate(models):
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    hashes[f"chunk{vp}.{name}"] = tensor_sha256(p)
            for name, module in model.named_modules():
                if isinstance(module, Qwen38NextFrozenNGramEmbedding):
                    assert module._loaded
                    for field in ("table", "layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
                        hashes[f"chunk{vp}.{name}.{field}"] = tensor_sha256(getattr(module, field))
        return hashes

    frozen_before = frozen_hashes()
    print(f"FULL48_FROZEN_HASHED rank={rank} tensors={len(frozen_before)}", flush=True)
    ddp = [
        DistributedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
            module=model,
            pg_collection=config._pg_collection,
        )
        for model in models
    ]
    stage("ddp_ready")
    replay_calls = [0]
    for model in models:
        decoder = model.language_model.decoder
        original = decoder._checkpointed_forward

        def replay(*args, _original=original, **kwargs):
            replay_calls[0] += 1
            return _original(*args, **kwargs)

        decoder._checkpointed_forward = replay
    normal_grads, normal_outputs = None, None
    for recompute in (False, True):
        for model in models:
            dc = model.language_model.decoder.config
            dc.recompute_granularity = "full" if recompute else None
            dc.recompute_method = "uniform" if recompute else None
            dc.recompute_num_layers = 1 if recompute else None
        observed.clear()
        replay_calls[0] = 0
        for module in ddp:
            module.train()
            module.zero_grad_buffer()
        stage(f"forward_backward_start_recompute_{recompute}")
        schedule(ddp, False)
        stage(f"forward_backward_done_recompute_{recompute}")
        finalize_model_grads(ddp, pg_collection=config._pg_collection)
        stage(f"finalize_grads_done_recompute_{recompute}")
        observed = {index: value.cpu() for index, value in observed.items()}
        assert bool(replay_calls[0]) == recompute, "Recompute toggle never reached the decoder"
        with pytest.raises(RuntimeError, match="no n-gram ids published"):
            current_ple_batch()
        assert all(not getattr(m, "_ple_recompute_fifo", []) for model in models for m in model.modules())
        grads = {name: p.main_grad.detach().cpu().clone() for name, p in trainable.items()}
        relative_errors = []
        for name, value in grads.items():
            if not bool(value.isfinite().all()):
                errors.append(f"nonfinite gradient: {name}")
            if normal_grads is not None:
                try:
                    relative_errors.append(check_recompute_gradient(value, normal_grads[name]))
                except AssertionError as error:
                    errors.append(f"recompute gradient {name}: {error}")
        for family in (".self_attention.", ".mlp.experts.", ".mlp.shared_experts."):
            if not any(family in name and bool((value != 0).any()) for name, value in grads.items()):
                errors.append(f"missing nonzero gradient: {family}")
        if normal_outputs is not None:
            for index, actual in observed.items():
                if not torch.equal(actual, normal_outputs[index]):
                    errors.append(f"recompute changed logits case={index}")
        agree(f"gradients recompute={recompute}")
        normal_grads, normal_outputs = grads, dict(observed)
        print(
            f"FULL48_GRADS rank={rank} recompute={recompute} tensors={len(grads)} "
            f"checkpoint_calls={replay_calls[0]} max_relative_l2={max(relative_errors, default=0.0):.9g}",
            flush=True,
        )
    del normal_grads, normal_outputs, grads
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-3, foreach=False)
    before_adapter = {name: tensor_sha256(p) for name, p in trainable.items()}
    for p in trainable.values():
        p.grad = p.main_grad.to(p.dtype)
    optimizer.step()
    assert any(before_adapter[name] != tensor_sha256(p) for name, p in trainable.items())
    del optimizer
    for module in wrapped:
        module.eval()
    with torch.no_grad():
        tuned = [value.cpu() for value in schedule(wrapped, True)]
        with peft.disable_adapter(models):
            disabled = [value.cpu() for value in schedule(wrapped, True)]
    if pp == 1:
        for index, (actual, original) in enumerate(zip(tuned, disabled, strict=True)):
            if torch.equal(actual, base[index]):
                errors.append(f"updated adapter has no output effect case={index}")
            if not torch.equal(original, base[index]):
                errors.append(f"disable differs from base case={index}")
    if frozen_hashes() != frozen_before:
        errors.append("frozen parameters or full PLE table/hash metadata changed")
    agree("updated adapter and frozen base")
    with torch.no_grad():
        tensors = dict(_iter_detached_export_weights(bridge.export_adapter_weights(ddp, cpu=True, show_progress=False)))
    assert len(tensors) == 936 and all(bool(x.isfinite().all()) for x in tensors.values())
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        (output / "adapter").mkdir()
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
            str(output / "raw_adapter.safetensors"),
        )
        manifest = {
            "layers": 48,
            "vocabulary": 248320,
            "lora_rank": 16,
            "lora_alpha": 32,
            "synthetic_probe_lr": 1e-3,
            "cases": [case["name"] for case in cases],
            "config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
            "index_sha256": hashlib.sha256((source / "model.safetensors.index.json").read_bytes()).hexdigest(),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    torch.distributed.barrier()
    (output / f"frozen-rank{rank}.json").write_text(json.dumps(frozen_before, sort_keys=True))
    if pp == 1 and tp == cp == 0:
        reference = {}
        for index, case in enumerate(cases):
            assert base[index].shape == tuned[index].shape == (len(case["input_ids"]), 248320)
            prefix = case["name"]
            reference[f"{prefix}.input_ids"] = case["input_ids"]
            reference[f"{prefix}.raw_input_ids"] = case["raw_input_ids"]
            reference[f"{prefix}.base"] = base[index][:-1].log_softmax(-1).contiguous()
            reference[f"{prefix}.adapter"] = tuned[index][:-1].log_softmax(-1).contiguous()
            if case["rgb"] is not None:
                reference[f"{prefix}.rgb"] = case["rgb"]
        save_file(reference, str(output / "reference.safetensors"))
    torch.distributed.barrier()
    print(f"QWEN38_FULL48_MCORE_NUMERICAL_PASS rank={rank}; vLLM parity is still separate", flush=True)
