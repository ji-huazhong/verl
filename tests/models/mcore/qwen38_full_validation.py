# SPDX-License-Identifier: Apache-2.0
"""Full-checkpoint numerical probe assets; never substitutes a reduced fixture.

Artifacts contain full per-token logprobs and must remain private. The hash
routine bounds transient copies, including for noncontiguous tensors, rather
than cloning a complete frozen model or its host-resident PLE table.
"""

import hashlib
import json
from pathlib import Path

import torch


def probe_inference(modules, forward):
    """PEFT installation resets train mode; no_grad alone does not disable Core checkpoints."""
    for module in modules:
        module.eval()
    with torch.no_grad():
        return forward()


def agree_probe_checks(errors, label, group):
    """Keep diagnostic control traffic off the model's CUDA/NCCL streams."""
    assert group is not None and torch.distributed.get_backend(group) == "gloo"
    valid = torch.tensor(int(not errors), dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN, group=group)
    assert valid.item(), (label, errors or "Another rank failed this check")


def tensor_sha256(tensor, *, chunk_bytes=64 * 1024**2):
    if tensor.is_meta or tensor.layout != torch.strided:
        raise ValueError("Hash requires materialized strided tensors")
    if chunk_bytes < tensor.element_size():
        raise ValueError("Hash chunk is smaller than one element")
    limit = chunk_bytes // tensor.element_size()
    digest = hashlib.sha256()

    def consume(value):
        if value.numel() <= limit:
            raw = value.detach().contiguous().reshape(-1).cpu().view(torch.uint8).numpy()
            digest.update(memoryview(raw))
        elif value.is_contiguous():
            flat = value.detach().view(-1)
            for start in range(0, flat.numel(), limit):
                consume(flat[start : start + limit])
        else:
            # Iterating slices avoids reshape's potentially full-sized copy.
            for index in range(value.shape[0]):
                consume(value[index])

    consume(tensor)
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": digest.hexdigest()}


def full_checkpoint_config(checkpoint):
    root = Path(checkpoint)
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", {})
    if config.get("model_type") != "qwen4_exp" or text.get("num_hidden_layers") != 48:
        raise ValueError("Full numerical gate requires the actual 48-layer Flash-Next checkpoint")
    if text.get("hidden_size") != 2560 or text.get("vocab_size") != 248320:
        raise ValueError("Do not replace full width/vocabulary with a small fixture")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if not index.get("weight_map") or not all((root / name).is_file() for name in set(index["weight_map"].values())):
        raise ValueError("Full checkpoint index or referenced shard files are missing")
    return config


def check_recompute_gradient(actual, expected):
    """An absolute tolerance alone can incorrectly accept erased small gradients."""
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-5)
    norm = expected.double().norm().item()
    delta = (actual.double() - expected.double()).norm().item()
    relative = delta / norm if norm else (0.0 if delta == 0 else float("inf"))
    assert relative < 0.02, f"Recompute gradient relative L2={relative}"
    return relative


def make_full_cases(checkpoint):
    """Real tokenizer/native processor, text and two different image pixels."""
    from PIL import Image

    import verl.models.mcore.qwen3_8_next.bridge  # noqa: F401 -- register the real HF config
    from tests.models.mcore.qwen38_vision_fixture import make_tiny_rgb
    from verl.utils.tokenizer import build_multimodal_processor_inputs, hf_processor

    processor = hf_processor(str(checkpoint), trust_remote_code=False, local_files_only=True)
    assert processor is not None, "A real processor is required, not a tiny tokenizer"
    rgb = make_tiny_rgb()
    texts = [
        "The answer to 2 + 3 is 5.",
        "A bag has 12 red marbles and 8 blue marbles. Four red marbles are removed. There are 16 marbles left.",
        "<|vision_start|><|image_pad|><|vision_end|>Describe the image briefly.",
        "<|vision_start|><|image_pad|><|vision_end|>Describe the image briefly.",
    ]
    cases = []
    for index, text in enumerate(texts):
        pixels = None if index < 2 else (rgb if index == 2 else 255 - rgb)
        images = None if pixels is None else [Image.fromarray(pixels)]
        encoded = build_multimodal_processor_inputs(
            processor,
            text=[text],
            images=images,
            mm_processor_kwargs={"min_pixels": 4096, "max_pixels": 4096},
        )
        ids = encoded["input_ids"][0].to(torch.long).cpu()
        assert 4 <= len(ids) <= 128
        mm = {key: encoded[key].cpu() for key in ("pixel_values", "image_grid_thw") if key in encoded}
        if pixels is not None:
            assert mm["image_grid_thw"].tolist() == [[1, 4, 4]]
            assert int((ids == processor.image_token_id).sum()) == 4
        cases.append(
            {
                "name": ("text_short", "text_long", "image", "changed_image")[index],
                "input_ids": ids,
                "raw_input_ids": torch.tensor(processor.tokenizer.encode(text, add_special_tokens=False)),
                "multimodal": mm,
                "rgb": None if pixels is None else torch.from_numpy(pixels.copy()),
            }
        )
    assert torch.equal(cases[2]["input_ids"], cases[3]["input_ids"])
    assert not torch.equal(cases[2]["multimodal"]["pixel_values"], cases[3]["multimodal"]["pixel_values"])
    return cases


def logprob_differences(actual, expected, token_ids):
    """Report local outliers and all-vocabulary errors, not sequence means only."""
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError("Logprob shapes must match [predicted_positions, vocabulary]")
    if len(token_ids) != len(actual):
        raise ValueError("One target token is required for each predicted position")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise ValueError("Nonfinite logprob in numerical comparison")
    delta = (actual.double() - expected.double()).abs()
    rows = torch.arange(len(token_ids))
    selected = delta[rows, token_ids]
    maximum = int(delta.argmax())
    return {
        "all_vocab_mean": delta.mean().item(),
        "all_vocab_max": delta.max().item(),
        "max_position": maximum // delta.shape[1],
        "max_vocab_id": maximum % delta.shape[1],
        "selected_token_mean": selected.mean().item(),
        "selected_token_max": selected.max().item(),
        "selected_token_abs_diff": selected.tolist(),
        "sequence_mean_abs_diff": abs((actual[rows, token_ids] - expected[rows, token_ids]).mean().item()),
    }
