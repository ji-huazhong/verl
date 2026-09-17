# SPDX-License-Identifier: Apache-2.0
"""Prepare a RANDOM Flash-Next trainer fixture with the real tokenizer assets.

This expands only the vocabulary of the exported four-layer GPU fixture. It
does not copy real model weights or claim real-model/accuracy acceptance.
Usage: python <this file> --fixture <GPU export> --assets <real checkpoint>
       --output <new directory> [--images]
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Deterministic synthetic signal; deliberately NOT a math/quality reward."""
    assert data_source == "qwen38_synthetic_integration"
    digest = hashlib.sha256(solution_str.encode("utf-8")).digest()
    return {"score": int.from_bytes(digest[:4], "little") / (2**32 - 1)}


def make_rows(images=False):
    rows = [
        {
            "data_source": "qwen38_synthetic_integration",
            "prompt": [{"role": "user", "content": f"Say a short word about number {i}."}],
            "ability": "synthetic_integration_only",
            "reward_model": {"style": "rule", "ground_truth": "not_an_accuracy_test"},
            "extra_info": {"index": i},
        }
        for i in range(16)
    ]
    if images:
        from io import BytesIO

        from PIL import Image

        from tests.models.mcore.qwen38_vision_fixture import make_tiny_rgb

        rgb = make_tiny_rgb()
        for index, row in enumerate(rows):
            encoded = BytesIO()
            Image.fromarray(255 - rgb if index % 2 else rgb).save(encoded, format="PNG")
            row["prompt"][0]["content"] = f"<image>Say a short word about this image and number {index}."
            row["images"] = [{"bytes": encoded.getvalue(), "min_pixels": 4096, "max_pixels": 4096}]
    return rows


def main():
    import pandas as pd
    import torch
    from safetensors.torch import load_file, save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", action="store_true", help="Use synthetic RGB images with real processor assets")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; do not overwrite a fixture or checkpoint")
    config = json.loads((args.fixture / "model" / "config.json").read_text())
    real = json.loads((args.assets / "config.json").read_text())
    assert config["model_type"] == real["model_type"] == "qwen4_exp"
    assert config["text_config"]["num_hidden_layers"] == 4
    vocab = real["text_config"]["vocab_size"]
    assert config["text_config"]["vocab_size"] < vocab
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        config[key] = real[key]
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        if key in real["text_config"]:
            config["text_config"][key] = real["text_config"][key]
    weights = load_file(str(args.fixture / "model" / "model.safetensors"))
    rng = torch.Generator().manual_seed(20260917)
    for key in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
        original = weights[key]
        expanded = (torch.randn(vocab, original.shape[1], generator=rng) * 0.02).to(original.dtype)
        expanded[: original.shape[0]].copy_(original)
        weights[key] = expanded
    model_dir = args.output / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(weights, str(model_dir / "model.safetensors"))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
    ):
        shutil.copy2(args.assets / name, model_dir / name)
    rows = make_rows(images=args.images)
    pd.DataFrame(rows).to_parquet(args.output / "train.parquet", index=False)
    pd.DataFrame(rows[:4]).to_parquet(args.output / "val.parquet", index=False)
    print(f"QWEN38_RANDOM_TRAINER_FIXTURE vocab={vocab} tensors={len(weights)} rows={len(rows)} images={args.images}")


if __name__ == "__main__":
    main()
