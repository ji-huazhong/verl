# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the tiny image fixture; not real-tokenizer acceptance."""

import os

import numpy as np
import pytest
import torch

pytest.importorskip("transformers.models.qwen3_vl.processing_qwen3_vl")

from tests.models.mcore.qwen38_vision_fixture import make_tiny_rgb, make_tiny_vision_processor


def test_tiny_image_processor_round_trip(tmp_path):
    from PIL import Image
    from transformers import AutoProcessor

    processor = make_tiny_vision_processor()
    prompt = "T3 T4 T5 <|vision_start|><|image_pad|><|vision_end|> T9"
    image = Image.fromarray(make_tiny_rgb())
    original = processor(text=[prompt], images=[image], return_tensors="pt")
    assert original["input_ids"].tolist() == [[3, 4, 5, 250, 252, 252, 252, 252, 251, 9]]
    assert original["pixel_values"].shape == (16, 1536)
    assert original["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert len(processor.tokenizer) == 256
    processor.save_pretrained(tmp_path)
    restored = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    actual = restored(text=[prompt], images=[image], return_tensors="pt")
    assert actual.keys() == original.keys()
    for name in original:
        torch.testing.assert_close(actual[name], original[name], rtol=0, atol=0)
    changed = restored(text=[prompt], images=[Image.fromarray(255 - np.asarray(image))], return_tensors="pt")
    assert not torch.equal(changed["pixel_values"], original["pixel_values"])
    torch.testing.assert_close(changed["input_ids"], original["input_ids"], rtol=0, atol=0)


def test_image_positions_have_three_distinct_axes():
    from verl.models.transformers.qwen3_vl import get_rope_index

    processor = make_tiny_vision_processor()
    ids = torch.tensor([3, 4, 5, 250, 252, 252, 252, 252, 251, 9])
    actual = get_rope_index(processor, ids, image_grid_thw=torch.tensor([[1, 4, 4]]))
    # Four merged image patches form a 1x2x2 grid after the text prefix.
    # Text after the image resumes at max(image position) + 1, not at token 8.
    expected = torch.tensor(
        [[0, 1, 2, 3, 4, 4, 4, 4, 6, 7], [0, 1, 2, 3, 4, 4, 5, 5, 6, 7], [0, 1, 2, 3, 4, 5, 4, 5, 6, 7]]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_trainer_image_rows_preserve_pixels_and_placeholders():
    from io import BytesIO

    from PIL import Image

    from tests.models.mcore.prepare_qwen38_next_trainer_fixture import make_rows

    text_rows, image_rows = make_rows(), make_rows(images=True)
    assert len(text_rows) == len(image_rows) == 16
    assert all("images" not in row for row in text_rows)
    original = make_tiny_rgb()
    for index, row in enumerate(image_rows):
        assert row["prompt"][0]["content"].count("<image>") == 1
        assert len(row["images"]) == 1
        payload = row["images"][0]
        assert payload["min_pixels"] == payload["max_pixels"] == 4096
        actual = np.asarray(Image.open(BytesIO(payload["bytes"])))
        np.testing.assert_array_equal(actual, 255 - original if index % 2 else original)
        assert row["reward_model"] == text_rows[index]["reward_model"]


def test_real_tokenizer_image_rows_fit_trainer_limit(tmp_path):
    checkpoint = os.environ.get("QWEN38_MODEL_PATH")
    if not checkpoint:
        pytest.skip("Set QWEN38_MODEL_PATH to local public processor/tokenizer assets")
    import pandas as pd
    from omegaconf import OmegaConf

    from tests.models.mcore.prepare_qwen38_next_trainer_fixture import make_rows
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer import build_multimodal_processor_inputs
    from verl.workers.config.model import HFModelConfig

    model = HFModelConfig(path=checkpoint, external_lib="verl.models.mcore.qwen3_8_next.bridge")
    processor = model.processor
    source = tmp_path / "images.parquet"
    pd.DataFrame(make_rows(images=True)).to_parquet(source, index=False)
    config = OmegaConf.create(
        {
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1,
            "max_prompt_length": 64,
            "image_patch_size": 16,
            "mm_processor_kwargs": {"min_pixels": 4096, "max_pixels": 4096},
        }
    )
    # Explicit negative control: the old text-only smoke limit filters every
    # image sample. Do not disable filtering or silently truncate image tokens.
    assert len(RLHFDataset(str(source), model.tokenizer, config, processor=processor)) == 0
    config.max_prompt_length = 96
    dataset = RLHFDataset(str(source), model.tokenizer, config, processor=processor)
    assert len(dataset) == 16
    lengths = []
    for index in range(len(dataset)):
        messages = dataset[index]["raw_prompt"]
        raw = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, audios = dataset._process_multi_modal_info(messages, 16, config)
        assert videos is None and audios is None
        inputs = build_multimodal_processor_inputs(
            processor, text=[raw], images=images, mm_processor_kwargs=config.mm_processor_kwargs
        )
        lengths.append(len(inputs["input_ids"][0]))
        assert inputs["image_grid_thw"].tolist() == [[1, 4, 4]]
        assert inputs["pixel_values"].shape == (16, 1536)
        assert (inputs["input_ids"] == processor.image_token_id).sum() == 4
    assert 64 < min(lengths) <= max(lengths) <= 96
    print(f"QWEN38_IMAGE_PROMPT_LENGTH min={min(lengths)} max={max(lengths)} rows={len(lengths)}")
