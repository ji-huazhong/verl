# SPDX-License-Identifier: Apache-2.0
"""Deterministic image/processor assets for a 256-token random VLM fixture.

The image/video processor implementations are real Transformers components;
the tiny WordLevel tokenizer is a test asset, not the production tokenizer.
"""


def make_tiny_vision_processor():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    tokens = ["[PAD]", "[BOS]", "[EOS]"] + [f"T{i}" for i in range(3, 249)]
    tokens += [
        "[UNK]",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|im_start|>",
        "<|im_end|>",
    ]
    assert len(tokens) == 256
    backend = Tokenizer(models.WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        additional_special_tokens=tokens[250:],
        model_max_length=256,
    )
    image_processor = Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 4096, "longest_edge": 4096},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    video_processor = Qwen3VLVideoProcessor(
        patch_size=16, temporal_patch_size=2, merge_size=2, size={"shortest_edge": 4096, "longest_edge": 4096}
    )
    processor = Qwen3VLProcessor(image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor)
    assert (processor.vision_start_token_id, processor.vision_end_token_id, processor.image_token_id) == (250, 251, 252)
    return processor


def make_tiny_rgb():
    import numpy as np

    y, x = np.indices((64, 64))
    return np.stack(((3 * x + y) % 256, (x + 5 * y) % 256, (7 * x + 11 * y) % 256), axis=-1).astype(np.uint8)
