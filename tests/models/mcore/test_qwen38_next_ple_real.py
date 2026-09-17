# SPDX-License-Identifier: Apache-2.0
"""Opt-in full frozen-table loading/lookup gate; not full-model GRPO acceptance.

Run under torchrun with RUN_QWEN38_REAL_PLE_TESTS=1 and QWEN38_MODEL_PATH.
The table remains host-resident; enough host RAM for the complete table is
required. This checks sampled rows, not a checksum of all checkpoint bytes.
"""

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_REAL_PLE_TESTS") != "1", reason="explicit full PLE table loading opt-in required"
)


def test_real_frozen_table_load_and_distributed_lookup():
    import psutil
    from safetensors import safe_open
    from transformers import AutoConfig

    from verl.models.mcore.qwen3_8_next.bridge import Qwen38NextBridge
    from verl.models.mcore.qwen3_8_next.ops.ple import Qwen38NextFrozenNGramEmbedding, build_ngram_contexts

    checkpoint = Path(os.environ["QWEN38_MODEL_PATH"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(1024**3 / total)
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
    try:
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        provider = Qwen38NextBridge().provider_bridge(
            SimpleNamespace(config=config, model_name_or_path=str(checkpoint))
        )
        assert len(config.text_config.ple_layer_ids) == 1
        layer_number = config.text_config.ple_layer_ids[0]
        prefix = f"model.language_model.layers.{layer_number - 1}.ple.ple_embedding.ngram_embedding"
        weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        shapes = []
        for shard in range(provider.qwen3_8_next_split_ngram_parts):
            name = f"{prefix}.shard_{shard}.weight"
            with safe_open(str(checkpoint / weight_map[name]), framework="pt", device="cpu") as file:
                shapes.append(file.get_slice(name).get_shape())
        assert all(shape == shapes[0] for shape in shapes)
        table_bytes = sum(rows * width * 2 for rows, width in shapes)
        # Each local rank allocates its own shard. Budget the complete table,
        # plus a second table's headroom, before any rank allocates pinned RAM.
        ready = torch.tensor(
            int(free >= 4 * 1024**3 and psutil.virtual_memory().available > 2 * table_bytes + 16 * 1024**3),
            device=device,
        )
        torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        if not ready.item():
            pytest.skip("Insufficient GPU/host headroom; never evict another job")

        started = time.perf_counter()
        embedding = Qwen38NextFrozenNGramEmbedding(provider, layer_number, tp_group=torch.distributed.group.WORLD)
        embedding = embedding.to(device)  # Only metadata buffers move; the table is not a buffer.
        embedding.load_from_hf(str(checkpoint))
        load_seconds = time.perf_counter() - started
        assert embedding._loaded and embedding.table.device.type == "cpu" and embedding.table.is_pinned()
        assert not list(embedding.parameters())
        assert all("ngram_embedding" not in name for name in embedding.state_dict())

        rows_per_shard = shapes[0][0]
        # Start, middle and last row of EVERY physical checkpoint shard.
        row_ids = [
            shard * rows_per_shard + offset
            for shard in range(len(shapes))
            for offset in (0, rows_per_shard // 2, rows_per_shard - 1)
        ]
        tokens = torch.tensor([1, 64, 65, embedding.eos_token_id, 100, 101, 102], device=device)
        contexts = build_ngram_contexts(tokens, embedding.ngram_size, embedding.eos_token_id)
        row_ids.extend(embedding.compute_ngram_ids(contexts).flatten().cpu().tolist())
        ids = torch.tensor(row_ids, device=device, dtype=torch.int64).unsqueeze(-1)
        actual = embedding(ids).cpu()
        expected = []
        for row_id in row_ids:
            shard, row = divmod(row_id, rows_per_shard)
            name = f"{prefix}.shard_{shard}.weight"
            with safe_open(str(checkpoint / weight_map[name]), framework="pt", device="cpu") as file:
                expected.append(file.get_slice(name)[row : row + 1])
        expected = torch.cat(expected)
        assert bool(actual.isfinite().all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(
            "QWEN38_REAL_PLE "
            + json.dumps(
                dict(
                    rank=torch.distributed.get_rank(),
                    world_size=torch.distributed.get_world_size(),
                    shards=len(shapes),
                    global_table_bytes=table_bytes,
                    local_table_bytes=embedding.table.numel() * embedding.table.element_size(),
                    sampled_rows=len(row_ids),
                    load_seconds=load_seconds,
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                    exact=True,
                )
            ),
            flush=True,
        )
    finally:
        torch.distributed.destroy_process_group()
