# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""GPU gate for trainable PLE, including distinct sequence-parallel consumers.

RUN_QWEN38_PLE_DISTRIBUTED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nproc-per-node=4 -m pytest -q -s <this file>
"""

import os
from datetime import timedelta

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_PLE_DISTRIBUTED") != "1", reason="explicit four-GPU PLE gate required"
)


@pytest.fixture(scope="module", autouse=True)
def parallel_context():
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=3))
    assert torch.distributed.get_world_size() == 4
    ps.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    yield ps
    ps.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_native_ple_lookup_gradient_and_optimizer(parallel_context, sequence_parallel):
    from megatron.core.transformer.transformer_config import TransformerConfig

    from verl.models.mcore.qwen3_8_next.ops.ple import Qwen38NextNGramEmbedding

    ps = parallel_context
    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        sequence_parallel=sequence_parallel,
        params_dtype=torch.float32,
        perform_initialization=True,
        use_cpu_initialization=False,
    )
    config.qwen3_8_next_train_ple = True
    config.qwen3_8_next_ngram_size = 3
    config.qwen3_8_next_heads_per_ngram = 1
    config.qwen3_8_next_ple_embed_dim = 4
    config.qwen3_8_next_split_ngram_parts = 4
    config.qwen3_8_next_ngram_rows_per_shard = 8
    embedding = Qwen38NextNGramEmbedding(config, 1, tp_group=ps.get_tensor_model_parallel_group()).cuda()
    reference = torch.nn.Embedding(32, 2, device="cuda")
    table = torch.arange(64, dtype=torch.float32, device="cuda").view(32, 2) / 64
    rank = ps.get_tensor_model_parallel_rank()
    with torch.no_grad():
        reference.weight.copy_(table)
        embedding.ngram_embedding.weight.copy_(table[rank * 16 : (rank + 1) * 16])
    ids = torch.tensor([[0, 17], [18, 1], [1, 19], [20, 0]], device="cuda")
    expected = reference(ids).flatten(-2)
    actual = embedding(ids)
    torch.testing.assert_close(actual, expected)
    # SP ranks own disjoint output rows, yet both consume table rows from both
    # vocab partitions. Missing the embedding gradient all-reduce fails here.
    loss = actual[rank * 2 : (rank + 1) * 2].square().sum() if sequence_parallel else actual.square().sum()
    loss.backward()
    expected.square().sum().backward()
    weight = embedding.ngram_embedding.weight
    torch.testing.assert_close(weight.grad, reference.weight.grad[rank * 16 : (rank + 1) * 16])
    assert weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(embedding.parameters(), lr=0.01)
    optimizer.step()
    torch.testing.assert_close(weight, (table - 0.01 * reference.weight.grad)[rank * 16 : (rank + 1) * 16])
    assert "ngram_embedding.weight" in embedding.sharded_state_dict()


def test_tp_pp_shards_round_trip_without_broadcasting_local_table(parallel_context):
    from verl.models.mcore.qwen3_8_next.param_mapping import PLENGramEmbeddingMapping

    ps = parallel_context
    mapping = PLENGramEmbeddingMapping("ple.weight", "ple.shard_{}.weight", 4)
    table = torch.arange(64, dtype=torch.float32, device="cuda").view(32, 2)
    rank = ps.get_tensor_model_parallel_rank()
    local_rows = table[rank * 16 : (rank + 1) * 16] if ps.get_pipeline_model_parallel_rank() == 0 else None
    values = mapping.megatron_to_hf(local_rows, None).with_dtype(torch.bfloat16)
    for index, (_, shard) in enumerate(values.items()):
        assert tuple(shard.shape) == (8, 2)
        torch.testing.assert_close(shard, table[index * 8 : (index + 1) * 8].to(torch.bfloat16))
        assert not shard.requires_grad
