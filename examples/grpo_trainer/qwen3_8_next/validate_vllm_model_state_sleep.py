# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""GPU regression for model-state constants across sleep and selective wake."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def run(output):
    import torch
    import vllm._C_stable_libtorch  # noqa: F401
    from vllm.device_allocator import get_mem_allocator_instance
    from vllm.device_allocator.sleep_mode_backend import CuMemBackend
    from vllm.v1.worker.gpu.model_states import init_model_state

    version = importlib.metadata.version("vllm")
    assert version == "0.30.1.dev0+g6e517b15.torch210.cu131.hcgdnfp32.plestate"
    source = Path(inspect.getfile(init_model_state))
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha256 == "ce7ab759968221debdab4e8a4baf9fead1a156287c92dba8cb8b4cc35816f8a2"

    class State:
        def __init__(self, config, model, encoder_cache, device):
            self.offsets = torch.arange(-2, 0, dtype=torch.int64, device=device)

    allocator = get_mem_allocator_instance()
    config = SimpleNamespace(model_config=SimpleNamespace(enable_sleep_mode=True, enable_prompt_embeds=False))
    model = SimpleNamespace(get_model_state_cls=lambda: State)
    with allocator.use_memory_pool("weights"):
        state = init_model_state(config, model, None, torch.device("cuda:0"))
    pointer = state.offsets.data_ptr()
    backend = CuMemBackend()
    cases = []
    for level in (1, 2):
        for tags in (["weights"], ["kv_cache"], None):
            backend.suspend(level=level)
            backend.resume(tags=tags)
            assert state.offsets.data_ptr() == pointer
            assert state.offsets.tolist() == [-2, -1]
            history = torch.tensor([11, 17, 23, 31], device="cuda:0")
            assert history[3 + state.offsets].tolist() == [17, 23]
            backend.resume()
            cases.append(dict(level=level, wake_tags=tags, values_preserved=True, address_preserved=True))
    report = dict(complete=True, vllm_version=version, source_sha256=source_sha256, cases=cases)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("QWEN38_REAL_PROMPT_SLEEP_POOL_PASS", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
