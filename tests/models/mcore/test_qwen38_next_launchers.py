# SPDX-License-Identifier: Apache-2.0
"""Launch argument contracts only; the Python preflight/trainer are not executed."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["default", "resident_ep1", "resume"])
def test_full_smoke_parallel_and_sleep_arguments(tmp_path, mode):
    root = Path(__file__).resolve().parents[3]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python3"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:] == ['-']:\n"
        "    sys.stdin.read()  # Deliberately bypass GPU preflight in this unit test.\n"
        "else:\n"
        "    Path(os.environ['QWEN38_CAPTURE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    stub.chmod(0o700)
    data = tmp_path / "data.parquet"
    data.touch()
    captured = tmp_path / "arguments.json"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "MODEL_PATH": str(tmp_path / "not_loaded"),
        "TRAIN_FILE": str(data),
        "VAL_FILE": str(data),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "QWEN38_FULL_OUTPUT": str(tmp_path / "output"),
        "QWEN38_RAY_TEMP": str(tmp_path / "ray"),
        "QWEN38_CAPTURE_ARGS": str(captured),
        "QWEN38_RESUME_FROM": "",
    }
    overrides = []
    if mode == "resident_ep1":
        overrides = [
            "actor_rollout_ref.rollout.expert_parallel_size=1",
            "actor_rollout_ref.rollout.free_cache_engine=False",
        ]
    if mode == "resume":
        checkpoint = tmp_path / "checkpoint"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "actor/ckpt_contents.json").write_text("{}")
        env["QWEN38_RESUME_FROM"] = str(checkpoint)
    subprocess.run(
        ["bash", "examples/tuning/lora/run_qwen38_flash_next_hybrid_smoke.sh", *overrides],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    args = json.loads(captured.read_text())
    assert args[:2] == ["-m", "verl.trainer.main_ppo"]
    effective = dict(arg.lstrip("+").split("=", 1) for arg in args[2:])
    assert effective["actor_rollout_ref.model.external_lib"] == "megatron.bridge.models.qwen38_next"
    rollout = "actor_rollout_ref.rollout."
    assert effective[rollout + "tensor_model_parallel_size"] == "8"
    assert effective[rollout + "data_parallel_size"] == "1"
    assert effective[rollout + "expert_parallel_size"] == ("1" if mode == "resident_ep1" else "8")
    assert effective[rollout + "free_cache_engine"] == ("False" if mode == "resident_ep1" else "True")
    assert effective[rollout + "engine_kwargs.vllm.all2all_backend"] == "allgather_reducescatter"
    assert effective[rollout + "engine_kwargs.vllm.fully_sharded_loras"] == "False"
    for role in ("actor", "ref"):
        for dimension in ("tensor", "pipeline", "expert", "virtual_pipeline"):
            assert effective[f"actor_rollout_ref.{role}.megatron.{dimension}_model_parallel_size"] == "2"
        assert effective[f"actor_rollout_ref.{role}.megatron.context_parallel_size"] == "2"
        assert effective[f"actor_rollout_ref.{role}.megatron.expert_tensor_parallel_size"] == "1"
    assert effective["trainer.resume_mode"] == ("resume_path" if mode == "resume" else "disable")
    if mode == "resume":
        assert effective["trainer.resume_from_path"] == env["QWEN38_RESUME_FROM"]


def test_geo3k_smoke_uses_separate_real_splits(tmp_path):
    root = Path(__file__).resolve().parents[3]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python3"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:] == ['-']:\n"
        "    sys.stdin.read()\n"
        "else:\n"
        "    Path(os.environ['QWEN38_CAPTURE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    stub.chmod(0o700)
    data = tmp_path / "geo3k"
    data.mkdir()
    for split in ("train", "test"):
        (data / f"{split}.parquet").touch()
    captured = tmp_path / "arguments.json"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "MODEL_PATH": str(tmp_path / "not_loaded"),
        "GEO3K_DIR": str(data),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "QWEN38_FULL_OUTPUT": str(tmp_path / "output"),
        "QWEN38_RAY_TEMP": str(tmp_path / "ray"),
        "QWEN38_CAPTURE_ARGS": str(captured),
        "QWEN38_RESUME_FROM": "",
    }
    subprocess.run(
        ["bash", "examples/tuning/lora/run_qwen38_flash_next_geo3k_smoke.sh"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    args = json.loads(captured.read_text())
    effective = dict(arg.lstrip("+").split("=", 1) for arg in args[2:])
    assert effective["data.train_files"] == str(data / "train.parquet")
    assert effective["data.val_files"] == str(data / "test.parquet")
    assert effective["data.truncation"] == "error"
    assert effective["data.filter_overlong_prompts"] == "True"
    assert effective["actor_rollout_ref.rollout.calculate_log_probs"] == "True"
    assert effective["trainer.total_training_steps"] == "2"
    assert effective["trainer.logger"] == "[console,tensorboard]"
