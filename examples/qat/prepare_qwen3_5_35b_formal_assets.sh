#!/usr/bin/env bash
# Prepare reproducible assets for the Qwen3.5-35B-A3B INT4-QAT RL study.
#
# The script is deliberately download-only: it does not start Ray, vLLM, or
# training.  It uses the post-trained checkpoint because it is the model that
# will produce GRPO rollouts.  QAT still keeps BF16 master weights at runtime.
#
# The H20 container used for the first run has no outbound HTTPS connectivity
# to huggingface.co.  ModelScope mirrors the same official Qwen and Byted
# assets, so it is the default downloader.  Set ASSET_DOWNLOADER=hf only on a
# host where Hugging Face connectivity has been checked.
#
# Assets created or verified:
#   * Qwen/Qwen3.5-35B-A3B (BF16 Hugging Face checkpoint)
#   * existing verl-formatted Geo3K Parquets (vision diagnostic track)
#   * BytedTsinghua-SIA/DAPO-Math-17k (long-run RL training track)
#   * BytedTsinghua-SIA/AIME-2024 (during-training validation)
#
# Usage on the H20 host:
#   bash examples/qat/prepare_qwen3_5_35b_formal_assets.sh
#   bash examples/qat/prepare_qwen3_5_35b_formal_assets.sh --data-only
#   bash examples/qat/prepare_qwen3_5_35b_formal_assets.sh --verify-only

set -euo pipefail

MODE="all"
case "${1:-}" in
    "" ) ;;
    --model-only) MODE="model" ;;
    --data-only) MODE="data" ;;
    --verify-only) MODE="verify" ;;
    *)
        echo "usage: $0 [--model-only|--data-only|--verify-only]" >&2
        exit 2
        ;;
esac

MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3.5-35B-A3B}
MODEL_REVISION=${MODEL_REVISION:-master}
MODEL_DIR=${MODEL_DIR:-/workspace/models/Qwen3.5-35B-A3B}

DATA_ROOT=${DATA_ROOT:-/workspace/data}
GEO3K_DIR=${GEO3K_DIR:-${DATA_ROOT}/geo3k}
DAPO_DIR=${DAPO_DIR:-${DATA_ROOT}/DAPO-Math-17k}
AIME_DIR=${AIME_DIR:-${DATA_ROOT}/AIME-2024}
ASSET_LOG_DIR=${ASSET_LOG_DIR:-${DATA_ROOT}/int4-qat-rl/20260902-qwen3_5_35b-formal-prep}

# Keep the large Xet/cache files off the container root filesystem.
export HF_HOME=${HF_HOME:-${DATA_ROOT}/hf_home}
ASSET_DOWNLOADER=${ASSET_DOWNLOADER:-modelscope}

mkdir -p "${MODEL_DIR}" "${DAPO_DIR}" "${AIME_DIR}" "${ASSET_LOG_DIR}" "${HF_HOME}"

require_file() {
    local path="$1"
    [[ -s "${path}" ]] || { echo "missing or empty: ${path}" >&2; return 1; }
}

download_asset() {
    local repo="$1"
    local repo_type="$2"
    local local_dir="$3"
    shift 3

    case "${ASSET_DOWNLOADER}" in
        modelscope)
            modelscope download "${repo}" \
                --repo-type "${repo_type}" \
                --revision "${MODEL_REVISION}" \
                --local-dir "${local_dir}" \
                "$@"
            ;;
        hf)
            hf download "${repo}" \
                --repo-type "${repo_type}" \
                --revision "${MODEL_REVISION}" \
                --local-dir "${local_dir}" \
                "$@"
            ;;
        *)
            echo "unsupported ASSET_DOWNLOADER=${ASSET_DOWNLOADER}; expected modelscope or hf" >&2
            return 2
            ;;
    esac
}

verify_model() {
    require_file "${MODEL_DIR}/config.json"
    require_file "${MODEL_DIR}/model.safetensors.index.json"
    local shard_count
    shard_count=$(find "${MODEL_DIR}" -maxdepth 1 -name 'model.safetensors-*-of-*.safetensors' -type f | wc -l)
    [[ "${shard_count}" -eq 14 ]] || {
        echo "expected 14 model shards, found ${shard_count} in ${MODEL_DIR}" >&2
        return 1
    }
}

verify_parquet() {
    local path="$1"
    local expected_rows="$2"
    python3 - "${path}" "${expected_rows}" <<'PY'
import sys
import pyarrow.parquet as pq

path, expected_rows = sys.argv[1], int(sys.argv[2])
pf = pq.ParquetFile(path)
if pf.metadata.num_rows != expected_rows:
    raise SystemExit(f"{path}: expected {expected_rows} rows, got {pf.metadata.num_rows}")
print(f"verified parquet rows={pf.metadata.num_rows} path={path}")
PY
}

verify_geo3k() {
    verify_parquet "${GEO3K_DIR}/train.parquet" 2101
    verify_parquet "${GEO3K_DIR}/test.parquet" 601
}

verify_dapo() {
    require_file "${DAPO_DIR}/data/dapo-math-17k.parquet"
    require_file "${AIME_DIR}/data/aime-2024.parquet"
    python3 - "${DAPO_DIR}/data/dapo-math-17k.parquet" "${AIME_DIR}/data/aime-2024.parquet" <<'PY'
import sys
import pyarrow.parquet as pq

for path in sys.argv[1:]:
    pf = pq.ParquetFile(path)
    if pf.metadata.num_rows <= 0:
        raise SystemExit(f"empty parquet: {path}")
    # ``ParquetFile.schema.names`` contains leaf field names for nested Arrow
    # structs (for example ``prompt.content``), not the user-facing top-level
    # verl columns.  Read the Arrow schema so valid nested prompt/reward_model
    # fields are not rejected.
    names = set(pq.read_schema(path).names)
    required = {"data_source", "prompt", "reward_model"}
    missing = required - names
    if missing:
        raise SystemExit(f"{path}: missing verl fields {sorted(missing)}")
    print(f"verified parquet rows={pf.metadata.num_rows} path={path}")
PY
}

if [[ "${MODE}" == "verify" ]]; then
    verify_model
    verify_geo3k
    verify_dapo
    du -sh "${MODEL_DIR}" "${GEO3K_DIR}" "${DAPO_DIR}" "${AIME_DIR}"
    exit 0
fi

if [[ "${MODE}" == "all" || "${MODE}" == "model" ]]; then
    download_asset "${MODEL_REPO}" model "${MODEL_DIR}"
    verify_model
fi

if [[ "${MODE}" == "all" || "${MODE}" == "data" ]]; then
    # Geo3K is preprocessed by examples/data_preprocess/geo3k.py and is already
    # present at the default H20 location.  Verify it instead of replacing it.
    verify_geo3k

    # These two datasets are published directly in the verl Parquet schema.
    download_asset BytedTsinghua-SIA/DAPO-Math-17k dataset "${DAPO_DIR}" \
        --include 'data/dapo-math-17k.parquet'
    download_asset BytedTsinghua-SIA/AIME-2024 dataset "${AIME_DIR}" \
        --include 'data/aime-2024.parquet'
    verify_dapo
fi

printf 'asset_downloader=%s\nmodel_repo=%s\nmodel_revision=%s\nmodel_dir=%s\ngeo3k_dir=%s\ndapo_dir=%s\naime_dir=%s\n' \
    "${ASSET_DOWNLOADER}" "${MODEL_REPO}" "${MODEL_REVISION}" "${MODEL_DIR}" "${GEO3K_DIR}" "${DAPO_DIR}" "${AIME_DIR}" \
    > "${ASSET_LOG_DIR}/asset-paths.env"

echo "asset preparation completed; manifest: ${ASSET_LOG_DIR}/asset-paths.env"
