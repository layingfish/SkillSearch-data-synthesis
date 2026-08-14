#!/usr/bin/env bash
set -euo pipefail

bash /mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh --with-dataset-rw || exit 1

CODE_ROOT=/workspace/SkillSearch-data-synthesis
cd "${CODE_ROOT}"

OUTPUT_ROOT="${DIVERSE_MINING_OUTPUT_ROOT:-/models_rw/SkillSearch-data-synthesis/experiments/a800_diverse_coverage_mining_20260814_v1}"
FILTER_ROOT="${DIVERSE_FILTER_OUTPUT_ROOT:-/dataset_rw/SkillSearch-data-synthesis/processed/diverse_coverage_20260814_v1/error_filtered_cum082_v1}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite diverse mining output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ -e "${FILTER_ROOT}" ]]; then
  echo "refusing to overwrite error-filtered output: ${FILTER_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/mining"
exec > >(tee "${OUTPUT_ROOT}/logs/node.log") 2>&1
echo "launch_commit=${DIVERSE_MINING_LAUNCH_COMMIT:-unknown}"

PY310_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1
PY310_RUNTIME_ARCHIVE="${PY310_RUNTIME_ROOT}/runtime.tar.zst"
PY310_RUNTIME_ARCHIVE_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
PY311_SITE_ARCHIVE="${PY311_RUNTIME_ROOT}/site.tar.zst"
PY311_SITE_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
BASE_MODEL_SHA256=a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4

PIPELINE="${CODE_ROOT}/scripts/skill_gr_pipeline.py"
PIPELINE_SHA256=5ee736cddebca4a7e7251d95e85bcdfbaeede3a6d60f78696481c64a81a66f59
MINER="${CODE_ROOT}/scripts/prepare_gr_self_mined_hard_negatives.py"
MINER_SHA256=05d1698619bea860fd344929a398cddd482e0e310332da32cdf71b742ce9a924
BUILDER="${CODE_ROOT}/scripts/build_diverse_coverage_queries.py"
BUILDER_SHA256=0da4db087d5b329d8ca532539b5ae668c5c16a11e507d075fcecdf4e00a47d06
BUILDER_TEST="${CODE_ROOT}/tests/test_build_diverse_coverage_queries.py"
BUILDER_TEST_SHA256=bdda279363ed6e84fe9dce0ab2e451f4c6abb442486d22dab7ae84e56a4d5738

DATA_ROOT=/mnt/cfs_algo_bj/workspace/chenyufei03/datasets/SkillSearch-data-synthesis
MINING_INPUT_ROOT="${DATA_ROOT}/processed/diverse_coverage_20260814_v1/mining_input_cum082_v1"
MINING_ROWS="${MINING_INPUT_ROOT}/mining_rows.jsonl"
MINING_ROWS_SHA256=9a442c06ee2793fa3bf792cc0373c59bff8f3f1b95bb3877f717eb517025f86e
MINING_INPUT_MANIFEST="${MINING_INPUT_ROOT}/manifest.json"
MINING_INPUT_MANIFEST_SHA256=c9ecac076b3399c41912c9c588087c3230f6f3ca5040bb11aa162adb4af87b7e
SEMANTIC_IDS=/models_rw/SkillSearch-data-synthesis/assets-models/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/semantic_ids.jsonl
SEMANTIC_IDS_SHA256=084cccd5033d0aee8fb7bfbf05d55508744376c50a3c9bef95803ee69b41fa61
MINING_MODEL=/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_self_mined_nearmiss_20260813_v1/selected/single_top1_best
MINING_MODEL_SHA256=b55092f2be32ff063bff3f4387f154343dcd5f652573e4d044274df33d39cf1d

check_sha256() {
  local path="$1" expected="$2" observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || {
    echo "SHA-256 mismatch for ${path}: expected=${expected}, observed=${observed}" >&2
    exit 1
  }
}

TEMP_ROOT="$(mktemp -d /tmp/diverse-coverage-mining.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/diverse-coverage-mining.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

check_sha256 "${PY310_RUNTIME_ARCHIVE}" "${PY310_RUNTIME_ARCHIVE_SHA256}"
check_sha256 "${PY311_SITE_ARCHIVE}" "${PY311_SITE_ARCHIVE_SHA256}"
mkdir -p "${TEMP_ROOT}/py310" "${TEMP_ROOT}/py311"
tar --zstd -xf "${PY310_RUNTIME_ARCHIVE}" -C "${TEMP_ROOT}/py310"
tar --zstd -xf "${PY311_SITE_ARCHIVE}" -C "${TEMP_ROOT}/py311"
PYTHON_BIN="${TEMP_ROOT}/py310/python/bin/python3"
PY310_SITE="${TEMP_ROOT}/py310/site"
PY311_SITE="${TEMP_ROOT}/py311/site"
NVIDIA_LIBRARY_PATHS="$(find "${PY311_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
[[ -n "${NVIDIA_LIBRARY_PATHS}" ]] || { echo "fixed NVIDIA libraries are missing" >&2; exit 1; }
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/scripts:${PY310_SITE}:${PY311_SITE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4

"${PYTHON_BIN}" - <<'PY'
import json
import platform
from importlib import metadata

import torch
import transformers

expected = {
    "python": "3.10.12",
    "torch": "2.10.0+cu128",
    "cuda": "12.8",
    "cudnn": 91002,
    "transformers": "5.3.0",
    "tokenizers": "0.22.2",
    "gpu_count": 8,
}
actual = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "tokenizers": metadata.version("tokenizers"),
    "gpu_count": torch.cuda.device_count(),
}
if actual != expected:
    raise SystemExit(f"diverse mining runtime mismatch: expected={expected}, actual={actual}")
print(json.dumps({"runtime": actual}, sort_keys=True))
PY

for required in \
  "${MINING_ROWS}" "${MINING_INPUT_MANIFEST}" "${SEMANTIC_IDS}" \
  "${MINING_MODEL}/structured_decoder.pt"; do
  [[ -f "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done
check_sha256 "${BASE_MODEL}/model.safetensors" "${BASE_MODEL_SHA256}"
check_sha256 "${PIPELINE}" "${PIPELINE_SHA256}"
check_sha256 "${MINER}" "${MINER_SHA256}"
check_sha256 "${BUILDER}" "${BUILDER_SHA256}"
check_sha256 "${BUILDER_TEST}" "${BUILDER_TEST_SHA256}"
check_sha256 "${MINING_ROWS}" "${MINING_ROWS_SHA256}"
check_sha256 "${MINING_INPUT_MANIFEST}" "${MINING_INPUT_MANIFEST_SHA256}"
check_sha256 "${SEMANTIC_IDS}" "${SEMANTIC_IDS_SHA256}"
check_sha256 "${MINING_MODEL}/structured_decoder.pt" "${MINING_MODEL_SHA256}"

MODEL_SAFE="${TEMP_ROOT}/t5-base-safetensors-only"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done
export SKILLGR_MODEL="${MODEL_SAFE}"

"${PYTHON_BIN}" -m py_compile "${PIPELINE}" "${MINER}" "${BUILDER}" "${BUILDER_TEST}"
PYTHONPATH="${CODE_ROOT}/tests:${CODE_ROOT}/scripts:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON_BIN}" -m unittest test_build_diverse_coverage_queries -v

SHARD_ROOT="${OUTPUT_ROOT}/mining/shards"
"${PYTHON_BIN}" "${MINER}" shard \
  --rows "${MINING_ROWS}" \
  --output-dir "${SHARD_ROOT}" \
  --num-shards 8

"${PYTHON_BIN}" - "${SHARD_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = [row["rows"] for row in manifest["shards"]]
if manifest["row_count"] != 10332 or manifest["num_shards"] != 8:
    raise SystemExit(f"diverse mining shard contract drifted: {manifest}")
if counts != [1292, 1292, 1292, 1292, 1291, 1291, 1291, 1291]:
    raise SystemExit(f"diverse mining shard balance drifted: {counts}")
print(json.dumps({"mining_shards_verified": True, "counts": counts}, sort_keys=True))
PY

mine_shard() {
  local gpu="$1" shard_label shard_rows shard_output
  shard_label="$(printf 'shard_%02d' "${gpu}")"
  shard_rows="${SHARD_ROOT}/${shard_label}.jsonl"
  shard_output="${OUTPUT_ROOT}/mining/predictions/${shard_label}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${PIPELINE}" eval-single-sid-decoder \
    --rows "${shard_rows}" \
    --model-dir "${MINING_MODEL}" \
    --semantic-ids "${SEMANTIC_IDS}" \
    --output-dir "${shard_output}" \
    --split-name "diverse_coverage_mining_${shard_label}" \
    --max-source-length 2048 \
    --num-beams 50 \
    --rank-aggregation beam_order \
    --decoder-head ce \
    --score-transform logprob \
    --k-values 1,10,50 \
    --log-every 250
}

unset CUBLAS_WORKSPACE_CONFIG || true
declare -a MINING_PIDS=()
for gpu in {0..7}; do
  (mine_shard "${gpu}") > >(tee "${OUTPUT_ROOT}/logs/mining_gpu${gpu}.log") 2>&1 &
  MINING_PIDS+=("$!")
done
mining_failures=0
for pid in "${MINING_PIDS[@]}"; do
  if ! wait "${pid}"; then
    mining_failures=$((mining_failures + 1))
  fi
done
if (( mining_failures > 0 )); then
  echo "diverse mining failed on ${mining_failures} workers" >&2
  exit 1
fi

declare -a PREDICTION_ARGS=()
for gpu in {0..7}; do
  shard_label="$(printf 'shard_%02d' "${gpu}")"
  PREDICTION_ARGS+=(--predictions "${OUTPUT_ROOT}/mining/predictions/${shard_label}/predictions.jsonl")
done
CANDIDATE_ROOT="${OUTPUT_ROOT}/mining/candidates"
"${PYTHON_BIN}" "${MINER}" compact \
  --rows "${MINING_ROWS}" \
  "${PREDICTION_ARGS[@]}" \
  --semantic-ids "${SEMANTIC_IDS}" \
  --model-dir "${MINING_MODEL}" \
  --output-dir "${CANDIDATE_ROOT}" \
  --expected-negatives 10 \
  --expected-beams 50

"${PYTHON_BIN}" "${BUILDER}" filter-errors \
  --mining-rows "${MINING_ROWS}" \
  --candidates "${CANDIDATE_ROOT}/candidates.jsonl" \
  --candidate-manifest "${CANDIDATE_ROOT}/manifest.json" \
  "${PREDICTION_ARGS[@]}" \
  --output-dir "${FILTER_ROOT}" \
  --expected-rows 10332 \
  --expected-targets 1722 \
  --views-per-target 6 \
  --expected-negatives 10 \
  --expected-beams 50 \
  --expected-checkpoint-sha256 "${MINING_MODEL_SHA256}"

"${PYTHON_BIN}" - "${FILTER_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest["input_rows"] != 10332 or manifest["target_skill_count"] != 1722:
    raise SystemExit(f"error-filtered input closure drifted: {manifest}")
if manifest["views_per_target"] != 6 or len(manifest["by_mode"]) != 6:
    raise SystemExit(f"error-filtered mode closure drifted: {manifest['by_mode']}")
if not 0 < manifest["error_rows"] < manifest["input_rows"]:
    raise SystemExit(f"unexpected retained error count: {manifest['error_rows']}")
if manifest["targets_with_errors"] <= 0 or manifest["negatives_per_error_row"] != 10:
    raise SystemExit(f"error-filtered candidate closure drifted: {manifest}")
print(
    json.dumps(
        {
            "error_rows": manifest["error_rows"],
            "gold_top1_rows": manifest["gold_top1_rows"],
            "targets_with_errors": manifest["targets_with_errors"],
            "by_mode": manifest["by_mode"],
        },
        sort_keys=True,
    )
)
PY

echo "diverse coverage mining complete filter_manifest=${FILTER_ROOT}/manifest.json"
