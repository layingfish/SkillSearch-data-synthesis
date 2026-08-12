#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_OUTPUT_BASE=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
OUTPUT_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/a800_legacy_core75_checkpoint_oracle_20260812_v1"
HISTORICAL_ARM_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/a800_legacy_conditional_prefix_hybrid_20260812_v1/arms/control_a"
HISTORICAL_ARM_SHA256=64d31c5edb39aaabe0c085b30d1e19ade43ccab79cbc241b3b654ca85646fd8b
HISTORICAL_RESULT_SHA256=996243be55e12fa53bcade55654bc46459491079cbab2842f37f7d83e9e22ec8
PY310_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1
PY310_RUNTIME_ARCHIVE="${PY310_RUNTIME_ROOT}/runtime.tar.zst"
PY310_RUNTIME_ARCHIVE_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
PY311_SITE_ARCHIVE="${PY311_RUNTIME_ROOT}/site.tar.zst"
PY311_SITE_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
BASE_MODEL_SHA256=a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4
PIPELINE_SHA256=7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76
CANDIDATE_ROOT=/models_rw/SkillSearch-data-synthesis/experiments/legacy_base_c1_hard_negative_coverage_20260811_v1
CANDIDATE_SUMMARY="${CANDIDATE_ROOT}/c1_hard_negative_coverage_summary.json"
CANDIDATE_SUMMARY_SHA256=72b0f525d6c3d9feabc7770c054ee70d8478f1b280d1a100793da8d5b6c6dd6d
CANDIDATE_MANIFEST="${CANDIDATE_ROOT}/c1_hard_negative_candidates.jsonl"
CANDIDATE_MANIFEST_SHA256=8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d
ORACLE_MAX_QUEUED=7
MIN_TMP_AVAILABLE_BYTES=$((20 * 1024 * 1024 * 1024))

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite checkpoint-oracle output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${HISTORICAL_ARM_ROOT}/full_query_arm.json" || ! -f "${HISTORICAL_ARM_ROOT}/result_summary.json" ]]; then
  echo "historical strict-base control artifact is incomplete: ${HISTORICAL_ARM_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/controls"
exec > >(tee "${OUTPUT_ROOT}/logs/node.log") 2>&1
echo "launch_commit=${SKILLGR_LAUNCH_COMMIT:-unknown}"

check_sha256() {
  local path="$1" expected="$2" observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || {
    echo "SHA-256 mismatch for ${path}: expected=${expected}, observed=${observed}" >&2
    exit 1
  }
}

TEMP_ROOT="$(mktemp -d /tmp/skillgr-core75-ckpt-oracle.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-core75-ckpt-oracle.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

check_sha256 "${PY310_RUNTIME_ARCHIVE}" "${PY310_RUNTIME_ARCHIVE_SHA256}"
check_sha256 "${PY311_SITE_ARCHIVE}" "${PY311_SITE_ARCHIVE_SHA256}"
check_sha256 "${HISTORICAL_ARM_ROOT}/full_query_arm.json" "${HISTORICAL_ARM_SHA256}"
check_sha256 "${HISTORICAL_ARM_ROOT}/result_summary.json" "${HISTORICAL_RESULT_SHA256}"
mkdir -p "${TEMP_ROOT}/py310" "${TEMP_ROOT}/py311"
tar --zstd -xf "${PY310_RUNTIME_ARCHIVE}" -C "${TEMP_ROOT}/py310"
tar --zstd -xf "${PY311_SITE_ARCHIVE}" -C "${TEMP_ROOT}/py311"
PYTHON_BIN="${TEMP_ROOT}/py310/python/bin/python3"
PY310_SITE="${TEMP_ROOT}/py310/site"
PY311_SITE="${TEMP_ROOT}/py311/site"
NVIDIA_LIBRARY_PATHS="$(find "${PY311_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
[[ -n "${NVIDIA_LIBRARY_PATHS}" ]] || { echo "fixed NVIDIA libraries are missing" >&2; exit 1; }
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${CODE_ROOT}:${PY310_SITE}:${PY311_SITE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

available_tmp_bytes="$(df --output=avail -B1 "${TEMP_ROOT}" | tail -n 1 | tr -d ' ')"
if [[ ! "${available_tmp_bytes}" =~ ^[0-9]+$ || "${available_tmp_bytes}" -lt "${MIN_TMP_AVAILABLE_BYTES}" ]]; then
  echo "insufficient node-local storage: available=${available_tmp_bytes}, required=${MIN_TMP_AVAILABLE_BYTES}" >&2
  exit 1
fi
echo "tmp_available_bytes=${available_tmp_bytes} required_bytes=${MIN_TMP_AVAILABLE_BYTES}"

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
    raise SystemExit(f"checkpoint-oracle runtime mismatch: expected={expected}, actual={actual}")
print(json.dumps({"runtime": actual, "torch_config": torch.__config__.show()}, sort_keys=True))
PY

MODEL_SAFE="${OUTPUT_ROOT}/controls/t5-base-safetensors-only"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done
check_sha256 "${BASE_MODEL}/model.safetensors" "${BASE_MODEL_SHA256}"

PIPELINE="${CODE_ROOT}/repro/20260714_letter_base_single_sid/skill_gr_pipeline.original_20260714.py"
EVALUATOR="${CODE_ROOT}/scripts/skill_gr_pipeline.py"
ORACLE="${CODE_ROOT}/scripts/legacy_core75_checkpoint_oracle.py"
check_sha256 "${PIPELINE}" "${PIPELINE_SHA256}"
check_sha256 "${CANDIDATE_SUMMARY}" "${CANDIDATE_SUMMARY_SHA256}"
check_sha256 "${CANDIDATE_MANIFEST}" "${CANDIDATE_MANIFEST_SHA256}"

set -a
export SKILLGR_EXACT_MODEL="${MODEL_SAFE}"
source "${CODE_ROOT}/configs/legacy_catalog_base_exact_runtime_20260808.env"
set +a
for required in \
  "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SKILLGR_OFFICIAL_ROWS}" \
  "${SKILLGR_SKILLS}" "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_ARCHIVED_OFFICIAL}"; do
  [[ -f "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done
check_sha256 "${SKILLGR_ROWS}" "${SKILLGR_ROWS_SHA256}"
check_sha256 "${SKILLGR_DEV_ROWS}" "${SKILLGR_DEV_ROWS_SHA256}"
check_sha256 "${SKILLGR_OFFICIAL_ROWS}" "${SKILLGR_OFFICIAL_ROWS_SHA256}"
check_sha256 "${SKILLGR_SKILLS}" "${SKILLGR_SKILLS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"
check_sha256 "${SKILLGR_ARCHIVED_OFFICIAL}" "${SKILLGR_ARCHIVED_OFFICIAL_SHA256}"

"${PYTHON_BIN}" -m py_compile \
  "${CODE_ROOT}/scripts/legacy_full_query_optimizer_lr.py" \
  "${ORACLE}" \
  "${EVALUATOR}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m unittest \
  tests.test_legacy_full_query_optimizer_lr \
  tests.test_legacy_core75_checkpoint_oracle -v

QUEUE_ROOT="${TEMP_ROOT}/oracle_queue"
mkdir -p "${QUEUE_ROOT}"
TRAIN_ARM_ROOT="${OUTPUT_ROOT}/train_arm"

declare -a WORKER_PIDS=()
terminate_workers() {
  local pid
  for pid in "${WORKER_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait || true
}
trap 'terminate_workers; cleanup; exit 130' INT TERM

run_worker() {
  local gpu="$1" worker_id="$2"
  exec > >(tee "${OUTPUT_ROOT}/logs/worker_${worker_id}.log") 2>&1
  export CUDA_VISIBLE_DEVICES="${gpu}"
  "${PYTHON_BIN}" "${ORACLE}" worker \
    --worker-id "${worker_id}" \
    --queue-root "${QUEUE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --evaluator "${EVALUATOR}" \
    --rows "${SKILLGR_OFFICIAL_ROWS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --anchor-step 38200 \
    --poll-seconds 0.5
}

for worker_id in 1 2 3 4 5 6 7; do
  (run_worker "${worker_id}" "${worker_id}") &
  WORKER_PIDS+=("$!")
  echo "launched oracle evaluator worker=${worker_id} on CUDA_VISIBLE_DEVICES=${worker_id}"
done

training_status=0
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:16:8 \
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/legacy_full_query_optimizer_lr.py" \
    --arm control_core75_checkpoint_oracle --role anchor --path workspace_16 \
    --optimizer-mode carry --query-lr 0.0003 \
    --prefix-rank-grouping fixed --prefix-rank-scope disabled --prefix-rank-weight 0 \
    --prefix-rank-single-scope sequence_only --prefix-rank-single-weight 0 \
    --prefix-rank-multi-scope cumulative --prefix-rank-multi-weight 0 \
    --prefix-rank-temperature 1.0 \
    --prefix-rank-summary "${CANDIDATE_SUMMARY}" \
    --prefix-rank-summary-sha256 "${CANDIDATE_SUMMARY_SHA256}" \
    --prefix-rank-manifest "${CANDIDATE_MANIFEST}" \
    --prefix-rank-manifest-sha256 "${CANDIDATE_MANIFEST_SHA256}" \
    --pipeline "${PIPELINE}" --rows "${SKILLGR_ROWS}" --dev-rows "${SKILLGR_DEV_ROWS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --model "${MODEL_SAFE}" --output-dir "${TRAIN_ARM_ROOT}" \
    --catalog-steps 2730 --query-steps 38805 --eval-steps 200 \
    --oracle-checkpoint-queue "${QUEUE_ROOT}" --oracle-max-queued "${ORACLE_MAX_QUEUED}" \
    --batch-size 16 --max-source-length 2048 --seed 20260626 --dropout-rate 0.1 \
    --cublas-workspace-config :16:8 || training_status=$?

if (( training_status != 0 )); then
  "${PYTHON_BIN}" - "${QUEUE_ROOT}/producer.failed.json" "${training_status}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps({"training_status": int(sys.argv[2])}, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  terminate_workers
  echo "checkpoint-oracle training failed with status ${training_status}" >&2
  exit "${training_status}"
fi
touch "${QUEUE_ROOT}/producer.done"

worker_failures=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    worker_failures=$((worker_failures + 1))
  fi
done
trap - INT TERM
if (( worker_failures > 0 )); then
  echo "checkpoint-oracle has ${worker_failures} failed evaluator workers" >&2
  exit 1
fi

if compgen -G "${QUEUE_ROOT}/step_*.ready.json" >/dev/null || \
   compgen -G "${QUEUE_ROOT}/step_*.claimed.*.json" >/dev/null || \
   compgen -G "${QUEUE_ROOT}/step_[0-9]*" >/dev/null; then
  echo "checkpoint-oracle queue is not drained" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ORACLE}" analyze \
  --output-root "${OUTPUT_ROOT}" \
  --train-arm-root "${TRAIN_ARM_ROOT}" \
  --historical-arm-root "${HISTORICAL_ARM_ROOT}" \
  --query-steps 38805 --eval-steps 200 --anchor-step 38200

echo "checkpoint-oracle completed: ${OUTPUT_ROOT}/checkpoint_oracle_summary.json"
