#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_OUTPUT_BASE=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
OUTPUT_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v4"
CONTROL_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/a800_legacy_core75_checkpoint_oracle_20260812_v1"
CONTROL_SUMMARY_SHA256=36ebaa1e7942867ac0a14b233024671d76bb1162e6de1975fd7606307bc85b0a
HISTORICAL_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/a800_legacy_conditional_prefix_hybrid_20260812_v1/arms"
SEQ_HISTORICAL_ROOT="${HISTORICAL_ROOT}/seq_010"
CUM_HISTORICAL_ROOT="${HISTORICAL_ROOT}/cum_002"
SEQ_ARM_SHA256=df60a1cd7020c4c3a5f514cd311a44ad66cdb795ba0b2e79dac9b5ad504364aa
SEQ_RESULT_SHA256=9eb246a965a0714f41f8fc6b7f9f5c32d07ccd8264c6181e6c760bb04d3dc301
SEQ_LOSS_SHA256=a108f3b132fe0fb56223d59628b2d574c2d55bf23be02a248c997e303172f5a6
CUM_ARM_SHA256=5196c5c1952e823d846c0aebb80fc61f005bed4b677567c340027291c8638395
CUM_RESULT_SHA256=85d2081aefac84c4cc52fe75902608e35ac83e356e77193ef5f96d9607465b64
CUM_LOSS_SHA256=c1c6cefc70e3749837c19f7c5fa73e7e300668c07d279a09170ac13d73f5308b
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
MIN_TMP_AVAILABLE_BYTES=$((100 * 1024 * 1024 * 1024))
JOINT_MIN_R10=0.69
JOINT_MIN_R50=0.75
SHARED_EVAL_GPU=7

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite sequence-weight checkpoint sweep: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/arms" "${OUTPUT_ROOT}/controls"
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

TEMP_ROOT="$(mktemp -d /tmp/skillgr-seq-weight-ckpt-sweep.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-seq-weight-ckpt-sweep.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

check_sha256 "${PY310_RUNTIME_ARCHIVE}" "${PY310_RUNTIME_ARCHIVE_SHA256}"
check_sha256 "${PY311_SITE_ARCHIVE}" "${PY311_SITE_ARCHIVE_SHA256}"
check_sha256 "${CONTROL_ROOT}/checkpoint_oracle_summary.json" "${CONTROL_SUMMARY_SHA256}"
check_sha256 "${SEQ_HISTORICAL_ROOT}/full_query_arm.json" "${SEQ_ARM_SHA256}"
check_sha256 "${SEQ_HISTORICAL_ROOT}/result_summary.json" "${SEQ_RESULT_SHA256}"
check_sha256 "${SEQ_HISTORICAL_ROOT}/train/loss_history.jsonl" "${SEQ_LOSS_SHA256}"
check_sha256 "${CUM_HISTORICAL_ROOT}/full_query_arm.json" "${CUM_ARM_SHA256}"
check_sha256 "${CUM_HISTORICAL_ROOT}/result_summary.json" "${CUM_RESULT_SHA256}"
check_sha256 "${CUM_HISTORICAL_ROOT}/train/loss_history.jsonl" "${CUM_LOSS_SHA256}"
mkdir -p "${TEMP_ROOT}/py310" "${TEMP_ROOT}/py311" "${TEMP_ROOT}/queues"
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
    raise SystemExit(f"sequence-weight sweep runtime mismatch: expected={expected}, actual={actual}")
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
SWEEP="${CODE_ROOT}/scripts/legacy_sequence_weight_checkpoint_sweep.py"
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
  "${CODE_ROOT}/scripts/legacy_ce_prefix_ranking.py" \
  "${CODE_ROOT}/scripts/legacy_full_query_optimizer_lr.py" \
  "${CODE_ROOT}/scripts/legacy_core75_checkpoint_oracle.py" \
  "${SWEEP}" \
  "${EVALUATOR}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m unittest \
  tests.test_legacy_ce_prefix_ranking \
  tests.test_legacy_full_query_optimizer_lr \
  tests.test_legacy_core75_checkpoint_oracle \
  tests.test_legacy_sequence_weight_checkpoint_sweep -v

"${PYTHON_BIN}" - "${OUTPUT_ROOT}/sweep_contract.json" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "artifact_format": "legacy_sequence_weight_checkpoint_sweep_contract_v1",
    "launch_commit": os.environ.get("SKILLGR_LAUNCH_COMMIT"),
    "new_arms": [
        {"arm": "cum_002", "scope": "cumulative", "weight": 0.02},
        {"arm": "seq_001", "scope": "sequence_only", "weight": 0.01},
        {"arm": "seq_002", "scope": "sequence_only", "weight": 0.02},
        {"arm": "seq_004", "scope": "sequence_only", "weight": 0.04},
        {"arm": "seq_006", "scope": "sequence_only", "weight": 0.06},
        {"arm": "seq_008", "scope": "sequence_only", "weight": 0.08},
        {"arm": "seq_010", "scope": "sequence_only", "weight": 0.10},
    ],
    "checkpoint_steps": list(range(200, 38801, 200)) + [38805],
    "joint_thresholds": {"r10": 0.69, "r50": 0.75},
    "evaluation": "Official core75=75 queries, beam=50, beam_order, CE log-prob",
    "scheduler": {
        "parallel_trainers": 7,
        "trainer_gpus": [0, 1, 2, 3, 4, 5, 6],
        "shared_evaluator_gpu": 7,
        "post_training_helper_evaluators": True,
    },
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

declare -a BACKGROUND_PIDS=()
terminate_background() {
  local pid
  for pid in "${BACKGROUND_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${BACKGROUND_PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
  BACKGROUND_PIDS=()
}
trap 'terminate_background; cleanup; exit 130' INT TERM

run_worker() {
  local gpu="$1" worker_id="$2" label="$3" queue_root="$4" arm_root="$5"
  shift 5
  export CUDA_VISIBLE_DEVICES="${gpu}"
  "${PYTHON_BIN}" "${SWEEP}" worker \
    --arm "${label}" \
    --worker-id "${worker_id}" \
    --queue-root "${queue_root}" \
    --output-root "${arm_root}" \
    --evaluator "${EVALUATOR}" \
    --rows "${SKILLGR_OFFICIAL_ROWS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --joint-min-r10 "${JOINT_MIN_R10}" \
    --joint-min-r50 "${JOINT_MIN_R50}" \
    --poll-seconds 0.5 \
    "$@"
}

run_training() {
  local gpu="$1" label="$2" scope="$3" weight="$4" arm_root="$5" queue_root="$6"
  CUDA_VISIBLE_DEVICES="${gpu}" CUBLAS_WORKSPACE_CONFIG=:16:8 \
    "${PYTHON_BIN}" "${CODE_ROOT}/scripts/legacy_full_query_optimizer_lr.py" \
      --arm "${label}" --role selection --path workspace_16 \
      --optimizer-mode carry --query-lr 0.0003 \
      --prefix-rank-grouping fixed --prefix-rank-scope "${scope}" --prefix-rank-weight "${weight}" \
      --prefix-rank-single-scope sequence_only --prefix-rank-single-weight 0 \
      --prefix-rank-multi-scope cumulative --prefix-rank-multi-weight 0 \
      --prefix-rank-temperature 1.0 \
      --prefix-rank-summary "${CANDIDATE_SUMMARY}" \
      --prefix-rank-summary-sha256 "${CANDIDATE_SUMMARY_SHA256}" \
      --prefix-rank-manifest "${CANDIDATE_MANIFEST}" \
      --prefix-rank-manifest-sha256 "${CANDIDATE_MANIFEST_SHA256}" \
      --pipeline "${PIPELINE}" --rows "${SKILLGR_ROWS}" --dev-rows "${SKILLGR_DEV_ROWS}" \
      --skills-jsonl "${SKILLGR_SKILLS}" --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
      --model "${MODEL_SAFE}" --output-dir "${arm_root}" \
      --catalog-steps 2730 --query-steps 38805 --eval-steps 200 \
      --oracle-checkpoint-queue "${queue_root}" --oracle-max-queued "${ORACLE_MAX_QUEUED}" \
      --batch-size 16 --max-source-length 2048 --seed 20260626 --dropout-rate 0.1 \
      --cublas-workspace-config :16:8
}

run_arm_pipeline() {
  local gpu="$1" helper_worker_id="$2" label="$3" scope="$4" weight="$5"
  local arm_root="${OUTPUT_ROOT}/arms/${label}"
  local eval_root="${OUTPUT_ROOT}/oracle/${label}"
  local queue_root="${TEMP_ROOT}/queues/${label}"
  local training_status=0

  run_training "${gpu}" "${label}" "${scope}" "${weight}" "${arm_root}" "${queue_root}" \
    > "${OUTPUT_ROOT}/logs/${label}_train.log" 2>&1 || training_status=$?

  if (( training_status != 0 )); then
    "${PYTHON_BIN}" - "${queue_root}/producer.failed.json" "${training_status}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps({"training_status": int(sys.argv[2])}, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
    echo "arm=${label} training failed with status ${training_status}" >&2
    return "${training_status}"
  fi
  touch "${queue_root}/producer.done"
  echo "arm=${label} training complete; GPU ${gpu} joins checkpoint evaluation"
  run_worker "${gpu}" "${helper_worker_id}" "${label}" "${queue_root}" "${eval_root}" \
    --allow-empty > "${OUTPUT_ROOT}/logs/${label}_worker_${helper_worker_id}.log" 2>&1
}

ARM_LABELS=(cum_002 seq_001 seq_002 seq_004 seq_006 seq_008 seq_010)
ARM_SCOPES=(cumulative sequence_only sequence_only sequence_only sequence_only sequence_only sequence_only)
ARM_WEIGHTS=(0.02 0.01 0.02 0.04 0.06 0.08 0.10)
ARM_GPUS=(0 1 2 3 4 5 6)
SHARED_EVAL_LOCK="${TEMP_ROOT}/gpu_${SHARED_EVAL_GPU}_evaluation.lock"

for index in "${!ARM_LABELS[@]}"; do
  label="${ARM_LABELS[index]}"
  queue_root="${TEMP_ROOT}/queues/${label}"
  eval_root="${OUTPUT_ROOT}/oracle/${label}"
  mkdir -p "${queue_root}" "${eval_root}"
  (run_worker "${SHARED_EVAL_GPU}" 8 "${label}" "${queue_root}" "${eval_root}" \
    --allow-empty --eval-lock "${SHARED_EVAL_LOCK}" \
    > "${OUTPUT_ROOT}/logs/${label}_worker_8.log" 2>&1) &
  BACKGROUND_PIDS+=("$!")
  echo "launched arm=${label} shared_evaluator=8 CUDA_VISIBLE_DEVICES=${SHARED_EVAL_GPU}"
done

for index in "${!ARM_LABELS[@]}"; do
  gpu="${ARM_GPUS[index]}"
  helper_worker_id=$((gpu + 1))
  (run_arm_pipeline "${gpu}" "${helper_worker_id}" \
    "${ARM_LABELS[index]}" "${ARM_SCOPES[index]}" "${ARM_WEIGHTS[index]}") &
  BACKGROUND_PIDS+=("$!")
  echo "launched arm=${ARM_LABELS[index]} trainer CUDA_VISIBLE_DEVICES=${gpu}"
done

declare -a ACTIVE_PIDS=("${BACKGROUND_PIDS[@]}")
while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  finished_pid=""
  if wait -n -p finished_pid "${ACTIVE_PIDS[@]}"; then
    status=0
  else
    status=$?
    echo "background process failed: pid=${finished_pid:-unknown} status=${status}" >&2
    terminate_background
    exit "${status}"
  fi
  declare -a remaining=()
  for pid in "${ACTIVE_PIDS[@]}"; do
    [[ "${pid}" == "${finished_pid}" ]] || remaining+=("${pid}")
  done
  ACTIVE_PIDS=("${remaining[@]}")
done
BACKGROUND_PIDS=()

for index in "${!ARM_LABELS[@]}"; do
  label="${ARM_LABELS[index]}"
  scope="${ARM_SCOPES[index]}"
  weight="${ARM_WEIGHTS[index]}"
  arm_root="${OUTPUT_ROOT}/arms/${label}"
  eval_root="${OUTPUT_ROOT}/oracle/${label}"
  queue_root="${TEMP_ROOT}/queues/${label}"
  historical_args=()
  if [[ "${label}" == seq_010 ]]; then
    historical_args=(--historical-arm-root "${SEQ_HISTORICAL_ROOT}")
  elif [[ "${label}" == cum_002 ]]; then
    historical_args=(--historical-arm-root "${CUM_HISTORICAL_ROOT}")
  fi
  if compgen -G "${queue_root}/step_*.ready.json" >/dev/null || \
     compgen -G "${queue_root}/step_*.claimed.*.json" >/dev/null || \
     compgen -G "${queue_root}/step_[0-9]*" >/dev/null; then
    echo "arm=${label} checkpoint queue is not drained" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SWEEP}" analyze-arm \
    --arm "${label}" --scope "${scope}" --weight "${weight}" \
    --arm-root "${arm_root}" --eval-root "${eval_root}" \
    --query-steps 38805 --eval-steps 200 --worker-count 2 \
    --joint-min-r10 "${JOINT_MIN_R10}" --joint-min-r50 "${JOINT_MIN_R50}" \
    "${historical_args[@]}"
  case "${queue_root}" in
    "${TEMP_ROOT}"/queues/*) rm -rf -- "${queue_root}" ;;
    *) echo "refusing to remove unexpected queue path: ${queue_root}" >&2; exit 1 ;;
  esac
  echo "completed arm=${label} summary=${arm_root}/checkpoint_sweep_summary.json"
done
trap - INT TERM

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SWEEP}" analyze-matrix \
  --root "${OUTPUT_ROOT}" \
  --control-root "${CONTROL_ROOT}" \
  --expected-new-arms cum_002,seq_001,seq_002,seq_004,seq_006,seq_008,seq_010 \
  --joint-min-r10 "${JOINT_MIN_R10}" --joint-min-r50 "${JOINT_MIN_R50}"

echo "sequence-weight checkpoint sweep completed: ${OUTPUT_ROOT}/sequence_weight_checkpoint_sweep_summary.json"
