#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_OUTPUT_BASE=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
OUTPUT_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/a800_legacy_ce_prefix_ranking_loss_20260811_v1"
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

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite CE-prefix output: ${OUTPUT_ROOT}" >&2
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

TEMP_ROOT="$(mktemp -d /tmp/skillgr-ce-prefix.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-ce-prefix.*) rm -rf -- "${TEMP_ROOT}" ;;
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
export PYTHONPATH="${CODE_ROOT}:${PY310_SITE}:${PY311_SITE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

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
    raise SystemExit(f"CE-prefix runtime mismatch: expected={expected}, actual={actual}")
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
  "${CODE_ROOT}/scripts/analyze_legacy_ce_prefix_ranking.py" \
  "${EVALUATOR}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m unittest \
  tests.test_legacy_ce_prefix_ranking \
  tests.test_analyze_legacy_ce_prefix_ranking \
  tests.test_legacy_c1_hard_negative_coverage -v

declare -a PIDS=()
declare -A LABEL_BY_PID=()
terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait || true
}
trap 'terminate_children; cleanup; exit 130' INT TERM

run_arm() {
  local gpu="$1" label="$2" role="$3" scope="$4" weight="$5"
  local arm_root="${OUTPUT_ROOT}/arms/${label}"
  local train_dir="${arm_root}/train"
  local eval_dir="${arm_root}/eval/official_core75"

  exec > >(tee "${OUTPUT_ROOT}/logs/${label}.log") 2>&1
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export CUBLAS_WORKSPACE_CONFIG=:16:8
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/legacy_full_query_optimizer_lr.py" \
    --arm "${label}" --role "${role}" --path workspace_16 \
    --optimizer-mode carry --query-lr 0.0003 \
    --prefix-rank-scope "${scope}" --prefix-rank-weight "${weight}" \
    --prefix-rank-temperature 1.0 \
    --prefix-rank-summary "${CANDIDATE_SUMMARY}" \
    --prefix-rank-summary-sha256 "${CANDIDATE_SUMMARY_SHA256}" \
    --prefix-rank-manifest "${CANDIDATE_MANIFEST}" \
    --prefix-rank-manifest-sha256 "${CANDIDATE_MANIFEST_SHA256}" \
    --pipeline "${PIPELINE}" --rows "${SKILLGR_ROWS}" --dev-rows "${SKILLGR_DEV_ROWS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --model "${MODEL_SAFE}" --output-dir "${arm_root}" \
    --catalog-steps 2730 --query-steps 38805 --eval-steps 200 \
    --batch-size 16 --max-source-length 2048 --seed 20260626 --dropout-rate 0.1 \
    --cublas-workspace-config :16:8
  cp "${arm_root}/full_query_arm.json" "${train_dir}/final/training_provenance.json"

  unset CUBLAS_WORKSPACE_CONFIG
  "${PYTHON_BIN}" "${EVALUATOR}" eval-single-sid-decoder \
    --rows "${SKILLGR_OFFICIAL_ROWS}" \
    --model-dir "${train_dir}/final" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${eval_dir}" \
    --split-name official_core75 \
    --max-source-length 2048 \
    --num-beams 50 \
    --rank-aggregation beam_order \
    --decoder-head ce \
    --score-transform logprob \
    --k-values 1,10,50 \
    --log-every 25

  local subset
  for subset in all single multi; do
    "${PYTHON_BIN}" "${EVALUATOR}" rescore-ranked \
      --predictions "${eval_dir}/predictions.jsonl" \
      --subset "${subset}" \
      --k-values 1,10,50 \
      --output "${eval_dir}/ranked_metrics_${subset}.json"
  done

  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/analyze_legacy_ce_prefix_ranking.py" arm \
    --arm-root "${arm_root}" \
    --archived-official "${SKILLGR_ARCHIVED_OFFICIAL}" \
    --evaluator "${EVALUATOR}"
  touch "${arm_root}/arm.done"
}

launch_arm() {
  local gpu="$1" label="$2" role="$3" scope="$4" weight="$5"
  (run_arm "${gpu}" "${label}" "${role}" "${scope}" "${weight}") &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}"
  echo "launched ${label} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_arm 0 control_a anchor disabled 0
launch_arm 1 control_b replica disabled 0
launch_arm 2 cum_002 selection cumulative 0.02
launch_arm 3 cum_005 selection cumulative 0.05
launch_arm 4 cum_010 selection cumulative 0.10
launch_arm 5 cum_020 selection cumulative 0.20
launch_arm 6 c1_010 selection c1_only 0.10
launch_arm 7 seq_010 selection sequence_only 0.10

remaining=${#PIDS[@]}
failures=0
while (( remaining > 0 )); do
  finished_pid=""
  if wait -n -p finished_pid; then
    echo "completed ${LABEL_BY_PID[${finished_pid}]}"
  else
    status=$?
    echo "failed ${LABEL_BY_PID[${finished_pid}]:-${finished_pid:-unknown}} with status ${status}" >&2
    failures=$((failures + 1))
  fi
  if [[ -n "${finished_pid}" ]]; then unset 'LABEL_BY_PID[$finished_pid]'; fi
  remaining=$((remaining - 1))
done
trap - INT TERM

if (( failures > 0 )); then
  echo "CE-prefix ranking sweep has ${failures} failed arms" >&2
  exit 1
fi

set +e
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/analyze_legacy_ce_prefix_ranking.py" matrix \
  --root "${OUTPUT_ROOT}"
matrix_status=$?
set -e
if (( matrix_status == 3 )); then
  touch "${OUTPUT_ROOT}/ce_prefix_ranking.no_qualified_arm"
  echo "sweep completed with no arm satisfying all selection gates"
  exit 0
fi
if (( matrix_status != 0 )); then
  exit "${matrix_status}"
fi
touch "${OUTPUT_ROOT}/ce_prefix_ranking.done"
