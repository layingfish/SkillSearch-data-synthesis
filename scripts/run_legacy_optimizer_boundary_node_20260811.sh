#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_OUTPUT_BASE=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
OUTPUT_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/legacy_a800_optimizer_boundary_20260811_v2"
PY310_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1
PY310_RUNTIME_ARCHIVE="${PY310_RUNTIME_ROOT}/runtime.tar.zst"
PY310_RUNTIME_ARCHIVE_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
PY311_SITE_ARCHIVE="${PY311_RUNTIME_ROOT}/site.tar.zst"
PY311_SITE_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
BASE_MODEL_SHA256=a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4
PIPELINE_SHA256=7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite optimizer-boundary output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/arms"
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

TEMP_ROOT="$(mktemp -d /tmp/skillgr-optimizer-boundary.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-optimizer-boundary.*) rm -rf -- "${TEMP_ROOT}" ;;
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
    raise SystemExit(f"optimizer-boundary runtime mismatch: expected={expected}, actual={actual}")
print(json.dumps({"runtime": actual, "torch_config": torch.__config__.show()}, sort_keys=True))
PY

MODEL_SAFE="${TEMP_ROOT}/model-safetensors"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done
check_sha256 "${BASE_MODEL}/model.safetensors" "${BASE_MODEL_SHA256}"

PIPELINE="${CODE_ROOT}/repro/20260714_letter_base_single_sid/skill_gr_pipeline.original_20260714.py"
check_sha256 "${PIPELINE}" "${PIPELINE_SHA256}"

set -a
export SKILLGR_EXACT_MODEL="${MODEL_SAFE}"
source "${CODE_ROOT}/configs/legacy_catalog_base_exact_runtime_20260808.env"
set +a
for required in "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SKILLGR_SKILLS}" "${SKILLGR_SEMANTIC_IDS}"; do
  [[ -f "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done
check_sha256 "${SKILLGR_ROWS}" "${SKILLGR_ROWS_SHA256}"
check_sha256 "${SKILLGR_DEV_ROWS}" "${SKILLGR_DEV_ROWS_SHA256}"
check_sha256 "${SKILLGR_SKILLS}" "${SKILLGR_SKILLS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"

"${PYTHON_BIN}" -m py_compile \
  "${CODE_ROOT}/scripts/legacy_optimizer_boundary_probe.py" \
  "${CODE_ROOT}/scripts/analyze_legacy_optimizer_boundary_matrix.py"

declare -a PIDS=()
declare -A LABEL_BY_PID=()
terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait || true
}
trap 'terminate_children; cleanup; exit 130' INT TERM

launch_arm() {
  local gpu="$1" label="$2" path="$3" mode="$4" query_lr="$5" workspace="$6"
  local output="${OUTPUT_ROOT}/arms/${label}"
  (
    exec > >(tee "${OUTPUT_ROOT}/logs/${label}.log") 2>&1
    export CUDA_VISIBLE_DEVICES="${gpu}"
    if [[ "${workspace}" == "auto" ]]; then
      unset CUBLAS_WORKSPACE_CONFIG || true
    else
      export CUBLAS_WORKSPACE_CONFIG="${workspace}"
    fi
    "${PYTHON_BIN}" "${CODE_ROOT}/scripts/legacy_optimizer_boundary_probe.py" \
      --arm "${label}" --path "${path}" --optimizer-mode "${mode}" --query-lr "${query_lr}" \
      --pipeline "${PIPELINE}" --rows "${SKILLGR_ROWS}" --dev-rows "${SKILLGR_DEV_ROWS}" \
      --skills-jsonl "${SKILLGR_SKILLS}" --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
      --model "${MODEL_SAFE}" --output-dir "${output}" \
      --catalog-steps 2730 --query-steps 50 --batch-size 16 --max-source-length 2048 \
      --seed 20260626 --dropout-rate 0.1 --cublas-workspace-config "${workspace}"
  ) &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}"
  echo "launched ${label} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_arm 0 default_carry_lr2e4 default carry 0.0002 auto
launch_arm 1 workspace_16_carry_lr2e4 workspace_16 carry 0.0002 :16:8
launch_arm 2 default_reset_lr2e4 default reset 0.0002 auto
launch_arm 3 workspace_16_reset_lr2e4 workspace_16 reset 0.0002 :16:8
launch_arm 4 default_carry_lr3e4 default carry 0.0003 auto
launch_arm 5 workspace_16_carry_lr3e4 workspace_16 carry 0.0003 :16:8
launch_arm 6 default_reset_lr3e4 default reset 0.0003 auto
launch_arm 7 workspace_16_reset_lr3e4 workspace_16 reset 0.0003 :16:8

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
  echo "optimizer-boundary matrix has ${failures} failed arms" >&2
  exit 1
fi

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/analyze_legacy_optimizer_boundary_matrix.py" \
  --root "${OUTPUT_ROOT}"
touch "${OUTPUT_ROOT}/optimizer_boundary.done"
