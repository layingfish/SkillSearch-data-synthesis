#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_OUTPUT_BASE=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
OUTPUT_ROOT="${SKILLGR_FIXED_POOL_OUTPUT_ROOT:-${PHYSICAL_OUTPUT_BASE}/experiments/legacy_core75_fixed_pool_scoring_20260811_v1}"
PY310_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1
PY310_RUNTIME_ARCHIVE="${PY310_RUNTIME_ROOT}/runtime.tar.zst"
PY310_RUNTIME_ARCHIVE_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
PY311_SITE_ARCHIVE="${PY311_RUNTIME_ROOT}/site.tar.zst"
PY311_SITE_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
BASE_MODEL_SHA256=a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4
BASE_MODEL_CONFIG_SHA256=46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033
TOKENIZER_JSON_SHA256=65430662114922dc57697bc782fc1122cd004547611abfcd520fe1a50f125bd9
RTX_TOKENIZER_CONFIG_SHA256=b9bd8a1f615bd012bfa742dbe9f303a5c6fffb10d9b48e07d1759912dd0701bb
A800_TOKENIZER_CONFIG_SHA256=ffad9890875f013e214d6f17a54dbb2a8e0870b5c568bdb591be770c2ccf2524

RTX_CHECKPOINT=/models_rw/SkillSearch-data-synthesis/assets-models/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/single_sid_d4_lightgcn_d32_letterofficial_task5_core75_gt2_t5base_cat3_querye5_b16/final
RTX_CHECKPOINT_SHA256=b92daf62e604f35dbecbb5c80897de9d212995d13c312f0f149f781caf5c0848
STABILITY_ROOT=/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_query_stability_20260811_v1
CONTROL_CHECKPOINT="${STABILITY_ROOT}/arms/workspace_16_control/train/final"
CONTROL_CHECKPOINT_SHA256=2362f0567854c0134f748dd44de3d284224057a7c93751c31b66594ae2662b87
WARMUP_CHECKPOINT="${STABILITY_ROOT}/arms/workspace_16_warmup500/train/final"
WARMUP_CHECKPOINT_SHA256=b76389494d94c966d53f2e9701993ba295da32034f8bd23dcea46ba95109e74b
CLIP_CHECKPOINT="${STABILITY_ROOT}/arms/workspace_16_clip1/train/final"
CLIP_CHECKPOINT_SHA256=22728b8d401d0e56843f85faab98416b7f530c55d718585d0f2bafb75a7240c5

RTX_PREDICTIONS=/models_rw/SkillSearch-data-synthesis/assets-models/outputs/eval/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714/core75_beam50/predictions.jsonl
RTX_PREDICTIONS_SHA256=167426964ab2938889526128a8119f39043a3d788d590105cd4a3be6cc31cf88
CONTROL_PREDICTIONS="${STABILITY_ROOT}/arms/workspace_16_control/eval/official_core75/predictions.jsonl"
CONTROL_PREDICTIONS_SHA256=3663a15ad9859abc76141d815f0da3c88bd9b3d5d9e914f09a3d8b0a3ed843f4
WARMUP_PREDICTIONS="${STABILITY_ROOT}/arms/workspace_16_warmup500/eval/official_core75/predictions.jsonl"
WARMUP_PREDICTIONS_SHA256=38cd43f3cdbfbad1f8d8d25f8c33ed5865ae1aa3a1f63585e868253ad99c6b54
CLIP_PREDICTIONS="${STABILITY_ROOT}/arms/workspace_16_clip1/eval/official_core75/predictions.jsonl"
CLIP_PREDICTIONS_SHA256=e205938980fcf2e5f11ccbdc507dd0340cfddbdec2110d834466a17b266fa91c

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite fixed-pool output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/shards"
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

TEMP_ROOT="$(mktemp -d /tmp/skillgr-fixed-pool.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-fixed-pool.*) rm -rf -- "${TEMP_ROOT}" ;;
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
export MKL_NUM_THREADS=4
unset CUBLAS_WORKSPACE_CONFIG || true

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
    raise SystemExit(f"fixed-pool runtime mismatch: expected={expected}, actual={actual}")
gpu_names = [torch.cuda.get_device_name(index) for index in range(actual["gpu_count"])]
if any("A800" not in name for name in gpu_names):
    raise SystemExit(f"fixed-pool GPU model mismatch: expected=A800, actual={gpu_names}")
print(
    json.dumps(
        {
            "runtime": actual,
            "gpu_names": gpu_names,
            "torch_config": torch.__config__.show(),
        },
        sort_keys=True,
    )
)
PY

set -a
export SKILLGR_EXACT_MODEL="${BASE_MODEL}"
source "${CODE_ROOT}/configs/legacy_catalog_base_exact_runtime_20260808.env"
set +a

check_sha256 "${BASE_MODEL}/model.safetensors" "${BASE_MODEL_SHA256}"
check_sha256 "${BASE_MODEL}/config.json" "${BASE_MODEL_CONFIG_SHA256}"
check_sha256 "${SKILLGR_OFFICIAL_ROWS}" "${SKILLGR_OFFICIAL_ROWS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"
check_sha256 "${RTX_CHECKPOINT}/structured_decoder.pt" "${RTX_CHECKPOINT_SHA256}"
check_sha256 "${CONTROL_CHECKPOINT}/structured_decoder.pt" "${CONTROL_CHECKPOINT_SHA256}"
check_sha256 "${WARMUP_CHECKPOINT}/structured_decoder.pt" "${WARMUP_CHECKPOINT_SHA256}"
check_sha256 "${CLIP_CHECKPOINT}/structured_decoder.pt" "${CLIP_CHECKPOINT_SHA256}"
check_sha256 "${RTX_CHECKPOINT}/tokenizer.json" "${TOKENIZER_JSON_SHA256}"
check_sha256 "${CONTROL_CHECKPOINT}/tokenizer.json" "${TOKENIZER_JSON_SHA256}"
check_sha256 "${WARMUP_CHECKPOINT}/tokenizer.json" "${TOKENIZER_JSON_SHA256}"
check_sha256 "${CLIP_CHECKPOINT}/tokenizer.json" "${TOKENIZER_JSON_SHA256}"
check_sha256 "${RTX_CHECKPOINT}/tokenizer_config.json" "${RTX_TOKENIZER_CONFIG_SHA256}"
check_sha256 "${CONTROL_CHECKPOINT}/tokenizer_config.json" "${A800_TOKENIZER_CONFIG_SHA256}"
check_sha256 "${WARMUP_CHECKPOINT}/tokenizer_config.json" "${A800_TOKENIZER_CONFIG_SHA256}"
check_sha256 "${CLIP_CHECKPOINT}/tokenizer_config.json" "${A800_TOKENIZER_CONFIG_SHA256}"
check_sha256 "${RTX_PREDICTIONS}" "${RTX_PREDICTIONS_SHA256}"
check_sha256 "${CONTROL_PREDICTIONS}" "${CONTROL_PREDICTIONS_SHA256}"
check_sha256 "${WARMUP_PREDICTIONS}" "${WARMUP_PREDICTIONS_SHA256}"
check_sha256 "${CLIP_PREDICTIONS}" "${CLIP_PREDICTIONS_SHA256}"

"${PYTHON_BIN}" -m py_compile \
  "${CODE_ROOT}/scripts/score_legacy_core75_fixed_pool.py" \
  "${CODE_ROOT}/scripts/analyze_legacy_core75_fixed_pool.py" \
  "${CODE_ROOT}/scripts/analyze_legacy_core75_position_scores.py"
"${PYTHON_BIN}" -m unittest discover -s tests -p 'test_legacy_core75_fixed_pool.py'
"${PYTHON_BIN}" -m unittest discover -s tests -p 'test_legacy_core75_position_scores.py'

declare -a PIDS=()
declare -A LABEL_BY_PID=()
terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait || true
}
trap 'terminate_children; cleanup; exit 130' INT TERM

run_shard() {
  local gpu="$1" label="$2" checkpoint="$3" checkpoint_sha="$4" shard_id="$5"
  local shard_root="${OUTPUT_ROOT}/shards/${label}/shard_${shard_id}"
  exec > >(tee "${OUTPUT_ROOT}/logs/${label}_shard_${shard_id}.log") 2>&1
  export CUDA_VISIBLE_DEVICES="${gpu}"
  unset CUBLAS_WORKSPACE_CONFIG || true
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/score_legacy_core75_fixed_pool.py" \
    --label "${label}" \
    --checkpoint-dir "${checkpoint}" \
    --checkpoint-sha256 "${checkpoint_sha}" \
    --base-model "${BASE_MODEL}" \
    --rows "${SKILLGR_OFFICIAL_ROWS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --predictions "rtx_archived=${RTX_PREDICTIONS}" \
    --predictions "workspace_control=${CONTROL_PREDICTIONS}" \
    --predictions "workspace_warmup500=${WARMUP_PREDICTIONS}" \
    --predictions "workspace_clip1=${CLIP_PREDICTIONS}" \
    --shard-id "${shard_id}" --num-shards 2 \
    --expected-rows 75 --prediction-depth 50 --max-source-length 2048 \
    --score-atol 0.00005 \
    --output-dir "${shard_root}"
}

launch_shard() {
  local gpu="$1" label="$2" checkpoint="$3" checkpoint_sha="$4" shard_id="$5"
  (
    run_shard "${gpu}" "${label}" "${checkpoint}" "${checkpoint_sha}" "${shard_id}"
  ) &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}/shard_${shard_id}"
  echo "launched ${label}/shard_${shard_id} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_shard 0 rtx_archived "${RTX_CHECKPOINT}" "${RTX_CHECKPOINT_SHA256}" 0
launch_shard 1 rtx_archived "${RTX_CHECKPOINT}" "${RTX_CHECKPOINT_SHA256}" 1
launch_shard 2 workspace_control "${CONTROL_CHECKPOINT}" "${CONTROL_CHECKPOINT_SHA256}" 0
launch_shard 3 workspace_control "${CONTROL_CHECKPOINT}" "${CONTROL_CHECKPOINT_SHA256}" 1
launch_shard 4 workspace_warmup500 "${WARMUP_CHECKPOINT}" "${WARMUP_CHECKPOINT_SHA256}" 0
launch_shard 5 workspace_warmup500 "${WARMUP_CHECKPOINT}" "${WARMUP_CHECKPOINT_SHA256}" 1
launch_shard 6 workspace_clip1 "${CLIP_CHECKPOINT}" "${CLIP_CHECKPOINT_SHA256}" 0
launch_shard 7 workspace_clip1 "${CLIP_CHECKPOINT}" "${CLIP_CHECKPOINT_SHA256}" 1

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
  echo "fixed-pool scoring has ${failures} failed shards" >&2
  exit 1
fi

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/analyze_legacy_core75_fixed_pool.py" \
  --shard "rtx_archived=${OUTPUT_ROOT}/shards/rtx_archived/shard_0" \
  --shard "rtx_archived=${OUTPUT_ROOT}/shards/rtx_archived/shard_1" \
  --shard "workspace_control=${OUTPUT_ROOT}/shards/workspace_control/shard_0" \
  --shard "workspace_control=${OUTPUT_ROOT}/shards/workspace_control/shard_1" \
  --shard "workspace_warmup500=${OUTPUT_ROOT}/shards/workspace_warmup500/shard_0" \
  --shard "workspace_warmup500=${OUTPUT_ROOT}/shards/workspace_warmup500/shard_1" \
  --shard "workspace_clip1=${OUTPUT_ROOT}/shards/workspace_clip1/shard_0" \
  --shard "workspace_clip1=${OUTPUT_ROOT}/shards/workspace_clip1/shard_1" \
  --reference-label rtx_archived --expected-rows 75 --expected-shards 2 \
  --output-root "${OUTPUT_ROOT}"

if [[ "${SKILLGR_POSITION_ANALYSIS:-0}" == "1" ]]; then
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/analyze_legacy_core75_position_scores.py" \
    --shard "rtx_archived=${OUTPUT_ROOT}/shards/rtx_archived/shard_0" \
    --shard "rtx_archived=${OUTPUT_ROOT}/shards/rtx_archived/shard_1" \
    --shard "workspace_control=${OUTPUT_ROOT}/shards/workspace_control/shard_0" \
    --shard "workspace_control=${OUTPUT_ROOT}/shards/workspace_control/shard_1" \
    --shard "workspace_warmup500=${OUTPUT_ROOT}/shards/workspace_warmup500/shard_0" \
    --shard "workspace_warmup500=${OUTPUT_ROOT}/shards/workspace_warmup500/shard_1" \
    --shard "workspace_clip1=${OUTPUT_ROOT}/shards/workspace_clip1/shard_0" \
    --shard "workspace_clip1=${OUTPUT_ROOT}/shards/workspace_clip1/shard_1" \
    --reference-label rtx_archived --expected-rows 75 --expected-shards 2 \
    --output-root "${OUTPUT_ROOT}"
fi
