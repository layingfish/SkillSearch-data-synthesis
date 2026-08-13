#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_OUTPUT_BASE=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
OUTPUT_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/legacy_core75_full_catalog_generator_scoring_20260813_v1"
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
BASE_MODEL_SHA256=a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4
CHECKPOINT_DIR=/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v5/selected/joint_best
CHECKPOINT_SHA256=a2a26ab5f64fbbe5b0df875afccad945684a2b5e2166f71b94a3e299c36f03c1
BEAM_PREDICTIONS=/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v5/oracle/cum_002/steps/step_00028400/predictions.jsonl
BEAM_PREDICTIONS_SHA256=78386bffac0944fed624db2d9948023cc6407fedea32ec988df6d99f66a29a7b
PY310_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1
PY310_RUNTIME_ARCHIVE="${PY310_RUNTIME_ROOT}/runtime.tar.zst"
PY310_RUNTIME_ARCHIVE_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
PY311_SITE_ARCHIVE="${PY311_RUNTIME_ROOT}/site.tar.zst"
PY311_SITE_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
MIN_TMP_AVAILABLE_BYTES=$((100 * 1024 * 1024 * 1024))
EXPECTED_CATALOG_SIZE=14551
FORWARD_BATCH_SIZE=256
BEAM_WIDTH=50

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite full-catalog output: ${OUTPUT_ROOT}" >&2
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

TEMP_ROOT="$(mktemp -d /tmp/skillgr-full-catalog.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-full-catalog.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

check_sha256 "${PY310_RUNTIME_ARCHIVE}" "${PY310_RUNTIME_ARCHIVE_SHA256}"
check_sha256 "${PY311_SITE_ARCHIVE}" "${PY311_SITE_ARCHIVE_SHA256}"
check_sha256 "${BASE_MODEL}/model.safetensors" "${BASE_MODEL_SHA256}"
check_sha256 "${CHECKPOINT_DIR}/structured_decoder.pt" "${CHECKPOINT_SHA256}"
check_sha256 "${BEAM_PREDICTIONS}" "${BEAM_PREDICTIONS_SHA256}"

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
export CUBLAS_WORKSPACE_CONFIG=:16:8

available_tmp_bytes="$(df --output=avail -B1 "${TEMP_ROOT}" | tail -n 1 | tr -d ' ')"
if [[ ! "${available_tmp_bytes}" =~ ^[0-9]+$ || "${available_tmp_bytes}" -lt "${MIN_TMP_AVAILABLE_BYTES}" ]]; then
  echo "insufficient node-local storage: available=${available_tmp_bytes}, required=${MIN_TMP_AVAILABLE_BYTES}" >&2
  exit 1
fi

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
gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
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
    raise SystemExit(f"full-catalog runtime mismatch: expected={expected}, actual={actual}")
if any("A800" not in name for name in gpu_names):
    raise SystemExit(f"expected eight A800 GPUs, got={gpu_names}")
print(json.dumps({"runtime": actual, "gpu_names": gpu_names}, sort_keys=True))
PY

set -a
export SKILLGR_EXACT_MODEL="${BASE_MODEL}"
source "${CODE_ROOT}/configs/legacy_catalog_base_exact_runtime_20260808.env"
set +a
for required in \
  "${SKILLGR_OFFICIAL_ROWS}" "${SKILLGR_SEMANTIC_IDS}"; do
  [[ -f "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done
check_sha256 "${SKILLGR_OFFICIAL_ROWS}" "${SKILLGR_OFFICIAL_ROWS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"

SCORER="${CODE_ROOT}/scripts/score_legacy_core75_full_catalog.py"
"${PYTHON_BIN}" -m py_compile \
  "${SCORER}" \
  "${CODE_ROOT}/scripts/score_legacy_core75_fixed_pool.py" \
  "${CODE_ROOT}/scripts/skill_gr_pipeline.py"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m unittest \
  tests.test_legacy_core75_fixed_pool \
  tests.test_legacy_core75_full_catalog -v

declare -a SHARD_PIDS=()
launch_shard() {
  local gpu="$1"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCORER}" score \
      --checkpoint-dir "${CHECKPOINT_DIR}" \
      --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
      --base-model "${BASE_MODEL}" \
      --rows "${SKILLGR_OFFICIAL_ROWS}" \
      --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
      --beam-predictions "${BEAM_PREDICTIONS}" \
      --shard-id "${gpu}" --num-shards 8 \
      --expected-rows 75 --expected-catalog-size "${EXPECTED_CATALOG_SIZE}" \
      --beam-width "${BEAM_WIDTH}" --max-source-length 2048 \
      --forward-batch-size "${FORWARD_BATCH_SIZE}" --store-top-k 100 \
      --score-atol 5e-5 \
      --output-dir "${OUTPUT_ROOT}/shards/shard_${gpu}"
  ) > "${OUTPUT_ROOT}/logs/shard_${gpu}.log" 2>&1 &
  SHARD_PIDS+=("$!")
}

launch_shard 0
launch_shard 1
launch_shard 2
launch_shard 3
launch_shard 4
launch_shard 5
launch_shard 6
launch_shard 7

for pid in "${SHARD_PIDS[@]}"; do
  if wait "${pid}"; then
    continue
  else
    status=$?
    for active_pid in "${SHARD_PIDS[@]}"; do kill "${active_pid}" 2>/dev/null || true; done
    for active_pid in "${SHARD_PIDS[@]}"; do wait "${active_pid}" 2>/dev/null || true; done
    echo "full-catalog shard failed with status=${status}" >&2
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SCORER}" analyze \
  --input-root "${OUTPUT_ROOT}" \
  --expected-shards 8 --expected-rows 75 --beam-width "${BEAM_WIDTH}"

echo "full-catalog generator scoring completed: ${OUTPUT_ROOT}/full_catalog_summary.json"
