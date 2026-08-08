#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/workspace/SkillSearch-data-synthesis
PHYSICAL_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
RUNTIME_ARCHIVE=${RUNTIME_ROOT}/site.tar.zst
RUNTIME_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
COMPAT_ARCHIVE=${RUNTIME_ROOT}/compat-overlay.tar.zst
COMPAT_ARCHIVE_SHA256=a6b2f56bb9b4496caa4be225b1c2630270cef1b9a23f82013eea72632bda6b20
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1

TEMP_ROOT="$(mktemp -d /tmp/skillgr-base-exact-runtime.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-base-exact-runtime.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

check_sha256() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || {
    echo "SHA-256 mismatch for ${path}: expected ${expected}, observed ${observed}" >&2
    exit 1
  }
}

check_sha256 "${RUNTIME_ARCHIVE}" "${RUNTIME_ARCHIVE_SHA256}"
check_sha256 "${COMPAT_ARCHIVE}" "${COMPAT_ARCHIVE_SHA256}"
tar --zstd -xf "${RUNTIME_ARCHIVE}" -C "${TEMP_ROOT}"
RUNTIME_SITE="${TEMP_ROOT}/site"
tar --zstd -xf "${COMPAT_ARCHIVE}" -C "${RUNTIME_SITE}"
export PYTHONPATH="${RUNTIME_SITE}:${PYTHONPATH:-}"

python3 - <<'PY'
import json
import platform

import huggingface_hub
import safetensors
import tokenizers
import torch
import transformers

expected = {
    "python": "3.11",
    "torch": "2.10.0+cu128",
    "cuda": "12.8",
    "cudnn": 91002,
    "transformers": "5.3.0",
    "tokenizers": "0.22.2",
    "huggingface_hub": "1.24.0",
    "safetensors": "0.8.0",
    "gpu_count": 8,
}
actual = {
    "python": ".".join(platform.python_version_tuple()[:2]),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "safetensors": safetensors.__version__,
    "gpu_count": torch.cuda.device_count(),
}
if actual != expected:
    raise SystemExit(f"archived runtime mismatch: expected={expected}, actual={actual}")
print(json.dumps(actual, sort_keys=True))
PY

MODEL_SAFE="${TEMP_ROOT}/model-safetensors"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done

export SKILLGR_CODE_ROOT="${ROOT_DIR}"
export SKILLGR_CONFIG="${ROOT_DIR}/configs/legacy_catalog_base_exact_runtime_20260808.env"
export SKILLGR_EXACT_MODEL="${MODEL_SAFE}"
export SKILLGR_PHYSICAL_OUTPUT_ROOT="${PHYSICAL_ROOT}/experiments/legacy_catalog_base_exact_runtime_20260808_v1"

bash "${ROOT_DIR}/scripts/run_legacy_catalog_final_effect_node_20260808.sh"
