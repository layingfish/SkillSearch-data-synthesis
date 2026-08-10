#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_REPO=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis
OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/legacy_dropout0_probe_20260810_v1
RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
PY310_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1
PY310_RUNTIME_ARCHIVE="${PY310_RUNTIME_ROOT}/runtime.tar.zst"
PY310_RUNTIME_ARCHIVE_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_SITE_ARCHIVE="${RUNTIME_ROOT}/site.tar.zst"
PY311_SITE_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
LEGACY_COMMIT=archive-original-20260714
LEGACY_PIPELINE_SHA256=7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite dropout-zero probe output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/probe"
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

TEMP_ROOT="$(mktemp -d /tmp/skillgr-dropout0.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-dropout0.*) rm -rf -- "${TEMP_ROOT}" ;;
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
import torch
import transformers
from importlib import metadata

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
    raise SystemExit(f"dropout-zero runtime mismatch: expected={expected}, actual={actual}")
print(json.dumps({"runtime": actual, "torch_config": torch.__config__.show()}, sort_keys=True))
PY

MODEL_SAFE="${TEMP_ROOT}/model-safetensors"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done

LEGACY_SOURCE="${TEMP_ROOT}/legacy-source"
mkdir -p "${LEGACY_SOURCE}"
mkdir -p "${LEGACY_SOURCE}/scripts"
ln -s "${CODE_ROOT}/repro/20260714_letter_base_single_sid/skill_gr_pipeline.original_20260714.py" \
  "${LEGACY_SOURCE}/scripts/skill_gr_pipeline.py"
check_sha256 "${LEGACY_SOURCE}/scripts/skill_gr_pipeline.py" "${LEGACY_PIPELINE_SHA256}"

export SKILLGR_EXACT_MODEL="${MODEL_SAFE}"
set -a
source "${CODE_ROOT}/configs/legacy_catalog_base_exact_runtime_20260808.env"
set +a
check_sha256 "${SKILLGR_ROWS}" "${SKILLGR_ROWS_SHA256}"
check_sha256 "${SKILLGR_DEV_ROWS}" "${SKILLGR_DEV_ROWS_SHA256}"
check_sha256 "${SKILLGR_SKILLS}" "${SKILLGR_SKILLS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"
check_sha256 "${BASE_MODEL}/model.safetensors" a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${CODE_ROOT}/scripts/legacy_runtime_matrix_probe.py" \
  --arm dropout0_bf16 \
  --legacy-source "${LEGACY_SOURCE}" \
  --legacy-commit "${LEGACY_COMMIT}" \
  --rows "${SKILLGR_ROWS}" \
  --dev-rows "${SKILLGR_DEV_ROWS}" \
  --skills-jsonl "${SKILLGR_SKILLS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --model "${MODEL_SAFE}" \
  --output-dir "${OUTPUT_ROOT}/probe" \
  --steps 1 \
  --seed 20260626 \
  --precision bf16 \
  --bf16-reduction auto \
  --tf32-matmul auto \
  --tf32-cudnn auto \
  --float32-matmul-precision auto \
  --foreach auto \
  --preferred-blas auto \
  --deterministic-algorithms auto \
  --cublas-workspace-config auto \
  --cuda-module-loading auto \
  --dropout-rate 0.0

"${PYTHON_BIN}" - "${OUTPUT_ROOT}/probe/runtime_probe.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload["losses"] != {"step_1": payload["losses"]["step_1"]}:
    raise SystemExit(f"unexpected probe loss events: {payload['losses']}")
child = payload["condition"]["child_effective_precision"]
if child["dropout_module_count"] <= 0 or child["dropout_module_rates"] != [0.0]:
    raise SystemExit(f"dropout override audit failed: {child}")
if payload["initialization"]["legacy_state_fingerprint"] != "c219d07a2ac425cffd677cefc0792a81f47aa3ad9d0fa76a0bfa37cdf9f5ebde":
    raise SystemExit("initial model fingerprint mismatch")
if payload["first_catalog_batch"]["row_ids_sha256"] != "721a0d93aaef2f9d0104adae8501fcfabcefcd6bd9f23c1e4a1aae701710d443":
    raise SystemExit("first catalog batch mismatch")
print(json.dumps({"verification": "passed", "step_1": payload["losses"]["step_1"]}, sort_keys=True))
PY
touch "${OUTPUT_ROOT}/probe.done"
