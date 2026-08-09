#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_REPO=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis
PHYSICAL_OUTPUT_BASE=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
RUNTIME_ARCHIVE="${RUNTIME_ROOT}/site.tar.zst"
RUNTIME_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
COMPAT_ARCHIVE="${RUNTIME_ROOT}/compat-overlay.tar.zst"
COMPAT_ARCHIVE_SHA256=a6b2f56bb9b4496caa4be225b1c2630270cef1b9a23f82013eea72632bda6b20
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
LEGACY_COMMIT=15f6c51131f3b6f3e8580aa4564b17cbe13a9790
LEGACY_PIPELINE_SHA256=5d60624b704e3563f1893b021e15c2c6da3cdb28a18a1ac3018ca3884de04f67
OUTPUT_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/legacy_blas_control_probe_20260809_v2"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite precision-control output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/arms"
exec > >(tee "${OUTPUT_ROOT}/logs/node.log") 2>&1

if [[ ! -d "${PHYSICAL_REPO}/.git" ]]; then
  echo "physical repository is unavailable: ${PHYSICAL_REPO}" >&2
  exit 1
fi
git -C "${PHYSICAL_REPO}" cat-file -e "${LEGACY_COMMIT}^{commit}"
echo "launch_commit=${SKILLGR_LAUNCH_COMMIT:-unknown}"

TEMP_ROOT="$(mktemp -d /tmp/skillgr-precision-control.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-precision-control.*) rm -rf -- "${TEMP_ROOT}" ;;
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
    echo "SHA-256 mismatch for ${path}: expected=${expected}, observed=${observed}" >&2
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
    raise SystemExit(f"precision-control runtime mismatch: expected={expected}, actual={actual}")
print(json.dumps(actual, sort_keys=True))
PY

MODEL_SAFE="${TEMP_ROOT}/model-safetensors"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done

LEGACY_SOURCE="${TEMP_ROOT}/legacy-source"
mkdir -p "${LEGACY_SOURCE}"
git -C "${PHYSICAL_REPO}" archive --format=tar "${LEGACY_COMMIT}" | tar -xf - -C "${LEGACY_SOURCE}"
observed_pipeline_sha256="$(sha256sum "${LEGACY_SOURCE}/scripts/skill_gr_pipeline.py" | awk '{print $1}')"
if [[ "${observed_pipeline_sha256}" != "${LEGACY_PIPELINE_SHA256}" ]]; then
  echo "legacy pipeline mismatch: expected=${LEGACY_PIPELINE_SHA256}, observed=${observed_pipeline_sha256}" >&2
  exit 1
fi

set -a
export SKILLGR_EXACT_MODEL="${MODEL_SAFE}"
source "${CODE_ROOT}/configs/legacy_catalog_base_exact_runtime_20260808.env"
set +a
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

for required in "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SKILLGR_SKILLS}" "${SKILLGR_SEMANTIC_IDS}" "${BASE_MODEL}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done
check_sha256 "${SKILLGR_ROWS}" "${SKILLGR_ROWS_SHA256}"
check_sha256 "${SKILLGR_DEV_ROWS}" "${SKILLGR_DEV_ROWS_SHA256}"
check_sha256 "${SKILLGR_SKILLS}" "${SKILLGR_SKILLS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"

python3 -m py_compile "${CODE_ROOT}/scripts/legacy_runtime_matrix_probe.py"

declare -a PIDS=()
declare -A LABEL_BY_PID=()
terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait || true
}
trap 'terminate_children; cleanup; exit 130' INT TERM

launch_arm() {
  local gpu="$1" label="$2" precision="$3" bf16_reduction="$4"
  local tf32_matmul="$5" tf32_cudnn="$6" fp32_matmul_precision="$7" foreach="$8"
  local preferred_blas="${9}" deterministic_algorithms="${10}"
  local cublas_workspace_config="${11}" cuda_module_loading="${12}"
  local output="${OUTPUT_ROOT}/arms/${label}"
  (
    exec > >(tee "${OUTPUT_ROOT}/logs/${label}.log") 2>&1
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}:${RUNTIME_SITE}:${PYTHONPATH:-}" \
      python3 "${CODE_ROOT}/scripts/legacy_runtime_matrix_probe.py" \
        --arm "${label}" --legacy-source "${LEGACY_SOURCE}" --legacy-commit "${LEGACY_COMMIT}" \
        --rows "${SKILLGR_ROWS}" --dev-rows "${SKILLGR_DEV_ROWS}" \
        --skills-jsonl "${SKILLGR_SKILLS}" --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
        --model "${MODEL_SAFE}" --output-dir "${output}" --steps 50 --seed 20260626 \
        --precision "${precision}" --bf16-reduction "${bf16_reduction}" \
        --tf32-matmul "${tf32_matmul}" --tf32-cudnn "${tf32_cudnn}" \
        --float32-matmul-precision "${fp32_matmul_precision}" --foreach "${foreach}" \
        --preferred-blas "${preferred_blas}" --deterministic-algorithms "${deterministic_algorithms}" \
        --cublas-workspace-config "${cublas_workspace_config}" \
        --cuda-module-loading "${cuda_module_loading}"
  ) &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}"
  echo "launched ${label} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_arm 0 default_bf16 bf16 auto auto auto auto auto auto auto auto auto
launch_arm 1 blas_cublas bf16 auto auto auto auto auto cublas auto auto auto
launch_arm 2 blas_cublaslt bf16 auto auto auto auto auto cublaslt auto auto auto
launch_arm 3 workspace_4096 bf16 auto auto auto auto auto auto auto :4096:8 auto
launch_arm 4 workspace_16 bf16 auto auto auto auto auto auto auto :16:8 auto
launch_arm 5 deterministic_4096 bf16 auto auto auto auto auto auto true :4096:8 auto
launch_arm 6 cublaslt_bf16_reduction_off bf16 false auto auto auto auto cublaslt auto auto auto
launch_arm 7 cuda_module_eager bf16 auto auto auto auto auto auto auto auto EAGER

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
python3 - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = []
for path in sorted((root / "arms").glob("*/runtime_probe.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    arms.append({
        "arm": payload["arm"],
        "runtime": payload["runtime"],
        "condition": payload["condition"],
        "initialization": payload["initialization"],
        "first_catalog_batch": payload["first_catalog_batch"],
        "losses": payload["losses"],
        "step_50_delta_from_archived_rtx": payload["losses"]["step_50"] - 4.203788757324219,
    })
if len(arms) != 8:
    raise SystemExit(f"expected 8 completed precision-control arms, found {len(arms)}")
expected_batch = "721a0d93aaef2f9d0104adae8501fcfabcefcd6bd9f23c1e4a1aae701710d443"
bad_batch = [a["arm"] for a in arms if a["first_catalog_batch"]["row_ids_sha256"] != expected_batch]
if bad_batch:
    raise SystemExit(f"catalog batch-order mismatch in arms: {bad_batch}")
summary = {
    "artifact_format": "legacy_blas_control_matrix_v2",
    "archived_rtx_reference": {
        "step_1": 4.528710842132568,
        "step_50": 4.203788757324219,
        "first_catalog_batch_sha256": expected_batch,
    },
    "arms": arms,
}
(root / "matrix_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"arms": len(arms), "batch_order": "matched"}, sort_keys=True))
PY

if (( failures > 0 )); then exit 1; fi
touch "${OUTPUT_ROOT}/matrix.done"
