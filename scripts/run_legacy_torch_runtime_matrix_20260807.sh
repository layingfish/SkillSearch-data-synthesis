#!/usr/bin/env bash
set -euo pipefail

LEGACY_COMMIT=15f6c51
PHYSICAL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/legacy_torch_runtime_matrix_20260807_v2
ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
RUNTIME_ARCHIVE=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1/site.tar.zst
RUNTIME_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
COMPAT_ARCHIVE=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1/compat-overlay.tar.zst
COMPAT_ARCHIVE_SHA256=a6b2f56bb9b4496caa4be225b1c2630270cef1b9a23f82013eea72632bda6b20

if [[ -e "${PHYSICAL_OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite torch runtime matrix output: ${PHYSICAL_OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${PHYSICAL_OUTPUT_ROOT}/logs" "${PHYSICAL_OUTPUT_ROOT}/arms"
exec > >(tee "${PHYSICAL_OUTPUT_ROOT}/logs/node.log") 2>&1

bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1
ROOT_DIR=/workspace/SkillSearch-data-synthesis
git config --global --add safe.directory "${ROOT_DIR}"
git config --global --add safe.directory "$(readlink -f "${ROOT_DIR}")"
cd "${ROOT_DIR}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "torch runtime matrix requires a clean immutable checkout" >&2
  git status --short
  exit 1
fi
git rev-parse HEAD
git cat-file -e "${LEGACY_COMMIT}^{commit}"

TEMP_ROOT="$(mktemp -d /tmp/skillgr-torch-runtime-matrix.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-torch-runtime-matrix.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

if [[ ! -f "${RUNTIME_ARCHIVE}" ]]; then
  echo "registered torch-2.10 runtime archive is missing: ${RUNTIME_ARCHIVE}" >&2
  exit 1
fi
archive_sha256="$(sha256sum "${RUNTIME_ARCHIVE}" | cut -d' ' -f1)"
if [[ "${archive_sha256}" != "${RUNTIME_ARCHIVE_SHA256}" ]]; then
  echo "torch-2.10 runtime archive mismatch: ${archive_sha256}" >&2
  exit 1
fi
if [[ ! -f "${COMPAT_ARCHIVE}" ]]; then
  echo "registered torch-2.10 compatibility archive is missing: ${COMPAT_ARCHIVE}" >&2
  exit 1
fi
compat_archive_sha256="$(sha256sum "${COMPAT_ARCHIVE}" | cut -d' ' -f1)"
if [[ "${compat_archive_sha256}" != "${COMPAT_ARCHIVE_SHA256}" ]]; then
  echo "torch-2.10 compatibility archive mismatch: ${compat_archive_sha256}" >&2
  exit 1
fi
tar --zstd -xf "${RUNTIME_ARCHIVE}" -C "${TEMP_ROOT}"
RUNTIME_SITE="${TEMP_ROOT}/site"
if [[ ! -d "${RUNTIME_SITE}/torch" || ! -d "${RUNTIME_SITE}/transformers" ]]; then
  echo "torch-2.10 runtime archive is incomplete after extraction" >&2
  exit 1
fi
tar --zstd -xf "${COMPAT_ARCHIVE}" -C "${RUNTIME_SITE}"
export PYTHONPATH="${RUNTIME_SITE}:${PYTHONPATH:-}"

python3 - <<'PY'
import scipy
import sklearn
import torch
import torchvision
import transformers
from transformers import T5ForConditionalGeneration

expected = (
    "2.10.0+cu128",
    "12.8",
    91002,
    "5.3.0",
    "0.25.0+cu128",
    "1.15.3",
    "1.7.2",
    "T5ForConditionalGeneration",
)
actual = (
    torch.__version__,
    torch.version.cuda,
    torch.backends.cudnn.version(),
    transformers.__version__,
    torchvision.__version__,
    scipy.__version__,
    sklearn.__version__,
    T5ForConditionalGeneration.__name__,
)
if actual != expected:
    raise SystemExit(f"torch-2.10 image mismatch: expected={expected}, actual={actual}")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 visible GPUs, found {torch.cuda.device_count()}")
print({"torch_2_10_image": actual, "gpu_count": torch.cuda.device_count()})
PY

set -a
source configs/legacy_catalog_sweep_20260807.env
set +a
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

LEGACY_SOURCE="${TEMP_ROOT}/legacy-source"
MODEL_SAFE="${TEMP_ROOT}/model-safetensors"
TF457_TARGET="${TEMP_ROOT}/transformers-4.57"
mkdir -p "${LEGACY_SOURCE}" "${MODEL_SAFE}" "${TF457_TARGET}"
git archive --format=tar "${LEGACY_COMMIT}" | tar -xf - -C "${LEGACY_SOURCE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${SKILLGR_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done

python3 -m pip install --no-deps --target "${TF457_TARGET}" -q \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers==4.57.6 \
  huggingface-hub==0.36.2 \
  tokenizers==0.22.2 \
  safetensors==0.8.0

PYTHONPATH="${TF457_TARGET}:${PYTHONPATH:-}" python3 - <<'PY'
import huggingface_hub
import safetensors
import tokenizers
import torch
import transformers
from transformers import T5ForConditionalGeneration

expected = (
    "2.10.0+cu128",
    "4.57.6",
    "0.22.2",
    "0.36.2",
    "0.8.0",
    "T5ForConditionalGeneration",
)
actual = (
    torch.__version__,
    transformers.__version__,
    tokenizers.__version__,
    huggingface_hub.__version__,
    safetensors.__version__,
    T5ForConditionalGeneration.__name__,
)
if actual != expected:
    raise SystemExit(f"Transformers 4.57 overlay mismatch: expected={expected}, actual={actual}")
print({"transformers_4_57_overlay": actual})
PY

declare -a PIDS=()
declare -A LABEL_BY_PID=()

terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
}
trap 'terminate_children; cleanup; exit 130' INT TERM

launch_arm() {
  local gpu="$1"
  local runtime="$2"
  local replica="$3"
  local python_path="${PYTHONPATH:-}"
  if [[ "${runtime}" == tf457 ]]; then
    python_path="${TF457_TARGET}:${python_path}"
  fi
  local label="torch210-${runtime}-r${replica}"
  local output="${PHYSICAL_OUTPUT_ROOT}/arms/${label}"
  (
    exec > >(tee "${PHYSICAL_OUTPUT_ROOT}/logs/${label}.log") 2>&1
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" \
      python3 scripts/legacy_runtime_matrix_probe.py \
        --arm "${label}" \
        --legacy-source "${LEGACY_SOURCE}" \
        --legacy-commit "$(git rev-parse "${LEGACY_COMMIT}^{commit}")" \
        --rows "${SKILLGR_ROWS}" \
        --dev-rows "${SKILLGR_DEV_ROWS}" \
        --skills-jsonl "${SKILLGR_SKILLS}" \
        --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
        --model "${MODEL_SAFE}" \
        --output-dir "${output}" \
        --steps 50 \
        --seed 20260626
  ) &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}"
  echo "launched ${label} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_arm 0 tf53 1
launch_arm 1 tf53 2
launch_arm 2 tf53 3
launch_arm 3 tf53 4
launch_arm 4 tf457 1
launch_arm 5 tf457 2
launch_arm 6 tf457 3
launch_arm 7 tf457 4

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
  if [[ -n "${finished_pid}" ]]; then
    unset 'LABEL_BY_PID[$finished_pid]'
  fi
  remaining=$((remaining - 1))
done

trap - INT TERM
python3 - "${PHYSICAL_OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = []
for path in sorted((root / "arms").glob("*/runtime_probe.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    arms.append(
        {
            "arm": payload["arm"],
            "runtime": payload["runtime"],
            "condition": payload["condition"],
            "initialization": payload["initialization"],
            "losses": payload["losses"],
            "step_50_delta_from_archived": payload["losses"]["step_50"] - 4.203789,
        }
    )
if len(arms) != 8:
    raise SystemExit(f"expected 8 completed torch runtime arms, found {len(arms)}")
(root / "matrix_summary.json").write_text(
    json.dumps({"arms": arms}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"arms": len(arms)}, sort_keys=True))
PY

if (( failures > 0 )); then
  exit 1
fi
touch "${PHYSICAL_OUTPUT_ROOT}/matrix.done"
