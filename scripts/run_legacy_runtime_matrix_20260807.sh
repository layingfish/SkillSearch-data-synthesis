#!/usr/bin/env bash
set -euo pipefail

LEGACY_COMMIT=15f6c51
PHYSICAL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/legacy_runtime_matrix_20260807_v1
ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh

if [[ -e "${PHYSICAL_OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite runtime matrix output: ${PHYSICAL_OUTPUT_ROOT}" >&2
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
  echo "runtime matrix requires a clean immutable checkout" >&2
  git status --short
  exit 1
fi
git rev-parse HEAD
git cat-file -e "${LEGACY_COMMIT}^{commit}"
python3 docker/smoke_test.py

set -a
source configs/legacy_catalog_sweep_20260807.env
set +a
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

TEMP_ROOT="$(mktemp -d /tmp/skillgr-runtime-matrix.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-runtime-matrix.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

LEGACY_SOURCE="${TEMP_ROOT}/legacy-source"
MODEL_SAFE="${TEMP_ROOT}/model-safetensors"
MODEL_BIN="${TEMP_ROOT}/model-pytorch-bin"
TF53_TARGET="${TEMP_ROOT}/transformers-5.3"
mkdir -p "${LEGACY_SOURCE}" "${MODEL_SAFE}" "${MODEL_BIN}" "${TF53_TARGET}"
git archive --format=tar "${LEGACY_COMMIT}" | tar -xf - -C "${LEGACY_SOURCE}"

for filename in config.json generation_config.json tokenizer.json spiece.model; do
  ln -s "${SKILLGR_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
  ln -s "${SKILLGR_MODEL}/${filename}" "${MODEL_BIN}/${filename}"
done
ln -s "${SKILLGR_MODEL}/model.safetensors" "${MODEL_SAFE}/model.safetensors"
ln -s "${SKILLGR_MODEL}/pytorch_model.bin" "${MODEL_BIN}/pytorch_model.bin"

python3 -m pip install --no-deps --target "${TF53_TARGET}" -q \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers==5.3.0 \
  huggingface-hub==1.24.0 \
  tokenizers==0.22.2 \
  safetensors==0.8.0

PYTHONPATH="${TF53_TARGET}:${PYTHONPATH:-}" python3 - <<'PY'
import huggingface_hub
import safetensors
import tokenizers
import transformers

expected = ("5.3.0", "0.22.2", "1.24.0", "0.8.0")
actual = (
    transformers.__version__,
    tokenizers.__version__,
    huggingface_hub.__version__,
    safetensors.__version__,
)
if actual != expected:
    raise SystemExit(f"Transformers 5.3 overlay mismatch: expected={expected}, actual={actual}")
print({"transformers_5_3_overlay": actual})
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
  local model_format="$3"
  local replica="$4"
  local model_dir
  local python_path="${PYTHONPATH:-}"
  if [[ "${runtime}" == tf53 ]]; then
    python_path="${TF53_TARGET}:${python_path}"
  fi
  case "${model_format}" in
    safe) model_dir="${MODEL_SAFE}" ;;
    bin) model_dir="${MODEL_BIN}" ;;
    *) echo "unknown model format: ${model_format}" >&2; exit 2 ;;
  esac
  local label="${runtime}-${model_format}-r${replica}"
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
        --model "${model_dir}" \
        --output-dir "${output}" \
        --steps 50 \
        --seed 20260626
  ) &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}"
  echo "launched ${label} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_arm 0 tf457 safe 1
launch_arm 1 tf457 safe 2
launch_arm 2 tf457 bin 1
launch_arm 3 tf457 bin 2
launch_arm 4 tf53 safe 1
launch_arm 5 tf53 safe 2
launch_arm 6 tf53 bin 1
launch_arm 7 tf53 bin 2

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
    raise SystemExit(f"expected 8 completed runtime arms, found {len(arms)}")
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
