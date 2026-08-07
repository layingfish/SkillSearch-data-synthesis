#!/usr/bin/env bash
set -euo pipefail

LEGACY_COMMIT=15f6c51
PHYSICAL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/legacy_baseline_parity_probe_20260807_v1
ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh

if [[ -e "${PHYSICAL_OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite parity probe output: ${PHYSICAL_OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${PHYSICAL_OUTPUT_ROOT}"
exec > >(tee "${PHYSICAL_OUTPUT_ROOT}/probe.log") 2>&1

bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1
ROOT_DIR=/workspace/SkillSearch-data-synthesis
git config --global --add safe.directory "${ROOT_DIR}"
git config --global --add safe.directory "$(readlink -f "${ROOT_DIR}")"
cd "${ROOT_DIR}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "parity probe requires a clean immutable checkout" >&2
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

LEGACY_SOURCE_ROOT="$(mktemp -d /tmp/skillgr-legacy-source.XXXXXX)"
TRAIN_ROOT="$(mktemp -d /tmp/skillgr-legacy-train.XXXXXX)"
cleanup() {
  local path
  for path in "${LEGACY_SOURCE_ROOT}" "${TRAIN_ROOT}"; do
    case "${path}" in
      /tmp/skillgr-legacy-source.*|/tmp/skillgr-legacy-train.*) rm -rf -- "${path}" ;;
      *) echo "refusing to remove unexpected temporary path: ${path}" >&2 ;;
    esac
  done
}
trap cleanup EXIT

git archive --format=tar "${LEGACY_COMMIT}" | tar -xf - -C "${LEGACY_SOURCE_ROOT}"

python3 - <<'PY'
import json
import platform
import torch
import tokenizers
import transformers

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "gpu_count": torch.cuda.device_count(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, sort_keys=True))
PY

python3 scripts/legacy_baseline_parity_probe.py audit \
  --legacy-source "${LEGACY_SOURCE_ROOT}" \
  --legacy-commit "$(git rev-parse "${LEGACY_COMMIT}^{commit}")" \
  --rows "${SKILLGR_ROWS}" \
  --dev-rows "${SKILLGR_DEV_ROWS}" \
  --skills-jsonl "${SKILLGR_SKILLS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --model "${SKILLGR_MODEL}" \
  --launcher scripts/run_legacy_baseline_parity_probe_20260807.sh \
  --catalog-epochs 3 \
  --batch-size 16 \
  --max-source-length 2048 \
  --lr 0.0003 \
  --weight-decay 0 \
  --seed 20260626 \
  --bf16 \
  --output "${PHYSICAL_OUTPUT_ROOT}/audit.json"

python3 "${LEGACY_SOURCE_ROOT}/scripts/skill_gr_pipeline.py" train-single-sid-decoder \
  --rows "${SKILLGR_ROWS}" \
  --dev-rows "${SKILLGR_DEV_ROWS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --skills-jsonl "${SKILLGR_SKILLS}" \
  --output-dir "${TRAIN_ROOT}/model" \
  --model "${SKILLGR_MODEL}" \
  --catalog-epochs 3 \
  --catalog-max-steps 950 \
  --epochs 0 \
  --max-source-length 2048 \
  --batch-size 16 \
  --eval-batch-size 128 \
  --lr 0.0003 \
  --weight-decay 0 \
  --logging-steps 1 \
  --eval-steps 1000 \
  --seed 20260626 \
  --bf16

python3 scripts/legacy_baseline_parity_probe.py finalize \
  --audit "${PHYSICAL_OUTPUT_ROOT}/audit.json" \
  --training-output "${TRAIN_ROOT}/model" \
  --archived-train-log \
    "${SKILLGR_OLD_TRAIN_LOG:-/models_rw/SkillSearch-data-synthesis/assets-models/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/single_sid_d4_lightgcn_d32_letterofficial_task5_isolatedmode3_core75_gt2_t5base_cat3_querye5_b16/train.log}" \
  --archived-train-log-sha256 \
    12a136a07d8986fd5cfcbd253b0d061791f734b05014af43d506755a16ee08b7 \
  --archived-manifest \
    /models_rw/SkillSearch-data-synthesis/assets-models/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/single_sid_d4_lightgcn_d32_letterofficial_task5_isolatedmode3_core75_gt2_t5base_cat3_querye5_b16/single_sid_decoder_train_manifest.json \
  --archived-manifest-sha256 \
    77dd35dc68f490fc1308cd52fd5773f849f9fe76a56bb187b175f59efa558ef1 \
  --output "${PHYSICAL_OUTPUT_ROOT}/parity_probe.json"

touch "${PHYSICAL_OUTPUT_ROOT}/probe.done"
