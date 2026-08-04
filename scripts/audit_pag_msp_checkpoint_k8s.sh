#!/usr/bin/env bash
set -euo pipefail

STEP="${PAG_MSP_AUDIT_STEP:?set PAG_MSP_AUDIT_STEP to a saved 25000-step multiple}"
if (( STEP <= 0 || STEP % 25000 != 0 || STEP > 200000 )); then
  echo "PAG_MSP_AUDIT_STEP must be a 25000-step multiple in [25000, 200000]" >&2
  exit 2
fi

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1

REPO=/workspace/SkillSearch-data-synthesis
EXPERIMENT=/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_pag_single_sid_20260804_v1
CHECKPOINT="${EXPERIMENT}/pag_msp/checkpoint-${STEP}/pag_sparse_training_state.pt"
SKILLS=/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/canonical_skills.jsonl
MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
OUTPUT="${EXPERIMENT}/pag_msp/audits/step-${STEP}.json"
LOG="${EXPERIMENT}/logs/pag-msp-audit-step-${STEP}.log"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint is not complete: ${CHECKPOINT}" >&2
  exit 3
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite existing audit: ${OUTPUT}" >&2
  exit 3
fi

mkdir -p "${EXPERIMENT}/logs"
exec > >(tee "${LOG}") 2>&1
cd "${REPO}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node=8 scripts/audit_pag_sparse_representations.py \
  --model "${MODEL}" \
  --skills-jsonl "${SKILLS}" \
  --checkpoint "${CHECKPOINT}" \
  --sample-size 512 \
  --batch-size 8 \
  --max-length 128 \
  --top-k 64 \
  --label "msp_step_${STEP}" \
  --output "${OUTPUT}"

test -s "${OUTPUT}"
