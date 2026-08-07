#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKILLGR_CODE_ROOT:-/workspace/SkillSearch-data-synthesis}"
PHYSICAL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/legacy_last_parity_20260808_v1
ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh

if [[ -e "${PHYSICAL_OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite experiment root: ${PHYSICAL_OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${PHYSICAL_OUTPUT_ROOT}/logs"
exec > >(tee -a "${PHYSICAL_OUTPUT_ROOT}/logs/job.log") 2>&1

bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1
git config --global --add safe.directory "${ROOT_DIR}"
git config --global --add safe.directory "$(readlink -f "${ROOT_DIR}")"
cd "${ROOT_DIR}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "parity diagnostic requires a clean immutable checkout" >&2
  git status --short
  exit 1
fi
git rev-parse HEAD | tee "${PHYSICAL_OUTPUT_ROOT}/launch_commit.txt"

set -a
source configs/legacy_catalog_final_effect_20260808.env
source configs/legacy_last_checkpoint_parity_20260808.env
set +a
if [[ "$(readlink -f "${SKILLGR_PARITY_OUTPUT_ROOT}")" != "${PHYSICAL_OUTPUT_ROOT}" ]]; then
  echo "configured parity output root does not match the physical output root" >&2
  exit 1
fi
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

for required in \
  "${SKILLGR_ARCHIVED_RUN}/final/structured_decoder.pt" \
  "${SKILLGR_ARCHIVED_RUN}/final/tokenizer.json" \
  "${SKILLGR_NEW_RUN}/last/structured_decoder.pt" \
  "${SKILLGR_NEW_RUN}/last/tokenizer.json" \
  "${SKILLGR_NEW_RUN}/single_sid_decoder_train_manifest.json" \
  "${SKILLGR_NEW_RUN}/loss_history.jsonl" \
  "${SKILLGR_ARCHIVED_EVAL_DIR}/predictions.jsonl" \
  "${SKILLGR_ARCHIVED_EVAL_DIR}/eval_manifest.json" \
  "${SKILLGR_ROWS}" \
  "${SKILLGR_DEV_ROWS}" \
  "${SKILLGR_OFFICIAL_ROWS}" \
  "${SKILLGR_SKILLS}" \
  "${SKILLGR_SEMANTIC_IDS}"; do
  [[ -f "${required}" ]] || { echo "required file is missing: ${required}" >&2; exit 1; }
done

LEGACY_SOURCE_ROOT="$(mktemp -d /tmp/skillgr-legacy-15f6c51.XXXXXX)"
git archive "${SKILLGR_LEGACY_COMMIT}" | tar -xf - -C "${LEGACY_SOURCE_ROOT}"
observed_pipeline_sha256="$(sha256sum "${LEGACY_SOURCE_ROOT}/scripts/skill_gr_pipeline.py" | awk '{print $1}')"
if [[ "${observed_pipeline_sha256}" != "${SKILLGR_LEGACY_PIPELINE_SHA256}" ]]; then
  echo "archived pipeline SHA-256 mismatch" >&2
  exit 1
fi

python3 scripts/validate_legacy_last_parity.py preflight \
  --new-run "${SKILLGR_NEW_RUN}" \
  --archived-run "${SKILLGR_ARCHIVED_RUN}" \
  --archived-eval-dir "${SKILLGR_ARCHIVED_EVAL_DIR}" \
  --rows "${SKILLGR_ROWS}" \
  --dev-rows "${SKILLGR_DEV_ROWS}" \
  --official-rows "${SKILLGR_OFFICIAL_ROWS}" \
  --skills-jsonl "${SKILLGR_SKILLS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --model "${SKILLGR_MODEL}" \
  --new-checkpoint-sha256 "${SKILLGR_NEW_LAST_CHECKPOINT_SHA256}" \
  --new-train-manifest-sha256 "${SKILLGR_NEW_TRAIN_MANIFEST_SHA256}" \
  --archived-predictions-sha256 "${SKILLGR_ARCHIVED_PREDICTIONS_SHA256}" \
  --archived-eval-manifest-sha256 "${SKILLGR_ARCHIVED_EVAL_MANIFEST_SHA256}" \
  --rows-sha256 "${SKILLGR_TRAIN_ROWS_SHA256}" \
  --dev-rows-sha256 "${SKILLGR_DEV_ROWS_SHA256}" \
  --official-rows-sha256 "${SKILLGR_OFFICIAL_ROWS_SHA256}" \
  --skills-sha256 "${SKILLGR_SKILLS_SHA256}" \
  --semantic-ids-file-sha256 "${SKILLGR_SEMANTIC_IDS_FILE_SHA256}" \
  --archived-tokenizer-json-sha256 "${SKILLGR_ARCHIVED_TOKENIZER_JSON_SHA256}" \
  --new-tokenizer-json-sha256 "${SKILLGR_NEW_TOKENIZER_JSON_SHA256}" \
  --output "${PHYSICAL_OUTPUT_ROOT}/provenance.json"

EVAL_DIR="${PHYSICAL_OUTPUT_ROOT}/eval/new_last_official_core75"
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py eval-single-sid-decoder \
  --rows "${SKILLGR_OFFICIAL_ROWS}" \
  --model-dir "${SKILLGR_NEW_RUN}/last" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --output-dir "${EVAL_DIR}" \
  --split-name official_core75 \
  --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
  --num-beams "${SKILLGR_NUM_BEAMS}" \
  --rank-aggregation beam_order \
  --decoder-head ce \
  --score-transform logprob \
  --k-values 1,10,50 \
  --log-every 10

python3 scripts/audit_legacy_tokenizer_parity.py \
  --pipeline "${LEGACY_SOURCE_ROOT}/scripts/skill_gr_pipeline.py" \
  --archived-tokenizer "${SKILLGR_ARCHIVED_RUN}/final" \
  --new-tokenizer "${SKILLGR_NEW_RUN}/last" \
  --rows "${SKILLGR_ROWS}" \
  --dev-rows "${SKILLGR_DEV_ROWS}" \
  --official-rows "${SKILLGR_OFFICIAL_ROWS}" \
  --skills-jsonl "${SKILLGR_SKILLS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
  --train-batch-size "${SKILLGR_TRAIN_BATCH}" \
  --eval-batch-size "${SKILLGR_EVAL_BATCH}" \
  --official-batch-size 1 \
  --output "${PHYSICAL_OUTPUT_ROOT}/tokenizer_parity.json"

for subset in all single multi; do
  python3 scripts/skill_gr_pipeline.py rescore-ranked \
    --predictions "${EVAL_DIR}/predictions.jsonl" \
    --subset "${subset}" \
    --k-values 1,10,50 \
    --output "${EVAL_DIR}/ranked_metrics_${subset}.json"
  python3 scripts/skill_gr_pipeline.py rescore-ranked \
    --predictions "${SKILLGR_ARCHIVED_EVAL_DIR}/predictions.jsonl" \
    --subset "${subset}" \
    --k-values 1,10,50 \
    --output "${PHYSICAL_OUTPUT_ROOT}/archived_ranked_metrics_${subset}.json"
done

python3 scripts/validate_legacy_last_parity.py summarize \
  --new-run "${SKILLGR_NEW_RUN}" \
  --archived-eval-dir "${SKILLGR_ARCHIVED_EVAL_DIR}" \
  --new-eval-dir "${EVAL_DIR}" \
  --official-rows "${SKILLGR_OFFICIAL_ROWS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --provenance "${PHYSICAL_OUTPUT_ROOT}/provenance.json" \
  --tokenizer-parity "${PHYSICAL_OUTPUT_ROOT}/tokenizer_parity.json" \
  --output-root "${PHYSICAL_OUTPUT_ROOT}" \
  --output "${PHYSICAL_OUTPUT_ROOT}/summary.json"

touch "${PHYSICAL_OUTPUT_ROOT}/done"
