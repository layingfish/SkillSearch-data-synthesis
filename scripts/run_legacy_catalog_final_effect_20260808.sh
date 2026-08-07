#!/usr/bin/env bash
set -euo pipefail

CATALOG_EPOCHS="${1:?usage: $0 0|3|6|12 seed}"
RUN_SEED="${2:?usage: $0 0|3|6|12 seed}"
case "${CATALOG_EPOCHS}" in
  0|3|6|12) ;;
  *) echo "catalog epochs must be one of 0, 3, 6, 12" >&2; exit 2 ;;
esac
[[ "${RUN_SEED}" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }

ROOT_DIR="${SKILLGR_CODE_ROOT:-/workspace/SkillSearch-data-synthesis}"
LEGACY_ROOT="${SKILLGR_LEGACY_SOURCE_ROOT:?legacy source root is required}"
set -a
source "${ROOT_DIR}/configs/legacy_catalog_final_effect_20260808.env"
set +a

ARM="legacy-c${CATALOG_EPOCHS}-q${SKILLGR_QUERY_EPOCHS}-seed${RUN_SEED}"
OUTPUT_DIR="${SKILLGR_OUTPUT_ROOT}/${ARM}"
EVAL_DIR="${SKILLGR_OUTPUT_ROOT}/eval/${ARM}/official_core75"
DONE_MARKER="${SKILLGR_OUTPUT_ROOT}/${ARM}.done"
mkdir -p "${SKILLGR_OUTPUT_ROOT}/logs"
exec > >(tee -a "${SKILLGR_OUTPUT_ROOT}/logs/${ARM}.log") 2>&1

require_file() {
  [[ -f "$1" ]] || { echo "required file is missing: $1" >&2; exit 1; }
}

require_absent() {
  [[ ! -e "$1" ]] || { echo "refusing to overwrite experiment artifact: $1" >&2; exit 1; }
}

for required in \
  "${SKILLGR_ROWS}" \
  "${SKILLGR_DEV_ROWS}" \
  "${SKILLGR_OFFICIAL_ROWS}" \
  "${SKILLGR_SKILLS}" \
  "${SKILLGR_SEMANTIC_IDS}" \
  "${SKILLGR_MODEL}/config.json" \
  "${LEGACY_ROOT}/scripts/skill_gr_pipeline.py"; do
  require_file "${required}"
done
require_absent "${OUTPUT_DIR}"
require_absent "${EVAL_DIR}"
require_absent "${DONE_MARKER}"

observed_pipeline_sha256="$(sha256sum "${LEGACY_ROOT}/scripts/skill_gr_pipeline.py" | awk '{print $1}')"
if [[ "${observed_pipeline_sha256}" != "${SKILLGR_LEGACY_PIPELINE_SHA256}" ]]; then
  echo "archived pipeline SHA-256 mismatch" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python3 "${LEGACY_ROOT}/scripts/skill_gr_pipeline.py" train-single-sid-decoder \
  --rows "${SKILLGR_ROWS}" \
  --dev-rows "${SKILLGR_DEV_ROWS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --skills-jsonl "${SKILLGR_SKILLS}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${SKILLGR_MODEL}" \
  --catalog-epochs "${CATALOG_EPOCHS}" \
  --epochs "${SKILLGR_QUERY_EPOCHS}" \
  --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
  --batch-size "${SKILLGR_TRAIN_BATCH}" \
  --eval-batch-size "${SKILLGR_EVAL_BATCH}" \
  --lr "${SKILLGR_LR}" \
  --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
  --logging-steps 50 \
  --eval-steps 1000 \
  --seed "${RUN_SEED}" \
  --load-best-model-at-end \
  --bf16

python3 - "${OUTPUT_DIR}" "${CATALOG_EPOCHS}" "${RUN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
catalog_epochs = int(sys.argv[2])
seed = int(sys.argv[3])
manifest = json.loads((output / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8"))
expected = {
    "batch_size": 16,
    "eval_batch_size": 128,
    "catalog_steps": 910 * catalog_epochs,
    "query_steps": 40420,
    "max_source_length": 2048,
    "lr": 0.0003,
    "weight_decay": 0.0,
    "seed": seed,
    "load_best_model_at_end": True,
    "formulation": "catalog indexing followed by query->single SID finetuning",
}
mismatches = {
    key: {"expected": value, "actual": manifest.get(key)}
    for key, value in expected.items()
    if manifest.get(key) != value
}
if mismatches:
    raise SystemExit(f"training manifest mismatch: {mismatches}")
if not (output / "final" / "structured_decoder.pt").is_file():
    raise SystemExit("final checkpoint is missing")
print(json.dumps({"catalog_epochs": catalog_epochs, "seed": seed, "manifest_verified": True}))
PY

cd "${ROOT_DIR}"
python3 scripts/skill_gr_pipeline.py eval-single-sid-decoder \
  --rows "${SKILLGR_OFFICIAL_ROWS}" \
  --model-dir "${OUTPUT_DIR}/final" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --output-dir "${EVAL_DIR}" \
  --split-name official_core75 \
  --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
  --num-beams "${SKILLGR_NUM_BEAMS}" \
  --rank-aggregation beam_order \
  --decoder-head ce \
  --score-transform logprob \
  --k-values 1,10,50 \
  --log-every 25

python3 scripts/skill_gr_pipeline.py rescore-ranked \
  --predictions "${EVAL_DIR}/predictions.jsonl" \
  --subset all \
  --k-values 1,10,50 \
  --output "${EVAL_DIR}/ranked_metrics_all.json"

python3 - \
  "${ARM}" \
  "${CATALOG_EPOCHS}" \
  "${RUN_SEED}" \
  "${OUTPUT_DIR}" \
  "${EVAL_DIR}" \
  "${SKILLGR_LEGACY_COMMIT}" \
  "${SKILLGR_ARCHIVED_OFFICIAL}" <<'PY'
import json
import sys
from pathlib import Path

arm = sys.argv[1]
catalog_epochs = int(sys.argv[2])
seed = int(sys.argv[3])
output = Path(sys.argv[4])
eval_dir = Path(sys.argv[5])
legacy_commit = sys.argv[6]
archive = Path(sys.argv[7])

predictions = [
    json.loads(line)
    for line in (eval_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(predictions) != 75:
    raise SystemExit(f"Official core75 prediction count mismatch: {len(predictions)}")
metrics = json.loads((eval_dir / "ranked_metrics_all.json").read_text(encoding="utf-8"))["overall"]
summary = {
    "arm": arm,
    "catalog_epochs": catalog_epochs,
    "seed": seed,
    "source_commit": legacy_commit,
    "protocol": {
        "global_batch_size": 16,
        "query_epochs": 5,
        "lr": 0.0003,
        "scheduler": "constant",
        "weight_decay": 0.0,
        "precision": "bf16",
        "selected_checkpoint": "dev_best",
        "evaluation": "Official core75, beam=50, beam_order, CE log-prob",
    },
    "overall": metrics,
}
if catalog_epochs == 3 and seed == 20260626 and archive.is_file():
    archived = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(archived) == len(predictions):
        exact_rows = sum(
            left.get("ranked_skill_ids") == right.get("ranked_skill_ids")
            for left, right in zip(archived, predictions)
        )
        summary["archived_top50_exact_rows"] = exact_rows

(output / "result_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY

touch "${DONE_MARKER}"
