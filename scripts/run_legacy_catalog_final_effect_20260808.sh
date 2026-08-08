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
CONFIG_PATH="${SKILLGR_CONFIG:-${ROOT_DIR}/configs/legacy_catalog_final_effect_20260808.env}"
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH}"
fi
set -a
source "${CONFIG_PATH}"
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

check_sha256() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: expected ${expected}, observed ${observed}" >&2
    exit 1
  fi
}

check_sha256 "${SKILLGR_ROWS}" "${SKILLGR_ROWS_SHA256}"
check_sha256 "${SKILLGR_DEV_ROWS}" "${SKILLGR_DEV_ROWS_SHA256}"
check_sha256 "${SKILLGR_OFFICIAL_ROWS}" "${SKILLGR_OFFICIAL_ROWS_SHA256}"
check_sha256 "${SKILLGR_SKILLS}" "${SKILLGR_SKILLS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"
check_sha256 "${SKILLGR_ARCHIVED_OFFICIAL}" "${SKILLGR_ARCHIVED_OFFICIAL_SHA256}"

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
  --eval-steps "${SKILLGR_EVAL_STEPS}" \
  --seed "${RUN_SEED}" \
  --load-best-model-at-end \
  --bf16

python3 - \
  "${OUTPUT_DIR}" \
  "${CATALOG_EPOCHS}" \
  "${RUN_SEED}" \
  "${SKILLGR_EXPECTED_SOURCE_ROWS}" \
  "${SKILLGR_EXPECTED_QUERY_TRAIN_ROWS}" \
  "${SKILLGR_EXPECTED_QUERY_STEPS}" \
  "${SKILLGR_EXPECTED_CATALOG_STEPS_PER_EPOCH}" \
  "${SKILLGR_TRAIN_BATCH}" \
  "${SKILLGR_EVAL_BATCH}" \
  "${SKILLGR_EVAL_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
catalog_epochs = int(sys.argv[2])
seed = int(sys.argv[3])
source_rows = int(sys.argv[4])
query_train_rows = int(sys.argv[5])
query_steps = int(sys.argv[6])
catalog_steps_per_epoch = int(sys.argv[7])
train_batch_size = int(sys.argv[8])
eval_batch_size = int(sys.argv[9])
eval_steps = int(sys.argv[10])
manifest = json.loads((output / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8"))
expected = {
    "batch_size": train_batch_size,
    "eval_batch_size": eval_batch_size,
    "source_train_rows": source_rows,
    "query_train_rows": query_train_rows,
    "catalog_steps": catalog_steps_per_epoch * catalog_epochs,
    "query_steps": query_steps,
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
events = [
    json.loads(line)
    for line in (output / "loss_history.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
actual_eval_steps = [
    int(event["step"])
    for event in events
    if event.get("stage") == "query" and event.get("event") == "eval"
]
expected_eval_steps = list(range(eval_steps, query_steps + 1, eval_steps))
if not expected_eval_steps or expected_eval_steps[-1] != query_steps:
    expected_eval_steps.append(query_steps)
if actual_eval_steps != expected_eval_steps:
    raise SystemExit(
        "query eval cadence mismatch: "
        f"expected={expected_eval_steps[:3]}...{expected_eval_steps[-3:]}, "
        f"actual={actual_eval_steps[:3]}...{actual_eval_steps[-3:]}"
    )
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

for subset in all single multi; do
  python3 scripts/skill_gr_pipeline.py rescore-ranked \
    --predictions "${EVAL_DIR}/predictions.jsonl" \
    --subset "${subset}" \
    --k-values 1,10,50 \
    --output "${EVAL_DIR}/ranked_metrics_${subset}.json"
done

python3 - \
  "${ARM}" \
  "${CATALOG_EPOCHS}" \
  "${RUN_SEED}" \
  "${OUTPUT_DIR}" \
  "${EVAL_DIR}" \
  "${SKILLGR_LEGACY_COMMIT}" \
  "${SKILLGR_ARCHIVED_OFFICIAL}" \
  "${SKILLGR_EVAL_BATCH}" \
  "${SKILLGR_EVAL_STEPS}" <<'PY'
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
eval_batch_size = int(sys.argv[8])
eval_steps = int(sys.argv[9])

predictions = [
    json.loads(line)
    for line in (eval_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(predictions) != 75:
    raise SystemExit(f"Official core75 prediction count mismatch: {len(predictions)}")
metrics = {
    subset: json.loads(
        (eval_dir / f"ranked_metrics_{metric_subset}.json").read_text(encoding="utf-8")
    )["overall"]
    for subset, metric_subset in (("overall", "all"), ("single", "single"), ("multi", "multi"))
}
summary = {
    "arm": arm,
    "catalog_epochs": catalog_epochs,
    "seed": seed,
    "source_commit": legacy_commit,
    "protocol": {
        "global_batch_size": 16,
        "eval_batch_size": eval_batch_size,
        "eval_steps": eval_steps,
        "query_epochs": 5,
        "lr": 0.0003,
        "scheduler": "constant",
        "weight_decay": 0.0,
        "precision": "bf16",
        "selected_checkpoint": "dev_best",
        "evaluation": "Official core75, beam=50, beam_order, CE log-prob",
    },
    **metrics,
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
