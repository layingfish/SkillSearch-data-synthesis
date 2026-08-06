#!/usr/bin/env bash
set -euo pipefail

ARM="${1:?usage: $0 query-only|mixed-c3|mixed-c6|mixed-c12}"
case "${ARM}" in
  query-only) CATALOG_PASSES=0; EXPECTED_SCHEDULE=0,0,0,0,0; EXPECTED_STEPS=40420 ;;
  mixed-c3) CATALOG_PASSES=3; EXPECTED_SCHEDULE=1,0,1,0,1; EXPECTED_STEPS=43150 ;;
  mixed-c6) CATALOG_PASSES=6; EXPECTED_SCHEDULE=1,1,2,1,1; EXPECTED_STEPS=45879 ;;
  mixed-c12) CATALOG_PASSES=12; EXPECTED_SCHEDULE=2,3,2,3,2; EXPECTED_STEPS=51335 ;;
  *) echo "unknown arm: ${ARM}" >&2; exit 2 ;;
esac

PHYSICAL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/document_augmentation_exposure_20260806_v1
mkdir -p "${PHYSICAL_OUTPUT_ROOT}/logs"
exec > >(tee -a "${PHYSICAL_OUTPUT_ROOT}/logs/${ARM}.log") 2>&1

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1

ROOT_DIR="${SKILLGR_CODE_ROOT:-/workspace/SkillSearch-data-synthesis}"
git config --global --add safe.directory "${ROOT_DIR}"
git config --global --add safe.directory "$(readlink -f "${ROOT_DIR}")"
cd "${ROOT_DIR}"

set -a
source configs/document_augmentation_exposure_20260806.env
set +a

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SKILLGR_MODEL

OUTPUT_DIR="${SKILLGR_OUTPUT_ROOT}/${ARM}"
EVAL_DIR="${SKILLGR_OUTPUT_ROOT}/eval/${ARM}"
DONE_MARKER="${SKILLGR_OUTPUT_ROOT}/${ARM}.done"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file is missing: $1" >&2
    exit 1
  fi
}

require_absent() {
  if [[ -e "$1" ]]; then
    echo "refusing to overwrite existing experiment artifact: $1" >&2
    exit 1
  fi
}

run_torch() {
  torchrun --standalone --nproc_per_node="${SKILLGR_WORLD_SIZE}" "$@"
}

record_runtime() {
  git status --short
  git rev-parse HEAD
  python3 docker/smoke_test.py
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
}

verify_schedule_contract() {
  python3 - \
    "${SKILLGR_QUERY_EPOCHS}" \
    "${CATALOG_PASSES}" \
    "${EXPECTED_SCHEDULE}" \
    "${SKILLGR_EXPECTED_QUERY_ROWS}" \
    "${SKILLGR_EXPECTED_CATALOG_ROWS}" \
    "${SKILLGR_WORLD_SIZE}" \
    "${SKILLGR_PER_DEVICE_BATCH}" \
    "${EXPECTED_STEPS}" <<'PY'
import json
import math
import sys

sys.path.insert(0, "scripts")
from skill_gr_pipeline import document_augmentation_catalog_pass_schedule

query_epochs = int(sys.argv[1])
catalog_passes = int(sys.argv[2])
expected_schedule = [int(value) for value in sys.argv[3].split(",")]
query_rows = int(sys.argv[4])
catalog_rows = int(sys.argv[5])
world_size = int(sys.argv[6])
per_device_batch = int(sys.argv[7])
expected_steps = int(sys.argv[8])
schedule = document_augmentation_catalog_pass_schedule(query_epochs, catalog_passes)
if schedule != expected_schedule:
    raise SystemExit(f"catalog schedule {schedule} != {expected_schedule}")
global_batch = world_size * per_device_batch
steps = sum(math.ceil((query_rows + passes * catalog_rows) / global_batch) for passes in schedule)
if steps != expected_steps:
    raise SystemExit(f"optimizer steps {steps} != {expected_steps}")
print(json.dumps({"catalog_passes": catalog_passes, "schedule": schedule, "steps": steps}, sort_keys=True))
PY
}

verify_training_artifacts() {
  python3 - \
    "${OUTPUT_DIR}" \
    "${CATALOG_PASSES}" \
    "${EXPECTED_SCHEDULE}" \
    "${EXPECTED_STEPS}" \
    "${SKILLGR_EXPECTED_QUERY_ROWS}" \
    "${SKILLGR_EXPECTED_CATALOG_ROWS}" \
    "${SKILLGR_QUERY_EPOCHS}" \
    "${SKILLGR_TARGET_TOKENS_PER_ROW}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
catalog_passes = int(sys.argv[2])
expected_schedule = [int(value) for value in sys.argv[3].split(",")]
expected_steps = int(sys.argv[4])
query_rows = int(sys.argv[5])
catalog_rows = int(sys.argv[6])
query_epochs = int(sys.argv[7])
target_tokens = int(sys.argv[8])

manifest_path = output_dir / "single_sid_decoder_train_manifest.json"
history_path = output_dir / "loss_history.jsonl"
checkpoint_path = output_dir / "final" / "structured_decoder.pt"
for path in (manifest_path, history_path, checkpoint_path):
    if not path.is_file():
        raise SystemExit(f"missing training artifact: {path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
augmentation = manifest.get("document_augmentation") or {}
expected_augmentation = {
    "enabled": True,
    "query_rows": query_rows,
    "catalog_rows": catalog_rows,
    "query_epochs": query_epochs,
    "catalog_epochs": catalog_passes,
    "catalog_pass_schedule": expected_schedule,
    "query_samples": query_rows * query_epochs,
    "catalog_samples": catalog_rows * catalog_passes,
}
mismatches = {
    key: {"expected": value, "observed": augmentation.get(key)}
    for key, value in expected_augmentation.items()
    if augmentation.get(key) != value
}
if mismatches:
    raise SystemExit(f"document augmentation metadata mismatch: {mismatches}")
if manifest.get("query_train_rows") != query_rows or manifest.get("catalog_rows") != catalog_rows:
    raise SystemExit("materialized training row counts do not match the registered protocol")
if manifest.get("query_steps") != expected_steps or manifest.get("catalog_steps") != 0:
    raise SystemExit("training step counts do not match the registered protocol")
if (manifest.get("optimizer_schedule") or {}).get("total_steps") != expected_steps:
    raise SystemExit("optimizer schedule total does not match the mixed-stream steps")
if manifest.get("init_model_dir") is not None:
    raise SystemExit("document augmentation arm did not start from the registered base model")
if manifest.get("load_best_model_at_end") is not False:
    raise SystemExit("final checkpoint selection is not the last training state")

events = [
    json.loads(line)
    for line in history_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
epochs = [row for row in events if row.get("stage") == "query" and row.get("event") == "epoch"]
if len(epochs) != query_epochs:
    raise SystemExit(f"query epoch event count {len(epochs)} != {query_epochs}")
for index, (event, passes) in enumerate(zip(epochs, expected_schedule), start=1):
    expected_ce_tokens = (query_rows + passes * catalog_rows) * target_tokens
    expected = {
        "epoch": index,
        "document_augmentation_query_rows": query_rows,
        "document_augmentation_catalog_rows": passes * catalog_rows,
        "document_augmentation_catalog_passes": passes,
        "ce_token_count": expected_ce_tokens,
    }
    bad = {key: {"expected": value, "observed": event.get(key)} for key, value in expected.items() if event.get(key) != value}
    if bad:
        raise SystemExit(f"epoch {index} exposure mismatch: {bad}")

verification = {
    "catalog_passes": catalog_passes,
    "catalog_pass_schedule": expected_schedule,
    "query_rows_per_epoch": query_rows,
    "catalog_rows": catalog_rows,
    "optimizer_steps": expected_steps,
    "query_ce_tokens": query_rows * query_epochs * target_tokens,
    "catalog_ce_tokens": catalog_rows * catalog_passes * target_tokens,
    "final_checkpoint": str(output_dir / "final"),
}
(output_dir / "exposure_verification.json").write_text(
    json.dumps(verification, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(verification, sort_keys=True))
PY
}

eval_split() {
  local split="$1"
  local rows="$2"
  local output="${EVAL_DIR}/${split}"
  require_file "${OUTPUT_DIR}/final/structured_decoder.pt"
  require_absent "${output}"
  run_torch scripts/skill_gr_pipeline.py eval-single-sid-decoder \
    --rows "${rows}" \
    --model-dir "${OUTPUT_DIR}/final" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${output}" \
    --split-name "${split}" \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --num-beams "${SKILLGR_NUM_BEAMS}" \
    --rank-aggregation beam_order \
    --k-values 1,10,50 \
    --log-every 25

  local subset
  for subset in all single multi; do
    python3 scripts/skill_gr_pipeline.py rescore-ranked \
      --predictions "${output}/predictions.jsonl" \
      --subset "${subset}" \
      --k-values 1,10,50 \
      --output "${output}/ranked_metrics_${subset}.json"
  done
}

write_result_summary() {
  python3 - "${ARM}" "${OUTPUT_DIR}" "${EVAL_DIR}" "${CATALOG_PASSES}" "${SKILLGR_NUM_BEAMS}" <<'PY'
import json
import sys
from pathlib import Path

arm = sys.argv[1]
output_dir = Path(sys.argv[2])
eval_dir = Path(sys.argv[3])
catalog_passes = int(sys.argv[4])
num_beams = int(sys.argv[5])
metrics = {}
expected_rows = {"private_test_1000": 1000, "official_core75": 75}
for split, row_count in expected_rows.items():
    split_dir = eval_dir / split
    predictions = [line for line in (split_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(predictions) != row_count:
        raise SystemExit(f"{split} prediction count {len(predictions)} != {row_count}")
    metrics[split] = {}
    for subset in ("all", "single", "multi"):
        payload = json.loads((split_dir / f"ranked_metrics_{subset}.json").read_text(encoding="utf-8"))
        metrics[split][subset] = payload["overall"]
    if metrics[split]["all"]["rows"] != row_count:
        raise SystemExit(f"{split} overall metric rows do not match predictions")
    if metrics[split]["single"]["rows"] + metrics[split]["multi"]["rows"] != row_count:
        raise SystemExit(f"{split} single/multi subsets do not partition the evaluation rows")

verification = json.loads((output_dir / "exposure_verification.json").read_text(encoding="utf-8"))
summary = {
    "arm": arm,
    "catalog_passes": catalog_passes,
    "training": verification,
    "evaluation": {
        "num_beams": num_beams,
        "rank_aggregation": "beam_order",
        "k_values": [1, 10, 50],
        "recall_aggregation": "macro",
        "metrics": metrics,
    },
}
(output_dir / "result_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY
}

for required in \
  "${SKILLGR_ROWS}" \
  "${SKILLGR_DEV_ROWS}" \
  "${SKILLGR_PRIVATE_TEST_ROWS}" \
  "${SKILLGR_EVAL_ROWS}" \
  "${SKILLGR_SKILLS}" \
  "${SKILLGR_SEMANTIC_IDS}" \
  "${SKILLGR_MODEL}/config.json"; do
  require_file "${required}"
done
require_absent "${OUTPUT_DIR}"
require_absent "${EVAL_DIR}"
require_absent "${DONE_MARKER}"
mkdir -p "${SKILLGR_OUTPUT_ROOT}" "$(dirname "${EVAL_DIR}")"

record_runtime
verify_schedule_contract
run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
  --rows "${SKILLGR_ROWS}" \
  --dev-rows "${SKILLGR_DEV_ROWS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --skills-jsonl "${SKILLGR_SKILLS}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${SKILLGR_MODEL}" \
  --query-loss expanded_ce \
  --catalog-epochs 0 \
  --catalog-max-steps 0 \
  --document-augmentation \
  --document-augmentation-catalog-epochs "${CATALOG_PASSES}" \
  --epochs "${SKILLGR_QUERY_EPOCHS}" \
  --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
  --batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
  --eval-batch-size "${SKILLGR_EVAL_PER_DEVICE_BATCH}" \
  --lr "${SKILLGR_LR}" \
  --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
  --lr-scheduler constant \
  --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
  --logging-steps "${SKILLGR_LOGGING_STEPS}" \
  --eval-steps "${SKILLGR_EVAL_STEPS}" \
  --save-steps 0 \
  --seed "${SKILLGR_SEED}" \
  --no-load-best-model-at-end \
  --bf16

verify_training_artifacts
eval_split private_test_1000 "${SKILLGR_PRIVATE_TEST_ROWS}"
eval_split official_core75 "${SKILLGR_EVAL_ROWS}"
write_result_summary
touch "${DONE_MARKER}"
