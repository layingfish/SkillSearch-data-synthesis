#!/usr/bin/env bash
set -euo pipefail

CATALOG_EPOCHS="${1:?usage: $0 0|3|6|12 [seed]}"
RUN_SEED="${2:-20260626}"
case "${CATALOG_EPOCHS}" in
  0|3|6|12) ;;
  *) echo "catalog epochs must be one of 0, 3, 6, 12" >&2; exit 2 ;;
esac
[[ "${RUN_SEED}" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }

ARM="legacy-c${CATALOG_EPOCHS}-q5-seed${RUN_SEED}"
PHYSICAL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260807_v2
mkdir -p "${PHYSICAL_OUTPUT_ROOT}/logs"
exec > >(tee -a "${PHYSICAL_OUTPUT_ROOT}/logs/${ARM}.log") 2>&1

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
if [[ "${SKILLGR_NODE_PREFLIGHT_DONE:-0}" != 1 ]]; then
  bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1
fi

ROOT_DIR="${SKILLGR_CODE_ROOT:-/workspace/SkillSearch-data-synthesis}"
if [[ "${SKILLGR_NODE_PREFLIGHT_DONE:-0}" != 1 ]]; then
  git config --global --add safe.directory "${ROOT_DIR}"
  git config --global --add safe.directory "$(readlink -f "${ROOT_DIR}")"
fi
cd "${ROOT_DIR}"

set -a
source configs/legacy_catalog_sweep_20260807.env
set +a
SKILLGR_SEED="${RUN_SEED}"
export SKILLGR_SEED

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SKILLGR_MODEL

OUTPUT_DIR="${SKILLGR_OUTPUT_ROOT}/${ARM}"
EVAL_DIR="${SKILLGR_OUTPUT_ROOT}/eval/${ARM}"
DONE_MARKER="${SKILLGR_OUTPUT_ROOT}/${ARM}.done"

require_file() {
  [[ -f "$1" ]] || { echo "required file is missing: $1" >&2; exit 1; }
}

require_absent() {
  [[ ! -e "$1" ]] || { echo "refusing to overwrite experiment artifact: $1" >&2; exit 1; }
}

for required in \
  "${SKILLGR_ROWS}" \
  "${SKILLGR_DEV_ROWS}" \
  "${SKILLGR_PRIVATE_TEST_ROWS}" \
  "${SKILLGR_OFFICIAL_ROWS}" \
  "${SKILLGR_SKILLS}" \
  "${SKILLGR_SEMANTIC_IDS}" \
  "${SKILLGR_MODEL}/config.json"; do
  require_file "${required}"
done
if [[ "${CATALOG_EPOCHS}" == 3 && "${SKILLGR_SEED}" == 20260626 ]]; then
  require_file "${SKILLGR_ARCHIVED_OFFICIAL}"
fi
require_absent "${OUTPUT_DIR}"
require_absent "${EVAL_DIR}"
require_absent "${DONE_MARKER}"
mkdir -p "${SKILLGR_OUTPUT_ROOT}" "$(dirname "${EVAL_DIR}")"

if [[ "${SKILLGR_NODE_PREFLIGHT_DONE:-0}" != 1 ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "legacy sweep requires a clean immutable checkout" >&2
    git status --short
    exit 1
  fi
  git rev-parse HEAD
  python3 docker/smoke_test.py
  python3 -m unittest \
    tests.test_skill_gr_pipeline.SkillGRPipelineTest.test_ripor_head_initialization_preserves_legacy_ce_rng_and_parameters
fi

python3 scripts/skill_gr_pipeline.py train-single-sid-decoder \
  --rows "${SKILLGR_ROWS}" \
  --dev-rows "${SKILLGR_DEV_ROWS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --skills-jsonl "${SKILLGR_SKILLS}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${SKILLGR_MODEL}" \
  --query-loss expanded_ce \
  --catalog-epochs "${CATALOG_EPOCHS}" \
  --epochs "${SKILLGR_QUERY_EPOCHS}" \
  --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
  --batch-size "${SKILLGR_TRAIN_BATCH}" \
  --eval-batch-size "${SKILLGR_EVAL_BATCH}" \
  --lr "${SKILLGR_LR}" \
  --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
  --lr-scheduler constant \
  --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
  --logging-steps 50 \
  --eval-steps 1000 \
  --save-steps 0 \
  --no-padding-distributed-sampler \
  --seed "${SKILLGR_SEED}" \
  --load-best-model-at-end \
  --bf16

python3 - "${OUTPUT_DIR}" "${CATALOG_EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
catalog_epochs = int(sys.argv[2])
manifest = json.loads((output / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8"))
if manifest.get("ddp_world_size") != 1 or manifest.get("global_batch_size") != 16:
    raise SystemExit("training did not use the registered single-process global batch 16")
if manifest.get("catalog_steps") != 910 * catalog_epochs:
    raise SystemExit("catalog optimizer-step count mismatch")
if manifest.get("query_steps") != 40420:
    raise SystemExit("query optimizer-step count mismatch")
if manifest.get("ripor_head_trainable") is not False:
    raise SystemExit("CE parity arm unexpectedly trained RIPOR heads")
if manifest.get("ripor_head_initialization_preserves_ce_rng") is not True:
    raise SystemExit("checkpoint does not record CE RNG-preserving auxiliary initialization")

events = [
    json.loads(line)
    for line in (output / "loss_history.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
catalog = [row for row in events if row.get("event") == "epoch" and row.get("stage") == "catalog"]
query = [row for row in events if row.get("event") == "epoch" and row.get("stage") == "query"]
if len(catalog) != catalog_epochs or any(row.get("ce_token_count") != 72755 for row in catalog):
    raise SystemExit("catalog exposure invariant failed")
if len(query) != 5 or any(row.get("ce_token_count") != 646705 for row in query):
    raise SystemExit("query exposure invariant failed")
print(json.dumps({"catalog_epochs": catalog_epochs, "catalog_steps": 910 * catalog_epochs, "query_steps": 40420}))
PY

eval_split() {
  local split="$1"
  local rows="$2"
  local output="${EVAL_DIR}/${split}"
  python3 scripts/skill_gr_pipeline.py eval-single-sid-decoder \
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

eval_split private_test_1000 "${SKILLGR_PRIVATE_TEST_ROWS}"
eval_split official_core75 "${SKILLGR_OFFICIAL_ROWS}"

python3 - \
  "${ARM}" \
  "${CATALOG_EPOCHS}" \
  "${SKILLGR_SEED}" \
  "${OUTPUT_DIR}" \
  "${EVAL_DIR}" \
  "${SKILLGR_ARCHIVED_OFFICIAL}" <<'PY'
import json
import sys
from pathlib import Path

arm = sys.argv[1]
catalog_epochs = int(sys.argv[2])
seed = int(sys.argv[3])
output = Path(sys.argv[4])
eval_dir = Path(sys.argv[5])
archive = Path(sys.argv[6])

metrics = {}
for split, expected_rows in (("private_test_1000", 1000), ("official_core75", 75)):
    split_dir = eval_dir / split
    predictions = [
        json.loads(line)
        for line in (split_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(predictions) != expected_rows:
        raise SystemExit(f"{split} prediction count mismatch")
    metrics[split] = {
        subset: json.loads((split_dir / f"ranked_metrics_{subset}.json").read_text(encoding="utf-8"))["overall"]
        for subset in ("all", "single", "multi")
    }

summary = {
    "arm": arm,
    "catalog_epochs": catalog_epochs,
    "protocol": {
        "training": "single_process_continuous_catalog_then_query",
        "global_batch_size": 16,
        "query_epochs": 5,
        "lr": 0.0003,
        "scheduler": "constant",
        "warmup_ratio": 0.0,
        "seed": seed,
        "selected_checkpoint": "dev_best",
    },
    "metrics": metrics,
}

if catalog_epochs == 3 and seed == 20260626:
    archived = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines() if line.strip()]
    current = [
        json.loads(line)
        for line in (eval_dir / "official_core75" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(archived) != len(current):
        raise SystemExit("archived/current Official prediction count mismatch")
    mismatches = [
        index
        for index, (left, right) in enumerate(zip(archived, current))
        if left.get("ranked_skill_ids") != right.get("ranked_skill_ids")
    ]
    summary["notion_cat3_parity"] = {
        "archived_rows": len(archived),
        "exact_top50_rows": len(archived) - len(mismatches),
        "mismatch_indices": mismatches,
        "exact_match": not mismatches,
    }

(output / "result_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
if catalog_epochs == 3 and seed == 20260626 and mismatches:
    raise SystemExit(f"Notion cat3 top-50 parity failed on {len(mismatches)} / {len(archived)} rows")
PY

touch "${DONE_MARKER}"
