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
cd "${ROOT_DIR}"
set -a
source configs/legacy_catalog_sweep_20260807.env
set +a

ARM="legacy-c${CATALOG_EPOCHS}-q5-seed${RUN_SEED}"
OUTPUT_DIR="${SKILLGR_OUTPUT_ROOT}/${ARM}"
EVAL_DIR="${SKILLGR_OUTPUT_ROOT}/eval/${ARM}"
DONE_MARKER="${SKILLGR_OUTPUT_ROOT}/${ARM}.done"

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SKILLGR_MODEL

python3 - "${OUTPUT_DIR}" "${CATALOG_EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
catalog_epochs = int(sys.argv[2])
manifest = json.loads((output / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8"))
expected = {
    "ddp_world_size": 1,
    "global_batch_size": 16,
    "catalog_steps": 910 * catalog_epochs,
    "query_steps": 40420,
    "ripor_head_trainable": False,
    "ripor_head_initialization_preserves_ce_rng": True,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"training invariant failed for {key}: {manifest.get(key)!r} != {value!r}")
if not (output / "final" / "structured_decoder.pt").is_file():
    raise SystemExit("final model checkpoint is missing")
PY

eval_split() {
  local split="$1"
  local rows="$2"
  local expected_rows="$3"
  local output="${EVAL_DIR}/${split}"
  local predictions="${output}/predictions.jsonl"
  if [[ ! -f "${predictions}" ]]; then
    [[ ! -e "${output}" ]] || { echo "partial evaluation without predictions: ${output}" >&2; exit 1; }
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
  fi
  [[ "$(wc -l < "${predictions}")" == "${expected_rows}" ]] || {
    echo "prediction count mismatch for ${split}" >&2
    exit 1
  }
  local subset
  for subset in all single multi; do
    if [[ ! -f "${output}/ranked_metrics_${subset}.json" ]]; then
      python3 scripts/skill_gr_pipeline.py rescore-ranked \
        --predictions "${predictions}" \
        --subset "${subset}" \
        --k-values 1,10,50 \
        --output "${output}/ranked_metrics_${subset}.json"
    fi
  done
}

eval_split private_test_1000 "${SKILLGR_PRIVATE_TEST_ROWS}" 1000
eval_split official_core75 "${SKILLGR_OFFICIAL_ROWS}" 75

python3 - \
  "${ARM}" \
  "${CATALOG_EPOCHS}" \
  "${RUN_SEED}" \
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
for split in ("private_test_1000", "official_core75"):
    split_dir = eval_dir / split
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
PY

touch "${DONE_MARKER}"
