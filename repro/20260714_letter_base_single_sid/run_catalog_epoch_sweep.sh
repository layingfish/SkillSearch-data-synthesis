#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/root/workplace/SkillSearch-data-synthesis}"
experiment_root="${2:-$repo_root/experiments/catalog_epoch_sweep_primary94912_rtxpro6000_20260809}"

pipeline="$repo_root/repro/20260714_letter_base_single_sid/skill_gr_pipeline.original_20260714.py"
asset_root="$repo_root/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713"
input_root="$asset_root/canonical_alias_core75_learned_d4_inputs"
semantic_ids="$asset_root/letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/semantic_ids.jsonl"
train_rows="$input_root/rows_000.decoder_rows.base_plus_task5_gt2.jsonl"
dev_rows="$input_root/rows_001.decoder_rows.base.jsonl"
official_rows="$input_root/rows_003.skillsbench_core75.rows.jsonl"
skills="$input_root/canonical_skills.jsonl"

expected_pipeline_sha="7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76"
actual_pipeline_sha="$(sha256sum "$pipeline" | awk '{print $1}')"
if [[ "$actual_pipeline_sha" != "$expected_pipeline_sha" ]]; then
  echo "pipeline SHA256 mismatch: $actual_pipeline_sha" >&2
  exit 1
fi

for required in "$train_rows" "$dev_rows" "$official_rows" "$skills" "$semantic_ids"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$experiment_root"

for catalog_epochs in 0 6 9 12; do
  run_name="catalog-epochs-${catalog_epochs}-query-epochs-5-seed-20260626"
  train_dir="$experiment_root/$run_name/train"
  eval_dir="$experiment_root/$run_name/official_core75_beam50"

  if [[ -e "$train_dir" || -e "$eval_dir" ]]; then
    echo "refusing to overwrite existing run: $experiment_root/$run_name" >&2
    exit 1
  fi

  mkdir -p "$train_dir" "$eval_dir"
  echo "START catalog_epochs=$catalog_epochs time=$(date --iso-8601=seconds)"

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python3 "$pipeline" train-single-sid-decoder \
    --rows "$train_rows" \
    --dev-rows "$dev_rows" \
    --semantic-ids "$semantic_ids" \
    --skills-jsonl "$skills" \
    --output-dir "$train_dir" \
    --model t5-base \
    --catalog-epochs "$catalog_epochs" \
    --epochs 5 \
    --batch-size 16 \
    --lr 3e-4 \
    --weight-decay 0 \
    --max-source-length 2048 \
    --logging-steps 50 \
    --eval-steps 200 \
    --seed 20260626 \
    --bf16 >"$train_dir/train.log" 2>&1

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python3 "$pipeline" eval-single-sid-decoder \
    --rows "$official_rows" \
    --model-dir "$train_dir/final" \
    --semantic-ids "$semantic_ids" \
    --output-dir "$eval_dir" \
    --split-name skillsbench_core75 \
    --max-source-length 2048 \
    --num-beams 50 \
    --k-values 1,10,50 \
    --rank-aggregation beam_order \
    --log-every 10 >"$eval_dir/eval.log" 2>&1

  python3 "$pipeline" rescore-ranked \
    --predictions "$eval_dir/predictions.jsonl" \
    --output "$eval_dir/ranked_metrics.single.json" \
    --subset single \
    --k-values 1,10,50 >>"$eval_dir/eval.log" 2>&1

  python3 "$pipeline" rescore-ranked \
    --predictions "$eval_dir/predictions.jsonl" \
    --output "$eval_dir/ranked_metrics.multi.json" \
    --subset multi \
    --k-values 1,10,50 >>"$eval_dir/eval.log" 2>&1

  touch "$eval_dir/COMPLETE"
  echo "DONE catalog_epochs=$catalog_epochs time=$(date --iso-8601=seconds)"
done

python3 - "$experiment_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for catalog_epochs in (0, 6, 9, 12):
    eval_dir = root / f"catalog-epochs-{catalog_epochs}-query-epochs-5-seed-20260626" / "official_core75_beam50"
    result = {"catalog_epochs": catalog_epochs}
    for subset, filename in (
        ("overall", "ranked_metrics.json"),
        ("single", "ranked_metrics.single.json"),
        ("multi", "ranked_metrics.multi.json"),
    ):
        metrics = json.loads((eval_dir / filename).read_text())["overall"]
        result[subset] = {f"R@{k}": metrics[f"@{k}"]["macro_recall"] for k in (1, 10, 50)}
    print(json.dumps(result, sort_keys=True))
PY
