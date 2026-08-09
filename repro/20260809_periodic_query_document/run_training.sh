#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/root/workplace/SkillSearch-data-synthesis}"
output_dir="${2:-$repo_root/experiments/query20_document_after_each_query_official_selection_rtxpro6000_20260809}"
asset_root="$repo_root/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713"
input_root="$asset_root/canonical_alias_core75_learned_d4_inputs"

train_rows="$input_root/rows_000.decoder_rows.base_plus_task5_gt2.jsonl"
official_rows="$input_root/rows_003.skillsbench_core75.rows.jsonl"
skills="$input_root/canonical_skills.jsonl"
semantic_ids="$asset_root/letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/semantic_ids.jsonl"

for required in "$train_rows" "$official_rows" "$skills" "$semantic_ids"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

if [[ -e "$output_dir/single_sid_decoder_train_manifest.json" ]]; then
  echo "refusing to overwrite completed run: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python3 "$repo_root/scripts/skill_gr_pipeline.py" train-single-sid-decoder \
  --rows "$train_rows" \
  --semantic-ids "$semantic_ids" \
  --skills-jsonl "$skills" \
  --official-eval-rows "$official_rows" \
  --output-dir "$output_dir" \
  --model t5-base \
  --catalog-epochs 0 \
  --periodic-document-epochs 1 \
  --epochs 20 \
  --query-batch-size 48 \
  --document-batch-size 16 \
  --disable-dev-eval \
  --official-eval-every-epochs 2 \
  --official-eval-num-beams 50 \
  --official-eval-k-values 1,10,50 \
  --official-selection-subset overall \
  --official-selection-k 1 \
  --lr 3e-4 \
  --weight-decay 0 \
  --max-source-length 2048 \
  --logging-steps 50 \
  --seed 20260626 \
  --bf16 \
  --wandb-project SKillGR \
  --wandb-entity cyf04 \
  --wandb-group periodic-query-document \
  --wandb-run-name 94k-query20-document-after-each-query-official-r1 \
  --wandb-tags 94k,periodic-document,query-20,official-selected,test-oracle,rtx-pro-6000
