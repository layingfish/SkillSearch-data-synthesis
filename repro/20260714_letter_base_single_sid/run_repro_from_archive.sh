#!/usr/bin/env bash
set -euo pipefail

cd /root/workplace/SkillSearch-data-synthesis
out=outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/single_sid_d4_lightgcn_d32_letterofficial_task5_core75_gt2_t5base_cat3_querye5_b16
if [[ -e "$out" ]]; then
  echo "OUTPUT_EXISTS $out" >&2
  exit 2
fi
mkdir -p "$out"
nohup env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  python3 repro/20260714_letter_base_single_sid/skill_gr_pipeline.original_20260714.py train-single-sid-decoder \
  --rows outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/rows_000.decoder_rows.base_plus_task5_gt2.jsonl \
  --dev-rows outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/rows_001.decoder_rows.base.jsonl \
  --semantic-ids outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/semantic_ids.jsonl \
  --skills-jsonl outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/canonical_skills.jsonl \
  --output-dir "$out" \
  --model t5-base \
  --catalog-epochs 3 \
  --epochs 5 \
  --batch-size 16 \
  --lr 3e-4 \
  --weight-decay 0 \
  --max-source-length 2048 \
  --logging-steps 50 \
  --eval-steps 200 \
  --seed 20260626 \
  --bf16 > "$out/train.log" 2>&1 &
echo $! > "$out/train.pid"
echo "STARTED pid=$(cat "$out/train.pid") out=$out"
