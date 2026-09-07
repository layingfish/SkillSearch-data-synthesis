#!/usr/bin/env bash
set -euo pipefail

cfs_agent_root="${LETTER_OPQ_CFS_AGENT_ROOT:-/mnt/cfs-agent}"
project_root="${RIPOR_PROJECT_ROOT:-$cfs_agent_root/workspace/chenyufei03/SkillSearch-data-synthesis}"
rows="${LETTER_OPQ_EVAL_ROWS:?LETTER_OPQ_EVAL_ROWS is required}"
split_name="${LETTER_OPQ_EVAL_SPLIT:?LETTER_OPQ_EVAL_SPLIT is required}"
semantic_ids="${LETTER_OPQ_EVAL_SEMANTIC_IDS:?LETTER_OPQ_EVAL_SEMANTIC_IDS is required}"
checkpoint="${LETTER_OPQ_EVAL_CHECKPOINT:?LETTER_OPQ_EVAL_CHECKPOINT is required}"
output_dir="${LETTER_OPQ_EVAL_OUTPUT:?LETTER_OPQ_EVAL_OUTPUT is required}"
ddp_launcher="$project_root/repro/20260903_letter_opq_ceiling/run_ddp_command.sh"

cd "$project_root"
export PYTHONPATH="$project_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
for required in "$rows" "$semantic_ids" "$checkpoint" "$ddp_launcher"; do
  [[ -f "$required" ]] || { echo "missing exact-eval input: $required" >&2; exit 1; }
done
if [[ -e "$output_dir" && ! -d "$output_dir" ]]; then
  echo "exact-eval output is not a directory: $output_dir" >&2
  exit 1
fi
if [[ ! -f "$output_dir/eval_manifest.json" || ! -f "$output_dir/ranked_metrics.json" ]]; then
  if [[ -d "$output_dir" && -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "partial exact-eval output exists; use a fresh directory: $output_dir" >&2
    exit 1
  fi
  bash "$ddp_launcher" scripts/skill_gr_opq_parallel.py eval \
    --rows "$rows" \
    --semantic-ids "$semantic_ids" \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --split-name "$split_name" \
    --batch-size 16 \
    --k-values 1,5,10,50
fi
