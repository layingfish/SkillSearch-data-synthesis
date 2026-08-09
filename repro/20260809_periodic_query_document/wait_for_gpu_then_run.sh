#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/root/workplace/SkillSearch-data-synthesis}"
output_dir="${2:-$repo_root/experiments/query20_document_after_each_query_official_selection_rtxpro6000_20260809}"
gpu_index="${GPU_INDEX:-0}"
poll_seconds="${GPU_POLL_SECONDS:-20}"
required_free_checks="${GPU_REQUIRED_FREE_CHECKS:-3}"
max_used_mib="${GPU_MAX_USED_MIB:-1024}"
max_utilization="${GPU_MAX_UTILIZATION:-5}"
launch_script="$repo_root/repro/20260809_periodic_query_document/run_training.sh"

free_checks=0
while true; do
  gpu_state="$(
    nvidia-smi \
      --id="$gpu_index" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )"
  used_mib="${gpu_state%%,*}"
  utilization="${gpu_state##*,}"
  used_mib="${used_mib//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  timestamp="$(date --iso-8601=seconds)"

  if (( used_mib <= max_used_mib && utilization <= max_utilization )); then
    free_checks=$((free_checks + 1))
    echo "$timestamp gpu_free_check=$free_checks/$required_free_checks used_mib=$used_mib utilization=$utilization"
  else
    free_checks=0
    echo "$timestamp gpu_busy used_mib=$used_mib utilization=$utilization"
  fi

  if (( free_checks >= required_free_checks )); then
    echo "$timestamp gpu_confirmed_free launching_training output_dir=$output_dir"
    mkdir -p "$output_dir"
    exec bash "$launch_script" "$repo_root" "$output_dir" >"$output_dir/train.log" 2>&1
  fi
  sleep "$poll_seconds"
done
