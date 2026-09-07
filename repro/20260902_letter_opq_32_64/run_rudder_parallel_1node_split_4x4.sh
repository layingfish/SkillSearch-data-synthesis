#!/usr/bin/env bash
set -euo pipefail

cfs_agent_root="${LETTER_OPQ_CFS_AGENT_ROOT:-/mnt/cfs-agent}"
project_root="${RIPOR_PROJECT_ROOT:-$cfs_agent_root/workspace/chenyufei03/SkillSearch-data-synthesis}"
model_root="$cfs_agent_root/dataset/chenyufei03/models/experiments/chenyufei03/SkillSearch-data-synthesis"
variant_runner="$project_root/repro/20260902_letter_opq_32_64/run_rudder_parallel_mtp.sh"
run_tag="${LETTER_OPQ_RUN_TAG:-rudder_r2_parallel_mtp_4gpu_20260902_v1}"
output_root="$model_root/letter_opq_parallel_runs/$run_tag"
log_root="$model_root/logs/$run_tag"

[[ -f "$variant_runner" ]] || {
  echo "missing OPQ parallel variant runner: $variant_runner" >&2
  exit 1
}
mkdir -p "$output_root" "$log_root"

declare -a child_pids=()
terminate_children() {
  local pid
  for pid in "${child_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap terminate_children INT TERM HUP

run_variant() {
  local depth="$1"
  local gpu_mask="$2"
  local master_port="$3"
  local experiment="$output_root/opq_m${depth}_parallel_t5base_seed20260626"
  local log_path="$log_root/opq_m${depth}_parallel.log"

  echo "letter_opq_parallel_start depth=$depth gpu_mask=$gpu_mask master_port=$master_port experiment=$experiment"
  env \
    CUDA_VISIBLE_DEVICES="$gpu_mask" \
    LETTER_OPQ_DEPTH="$depth" \
    LETTER_OPQ_WORLD_SIZE=4 \
    LETTER_OPQ_PER_DEVICE_BATCH_SIZE=4 \
    LETTER_OPQ_MASTER_PORT="$master_port" \
    LETTER_OPQ_CFS_AGENT_ROOT="$cfs_agent_root" \
    RIPOR_PROJECT_ROOT="$project_root" \
    RIPOR_EXPERIMENT="$experiment" \
    bash "$variant_runner" 2>&1 | tee -a "$log_path"
}

run_variant 32 0,1,2,3 29632 &
pid32="$!"
child_pids+=("$pid32")
run_variant 64 4,5,6,7 29664 &
pid64="$!"
child_pids+=("$pid64")

set +e
wait "$pid32"
status32=$?
wait "$pid64"
status64=$?
set -e

trap - INT TERM HUP
echo "letter_opq_parallel_exit depth=32 status=$status32"
echo "letter_opq_parallel_exit depth=64 status=$status64"
if [[ "$status32" -ne 0 || "$status64" -ne 0 ]]; then
  exit 1
fi
