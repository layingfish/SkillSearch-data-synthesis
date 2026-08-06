#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKILLGR_CODE_ROOT:-/workspace/SkillSearch-data-synthesis}"
PHYSICAL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260807_v2
ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh

mkdir -p "${PHYSICAL_OUTPUT_ROOT}/logs"
exec > >(tee -a "${PHYSICAL_OUTPUT_ROOT}/logs/node-8gpu.log") 2>&1

bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1
git config --global --add safe.directory "${ROOT_DIR}"
git config --global --add safe.directory "$(readlink -f "${ROOT_DIR}")"
cd "${ROOT_DIR}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "legacy sweep requires a clean immutable checkout" >&2
  git status --short
  exit 1
fi
git rev-parse HEAD
python3 docker/smoke_test.py
python3 -m unittest \
  tests.test_skill_gr_pipeline.SkillGRPipelineTest.test_ripor_head_initialization_preserves_legacy_ce_rng_and_parameters

export SKILLGR_NODE_PREFLIGHT_DONE=1
declare -a PIDS=()
declare -A LABEL_BY_PID=()

terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
}
trap 'terminate_children; exit 130' INT TERM

launch_arm() {
  local gpu="$1"
  local catalog_epochs="$2"
  local seed="$3"
  local label="c${catalog_epochs}-seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    bash scripts/run_legacy_catalog_sweep_20260807.sh "${catalog_epochs}" "${seed}" &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}"
  echo "launched ${label} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_arm 0 0 20260626
launch_arm 1 3 20260626
launch_arm 2 6 20260626
launch_arm 3 12 20260626
launch_arm 4 0 20260627
launch_arm 5 3 20260627
launch_arm 6 6 20260627
launch_arm 7 12 20260627

remaining=${#PIDS[@]}
failures=0
while (( remaining > 0 )); do
  finished_pid=""
  if wait -n -p finished_pid; then
    echo "completed ${LABEL_BY_PID[${finished_pid}]}"
  else
    status=$?
    echo "failed ${LABEL_BY_PID[${finished_pid}]:-${finished_pid:-unknown}} with status ${status}" >&2
    failures=$((failures + 1))
  fi
  if [[ -n "${finished_pid}" ]]; then
    unset 'LABEL_BY_PID[$finished_pid]'
  fi
  remaining=$((remaining - 1))
done

trap - INT TERM
if (( failures > 0 )); then
  exit 1
fi
