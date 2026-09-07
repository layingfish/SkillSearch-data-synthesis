#!/usr/bin/env bash
set -uo pipefail

launch_log="${LETTER_OPQ_LAUNCH_LOG_PHYSICAL:?LETTER_OPQ_LAUNCH_LOG_PHYSICAL is required}"
mkdir -p "$(dirname "$launch_log")"
exec > >(tee -a "$launch_log") 2>&1
echo "letter_opq_cooldown_node_guard_start=$(date -Iseconds)"

expected_guard_sha="${LETTER_OPQ_GUARD_SHA256:?LETTER_OPQ_GUARD_SHA256 is required}"
actual_guard_sha="$(sha256sum "$0" | awk '{print $1}')"
if [[ "$actual_guard_sha" != "$expected_guard_sha" ]]; then
  echo "node guard SHA256 mismatch: expected=$expected_guard_sha actual=$actual_guard_sha" >&2
  sleep 300
  exit 1
fi

project_root="${RIPOR_PROJECT_ROOT:?RIPOR_PROJECT_ROOT is required}"
bundle_files=(
  "$project_root/repro/20260903_letter_opq_cooldown_calibration/run_1node_4x4.sh"
  "$project_root/repro/20260903_letter_opq_cooldown_calibration/run_candidate.sh"
  "$project_root/repro/20260903_letter_opq_ceiling/run_ddp_command.sh"
  "$project_root/repro/20260903_letter_opq_ceiling/run_exact_eval.sh"
  "$project_root/scripts/select_letter_opq_ceiling.py"
  "$project_root/scripts/skill_gr_opq_parallel.py"
  "$project_root/scripts/skill_gr_pipeline.py"
  "$project_root/scripts/distributed_utils.py"
)
for bundle_file in "${bundle_files[@]}"; do
  [[ -f "$bundle_file" ]] || {
    echo "missing code-bundle file: $bundle_file" >&2
    sleep 300
    exit 1
  }
done
expected_bundle_sha="${LETTER_OPQ_CODE_BUNDLE_SHA256:?LETTER_OPQ_CODE_BUNDLE_SHA256 is required}"
actual_bundle_sha="$(sha256sum "${bundle_files[@]}" | awk '{print $1}' | sha256sum | awk '{print $1}')"
if [[ "$actual_bundle_sha" != "$expected_bundle_sha" ]]; then
  echo "code bundle SHA256 mismatch: expected=$expected_bundle_sha actual=$actual_bundle_sha" >&2
  sleep 300
  exit 1
fi

node_wrapper="${LETTER_OPQ_NODE_WRAPPER:-/mnt/cfs-agent/workspace/chenyufei03/SkillSearch-data-synthesis/.repro_worktrees/ripor-m0qwen-cf-multisid-20260816/repro/20260816_ripor_cfaware_multisid_testselected/run_rudder_8gpu_node.sh}"
if [[ ! -f "$node_wrapper" ]]; then
  echo "missing exact-runtime node wrapper: $node_wrapper" >&2
  sleep 300
  exit 1
fi
expected_wrapper_sha="${LETTER_OPQ_NODE_WRAPPER_SHA256:?LETTER_OPQ_NODE_WRAPPER_SHA256 is required}"
actual_wrapper_sha="$(sha256sum "$node_wrapper" | awk '{print $1}')"
if [[ "$actual_wrapper_sha" != "$expected_wrapper_sha" ]]; then
  echo "exact-runtime wrapper SHA256 mismatch: expected=$expected_wrapper_sha actual=$actual_wrapper_sha" >&2
  sleep 300
  exit 1
fi

bash "$node_wrapper"
status=$?
echo "letter_opq_cooldown_node_wrapper_exit=$status"
if [[ "$status" -ne 0 ]]; then
  echo "holding failed pod for 300 seconds so logs and events remain inspectable"
  sleep 300
fi
exit "$status"
