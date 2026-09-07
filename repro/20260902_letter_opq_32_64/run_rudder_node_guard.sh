#!/usr/bin/env bash
set -uo pipefail

launch_log="${LETTER_OPQ_LAUNCH_LOG_PHYSICAL:?LETTER_OPQ_LAUNCH_LOG_PHYSICAL is required}"
mkdir -p "$(dirname "$launch_log")"
exec > >(tee -a "$launch_log") 2>&1

echo "letter_opq_node_guard_start=$(date -Iseconds)"

node_wrapper="${LETTER_OPQ_NODE_WRAPPER:-/mnt/cfs-agent/workspace/chenyufei03/SkillSearch-data-synthesis/.repro_worktrees/ripor-m0qwen-cf-multisid-20260816/repro/20260816_ripor_cfaware_multisid_testselected/run_rudder_8gpu_node.sh}"
if [[ ! -f "$node_wrapper" ]]; then
  echo "missing exact-runtime node wrapper: $node_wrapper" >&2
  sleep 300
  exit 1
fi

bash "$node_wrapper"
status=$?
echo "letter_opq_node_wrapper_exit=$status"
if [[ "$status" -ne 0 ]]; then
  echo "holding failed pod for 300 seconds so logs and events remain inspectable"
  sleep 300
fi
exit "$status"
