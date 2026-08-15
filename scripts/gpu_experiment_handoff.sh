#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="${SKILLGR_GPU_STATE_DIR:-/tmp/skillgr_gpu_handoff}"
log_dir="${SKILLGR_GPU_LOG_DIR:-$repo_root/experiments/gpu_handoff}"
worker_script="$repo_root/scripts/skillgr_dense_teacher_throughput.py"
standby_pid_file="$state_dir/standby.pid"
standby_ready_file="$state_dir/standby.ready.json"
experiment_pid_file="$state_dir/experiment.pid"
experiment_name_file="$state_dir/experiment.name"
lock_file="$state_dir/handoff.lock"
standby_log="$log_dir/skillgr_dense_teacher_throughput.log"
gpu_index="${SKILLGR_GPU_INDEX:-0}"
resident_gib="${SKILLGR_GPU_RESIDENT_GIB:-74}"
standby_batch_size="${SKILLGR_GPU_STANDBY_BATCH_SIZE:-8}"
standby_max_length="${SKILLGR_GPU_STANDBY_MAX_LENGTH:-2048}"
standby_step_sleep_seconds="${SKILLGR_GPU_STANDBY_STEP_SLEEP_SECONDS:-0.1}"
ready_timeout="${SKILLGR_GPU_READY_TIMEOUT:-240}"
stop_timeout="${SKILLGR_GPU_STOP_TIMEOUT:-60}"
handoff_experiment_pid=""
handoff_restore_needed=0

mkdir -p "$state_dir" "$log_dir"

usage() {
  echo "usage: $0 {start|stop|status|run -- COMMAND...|submit NAME LOG -- COMMAND...}"
}

read_pid() {
  local path="$1"
  if [[ -f "$path" ]]; then
    tr -d '[:space:]' <"$path"
  fi
}

standby_is_live() {
  local pid
  local cmdline
  pid="$(read_pid "$standby_pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  [[ "$cmdline" == *"$worker_script"* ]]
}

start_standby_unlocked() {
  if standby_is_live && [[ -s "$standby_ready_file" ]]; then
    echo "standby_already_ready pid=$(read_pid "$standby_pid_file")"
    return 0
  fi

  rm -f "$standby_pid_file" "$standby_ready_file"
  nohup env CUDA_VISIBLE_DEVICES="$gpu_index" \
    python3 -u "$worker_script" \
      --gpu-index "$gpu_index" \
      --resident-gib "$resident_gib" \
      --batch-size "$standby_batch_size" \
      --max-length "$standby_max_length" \
      --step-sleep-seconds "$standby_step_sleep_seconds" \
      --ready-file "$standby_ready_file" \
      >>"$standby_log" 2>&1 9>&- &
  local pid=$!
  printf '%s\n' "$pid" >"$standby_pid_file"

  local waited=0
  while (( waited < ready_timeout )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "standby_failed pid=$pid log=$standby_log" >&2
      return 1
    fi
    if [[ -s "$standby_ready_file" ]]; then
      echo "standby_ready pid=$pid ready_file=$standby_ready_file log=$standby_log"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "standby_ready_timeout pid=$pid timeout_seconds=$ready_timeout log=$standby_log" >&2
  return 1
}

stop_standby_unlocked() {
  local pid
  pid="$(read_pid "$standby_pid_file")"
  if ! standby_is_live; then
    rm -f "$standby_pid_file" "$standby_ready_file"
    echo "standby_not_running"
    return 0
  fi

  kill -TERM "$pid"
  local waited=0
  while kill -0 "$pid" 2>/dev/null && (( waited < stop_timeout )); do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "standby_sigterm_timeout pid=$pid; sending SIGKILL" >&2
    kill -KILL "$pid"
    wait "$pid" 2>/dev/null || true
  fi
  rm -f "$standby_pid_file" "$standby_ready_file"
  echo "standby_stopped pid=$pid wait_seconds=$waited"
}

with_lock() {
  exec 9>"$lock_file"
  flock 9
  "$@"
}

status() {
  local standby_state="stopped"
  local experiment_state="stopped"
  local standby_pid
  local experiment_pid
  local experiment_name="none"
  standby_pid="$(read_pid "$standby_pid_file")"
  experiment_pid="$(read_pid "$experiment_pid_file")"
  if [[ -f "$experiment_name_file" ]]; then
    experiment_name="$(tr -d '\n' <"$experiment_name_file")"
  fi
  if standby_is_live; then
    standby_state="running"
  fi
  if [[ "$experiment_pid" =~ ^[0-9]+$ ]] && kill -0 "$experiment_pid" 2>/dev/null; then
    experiment_state="running"
  fi
  echo "standby_state=$standby_state standby_pid=${standby_pid:-none} standby_log=$standby_log"
  echo "experiment_state=$experiment_state experiment_pid=${experiment_pid:-none} experiment_name=$experiment_name"
  nvidia-smi \
    --id="$gpu_index" \
    --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits
  nvidia-smi \
    --id="$gpu_index" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits || true
}

run_experiment_unlocked() {
  local experiment_name="${SKILLGR_HANDOFF_EXPERIMENT_NAME:-skillgr_experiment}"
  if (( $# == 0 )); then
    echo "missing experiment command" >&2
    return 2
  fi

  stop_standby_unlocked
  printf '%s\n' "$experiment_name" >"$experiment_name_file"

  handoff_experiment_pid=""
  handoff_restore_needed=1
  restore_standby() {
    local original_status=$?
    trap - EXIT INT TERM
    rm -f "$experiment_pid_file" "$experiment_name_file"
    if (( handoff_restore_needed == 1 )); then
      handoff_restore_needed=0
      if ! start_standby_unlocked; then
        echo "critical: experiment exited but standby restart failed" >&2
      fi
    fi
    exit "$original_status"
  }
  forward_signal() {
    local signal_name="$1"
    if [[ "$handoff_experiment_pid" =~ ^[0-9]+$ ]] && kill -0 "$handoff_experiment_pid" 2>/dev/null; then
      kill -"$signal_name" "$handoff_experiment_pid" 2>/dev/null || true
    fi
  }
  trap restore_standby EXIT
  trap 'forward_signal TERM' TERM
  trap 'forward_signal INT' INT

  echo "experiment_launching name=$experiment_name command=$(printf '%q ' "$@")"
  "$@" 9>&- &
  handoff_experiment_pid=$!
  printf '%s\n' "$handoff_experiment_pid" >"$experiment_pid_file"
  echo "experiment_started name=$experiment_name pid=$handoff_experiment_pid"

  local experiment_status=0
  wait "$handoff_experiment_pid" || experiment_status=$?
  echo "experiment_finished name=$experiment_name pid=$handoff_experiment_pid exit_status=$experiment_status"
  return "$experiment_status"
}

command="${1:-}"
case "$command" in
  start)
    shift
    (( $# == 0 )) || { usage >&2; exit 2; }
    with_lock start_standby_unlocked
    ;;
  stop)
    shift
    (( $# == 0 )) || { usage >&2; exit 2; }
    with_lock stop_standby_unlocked
    ;;
  status)
    shift
    (( $# == 0 )) || { usage >&2; exit 2; }
    status
    ;;
  run)
    shift
    [[ "${1:-}" == "--" ]] && shift
    with_lock run_experiment_unlocked "$@"
    ;;
  submit)
    shift
    experiment_name="${1:-}"
    experiment_log="${2:-}"
    if [[ -z "$experiment_name" || -z "$experiment_log" ]]; then
      usage >&2
      exit 2
    fi
    shift 2
    [[ "${1:-}" == "--" ]] && shift
    (( $# > 0 )) || { usage >&2; exit 2; }
    mkdir -p "$(dirname "$experiment_log")"
    nohup env SKILLGR_HANDOFF_EXPERIMENT_NAME="$experiment_name" \
      "$0" run -- "$@" >"$experiment_log" 2>&1 &
    supervisor_pid=$!
    echo "experiment_submitted name=$experiment_name supervisor_pid=$supervisor_pid log=$experiment_log"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
