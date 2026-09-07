#!/usr/bin/env bash
set -euo pipefail

python_bin="${RIPOR_PYTHON_BIN:-python3}"
world_size="${LETTER_OPQ_WORLD_SIZE:-4}"
master_port="${LETTER_OPQ_MASTER_PORT:?LETTER_OPQ_MASTER_PORT is required}"
[[ "$world_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid LETTER_OPQ_WORLD_SIZE: $world_size" >&2
  exit 2
}
[[ "$master_port" =~ ^[0-9]+$ ]] || {
  echo "invalid LETTER_OPQ_MASTER_PORT: $master_port" >&2
  exit 2
}
[[ "$#" -gt 0 ]] || {
  echo "usage: run_ddp_command.sh PYTHON_ARGS..." >&2
  exit 2
}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "${#visible_devices[@]}" -ne "$world_size" ]]; then
    echo "CUDA_VISIBLE_DEVICES/world-size mismatch: visible=${#visible_devices[@]} world=$world_size" >&2
    exit 2
  fi
fi

status_dir="$(mktemp -d /tmp/letter-opq-ceiling-ddp.XXXXXX)"
declare -a child_pids=()
cleanup() {
  local pid
  for pid in "${child_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  case "$status_dir" in
    /tmp/letter-opq-ceiling-ddp.*) rm -rf -- "$status_dir" ;;
    *) echo "refusing unexpected temporary path: $status_dir" >&2 ;;
  esac
}
trap cleanup EXIT INT TERM HUP

echo "local_ddp_launch world_size=$world_size master=127.0.0.1:$master_port command=$1"
for ((rank = 0; rank < world_size; rank++)); do
  status_file="$status_dir/rank-${rank}.status"
  (
    set +e
    env \
      -u TORCHELASTIC_RUN_ID -u TORCHELASTIC_RESTART_COUNT \
      -u TORCHELASTIC_MAX_RESTARTS -u TORCHELASTIC_ERROR_FILE \
      -u PET_MASTER_ADDR -u PET_MASTER_PORT -u PET_NNODES -u PET_NODE_RANK \
      -u PET_NPROC_PER_NODE \
      MASTER_ADDR=127.0.0.1 \
      MASTER_PORT="$master_port" \
      WORLD_SIZE="$world_size" \
      RANK="$rank" \
      LOCAL_RANK="$rank" \
      LOCAL_WORLD_SIZE="$world_size" \
      GROUP_RANK=0 \
      ROLE_RANK="$rank" \
      ROLE_WORLD_SIZE="$world_size" \
      "$python_bin" "$@"
    child_status=$?
    printf '%s\n' "$child_status" > "$status_file"
    exit "$child_status"
  ) &
  child_pids+=("$!")
done

remaining="$world_size"
failed=0
while [[ "$remaining" -gt 0 && "$failed" -eq 0 ]]; do
  remaining=0
  for ((rank = 0; rank < world_size; rank++)); do
    status_file="$status_dir/rank-${rank}.status"
    if [[ ! -f "$status_file" ]]; then
      remaining=$((remaining + 1))
      continue
    fi
    status="$(<"$status_file")"
    if [[ "$status" -ne 0 ]]; then
      echo "local DDP rank failed: rank=$rank status=$status" >&2
      failed="$status"
      break
    fi
  done
  if [[ "$remaining" -gt 0 && "$failed" -eq 0 ]]; then
    sleep 1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit "$failed"
fi
for pid in "${child_pids[@]}"; do
  wait "$pid"
done
child_pids=()

