#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: $0 preflight|preflight_eval|catalog|catalog_long_e30|catalog_extend_e60|catalog_resume_diagnostics|query_expanded_ce|query_sibling_masked_ce|query_expanded_ce_e6|query_expanded_ce_long_selected}"
PHYSICAL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/multigt_loss_catalog_convergence_20260805_v1
mkdir -p "${PHYSICAL_OUTPUT_ROOT}/logs"
exec > >(tee -a "${PHYSICAL_OUTPUT_ROOT}/logs/${STAGE}.log") 2>&1

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1

ROOT_DIR="${SKILLGR_CODE_ROOT:-/workspace/SkillSearch-data-synthesis}"
git config --global --add safe.directory "${ROOT_DIR}"
git config --global --add safe.directory "$(readlink -f "${ROOT_DIR}")"
cd "${ROOT_DIR}"

set -a
source configs/multigt_loss_catalog_20260805.env
set +a

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SKILLGR_MODEL

CATALOG_DIR="${SKILLGR_OUTPUT_ROOT}/catalog_e6"
LONG_CATALOG_DIR="${SKILLGR_OUTPUT_ROOT}/catalog_long_e30"
EXTENDED_CATALOG_DIR="${SKILLGR_OUTPUT_ROOT}/catalog_long_e60_from_e27"
RESUME_PROBE_DIR="${SKILLGR_OUTPUT_ROOT}/catalog_resume_probe_e30_b"
CATALOG_ROWS="${SKILLGR_OUTPUT_ROOT}/catalog_rows.jsonl"
EVAL_ROOT="${SKILLGR_OUTPUT_ROOT}/eval"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file is missing: $1" >&2
    exit 1
  fi
}

require_absent() {
  if [[ -e "$1" ]]; then
    echo "refusing to overwrite existing experiment artifact: $1" >&2
    exit 1
  fi
}

run_torch() {
  torchrun --standalone --nproc_per_node="${SKILLGR_WORLD_SIZE}" "$@"
}

record_runtime() {
  git status --short
  git rev-parse HEAD
  python3 docker/smoke_test.py
}

verify_epoch_history() {
  local history="$1"
  local stage="$2"
  local expected_epochs="$3"
  local expected_tokens="$4"
  python3 - "${history}" "${stage}" "${expected_epochs}" "${expected_tokens}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
stage = sys.argv[2]
expected_epochs = int(sys.argv[3])
expected_tokens = int(sys.argv[4])
events = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
epochs = [row for row in events if row.get("stage") == stage and row.get("event") == "epoch"]
if len(epochs) != expected_epochs:
    raise SystemExit(f"{stage} epoch event count {len(epochs)} != {expected_epochs}")
bad = [row for row in epochs if int(row.get("ce_token_count", -1)) != expected_tokens]
if bad:
    raise SystemExit(f"{stage} epoch token count mismatch: {bad}")
print(json.dumps({"stage": stage, "epochs": expected_epochs, "ce_tokens_per_epoch": expected_tokens}))
PY
}

verify_epoch_range_history() {
  local history="$1"
  local stage="$2"
  local first_epoch="$3"
  local last_epoch="$4"
  local expected_tokens="$5"
  python3 - "${history}" "${stage}" "${first_epoch}" "${last_epoch}" "${expected_tokens}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
stage = sys.argv[2]
first_epoch = int(sys.argv[3])
last_epoch = int(sys.argv[4])
expected_tokens = int(sys.argv[5])
events = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
epochs = [row for row in events if row.get("stage") == stage and row.get("event") == "epoch"]
observed = [int(row["epoch"]) for row in epochs]
expected = list(range(first_epoch, last_epoch + 1))
if observed != expected:
    raise SystemExit(f"{stage} epoch sequence {observed} != {expected}")
bad = [row for row in epochs if int(row.get("ce_token_count", -1)) != expected_tokens]
if bad:
    raise SystemExit(f"{stage} epoch token count mismatch: {bad}")
print(
    json.dumps(
        {
            "stage": stage,
            "first_epoch": first_epoch,
            "last_epoch": last_epoch,
            "ce_tokens_per_epoch": expected_tokens,
        },
        sort_keys=True,
    )
)
PY
}

verify_resume_model_parity() {
  local original_checkpoint="$1"
  local resumed_checkpoint="$2"
  local output="$3"
  require_absent "${output}"
  python3 - "${original_checkpoint}" "${resumed_checkpoint}" "${output}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch


def state_hash(path: Path) -> str:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    digest = hashlib.sha256()
    for key, value in sorted(payload["state_dict"].items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


original = Path(sys.argv[1])
resumed = Path(sys.argv[2])
output = Path(sys.argv[3])
original_hash = state_hash(original)
resumed_hash = state_hash(resumed)
result = {
    "format": "structured_catalog_resume_parity_v1",
    "original_checkpoint": str(original),
    "original_model_state_sha256": original_hash,
    "resumed_checkpoint": str(resumed),
    "resumed_model_state_sha256": resumed_hash,
    "model_state_exact_match": original_hash == resumed_hash,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True), flush=True)
if original_hash != resumed_hash:
    raise SystemExit("resumed epoch-30 model state does not match the uninterrupted run")
PY
}

verify_preflight_history() {
  local history="$1"
  python3 - "${history}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
events = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
expected = {"catalog": 1, "query": 2}
observed = {}
for stage, count in expected.items():
    train_events = [
        row
        for row in events
        if row.get("stage") == stage and row.get("event") == "train"
    ]
    if len(train_events) != count:
        raise SystemExit(f"{stage} train event count {len(train_events)} != {count}")
    if any(not math.isfinite(float(row["train_loss"])) for row in train_events):
        raise SystemExit(f"{stage} preflight has a non-finite train loss")
    observed[stage] = len(train_events)
print(json.dumps({"preflight_train_events": observed}, sort_keys=True))
PY
}

catalog_checkpoint() {
  case "$1" in
    1) echo "${CATALOG_DIR}/checkpoint-$((SKILLGR_CATALOG_STEPS_PER_EPOCH))" ;;
    2) echo "${CATALOG_DIR}/checkpoint-$((2 * SKILLGR_CATALOG_STEPS_PER_EPOCH))" ;;
    3) echo "${CATALOG_DIR}/checkpoint-$((3 * SKILLGR_CATALOG_STEPS_PER_EPOCH))" ;;
    6) echo "${CATALOG_DIR}/final" ;;
    *) echo "unsupported catalog epoch: $1" >&2; exit 1 ;;
  esac
}

long_catalog_checkpoint() {
  local epoch="$1"
  if (( epoch <= 0 || epoch > SKILLGR_LONG_CATALOG_EPOCHS )); then
    echo "unsupported long catalog epoch: ${epoch}" >&2
    exit 1
  fi
  if (( epoch == SKILLGR_LONG_CATALOG_EPOCHS )); then
    echo "${LONG_CATALOG_DIR}/final"
  elif (( epoch % SKILLGR_LONG_CATALOG_EVAL_INTERVAL == 0 )); then
    echo "${LONG_CATALOG_DIR}/checkpoint-$((epoch * SKILLGR_CATALOG_STEPS_PER_EPOCH))"
  else
    echo "long catalog epoch is not a saved evaluation checkpoint: ${epoch}" >&2
    exit 1
  fi
}

extended_catalog_checkpoint() {
  local epoch="$1"
  if (( epoch <= 0 || epoch > SKILLGR_EXTENDED_CATALOG_EPOCHS )); then
    echo "unsupported extended catalog epoch: ${epoch}" >&2
    exit 1
  fi
  if (( epoch == SKILLGR_EXTENDED_CATALOG_EPOCHS )); then
    echo "${EXTENDED_CATALOG_DIR}/final"
  elif (( epoch % SKILLGR_LONG_CATALOG_EVAL_INTERVAL == 0 )); then
    echo "${EXTENDED_CATALOG_DIR}/checkpoint-$((epoch * SKILLGR_CATALOG_STEPS_PER_EPOCH))"
  else
    echo "extended catalog epoch is not a saved evaluation checkpoint: ${epoch}" >&2
    exit 1
  fi
}

eval_catalog_teacher_forced() {
  local label="$1"
  local model_dir="$2"
  local output_group="${3:-catalog_teacher_forced}"
  local output="${EVAL_ROOT}/${output_group}/${label}.json"
  local materialize_args=()
  require_file "${model_dir}/structured_decoder.pt"
  require_absent "${output}"
  if [[ ! -f "${CATALOG_ROWS}" ]]; then
    materialize_args=(--materialized-rows "${CATALOG_ROWS}")
  fi
  run_torch scripts/evaluate_catalog_indexing.py \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --model-dir "${model_dir}" \
    --architecture-model "${SKILLGR_MODEL}" \
    --output "${output}" \
    "${materialize_args[@]}" \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --bf16
}

eval_catalog_constrained_greedy() {
  local label="$1"
  local model_dir="$2"
  local output_group="${3:-catalog_constrained_greedy}"
  local output="${EVAL_ROOT}/${output_group}/${label}"
  require_file "${model_dir}/structured_decoder.pt"
  require_file "${CATALOG_ROWS}"
  require_absent "${output}"
  run_torch scripts/skill_gr_pipeline.py eval-single-sid-decoder \
    --rows "${CATALOG_ROWS}" \
    --model-dir "${model_dir}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${output}" \
    --split-name "catalog_${label}" \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --num-beams 1 --rank-aggregation beam_order --k-values 1 \
    --log-every 250
}

run_catalog() {
  require_file "${SKILLGR_OUTPUT_ROOT}/preflight_v3/preflight.done"
  require_absent "${CATALOG_DIR}"
  mkdir -p "${SKILLGR_OUTPUT_ROOT}"
  record_runtime
  run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
    --rows "${SKILLGR_ROWS}" \
    --dev-rows "${SKILLGR_DEV_ROWS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --output-dir "${CATALOG_DIR}" \
    --model "${SKILLGR_MODEL}" \
    --query-loss expanded_ce \
    --catalog-epochs "${SKILLGR_CATALOG_EPOCHS}" \
    --epochs 0 \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --eval-batch-size "${SKILLGR_EVAL_PER_DEVICE_BATCH}" \
    --lr "${SKILLGR_LR}" \
    --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
    --lr-scheduler constant \
    --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
    --logging-steps 50 \
    --eval-steps 0 \
    --save-steps "${SKILLGR_CATALOG_STEPS_PER_EPOCH}" \
    --no-padding-distributed-sampler \
    --seed "${SKILLGR_SEED}" \
    --no-load-best-model-at-end \
    --bf16
  verify_epoch_history "${CATALOG_DIR}/loss_history.jsonl" catalog 6 72755

  local epoch
  for epoch in 1 2 3 6; do
    eval_catalog_teacher_forced "epoch${epoch}" "$(catalog_checkpoint "${epoch}")"
  done
  eval_catalog_teacher_forced old_query_final "${SKILLGR_OLD_FINAL}"
  eval_catalog_constrained_greedy epoch3 "$(catalog_checkpoint 3)"
  eval_catalog_constrained_greedy epoch6 "$(catalog_checkpoint 6)"
  touch "${SKILLGR_OUTPUT_ROOT}/catalog.done"
}

write_long_catalog_convergence_summary() {
  local output="${EVAL_ROOT}/catalog_long_convergence.json"
  require_absent "${output}"
  python3 scripts/analyze_catalog_convergence.py \
    --metrics-dir "${EVAL_ROOT}/catalog_long_teacher_forced" \
    --output "${output}" \
    --checkpoint-interval "${SKILLGR_LONG_CATALOG_EVAL_INTERVAL}" \
    --max-epoch "${SKILLGR_LONG_CATALOG_EPOCHS}" \
    --ce-relative-threshold 0.01 \
    --sid-exact-absolute-threshold 0.005 \
    --consecutive-stable-intervals 3
}

run_catalog_long() {
  require_file "${SKILLGR_OUTPUT_ROOT}/preflight_v3/preflight.done"
  require_file "${CATALOG_ROWS}"
  require_absent "${LONG_CATALOG_DIR}"
  require_absent "${EVAL_ROOT}/catalog_long_teacher_forced"
  require_absent "${EVAL_ROOT}/catalog_long_constrained_greedy"
  require_absent "${EVAL_ROOT}/catalog_long_convergence.json"
  require_absent "${SKILLGR_OUTPUT_ROOT}/catalog_long_e30.done"
  mkdir -p "${SKILLGR_OUTPUT_ROOT}"
  record_runtime
  run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
    --rows "${SKILLGR_ROWS}" \
    --dev-rows "${SKILLGR_DEV_ROWS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --output-dir "${LONG_CATALOG_DIR}" \
    --model "${SKILLGR_MODEL}" \
    --query-loss expanded_ce \
    --catalog-epochs "${SKILLGR_LONG_CATALOG_EPOCHS}" \
    --epochs 0 \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --eval-batch-size "${SKILLGR_EVAL_PER_DEVICE_BATCH}" \
    --lr "${SKILLGR_LR}" \
    --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
    --lr-scheduler constant \
    --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
    --logging-steps 50 \
    --eval-steps 0 \
    --save-steps "$((SKILLGR_LONG_CATALOG_EVAL_INTERVAL * SKILLGR_CATALOG_STEPS_PER_EPOCH))" \
    --no-padding-distributed-sampler \
    --seed "${SKILLGR_SEED}" \
    --no-load-best-model-at-end \
    --bf16
  verify_epoch_history \
    "${LONG_CATALOG_DIR}/loss_history.jsonl" catalog "${SKILLGR_LONG_CATALOG_EPOCHS}" 72755

  local epoch
  for ((
    epoch = SKILLGR_LONG_CATALOG_EVAL_INTERVAL;
    epoch <= SKILLGR_LONG_CATALOG_EPOCHS;
    epoch += SKILLGR_LONG_CATALOG_EVAL_INTERVAL
  )); do
    eval_catalog_teacher_forced \
      "epoch${epoch}" "$(long_catalog_checkpoint "${epoch}")" catalog_long_teacher_forced
  done
  for epoch in 6 12 18 24 30; do
    eval_catalog_constrained_greedy \
      "epoch${epoch}" "$(long_catalog_checkpoint "${epoch}")" catalog_long_constrained_greedy
  done
  write_long_catalog_convergence_summary
  touch "${SKILLGR_OUTPUT_ROOT}/catalog_long_e30.done"
}

run_catalog_extend_e60() {
  local original_summary="${EVAL_ROOT}/catalog_long_convergence.json"
  local resume_epoch="${SKILLGR_CATALOG_RESUME_EPOCH}"
  local resume_dir
  local parity_output="${EVAL_ROOT}/catalog_long_e60_resume_parity.json"
  local teacher_group="catalog_long_e60_teacher_forced"
  local teacher_dir="${EVAL_ROOT}/${teacher_group}"
  local combined_dir="${EVAL_ROOT}/catalog_long_e60_combined_teacher_forced"
  local greedy_group="catalog_long_e60_constrained_greedy"
  local convergence_output="${EVAL_ROOT}/catalog_long_e60_convergence.json"
  resume_dir="$(long_catalog_checkpoint "${resume_epoch}")"

  require_file "${SKILLGR_OUTPUT_ROOT}/catalog_long_e30.done"
  require_file "${original_summary}"
  require_file "${resume_dir}/structured_decoder.pt"
  python3 - "${original_summary}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("converged") is not False or payload.get("selected_epoch") is not None:
    raise SystemExit("epoch-30 catalog result is not the expected non-converged extension input")
if not payload.get("checkpoints") or int(payload["checkpoints"][-1]["epoch"]) != 30:
    raise SystemExit("epoch-30 catalog convergence summary is incomplete")
PY
  python3 - \
    "${resume_dir}/structured_decoder.pt" \
    "${resume_epoch}" \
    "${SKILLGR_CATALOG_STEPS_PER_EPOCH}" \
    "${SKILLGR_WORLD_SIZE}" \
    "${SKILLGR_PER_DEVICE_BATCH}" <<'PY'
import json
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1])
resume_epoch = int(sys.argv[2])
steps_per_epoch = int(sys.argv[3])
world_size = int(sys.argv[4])
per_device_batch = int(sys.argv[5])
try:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(checkpoint_path, map_location="cpu")
state = payload.get("training_state")
if not isinstance(state, dict):
    raise SystemExit("catalog resume checkpoint is missing training_state")
expected = {
    "stage": "catalog",
    "completed_steps": resume_epoch * steps_per_epoch,
    "next_epoch": resume_epoch,
    "next_batch_index": 0,
    "world_size": world_size,
    "per_device_batch_size": per_device_batch,
}
mismatches = {
    key: {"expected": value, "observed": state.get(key)}
    for key, value in expected.items()
    if state.get(key) != value
}
if mismatches:
    raise SystemExit(f"catalog resume checkpoint state mismatch: {mismatches}")
if not isinstance(payload.get("optimizer_state_dict"), dict):
    raise SystemExit("catalog resume checkpoint is missing optimizer state")
if not isinstance(state.get("scheduler_state_dict"), dict):
    raise SystemExit("catalog resume checkpoint is missing scheduler state")
if len(state.get("rng_states") or []) != world_size:
    raise SystemExit("catalog resume checkpoint is missing per-rank RNG states")
print(
    json.dumps(
        {
            "resume_checkpoint": str(checkpoint_path),
            **expected,
            "optimizer_state": "present",
            "scheduler_state": "present",
            "rng_state_count": world_size,
        },
        sort_keys=True,
    ),
    flush=True,
)
PY
  require_absent "${EXTENDED_CATALOG_DIR}"
  require_absent "${parity_output}"
  require_absent "${teacher_dir}"
  require_absent "${combined_dir}"
  require_absent "${EVAL_ROOT}/${greedy_group}"
  require_absent "${convergence_output}"
  require_absent "${SKILLGR_OUTPUT_ROOT}/catalog_extend_e60.done"
  record_runtime

  run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
    --rows "${SKILLGR_ROWS}" \
    --dev-rows "${SKILLGR_DEV_ROWS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --output-dir "${EXTENDED_CATALOG_DIR}" \
    --model "${SKILLGR_MODEL}" \
    --init-model-dir "${resume_dir}" \
    --resume-from-checkpoint "${resume_dir}" \
    --allow-resume-total-steps-extension \
    --no-reset-optimizer \
    --query-loss expanded_ce \
    --catalog-epochs "${SKILLGR_EXTENDED_CATALOG_EPOCHS}" \
    --epochs 0 \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --eval-batch-size "${SKILLGR_EVAL_PER_DEVICE_BATCH}" \
    --lr "${SKILLGR_LR}" \
    --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
    --lr-scheduler constant \
    --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
    --logging-steps 50 \
    --eval-steps 0 \
    --save-steps "$((SKILLGR_LONG_CATALOG_EVAL_INTERVAL * SKILLGR_CATALOG_STEPS_PER_EPOCH))" \
    --no-padding-distributed-sampler \
    --seed "${SKILLGR_SEED}" \
    --no-load-best-model-at-end \
    --bf16
  verify_epoch_range_history \
    "${EXTENDED_CATALOG_DIR}/loss_history.jsonl" \
    catalog \
    "$((resume_epoch + 1))" \
    "${SKILLGR_EXTENDED_CATALOG_EPOCHS}" \
    72755
  verify_resume_model_parity \
    "${LONG_CATALOG_DIR}/final/structured_decoder.pt" \
    "$(extended_catalog_checkpoint 30)/structured_decoder.pt" \
    "${parity_output}"

  local epoch
  for ((
    epoch = 30;
    epoch <= SKILLGR_EXTENDED_CATALOG_EPOCHS;
    epoch += SKILLGR_LONG_CATALOG_EVAL_INTERVAL
  )); do
    eval_catalog_teacher_forced \
      "epoch${epoch}" "$(extended_catalog_checkpoint "${epoch}")" "${teacher_group}"
  done

  mkdir -p "${combined_dir}"
  for ((epoch = SKILLGR_LONG_CATALOG_EVAL_INTERVAL; epoch < 30; epoch += SKILLGR_LONG_CATALOG_EVAL_INTERVAL)); do
    install -m 0644 \
      "${EVAL_ROOT}/catalog_long_teacher_forced/epoch${epoch}.json" \
      "${combined_dir}/epoch${epoch}.json"
  done
  for ((epoch = 30; epoch <= SKILLGR_EXTENDED_CATALOG_EPOCHS; epoch += SKILLGR_LONG_CATALOG_EVAL_INTERVAL)); do
    install -m 0644 \
      "${teacher_dir}/epoch${epoch}.json" \
      "${combined_dir}/epoch${epoch}.json"
  done
  python3 scripts/analyze_catalog_convergence.py \
    --metrics-dir "${combined_dir}" \
    --output "${convergence_output}" \
    --checkpoint-interval "${SKILLGR_LONG_CATALOG_EVAL_INTERVAL}" \
    --max-epoch "${SKILLGR_EXTENDED_CATALOG_EPOCHS}" \
    --ce-relative-threshold 0.01 \
    --sid-exact-absolute-threshold 0.005 \
    --consecutive-stable-intervals 3
  for epoch in 36 42 48 54 60; do
    eval_catalog_constrained_greedy \
      "epoch${epoch}" "$(extended_catalog_checkpoint "${epoch}")" "${greedy_group}"
  done
  touch "${SKILLGR_OUTPUT_ROOT}/catalog_extend_e60.done"
}

run_catalog_resume_diagnostics() {
  local resume_epoch="${SKILLGR_CATALOG_RESUME_EPOCH}"
  local resume_dir
  local comparison_dir="${EVAL_ROOT}/catalog_resume_diagnostics"
  local teacher_group="catalog_resume_diagnostics_teacher_forced"
  resume_dir="$(long_catalog_checkpoint "${resume_epoch}")"

  require_file "${SKILLGR_OUTPUT_ROOT}/catalog_long_e30.done"
  require_file "${resume_dir}/structured_decoder.pt"
  require_file "${LONG_CATALOG_DIR}/final/structured_decoder.pt"
  require_file "$(extended_catalog_checkpoint 30)/structured_decoder.pt"
  require_file "${EVAL_ROOT}/catalog_long_e60_resume_parity.json"
  require_file "${EVAL_ROOT}/catalog_long_teacher_forced/epoch30.json"
  require_absent "${SKILLGR_OUTPUT_ROOT}/catalog_resume_diagnostics.done"
  python3 - "${EVAL_ROOT}/catalog_long_e60_resume_parity.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("model_state_exact_match") is not False:
    raise SystemExit("resume diagnostics require the observed failed exact-hash parity result")
PY
  record_runtime
  python3 -m unittest tests.test_compare_catalog_resume_checkpoints

  if [[ ! -d "${RESUME_PROBE_DIR}" ]]; then
    run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
      --rows "${SKILLGR_ROWS}" \
      --dev-rows "${SKILLGR_DEV_ROWS}" \
      --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
      --skills-jsonl "${SKILLGR_SKILLS}" \
      --output-dir "${RESUME_PROBE_DIR}" \
      --model "${SKILLGR_MODEL}" \
      --init-model-dir "${resume_dir}" \
      --resume-from-checkpoint "${resume_dir}" \
      --no-reset-optimizer \
      --query-loss expanded_ce \
      --catalog-epochs "${SKILLGR_LONG_CATALOG_EPOCHS}" \
      --epochs 0 \
      --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
      --batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
      --eval-batch-size "${SKILLGR_EVAL_PER_DEVICE_BATCH}" \
      --lr "${SKILLGR_LR}" \
      --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
      --lr-scheduler constant \
      --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
      --logging-steps 50 \
      --eval-steps 0 \
      --save-steps "$((SKILLGR_LONG_CATALOG_EVAL_INTERVAL * SKILLGR_CATALOG_STEPS_PER_EPOCH))" \
      --no-padding-distributed-sampler \
      --seed "${SKILLGR_SEED}" \
      --no-load-best-model-at-end \
      --bf16
  fi
  verify_epoch_range_history \
    "${RESUME_PROBE_DIR}/loss_history.jsonl" \
    catalog \
    "$((resume_epoch + 1))" \
    "${SKILLGR_LONG_CATALOG_EPOCHS}" \
    72755

  if [[ ! -f "${EVAL_ROOT}/${teacher_group}/resume_a_epoch30.json" ]]; then
    eval_catalog_teacher_forced \
      resume_a_epoch30 "$(extended_catalog_checkpoint 30)" "${teacher_group}"
  fi
  if [[ ! -f "${EVAL_ROOT}/${teacher_group}/resume_b_epoch30.json" ]]; then
    eval_catalog_teacher_forced \
      resume_b_epoch30 "${RESUME_PROBE_DIR}/final" "${teacher_group}"
  fi
  if [[ ! -f "${comparison_dir}/checkpoint_comparison.json" ]]; then
    python3 scripts/compare_catalog_resume_checkpoints.py \
      --checkpoint "original_epoch30=${LONG_CATALOG_DIR}/final/structured_decoder.pt" \
      --checkpoint "resume_a_epoch30=$(extended_catalog_checkpoint 30)/structured_decoder.pt" \
      --checkpoint "resume_b_epoch30=${RESUME_PROBE_DIR}/final/structured_decoder.pt" \
      --metric "original_epoch30=${EVAL_ROOT}/catalog_long_teacher_forced/epoch30.json" \
      --metric "resume_a_epoch30=${EVAL_ROOT}/${teacher_group}/resume_a_epoch30.json" \
      --metric "resume_b_epoch30=${EVAL_ROOT}/${teacher_group}/resume_b_epoch30.json" \
      --input-file "query_rows=${SKILLGR_ROWS}" \
      --input-file "query_dev_rows=${SKILLGR_DEV_ROWS}" \
      --input-file "skills=${SKILLGR_SKILLS}" \
      --input-file "semantic_ids=${SKILLGR_SEMANTIC_IDS}" \
      --input-file "model_config=${SKILLGR_MODEL}/config.json" \
      --input-file "tokenizer_json=${SKILLGR_MODEL}/tokenizer.json" \
      --input-file "tokenizer_spiece=${SKILLGR_MODEL}/spiece.model" \
      --input-file "experiment_config=${ROOT_DIR}/configs/multigt_loss_catalog_20260805.env" \
      --output "${comparison_dir}/checkpoint_comparison.json"
  fi
  touch "${SKILLGR_OUTPUT_ROOT}/catalog_resume_diagnostics.done"
}

eval_preflight_artifacts() {
  local output="${SKILLGR_OUTPUT_ROOT}/preflight_v3"
  require_file "${output}/model/loss_history.jsonl"
  require_file "${output}/model/final/structured_decoder.pt"
  require_absent "${output}/catalog_teacher_forced.json"
  verify_preflight_history "${output}/model/loss_history.jsonl"
  run_torch scripts/evaluate_catalog_indexing.py \
    --skills-jsonl "${output}/data/skills.jsonl" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --model-dir "${output}/model/final" \
    --architecture-model "${SKILLGR_MODEL}" \
    --output "${output}/catalog_teacher_forced.json" \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --batch-size 1 --bf16
  touch "${output}/preflight.done"
}

run_preflight_eval() {
  record_runtime
  eval_preflight_artifacts
}

run_preflight() {
  local output="${SKILLGR_OUTPUT_ROOT}/preflight_v3"
  require_absent "${output}"
  mkdir -p "${output}/data"
  record_runtime
  python3 - "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SKILLGR_SKILLS}" "${output}/data" <<'PY'
import json
import sys
from pathlib import Path

train_path, dev_path, skills_path, output_dir = map(Path, sys.argv[1:])
output_dir.mkdir(parents=True, exist_ok=True)

def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def selected(row):
    generated = row.get("generated") if isinstance(row.get("generated"), dict) else {}
    return row.get("selected_skill_ids") or generated.get("selected_skill_ids") or []

def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

train = [row for row in read_rows(train_path) if len(selected(row)) > 1][:16]
dev = [row for row in read_rows(dev_path) if len(selected(row)) > 1][:8]
skills = read_rows(skills_path)[:16]
if len(train) != 16 or len(dev) != 8 or len(skills) != 16:
    raise SystemExit("preflight data selection did not produce 16 train, 8 dev, and 16 catalog rows")
write_rows(output_dir / "train.jsonl", train)
write_rows(output_dir / "dev.jsonl", dev)
write_rows(output_dir / "skills.jsonl", skills)
PY

  run_torch -m unittest tests.test_skill_gr_pipeline
  run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
    --rows "${output}/data/train.jsonl" \
    --dev-rows "${output}/data/dev.jsonl" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${output}/data/skills.jsonl" \
    --output-dir "${output}/model" \
    --model "${SKILLGR_MODEL}" \
    --query-loss sibling_masked_ce \
    --catalog-epochs 1 --catalog-max-steps 1 \
    --epochs 1 --max-steps 2 \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --batch-size 1 --eval-batch-size 2 \
    --lr "${SKILLGR_LR}" \
    --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
    --lr-scheduler constant \
    --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
    --logging-steps 1 --eval-steps 1 --save-steps 0 \
    --no-padding-distributed-sampler \
    --seed "${SKILLGR_SEED}" \
    --load-best-model-at-end \
    --bf16
  eval_preflight_artifacts
}

eval_query_checkpoint() {
  local arm="$1"
  local model_dir="$2"
  local split="$3"
  local rows="$4"
  local output="${EVAL_ROOT}/${arm}/${split}"
  require_absent "${output}"
  run_torch scripts/skill_gr_pipeline.py eval-single-sid-decoder \
    --rows "${rows}" \
    --model-dir "${model_dir}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${output}" \
    --split-name "${split}" \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --num-beams "${SKILLGR_NUM_BEAMS}" \
    --rank-aggregation beam_order --k-values 1,10,50 \
    --log-every 25
}

run_query_from_init() {
  local query_loss="$1"
  local arm="$2"
  local init="$3"
  local prerequisite="$4"
  local output="${SKILLGR_OUTPUT_ROOT}/${arm}"
  require_file "${prerequisite}"
  require_file "${init}/structured_decoder.pt"
  require_absent "${output}"
  record_runtime
  run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
    --rows "${SKILLGR_ROWS}" \
    --dev-rows "${SKILLGR_DEV_ROWS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --output-dir "${output}" \
    --model "${SKILLGR_MODEL}" \
    --init-model-dir "${init}" \
    --no-reset-optimizer \
    --query-loss "${query_loss}" \
    --catalog-epochs 0 --catalog-max-steps 0 \
    --epochs "${SKILLGR_QUERY_EPOCHS}" \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --eval-batch-size "${SKILLGR_EVAL_PER_DEVICE_BATCH}" \
    --lr "${SKILLGR_LR}" \
    --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
    --lr-scheduler constant \
    --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
    --logging-steps 50 \
    --eval-steps 1000 \
    --save-steps 0 \
    --no-padding-distributed-sampler \
    --seed "${SKILLGR_SEED}" \
    --load-best-model-at-end \
    --bf16

  verify_epoch_history \
    "${output}/loss_history.jsonl" query "${SKILLGR_QUERY_EPOCHS}" 646705
  eval_query_checkpoint "${arm}" "${output}/final" private_test_1000 "${SKILLGR_PRIVATE_TEST_ROWS}"
  eval_query_checkpoint "${arm}" "${output}/final" official_core75 "${SKILLGR_EVAL_ROWS}"
  eval_catalog_teacher_forced "${arm}_final" "${output}/final"
  touch "${SKILLGR_OUTPUT_ROOT}/${arm}.done"
}

run_query() {
  local query_loss="$1"
  local catalog_epoch="$2"
  local arm="query_${query_loss}_from_catalog_e${catalog_epoch}"
  run_query_from_init \
    "${query_loss}" \
    "${arm}" \
    "$(catalog_checkpoint "${catalog_epoch}")" \
    "${SKILLGR_OUTPUT_ROOT}/catalog.done"
}

run_query_from_selected_long_catalog() {
  local convergence_summary
  local prerequisite
  local selected_epoch
  local selected_checkpoint
  if [[ -f "${SKILLGR_OUTPUT_ROOT}/catalog_extend_e60.done" ]]; then
    prerequisite="${SKILLGR_OUTPUT_ROOT}/catalog_extend_e60.done"
    convergence_summary="${EVAL_ROOT}/catalog_long_e60_convergence.json"
  else
    prerequisite="${SKILLGR_OUTPUT_ROOT}/catalog_long_e30.done"
    convergence_summary="${EVAL_ROOT}/catalog_long_convergence.json"
  fi
  require_file "${prerequisite}"
  require_file "${convergence_summary}"
  selected_epoch="$(python3 - "${convergence_summary}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected_epoch = payload.get("selected_epoch")
if payload.get("converged") is not True or selected_epoch is None:
    raise SystemExit("long catalog has no checkpoint satisfying the convergence criterion")
if not isinstance(selected_epoch, int) or isinstance(selected_epoch, bool):
    raise SystemExit(f"invalid selected catalog epoch: {selected_epoch!r}")
print(selected_epoch)
PY
)"
  if (( selected_epoch < 30 )); then
    selected_checkpoint="$(long_catalog_checkpoint "${selected_epoch}")"
  elif [[ "${prerequisite}" == "${SKILLGR_OUTPUT_ROOT}/catalog_extend_e60.done" ]]; then
    selected_checkpoint="$(extended_catalog_checkpoint "${selected_epoch}")"
  else
    selected_checkpoint="$(long_catalog_checkpoint "${selected_epoch}")"
  fi
  run_query_from_init \
    expanded_ce \
    "query_expanded_ce_from_catalog_long_e${selected_epoch}" \
    "${selected_checkpoint}" \
    "${prerequisite}"
}

case "${STAGE}" in
  preflight) run_preflight ;;
  preflight_eval) run_preflight_eval ;;
  catalog) run_catalog ;;
  catalog_long_e30) run_catalog_long ;;
  catalog_extend_e60) run_catalog_extend_e60 ;;
  catalog_resume_diagnostics) run_catalog_resume_diagnostics ;;
  query_expanded_ce) run_query expanded_ce 3 ;;
  query_sibling_masked_ce) run_query sibling_masked_ce 3 ;;
  query_expanded_ce_e6) run_query expanded_ce 6 ;;
  query_expanded_ce_long_selected) run_query_from_selected_long_catalog ;;
  *) echo "unknown stage: ${STAGE}" >&2; exit 1 ;;
esac
