#!/usr/bin/env bash
set -euo pipefail

# Stage-oriented launcher for one 8xA800 rudder-r2 node.  Submit independent
# one-node jobs for each DAG stage; every GPU-capable command uses all 8 GPUs.

STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
  echo "usage: $0 {teacher|preflight|pag_shared|ripor_mseq|final_ripor|final_pag|final_ripor_pag|eval_baseline|eval_ripor|eval_pag|eval_ripor_pag|all}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${SKILLGR_CONFIG:-${REPO_ROOT}/configs/letter_ripor_pag_20260804.env}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "missing locked experiment config: ${CONFIG_PATH}" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set +a

required_vars=(
  SKILLGR_ROWS SKILLGR_DEV_ROWS SKILLGR_SKILLS SKILLGR_SEMANTIC_IDS
  SKILLGR_LETTER_MANIFEST SKILLGR_BASE_DECODER SKILLGR_TEACHER_MODEL
  SKILLGR_MODEL SKILLGR_OUTPUT_ROOT SKILLGR_MSP_STEPS SKILLGR_MSET_STEPS
  SKILLGR_RIPOR_STAGE_STEPS SKILLGR_FINAL_STAGE_STEPS
)
for variable_name in "${required_vars[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "missing required experiment variable: ${variable_name}" >&2
    exit 2
  fi
done
if [[ "${SKILLGR_OUTPUT_ROOT}" != /models_rw/* ]]; then
  echo "SKILLGR_OUTPUT_ROOT must be under /models_rw" >&2
  exit 2
fi

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
bash "${ALIAS_HELPER}" || exit 1

for input_path in \
  "${SKILLGR_ROWS}" \
  "${SKILLGR_DEV_ROWS}" \
  "${SKILLGR_SKILLS}" \
  "${SKILLGR_SEMANTIC_IDS}" \
  "${SKILLGR_LETTER_MANIFEST}" \
  "${SKILLGR_BASE_DECODER}/structured_decoder.pt" \
  "${SKILLGR_TEACHER_MODEL}/model.safetensors" \
  "${SKILLGR_MODEL}/config.json" \
  "${SKILLGR_MODEL}/model.safetensors"; do
  if [[ ! -e "${input_path}" ]]; then
    echo "missing required input: ${input_path}" >&2
    exit 2
  fi
done

mkdir -p "${SKILLGR_OUTPUT_ROOT}/logs"
JOB_LABEL="${SKILLGR_JOB_NAME:-${STAGE}}"
exec > >(tee "${SKILLGR_OUTPUT_ROOT}/logs/${JOB_LABEL}.log") 2>&1
cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

MODEL="${SKILLGR_MODEL}"
PER_DEVICE_BATCH="${SKILLGR_PER_DEVICE_BATCH:-8}"
MAX_SOURCE_LENGTH="${SKILLGR_MAX_SOURCE_LENGTH:-2048}"
SEED="${SKILLGR_SEED:-20260626}"
LR="${SKILLGR_LR:-0.0005}"
WEIGHT_DECAY="${SKILLGR_WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${SKILLGR_WARMUP_RATIO:-0.045}"
PAG_SET_SIZE="${SKILLGR_PAG_SET_SIZE:-64}"
PAG_TOP_N="${SKILLGR_PAG_TOP_N:-1000}"
PAG_SET_WEIGHT="${SKILLGR_PAG_SET_WEIGHT:-1.0}"
NUM_BEAMS="${SKILLGR_NUM_BEAMS:-50}"
TEACHER_QUERY_MAX_CHARS="${SKILLGR_TEACHER_QUERY_MAX_CHARS:-1500}"
TEACHER_DESC_MAX="${SKILLGR_TEACHER_DESC_MAX:-300}"
TEACHER_BODY_MAX="${SKILLGR_TEACHER_BODY_MAX:-2500}"

TEACHER_DIR="${SKILLGR_OUTPUT_ROOT}/teacher_panel"
TEACHER_PANEL="${TEACHER_DIR}/predictions.jsonl"
SETTING_CONTRACT="${SKILLGR_OUTPUT_ROOT}/setting_contract.json"
MSP_DIR="${SKILLGR_OUTPUT_ROOT}/pag_msp"
SET_IDS="${MSP_DIR}/pag_set_ids.json"
MSET_DIR="${SKILLGR_OUTPUT_ROOT}/pag_mset"
RIPOR_MSEQ_DIR="${SKILLGR_OUTPUT_ROOT}/ripor_mseq"
RIPOR_ONLY_DIR="${SKILLGR_OUTPUT_ROOT}/ripor_only"
PAG_ONLY_DIR="${SKILLGR_OUTPUT_ROOT}/pag_only"
RIPOR_PAG_DIR="${SKILLGR_OUTPUT_ROOT}/ripor_pag"
PREFLIGHT_DIR="${SKILLGR_OUTPUT_ROOT}/preflight"

run_torch() {
  torchrun --standalone --nproc_per_node=8 "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required dependency is not complete: $1" >&2
    exit 3
  fi
}

ensure_fresh_dir() {
  local target="$1"
  if [[ -e "${target}" ]] && [[ -n "$(find "${target}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing to overwrite non-empty stage output: ${target}" >&2
    exit 3
  fi
  mkdir -p "${target}"
}

verify_setting_contract() {
  require_file "${SETTING_CONTRACT}"
  python3 - "${SETTING_CONTRACT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

contract_path = Path(sys.argv[1])
payload = json.loads(contract_path.read_text(encoding="utf-8"))
expected_paths = {
    "rows": os.environ["SKILLGR_ROWS"],
    "dev_rows": os.environ["SKILLGR_DEV_ROWS"],
    "skills": os.environ["SKILLGR_SKILLS"],
    "semantic_ids": os.environ["SKILLGR_SEMANTIC_IDS"],
    "letter_manifest": os.environ["SKILLGR_LETTER_MANIFEST"],
    "base_decoder": str(Path(os.environ["SKILLGR_BASE_DECODER"]) / "structured_decoder.pt"),
    "teacher_panel": str(Path(os.environ["SKILLGR_OUTPUT_ROOT"]) / "teacher_panel" / "predictions.jsonl"),
    "t5_config": str(Path(os.environ["SKILLGR_MODEL"]) / "config.json"),
    "t5_weights": str(Path(os.environ["SKILLGR_MODEL"]) / "model.safetensors"),
}
recorded = payload.get("input_files") or {}
for name, path in expected_paths.items():
    item = recorded.get(name) or {}
    if item.get("path") != path:
        raise SystemExit(f"setting contract path drift for {name}: {item.get('path')} != {path}")
    actual = sha256(path)
    if item.get("sha256") != actual:
        raise SystemExit(f"setting contract SHA-256 drift for {name}: {actual}")
expected_scalars = {
    "seed": int(os.environ["SKILLGR_SEED"]),
    "per_device_batch": int(os.environ["SKILLGR_PER_DEVICE_BATCH"]),
    "world_size": 8,
    "global_batch": 8 * int(os.environ["SKILLGR_PER_DEVICE_BATCH"]),
    "lr": float(os.environ["SKILLGR_LR"]),
    "weight_decay": float(os.environ["SKILLGR_WEIGHT_DECAY"]),
    "warmup_ratio": float(os.environ["SKILLGR_WARMUP_RATIO"]),
    "max_source_length": int(os.environ["SKILLGR_MAX_SOURCE_LENGTH"]),
    "teacher_query_max_chars": int(os.environ["SKILLGR_TEACHER_QUERY_MAX_CHARS"]),
    "teacher_desc_max": int(os.environ["SKILLGR_TEACHER_DESC_MAX"]),
    "teacher_body_max": int(os.environ["SKILLGR_TEACHER_BODY_MAX"]),
    "msp_steps": int(os.environ["SKILLGR_MSP_STEPS"]),
    "mset_steps": int(os.environ["SKILLGR_MSET_STEPS"]),
    "ripor_stage_steps": int(os.environ["SKILLGR_RIPOR_STAGE_STEPS"]),
    "final_stage_steps": int(os.environ["SKILLGR_FINAL_STAGE_STEPS"]),
    "pag_set_size": int(os.environ["SKILLGR_PAG_SET_SIZE"]),
    "pag_top_n": int(os.environ["SKILLGR_PAG_TOP_N"]),
    "pag_set_weight": float(os.environ["SKILLGR_PAG_SET_WEIGHT"]),
    "num_beams": int(os.environ["SKILLGR_NUM_BEAMS"]),
}
for name, expected in expected_scalars.items():
    if payload.get(name) != expected:
        raise SystemExit(f"setting contract scalar drift for {name}: {payload.get(name)} != {expected}")
print(f"verified setting contract: {contract_path}", flush=True)
PY
}

stage_teacher() {
  ensure_fresh_dir "${TEACHER_DIR}"
  run_torch scripts/skillrouter_embedding_train.py eval \
    --rows "${SKILLGR_ROWS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --model "${SKILLGR_TEACHER_MODEL}" \
    --output-dir "${TEACHER_DIR}" \
    --split-name train_teacher \
    --k-values 1,50 \
    --max-query-length "${MAX_SOURCE_LENGTH}" \
    --max-skill-length "${MAX_SOURCE_LENGTH}" \
    --query-max-chars "${TEACHER_QUERY_MAX_CHARS}" \
    --desc-max "${TEACHER_DESC_MAX}" \
    --body-max "${TEACHER_BODY_MAX}" \
    --batch-size "${PER_DEVICE_BATCH}" \
    --bf16
  require_file "${TEACHER_PANEL}"
  python3 - "${SETTING_CONTRACT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

root = Path(os.environ["SKILLGR_OUTPUT_ROOT"])
paths = {
    "rows": os.environ["SKILLGR_ROWS"],
    "dev_rows": os.environ["SKILLGR_DEV_ROWS"],
    "skills": os.environ["SKILLGR_SKILLS"],
    "semantic_ids": os.environ["SKILLGR_SEMANTIC_IDS"],
    "letter_manifest": os.environ["SKILLGR_LETTER_MANIFEST"],
    "base_decoder": str(Path(os.environ["SKILLGR_BASE_DECODER"]) / "structured_decoder.pt"),
    "teacher_panel": str(root / "teacher_panel" / "predictions.jsonl"),
    "t5_config": str(Path(os.environ["SKILLGR_MODEL"]) / "config.json"),
    "t5_weights": str(Path(os.environ["SKILLGR_MODEL"]) / "model.safetensors"),
}
payload = {
    "settings": ["ripor", "pag", "ripor_pag"],
    "fixed_letter_semantic_ids": True,
    "query_target_mode": "single_sid",
    "multi_ground_truth_policy": "K expanded examples; current target is the sole positive; all source GT are negative exclusions",
    "shared_ripor_mseq": str(root / "ripor_mseq" / "final"),
    "shared_pag_mset": str(root / "pag_mset" / "final"),
    "reset_optimizer_each_stage": True,
    "scheduler": "linear_warmup_linear_decay",
    "input_files": {name: {"path": path, "sha256": sha256(path)} for name, path in paths.items()},
    "seed": int(os.environ["SKILLGR_SEED"]),
    "per_device_batch": int(os.environ["SKILLGR_PER_DEVICE_BATCH"]),
    "world_size": 8,
    "global_batch": 8 * int(os.environ["SKILLGR_PER_DEVICE_BATCH"]),
    "lr": float(os.environ["SKILLGR_LR"]),
    "weight_decay": float(os.environ["SKILLGR_WEIGHT_DECAY"]),
    "warmup_ratio": float(os.environ["SKILLGR_WARMUP_RATIO"]),
    "max_source_length": int(os.environ["SKILLGR_MAX_SOURCE_LENGTH"]),
    "teacher_query_max_chars": int(os.environ["SKILLGR_TEACHER_QUERY_MAX_CHARS"]),
    "teacher_desc_max": int(os.environ["SKILLGR_TEACHER_DESC_MAX"]),
    "teacher_body_max": int(os.environ["SKILLGR_TEACHER_BODY_MAX"]),
    "msp_steps": int(os.environ["SKILLGR_MSP_STEPS"]),
    "mset_steps": int(os.environ["SKILLGR_MSET_STEPS"]),
    "ripor_stage_steps": int(os.environ["SKILLGR_RIPOR_STAGE_STEPS"]),
    "final_stage_steps": int(os.environ["SKILLGR_FINAL_STAGE_STEPS"]),
    "pag_set_size": int(os.environ["SKILLGR_PAG_SET_SIZE"]),
    "pag_top_n": int(os.environ["SKILLGR_PAG_TOP_N"]),
    "pag_set_weight": float(os.environ["SKILLGR_PAG_SET_WEIGHT"]),
    "num_beams": int(os.environ["SKILLGR_NUM_BEAMS"]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

train_msp() {
  local rows="$1" output="$2" steps="$3"
  ensure_fresh_dir "${output}"
  run_torch scripts/skill_gr_pipeline.py train-pag-sparse-encoder \
    --rows "${rows}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --hard-negatives "${TEACHER_PANEL}" \
    --output-dir "${output}" \
    --setting-contract "${SETTING_CONTRACT}" \
    --model "${MODEL}" \
    --max-length "${MAX_SOURCE_LENGTH}" \
    --top-k "${PAG_SET_SIZE}" \
    --max-steps "${steps}" \
    --batch-size "${PER_DEVICE_BATCH}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --seed "${SEED}" \
    --bf16
  require_file "${output}/pag_sparse_encoder.pt"
  require_file "${output}/pag_set_ids.json"
}

train_mset() {
  local rows="$1" dev_rows="$2" set_ids="$3" msp_dir="$4" output="$5" steps="$6"
  ensure_fresh_dir "${output}"
  run_torch scripts/skill_gr_pipeline.py train-structured-decoder \
    --query-target-mode single_sid \
    --rows "${rows}" \
    --dev-rows "${dev_rows}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --pag-set-ids "${set_ids}" \
    --pag-set-hard-negatives "${TEACHER_PANEL}" \
    --pag-set-loss-weight 1 \
    --pag-set-stage-only \
    --setting-contract "${SETTING_CONTRACT}" \
    --pag-sparse-checkpoint "${msp_dir}" \
    --sequential-ce-weight 0 \
    --catalog-epochs 0 \
    --max-steps "${steps}" \
    --eval-steps 0 \
    --no-load-best-model-at-end \
    --output-dir "${output}" \
    --model "${MODEL}" \
    --max-source-length "${MAX_SOURCE_LENGTH}" \
    --batch-size "${PER_DEVICE_BATCH}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --seed "${SEED}" \
    --bf16
  require_file "${output}/final/structured_decoder.pt"
}

train_ripor_stage() {
  local rows="$1" dev_rows="$2" init_dir="$3" output="$4" steps="$5" step_offset="$6"
  ensure_fresh_dir "${output}"
  run_torch scripts/skill_gr_pipeline.py train-structured-decoder \
    --method-setting ripor \
    --query-target-mode single_sid \
    --letter-tokenizer-manifest "${SKILLGR_LETTER_MANIFEST}" \
    --setting-contract "${SETTING_CONTRACT}" \
    --rows "${rows}" \
    --dev-rows "${dev_rows}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --init-model-dir "${init_dir}" \
    --reset-optimizer \
    --ripor-hard-negatives "${TEACHER_PANEL}" \
    --ripor-loss-weight 1 \
    --sequential-ce-weight 0 \
    --catalog-epochs 0 \
    --max-steps "${steps}" \
    --step-offset "${step_offset}" \
    --eval-steps 0 \
    --no-load-best-model-at-end \
    --output-dir "${output}" \
    --model "${MODEL}" \
    --max-source-length "${MAX_SOURCE_LENGTH}" \
    --batch-size "${PER_DEVICE_BATCH}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --seed "${SEED}" \
    --bf16
  require_file "${output}/final/structured_decoder.pt"
}

train_pag_joint() {
  local method="$1" rows="$2" dev_rows="$3" set_ids="$4" mset="$5" mseq="$6" output="$7" steps="$8"
  local ripor_weight=0 ce_weight=1
  if [[ "${method}" == "ripor_pag" ]]; then
    ripor_weight=1
    ce_weight=0
  fi
  ensure_fresh_dir "${output}"
  run_torch scripts/skill_gr_pipeline.py train-structured-decoder \
    --method-setting "${method}" \
    --query-target-mode single_sid \
    --letter-tokenizer-manifest "${SKILLGR_LETTER_MANIFEST}" \
    --setting-contract "${SETTING_CONTRACT}" \
    --rows "${rows}" \
    --dev-rows "${dev_rows}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --pag-set-ids "${set_ids}" \
    --pag-set-hard-negatives "${TEACHER_PANEL}" \
    --pag-set-loss-weight 1 \
    --pag-set-init-model-dir "${mset}" \
    --pag-seq-init-model-dir "${mseq}" \
    --reset-optimizer \
    --ripor-hard-negatives "${TEACHER_PANEL}" \
    --ripor-loss-weight "${ripor_weight}" \
    --sequential-ce-weight "${ce_weight}" \
    --catalog-epochs 0 \
    --max-steps "${steps}" \
    --eval-steps 0 \
    --no-load-best-model-at-end \
    --output-dir "${output}" \
    --model "${MODEL}" \
    --max-source-length "${MAX_SOURCE_LENGTH}" \
    --batch-size "${PER_DEVICE_BATCH}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --seed "${SEED}" \
    --bf16
  require_file "${output}/final/structured_decoder.pt"
}

eval_plain() {
  local rows="$1" model_dir="$2" output="$3" expected="$4"
  ensure_fresh_dir "${output}"
  local expected_args=()
  if [[ "${expected}" != custom ]]; then
    expected_args=(--expected-method-setting "${expected}")
  fi
  run_torch scripts/skill_gr_pipeline.py eval-single-sid-decoder \
    "${expected_args[@]}" \
    --rows "${rows}" \
    --model-dir "${model_dir}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${output}" \
    --num-beams "${NUM_BEAMS}" \
    --k-values 1,10,50
}

eval_pag() {
  local rows="$1" model_dir="$2" output_root="$3" expected="$4" set_ids="$5"
  ensure_fresh_dir "${output_root}"
  run_torch scripts/skill_gr_pipeline.py eval-single-sid-decoder \
    --expected-method-setting "${expected}" \
    --rows "${rows}" \
    --model-dir "${model_dir}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --pag-set-ids "${set_ids}" \
    --pag-set-weight "${PAG_SET_WEIGHT}" \
    --pag-set-top-n "${PAG_TOP_N}" \
    --output-dir "${output_root}/planning" \
    --num-beams "${NUM_BEAMS}" \
    --k-values 1,10,50
  run_torch scripts/skill_gr_pipeline.py eval-single-sid-decoder \
    --expected-method-setting "${expected}" \
    --rows "${rows}" \
    --model-dir "${model_dir}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --pag-set-ids "${set_ids}" \
    --pag-set-weight 0 \
    --pag-set-top-n 0 \
    --allow-pag-weight-zero-diagnostic \
    --output-dir "${output_root}/weight0_diagnostic" \
    --num-beams "${NUM_BEAMS}" \
    --k-values 1,10,50
}

stage_preflight() {
  require_file "${TEACHER_PANEL}"
  ensure_fresh_dir "${PREFLIGHT_DIR}"
  python3 - "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${PREFLIGHT_DIR}" <<'PY'
import sys
from pathlib import Path

train_source, dev_source, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
for source, target, count in (
    (train_source, output / "train32.jsonl", 32),
    (dev_source, output / "dev8.jsonl", 8),
):
    with source.open("r", encoding="utf-8") as handle:
        rows = [next(handle) for _ in range(count)]
    target.write_text("".join(rows), encoding="utf-8")
PY
  python3 -m unittest tests.test_skill_gr_pipeline
  run_torch scripts/ddp_objective_preflight.py
  local train_rows="${PREFLIGHT_DIR}/train32.jsonl"
  local dev_rows="${PREFLIGHT_DIR}/dev8.jsonl"
  local msp="${PREFLIGHT_DIR}/pag_msp"
  local set_ids="${msp}/pag_set_ids.json"
  local mset="${PREFLIGHT_DIR}/pag_mset"
  local mseq="${PREFLIGHT_DIR}/ripor_mseq"
  train_msp "${train_rows}" "${msp}" 2
  train_mset "${train_rows}" "${dev_rows}" "${set_ids}" "${msp}" "${mset}" 2
  train_ripor_stage "${train_rows}" "${dev_rows}" "${SKILLGR_BASE_DECODER}" "${mseq}" 2 0
  train_ripor_stage "${train_rows}" "${dev_rows}" "${mseq}/final" "${PREFLIGHT_DIR}/ripor_only" 2 2
  train_pag_joint pag "${train_rows}" "${dev_rows}" "${set_ids}" "${mset}/final" "${SKILLGR_BASE_DECODER}" "${PREFLIGHT_DIR}/pag_only" 2
  train_pag_joint ripor_pag "${train_rows}" "${dev_rows}" "${set_ids}" "${mset}/final" "${mseq}/final" "${PREFLIGHT_DIR}/ripor_pag" 2
  eval_pag "${dev_rows}" "${PREFLIGHT_DIR}/ripor_pag/final" "${PREFLIGHT_DIR}/eval_ripor_pag" ripor_pag "${set_ids}"
  touch "${PREFLIGHT_DIR}/PASSED"
}

run_stage() {
  if [[ "$1" != teacher ]]; then
    verify_setting_contract
  fi
  case "$1" in
    teacher)
      stage_teacher
      ;;
    preflight)
      stage_preflight
      ;;
    pag_shared)
      require_file "${PREFLIGHT_DIR}/PASSED"
      require_file "${TEACHER_PANEL}"
      train_msp "${SKILLGR_ROWS}" "${MSP_DIR}" "${SKILLGR_MSP_STEPS}"
      train_mset "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SET_IDS}" "${MSP_DIR}" "${MSET_DIR}" "${SKILLGR_MSET_STEPS}"
      ;;
    ripor_mseq)
      require_file "${PREFLIGHT_DIR}/PASSED"
      require_file "${TEACHER_PANEL}"
      train_ripor_stage "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SKILLGR_BASE_DECODER}" "${RIPOR_MSEQ_DIR}" "${SKILLGR_RIPOR_STAGE_STEPS}" 0
      ;;
    final_ripor)
      require_file "${RIPOR_MSEQ_DIR}/final/structured_decoder.pt"
      train_ripor_stage "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${RIPOR_MSEQ_DIR}/final" "${RIPOR_ONLY_DIR}" "${SKILLGR_FINAL_STAGE_STEPS}" "${SKILLGR_RIPOR_STAGE_STEPS}"
      ;;
    final_pag)
      require_file "${MSET_DIR}/final/structured_decoder.pt"
      require_file "${SET_IDS}"
      train_pag_joint pag "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SET_IDS}" "${MSET_DIR}/final" "${SKILLGR_BASE_DECODER}" "${PAG_ONLY_DIR}" "${SKILLGR_FINAL_STAGE_STEPS}"
      ;;
    final_ripor_pag)
      require_file "${MSET_DIR}/final/structured_decoder.pt"
      require_file "${RIPOR_MSEQ_DIR}/final/structured_decoder.pt"
      require_file "${SET_IDS}"
      train_pag_joint ripor_pag "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SET_IDS}" "${MSET_DIR}/final" "${RIPOR_MSEQ_DIR}/final" "${RIPOR_PAG_DIR}" "${SKILLGR_FINAL_STAGE_STEPS}"
      ;;
    eval_baseline)
      eval_plain "${SKILLGR_DEV_ROWS}" "${SKILLGR_BASE_DECODER}" "${SKILLGR_OUTPUT_ROOT}/baseline_eval" custom
      ;;
    eval_ripor)
      require_file "${RIPOR_ONLY_DIR}/final/structured_decoder.pt"
      eval_plain "${SKILLGR_DEV_ROWS}" "${RIPOR_ONLY_DIR}/final" "${RIPOR_ONLY_DIR}/eval" ripor
      ;;
    eval_pag)
      require_file "${PAG_ONLY_DIR}/final/structured_decoder.pt"
      require_file "${SET_IDS}"
      eval_pag "${SKILLGR_DEV_ROWS}" "${PAG_ONLY_DIR}/final" "${PAG_ONLY_DIR}/eval" pag "${SET_IDS}"
      ;;
    eval_ripor_pag)
      require_file "${RIPOR_PAG_DIR}/final/structured_decoder.pt"
      require_file "${SET_IDS}"
      eval_pag "${SKILLGR_DEV_ROWS}" "${RIPOR_PAG_DIR}/final" "${RIPOR_PAG_DIR}/eval" ripor_pag "${SET_IDS}"
      ;;
    *)
      echo "unknown stage: $1" >&2
      exit 2
      ;;
  esac
}

if [[ "${STAGE}" == all ]]; then
  for stage_name in teacher preflight pag_shared ripor_mseq final_ripor final_pag final_ripor_pag eval_baseline eval_ripor eval_pag eval_ripor_pag; do
    run_stage "${stage_name}"
  done
else
  run_stage "${STAGE}"
fi
