#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
ROOT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
git config --global --add safe.directory "${ROOT_DIR}"
cd "${ROOT_DIR}"

set -a
source configs/letter_ripor_anchored_20260805.env
set +a

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SKILLGR_MODEL

WORLD_SIZE=8
CONTRACT="${SKILLGR_OUTPUT_ROOT}/setting_contract.json"
LOG_DIR="${SKILLGR_OUTPUT_ROOT}/logs"
PREFLIGHT="${SKILLGR_OUTPUT_ROOT}/preflight"
M2="${SKILLGR_OUTPUT_ROOT}/m2"
M3_P1="${SKILLGR_OUTPUT_ROOT}/m3_prefix1"
M3_P2="${SKILLGR_OUTPUT_ROOT}/m3_prefix2"
M3_P3="${SKILLGR_OUTPUT_ROOT}/m3_prefix3"
M3_P4="${SKILLGR_OUTPUT_ROOT}/m3_prefix4"
MINING="${SKILLGR_OUTPUT_ROOT}/m3_self_negative_mining"
SELFNEG_PANEL="${MINING}/self_negatives.jsonl"
SELFNEG_ROWS="${MINING}/eligible_rows.jsonl"
SELFNEG_MANIFEST="${MINING}/self_negatives.manifest.json"
M4="${SKILLGR_OUTPUT_ROOT}/m4"
EVAL_ROOT="${SKILLGR_OUTPUT_ROOT}/eval"

mkdir -p "${LOG_DIR}"

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
  torchrun --standalone --nproc_per_node="${WORLD_SIZE}" "$@"
}

write_contract() {
  require_absent "${CONTRACT}"
  if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "setting contract requires a clean, fully committed worktree" >&2
    git status --short >&2
    exit 1
  fi
  python3 - "${CONTRACT}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

contract = Path(sys.argv[1])
paths = {
    "rows": os.environ["SKILLGR_ROWS"],
    "dev_rows": os.environ["SKILLGR_DEV_ROWS"],
    "private_test_rows": os.environ["SKILLGR_PRIVATE_TEST_ROWS"],
    "official_rows": os.environ["SKILLGR_EVAL_ROWS"],
    "skills": os.environ["SKILLGR_SKILLS"],
    "semantic_ids": os.environ["SKILLGR_SEMANTIC_IDS"],
    "letter_manifest": os.environ["SKILLGR_LETTER_MANIFEST"],
    "base_checkpoint": str(Path(os.environ["SKILLGR_BASE_DECODER"]) / "structured_decoder.pt"),
    "base_tokenizer": str(Path(os.environ["SKILLGR_BASE_DECODER"]) / "tokenizer.json"),
    "base_tokenizer_config": str(Path(os.environ["SKILLGR_BASE_DECODER"]) / "tokenizer_config.json"),
    "archived_official_predictions": os.environ["SKILLGR_ARCHIVED_OFFICIAL_PREDICTIONS"],
    "teacher_panel": os.environ["SKILLGR_TEACHER_PANEL"],
    "t5_config": str(Path(os.environ["SKILLGR_MODEL"]) / "config.json"),
    "t5_weights": str(Path(os.environ["SKILLGR_MODEL"]) / "model.safetensors"),
    "t5_tokenizer": str(Path(os.environ["SKILLGR_MODEL"]) / "tokenizer.json"),
    "t5_spiece": str(Path(os.environ["SKILLGR_MODEL"]) / "spiece.model"),
}
payload = {
    "artifact_format": "fixed_letter_ripor_anchored_contract_v1",
    "settings": ["B", "R_raw", "R_logprob_diagnostic", "B_union_R_anchor"],
    "fixed_letter_semantic_ids": True,
    "query_target_mode": "single_sid",
    "immutable_baseline": True,
    "baseline_score": "sum_conditional_log_probability_with_historical_control_semantics",
    "ripor_score": "bias_free_position_specific_sum_raw_conditional_logits_codes_only",
    "ripor_head_initialization": "copy_fixed_letter_ce_code_head_weight_without_bias",
    "stages": ["m2_full", "m3_prefix1", "m3_prefix2", "m3_prefix3", "m3_prefix4", "m4_self_negative_full"],
    "stage_steps": {
        "m2": int(os.environ["SKILLGR_M2_STEPS"]),
        "m3_each": int(os.environ["SKILLGR_M3_PREFIX_STEPS"]),
        "m4": int(os.environ["SKILLGR_M4_STEPS"]),
    },
    "optimizer": {
        "name": "AdamW",
        "lr": float(os.environ["SKILLGR_LR"]),
        "weight_decay": float(os.environ["SKILLGR_WEIGHT_DECAY"]),
        "warmup_ratio": float(os.environ["SKILLGR_WARMUP_RATIO"]),
        "scheduler": "linear",
        "reset_each_stage": True,
    },
    "distributed": {
        "world_size": 8,
        "per_device_batch": int(os.environ["SKILLGR_PER_DEVICE_BATCH"]),
        "global_batch": 8 * int(os.environ["SKILLGR_PER_DEVICE_BATCH"]),
        "precision": "bf16",
        "queue": "rudder-r1",
        "nodes": 1,
    },
    "negatives": {
        "per_positive": int(os.environ["SKILLGR_RIPOR_MAX_NEGATIVES"]),
        "sampling": "deterministic_random_per_presentation",
        "self_negative_pool": int(os.environ["SKILLGR_SELF_NEGATIVE_POOL"]),
        "sibling_gt": "excluded",
        "missing_teacher_score": "fail_or_audited_skip_before_M4",
    },
    "anchor": {
        "weight": float(os.environ["SKILLGR_ANCHOR_WEIGHT"]),
        "formula": "zscore(B_logprob)+lambda*zscore(R_raw); missing_branch=min_zscore-1",
        "candidate_domain": "union(B_logprob_top50,R_raw_top50)",
        "tie_break": "descending_score_then_ascending_skill_id",
        "gate_zero": "exact_baseline_bypass",
    },
    "seed": int(os.environ["SKILLGR_SEED"]),
    "max_source_length": int(os.environ["SKILLGR_MAX_SOURCE_LENGTH"]),
    "num_beams": int(os.environ["SKILLGR_NUM_BEAMS"]),
    "input_files": {name: {"path": path, "sha256": sha(path)} for name, path in paths.items()},
    "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "ripor_official_commit": "03fd0cbfafdeeb9cdd866e0c44727fec39a96a52",
}
contract.parent.mkdir(parents=True, exist_ok=True)
contract.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"contract": str(contract), "sha256": sha(contract)}, sort_keys=True))
PY
}

verify_contract() {
  require_file "${CONTRACT}"
  python3 - "${CONTRACT}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "m2": int(os.environ["SKILLGR_M2_STEPS"]),
    "m3_each": int(os.environ["SKILLGR_M3_PREFIX_STEPS"]),
    "m4": int(os.environ["SKILLGR_M4_STEPS"]),
}
if p.get("artifact_format") != "fixed_letter_ripor_anchored_contract_v1":
    raise SystemExit("wrong setting contract format")
if p.get("stage_steps") != expected:
    raise SystemExit(f"stage budget drift: {p.get('stage_steps')} != {expected}")
if p.get("distributed", {}).get("world_size") != 8:
    raise SystemExit("contract is not a one-node 8-GPU run")
if p.get("distributed", {}).get("queue") != "rudder-r1":
    raise SystemExit("contract is not bound to the requested rudder-r1 queue")
status = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=all"], text=True
).strip()
if status:
    raise SystemExit(f"worktree drift after contract creation:\n{status}")
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if p.get("code_commit") != head:
    raise SystemExit(f"code commit drift: {p.get('code_commit')} != {head}")
current_paths = {
    "rows": os.environ["SKILLGR_ROWS"],
    "dev_rows": os.environ["SKILLGR_DEV_ROWS"],
    "private_test_rows": os.environ["SKILLGR_PRIVATE_TEST_ROWS"],
    "official_rows": os.environ["SKILLGR_EVAL_ROWS"],
    "skills": os.environ["SKILLGR_SKILLS"],
    "semantic_ids": os.environ["SKILLGR_SEMANTIC_IDS"],
    "letter_manifest": os.environ["SKILLGR_LETTER_MANIFEST"],
    "base_checkpoint": str(Path(os.environ["SKILLGR_BASE_DECODER"]) / "structured_decoder.pt"),
    "base_tokenizer": str(Path(os.environ["SKILLGR_BASE_DECODER"]) / "tokenizer.json"),
    "base_tokenizer_config": str(Path(os.environ["SKILLGR_BASE_DECODER"]) / "tokenizer_config.json"),
    "archived_official_predictions": os.environ["SKILLGR_ARCHIVED_OFFICIAL_PREDICTIONS"],
    "teacher_panel": os.environ["SKILLGR_TEACHER_PANEL"],
    "t5_config": str(Path(os.environ["SKILLGR_MODEL"]) / "config.json"),
    "t5_weights": str(Path(os.environ["SKILLGR_MODEL"]) / "model.safetensors"),
    "t5_tokenizer": str(Path(os.environ["SKILLGR_MODEL"]) / "tokenizer.json"),
    "t5_spiece": str(Path(os.environ["SKILLGR_MODEL"]) / "spiece.model"),
}
recorded_files = p.get("input_files") or {}
if set(recorded_files) != set(current_paths):
    raise SystemExit("setting contract input-file schema drift")
for name, current_path in current_paths.items():
    recorded = recorded_files[name]
    if Path(str(recorded.get("path"))).resolve(strict=True) != Path(current_path).resolve(strict=True):
        raise SystemExit(f"setting contract input path drift: {name}")
    actual_sha = sha(current_path)
    if recorded.get("sha256") != actual_sha:
        raise SystemExit(f"setting contract input content drift: {name}")
print(f"verified contract: {sys.argv[1]}", flush=True)
PY
}

require_preflight() {
  require_file "${PREFLIGHT}/PASSED"
}

verify_stage_checkpoint() {
  local model_dir="$1" expected_prefix="$2" expected_init_dir="$3"
  require_file "${model_dir}/structured_decoder.pt"
  require_file "${expected_init_dir}/structured_decoder.pt"
  python3 - "${model_dir}" "${expected_prefix}" "${expected_init_dir}" "${CONTRACT}" <<'PY'
import hashlib
import sys
from pathlib import Path
import torch

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

model_dir, expected_prefix, init_dir, contract = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
payload = torch.load(model_dir / "structured_decoder.pt", map_location="cpu", mmap=True)
if payload.get("method_setting") != "ripor":
    raise SystemExit("stage checkpoint is not tagged as fixed-LETTER RIPOR")
if (payload.get("setting_contract") or {}).get("sha256") != sha(contract):
    raise SystemExit("stage checkpoint setting-contract SHA mismatch")
ripor = payload.get("ripor") or {}
expected_ripor = {
    "enabled": True,
    "loss_weight": 1.0,
    "fixed_prefix_max_length": expected_prefix,
    "score_head": "position_specific_linear_without_bias",
    "docid_score": "sum_raw_conditional_logits_codes_only",
    "control_score": "excluded",
    "strict_teacher_scores": True,
}
for key, expected in expected_ripor.items():
    if ripor.get(key) != expected:
        raise SystemExit(f"stage RIPOR metadata mismatch for {key}: {ripor.get(key)!r} != {expected!r}")
initialization = payload.get("initialization") or {}
recorded_init = initialization.get("init_checkpoint") or {}
expected_init_file = (init_dir / "structured_decoder.pt").resolve(strict=True)
if Path(str(recorded_init.get("path") or "")).resolve(strict=True) != expected_init_file:
    raise SystemExit("stage initialization checkpoint path mismatch")
if recorded_init.get("sha256") != sha(expected_init_file):
    raise SystemExit("stage initialization checkpoint SHA mismatch")
if payload.get("distributed_sampler") != "no_padding_graph_zero_tail_v1":
    raise SystemExit("stage did not use the no-repeat DDP sampler")
print(f"verified stage checkpoint: {model_dir}", flush=True)
PY
}

make_preflight_rows() {
  mkdir -p "${PREFLIGHT}"
  python3 - "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${PREFLIGHT}" <<'PY'
import sys
from pathlib import Path
train, dev, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for source, name, count in ((train, "train32.jsonl", 32), (dev, "dev8.jsonl", 8)):
    with source.open(encoding="utf-8") as handle:
        values = [next(handle) for _ in range(count)]
    (out / name).write_text("".join(values), encoding="utf-8")
PY
}

train_ripor() {
  local rows="$1" init="$2" output="$3" steps="$4" prefix="$5" negatives="$6" offset="$7" save_steps="$8"
  require_absent "${output}"
  run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
    --rows "${rows}" \
    --dev-rows "${SKILLGR_DEV_ROWS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --output-dir "${output}" \
    --model "${SKILLGR_MODEL}" \
    --init-model-dir "${init}" \
    --method-setting ripor \
    --letter-tokenizer-manifest "${SKILLGR_LETTER_MANIFEST}" \
    --setting-contract "${CONTRACT}" \
    --reset-optimizer \
    --catalog-epochs 0 \
    --catalog-max-steps 0 \
    --max-steps "${steps}" \
    --batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --eval-batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --lr "${SKILLGR_LR}" \
    --weight-decay "${SKILLGR_WEIGHT_DECAY}" \
    --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --ripor-loss-weight 1 \
    --sequential-ce-weight 0 \
    --ripor-hard-negatives "${negatives}" \
    --ripor-max-negatives "${SKILLGR_RIPOR_MAX_NEGATIVES}" \
    --ripor-negative-sampling random \
    --ripor-prefix-max-length "${prefix}" \
    --no-ripor-progressive-prefix \
    --step-offset "${offset}" \
    --eval-steps 0 \
    --no-load-best-model-at-end \
    --logging-steps 25 \
    --save-steps "${save_steps}" \
    --seed "${SKILLGR_SEED}" \
    --bf16
}

eval_decoder() {
  local rows="$1" model="$2" output="$3" expected="$4" head="$5" transform="$6"
  require_absent "${output}"
  local expected_args=()
  if [[ "${expected}" != custom ]]; then
    expected_args=(--expected-method-setting "${expected}")
  fi
  run_torch scripts/skill_gr_pipeline.py eval-single-sid-decoder \
    "${expected_args[@]}" \
    --rows "${rows}" \
    --model-dir "${model}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${output}" \
    --num-beams "${SKILLGR_NUM_BEAMS}" \
    --k-values 1,10,50 \
    --decoder-head "${head}" \
    --score-transform "${transform}"
}

eval_decoder_single_gpu() {
  local rows="$1" model="$2" output="$3" expected="$4" head="$5" transform="$6"
  require_absent "${output}"
  local expected_args=()
  if [[ "${expected}" != custom ]]; then
    expected_args=(--expected-method-setting "${expected}")
  fi
  python3 scripts/skill_gr_pipeline.py eval-single-sid-decoder \
    "${expected_args[@]}" \
    --rows "${rows}" \
    --model-dir "${model}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${output}" \
    --num-beams "${SKILLGR_NUM_BEAMS}" \
    --k-values 1,10,50 \
    --decoder-head "${head}" \
    --score-transform "${transform}"
}

compare_prediction_rankings() {
  local expected="$1" actual="$2" output="$3"
  python3 - "${expected}" "${actual}" "${output}" <<'PY'
import json, sys
import math
from pathlib import Path
def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
left, right = load(sys.argv[1]), load(sys.argv[2])
if len(left) != len(right):
    raise SystemExit(f"prediction count mismatch: {len(left)} != {len(right)}")
fields = ("index", "predicted_semantic_ids", "predicted_skill_ids", "ranked_skill_ids")
for i, (a, b) in enumerate(zip(left, right)):
    for field in fields:
        if a.get(field) != b.get(field):
            raise SystemExit(f"historical baseline mismatch row={i}, field={field}")
    a_scores = a.get("ranked_skill_scores") or {}
    b_scores = b.get("ranked_skill_scores") or {}
    if set(a_scores) != set(b_scores):
        raise SystemExit(f"historical baseline score-domain mismatch row={i}")
    for skill_id in a_scores:
        if not math.isclose(float(a_scores[skill_id]), float(b_scores[skill_id]), rel_tol=0.0, abs_tol=1e-6):
            raise SystemExit(f"historical baseline score mismatch row={i}, skill={skill_id}")
    if not math.isclose(float(a.get("score")), float(b.get("score")), rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(f"historical baseline best-beam score mismatch row={i}")
payload = {
    "status": "passed",
    "rows": len(left),
    "fields": [*fields, "ranked_skill_scores", "score"],
    "score_atol": 1e-6,
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
}

compare_current_predictions() {
  local single_gpu="$1" ddp="$2" output="$3"
  python3 - "${single_gpu}" "${ddp}" "${output}" <<'PY'
import json
import math
import sys
from pathlib import Path

def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

left, right = load(sys.argv[1]), load(sys.argv[2])
if len(left) != len(right):
    raise SystemExit(f"single/DDP prediction count mismatch: {len(left)} != {len(right)}")
fields = ("index", "predicted_semantic_ids", "predicted_skill_ids", "ranked_skill_ids")
maximum_score_error = 0.0
for row_index, (single, distributed) in enumerate(zip(left, right)):
    for field in fields:
        if single.get(field) != distributed.get(field):
            raise SystemExit(f"single/DDP mismatch row={row_index}, field={field}")
    single_scores = single.get("ranked_skill_scores") or {}
    distributed_scores = distributed.get("ranked_skill_scores") or {}
    if set(single_scores) != set(distributed_scores):
        raise SystemExit(f"single/DDP candidate-score domain mismatch row={row_index}")
    for skill_id in single_scores:
        error = abs(float(single_scores[skill_id]) - float(distributed_scores[skill_id]))
        if not math.isfinite(error):
            raise SystemExit(f"non-finite single/DDP score comparison row={row_index}, skill={skill_id}")
        maximum_score_error = max(maximum_score_error, error)
if maximum_score_error > 1e-6:
    raise SystemExit(f"single/DDP score error {maximum_score_error} exceeds 1e-6")
payload = {
    "status": "passed",
    "rows": len(left),
    "fields": list(fields),
    "max_score_error": maximum_score_error,
    "atol": 1e-6,
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
}

stage_preflight() {
  require_absent "${PREFLIGHT}"
  make_preflight_rows
  python3 -m unittest discover -s tests -p 'test_skill_gr_pipeline.py'
  python3 -m unittest discover -s tests -p 'test_ripor_anchored_artifacts.py'
  run_torch scripts/ddp_objective_preflight.py
  # Regression smoke for the CE-only fixed-LETTER path: RIPOR-only heads are
  # frozen, so DDP with find_unused_parameters=False must complete two steps.
  run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
    --rows "${PREFLIGHT}/train32.jsonl" \
    --dev-rows "${PREFLIGHT}/dev8.jsonl" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --output-dir "${PREFLIGHT}/ce_ddp_smoke" \
    --model "${SKILLGR_MODEL}" \
    --init-model-dir "${SKILLGR_BASE_DECODER}" \
    --method-setting custom \
    --letter-tokenizer-manifest "${SKILLGR_LETTER_MANIFEST}" \
    --setting-contract "${CONTRACT}" \
    --reset-optimizer \
    --catalog-epochs 0 --catalog-max-steps 0 --max-steps 2 \
    --batch-size "${SKILLGR_PER_DEVICE_BATCH}" --eval-batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --lr "${SKILLGR_LR}" --weight-decay "${SKILLGR_WEIGHT_DECAY}" --warmup-ratio 0 \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --ripor-loss-weight 0 --sequential-ce-weight 1 \
    --ripor-prefix-max-length 4 --no-ripor-progressive-prefix \
    --eval-steps 0 --no-load-best-model-at-end --logging-steps 1 --save-steps 0 \
    --seed "${SKILLGR_SEED}" --bf16
  # Supported custom catalog→RIPOR path: catalog CE must use the code heads
  # while the later ranking heads remain graph-connected under strict DDP.
  run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
    --rows "${PREFLIGHT}/train32.jsonl" \
    --dev-rows "${PREFLIGHT}/dev8.jsonl" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --output-dir "${PREFLIGHT}/catalog_then_ripor_ddp_smoke" \
    --model "${SKILLGR_MODEL}" \
    --init-model-dir "${SKILLGR_BASE_DECODER}" \
    --method-setting custom \
    --setting-contract "${CONTRACT}" \
    --reset-optimizer \
    --catalog-epochs 1 --catalog-max-steps 1 --max-steps 1 \
    --batch-size "${SKILLGR_PER_DEVICE_BATCH}" --eval-batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --lr "${SKILLGR_LR}" --weight-decay "${SKILLGR_WEIGHT_DECAY}" --warmup-ratio 0 \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --ripor-loss-weight 1 --sequential-ce-weight 0 \
    --ripor-hard-negatives "${SKILLGR_TEACHER_PANEL}" --ripor-max-negatives 1 \
    --ripor-negative-sampling random --ripor-prefix-max-length 4 --no-ripor-progressive-prefix \
    --eval-steps 0 --no-load-best-model-at-end --logging-steps 1 --save-steps 0 \
    --seed "${SKILLGR_SEED}" --bf16
  run_torch scripts/ripor_real_model_preflight.py \
    --base-decoder "${SKILLGR_BASE_DECODER}" \
    --architecture-model "${SKILLGR_MODEL}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --rows "${PREFLIGHT}/dev8.jsonl" \
    --output "${PREFLIGHT}/ripor_real_model_parity.json" \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}"

  eval_decoder_single_gpu "${PREFLIGHT}/dev8.jsonl" "${SKILLGR_BASE_DECODER}" "${PREFLIGHT}/baseline_single_gpu" custom ce logprob
  eval_decoder "${PREFLIGHT}/dev8.jsonl" "${SKILLGR_BASE_DECODER}" "${PREFLIGHT}/baseline_ddp" custom ce logprob
  compare_current_predictions \
    "${PREFLIGHT}/baseline_single_gpu/predictions.jsonl" \
    "${PREFLIGHT}/baseline_ddp/predictions.jsonl" \
    "${PREFLIGHT}/single_vs_ddp_score_parity.json"

  eval_decoder "${SKILLGR_EVAL_ROWS}" "${SKILLGR_BASE_DECODER}" "${PREFLIGHT}/baseline_official" custom ce logprob
  compare_prediction_rankings \
    "${SKILLGR_ARCHIVED_OFFICIAL_PREDICTIONS}" \
    "${PREFLIGHT}/baseline_official/predictions.jsonl" \
    "${PREFLIGHT}/historical_baseline_parity.json"
  python3 scripts/ripor_anchored_artifacts.py anchor \
    --rows "${SKILLGR_EVAL_ROWS}" \
    --baseline-predictions "${PREFLIGHT}/baseline_official/predictions.jsonl" \
    --ripor-predictions "${PREFLIGHT}/baseline_official/predictions.jsonl" \
    --output-dir "${PREFLIGHT}/gate0" \
    --anchor-weight 0

  train_ripor "${PREFLIGHT}/train32.jsonl" "${SKILLGR_BASE_DECODER}" "${PREFLIGHT}/resume_source" 2 4 "${SKILLGR_TEACHER_PANEL}" 0 1
  run_torch scripts/skill_gr_pipeline.py train-single-sid-decoder \
    --rows "${PREFLIGHT}/train32.jsonl" \
    --dev-rows "${SKILLGR_DEV_ROWS}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --skills-jsonl "${SKILLGR_SKILLS}" \
    --output-dir "${PREFLIGHT}/resume_target" \
    --model "${SKILLGR_MODEL}" \
    --init-model-dir "${PREFLIGHT}/resume_source/checkpoint-1" \
    --resume-from-checkpoint "${PREFLIGHT}/resume_source/checkpoint-1" \
    --method-setting ripor \
    --letter-tokenizer-manifest "${SKILLGR_LETTER_MANIFEST}" \
    --setting-contract "${CONTRACT}" \
    --no-reset-optimizer \
    --catalog-epochs 0 --catalog-max-steps 0 --max-steps 2 \
    --batch-size "${SKILLGR_PER_DEVICE_BATCH}" --eval-batch-size "${SKILLGR_PER_DEVICE_BATCH}" \
    --lr "${SKILLGR_LR}" --weight-decay "${SKILLGR_WEIGHT_DECAY}" --warmup-ratio "${SKILLGR_WARMUP_RATIO}" \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --ripor-loss-weight 1 --sequential-ce-weight 0 \
    --ripor-hard-negatives "${SKILLGR_TEACHER_PANEL}" --ripor-max-negatives 1 \
    --ripor-negative-sampling random --ripor-prefix-max-length 4 --no-ripor-progressive-prefix \
    --eval-steps 0 --no-load-best-model-at-end --logging-steps 1 --save-steps 0 \
    --seed "${SKILLGR_SEED}" --bf16
  python3 - "${PREFLIGHT}/resume_source/final/structured_decoder.pt" "${PREFLIGHT}/resume_target/final/structured_decoder.pt" "${PREFLIGHT}/resume_parity.json" <<'PY'
import json, sys, torch
left = torch.load(sys.argv[1], map_location="cpu", mmap=True)
right = torch.load(sys.argv[2], map_location="cpu", mmap=True)
if set(left["state_dict"]) != set(right["state_dict"]):
    raise SystemExit("resume state keys differ")
maximum = 0.0
for key in left["state_dict"]:
    maximum = max(maximum, float((left["state_dict"][key] - right["state_dict"][key]).abs().max()))
if maximum > 1e-8:
    raise SystemExit(f"resume parameter error {maximum} exceeds 1e-8")
payload = {"status": "passed", "max_parameter_error": maximum, "atol": 1e-8}
open(sys.argv[3], "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
  touch "${PREFLIGHT}/PASSED"
}

stage_m2() {
  train_ripor "${SKILLGR_ROWS}" "${SKILLGR_BASE_DECODER}" "${M2}" "${SKILLGR_M2_STEPS}" 4 "${SKILLGR_TEACHER_PANEL}" 0 "${SKILLGR_SAVE_STEPS}"
  verify_stage_checkpoint "${M2}/final" 4 "${SKILLGR_BASE_DECODER}"
}

stage_m3() {
  verify_stage_checkpoint "${M2}/final" 4 "${SKILLGR_BASE_DECODER}"
  local offset="${SKILLGR_M2_STEPS}"
  train_ripor "${SKILLGR_ROWS}" "${M2}/final" "${M3_P1}" "${SKILLGR_M3_PREFIX_STEPS}" 1 "${SKILLGR_TEACHER_PANEL}" "${offset}" "${SKILLGR_SAVE_STEPS}"
  verify_stage_checkpoint "${M3_P1}/final" 1 "${M2}/final"
  offset=$((offset + SKILLGR_M3_PREFIX_STEPS))
  train_ripor "${SKILLGR_ROWS}" "${M3_P1}/final" "${M3_P2}" "${SKILLGR_M3_PREFIX_STEPS}" 2 "${SKILLGR_TEACHER_PANEL}" "${offset}" "${SKILLGR_SAVE_STEPS}"
  verify_stage_checkpoint "${M3_P2}/final" 2 "${M3_P1}/final"
  offset=$((offset + SKILLGR_M3_PREFIX_STEPS))
  train_ripor "${SKILLGR_ROWS}" "${M3_P2}/final" "${M3_P3}" "${SKILLGR_M3_PREFIX_STEPS}" 3 "${SKILLGR_TEACHER_PANEL}" "${offset}" "${SKILLGR_SAVE_STEPS}"
  verify_stage_checkpoint "${M3_P3}/final" 3 "${M3_P2}/final"
  offset=$((offset + SKILLGR_M3_PREFIX_STEPS))
  train_ripor "${SKILLGR_ROWS}" "${M3_P3}/final" "${M3_P4}" "${SKILLGR_M3_PREFIX_STEPS}" 4 "${SKILLGR_TEACHER_PANEL}" "${offset}" "${SKILLGR_SAVE_STEPS}"
  verify_stage_checkpoint "${M3_P4}/final" 4 "${M3_P3}/final"
}

stage_mine() {
  verify_stage_checkpoint "${M3_P4}/final" 4 "${M3_P3}/final"
  require_absent "${MINING}"
  eval_decoder "${SKILLGR_ROWS}" "${M3_P4}/final" "${MINING}/raw_predictions" ripor ripor raw
  python3 scripts/ripor_anchored_artifacts.py seal-self-negatives \
    --rows "${SKILLGR_ROWS}" \
    --miner-predictions "${MINING}/raw_predictions/predictions.jsonl" \
    --miner-eval-manifest "${MINING}/raw_predictions/eval_manifest.json" \
    --miner-checkpoint "${M3_P4}/final" \
    --teacher-panel "${SKILLGR_TEACHER_PANEL}" \
    --output-panel "${SELFNEG_PANEL}" \
    --output-rows "${SELFNEG_ROWS}" \
    --output-manifest "${SELFNEG_MANIFEST}" \
    --max-negatives "${SKILLGR_SELF_NEGATIVE_POOL}"
}

stage_m4() {
  require_file "${SELFNEG_MANIFEST}"
  verify_stage_checkpoint "${M3_P4}/final" 4 "${M3_P3}/final"
  python3 scripts/ripor_anchored_artifacts.py verify-self-negatives \
    --manifest "${SELFNEG_MANIFEST}" \
    --rows "${SKILLGR_ROWS}" \
    --miner-predictions "${MINING}/raw_predictions/predictions.jsonl" \
    --miner-eval-manifest "${MINING}/raw_predictions/eval_manifest.json" \
    --miner-checkpoint "${M3_P4}/final" \
    --teacher-panel "${SKILLGR_TEACHER_PANEL}" \
    --output-panel "${SELFNEG_PANEL}" \
    --output-rows "${SELFNEG_ROWS}"
  local offset=$((SKILLGR_M2_STEPS + 4 * SKILLGR_M3_PREFIX_STEPS))
  train_ripor "${SELFNEG_ROWS}" "${M3_P4}/final" "${M4}" "${SKILLGR_M4_STEPS}" 4 "${SELFNEG_PANEL}" "${offset}" "${SKILLGR_SAVE_STEPS}"
  verify_stage_checkpoint "${M4}/final" 4 "${M3_P4}/final"
}

eval_split() {
  local name="$1" rows="$2"
  eval_decoder "${rows}" "${SKILLGR_BASE_DECODER}" "${EVAL_ROOT}/${name}/B" custom ce logprob
  eval_decoder "${rows}" "${M4}/final" "${EVAL_ROOT}/${name}/R_raw" ripor ripor raw
  eval_decoder "${rows}" "${M4}/final" "${EVAL_ROOT}/${name}/R_logprob" ripor ripor logprob
  python3 scripts/ripor_anchored_artifacts.py anchor \
    --rows "${rows}" \
    --baseline-predictions "${EVAL_ROOT}/${name}/B/predictions.jsonl" \
    --ripor-predictions "${EVAL_ROOT}/${name}/R_raw/predictions.jsonl" \
    --output-dir "${EVAL_ROOT}/${name}/B_union_R_anchor" \
    --anchor-weight "${SKILLGR_ANCHOR_WEIGHT}" \
    --max-rank 50 --k-values 1,10,50
  python3 scripts/ripor_anchored_artifacts.py anchor \
    --rows "${rows}" \
    --baseline-predictions "${EVAL_ROOT}/${name}/B/predictions.jsonl" \
    --ripor-predictions "${EVAL_ROOT}/${name}/R_raw/predictions.jsonl" \
    --output-dir "${EVAL_ROOT}/${name}/gate0" \
    --anchor-weight 0
}

stage_eval() {
  verify_stage_checkpoint "${M4}/final" 4 "${M3_P4}/final"
  require_absent "${EVAL_ROOT}"
  eval_split dev "${SKILLGR_DEV_ROWS}"
  eval_split official_core75 "${SKILLGR_EVAL_ROWS}"
  eval_split private_test "${SKILLGR_PRIVATE_TEST_ROWS}"
}

run_stage() {
  if [[ "$1" != contract ]]; then
    verify_contract
  fi
  case "$1" in
    m2|m3|mine|m4|eval) require_preflight ;;
  esac
  case "$1" in
    contract) write_contract ;;
    preflight) stage_preflight ;;
    m2) stage_m2 ;;
    m3) stage_m3 ;;
    mine) stage_mine ;;
    m4) stage_m4 ;;
    eval) stage_eval ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

if [[ "${STAGE}" == all ]]; then
  for stage_name in contract preflight m2 m3 mine m4 eval; do
    run_stage "${stage_name}"
  done
else
  run_stage "${STAGE}"
fi
