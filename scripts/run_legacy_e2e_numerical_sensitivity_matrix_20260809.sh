#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_REPO=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis
PHYSICAL_OUTPUT_BASE=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
OUTPUT_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/legacy_e2e_numerical_sensitivity_20260809_v1"

PY310_RUNTIME=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1/runtime.tar.zst
PY310_RUNTIME_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_RUNTIME=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1/site.tar.zst
PY311_RUNTIME_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
LEGACY_COMMIT=15f6c51131f3b6f3e8580aa4564b17cbe13a9790
LEGACY_PIPELINE_SHA256=5d60624b704e3563f1893b021e15c2c6da3cdb28a18a1ac3018ca3884de04f67

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite numerical-sensitivity output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/controls" "${OUTPUT_ROOT}/arms"
exec > >(tee "${OUTPUT_ROOT}/logs/node-8gpu.log") 2>&1
echo "launch_commit=${SKILLGR_LAUNCH_COMMIT:-unknown}"

TEMP_ROOT="$(mktemp -d /tmp/skillgr-e2e-numerical.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-e2e-numerical.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

check_sha256() {
  local path="$1" expected="$2" observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || {
    echo "SHA-256 mismatch for ${path}: expected=${expected}, observed=${observed}" >&2
    exit 1
  }
}

check_sha256 "${PY310_RUNTIME}" "${PY310_RUNTIME_SHA256}"
check_sha256 "${PY311_RUNTIME}" "${PY311_RUNTIME_SHA256}"
mkdir -p "${TEMP_ROOT}/py310" "${TEMP_ROOT}/py311"
tar --zstd -xf "${PY310_RUNTIME}" -C "${TEMP_ROOT}/py310"
tar --zstd -xf "${PY311_RUNTIME}" -C "${TEMP_ROOT}/py311"

PYTHON_BIN="${TEMP_ROOT}/py310/python/bin/python3"
PY310_SITE="${TEMP_ROOT}/py310/site"
PY311_SITE="${TEMP_ROOT}/py311/site"
NVIDIA_LIBRARY_PATHS="$(find "${PY311_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
[[ -n "${NVIDIA_LIBRARY_PATHS}" ]] || { echo "fixed NVIDIA libraries are missing" >&2; exit 1; }
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${CODE_ROOT}:${PY310_SITE}:${PY311_SITE}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

EXPECTED_NVIDIA_ROOT="${PY311_SITE}/nvidia" "${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch
import transformers
from transformers import T5Config, T5ForConditionalGeneration

expected = {
    "python": "3.10.12",
    "torch": "2.10.0+cu128",
    "cuda": "12.8",
    "cudnn": 91002,
    "transformers": "5.3.0",
    "cublas": "12.8.4.1",
    "cuda_runtime": "12.8.90",
    "cudnn_package": "9.10.2.21",
    "gpu_count": 8,
}
actual = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "cublas": importlib.metadata.version("nvidia-cublas-cu12"),
    "cuda_runtime": importlib.metadata.version("nvidia-cuda-runtime-cu12"),
    "cudnn_package": importlib.metadata.version("nvidia-cudnn-cu12"),
    "gpu_count": torch.cuda.device_count(),
}
if actual != expected:
    raise SystemExit(f"runtime mismatch: expected={expected}, actual={actual}")
matrix = torch.ones((8, 8), dtype=torch.bfloat16, device="cuda")
torch.matmul(matrix, matrix)
torch.cuda.synchronize()
tiny_t5 = T5ForConditionalGeneration(T5Config(
    vocab_size=32,
    d_model=16,
    d_kv=4,
    d_ff=32,
    num_layers=1,
    num_decoder_layers=1,
    num_heads=4,
))
del tiny_t5
library_names = ("libcublas.so", "libcublasLt.so", "libcudnn.so")
loaded = sorted({
    fields[-1]
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    if len(fields := line.split()) >= 6
    and any(name in fields[-1] for name in library_names)
})
root = Path(os.environ["EXPECTED_NVIDIA_ROOT"]).resolve()
for name in library_names:
    matches = [Path(path).resolve() for path in loaded if name in path]
    if not matches or any(not path.is_relative_to(root) for path in matches):
        raise SystemExit(f"loaded {name} outside fixed runtime: {matches}")
print(json.dumps({**actual, "gpu": torch.cuda.get_device_name(0), "libraries": loaded}, sort_keys=True))
PY

MODEL_SAFE="${TEMP_ROOT}/model-safetensors"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done

LEGACY_SOURCE="${TEMP_ROOT}/legacy-source"
mkdir -p "${LEGACY_SOURCE}"
git -C "${PHYSICAL_REPO}" archive --format=tar "${LEGACY_COMMIT}" | tar -xf - -C "${LEGACY_SOURCE}"
check_sha256 "${LEGACY_SOURCE}/scripts/skill_gr_pipeline.py" "${LEGACY_PIPELINE_SHA256}"

set -a
export SKILLGR_EXACT_MODEL="${MODEL_SAFE}"
source "${CODE_ROOT}/configs/legacy_catalog_base_exact_runtime_20260808.env"
set +a

for required in \
  "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SKILLGR_OFFICIAL_ROWS}" \
  "${SKILLGR_SKILLS}" "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_ARCHIVED_OFFICIAL}"; do
  [[ -f "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done
check_sha256 "${SKILLGR_ROWS}" "${SKILLGR_ROWS_SHA256}"
check_sha256 "${SKILLGR_DEV_ROWS}" "${SKILLGR_DEV_ROWS_SHA256}"
check_sha256 "${SKILLGR_OFFICIAL_ROWS}" "${SKILLGR_OFFICIAL_ROWS_SHA256}"
check_sha256 "${SKILLGR_SKILLS}" "${SKILLGR_SKILLS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"
check_sha256 "${SKILLGR_ARCHIVED_OFFICIAL}" "${SKILLGR_ARCHIVED_OFFICIAL_SHA256}"
"${PYTHON_BIN}" -m py_compile \
  "${CODE_ROOT}/scripts/run_controlled_legacy_pipeline.py" \
  "${LEGACY_SOURCE}/scripts/skill_gr_pipeline.py" \
  "${CODE_ROOT}/scripts/skill_gr_pipeline.py"

run_arm() {
  local label="$1" precision="$2" bf16_reduction="$3" foreach="$4"
  local preferred_blas="$5" workspace="$6"
  local arm_root="${OUTPUT_ROOT}/arms/${label}"
  local train_dir="${arm_root}/train"
  local eval_dir="${arm_root}/eval/official_core75"
  local audit="${OUTPUT_ROOT}/controls/${label}.json"
  mkdir -p "${arm_root}"
  exec > >(tee "${OUTPUT_ROOT}/logs/${label}.log") 2>&1

  local -a train_args=(
    train-single-sid-decoder
    --rows "${SKILLGR_ROWS}"
    --dev-rows "${SKILLGR_DEV_ROWS}"
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}"
    --skills-jsonl "${SKILLGR_SKILLS}"
    --output-dir "${train_dir}"
    --model "${MODEL_SAFE}"
    --catalog-epochs 3
    --epochs 5
    --max-source-length 2048
    --batch-size 16
    --eval-batch-size 16
    --lr 0.0003
    --weight-decay 0
    --logging-steps 50
    --eval-steps 200
    --seed 20260626
    --load-best-model-at-end
  )
  if [[ "${precision}" == "bf16" ]]; then
    train_args+=(--bf16)
  elif [[ "${precision}" != "fp32" ]]; then
    echo "unsupported precision: ${precision}" >&2
    return 2
  fi

  local -a controlled_command=(
    "${PYTHON_BIN}" "${CODE_ROOT}/scripts/run_controlled_legacy_pipeline.py"
    --audit "${audit}"
    --bf16-reduction "${bf16_reduction}"
    --foreach "${foreach}"
    --preferred-blas "${preferred_blas}"
    "${LEGACY_SOURCE}/scripts/skill_gr_pipeline.py"
    "${train_args[@]}"
  )
  if [[ "${workspace}" == "auto" ]]; then
    "${controlled_command[@]}"
  else
    CUBLAS_WORKSPACE_CONFIG="${workspace}" "${controlled_command[@]}"
  fi

  "${PYTHON_BIN}" - "${train_dir}" "${precision}" <<'PY'
import json
import sys
from pathlib import Path

train = Path(sys.argv[1])
precision = sys.argv[2]
manifest = json.loads((train / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8"))
expected = {
    "batch_size": 16,
    "eval_batch_size": 16,
    "source_train_rows": 94912,
    "query_train_rows": 124175,
    "catalog_steps": 2730,
    "query_steps": 38805,
    "max_source_length": 2048,
    "lr": 0.0003,
    "weight_decay": 0.0,
    "seed": 20260626,
    "load_best_model_at_end": True,
    "loaded_optimizer_state": False,
    "init_model_dir": None,
    "formulation": "catalog indexing followed by query->single SID finetuning",
}
mismatches = {
    key: {"expected": value, "actual": manifest.get(key)}
    for key, value in expected.items()
    if manifest.get(key) != value
}
if mismatches:
    raise SystemExit(f"training manifest mismatch: {mismatches}")
events = [
    json.loads(line)
    for line in (train / "loss_history.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
eval_steps = [
    int(row["step"])
    for row in events
    if row.get("stage") == "query" and row.get("event") == "eval"
]
expected_eval_steps = list(range(200, 38806, 200)) + [38805]
if eval_steps != expected_eval_steps:
    raise SystemExit(
        f"query eval cadence mismatch: expected_tail={expected_eval_steps[-3:]}, "
        f"actual_tail={eval_steps[-3:]}"
    )
if not (train / "final" / "structured_decoder.pt").is_file():
    raise SystemExit("final checkpoint is missing")
print(json.dumps({"manifest_verified": True, "precision": precision}, sort_keys=True))
PY

  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/skill_gr_pipeline.py" eval-single-sid-decoder \
    --rows "${SKILLGR_OFFICIAL_ROWS}" \
    --model-dir "${train_dir}/final" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${eval_dir}" \
    --split-name official_core75 \
    --max-source-length 2048 \
    --num-beams 50 \
    --rank-aggregation beam_order \
    --decoder-head ce \
    --score-transform logprob \
    --k-values 1,10,50 \
    --log-every 25

  local subset
  for subset in all single multi; do
    "${PYTHON_BIN}" "${CODE_ROOT}/scripts/skill_gr_pipeline.py" rescore-ranked \
      --predictions "${eval_dir}/predictions.jsonl" \
      --subset "${subset}" \
      --k-values 1,10,50 \
      --output "${eval_dir}/ranked_metrics_${subset}.json"
  done

  "${PYTHON_BIN}" - \
    "${label}" "${precision}" "${audit}" "${train_dir}" "${eval_dir}" \
    "${SKILLGR_ARCHIVED_OFFICIAL}" <<'PY'
import json
import sys
from pathlib import Path

label, precision = sys.argv[1:3]
audit_path, train, evaluation, archive = map(Path, sys.argv[3:7])
manifest = json.loads((train / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
predictions = [
    json.loads(line)
    for line in (evaluation / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(predictions) != 75:
    raise SystemExit(f"Official core75 prediction count mismatch: {len(predictions)}")
metrics = {
    subset: json.loads(
        (evaluation / f"ranked_metrics_{metric_subset}.json").read_text(encoding="utf-8")
    )["overall"]
    for subset, metric_subset in (("overall", "all"), ("single", "single"), ("multi", "multi"))
}
archived = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines() if line.strip()]
exact_rows = sum(
    left.get("ranked_skill_ids") == right.get("ranked_skill_ids")
    for left, right in zip(archived, predictions)
)
events = [
    json.loads(line)
    for line in (train / "loss_history.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
best = min(
    (row for row in events if row.get("stage") == "query" and row.get("event") == "eval"),
    key=lambda row: row["dev_loss"],
)
summary = {
    "arm": label,
    "controls": audit,
    "training": {
        "catalog_epochs": 3,
        "query_epochs": 5,
        "seed": 20260626,
        "precision": precision,
        "query_best_dev_loss": manifest["query_best_dev_loss"],
        "query_best_step": best["step"],
    },
    "evaluation": "Official core75, beam=50, beam_order, CE log-prob",
    "archived_top50_exact_rows": exact_rows,
    **metrics,
}
(train.parent / "result_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY
  touch "${arm_root}/arm.done"
}

declare -a PIDS=()
declare -A LABEL_BY_PID=()
terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait || true
}
trap 'terminate_children; cleanup; exit 130' INT TERM

launch_arm() {
  local gpu="$1" label="$2" precision="$3" bf16_reduction="$4"
  local foreach="$5" preferred_blas="$6" workspace="$7"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" run_arm \
      "${label}" "${precision}" "${bf16_reduction}" "${foreach}" \
      "${preferred_blas}" "${workspace}"
  ) &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}"
  echo "launched ${label} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_arm 0 default_bf16_r0 bf16 auto auto auto auto
launch_arm 1 default_bf16_r1 bf16 auto auto auto auto
launch_arm 2 bf16_reduction_off bf16 false auto auto auto
launch_arm 3 fp32 fp32 auto auto auto auto
launch_arm 4 cublaslt bf16 auto auto cublaslt auto
launch_arm 5 cublaslt_bf16_reduction_off bf16 false auto cublaslt auto
launch_arm 6 workspace_16 bf16 auto auto auto :16:8
launch_arm 7 foreach_false bf16 auto false auto auto

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
  if [[ -n "${finished_pid}" ]]; then unset 'LABEL_BY_PID[$finished_pid]'; fi
  remaining=$((remaining - 1))
done
trap - INT TERM
if (( failures > 0 )); then exit 1; fi

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = (
    "default_bf16_r0",
    "default_bf16_r1",
    "bf16_reduction_off",
    "fp32",
    "cublaslt",
    "cublaslt_bf16_reduction_off",
    "workspace_16",
    "foreach_false",
)
arms = [
    json.loads((root / "arms" / label / "result_summary.json").read_text(encoding="utf-8"))
    for label in labels
]
ranges = {
    subset: {
        f"recall@{cutoff}": {
            "min": min(float(arm[subset][f"@{cutoff}"]["recall"]) for arm in arms),
            "max": max(float(arm[subset][f"@{cutoff}"]["recall"]) for arm in arms),
        }
        for cutoff in (1, 10, 50)
    }
    for subset in ("overall", "single", "multi")
}
summary = {
    "artifact_format": "legacy_e2e_numerical_sensitivity_v1",
    "protocol": {
        "source_rows": 94912,
        "expanded_query_rows": 124175,
        "catalog_epochs": 3,
        "query_epochs": 5,
        "batch_size": 16,
        "lr": 0.0003,
        "weight_decay": 0.0,
        "seed": 20260626,
        "checkpoint": "dev_best",
        "evaluation": "Official core75, beam=50, beam_order, CE log-prob",
    },
    "arms": arms,
    "metric_ranges": ranges,
}
(root / "matrix_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"arms": len(arms), "metric_ranges": ranges}, sort_keys=True))
PY
touch "${OUTPUT_ROOT}/matrix.done"
