#!/usr/bin/env bash
set -euo pipefail

bash /mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh || exit 1

CODE_ROOT=/workspace/SkillSearch-data-synthesis
cd "${CODE_ROOT}"

OUTPUT_ROOT="${SKILLGR_SELF_MINED_OUTPUT_ROOT:-/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_self_mined_nearmiss_20260813_v1}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite GR-self-mined output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/controls" "${OUTPUT_ROOT}/mining" "${OUTPUT_ROOT}/eval"
exec > >(tee "${OUTPUT_ROOT}/logs/node.log") 2>&1
echo "launch_commit=${SKILLGR_LAUNCH_COMMIT:-unknown}"

PY310_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1
PY310_RUNTIME_ARCHIVE="${PY310_RUNTIME_ROOT}/runtime.tar.zst"
PY310_RUNTIME_ARCHIVE_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
PY311_SITE_ARCHIVE="${PY311_RUNTIME_ROOT}/site.tar.zst"
PY311_SITE_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
BASE_MODEL_SHA256=a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4

PIPELINE="${CODE_ROOT}/scripts/skill_gr_pipeline.py"
PIPELINE_SHA256=c37d7b809c1b689c0ae268d87aaa65fb001b54885e6c18ed171390a71a7d650e
MINER="${CODE_ROOT}/scripts/prepare_gr_self_mined_hard_negatives.py"
MINER_SHA256=05d1698619bea860fd344929a398cddd482e0e310332da32cdf71b742ce9a924
ANALYZER="${CODE_ROOT}/scripts/analyze_sequence_contrastive_checkpoint_sweep.py"
ANALYZER_SHA256=a528145faa614125e1118fb73ce819744e67284d081d1db4676a31bf12ca6d7b

DATASET_RW_ROOT=/mnt/cfs_algo_bj/workspace/chenyufei03/datasets
SINGLE_ROWS="${DATASET_RW_ROOT}/SkillSearch-data-synthesis/assets-data/outputs/train/skillrouter_repro_task5_mode3_single82621_20260716/inputs/decoder_rows.single_only.jsonl"
SINGLE_ROWS_SHA256=df7046845745d144592a9d91469be5fa9447b8a1c61df8724faf5a20f1129cb0
BASELINE_MODEL_DIR=/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v5/selected/joint_best
BASELINE_CHECKPOINT_SHA256=a2a26ab5f64fbbe5b0df875afccad945684a2b5e2166f71b94a3e299c36f03c1
MINING_MODEL_DIR=/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_contrastive_hn10_20260813_v2/selected/single_top1_best
MINING_CHECKPOINT_SHA256=9d40adfd3125f4ec60c8c56940b4f0018fb4041ae3e07f6736d826cdd4c1b7e8

check_sha256() {
  local path="$1" expected="$2" observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || {
    echo "SHA-256 mismatch for ${path}: expected=${expected}, observed=${observed}" >&2
    exit 1
  }
}

TEMP_ROOT="$(mktemp -d /tmp/skillgr-self-mined.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-self-mined.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

check_sha256 "${PY310_RUNTIME_ARCHIVE}" "${PY310_RUNTIME_ARCHIVE_SHA256}"
check_sha256 "${PY311_SITE_ARCHIVE}" "${PY311_SITE_ARCHIVE_SHA256}"
mkdir -p "${TEMP_ROOT}/py310" "${TEMP_ROOT}/py311"
tar --zstd -xf "${PY310_RUNTIME_ARCHIVE}" -C "${TEMP_ROOT}/py310"
tar --zstd -xf "${PY311_SITE_ARCHIVE}" -C "${TEMP_ROOT}/py311"
PYTHON_BIN="${TEMP_ROOT}/py310/python/bin/python3"
PY310_SITE="${TEMP_ROOT}/py310/site"
PY311_SITE="${TEMP_ROOT}/py311/site"
NVIDIA_LIBRARY_PATHS="$(find "${PY311_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
[[ -n "${NVIDIA_LIBRARY_PATHS}" ]] || { echo "fixed NVIDIA libraries are missing" >&2; exit 1; }
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/scripts:${PY310_SITE}:${PY311_SITE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4

"${PYTHON_BIN}" - <<'PY'
import json
import platform
from importlib import metadata

import torch
import transformers

expected = {
    "python": "3.10.12",
    "torch": "2.10.0+cu128",
    "cuda": "12.8",
    "cudnn": 91002,
    "transformers": "5.3.0",
    "tokenizers": "0.22.2",
    "gpu_count": 8,
}
actual = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "tokenizers": metadata.version("tokenizers"),
    "gpu_count": torch.cuda.device_count(),
}
if actual != expected:
    raise SystemExit(f"GR-self-mined runtime mismatch: expected={expected}, actual={actual}")
print(json.dumps({"runtime": actual, "torch_config": torch.__config__.show()}, sort_keys=True))
PY

MODEL_SAFE="${OUTPUT_ROOT}/controls/t5-base-safetensors-only"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done

set -a
export SKILLGR_EXACT_MODEL="${MODEL_SAFE}"
source "${CODE_ROOT}/configs/legacy_catalog_base_exact_runtime_20260808.env"
set +a
SKILLGR_OFFICIAL_ROWS="${DATASET_RW_ROOT}/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/rows_003.skillsbench_core75.rows.jsonl"
SKILLGR_SKILLS="${DATASET_RW_ROOT}/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/canonical_skills.jsonl"

for required in \
  "${SINGLE_ROWS}" "${SKILLGR_OFFICIAL_ROWS}" "${SKILLGR_SKILLS}" "${SKILLGR_SEMANTIC_IDS}" \
  "${BASELINE_MODEL_DIR}/structured_decoder.pt" "${MINING_MODEL_DIR}/structured_decoder.pt"; do
  [[ -f "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done
check_sha256 "${BASE_MODEL}/model.safetensors" "${BASE_MODEL_SHA256}"
check_sha256 "${PIPELINE}" "${PIPELINE_SHA256}"
check_sha256 "${MINER}" "${MINER_SHA256}"
check_sha256 "${ANALYZER}" "${ANALYZER_SHA256}"
check_sha256 "${SINGLE_ROWS}" "${SINGLE_ROWS_SHA256}"
check_sha256 "${SKILLGR_OFFICIAL_ROWS}" "${SKILLGR_OFFICIAL_ROWS_SHA256}"
check_sha256 "${SKILLGR_SKILLS}" "${SKILLGR_SKILLS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"
check_sha256 "${BASELINE_MODEL_DIR}/structured_decoder.pt" "${BASELINE_CHECKPOINT_SHA256}"
check_sha256 "${MINING_MODEL_DIR}/structured_decoder.pt" "${MINING_CHECKPOINT_SHA256}"

"${PYTHON_BIN}" -m py_compile "${PIPELINE}" "${MINER}" "${ANALYZER}"
PYTHONPATH="${CODE_ROOT}/tests:${CODE_ROOT}/scripts:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON_BIN}" -m unittest \
  test_prepare_gr_self_mined_hard_negatives \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_sequence_candidate_scores_match_stepwise_sid_and_stop_logprob \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_sequence_contrastive_zero_weight_preserves_forward_exactly \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_sequence_listwise_loss_is_candidate_cross_entropy_and_honors_zero_weight \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_sequence_contrastive_candidate_manifest_closes_and_builds_exact_paths \
  test_analyze_sequence_contrastive_checkpoint_sweep -v

SHARD_ROOT="${OUTPUT_ROOT}/mining/shards"
"${PYTHON_BIN}" "${MINER}" shard \
  --rows "${SINGLE_ROWS}" \
  --output-dir "${SHARD_ROOT}" \
  --num-shards 8

"${PYTHON_BIN}" - "${SHARD_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest["row_count"] != 82621 or manifest["num_shards"] != 8:
    raise SystemExit(f"mining shard contract drifted: {manifest}")
counts = [row["rows"] for row in manifest["shards"]]
if sum(counts) != 82621 or max(counts) - min(counts) > 1:
    raise SystemExit(f"mining shard balance drifted: {counts}")
print(json.dumps({"mining_shards_verified": True, "counts": counts}, sort_keys=True))
PY

mine_shard() {
  local gpu="$1" shard_label shard_rows shard_output
  shard_label="$(printf 'shard_%02d' "${gpu}")"
  shard_rows="${SHARD_ROOT}/${shard_label}.jsonl"
  shard_output="${OUTPUT_ROOT}/mining/predictions/${shard_label}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${PIPELINE}" eval-single-sid-decoder \
    --rows "${shard_rows}" \
    --model-dir "${MINING_MODEL_DIR}" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${shard_output}" \
    --split-name "train_self_mining_${shard_label}" \
    --max-source-length 2048 \
    --num-beams 50 \
    --rank-aggregation beam_order \
    --decoder-head ce \
    --score-transform logprob \
    --k-values 1,10,50 \
    --log-every 1000
}

unset CUBLAS_WORKSPACE_CONFIG || true
declare -a MINING_PIDS=()
for gpu in {0..7}; do
  (mine_shard "${gpu}") > >(tee "${OUTPUT_ROOT}/logs/mining_gpu${gpu}.log") 2>&1 &
  MINING_PIDS+=("$!")
done
mining_failures=0
for pid in "${MINING_PIDS[@]}"; do
  if ! wait "${pid}"; then
    mining_failures=$((mining_failures + 1))
  fi
done
if (( mining_failures > 0 )); then
  echo "GR self-mining failed on ${mining_failures} workers" >&2
  exit 1
fi

declare -a PREDICTION_ARGS=()
for gpu in {0..7}; do
  shard_label="$(printf 'shard_%02d' "${gpu}")"
  PREDICTION_ARGS+=(--predictions "${OUTPUT_ROOT}/mining/predictions/${shard_label}/predictions.jsonl")
done
CANDIDATE_ROOT="${OUTPUT_ROOT}/mining/candidates"
"${PYTHON_BIN}" "${MINER}" compact \
  --rows "${SINGLE_ROWS}" \
  "${PREDICTION_ARGS[@]}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --model-dir "${MINING_MODEL_DIR}" \
  --output-dir "${CANDIDATE_ROOT}" \
  --expected-negatives 10 \
  --expected-beams 50
CANDIDATES="${CANDIDATE_ROOT}/candidates.jsonl"
CANDIDATE_MANIFEST="${CANDIDATE_ROOT}/manifest.json"

"${PYTHON_BIN}" - "${CANDIDATE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
audit = manifest["audit"]
if audit["rows"] != 82621 or audit["training_rows"] != 82621:
    raise SystemExit(f"self-mined row closure drifted: {audit}")
if audit["negatives_per_row"] != 10 or not audit["known_unique_non_gold_negatives"]:
    raise SystemExit(f"self-mined candidate contract drifted: {audit}")
if manifest["mining_checkpoint_sha256"] != "9d40adfd3125f4ec60c8c56940b4f0018fb4041ae3e07f6736d826cdd4c1b7e8":
    raise SystemExit("self-mined checkpoint provenance drifted")
print(json.dumps({"self_mined_candidates_verified": True, "audit": audit}, sort_keys=True))
PY

TRAIN_ROOT="${OUTPUT_ROOT}/train"
export CUBLAS_WORKSPACE_CONFIG=:16:8
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 "${PIPELINE}" \
  train-single-sid-decoder \
  --rows "${SINGLE_ROWS}" \
  --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
  --skills-jsonl "${SKILLGR_SKILLS}" \
  --output-dir "${TRAIN_ROOT}" \
  --model "${MODEL_SAFE}" \
  --query-loss expanded_ce \
  --catalog-epochs 0 \
  --catalog-max-steps 0 \
  --max-source-length 2048 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --epochs 1 \
  --max-steps -1 \
  --lr 1e-5 \
  --warmup-ratio 0 \
  --lr-scheduler constant \
  --weight-decay 0 \
  --dev-ratio 0 \
  --logging-steps 50 \
  --eval-steps 0 \
  --save-steps 200 \
  --lightweight-checkpoints \
  --no-padding-distributed-sampler \
  --seed 20260626 \
  --init-model-dir "${MINING_MODEL_DIR}" \
  --reset-optimizer \
  --no-load-best-model-at-end \
  --bf16 \
  --sequence-contrastive-loss-weight 1.0 \
  --sequence-contrastive-temperature 0.5 \
  --sequence-contrastive-max-negatives 10 \
  --sequence-contrastive-hard-negatives "${CANDIDATES}"
unset CUBLAS_WORKSPACE_CONFIG

"${PYTHON_BIN}" - "${TRAIN_ROOT}" "${CANDIDATES}" <<'PY'
import json
import sys
from pathlib import Path

from skill_gr_pipeline import file_sha256

root = Path(sys.argv[1])
candidates = Path(sys.argv[2])
manifest = json.loads((root / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8"))
if manifest["source_train_rows"] != 82621 or manifest["query_train_rows"] != 82621:
    raise SystemExit(f"training row count drifted: {manifest['source_train_rows']}/{manifest['query_train_rows']}")
if manifest["query_steps"] != 5164:
    raise SystemExit(f"optimizer step count drifted: {manifest['query_steps']}")
if manifest["global_batch_size"] != 16 or manifest["ddp_world_size"] != 8:
    raise SystemExit("distributed batch contract drifted")
sequence = manifest["sequence_contrastive"]
if sequence["candidate_manifest_sha256"] != file_sha256(candidates):
    raise SystemExit("training candidate provenance drifted")
if sequence["temperature"] != 0.5 or sequence["loss_weight"] != 1.0:
    raise SystemExit(f"sequence objective drifted: {sequence}")
periodic = list(root.glob("checkpoint-*"))
if len(periodic) != 25:
    raise SystemExit(f"expected 25 periodic checkpoints, got {len(periodic)}")
if not (root / "final" / "structured_decoder.pt").is_file():
    raise SystemExit("final checkpoint is missing")
print(json.dumps({"training_verified": True, "query_steps": 5164, "periodic_checkpoints": 25}))
PY

write_eval_provenance() {
  local label="$1" step="$2" checkpoint="$3" eval_dir="$4"
  "${PYTHON_BIN}" - "${label}" "${step}" "${checkpoint}" "${eval_dir}" <<'PY'
import json
import sys
from pathlib import Path

label, step, checkpoint, eval_dir = sys.argv[1:]
path = Path(eval_dir) / "checkpoint_provenance.json"
path.write_text(
    json.dumps(
        {"label": label, "step": int(step), "checkpoint": str(Path(checkpoint).resolve(strict=True))},
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

rescore_eval() {
  local eval_dir="$1" subset
  for subset in all single multi; do
    "${PYTHON_BIN}" "${PIPELINE}" rescore-ranked \
      --predictions "${eval_dir}/predictions.jsonl" \
      --subset "${subset}" \
      --k-values 1,10,50 \
      --output "${eval_dir}/ranked_metrics_${subset}.json"
  done
}

eval_checkpoint() {
  local gpu="$1" label="$2" step="$3" checkpoint="$4"
  local eval_dir="${OUTPUT_ROOT}/eval/${label}"
  [[ ! -e "${eval_dir}" ]] || { echo "refusing to overwrite eval output: ${eval_dir}" >&2; return 1; }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${PIPELINE}" eval-single-sid-decoder \
    --rows "${SKILLGR_OFFICIAL_ROWS}" \
    --model-dir "${checkpoint}" \
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
  rescore_eval "${eval_dir}"
  write_eval_provenance "${label}" "${step}" "${checkpoint}" "${eval_dir}"
}

declare -a CHECKPOINTS=("${BASELINE_MODEL_DIR}" "${MINING_MODEL_DIR}")
declare -a LABELS=(init prior_best)
declare -a STEPS=(0 0)
for checkpoint in "${TRAIN_ROOT}"/checkpoint-*; do
  step="${checkpoint##*-}"
  CHECKPOINTS+=("${checkpoint}")
  LABELS+=("$(printf 'step_%08d' "${step}")")
  STEPS+=("${step}")
done
CHECKPOINTS+=("${TRAIN_ROOT}/final")
LABELS+=("step_00005164")
STEPS+=("5164")
if (( ${#CHECKPOINTS[@]} != 28 )); then
  echo "expected 28 evaluations, got ${#CHECKPOINTS[@]}" >&2
  exit 1
fi

eval_worker() {
  local gpu="$1" index
  for ((index=gpu; index<${#CHECKPOINTS[@]}; index+=8)); do
    eval_checkpoint \
      "${gpu}" \
      "${LABELS[index]}" \
      "${STEPS[index]}" \
      "${CHECKPOINTS[index]}"
  done
}

declare -a EVAL_PIDS=()
for gpu in {0..7}; do
  (eval_worker "${gpu}") > >(tee "${OUTPUT_ROOT}/logs/eval_gpu${gpu}.log") 2>&1 &
  EVAL_PIDS+=("$!")
done
eval_failures=0
for pid in "${EVAL_PIDS[@]}"; do
  if ! wait "${pid}"; then
    eval_failures=$((eval_failures + 1))
  fi
done
if (( eval_failures > 0 )); then
  echo "checkpoint evaluation failed on ${eval_failures} workers" >&2
  exit 1
fi

eval_count="$(find "${OUTPUT_ROOT}/eval" -mindepth 1 -maxdepth 1 -type d | wc -l)"
[[ "${eval_count}" == 28 ]] || { echo "expected 28 eval directories, got ${eval_count}" >&2; exit 1; }

"${PYTHON_BIN}" "${ANALYZER}" \
  --eval-root "${OUTPUT_ROOT}/eval" \
  --output "${OUTPUT_ROOT}/sequence_contrastive_checkpoint_sweep_summary.json" \
  --selected-root "${OUTPUT_ROOT}/selected"

touch "${OUTPUT_ROOT}/experiment.done"
echo "GR-self-mined experiment completed: ${OUTPUT_ROOT}"
