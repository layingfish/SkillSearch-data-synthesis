#!/usr/bin/env bash
set -euo pipefail

bash /mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh --with-dataset-rw || exit 1

STAGE="${1:-}"
if [[ "${STAGE}" != preflight && "${STAGE}" != full && "${STAGE}" != resume-rq && "${STAGE}" != resume-rq-preflight && "${STAGE}" != resume-decoder && "${STAGE}" != resume-eval ]]; then
  echo "usage: $0 {preflight|full|resume-rq|resume-rq-preflight|resume-decoder|resume-eval}" >&2
  exit 2
fi

CODE_ROOT=/workspace/SkillSearch-data-synthesis
cd "${CODE_ROOT}"
CODE_ROOT_PHYSICAL="$(pwd -P)"
git config --global --add safe.directory "${CODE_ROOT_PHYSICAL}"

EXPECTED_COMMIT="${SKILLGR_LAUNCH_COMMIT:-}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ -z "${EXPECTED_COMMIT}" || "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "launch commit mismatch: expected=${EXPECTED_COMMIT:-missing}, actual=${ACTUAL_COMMIT}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "refusing to launch from a dirty worktree" >&2
  exit 1
fi

if [[ "${STAGE}" == preflight ]]; then
  DEFAULT_OUTPUT=/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_preflight_v1
elif [[ "${STAGE}" == resume-rq-preflight ]]; then
  DEFAULT_OUTPUT=/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_recovery_preflight_v2
elif [[ "${STAGE}" == resume-rq ]]; then
  DEFAULT_OUTPUT=/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v2
elif [[ "${STAGE}" == resume-decoder ]]; then
  DEFAULT_OUTPUT=/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v3
elif [[ "${STAGE}" == resume-eval ]]; then
  DEFAULT_OUTPUT=/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v4
else
  DEFAULT_OUTPUT=/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v1
fi
OUTPUT_ROOT="${SKILLGR_QUERY_ALIGNED_OUTPUT_ROOT:-${DEFAULT_OUTPUT}}"
if [[ "${OUTPUT_ROOT}" != /models_rw/* ]]; then
  echo "output root must be under /models_rw: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/controls"
exec > >(tee "${OUTPUT_ROOT}/logs/node.log") 2>&1

PY310_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1
PY310_RUNTIME_ARCHIVE="${PY310_RUNTIME_ROOT}/runtime.tar.zst"
PY310_RUNTIME_ARCHIVE_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
PY311_SITE_ARCHIVE="${PY311_RUNTIME_ROOT}/site.tar.zst"
PY311_SITE_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
BASE_MODEL_SHA256=a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4

DATA_ROOT=/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713
TRAIN_ROWS="${DATA_ROOT}/isolated_mode3_gpt4o_20260714/decoder_rows.base_plus_task5_gt2.plus_isolated_mode3.jsonl"
DEV_ROWS="${DATA_ROOT}/canonical_alias_core75_learned_d4_inputs/rows_001.decoder_rows.base.jsonl"
OFFICIAL_ROWS="${DATA_ROOT}/canonical_alias_core75_learned_d4_inputs/rows_003.skillsbench_core75.rows.jsonl"
SKILLS="${DATA_ROOT}/canonical_alias_core75_learned_d4_inputs/canonical_skills.jsonl"
SEMANTIC_EMBEDDINGS="${DATA_ROOT}/canonical_alias_core75_learned_d4_inputs/canonical_skill_embeddings.pt"
MODEL_ASSET_ROOT=/models_rw/SkillSearch-data-synthesis/assets-models/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713
CF_ROOT="${MODEL_ASSET_ROOT}/lightgcn_cf_d32_l3_e500_seed42_core75_gt2"
CF_EMBEDDINGS="${CF_ROOT}/lightgcn_cf_embeddings.pt"
CF_SKILL_IDS="${CF_ROOT}/skill_ids.jsonl"
TEACHER_PANEL=/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_pag_single_sid_20260804_v1/teacher_panel/predictions.jsonl

check_sha256() {
  local path="$1" expected="$2" observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: expected=${expected}, observed=${observed}" >&2
    exit 1
  fi
}

check_sha256 "${PY310_RUNTIME_ARCHIVE}" "${PY310_RUNTIME_ARCHIVE_SHA256}"
check_sha256 "${PY311_SITE_ARCHIVE}" "${PY311_SITE_ARCHIVE_SHA256}"
check_sha256 "${BASE_MODEL}/model.safetensors" "${BASE_MODEL_SHA256}"
check_sha256 "${TRAIN_ROWS}" 55b6539c3379523f78c2326580fa690500f9b213d1d620b250ba333b65bcba08
check_sha256 "${DEV_ROWS}" a4d7b7c8223a79e1ac3157b2bc72121e64590383a52bb7bde1c540d51f2e7aed
check_sha256 "${OFFICIAL_ROWS}" 4adcd50e359e806d824ffa6f5ee29b4573321033e47ffa2a31780e65ff3cb8ba
check_sha256 "${SKILLS}" 1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3
check_sha256 "${SEMANTIC_EMBEDDINGS}" bb623a18cc0fc24d20b492db4730c484cfdd93422f49c72fead9b88fb3370f20
check_sha256 "${CF_EMBEDDINGS}" b4d6a16150f79c202314a8ba1f3917ae3106876ee9e8a69ff6f9d68edb928f84
check_sha256 "${CF_SKILL_IDS}" 26daa59096f84c50e0c6c07e58736f37f4d36a7f753fdec55d306a31d8276b50
check_sha256 "${TEACHER_PANEL}" 6ac2759c4fc08fd5be2e881f4ce90f20627d5fa78b160abb38f726153af64b63

TEMP_ROOT="$(mktemp -d /tmp/query-aligned-rq-pro.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/query-aligned-rq-pro.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT
mkdir -p "${TEMP_ROOT}/py310" "${TEMP_ROOT}/py311"
tar --zstd -xf "${PY310_RUNTIME_ARCHIVE}" -C "${TEMP_ROOT}/py310"
tar --zstd -xf "${PY311_SITE_ARCHIVE}" -C "${TEMP_ROOT}/py311"
PYTHON_BIN="${TEMP_ROOT}/py310/python/bin/python3"
PY310_SITE="${TEMP_ROOT}/py310/site"
PY311_SITE="${TEMP_ROOT}/py311/site"
NVIDIA_LIBRARY_PATHS="$(find "${PY311_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
if [[ -z "${NVIDIA_LIBRARY_PATHS}" ]]; then
  echo "fixed NVIDIA libraries are missing" >&2
  exit 1
fi
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

import scipy
import torch
import transformers

actual = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "tokenizers": metadata.version("tokenizers"),
    "scipy": scipy.__version__,
    "gpu_count": torch.cuda.device_count(),
}
expected = {
    "python": "3.10.12",
    "torch": "2.10.0+cu128",
    "cuda": "12.8",
    "cudnn": 91002,
    "transformers": "5.3.0",
    "tokenizers": "0.22.2",
    "gpu_count": 8,
}
for key, value in expected.items():
    if actual[key] != value:
        raise SystemExit(f"runtime mismatch for {key}: expected={value}, actual={actual[key]}")
print(json.dumps({"runtime": actual}, sort_keys=True))
PY

MODEL_SAFE="${OUTPUT_ROOT}/controls/t5-base-safetensors-only"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done

PIPELINE="${CODE_ROOT}/scripts/skill_gr_pipeline.py"
RQ_TRAINER="${CODE_ROOT}/scripts/train_query_aligned_rq.py"
"${PYTHON_BIN}" -m py_compile "${PIPELINE}" "${RQ_TRAINER}"
PYTHONPATH="${CODE_ROOT}/tests:${CODE_ROOT}/scripts:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON_BIN}" -m unittest \
  test_query_aligned_rq \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_pro_residual_geometry_matches_closed_form_and_legal_order \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_frozen_pro_query_vector_reproduces_m0_decoder_start_pooling \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_pro_fusion_zero_weight_preserves_beam_scores_and_order \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_model_config_provenance_allows_only_local_load_path_difference \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_model_config_provenance_verifies_recorded_full_hash \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_prefix_distribution_distillation_aggregates_teacher_mass_by_parent \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_strict_ripor_candidate_batch_uses_all_available_scored_negatives_up_to_max \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_ripor_teacher_candidate_audit_records_variable_strict_coverage \
  test_skill_gr_pipeline.SkillGRPipelineTest.test_sequence_contrastive_zero_weight_preserves_forward_exactly -v

run_torch() {
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 "$@"
}

ACTIVE_ROWS="${TRAIN_ROWS}"
ACTIVE_DEV_ROWS="${DEV_ROWS}"
M0_STEPS=-1
M0_EPOCHS=3
M0_BATCH=8
RQ_STEPS=-1
RQ_EPOCHS=3
RQ_KMEANS_ITERS=25
RQ_MAX_CANDIDATES=50
DECODER_CATALOG_EPOCHS=3
DECODER_EPOCHS=5
DECODER_STEPS=-1
DECODER_BATCH=4
DECODER_MAX_NEGATIVES=49
DECODER_SAVE_STEPS=400
EVAL_ROWS="${OFFICIAL_ROWS}"
EVAL_BEAMS=50
EVAL_K_VALUES=1,10,50

if [[ "${STAGE}" == preflight ]]; then
  ACTIVE_ROWS="${OUTPUT_ROOT}/controls/train64.jsonl"
  ACTIVE_DEV_ROWS="${OUTPUT_ROOT}/controls/dev8.jsonl"
  EVAL_ROWS="${OUTPUT_ROOT}/controls/core8.jsonl"
  "${PYTHON_BIN}" - "${TRAIN_ROWS}" "${DEV_ROWS}" "${OFFICIAL_ROWS}" \
    "${ACTIVE_ROWS}" "${ACTIVE_DEV_ROWS}" "${EVAL_ROWS}" <<'PY'
import sys
from pathlib import Path

train_source, dev_source, eval_source, train_target, dev_target, eval_target = map(Path, sys.argv[1:])
for source, target, count in (
    (train_source, train_target, 64),
    (dev_source, dev_target, 8),
    (eval_source, eval_target, 8),
):
    with source.open("r", encoding="utf-8") as handle:
        rows = [next(handle) for _ in range(count)]
    target.write_text("".join(rows), encoding="utf-8")
PY
  M0_STEPS=2
  M0_EPOCHS=1
  M0_BATCH=2
  RQ_STEPS=2
  RQ_EPOCHS=1
  RQ_KMEANS_ITERS=2
  RQ_MAX_CANDIDATES=10
  DECODER_CATALOG_EPOCHS=0
  DECODER_EPOCHS=1
  DECODER_STEPS=2
  DECODER_BATCH=1
  DECODER_MAX_NEGATIVES=5
  DECODER_SAVE_STEPS=0
  EVAL_BEAMS=10
  EVAL_K_VALUES=1,10
elif [[ "${STAGE}" == resume-rq-preflight ]]; then
  RQ_STEPS=2
  RQ_EPOCHS=1
  DECODER_CATALOG_EPOCHS=0
  DECODER_EPOCHS=1
  DECODER_STEPS=2
  DECODER_BATCH=1
  DECODER_MAX_NEGATIVES=5
  DECODER_SAVE_STEPS=0
fi

if [[ "${STAGE}" == resume-rq || "${STAGE}" == resume-rq-preflight || "${STAGE}" == resume-decoder || "${STAGE}" == resume-eval ]]; then
  M0_ROOT="${SKILLGR_QUERY_ALIGNED_M0_ROOT:-/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v1/m0}"
  if [[ "${M0_ROOT}" != /models_rw/* ]]; then
    echo "resume M0 root must be under /models_rw: ${M0_ROOT}" >&2
    exit 1
  fi
  check_sha256 "${M0_ROOT}/relevance_encoder.pt" c08d6ce0d3f7ca26b20938e37ff72aac828f2541ecfc50a047b709696ca0d519
  check_sha256 "${M0_ROOT}/skill_embeddings.pt" 283641ad3069333997c33e5407daf535d246b800152d3d30b49afa50f495b39a
  check_sha256 "${M0_ROOT}/query_embeddings.pt" cf0c84d18c524ff6782d6e28dece32088ef7b824317cb64815b1297334225fad
  echo "reusing verified M0 root: ${M0_ROOT}"
else
  M0_ROOT="${OUTPUT_ROOT}/m0"
  run_torch "${PIPELINE}" train-ripor-relevance-encoder \
    --rows "${ACTIVE_ROWS}" \
    --skills-jsonl "${SKILLS}" \
    --hard-negatives "${TEACHER_PANEL}" \
    --output-dir "${M0_ROOT}" \
    --model "${MODEL_SAFE}" \
    --max-query-length 512 \
    --max-skill-length 512 \
    --decoder-length 1 \
    --batch-size "${M0_BATCH}" \
    --epochs "${M0_EPOCHS}" \
    --max-steps "${M0_STEPS}" \
    --lr 3e-5 \
    --weight-decay 0.01 \
    --max-negatives 1 \
    --logging-steps 25 \
    --seed 20260814 \
    --bf16
fi

for required in relevance_encoder.pt skill_embeddings.pt query_embeddings.pt; do
  if [[ ! -f "${M0_ROOT}/${required}" ]]; then
    echo "M0 output is missing: ${M0_ROOT}/${required}" >&2
    exit 1
  fi
done

if [[ "${STAGE}" == resume-decoder || "${STAGE}" == resume-eval ]]; then
  RQ_ROOT="${SKILLGR_QUERY_ALIGNED_RQ_ROOT:-/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v2/rq}"
  if [[ "${RQ_ROOT}" != /models_rw/* ]]; then
    echo "resume RQ root must be under /models_rw: ${RQ_ROOT}" >&2
    exit 1
  fi
  check_sha256 "${RQ_ROOT}/rqvae.pt" 14b23af2101a1c4aa38416d6d984a1f8f3ab683ffceb2cab3c9b4cdac3ad875c
  check_sha256 "${RQ_ROOT}/codebook_embeddings.pt" d6eb64f9d888f40db2334f53906fd5b19dc035cc82b4af4d935fe2cd895fa0a3
  check_sha256 "${RQ_ROOT}/semantic_ids.jsonl" 8c0e95b30940669b3e74271e6a6d667036568a0b998a5069be49cab4ee96a451
  check_sha256 "${RQ_ROOT}/rqvae_manifest.json" eda8b99dbeeb9e35e94c4aa49d9bedef240c35579735ca81b3bf6ae54d0e0828
  echo "reusing verified RQ root: ${RQ_ROOT}"
else
  RQ_ROOT="${OUTPUT_ROOT}/rq"
  run_torch "${RQ_TRAINER}" \
    --relevance-embeddings "${M0_ROOT}/skill_embeddings.pt" \
    --semantic-embeddings "${SEMANTIC_EMBEDDINGS}" \
    --cf-embeddings "${CF_EMBEDDINGS}" \
    --cf-skill-ids "${CF_SKILL_IDS}" \
    --query-embeddings "${M0_ROOT}/query_embeddings.pt" \
    --rows "${ACTIVE_ROWS}" \
    --hard-negatives "${TEACHER_PANEL}" \
    --output-dir "${RQ_ROOT}" \
    --depth 4 \
    --codebook-size 256 \
    --kmeans-iters "${RQ_KMEANS_ITERS}" \
    --batch-size 512 \
    --query-batch-size 8 \
    --max-candidates "${RQ_MAX_CANDIDATES}" \
    --epochs "${RQ_EPOCHS}" \
    --max-steps "${RQ_STEPS}" \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --semantic-weight 0.01 \
    --cf-weight 0.01 \
    --prefix-weight 0.1 \
    --teacher-temperature 0.05 \
    --student-temperature 1.0 \
    --logging-steps 25 \
    --seed 20260814
fi

for required in codebook_embeddings.pt semantic_ids.jsonl rqvae_manifest.json; do
  if [[ ! -f "${RQ_ROOT}/${required}" ]]; then
    echo "RQ output is missing: ${RQ_ROOT}/${required}" >&2
    exit 1
  fi
done

if [[ "${STAGE}" == resume-eval ]]; then
  DECODER_ROOT="${SKILLGR_QUERY_ALIGNED_DECODER_ROOT:-/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v3/decoder}"
  if [[ "${DECODER_ROOT}" != /models_rw/* ]]; then
    echo "resume decoder root must be under /models_rw: ${DECODER_ROOT}" >&2
    exit 1
  fi
  check_sha256 "${DECODER_ROOT}/single_sid_decoder_train_manifest.json" a50b0c609dce1e199aece92e23003929720f2ed83538ea3e83c6094518728917
  check_sha256 "${DECODER_ROOT}/final/structured_decoder.pt" 3fb015dcdb46f02bdf100f9ecb0b3e834bbb5c79b9e71205b314ec032066fb67
  for step in $(seq 400 400 20000); do
    if [[ ! -s "${DECODER_ROOT}/checkpoint-${step}/structured_decoder.pt" ]]; then
      echo "decoder checkpoint is missing or empty: checkpoint-${step}" >&2
      exit 1
    fi
  done
  mapfile -t PERIODIC_CHECKPOINTS < <(
    find "${DECODER_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print
  )
  if [[ "${#PERIODIC_CHECKPOINTS[@]}" -ne 50 ]]; then
    echo "expected exactly 50 periodic decoder checkpoints, found ${#PERIODIC_CHECKPOINTS[@]}" >&2
    exit 1
  fi
  echo "reusing verified decoder root: ${DECODER_ROOT}"
else
  DECODER_ROOT="${OUTPUT_ROOT}/decoder"
  run_torch "${PIPELINE}" train-single-sid-decoder \
    --rows "${ACTIVE_ROWS}" \
    --dev-rows "${ACTIVE_DEV_ROWS}" \
    --semantic-ids "${RQ_ROOT}/semantic_ids.jsonl" \
    --skills-jsonl "${SKILLS}" \
    --output-dir "${DECODER_ROOT}" \
    --model "${MODEL_SAFE}" \
    --query-loss expanded_ce \
    --catalog-epochs "${DECODER_CATALOG_EPOCHS}" \
    --catalog-max-steps -1 \
    --max-source-length 2048 \
    --batch-size "${DECODER_BATCH}" \
    --eval-batch-size "${DECODER_BATCH}" \
    --epochs "${DECODER_EPOCHS}" \
    --max-steps "${DECODER_STEPS}" \
    --lr 3e-4 \
    --warmup-ratio 0 \
    --lr-scheduler linear \
    --weight-decay 0 \
    --logging-steps 25 \
    --eval-steps 0 \
    --save-steps "${DECODER_SAVE_STEPS}" \
    --lightweight-checkpoints \
    --no-padding-distributed-sampler \
    --seed 20260814 \
    --ripor-relevance-checkpoint "${M0_ROOT}/relevance_encoder.pt" \
    --ripor-codebook-embeddings "${RQ_ROOT}/codebook_embeddings.pt" \
    --ripor-loss-weight 0 \
    --sequential-ce-weight 1 \
    --ripor-max-negatives "${DECODER_MAX_NEGATIVES}" \
    --ripor-negative-sampling top \
    --ripor-hard-negatives "${TEACHER_PANEL}" \
    --prefix-distill-loss-weight 1 \
    --prefix-distill-teacher-temperature 0.05 \
    --prefix-distill-student-temperature 1 \
    --no-load-best-model-at-end \
    --bf16
fi

if [[ ! -f "${DECODER_ROOT}/final/structured_decoder.pt" ]]; then
  echo "decoder final checkpoint is missing" >&2
  exit 1
fi

if [[ "${STAGE}" == resume-rq-preflight ]]; then
  touch "${OUTPUT_ROOT}/PASSED"
  exit 0
fi

if [[ "${STAGE}" == preflight ]]; then
  EVAL_ROOT="${OUTPUT_ROOT}/eval/final_omega10"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${PIPELINE}" eval-single-sid-decoder \
    --rows "${EVAL_ROWS}" \
    --model-dir "${DECODER_ROOT}/final" \
    --semantic-ids "${RQ_ROOT}/semantic_ids.jsonl" \
    --output-dir "${EVAL_ROOT}" \
    --num-beams "${EVAL_BEAMS}" \
    --k-values "${EVAL_K_VALUES}" \
    --decoder-head ce \
    --score-transform logprob \
    --pro-fusion-weight 10
  touch "${OUTPUT_ROOT}/PASSED"
  exit 0
fi

EVAL_ROOT="${OUTPUT_ROOT}/eval"
mkdir -p "${EVAL_ROOT}"
mapfile -t CHECKPOINTS < <(
  find "${DECODER_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    \( -name 'checkpoint-*' -o -name final \) -print | sort
)
if [[ "${#CHECKPOINTS[@]}" -lt 2 ]]; then
  echo "checkpoint sweep requires periodic checkpoints plus final" >&2
  exit 1
fi
FUSION_WEIGHTS=(0 2 5 10 20)

eval_worker() {
  local gpu="$1" task_index=0 checkpoint weight label eval_dir
  for checkpoint in "${CHECKPOINTS[@]}"; do
    for weight in "${FUSION_WEIGHTS[@]}"; do
      if (( task_index % 8 != gpu )); then
        task_index=$((task_index + 1))
        continue
      fi
      label="$(basename "${checkpoint}")_omega${weight}"
      eval_dir="${EVAL_ROOT}/${label}"
      CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=2 "${PYTHON_BIN}" "${PIPELINE}" \
        eval-single-sid-decoder \
        --rows "${OFFICIAL_ROWS}" \
        --model-dir "${checkpoint}" \
        --semantic-ids "${RQ_ROOT}/semantic_ids.jsonl" \
        --output-dir "${eval_dir}" \
        --num-beams 50 \
        --k-values 1,10,50 \
        --rank-aggregation beam_order \
        --decoder-head ce \
        --score-transform logprob \
        --pro-fusion-weight "${weight}" \
        --log-every 75
      "${PYTHON_BIN}" "${PIPELINE}" rescore-ranked \
        --predictions "${eval_dir}/predictions.jsonl" \
        --subset single \
        --k-values 1,10,50 \
        --output "${eval_dir}/ranked_metrics_single.json"
      task_index=$((task_index + 1))
    done
  done
}

worker_pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  eval_worker "${gpu}" > "${OUTPUT_ROOT}/logs/eval_gpu${gpu}.log" 2>&1 &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do
  wait "${pid}"
done

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${DECODER_ROOT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
decoder_root = Path(sys.argv[2])
records = []
for metrics_path in sorted((root / "eval").glob("*/ranked_metrics_single.json")):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["overall"]
    eval_dir = metrics_path.parent
    manifest = json.loads((eval_dir / "eval_manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["model_dir"]).resolve(strict=True)
    records.append(
        {
            "label": eval_dir.name,
            "checkpoint": str(checkpoint),
            "fusion_weight": manifest["pro_fusion_weight"],
            "r1": metrics["@1"]["recall"],
            "r10": metrics["@10"]["recall"],
            "r50": metrics["@50"]["recall"],
        }
    )
fusion_weights = [0, 2, 5, 10, 20]
checkpoint_labels = [path.name for path in decoder_root.glob("checkpoint-*")] + ["final"]
expected_labels = {
    f"{checkpoint_label}_omega{weight}"
    for checkpoint_label in checkpoint_labels
    for weight in fusion_weights
}
observed_labels = {record["label"] for record in records}
if len(records) != len(expected_labels) or observed_labels != expected_labels:
    missing = sorted(expected_labels - observed_labels)
    extra = sorted(observed_labels - expected_labels)
    raise SystemExit(
        "evaluation matrix incomplete: "
        f"expected={len(expected_labels)}, observed={len(records)}, "
        f"missing={missing[:5]}, extra={extra[:5]}"
    )
best = max(records, key=lambda row: (row["r1"], row["r10"], row["r50"], -float(row["fusion_weight"])))
checkpoint_file = Path(best["checkpoint"]) / "structured_decoder.pt"
hasher = hashlib.sha256()
with checkpoint_file.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
digest = hasher.hexdigest()
selection = {
    "protocol": {
        "rows": 75,
        "single_rows": 24,
        "beam": 50,
        "rank_aggregation": "beam_order",
        "decoder_head": "ce",
        "score_transform": "logprob",
        "checkpoint_selection": "Official core75 single R@1, then R@10, then R@50",
        "fusion_weights": fusion_weights,
    },
    "best": {**best, "checkpoint_sha256": digest},
    "records": records,
}
selected_root = root / "selected"
selected_root.mkdir(parents=True, exist_ok=True)
(selected_root / "selection.json").write_text(
    json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
link = selected_root / "single_top1_best"
os.symlink(best["checkpoint"], link, target_is_directory=True)
print(json.dumps(selection["best"], sort_keys=True))
PY

touch "${OUTPUT_ROOT}/COMPLETED"
