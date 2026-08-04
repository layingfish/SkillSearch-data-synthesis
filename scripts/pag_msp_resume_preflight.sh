#!/usr/bin/env bash
set -euo pipefail

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1

REPO=/workspace/SkillSearch-data-synthesis
EXPERIMENT=/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_pag_single_sid_20260804_v1
OUTPUT="${EXPERIMENT}/preflight_pag_msp_resume_0804b"
LOG="${EXPERIMENT}/logs/pag-msp-resume-preflight-0804b.log"
ROWS=/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/isolated_mode3_gpt4o_20260714/decoder_rows.base_plus_task5_gt2.plus_isolated_mode3.jsonl
SKILLS=/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/canonical_skills.jsonl
TEACHER="${EXPERIMENT}/teacher_panel/predictions.jsonl"
MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1

mkdir -p "${EXPERIMENT}/logs"
exec > >(tee "${LOG}") 2>&1

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite preflight output: ${OUTPUT}" >&2
  exit 3
fi
mkdir -p "${OUTPUT}/continuous" "${OUTPUT}/split" "${OUTPUT}/resumed"
cd "${REPO}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

common_args=(
  scripts/skill_gr_pipeline.py train-pag-sparse-encoder
  --rows "${ROWS}"
  --skills-jsonl "${SKILLS}"
  --hard-negatives "${TEACHER}"
  --model "${MODEL}"
  --max-length 128
  --top-k 64
  --max-steps 4
  --batch-size 32
  --lr 0.0005
  --weight-decay 0
  --warmup-ratio 0
  --seed 20260626
  --bf16
  --skip-set-id-build
)

torchrun --standalone --nproc_per_node=8 "${common_args[@]}" \
  --output-dir "${OUTPUT}/continuous"

torchrun --standalone --nproc_per_node=8 "${common_args[@]}" \
  --output-dir "${OUTPUT}/split" \
  --save-steps 2 \
  --stop-after-steps 2

torchrun --standalone --nproc_per_node=8 "${common_args[@]}" \
  --output-dir "${OUTPUT}/resumed" \
  --save-steps 2 \
  --resume-checkpoint "${OUTPUT}/split/checkpoint-2"

python3 - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
continuous = torch.load(root / "continuous" / "pag_sparse_encoder.pt", map_location="cpu", weights_only=False)
resumed = torch.load(root / "resumed" / "pag_sparse_encoder.pt", map_location="cpu", weights_only=False)
continuous_state = continuous["state_dict"]
resumed_state = resumed["state_dict"]
if continuous_state.keys() != resumed_state.keys():
    raise SystemExit("resume invariant failed: model state keys differ")
max_error = 0.0
for key in continuous_state:
    left = continuous_state[key]
    right = resumed_state[key]
    if left.shape != right.shape:
        raise SystemExit(f"resume invariant failed: shape mismatch for {key}")
    if left.numel():
        max_error = max(max_error, float((left.float() - right.float()).abs().max()))
tolerance = 1e-8
if max_error > tolerance:
    raise SystemExit(
        f"resume invariant failed: max_parameter_error={max_error} exceeds tolerance={tolerance}"
    )
result = {
    "world_size": 8,
    "per_device_batch": 32,
    "continuous_steps": 4,
    "split_step": 2,
    "resumed_steps": 4,
    "max_parameter_error": max_error,
    "allclose_atol": tolerance,
}
(root / "PASSED.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True), flush=True)
PY
