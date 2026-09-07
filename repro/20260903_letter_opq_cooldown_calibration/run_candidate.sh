#!/usr/bin/env bash
set -euo pipefail

cfs_agent_root="${LETTER_OPQ_CFS_AGENT_ROOT:-/mnt/cfs-agent}"
project_root="${RIPOR_PROJECT_ROOT:-$cfs_agent_root/workspace/chenyufei03/SkillSearch-data-synthesis}"
variant="${LETTER_OPQ_VARIANT:?LETTER_OPQ_VARIANT is required}"
head_type="${LETTER_OPQ_HEAD_TYPE:?LETTER_OPQ_HEAD_TYPE is required}"
catalog_loss_weight="${LETTER_OPQ_CATALOG_LOSS_WEIGHT:?LETTER_OPQ_CATALOG_LOSS_WEIGHT is required}"
position_kl_weight="${LETTER_OPQ_POSITION_KL_WEIGHT:?LETTER_OPQ_POSITION_KL_WEIGHT is required}"
experiment="${LETTER_OPQ_VARIANT_EXPERIMENT:?LETTER_OPQ_VARIANT_EXPERIMENT is required}"
world_size="${LETTER_OPQ_WORLD_SIZE:-4}"
per_device_batch_size="${LETTER_OPQ_PER_DEVICE_BATCH_SIZE:-4}"

case "$head_type" in
  rpg_cosine_residual|rpg_cosine_residual_calibrated) ;;
  *) echo "invalid continuation head: $head_type" >&2; exit 2 ;;
esac
if [[ $((world_size * per_device_batch_size)) -ne 16 ]]; then
  echo "global batch must be 16: world=$world_size per_device=$per_device_batch_size" >&2
  exit 2
fi

cd "$project_root"
export PYTHONPATH="$project_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

data_root="${LETTER_OPQ_DATA_ROOT:-$cfs_agent_root/dataset/chenyufei03/workspace/chenyufei03/datasets/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs}"
model_root="${LETTER_OPQ_MODEL_ROOT:-$cfs_agent_root/dataset/chenyufei03/models/experiments/chenyufei03/SkillSearch-data-synthesis}"
docid_root="${LETTER_OPQ_CF20_ROOT:-$data_root/letter_opq_m64_semantic_cf20_c256_seed42_20260903}"
baseline_root="${LETTER_OPQ_BASELINE_ROOT:-$model_root/letter_opq_ceiling_runs/rudder_r2_opq_ceiling_m64_20260903_v1}"
train_rows="$data_root/rows_000.decoder_rows.base_plus_task5_gt2.jsonl"
dev_rows="$data_root/rows_001.decoder_rows.base.jsonl"
skills="$data_root/canonical_skills.jsonl"
base_model="$model_root/hf-models/t5-base-a9723ea"
semantic_ids="$docid_root/semantic_ids.jsonl"
docid_manifest="$docid_root/opq_docid_manifest.json"
initial_checkpoint="$baseline_root/candidates/semantic_cf20_rpg_e10/decoder/final/parallel_decoder.pt"
decoder="$experiment/decoder"
dev_eval="$experiment/dev_exact"
ddp_launcher="$project_root/repro/20260903_letter_opq_ceiling/run_ddp_command.sh"
eval_runner="$project_root/repro/20260903_letter_opq_ceiling/run_exact_eval.sh"

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: path=$path expected=$expected actual=$actual" >&2
    exit 1
  }
}

for required in \
  "$train_rows" "$dev_rows" "$skills" "$semantic_ids" "$docid_manifest" \
  "$initial_checkpoint" "$ddp_launcher" "$eval_runner" \
  "$base_model/config.json" "$base_model/model.safetensors" \
  "$base_model/tokenizer.json" "$base_model/spiece.model"; do
  [[ -f "$required" ]] || { echo "missing continuation input: $required" >&2; exit 1; }
done
check_sha "ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5" "$train_rows"
check_sha "a4d7b7c8223a79e1ac3157b2bc72121e64590383a52bb7bde1c540d51f2e7aed" "$dev_rows"
check_sha "1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3" "$skills"
check_sha "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4" "$base_model/model.safetensors"
check_sha "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033" "$base_model/config.json"
check_sha "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066" "$base_model/tokenizer.json"
check_sha "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86" "$base_model/spiece.model"
check_sha "8987d93cf89a93e80f643c84fa389c511dff35a883e990ad723c2d9c9fa3f7d1" "$semantic_ids"
check_sha "548d01f677cd60448789f7b62eb32bdd1121418d407262319b8c66a69922e7f0" "$docid_manifest"
check_sha "e86f5e393d21364afd65a24379ddb081958bfe71f177a6a8bd5719ac5e1a9662" "$initial_checkpoint"

mkdir -p "$experiment"
"${RIPOR_PYTHON_BIN:-python3}" - \
  "$variant" "$head_type" "$catalog_loss_weight" "$position_kl_weight" \
  "$semantic_ids" "$docid_manifest" "$initial_checkpoint" \
  "$experiment/experiment_contract.json" "$world_size" "$per_device_batch_size" <<'PY'
import json
import sys
from pathlib import Path

import torch

import skill_gr_opq_parallel as parallel
import skill_gr_pipeline as pipeline

variant, head_type = sys.argv[1:3]
catalog_loss_weight = float(sys.argv[3])
position_kl_weight = float(sys.argv[4])
semantic_path = Path(sys.argv[5])
manifest_path = Path(sys.argv[6])
initial_checkpoint_path = Path(sys.argv[7])
contract_path = Path(sys.argv[8])
world_size = int(sys.argv[9])
per_device_batch_size = int(sys.argv[10])
semantic_ids, mode = pipeline.load_semantic_ids_with_mode(semantic_path)
audit = parallel.validate_parallel_semantic_ids(
    semantic_ids, mode, depth=64, codebook_size=256, expected_catalog_size=14551
)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
fusion = manifest.get("fusion", {})
if manifest.get("method") != "letter_opq_docids":
    raise SystemExit("not a LETTER OPQ DocID manifest")
if fusion.get("mode") != "weighted_l2_normalized_semantic_cf_concat_v1":
    raise SystemExit("continuation is not using the frozen CF-aware OPQ tokenizer")
if float(fusion.get("cf_energy_fraction", -1)) != 0.2:
    raise SystemExit("CF energy fraction changed")
if manifest.get("semantic_ids_sha256") != audit["semantic_ids_sha256"]:
    raise SystemExit("OPQ manifest logical semantic-ID hash mismatch")
initial = torch.load(initial_checkpoint_path, map_location="cpu", weights_only=False)
if initial.get("head_type") != parallel.RPG_HEAD:
    raise SystemExit("initializer is not the frozen Phase 12 RPG winner")
if initial.get("semantic_ids_sha256") != audit["semantic_ids_sha256"]:
    raise SystemExit("initializer semantic-ID hash mismatch")
if float(initial.get("temperature", -1)) != 0.03:
    raise SystemExit("initializer temperature mismatch")
calibrated = head_type == parallel.RPG_CALIBRATED_HEAD
expected_catalog_loss = 0.1 if calibrated else 0.0
expected_kl = 0.01 if calibrated else 0.0
if catalog_loss_weight != expected_catalog_loss or position_kl_weight != expected_kl:
    raise SystemExit("candidate calibration weights violate the frozen protocol")
current = {
    "status": "complete",
    "study": "letter_opq_winner_cooldown_position_calibration",
    "variant": variant,
    "head_type": head_type,
    "architecture": parallel.architecture_for_head(head_type),
    "depth": 64,
    "codebook_size": 256,
    "semantic_ids_sha256": audit["semantic_ids_sha256"],
    "initializer": {
        "variant": "semantic_cf20_rpg_e10",
        "checkpoint_sha256": "e86f5e393d21364afd65a24379ddb081958bfe71f177a6a8bd5719ac5e1a9662",
    },
    "decoder": {
        "catalog_epochs": 0,
        "query_epochs": 3,
        "query_epoch_offset": 10,
        "global_batch_size": world_size * per_device_batch_size,
        "optimizer": "AdamW",
        "peak_lr": 0.0001,
        "min_lr": 0.00001,
        "lr_scheduler": "cosine",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 20260626,
        "bf16": True,
        "catalog_loss_weight": catalog_loss_weight,
        "position_weight_kl_weight": position_kl_weight,
        "training_objective": parallel.training_objective_for_head(head_type),
        "catalog_scoring": parallel.catalog_scoring_for_head(head_type),
    },
    "selection_evaluation": {
        "split": "dev_rows_001",
        "selection_key": ["R@10", "R@5", "R@1", "R@50"],
    },
    "official_evaluation": "not_performed_by_candidate_runner",
}
if contract_path.exists():
    previous = json.loads(contract_path.read_text(encoding="utf-8"))
    if previous != current:
        raise SystemExit("existing candidate contract differs; use a fresh output directory")
else:
    contract_path.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
print(json.dumps({"candidate_contract": current}, sort_keys=True))
PY

if [[ ! -f "$decoder/final/parallel_decoder.pt" ]]; then
  if [[ -d "$decoder" && -n "$(find "$decoder" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "partial decoder output exists; use a fresh experiment directory: $decoder" >&2
    exit 1
  fi
  LETTER_OPQ_MASTER_PORT="${LETTER_OPQ_MASTER_PORT:?}" \
    bash "$ddp_launcher" scripts/skill_gr_opq_parallel.py train \
      --rows "$train_rows" \
      --dev-rows "$dev_rows" \
      --semantic-ids "$semantic_ids" \
      --skills-jsonl "$skills" \
      --output-dir "$decoder" \
      --model "$base_model" \
      --depth 64 \
      --codebook-size 256 \
      --catalog-epochs 0 \
      --epochs 3 \
      --batch-size "$per_device_batch_size" \
      --expected-global-batch-size 16 \
      --eval-batch-size 16 \
      --lr 1e-4 \
      --min-lr 1e-5 \
      --lr-scheduler cosine \
      --weight-decay 0 \
      --max-grad-norm 1 \
      --epoch-offset 10 \
      --head-type "$head_type" \
      --temperature 0.03 \
      --catalog-loss-weight "$catalog_loss_weight" \
      --position-weight-kl-weight "$position_kl_weight" \
      --initial-checkpoint "$initial_checkpoint" \
      --max-source-length 2048 \
      --logging-steps 50 \
      --eval-steps 1000 \
      --seed 20260626 \
      --bf16
fi

"${RIPOR_PYTHON_BIN:-python3}" - \
  "$head_type" "$catalog_loss_weight" "$position_kl_weight" "$semantic_ids" \
  "$initial_checkpoint" "$decoder/final/parallel_decoder.pt" <<'PY'
import sys
from pathlib import Path

import torch

import skill_gr_opq_parallel as parallel
import skill_gr_pipeline as pipeline

head_type = sys.argv[1]
catalog_loss_weight = float(sys.argv[2])
position_kl_weight = float(sys.argv[3])
semantic_ids, mode = pipeline.load_semantic_ids_with_mode(Path(sys.argv[4]))
initial_path = Path(sys.argv[5])
checkpoint = torch.load(Path(sys.argv[6]), map_location="cpu", weights_only=False)
audit = parallel.validate_parallel_semantic_ids(
    semantic_ids, mode, depth=64, codebook_size=256, expected_catalog_size=14551
)
if checkpoint.get("architecture") != parallel.architecture_for_head(head_type):
    raise SystemExit("final checkpoint architecture mismatch")
if checkpoint.get("head_type") != head_type:
    raise SystemExit("final checkpoint head type mismatch")
if checkpoint.get("semantic_ids_sha256") != audit["semantic_ids_sha256"]:
    raise SystemExit("final checkpoint semantic-ID hash mismatch")
schedule = checkpoint.get("optimizer_schedule", {})
expected_schedule = {
    "name": "cosine",
    "lr": 0.0001,
    "min_lr": 0.00001,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "catalog_epochs": 0,
    "query_epochs": 3,
    "query_epoch_offset": 10,
}
for key, expected in expected_schedule.items():
    if schedule.get(key) != expected:
        raise SystemExit(f"final checkpoint schedule mismatch: {key}")
calibration = checkpoint.get("position_calibration", {})
if calibration.get("catalog_loss_weight") != catalog_loss_weight:
    raise SystemExit("final checkpoint catalog-loss mismatch")
if calibration.get("position_weight_kl_weight") != position_kl_weight:
    raise SystemExit("final checkpoint position-KL mismatch")
initializer = checkpoint.get("inputs", {}).get("initial_checkpoint", {})
if initializer.get("sha256") != parallel.file_sha256(initial_path):
    raise SystemExit("final checkpoint initializer provenance mismatch")
print(f"verified continuation checkpoint head={head_type}")
PY

if [[ ! -f "$dev_eval/eval_manifest.json" || ! -f "$dev_eval/ranked_metrics.json" ]]; then
  env \
    LETTER_OPQ_EVAL_ROWS="$dev_rows" \
    LETTER_OPQ_EVAL_SPLIT=dev_rows_001 \
    LETTER_OPQ_EVAL_SEMANTIC_IDS="$semantic_ids" \
    LETTER_OPQ_EVAL_CHECKPOINT="$decoder/final/parallel_decoder.pt" \
    LETTER_OPQ_EVAL_OUTPUT="$dev_eval" \
    LETTER_OPQ_MASTER_PORT="$((LETTER_OPQ_MASTER_PORT + 20))" \
    LETTER_OPQ_WORLD_SIZE="$world_size" \
    LETTER_OPQ_CFS_AGENT_ROOT="$cfs_agent_root" \
    RIPOR_PROJECT_ROOT="$project_root" \
    bash "$eval_runner"
fi

"${RIPOR_PYTHON_BIN:-python3}" - "$experiment/completion.json" "$variant" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = path.parent
required = {
    "checkpoint": root / "decoder/final/parallel_decoder.pt",
    "train_manifest": root / "decoder/parallel_decoder_train_manifest.json",
    "dev_metrics": root / "dev_exact/ranked_metrics.json",
    "dev_eval_manifest": root / "dev_exact/eval_manifest.json",
}
missing = [str(value) for value in required.values() if not value.is_file()]
if missing:
    raise SystemExit(f"continuation candidate is incomplete: {missing}")

def sha256(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "status": "complete",
    "variant": sys.argv[2],
    "selection_split": "dev_rows_001",
    "official_evaluation": False,
    "artifacts": {
        key: {"path": str(value), "sha256": sha256(value)}
        for key, value in required.items()
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
