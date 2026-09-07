#!/usr/bin/env bash
set -euo pipefail

cfs_agent_root="${LETTER_OPQ_CFS_AGENT_ROOT:-/mnt/cfs-agent}"
project_root="${RIPOR_PROJECT_ROOT:-$cfs_agent_root/workspace/chenyufei03/SkillSearch-data-synthesis}"
variant="${LETTER_OPQ_VARIANT:?LETTER_OPQ_VARIANT is required}"
head_type="${LETTER_OPQ_HEAD_TYPE:?LETTER_OPQ_HEAD_TYPE is required}"
docid_kind="${LETTER_OPQ_DOCID_KIND:?LETTER_OPQ_DOCID_KIND is required}"
docid_root="${LETTER_OPQ_DOCID_ROOT:?LETTER_OPQ_DOCID_ROOT is required}"
expected_sid_file_sha="${LETTER_OPQ_EXPECTED_SID_FILE_SHA:?LETTER_OPQ_EXPECTED_SID_FILE_SHA is required}"
expected_manifest_sha="${LETTER_OPQ_EXPECTED_MANIFEST_SHA:?LETTER_OPQ_EXPECTED_MANIFEST_SHA is required}"
experiment="${LETTER_OPQ_VARIANT_EXPERIMENT:?LETTER_OPQ_VARIANT_EXPERIMENT is required}"
world_size="${LETTER_OPQ_WORLD_SIZE:-4}"
per_device_batch_size="${LETTER_OPQ_PER_DEVICE_BATCH_SIZE:-4}"
temperature="${LETTER_OPQ_TEMPERATURE:-0.03}"

case "$head_type" in
  linear|rpg_cosine_residual) ;;
  *) echo "invalid LETTER_OPQ_HEAD_TYPE: $head_type" >&2; exit 2 ;;
esac
case "$docid_kind" in
  semantic_only|semantic_cf20) ;;
  *) echo "invalid LETTER_OPQ_DOCID_KIND: $docid_kind" >&2; exit 2 ;;
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

inputs="$cfs_agent_root/dataset/chenyufei03/workspace/chenyufei03/datasets/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs"
train_rows="$inputs/rows_000.decoder_rows.base_plus_task5_gt2.jsonl"
dev_rows="$inputs/rows_001.decoder_rows.base.jsonl"
skills="$inputs/canonical_skills.jsonl"
base_model="$cfs_agent_root/dataset/chenyufei03/models/experiments/chenyufei03/SkillSearch-data-synthesis/hf-models/t5-base-a9723ea"
semantic_ids="$docid_root/semantic_ids.jsonl"
docid_manifest="$docid_root/opq_docid_manifest.json"
decoder="$experiment/decoder"
dev_eval="$experiment/dev_exact"
ddp_launcher="$project_root/repro/20260903_letter_opq_ceiling/run_ddp_command.sh"

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
  "$ddp_launcher" "$base_model/config.json" "$base_model/model.safetensors" \
  "$base_model/tokenizer.json" "$base_model/spiece.model"; do
  [[ -f "$required" ]] || {
    echo "missing candidate input: $required" >&2
    exit 1
  }
done
check_sha "ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5" "$train_rows"
check_sha "a4d7b7c8223a79e1ac3157b2bc72121e64590383a52bb7bde1c540d51f2e7aed" "$dev_rows"
check_sha "1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3" "$skills"
check_sha "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4" "$base_model/model.safetensors"
check_sha "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033" "$base_model/config.json"
check_sha "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066" "$base_model/tokenizer.json"
check_sha "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86" "$base_model/spiece.model"
check_sha "$expected_sid_file_sha" "$semantic_ids"
check_sha "$expected_manifest_sha" "$docid_manifest"

mkdir -p "$experiment"
"${RIPOR_PYTHON_BIN:-python3}" - \
  "$variant" "$head_type" "$docid_kind" "$semantic_ids" "$docid_manifest" \
  "$experiment/experiment_contract.json" "$world_size" "$per_device_batch_size" \
  "$temperature" <<'PY'
import json
import sys
from pathlib import Path

import skill_gr_opq_parallel as parallel
import skill_gr_pipeline as pipeline

variant, head_type, docid_kind = sys.argv[1:4]
semantic_path = Path(sys.argv[4])
manifest_path = Path(sys.argv[5])
contract_path = Path(sys.argv[6])
world_size = int(sys.argv[7])
per_device_batch_size = int(sys.argv[8])
temperature = float(sys.argv[9])
semantic_ids, mode = pipeline.load_semantic_ids_with_mode(semantic_path)
audit = parallel.validate_parallel_semantic_ids(
    semantic_ids,
    mode,
    depth=64,
    codebook_size=256,
    expected_catalog_size=14551,
)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("method") != "letter_opq_docids":
    raise SystemExit("not a LETTER OPQ DocID manifest")
if int(manifest.get("depth", -1)) != 64 or int(manifest.get("codebook_size", -1)) != 256:
    raise SystemExit("OPQ manifest depth/codebook mismatch")
if manifest.get("position_order_policy") != "faiss_native_no_search":
    raise SystemExit("OPQ manifest position order changed")
if manifest["integrity"]["resolved_full_tuple"]["collision_groups"] != 0:
    raise SystemExit("OPQ manifest contains collisions")
if manifest.get("semantic_ids_sha256") != audit["semantic_ids_sha256"]:
    raise SystemExit("OPQ manifest logical semantic-ID hash mismatch")
fusion_mode = manifest.get("fusion", {}).get("mode", "semantic_only")
if docid_kind == "semantic_only" and fusion_mode != "semantic_only":
    raise SystemExit(f"semantic-only candidate has unexpected fusion={fusion_mode}")
if docid_kind == "semantic_cf20":
    fusion = manifest.get("fusion", {})
    if fusion_mode != "weighted_l2_normalized_semantic_cf_concat_v1":
        raise SystemExit(f"CF candidate has unexpected fusion={fusion_mode}")
    if float(fusion.get("cf_energy_fraction", -1)) != 0.2:
        raise SystemExit("CF candidate energy fraction is not 0.2")
    if int(fusion.get("quantizer_dim", -1)) != 1088:
        raise SystemExit("CF candidate quantizer dimension is not 1088")

current = {
    "status": "complete",
    "study": "letter_opq_base_pipeline_ceiling",
    "variant": variant,
    "docid_kind": docid_kind,
    "head_type": head_type,
    "architecture": parallel.architecture_for_head(head_type),
    "temperature": temperature,
    "depth": 64,
    "codebook_size": 256,
    "semantic_ids_sha256": audit["semantic_ids_sha256"],
    "decoder": {
        "base_model": "t5-base@a9723ea",
        "catalog_epochs": 3,
        "query_epochs": 10,
        "global_batch_size": world_size * per_device_batch_size,
        "lr": 0.0003,
        "lr_scheduler": "constant",
        "weight_decay": 0.0,
        "seed": 20260626,
        "bf16": True,
        "objective": parallel.TRAINING_OBJECTIVE,
    },
    "selection_evaluation": {
        "split": "dev_rows_001",
        "catalog_scoring": parallel.CATALOG_SCORING,
        "k_values": [1, 5, 10, 50],
    },
    "official_evaluation": "not_performed_by_candidate_runner",
}
if contract_path.exists():
    previous = json.loads(contract_path.read_text(encoding="utf-8"))
    if previous != current:
        raise SystemExit("existing candidate contract differs; use a fresh output directory")
else:
    contract_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
      --catalog-epochs 3 \
      --epochs 10 \
      --batch-size "$per_device_batch_size" \
      --expected-global-batch-size 16 \
      --eval-batch-size 16 \
      --lr 3e-4 \
      --weight-decay 0 \
      --head-type "$head_type" \
      --temperature "$temperature" \
      --max-source-length 2048 \
      --logging-steps 50 \
      --eval-steps 1000 \
      --seed 20260626 \
      --bf16
fi

"${RIPOR_PYTHON_BIN:-python3}" - "$head_type" "$semantic_ids" "$decoder/final/parallel_decoder.pt" <<'PY'
import sys
from pathlib import Path

import torch

import skill_gr_opq_parallel as parallel
import skill_gr_pipeline as pipeline

head_type = sys.argv[1]
semantic_ids, mode = pipeline.load_semantic_ids_with_mode(Path(sys.argv[2]))
checkpoint = torch.load(Path(sys.argv[3]), map_location="cpu", weights_only=False)
audit = parallel.validate_parallel_semantic_ids(
    semantic_ids, mode, depth=64, codebook_size=256, expected_catalog_size=14551
)
if checkpoint.get("architecture") != parallel.architecture_for_head(head_type):
    raise SystemExit("final checkpoint architecture mismatch")
if checkpoint.get("head_type", parallel.LINEAR_HEAD) != head_type:
    raise SystemExit("final checkpoint head type mismatch")
if checkpoint.get("semantic_ids_sha256") != audit["semantic_ids_sha256"]:
    raise SystemExit("final checkpoint semantic-ID hash mismatch")
schedule = checkpoint.get("optimizer_schedule", {})
if schedule.get("name") != "constant" or int(schedule.get("query_epochs", -1)) != 10:
    raise SystemExit("final checkpoint training schedule mismatch")
print(f"verified candidate checkpoint head={head_type}")
PY

if [[ ! -f "$dev_eval/eval_manifest.json" || ! -f "$dev_eval/ranked_metrics.json" ]]; then
  LETTER_OPQ_MASTER_PORT="$((LETTER_OPQ_MASTER_PORT + 20))" \
    bash "$ddp_launcher" scripts/skill_gr_opq_parallel.py eval \
      --rows "$dev_rows" \
      --semantic-ids "$semantic_ids" \
      --checkpoint "$decoder/final/parallel_decoder.pt" \
      --output-dir "$dev_eval" \
      --split-name dev_rows_001 \
      --batch-size 16 \
      --k-values 1,5,10,50
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
    raise SystemExit(f"candidate is incomplete: {missing}")

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

