#!/usr/bin/env bash
set -euo pipefail

cfs_agent_root="${LETTER_OPQ_CFS_AGENT_ROOT:-/mnt/cfs-agent}"
project_root="${RIPOR_PROJECT_ROOT:-$cfs_agent_root/workspace/chenyufei03/SkillSearch-data-synthesis}"
python_bin="${RIPOR_PYTHON_BIN:-python3}"
depth="${LETTER_OPQ_DEPTH:?LETTER_OPQ_DEPTH must be 32 or 64}"
case "$depth" in
  32|64) ;;
  *) echo "LETTER_OPQ_DEPTH must be 32 or 64, got: $depth" >&2; exit 2 ;;
esac
world_size="${LETTER_OPQ_WORLD_SIZE:-4}"
per_device_batch_size="${LETTER_OPQ_PER_DEVICE_BATCH_SIZE:-4}"
master_port="${LETTER_OPQ_MASTER_PORT:-296${depth}}"
if [[ $((world_size * per_device_batch_size)) -ne 16 ]]; then
  echo "OPQ parallel global batch must remain 16: world_size=$world_size per_device_batch_size=$per_device_batch_size" >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "${#visible_devices[@]}" -ne "$world_size" ]]; then
    echo "CUDA_VISIBLE_DEVICES/world-size mismatch: visible=${#visible_devices[@]} world_size=$world_size" >&2
    exit 2
  fi
fi

cd "$project_root"
export PYTHONPATH="$project_root/scripts${PYTHONPATH:+:$PYTHONPATH}"

inputs="$cfs_agent_root/dataset/chenyufei03/workspace/chenyufei03/datasets/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs"
train_rows="$inputs/rows_000.decoder_rows.base_plus_task5_gt2.jsonl"
dev_rows="$inputs/rows_001.decoder_rows.base.jsonl"
core75_rows="$inputs/rows_003.skillsbench_core75.rows.jsonl"
skills="$inputs/canonical_skills.jsonl"
base_model="$cfs_agent_root/dataset/chenyufei03/models/experiments/chenyufei03/SkillSearch-data-synthesis/hf-models/t5-base-a9723ea"
docid_root="$project_root/.tmp/letter_opq_m${depth}_semantic_c256_seed42_20260901"
semantic_ids="$docid_root/semantic_ids.jsonl"
docid_manifest="$docid_root/opq_docid_manifest.json"
experiment="${RIPOR_EXPERIMENT:?RIPOR_EXPERIMENT is required}"
decoder="$experiment/decoder"
official_eval="$experiment/official_core75_exact"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

declare -a ddp_pids=()
terminate_ddp_children() {
  local pid
  for pid in "${ddp_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  ddp_pids=()
}
trap terminate_ddp_children INT TERM HUP EXIT

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
  "$train_rows" "$dev_rows" "$core75_rows" "$skills" \
  "$semantic_ids" "$docid_manifest" \
  "$base_model/config.json" "$base_model/model.safetensors" \
  "$base_model/tokenizer.json" "$base_model/spiece.model"; do
  [[ -f "$required" ]] || {
    echo "missing required OPQ parallel input: $required" >&2
    exit 1
  }
done

check_sha "ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5" "$train_rows"
check_sha "a4d7b7c8223a79e1ac3157b2bc72121e64590383a52bb7bde1c540d51f2e7aed" "$dev_rows"
check_sha "4adcd50e359e806d824ffa6f5ee29b4573321033e47ffa2a31780e65ff3cb8ba" "$core75_rows"
check_sha "1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3" "$skills"
check_sha "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4" "$base_model/model.safetensors"
check_sha "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033" "$base_model/config.json"
check_sha "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066" "$base_model/tokenizer.json"
check_sha "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86" "$base_model/spiece.model"
if [[ "$depth" == "32" ]]; then
  check_sha "464ddc9840fef2054dc617da04b617ba7acea6156e3253f3ae3ff915947f9b08" "$semantic_ids"
  check_sha "c6605c16f3ff17ab95f66e47f2099afbed19b9bb66afb76c76721ccfadfb491e" "$docid_manifest"
else
  check_sha "d059d380a15f6cc9adfb66c0476f0d4ecfc0b40127585d62c0b85c98a21d6134" "$semantic_ids"
  check_sha "34c088b72abfa755196830c5b442c2668cb7ed863e4118afab1626aa588e8c8d" "$docid_manifest"
fi

mkdir -p "$experiment"

"$python_bin" - \
  "$depth" "$semantic_ids" "$docid_manifest" "$experiment/experiment_contract.json" \
  "$world_size" "$per_device_batch_size" <<'PY'
import json
import sys
from pathlib import Path

import skill_gr_opq_parallel as parallel
import skill_gr_pipeline as pipeline

depth = int(sys.argv[1])
semantic_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
contract_path = Path(sys.argv[4])
world_size = int(sys.argv[5])
per_device_batch_size = int(sys.argv[6])
semantic_ids, mode = pipeline.load_semantic_ids_with_mode(semantic_path)
audit = parallel.validate_parallel_semantic_ids(
    semantic_ids,
    mode,
    depth=depth,
    codebook_size=256,
    expected_catalog_size=14551,
)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("method") != "letter_opq_docids":
    raise SystemExit("not a frozen LETTER OPQ DocID manifest")
if int(manifest.get("depth", -1)) != depth or int(manifest.get("codebook_size", -1)) != 256:
    raise SystemExit("OPQ manifest depth/codebook mismatch")
if manifest.get("position_order_policy") != "faiss_native_no_search":
    raise SystemExit("OPQ manifest position-order policy changed")
if manifest["integrity"]["resolved_full_tuple"]["collision_groups"] != 0:
    raise SystemExit("OPQ manifest contains non-unique resolved DocIDs")
if manifest.get("semantic_ids_sha256") != audit["semantic_ids_sha256"]:
    raise SystemExit("OPQ manifest logical semantic-ID hash mismatch")
current = {
    "status": "complete",
    "comparison": "opq_32_vs_opq_64_parallel_mtp",
    "only_variable": "opq_depth",
    "depth": depth,
    "codebook_size": 256,
    "semantic_id_mode": mode,
    "skills": len(semantic_ids),
    "semantic_ids_sha256": audit["semantic_ids_sha256"],
    "decoder": {
        "architecture": parallel.ARCHITECTURE,
        "shared_representation": "t5_decoder_start_hidden_state",
        "position_heads": depth,
        "base_model": "t5-base@a9723ea",
        "catalog_epochs": 3,
        "query_epochs": 5,
        "global_batch_size": world_size * per_device_batch_size,
        "per_device_batch_size": per_device_batch_size,
        "world_size": world_size,
        "lr": 0.0003,
        "lr_scheduler": "constant",
        "weight_decay": 0.0,
        "max_source_length": 2048,
        "seed": 20260626,
        "bf16": True,
        "query_units": "expanded_single_sid",
        "loss": parallel.TRAINING_OBJECTIVE,
    },
    "evaluation": {
        "splits": ["official_core75"],
        "catalog_scoring": parallel.CATALOG_SCORING,
        "catalog_items_per_query": 14551,
        "k_values": [1, 5, 10, 50],
        "private_set": False,
    },
}
if contract_path.exists():
    previous = json.loads(contract_path.read_text(encoding="utf-8"))
    if previous != current:
        raise SystemExit("existing parallel experiment contract differs; use a fresh output directory")
else:
    contract_path.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
print(json.dumps({"experiment_contract": current}, sort_keys=True))
PY

run_distributed() {
  local status_dir rank status_file remaining failed pid
  status_dir="$(mktemp -d /tmp/letter-opq-ddp.XXXXXX)"
  ddp_pids=()
  echo "local_ddp_launch depth=$depth world_size=$world_size master=127.0.0.1:$master_port"
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
    ddp_pids+=("$!")
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
        echo "local DDP rank failed: depth=$depth rank=$rank status=$status" >&2
        failed="$status"
        break
      fi
    done
    if [[ "$remaining" -gt 0 && "$failed" -eq 0 ]]; then
      sleep 1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    terminate_ddp_children
  else
    for pid in "${ddp_pids[@]}"; do
      wait "$pid"
    done
    ddp_pids=()
  fi
  case "$status_dir" in
    /tmp/letter-opq-ddp.*) rm -rf -- "$status_dir" ;;
    *) echo "refusing to remove unexpected DDP status directory: $status_dir" >&2; return 1 ;;
  esac
  if [[ "$failed" -ne 0 ]]; then
    return "$failed"
  fi
}

if [[ ! -f "$decoder/final/parallel_decoder.pt" ]]; then
  if [[ -d "$decoder" && -n "$(find "$decoder" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "partial parallel decoder output exists without final checkpoint; use a fresh experiment directory: $decoder" >&2
    exit 1
  fi
  run_distributed scripts/skill_gr_opq_parallel.py train \
    --rows "$train_rows" \
    --dev-rows "$dev_rows" \
    --semantic-ids "$semantic_ids" \
    --skills-jsonl "$skills" \
    --output-dir "$decoder" \
    --model "$base_model" \
    --depth "$depth" \
    --codebook-size 256 \
    --catalog-epochs 3 \
    --epochs 5 \
    --batch-size "$per_device_batch_size" \
    --expected-global-batch-size 16 \
    --eval-batch-size 16 \
    --lr 3e-4 \
    --weight-decay 0 \
    --max-source-length 2048 \
    --logging-steps 50 \
    --eval-steps 1000 \
    --seed 20260626 \
    --bf16
fi

"$python_bin" - "$depth" "$semantic_ids" "$decoder/final/parallel_decoder.pt" <<'PY'
import sys
from pathlib import Path

import torch

import skill_gr_opq_parallel as parallel
import skill_gr_pipeline as pipeline

depth = int(sys.argv[1])
semantic_ids, mode = pipeline.load_semantic_ids_with_mode(Path(sys.argv[2]))
checkpoint = torch.load(Path(sys.argv[3]), map_location="cpu", weights_only=False)
audit = parallel.validate_parallel_semantic_ids(
    semantic_ids, mode, depth=depth, codebook_size=256, expected_catalog_size=14551
)
if checkpoint.get("architecture") != parallel.ARCHITECTURE:
    raise SystemExit("final checkpoint is not the parallel-MTP architecture")
if int(checkpoint.get("depth", -1)) != depth or int(checkpoint.get("codebook_size", -1)) != 256:
    raise SystemExit("final parallel decoder schema mismatch")
if checkpoint.get("semantic_ids_sha256") != audit["semantic_ids_sha256"]:
    raise SystemExit("final parallel decoder semantic-ID hash mismatch")
if checkpoint.get("optimizer_schedule", {}).get("name") != "constant":
    raise SystemExit("final parallel decoder did not preserve constant LR")
print(f"verified final OPQ-{depth} parallel decoder")
PY

if [[ ! -f "$official_eval/eval_manifest.json" || ! -f "$official_eval/ranked_metrics.json" ]]; then
  run_distributed scripts/skill_gr_opq_parallel.py eval \
    --rows "$core75_rows" \
    --semantic-ids "$semantic_ids" \
    --checkpoint "$decoder/final/parallel_decoder.pt" \
    --output-dir "$official_eval" \
    --split-name official_core75 \
    --batch-size 16 \
    --k-values 1,5,10,50
fi

"$python_bin" - "$experiment/completion.json" "$depth" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = path.parent
required = {
    "decoder": root / "decoder/final/parallel_decoder.pt",
    "train_manifest": root / "decoder/parallel_decoder_train_manifest.json",
    "official_core75": root / "official_core75_exact/ranked_metrics.json",
    "official_eval_manifest": root / "official_core75_exact/eval_manifest.json",
}
missing = [str(value) for value in required.values() if not value.is_file()]
if missing:
    raise SystemExit(f"OPQ parallel experiment is incomplete: {missing}")

def sha256(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "status": "complete",
    "depth": int(sys.argv[2]),
    "evaluation_scope": "official_core75_only",
    "artifacts": {
        key: {"path": str(value), "sha256": sha256(value)}
        for key, value in required.items()
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
