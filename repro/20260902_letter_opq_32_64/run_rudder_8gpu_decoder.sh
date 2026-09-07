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
world_size="${LETTER_OPQ_WORLD_SIZE:-8}"
per_device_batch_size="${LETTER_OPQ_PER_DEVICE_BATCH_SIZE:-2}"
master_port="${LETTER_OPQ_MASTER_PORT:-295${depth}}"
if [[ $((world_size * per_device_batch_size)) -ne 16 ]]; then
  echo "OPQ decoder global batch must remain 16: world_size=$world_size per_device_batch_size=$per_device_batch_size" >&2
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

inputs="$cfs_agent_root/dataset/chenyufei03/workspace/chenyufei03/datasets/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs"
train_rows="$inputs/rows_000.decoder_rows.base_plus_task5_gt2.jsonl"
dev_rows="$inputs/rows_001.decoder_rows.base.jsonl"
test_rows="$inputs/rows_002.decoder_rows.base.jsonl"
core75_rows="$inputs/rows_003.skillsbench_core75.rows.jsonl"
skills="$inputs/canonical_skills.jsonl"
base_model="$cfs_agent_root/dataset/chenyufei03/models/experiments/chenyufei03/SkillSearch-data-synthesis/hf-models/t5-base-a9723ea"
docid_root="$project_root/.tmp/letter_opq_m${depth}_semantic_c256_seed42_20260901"
semantic_ids="$docid_root/semantic_ids.jsonl"
docid_manifest="$docid_root/opq_docid_manifest.json"
experiment="${RIPOR_EXPERIMENT:?RIPOR_EXPERIMENT is required}"
decoder="$experiment/decoder"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

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
  "$train_rows" "$dev_rows" "$test_rows" "$core75_rows" "$skills" \
  "$semantic_ids" "$docid_manifest" \
  "$base_model/config.json" "$base_model/model.safetensors" \
  "$base_model/tokenizer.json" "$base_model/spiece.model"; do
  [[ -f "$required" ]] || {
    echo "missing required OPQ decoder input: $required" >&2
    exit 1
  }
done

check_sha "ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5" "$train_rows"
check_sha "a4d7b7c8223a79e1ac3157b2bc72121e64590383a52bb7bde1c540d51f2e7aed" "$dev_rows"
check_sha "f6bff43b0c47cb576e35042ae73982fac7305a8e553a1c0afe14de783bce4de8" "$test_rows"
check_sha "4adcd50e359e806d824ffa6f5ee29b4573321033e47ffa2a31780e65ff3cb8ba" "$core75_rows"
check_sha "1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3" "$skills"
check_sha "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4" "$base_model/model.safetensors"
check_sha "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033" "$base_model/config.json"
check_sha "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066" "$base_model/tokenizer.json"
check_sha "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86" "$base_model/spiece.model"
if [[ "$depth" == "32" ]]; then
  check_sha "464ddc9840fef2054dc617da04b617ba7acea6156e3253f3ae3ff915947f9b08" "$semantic_ids"
else
  check_sha "d059d380a15f6cc9adfb66c0476f0d4ecfc0b40127585d62c0b85c98a21d6134" "$semantic_ids"
fi

mkdir -p "$experiment"

"$python_bin" - \
  "$depth" "$semantic_ids" "$docid_manifest" "$experiment/experiment_contract.json" \
  "$world_size" "$per_device_batch_size" <<'PY'
import json
import sys
from pathlib import Path

import skill_gr_pipeline as pipeline

depth = int(sys.argv[1])
semantic_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
contract_path = Path(sys.argv[4])
world_size = int(sys.argv[5])
per_device_batch_size = int(sys.argv[6])
semantic_ids, mode = pipeline.load_semantic_ids_with_mode(semantic_path)
config = pipeline.infer_structured_sid_config(
    semantic_ids,
    codebook_size=256,
    dedup_size=0,
    semantic_id_mode=mode,
)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if config["depth"] != depth or manifest["depth"] != depth:
    raise SystemExit(
        f"OPQ depth mismatch: requested={depth} ids={config['depth']} manifest={manifest['depth']}"
    )
if len(semantic_ids) != 14551:
    raise SystemExit(f"OPQ catalog size mismatch: {len(semantic_ids)}")
logical_sha = pipeline.semantic_ids_sha256(semantic_ids, mode)
if logical_sha != manifest["semantic_ids_sha256"]:
    raise SystemExit("OPQ logical semantic-ID hash mismatch")
if manifest["integrity"]["resolved_full_tuple"]["collision_groups"] != 0:
    raise SystemExit("OPQ manifest contains non-unique resolved DocIDs")
current = {
    "status": "complete",
    "comparison": "opq_32_vs_opq_64",
    "only_variable": "opq_depth",
    "depth": depth,
    "codebook_size": 256,
    "semantic_id_mode": mode,
    "skills": len(semantic_ids),
    "semantic_ids_sha256": logical_sha,
    "decoder": {
        "architecture": "letter_sequential_autoregressive_t5_trie",
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
        "query_loss": "expanded_ce",
    },
    "evaluation": {
        "splits": ["official_core75", "private_test1000"],
        "num_beams": 50,
        "rank_aggregation": "beam_order",
        "k_values": [1, 5, 10, 50],
    },
}
if contract_path.exists():
    previous = json.loads(contract_path.read_text(encoding="utf-8"))
    if previous != current:
        raise SystemExit("existing experiment contract differs; use a fresh output directory")
else:
    contract_path.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
print(json.dumps({"experiment_contract": current}, sort_keys=True))
PY

if [[ ! -f "$decoder/final/structured_decoder.pt" ]]; then
  if [[ -d "$decoder" && -n "$(find "$decoder" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "partial decoder output exists without final checkpoint; use a fresh experiment directory: $decoder" >&2
    exit 1
  fi
  "$python_bin" -m torch.distributed.run \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$master_port" \
    --nproc_per_node="$world_size" \
    scripts/skill_gr_pipeline.py train-single-sid-decoder \
      --rows "$train_rows" \
      --dev-rows "$dev_rows" \
      --semantic-ids "$semantic_ids" \
      --skills-jsonl "$skills" \
      --output-dir "$decoder" \
      --model "$base_model" \
      --codebook-size 256 \
      --dedup-size 0 \
      --query-loss expanded_ce \
      --catalog-epochs 3 \
      --epochs 5 \
      --batch-size "$per_device_batch_size" \
      --eval-batch-size 16 \
      --lr 3e-4 \
      --lr-scheduler constant \
      --warmup-ratio 0 \
      --weight-decay 0 \
      --max-source-length 2048 \
      --logging-steps 50 \
      --eval-steps 1000 \
      --save-steps 0 \
      --no-padding-distributed-sampler \
      --seed 20260626 \
      --bf16
fi

"$python_bin" - "$depth" "$semantic_ids" "$decoder/final/structured_decoder.pt" <<'PY'
import sys
from pathlib import Path

import torch

import skill_gr_pipeline as pipeline

depth = int(sys.argv[1])
semantic_ids, mode = pipeline.load_semantic_ids_with_mode(Path(sys.argv[2]))
checkpoint = torch.load(Path(sys.argv[3]), map_location="cpu", weights_only=False)
if int(checkpoint["depth"]) != depth:
    raise SystemExit("final decoder depth mismatch")
if int(checkpoint["codebook_size"]) != 256 or int(checkpoint.get("dedup_size", -1)) != 0:
    raise SystemExit("final decoder codebook schema mismatch")
if not pipeline.validate_checkpoint_semantic_ids(checkpoint, semantic_ids, mode):
    raise SystemExit("final decoder does not verify its OPQ semantic IDs")
if checkpoint.get("optimizer_schedule", {}).get("name") != "constant":
    raise SystemExit("final decoder did not preserve the constant-LR protocol")
print(f"verified final OPQ-{depth} decoder")
PY

run_eval() {
  local rows="$1"
  local split="$2"
  local output="$3"
  if [[ -f "$output/eval_manifest.json" && -f "$output/ranked_metrics.json" ]]; then
    echo "skip completed eval: $output"
    return
  fi
  "$python_bin" -m torch.distributed.run \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$master_port" \
    --nproc_per_node="$world_size" \
    scripts/skill_gr_pipeline.py eval-single-sid-decoder \
      --rows "$rows" \
      --model-dir "$decoder/final" \
      --semantic-ids "$semantic_ids" \
      --output-dir "$output" \
      --split-name "$split" \
      --num-beams 50 \
      --max-skills 1 \
      --rank-aggregation beam_order \
      --k-values 1,5,10,50
}

run_eval "$core75_rows" "official_core75" "$experiment/official_core75_beam50"
run_eval "$test_rows" "private_test1000" "$experiment/private_test1000_beam50"

"$python_bin" - "$experiment/completion.json" "$depth" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = path.parent
required = {
    "decoder": root / "decoder/final/structured_decoder.pt",
    "core75": root / "official_core75_beam50/ranked_metrics.json",
    "test1000": root / "private_test1000_beam50/ranked_metrics.json",
}
missing = [str(value) for value in required.values() if not value.is_file()]
if missing:
    raise SystemExit(f"OPQ experiment is incomplete: {missing}")

def sha256(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "status": "complete",
    "depth": int(sys.argv[2]),
    "artifacts": {
        key: {"path": str(value), "sha256": sha256(value)}
        for key, value in required.items()
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
