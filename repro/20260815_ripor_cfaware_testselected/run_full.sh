#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

asset_root="outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713"
inputs="$asset_root/canonical_alias_core75_learned_d4_inputs"
train_rows="$inputs/rows_000.decoder_rows.base_plus_task5_gt2.jsonl"
official_rows="$inputs/rows_003.skillsbench_core75.rows.jsonl"
skills="$inputs/canonical_skills.jsonl"
cf_root="$asset_root/lightgcn_cf_d32_l3_e500_seed42_core75_gt2"
cf_embeddings="$cf_root/lightgcn_cf_embeddings.pt"
cf_manifest="$cf_root/manifest.json"
cf_skill_ids="$cf_root/skill_ids.jsonl"
qwen_model="Qwen/Qwen3-Embedding-0.6B"

source_experiment="experiments/ripor_faithful_cat3query5_20260813"
source_m0_dense="$source_experiment/m0_dense"
source_m0_final_mined="$source_experiment/assets/m0_final_mined"
source_dense_rank_scored="$source_experiment/assets/dense_rank_scored"
source_rq="$source_experiment/rq_depth8_vocab256"
source_m2="$source_experiment/m2_initial_rank"

experiment="${RIPOR_EXPERIMENT:-experiments/ripor_cfaware_testselected_cat3query5_20260815}"
reuse_rq="${RIPOR_REUSE_RQ:-}"
reuse_m1="${RIPOR_REUSE_M1:-}"
query_balanced_multi_positive="${RIPOR_QUERY_BALANCED_MULTI_POSITIVE:-0}"
repro_readme="${RIPOR_REPRO_README:-repro/20260815_ripor_cfaware_testselected/README.md}"
launcher_path="${RIPOR_LAUNCHER_PATH:-${BASH_SOURCE[0]}}"
assets="$experiment/assets"
teacher_root="experiments/ripor_skillrouter_teacher_cat3query5_20260813/assets"
teacher_queries="$teacher_root/teacher_query_embeddings"
teacher_skills="$teacher_root/skillrouter_skill_embeddings.npy"
teacher_skill_manifest="$teacher_root/skillrouter_skill_embeddings.manifest.json"
decoder_start="repro/20260813_ripor_faithful/t5-base.npy.b64"

expected_train_sha="ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5"
expected_official_sha="4adcd50e359e806d824ffa6f5ee29b4573321033e47ffa2a31780e65ff3cb8ba"
expected_skills_sha="1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3"
expected_teacher_skills_sha="269ab5985dbb68e38326a3391656b43dbe3c7203d2583ae273ccfd7c49340df7"
expected_decoder_start_sha="fe928c4929126abb23e1b22532d21aec65c9a723f2e5fb1f035cfea7da211f02"
expected_cf_sha="b4d6a16150f79c202314a8ba1f3917ae3106876ee9e8a69ff6f9d68edb928f84"
expected_cf_manifest_sha="151774db3e8badb6174646783f6c8bb2bacd5a0a5cb59f45c983aa26f61e83b4"
expected_cf_skill_ids_sha="26daa59096f84c50e0c6c07e58736f37f4d36a7f753fdec55d306a31d8276b50"
expected_m0_checkpoint_sha="7aa06b562988a55b1a953310b7e89412b4b59c9bfaaa6f7350ee57997d9be9d3"
expected_m0_embeddings_sha="34b2ec494ed8679e0552414f137b0530f6d122faac31f5fd870d6b33c9287df1"
expected_m0_embeddings_manifest_sha="6e31234581291cc1576e20b72dfbfaf66b8de44eca329b9bc9d9e5b7a30ad079"

base_model="t5-base"
seed=42
depth=8
codebook_size=256
m0_weight="${RIPOR_M0_WEIGHT:-0.5}"
qwen_weight="${RIPOR_QWEN_WEIGHT:-0.3}"
cf_weight="${RIPOR_CF_WEIGHT:-0.5}"
qwen_three_way="${RIPOR_QWEN3_THREE_WAY:-0}"
qwen_batch=32
qwen_max_length=2048
beam_mining_top_k=100
m1_epochs=5
m1_batch=32
m1_lr="1e-3"
m1_max_length=768
rank_epochs=5
rank_batch=64
rank_microbatch=32
rank_lr="1e-4"
rank_max_length=768
warmup_ratio="0.045"
max_grad_norm="1.0"

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch: path=$path expected=$expected actual=$actual" >&2
    exit 1
  fi
}

json_complete() {
  local manifest="$1"
  [[ -f "$manifest" ]] || return 1
  python3 - "$manifest" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "complete" else 1)
PY
}

run_unless_complete() {
  local manifest="$1"
  shift
  if json_complete "$manifest"; then
    echo "stage_already_complete manifest=$manifest"
  else
    "$@"
  fi
}

assert_selected_checkpoint() {
  local stage_dir="$1"
  python3 - "$stage_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

stage_dir = Path(sys.argv[1])
manifest_path = stage_dir / "train_manifest.json"
checkpoint_path = stage_dir / "best" / "structured_decoder.pt"
manifest = json.load(open(manifest_path, encoding="utf-8"))
if manifest.get("status") != "complete":
    raise SystemExit(f"incomplete selected-checkpoint stage: {stage_dir}")
epoch = manifest.get("checkpoint_selection_epoch")
if not isinstance(epoch, int) or epoch < 1:
    raise SystemExit(f"missing checkpoint_selection_epoch: {manifest_path}")
if not checkpoint_path.is_file():
    raise SystemExit(f"missing selected checkpoint: {checkpoint_path}")
digest_hash = hashlib.sha256()
with checkpoint_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest_hash.update(chunk)
digest = digest_hash.hexdigest()
print(f"selected_checkpoint_ready stage={stage_dir.name} epoch={epoch} sha256={digest}")
PY
}

assert_parent_checkpoint() {
  local child_manifest="$1"
  local parent_dir="$2"
  python3 - "$child_manifest" "$parent_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import torch

manifest_path = Path(sys.argv[1])
parent_dir = Path(sys.argv[2])
checkpoint_path = parent_dir / "structured_decoder.pt"
manifest = json.load(open(manifest_path, encoding="utf-8"))
expected_hash = hashlib.sha256()
with checkpoint_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        expected_hash.update(chunk)
expected_sha = expected_hash.hexdigest()
actual_dir = manifest.get("parent_model_dir")
actual_sha = manifest.get("parent_checkpoint_sha256")
if actual_dir == str(parent_dir) and actual_sha == expected_sha:
    print(f"parent_checkpoint_verified manifest={manifest_path} sha256={expected_sha}")
    raise SystemExit(0)

actual_checkpoint_path = Path(actual_dir) / "structured_decoder.pt"
if not actual_checkpoint_path.is_file():
    raise SystemExit(
        "parent checkpoint mismatch: "
        f"manifest={manifest_path} expected_dir={parent_dir} actual_dir={actual_dir} "
        f"expected_sha={expected_sha} actual_sha={actual_sha}"
    )
actual_hash = hashlib.sha256()
with actual_checkpoint_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        actual_hash.update(chunk)
if actual_hash.hexdigest() != actual_sha:
    raise SystemExit(
        f"recorded parent hash mismatch: manifest={manifest_path} "
        f"recorded_sha={actual_sha} file_sha={actual_hash.hexdigest()}"
    )

expected_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
actual_state = torch.load(actual_checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
if expected_state.keys() != actual_state.keys() or any(
    not torch.equal(expected_state[key], actual_state[key]) for key in expected_state
):
    raise SystemExit(
        "parent checkpoint tensor mismatch: "
        f"manifest={manifest_path} expected_dir={parent_dir} actual_dir={actual_dir}"
    )
print(
    "parent_checkpoint_tensor_equivalent "
    f"manifest={manifest_path} expected_dir={parent_dir} actual_dir={actual_dir}"
)
PY
}

for manifest in \
  "$source_m0_dense/train_manifest.json" \
  "$source_m0_final_mined/candidates/manifest.json" \
  "$source_dense_rank_scored/manifest.json" \
  "$source_rq/manifest.json" \
  "$source_m2/train_manifest.json"; do
  json_complete "$manifest" || { echo "incomplete source artifact: $manifest" >&2; exit 1; }
done
check_sha "$expected_train_sha" "$train_rows"
check_sha "$expected_official_sha" "$official_rows"
check_sha "$expected_skills_sha" "$skills"
check_sha "$expected_teacher_skills_sha" "$teacher_skills"
check_sha "$expected_decoder_start_sha" <(base64 --decode < "$decoder_start")
check_sha "$expected_cf_sha" "$cf_embeddings"
check_sha "$expected_cf_manifest_sha" "$cf_manifest"
check_sha "$expected_cf_skill_ids_sha" "$cf_skill_ids"
check_sha "$expected_m0_checkpoint_sha" "$source_m0_dense/final/dense_model.pt"
check_sha "$expected_m0_embeddings_sha" "$source_m0_final_mined/m0_skill_embeddings.npy"
check_sha "$expected_m0_embeddings_manifest_sha" \
  "$source_m0_final_mined/m0_skill_embeddings.manifest.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$assets" "$experiment/source_snapshot"
cp scripts/skill_gr_ripor.py "$experiment/source_snapshot/skill_gr_ripor.py"
cp scripts/skill_gr_dgr.py "$experiment/source_snapshot/skill_gr_dgr.py"
cp scripts/skill_gr_pipeline.py "$experiment/source_snapshot/skill_gr_pipeline.py"
cp tests/test_skill_gr_ripor.py "$experiment/source_snapshot/test_skill_gr_ripor.py"
cp tests/test_skill_gr_dgr.py "$experiment/source_snapshot/test_skill_gr_dgr.py"
cp "$repro_readme" "$experiment/source_snapshot/README.md"
cp "${BASH_SOURCE[0]}" "$experiment/source_snapshot/run_full.sh"
if [[ "$launcher_path" != "${BASH_SOURCE[0]}" ]]; then
  cp "$launcher_path" "$experiment/source_snapshot/launcher.sh"
fi

rank_objective_args=()
if [[ "$query_balanced_multi_positive" == "1" ]]; then
  rank_objective_args+=(--query-balanced-multi-positive)
elif [[ "$query_balanced_multi_positive" != "0" ]]; then
  echo "RIPOR_QUERY_BALANCED_MULTI_POSITIVE must be 0 or 1" >&2
  exit 1
fi

qwen_args=()
if [[ "$qwen_three_way" == "1" ]]; then
  if [[ "$cf_weight" != "0.2" || "$m0_weight" != "0.5" || "$qwen_weight" != "0.3" ]]; then
    echo "three-way RQ requires M0/Qwen/CF weights 0.5/0.3/0.2" >&2
    exit 1
  fi
elif [[ "$qwen_three_way" != "0" ]]; then
  echo "RIPOR_QWEN3_THREE_WAY must be 0 or 1" >&2
  exit 1
fi

python3 -m pytest -q \
  tests/test_skill_gr_ripor.py \
  tests/test_skill_gr_dgr.py \
  tests/test_skill_gr_pipeline.py \
  | tee "$experiment/preflight_tests.log"

# Smoke-test the unchanged query-batch=4 beam path on the fixed Official rows.
preflight_q4="$assets/preflight_beam_q4"
run_unless_complete "$preflight_q4/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py mine-beam-candidates \
    --rows "$official_rows" \
    --skills-jsonl "$skills" \
    --model-dir "$source_m2/final" \
    --rq-dir "$source_rq" \
    --expected-stage M2 \
    --top-k 100 \
    --query-batch-size 4 \
    --decoder-batch-size 1024 \
    --max-source-length "$rank_max_length" \
    --shard-size 4096 \
    --bf16 \
    --output-dir "$preflight_q4"
python3 - "$preflight_q4" <<'PY'
import json
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
manifest = json.load(open(root / "manifest.json", encoding="utf-8"))
payloads = [
    torch.load(path, map_location="cpu", weights_only=False)
    for path in sorted((root / "shards").glob("*.pt"))
]
counts = torch.cat([payload["candidate_counts"] for payload in payloads])
if manifest.get("status") != "complete" or len(counts) != 75 or int(counts.min()) < 100:
    raise SystemExit("query-batch=4 beam smoke test failed")
print("beam_smoke=passed rows=75 top_k_sids=100 query_batch=4")
PY

test_selection_args=(
  --test-selection-rows "$official_rows"
  --test-selection-expected-rows-sha256 "$expected_official_sha"
  --test-selection-expected-skills-sha256 "$expected_skills_sha"
  --test-selection-num-beams 50
  --test-selection-query-batch-size 1
  --test-selection-decoder-batch-size 1024
  --test-selection-max-source-length "$rank_max_length"
)

if [[ "$qwen_three_way" == "1" ]]; then
  qwen_root="$assets/qwen3_base_skill_embeddings"
  qwen_embeddings="$qwen_root/skill_embeddings.pt"
  qwen_manifest="$qwen_root/embedding_manifest.json"
  qwen_complete=0
  if [[ -f "$qwen_embeddings" && -f "$qwen_manifest" ]]; then
    if python3 - "$qwen_embeddings" "$qwen_manifest" "$skills" <<'PY'
import json
import sys
from pathlib import Path
import torch

embedding_path, manifest_path, skills_path = map(Path, sys.argv[1:])
manifest = json.load(open(manifest_path, encoding="utf-8"))
skills = [json.loads(line) for line in open(skills_path, encoding="utf-8") if line.strip()]
skill_ids = [str(row.get("skill_id") or row.get("id")) for row in skills]
payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
valid = (
    manifest.get("model") == "Qwen/Qwen3-Embedding-0.6B"
    and manifest.get("skills") == len(skill_ids)
    and manifest.get("max_length") == 2048
    and payload.get("skill_ids") == skill_ids
    and tuple(payload["embeddings"].shape) == (len(skill_ids), 1024)
    and bool(torch.isfinite(payload["embeddings"]).all())
)
raise SystemExit(0 if valid else 1)
PY
    then
      qwen_complete=1
    fi
  fi
  if [[ "$qwen_complete" == "0" ]]; then
    if [[ -e "$qwen_root" ]]; then
      echo "refusing to overwrite incomplete Qwen embedding artifact: $qwen_root" >&2
      exit 1
    fi
    python3 -u scripts/skill_gr_pipeline.py encode-skills \
      --skills-jsonl "$skills" \
      --model "$qwen_model" \
      --max-length "$qwen_max_length" \
      --batch-size "$qwen_batch" \
      --output-dir "$qwen_root"
  fi
  qwen_args=(
    --qwen-embeddings "$qwen_embeddings"
    --qwen-manifest "$qwen_manifest"
    --m0-weight "$m0_weight"
    --qwen-weight "$qwen_weight"
  )
fi

if [[ -n "$reuse_rq" ]]; then
  rq="$reuse_rq"
  json_complete "$rq/manifest.json" || {
    echo "incomplete reused RQ artifact: $rq/manifest.json" >&2
    exit 1
  }
else
  if [[ "$qwen_three_way" == "1" ]]; then
    rq="$experiment/rq_m0w05_qwenw03_cfw02_depth8_vocab256"
  else
    rq="$experiment/rq_cf_procrustes_w05_depth8_vocab256"
  fi
  run_unless_complete "$rq/manifest.json" \
    python3 -u scripts/skill_gr_ripor.py build-rq \
      --embeddings "$source_m0_final_mined/m0_skill_embeddings.npy" \
      --embeddings-manifest "$source_m0_final_mined/m0_skill_embeddings.manifest.json" \
      --skills-jsonl "$skills" \
      --cf-embeddings "$cf_embeddings" \
      --cf-manifest "$cf_manifest" \
      --cf-skill-ids "$cf_skill_ids" \
      --cf-weight "$cf_weight" \
      "${qwen_args[@]}" \
      --depth "$depth" \
      --codebook-size "$codebook_size" \
      --seed "$seed" \
      --faiss-threads 32 \
      --output-dir "$rq"
fi

if [[ -n "$reuse_m1" ]]; then
  m1="$reuse_m1"
  json_complete "$m1/train_manifest.json" || {
    echo "incomplete reused M1 artifact: $m1/train_manifest.json" >&2
    exit 1
  }
else
  m1="$experiment/m1_seq2seq"
  run_unless_complete "$m1/train_manifest.json" \
    python3 -u scripts/skill_gr_ripor.py train-m1 \
      --rows "$train_rows" \
      --skills-jsonl "$skills" \
      --base-model "$base_model" \
      --m0-model-dir "$source_m0_dense/final" \
      --decoder-start-token "$decoder_start" \
      --rq-dir "$rq" \
      --epochs "$m1_epochs" \
      --batch-size "$m1_batch" \
      --lr "$m1_lr" \
      --weight-decay 0 \
      --warmup-ratio "$warmup_ratio" \
      --max-grad-norm "$max_grad_norm" \
      --max-source-length "$m1_max_length" \
      --seed "$seed" \
      --bf16 \
      "${test_selection_args[@]}" \
      --output-dir "$m1"
fi
assert_selected_checkpoint "$m1"

m2="$experiment/m2_initial_rank"
run_unless_complete "$m2/train_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py train-rank \
    --stage-name M2 \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --parent-model-dir "$m1/best" \
    --rq-dir "$rq" \
    --candidates-dir "$source_dense_rank_scored" \
    --active-max-prefix "$depth" \
    --prefix-beta 2 \
    --epochs "$rank_epochs" \
    --batch-size "$rank_batch" \
    --microbatch-size "$rank_microbatch" \
    --lr "$rank_lr" \
    --weight-decay 0 \
    --warmup-ratio "$warmup_ratio" \
    --max-grad-norm "$max_grad_norm" \
    --max-source-length "$rank_max_length" \
    --seed "$seed" \
    --bf16 \
    "${rank_objective_args[@]}" \
    "${test_selection_args[@]}" \
    --output-dir "$m2"
assert_selected_checkpoint "$m2"
assert_parent_checkpoint "$m2/train_manifest.json" "$m1/best"

m2_beam="$assets/m2_beam"
run_unless_complete "$m2_beam/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py mine-beam-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --model-dir "$m2/best" \
    --rq-dir "$rq" \
    --expected-stage M2 \
    --top-k "$beam_mining_top_k" \
    --query-batch-size 4 \
    --decoder-batch-size 1024 \
    --max-source-length "$rank_max_length" \
    --shard-size 4096 \
    --bf16 \
    --output-dir "$m2_beam"

m3_union="$assets/m3_dense_beam_union"
run_unless_complete "$m3_union/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py union-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --first-candidates-dir "$source_m0_final_mined/candidates" \
    --second-candidates-dir "$m2_beam" \
    --shard-size 4096 \
    --output-dir "$m3_union"

m3_scored="$assets/m3_dense_beam_union_scored"
run_unless_complete "$m3_scored/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py annotate-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --candidates-dir "$m3_union" \
    --teacher-query-embeddings-dir "$teacher_queries" \
    --teacher-skill-embeddings "$teacher_skills" \
    --teacher-skill-manifest "$teacher_skill_manifest" \
    --output-dir "$m3_scored" \
    --batch-size 1024 \
    --shard-size 4096

train_rank_stage() {
  local stage_name="$1"
  local output_dir="$2"
  local parent_dir="$3"
  local candidates_dir="$4"
  local active_prefix="$5"
  run_unless_complete "$output_dir/train_manifest.json" \
    python3 -u scripts/skill_gr_ripor.py train-rank \
      --stage-name "$stage_name" \
      --rows "$train_rows" \
      --skills-jsonl "$skills" \
      --parent-model-dir "$parent_dir" \
      --rq-dir "$rq" \
      --candidates-dir "$candidates_dir" \
      --active-max-prefix "$active_prefix" \
      --prefix-beta 2 \
      --epochs "$rank_epochs" \
      --batch-size "$rank_batch" \
      --microbatch-size "$rank_microbatch" \
      --lr "$rank_lr" \
      --weight-decay 0 \
      --warmup-ratio "$warmup_ratio" \
      --max-grad-norm "$max_grad_norm" \
      --max-source-length "$rank_max_length" \
      --seed "$seed" \
      --bf16 \
      "${rank_objective_args[@]}" \
      "${test_selection_args[@]}" \
      --output-dir "$output_dir"
}

m3_prefix4="$experiment/m3_prefix4"
train_rank_stage M3_prefix4 "$m3_prefix4" "$m2/best" "$m3_scored" 4
assert_selected_checkpoint "$m3_prefix4"
assert_parent_checkpoint "$m3_prefix4/train_manifest.json" "$m2/best"
m3_full="$experiment/m3_full"
train_rank_stage M3_full "$m3_full" "$m3_prefix4/best" "$m3_scored" "$depth"
assert_selected_checkpoint "$m3_full"
assert_parent_checkpoint "$m3_full/train_manifest.json" "$m3_prefix4/best"

m3_beam="$assets/m3_beam"
run_unless_complete "$m3_beam/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py mine-beam-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --model-dir "$m3_full/best" \
    --rq-dir "$rq" \
    --expected-stage M3_full \
    --top-k "$beam_mining_top_k" \
    --query-batch-size 4 \
    --decoder-batch-size 1024 \
    --max-source-length "$rank_max_length" \
    --shard-size 4096 \
    --bf16 \
    --output-dir "$m3_beam"

m4_union="$assets/m4_self_negative_union"
run_unless_complete "$m4_union/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py union-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --first-candidates-dir "$m3_union" \
    --second-candidates-dir "$m3_beam" \
    --shard-size 4096 \
    --output-dir "$m4_union"

m4_scored="$assets/m4_self_negative_union_scored"
run_unless_complete "$m4_scored/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py annotate-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --candidates-dir "$m4_union" \
    --teacher-query-embeddings-dir "$teacher_queries" \
    --teacher-skill-embeddings "$teacher_skills" \
    --teacher-skill-manifest "$teacher_skill_manifest" \
    --output-dir "$m4_scored" \
    --batch-size 1024 \
    --shard-size 4096

m4="$experiment/m4_self_negative"
train_rank_stage M4 "$m4" "$m3_full/best" "$m4_scored" "$depth"
assert_selected_checkpoint "$m4"
assert_parent_checkpoint "$m4/train_manifest.json" "$m3_full/best"

official_eval="$experiment/official_core75_beam50"
run_unless_complete "$official_eval/eval_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py eval \
    --rows "$official_rows" \
    --skills-jsonl "$skills" \
    --model-dir "$m4/best" \
    --rq-dir "$rq" \
    --num-beams 50 \
    --k-values 1,5,10,50 \
    --query-batch-size 1 \
    --decoder-batch-size 1024 \
    --max-source-length "$rank_max_length" \
    --expected-rows-sha256 "$expected_official_sha" \
    --expected-skills-sha256 "$expected_skills_sha" \
    --bf16 \
    --output-dir "$official_eval"

touch "$experiment/COMPLETE"
find "$experiment" -type f \
  ! -name 'structured_decoder.pt' \
  ! -name 'dense_model.pt' \
  ! -name '*.npy' \
  ! -name '*.index' \
  ! -name '*.pt' \
  ! -name 'output_sha256.txt' \
  -print0 | sort -z | xargs -0 sha256sum > "$experiment/output_sha256.txt"
