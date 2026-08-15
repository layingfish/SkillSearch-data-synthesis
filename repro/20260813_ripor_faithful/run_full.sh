#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

asset_root="outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713"
inputs="$asset_root/canonical_alias_core75_learned_d4_inputs"
train_rows="$inputs/rows_000.decoder_rows.base_plus_task5_gt2.jsonl"
official_rows="$inputs/rows_003.skillsbench_core75.rows.jsonl"
skills="$inputs/canonical_skills.jsonl"

experiment="experiments/ripor_faithful_cat3query5_20260813"
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

base_model="t5-base"
seed=42
depth=8
codebook_size=256
teacher_top_k=100
beam_mining_top_k=100

# Fixed before Official evaluation.  These values preserve the complete RIPOR
# phase sequence while adapting the original eight-A100 schedule to one GPU and
# the fixed 124,175 query-positive pairs.
m0_epochs=5
m0_batch=16
m0_microbatch=8
m0_lr="1e-4"
m0_max_length=2048
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

check_sha "$expected_train_sha" "$train_rows"
check_sha "$expected_official_sha" "$official_rows"
check_sha "$expected_skills_sha" "$skills"
check_sha "$expected_teacher_skills_sha" "$teacher_skills"
check_sha "$expected_decoder_start_sha" <(base64 --decode < "$decoder_start")

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$assets" "$experiment/source_snapshot"
cp scripts/skill_gr_ripor.py "$experiment/source_snapshot/skill_gr_ripor.py"
cp tests/test_skill_gr_ripor.py "$experiment/source_snapshot/test_skill_gr_ripor.py"
cp repro/20260813_ripor_faithful/README.md "$experiment/source_snapshot/README.md"
cp repro/20260813_ripor_faithful/FAITHFULNESS_AUDIT.md \
  "$experiment/source_snapshot/FAITHFULNESS_AUDIT.md"
cp "${BASH_SOURCE[0]}" "$experiment/source_snapshot/run_full.sh"

python3 -m pytest -q \
  tests/test_skill_gr_ripor.py \
  tests/test_skill_gr_dgr.py \
  tests/test_skill_gr_pipeline.py \
  | tee "$experiment/preflight_tests.log"

run_unless_complete "$experiment/input_length_audit.json" \
  python3 -u scripts/skill_gr_ripor.py audit-inputs \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --base-model "$base_model" \
    --query-limit "$rank_max_length" \
    --skill-limit "$m0_max_length" \
    --batch-size 512 \
    --output "$experiment/input_length_audit.json"

bm25_unscored="$assets/bm25_unscored"
run_unless_complete "$bm25_unscored/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py build-bm25-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --output-dir "$bm25_unscored" \
    --top-k "$teacher_top_k" \
    --k1 1.5 \
    --b 0.75 \
    --shard-size 4096 \
    --expected-rows-sha256 "$expected_train_sha" \
    --expected-skills-sha256 "$expected_skills_sha"

bm25_scored="$assets/bm25_scored"
run_unless_complete "$bm25_scored/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py annotate-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --candidates-dir "$bm25_unscored" \
    --teacher-query-embeddings-dir "$teacher_queries" \
    --teacher-skill-embeddings "$teacher_skills" \
    --teacher-skill-manifest "$teacher_skill_manifest" \
    --output-dir "$bm25_scored" \
    --batch-size 2048 \
    --shard-size 4096

m0_bm25="$experiment/m0_bm25"
run_unless_complete "$m0_bm25/train_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py train-m0 \
    --stage-name M0_bm25 \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --candidates-dir "$bm25_scored" \
    --base-model "$base_model" \
    --decoder-start-token "$decoder_start" \
    --epochs "$m0_epochs" \
    --batch-size "$m0_batch" \
    --microbatch-size "$m0_microbatch" \
    --lr "$m0_lr" \
    --weight-decay 0 \
    --warmup-ratio "$warmup_ratio" \
    --max-grad-norm "$max_grad_norm" \
    --max-source-length "$m0_max_length" \
    --seed "$seed" \
    --bf16 \
    --output-dir "$m0_bm25"

m0_stage1_mined="$assets/m0_stage1_mined"
run_unless_complete "$m0_stage1_mined/candidates/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py mine-dense-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --model-dir "$m0_bm25/final" \
    --decoder-start-token "$decoder_start" \
    --top-k "$teacher_top_k" \
    --query-batch-size 32 \
    --skill-batch-size 16 \
    --max-source-length "$m0_max_length" \
    --shard-size 4096 \
    --bf16 \
    --output-dir "$m0_stage1_mined"

m0_stage1_scored="$assets/m0_stage1_scored"
run_unless_complete "$m0_stage1_scored/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py annotate-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --candidates-dir "$m0_stage1_mined/candidates" \
    --teacher-query-embeddings-dir "$teacher_queries" \
    --teacher-skill-embeddings "$teacher_skills" \
    --teacher-skill-manifest "$teacher_skill_manifest" \
    --output-dir "$m0_stage1_scored" \
    --batch-size 2048 \
    --shard-size 4096

m0_dense="$experiment/m0_dense"
run_unless_complete "$m0_dense/train_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py train-m0 \
    --stage-name M0_dense \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --candidates-dir "$m0_stage1_scored" \
    --base-model "$base_model" \
    --decoder-start-token "$decoder_start" \
    --parent-model-dir "$m0_bm25/final" \
    --epochs "$m0_epochs" \
    --batch-size "$m0_batch" \
    --microbatch-size "$m0_microbatch" \
    --lr "$m0_lr" \
    --weight-decay 0 \
    --warmup-ratio "$warmup_ratio" \
    --max-grad-norm "$max_grad_norm" \
    --max-source-length "$m0_max_length" \
    --seed "$seed" \
    --bf16 \
    --output-dir "$m0_dense"

m0_final_mined="$assets/m0_final_mined"
run_unless_complete "$m0_final_mined/candidates/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py mine-dense-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --model-dir "$m0_dense/final" \
    --decoder-start-token "$decoder_start" \
    --top-k "$teacher_top_k" \
    --query-batch-size 32 \
    --skill-batch-size 16 \
    --max-source-length "$m0_max_length" \
    --shard-size 4096 \
    --bf16 \
    --output-dir "$m0_final_mined"

dense_rank_scored="$assets/dense_rank_scored"
run_unless_complete "$dense_rank_scored/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py annotate-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --candidates-dir "$m0_final_mined/candidates" \
    --teacher-query-embeddings-dir "$teacher_queries" \
    --teacher-skill-embeddings "$teacher_skills" \
    --teacher-skill-manifest "$teacher_skill_manifest" \
    --output-dir "$dense_rank_scored" \
    --batch-size 2048 \
    --shard-size 4096

rq="$experiment/rq_depth8_vocab256"
run_unless_complete "$rq/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py build-rq \
    --embeddings "$m0_final_mined/m0_skill_embeddings.npy" \
    --embeddings-manifest "$m0_final_mined/m0_skill_embeddings.manifest.json" \
    --skills-jsonl "$skills" \
    --depth "$depth" \
    --codebook-size "$codebook_size" \
    --seed "$seed" \
    --faiss-threads 32 \
    --output-dir "$rq"

m1="$experiment/m1_seq2seq"
run_unless_complete "$m1/train_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py train-m1 \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --base-model "$base_model" \
    --m0-model-dir "$m0_dense/final" \
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
    --output-dir "$m1"

m2="$experiment/m2_initial_rank"
run_unless_complete "$m2/train_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py train-rank \
    --stage-name M2 \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --parent-model-dir "$m1/final" \
    --rq-dir "$rq" \
    --candidates-dir "$dense_rank_scored" \
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
    --output-dir "$m2"

m2_beam="$assets/m2_beam"
run_unless_complete "$m2_beam/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py mine-beam-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --model-dir "$m2/final" \
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
    --first-candidates-dir "$m0_final_mined/candidates" \
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

m3_prefix4="$experiment/m3_prefix4"
run_unless_complete "$m3_prefix4/train_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py train-rank \
    --stage-name M3_prefix4 \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --parent-model-dir "$m2/final" \
    --rq-dir "$rq" \
    --candidates-dir "$m3_scored" \
    --active-max-prefix 4 \
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
    --output-dir "$m3_prefix4"

m3_full="$experiment/m3_full"
run_unless_complete "$m3_full/train_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py train-rank \
    --stage-name M3_full \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --parent-model-dir "$m3_prefix4/final" \
    --rq-dir "$rq" \
    --candidates-dir "$m3_scored" \
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
    --output-dir "$m3_full"

m3_beam="$assets/m3_beam"
run_unless_complete "$m3_beam/manifest.json" \
  python3 -u scripts/skill_gr_ripor.py mine-beam-candidates \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --model-dir "$m3_full/final" \
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
run_unless_complete "$m4/train_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py train-rank \
    --stage-name M4 \
    --rows "$train_rows" \
    --skills-jsonl "$skills" \
    --parent-model-dir "$m3_full/final" \
    --rq-dir "$rq" \
    --candidates-dir "$m4_scored" \
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
    --output-dir "$m4"

official_eval="$experiment/official_core75_beam50"
run_unless_complete "$official_eval/eval_manifest.json" \
  python3 -u scripts/skill_gr_ripor.py eval \
    --rows "$official_rows" \
    --skills-jsonl "$skills" \
    --model-dir "$m4/final" \
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
