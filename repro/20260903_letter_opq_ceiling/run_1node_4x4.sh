#!/usr/bin/env bash
set -euo pipefail

cfs_agent_root="${LETTER_OPQ_CFS_AGENT_ROOT:-/mnt/cfs-agent}"
project_root="${RIPOR_PROJECT_ROOT:-$cfs_agent_root/workspace/chenyufei03/SkillSearch-data-synthesis}"
output_root="${RIPOR_EXPERIMENT:?RIPOR_EXPERIMENT is required}"
model_root="$cfs_agent_root/dataset/chenyufei03/models/experiments/chenyufei03/SkillSearch-data-synthesis"
default_data_root="$cfs_agent_root/dataset/chenyufei03/workspace/chenyufei03/datasets/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs"
data_root="${LETTER_OPQ_DATA_ROOT:-$default_data_root}"
candidate_runner="${LETTER_OPQ_CANDIDATE_RUNNER:-$project_root/repro/20260903_letter_opq_ceiling/run_candidate.sh}"
eval_runner="${LETTER_OPQ_EVAL_RUNNER:-$project_root/repro/20260903_letter_opq_ceiling/run_exact_eval.sh}"
selector="${LETTER_OPQ_SELECTOR:-$project_root/scripts/select_letter_opq_ceiling.py}"
log_root="${LETTER_OPQ_LOG_ROOT:-$model_root/logs/${LETTER_OPQ_RUN_TAG:?LETTER_OPQ_RUN_TAG is required}}"
semantic_root="${LETTER_OPQ_SEMANTIC_ROOT:-$project_root/.tmp/letter_opq_m64_semantic_c256_seed42_20260901}"
cf20_root="${LETTER_OPQ_CF20_ROOT:-$data_root/letter_opq_m64_semantic_cf20_c256_seed42_20260903}"
default_reference_root="$model_root/letter_opq_parallel_runs/rudder_r2_parallel_mtp_4gpu_20260902_v3/opq_m64_parallel_t5base_seed20260626"
reference_root="${LETTER_OPQ_REFERENCE_ROOT:-$default_reference_root}"

for required in "$candidate_runner" "$eval_runner" "$selector"; do
  [[ -f "$required" ]] || { echo "missing ceiling runner: $required" >&2; exit 1; }
done
mkdir -p "$output_root/candidates" "$output_root/reference_semantic_linear_e5" "$log_root"

declare -a active_pids=()
terminate_children() {
  local pid
  for pid in "${active_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  active_pids=()
}
trap terminate_children INT TERM HUP

run_candidate() {
  local name="$1"
  local head_type="$2"
  local docid_kind="$3"
  local docid_root="$4"
  local sid_file_sha="$5"
  local manifest_sha="$6"
  local gpu_mask="$7"
  local master_port="$8"
  local experiment="$output_root/candidates/$name"
  local log_path="$log_root/$name.log"
  echo "ceiling_candidate_start name=$name head=$head_type docids=$docid_kind gpu_mask=$gpu_mask port=$master_port"
  env \
    CUDA_VISIBLE_DEVICES="$gpu_mask" \
    LETTER_OPQ_VARIANT="$name" \
    LETTER_OPQ_HEAD_TYPE="$head_type" \
    LETTER_OPQ_DOCID_KIND="$docid_kind" \
    LETTER_OPQ_DOCID_ROOT="$docid_root" \
    LETTER_OPQ_EXPECTED_SID_FILE_SHA="$sid_file_sha" \
    LETTER_OPQ_EXPECTED_MANIFEST_SHA="$manifest_sha" \
    LETTER_OPQ_VARIANT_EXPERIMENT="$experiment" \
    LETTER_OPQ_WORLD_SIZE=4 \
    LETTER_OPQ_PER_DEVICE_BATCH_SIZE=4 \
    LETTER_OPQ_MASTER_PORT="$master_port" \
    LETTER_OPQ_TEMPERATURE=0.03 \
    LETTER_OPQ_CFS_AGENT_ROOT="$cfs_agent_root" \
    RIPOR_PROJECT_ROOT="$project_root" \
    bash "$candidate_runner" 2>&1 | tee -a "$log_path"
}

run_wave() {
  local wave="$1"
  shift
  if [[ "$#" -ne 16 ]]; then
    echo "wave $wave requires exactly two 8-field candidate specifications" >&2
    exit 2
  fi
  local -a left_spec=("${@:1:8}")
  local -a right_spec=("${@:9:8}")
  echo "ceiling_wave_start wave=$wave"
  run_candidate "${left_spec[@]}" &
  left_pid="$!"
  active_pids+=("$left_pid")
  run_candidate "${right_spec[@]}" &
  right_pid="$!"
  active_pids+=("$right_pid")

  set +e
  wait "$left_pid"
  left_status=$?
  wait "$right_pid"
  right_status=$?
  set -e
  active_pids=()
  echo "ceiling_wave_exit wave=$wave left_status=$left_status right_status=$right_status"
  if [[ "$left_status" -ne 0 || "$right_status" -ne 0 ]]; then
    exit 1
  fi
}

# Wave 1 isolates head design while holding the semantic-only OPQ tokenizer fixed.
run_wave 1 \
  semantic_linear_e10 linear semantic_only "$semantic_root" \
    d059d380a15f6cc9adfb66c0476f0d4ecfc0b40127585d62c0b85c98a21d6134 \
    34c088b72abfa755196830c5b442c2668cb7ed863e4118afab1626aa588e8c8d \
    0,1,2,3 29701 \
  semantic_rpg_e10 rpg_cosine_residual semantic_only "$semantic_root" \
    d059d380a15f6cc9adfb66c0476f0d4ecfc0b40127585d62c0b85c98a21d6134 \
    34c088b72abfa755196830c5b442c2668cb7ed863e4118afab1626aa588e8c8d \
    4,5,6,7 29702

# Wave 2 isolates the same heads with the frozen semantic+CF20 OPQ tokenizer.
run_wave 2 \
  semantic_cf20_linear_e10 linear semantic_cf20 "$cf20_root" \
    8987d93cf89a93e80f643c84fa389c511dff35a883e990ad723c2d9c9fa3f7d1 \
    548d01f677cd60448789f7b62eb32bdd1121418d407262319b8c66a69922e7f0 \
    0,1,2,3 29711 \
  semantic_cf20_rpg_e10 rpg_cosine_residual semantic_cf20 "$cf20_root" \
    8987d93cf89a93e80f643c84fa389c511dff35a883e990ad723c2d9c9fa3f7d1 \
    548d01f677cd60448789f7b62eb32bdd1121418d407262319b8c66a69922e7f0 \
    4,5,6,7 29712

# Add the already-completed 5-epoch semantic-linear model to the Dev candidate
# pool. Its prior Official artifact is deliberately not read here.
reference_checkpoint="$reference_root/decoder/final/parallel_decoder.pt"
reference_dev="$output_root/reference_semantic_linear_e5/dev_exact"
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LETTER_OPQ_WORLD_SIZE=4 \
  LETTER_OPQ_MASTER_PORT=29721 \
  LETTER_OPQ_EVAL_ROWS="$data_root/rows_001.decoder_rows.base.jsonl" \
  LETTER_OPQ_EVAL_SPLIT=dev_rows_001 \
  LETTER_OPQ_EVAL_SEMANTIC_IDS="$semantic_root/semantic_ids.jsonl" \
  LETTER_OPQ_EVAL_CHECKPOINT="$reference_checkpoint" \
  LETTER_OPQ_EVAL_OUTPUT="$reference_dev" \
  LETTER_OPQ_CFS_AGENT_ROOT="$cfs_agent_root" \
  RIPOR_PROJECT_ROOT="$project_root" \
  bash "$eval_runner" 2>&1 | tee -a "$log_root/reference_semantic_linear_e5_dev.log"

registry="$output_root/dev_candidate_registry.json"
"${RIPOR_PYTHON_BIN:-python3}" - "$registry" "$output_root" "$semantic_root" "$cf20_root" "$reference_root" <<'PY'
import json
import sys
from pathlib import Path

registry = Path(sys.argv[1])
root = Path(sys.argv[2])
semantic_root = Path(sys.argv[3])
cf20_root = Path(sys.argv[4])
reference_root = Path(sys.argv[5])
specs = [
    ("semantic_linear_e10", "semantic_only", "linear", semantic_root),
    ("semantic_cf20_linear_e10", "semantic_cf20", "linear", cf20_root),
    ("semantic_rpg_e10", "semantic_only", "rpg_cosine_residual", semantic_root),
    ("semantic_cf20_rpg_e10", "semantic_cf20", "rpg_cosine_residual", cf20_root),
]
candidates = []
for name, docid_kind, head_type, docid_root in specs:
    experiment = root / "candidates" / name
    completion_path = experiment / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("official_evaluation") is not False:
        raise SystemExit(f"candidate completion contract failed: {name}")
    candidates.append(
        {
            "name": name,
            "docid_kind": docid_kind,
            "head_type": head_type,
            "query_epochs": 10,
            "metrics_path": str(experiment / "dev_exact/ranked_metrics.json"),
            "checkpoint_path": str(experiment / "decoder/final/parallel_decoder.pt"),
            "semantic_ids_path": str(docid_root / "semantic_ids.jsonl"),
        }
    )
candidates.append(
    {
        "name": "reference_semantic_linear_e5",
        "docid_kind": "semantic_only",
        "head_type": "linear",
        "query_epochs": 5,
        "metrics_path": str(root / "reference_semantic_linear_e5/dev_exact/ranked_metrics.json"),
        "checkpoint_path": str(reference_root / "decoder/final/parallel_decoder.pt"),
        "semantic_ids_path": str(semantic_root / "semantic_ids.jsonl"),
    }
)
registry.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"candidate_registry": str(registry), "count": len(candidates)}, sort_keys=True))
PY

selection="$output_root/dev_selection.json"
"${RIPOR_PYTHON_BIN:-python3}" "$selector" --registry "$registry" --output "$selection"

mapfile -t winner_fields < <("${RIPOR_PYTHON_BIN:-python3}" - "$selection" <<'PY'
import json
import sys
from pathlib import Path

winner = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["winner"]
for key in ("name", "checkpoint_path", "semantic_ids_path"):
    value = str(winner[key])
    if "\n" in value:
        raise SystemExit(f"newline in winner field: {key}")
    print(value)
PY
)
[[ "${#winner_fields[@]}" -eq 3 ]] || { echo "invalid winner selection output" >&2; exit 1; }
winner_name="${winner_fields[0]}"
winner_checkpoint="${winner_fields[1]}"
winner_semantic_ids="${winner_fields[2]}"
echo "ceiling_dev_winner=$winner_name"

# Official becomes reachable only after the Dev winner is frozen above.
official_rows="$data_root/rows_003.skillsbench_core75.rows.jsonl"
actual_official_sha="$(sha256sum "$official_rows" | awk '{print $1}')"
[[ "$actual_official_sha" == "4adcd50e359e806d824ffa6f5ee29b4573321033e47ffa2a31780e65ff3cb8ba" ]] || {
  echo "Official Core75 SHA256 mismatch: $actual_official_sha" >&2
  exit 1
}
official_output="$output_root/official_winner_exact"
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LETTER_OPQ_WORLD_SIZE=4 \
  LETTER_OPQ_MASTER_PORT=29731 \
  LETTER_OPQ_EVAL_ROWS="$official_rows" \
  LETTER_OPQ_EVAL_SPLIT=official_core75 \
  LETTER_OPQ_EVAL_SEMANTIC_IDS="$winner_semantic_ids" \
  LETTER_OPQ_EVAL_CHECKPOINT="$winner_checkpoint" \
  LETTER_OPQ_EVAL_OUTPUT="$official_output" \
  LETTER_OPQ_CFS_AGENT_ROOT="$cfs_agent_root" \
  RIPOR_PROJECT_ROOT="$project_root" \
  bash "$eval_runner" 2>&1 | tee -a "$log_root/official_winner_exact.log"

"${RIPOR_PYTHON_BIN:-python3}" - "$output_root/completion.json" "$selection" "$official_output" "$winner_name" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

completion_path = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
official_root = Path(sys.argv[3])
winner_name = sys.argv[4]
metrics_path = official_root / "ranked_metrics.json"
manifest_path = official_root / "eval_manifest.json"
for required in (selection_path, metrics_path, manifest_path):
    if not required.is_file():
        raise SystemExit(f"ceiling run is incomplete: {required}")
selection = json.loads(selection_path.read_text(encoding="utf-8"))
if selection["winner"]["name"] != winner_name:
    raise SystemExit("winner changed between selection and Official evaluation")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "status": "complete",
    "study": "letter_opq_base_pipeline_ceiling",
    "winner": winner_name,
    "selection_split": "dev_rows_001",
    "official_scope": "official_core75_winner_only",
    "official_macro_recall": {
        f"R@{k}": float(metrics["overall"][f"@{k}"]["macro_recall"])
        for k in (1, 5, 10, 50)
    },
    "artifacts": {
        "selection": {"path": str(selection_path), "sha256": sha256(selection_path)},
        "official_metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
        "official_eval_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "winner_checkpoint": {
            "path": selection["winner"]["checkpoint_path"],
            "sha256": sha256(Path(selection["winner"]["checkpoint_path"])),
        },
    },
}
completion_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

trap - INT TERM HUP
echo "letter_opq_ceiling_complete=$(date -Iseconds) winner=$winner_name"
