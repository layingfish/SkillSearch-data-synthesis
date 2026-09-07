#!/usr/bin/env bash
set -euo pipefail

cfs_agent_root="${LETTER_OPQ_CFS_AGENT_ROOT:-/mnt/cfs-agent}"
project_root="${RIPOR_PROJECT_ROOT:-$cfs_agent_root/workspace/chenyufei03/SkillSearch-data-synthesis}"
output_root="${RIPOR_EXPERIMENT:?RIPOR_EXPERIMENT is required}"
model_root="${LETTER_OPQ_MODEL_ROOT:-$cfs_agent_root/dataset/chenyufei03/models/experiments/chenyufei03/SkillSearch-data-synthesis}"
data_root="${LETTER_OPQ_DATA_ROOT:-$cfs_agent_root/dataset/chenyufei03/workspace/chenyufei03/datasets/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs}"
cf20_root="${LETTER_OPQ_CF20_ROOT:-$data_root/letter_opq_m64_semantic_cf20_c256_seed42_20260903}"
baseline_root="${LETTER_OPQ_BASELINE_ROOT:-$model_root/letter_opq_ceiling_runs/rudder_r2_opq_ceiling_m64_20260903_v1}"
candidate_runner="${LETTER_OPQ_CANDIDATE_RUNNER:-$project_root/repro/20260903_letter_opq_cooldown_calibration/run_candidate.sh}"
eval_runner="${LETTER_OPQ_EVAL_RUNNER:-$project_root/repro/20260903_letter_opq_ceiling/run_exact_eval.sh}"
selector="${LETTER_OPQ_SELECTOR:-$project_root/scripts/select_letter_opq_ceiling.py}"
log_root="${LETTER_OPQ_LOG_ROOT:-$model_root/logs/${LETTER_OPQ_RUN_TAG:?LETTER_OPQ_RUN_TAG is required}}"

for required in "$candidate_runner" "$eval_runner" "$selector"; do
  [[ -f "$required" ]] || { echo "missing cooldown runner: $required" >&2; exit 1; }
done
mkdir -p "$output_root/candidates" "$log_root"

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
  local catalog_loss_weight="$3"
  local position_kl_weight="$4"
  local gpu_mask="$5"
  local master_port="$6"
  local experiment="$output_root/candidates/$name"
  local log_path="$log_root/$name.log"
  echo "cooldown_candidate_start name=$name head=$head_type gpu_mask=$gpu_mask port=$master_port"
  env \
    CUDA_VISIBLE_DEVICES="$gpu_mask" \
    LETTER_OPQ_VARIANT="$name" \
    LETTER_OPQ_HEAD_TYPE="$head_type" \
    LETTER_OPQ_CATALOG_LOSS_WEIGHT="$catalog_loss_weight" \
    LETTER_OPQ_POSITION_KL_WEIGHT="$position_kl_weight" \
    LETTER_OPQ_VARIANT_EXPERIMENT="$experiment" \
    LETTER_OPQ_WORLD_SIZE=4 \
    LETTER_OPQ_PER_DEVICE_BATCH_SIZE=4 \
    LETTER_OPQ_MASTER_PORT="$master_port" \
    LETTER_OPQ_CFS_AGENT_ROOT="$cfs_agent_root" \
    LETTER_OPQ_MODEL_ROOT="$model_root" \
    LETTER_OPQ_DATA_ROOT="$data_root" \
    LETTER_OPQ_CF20_ROOT="$cf20_root" \
    LETTER_OPQ_BASELINE_ROOT="$baseline_root" \
    RIPOR_PROJECT_ROOT="$project_root" \
    bash "$candidate_runner" 2>&1 | tee -a "$log_path"
}

run_candidate cf20_rpg_cooldown_e3 \
  rpg_cosine_residual 0 0 0,1,2,3 29801 &
left_pid="$!"
active_pids+=("$left_pid")
run_candidate cf20_rpg_calibrated_e3 \
  rpg_cosine_residual_calibrated 0.1 0.01 4,5,6,7 29802 &
right_pid="$!"
active_pids+=("$right_pid")

set +e
wait "$left_pid"
left_status=$?
wait "$right_pid"
right_status=$?
set -e
active_pids=()
echo "cooldown_candidates_exit left_status=$left_status right_status=$right_status"
if [[ "$left_status" -ne 0 || "$right_status" -ne 0 ]]; then
  exit 1
fi

registry="$output_root/dev_candidate_registry.json"
"${RIPOR_PYTHON_BIN:-python3}" - "$registry" "$output_root" "$baseline_root" "$cf20_root" <<'PY'
import json
import sys
from pathlib import Path

registry = Path(sys.argv[1])
root = Path(sys.argv[2])
baseline_root = Path(sys.argv[3])
cf20_root = Path(sys.argv[4])
specs = [
    ("cf20_rpg_cooldown_e3", "rpg_cosine_residual"),
    ("cf20_rpg_calibrated_e3", "rpg_cosine_residual_calibrated"),
]
candidates = []
for name, head_type in specs:
    experiment = root / "candidates" / name
    completion = json.loads(
        (experiment / "completion.json").read_text(encoding="utf-8")
    )
    if completion.get("status") != "complete" or completion.get("official_evaluation") is not False:
        raise SystemExit(f"candidate completion contract failed: {name}")
    candidates.append(
        {
            "name": name,
            "docid_kind": "semantic_cf20",
            "head_type": head_type,
            "query_epochs": 3,
            "metrics_path": str(experiment / "dev_exact/ranked_metrics.json"),
            "checkpoint_path": str(experiment / "decoder/final/parallel_decoder.pt"),
            "semantic_ids_path": str(cf20_root / "semantic_ids.jsonl"),
        }
    )
candidates.append(
    {
        "name": "baseline_cf20_rpg_e10",
        "docid_kind": "semantic_cf20",
        "head_type": "rpg_cosine_residual",
        "query_epochs": 10,
        "metrics_path": str(
            baseline_root
            / "candidates/semantic_cf20_rpg_e10/dev_exact/ranked_metrics.json"
        ),
        "checkpoint_path": str(
            baseline_root
            / "candidates/semantic_cf20_rpg_e10/decoder/final/parallel_decoder.pt"
        ),
        "semantic_ids_path": str(cf20_root / "semantic_ids.jsonl"),
    }
)
for candidate in candidates:
    for key in ("metrics_path", "checkpoint_path", "semantic_ids_path"):
        if not Path(candidate[key]).is_file():
            raise SystemExit(f"missing candidate artifact: {candidate['name']} {key}")
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
echo "cooldown_dev_winner=$winner_name"

# Official is unreachable until the Dev winner above has been frozen. If the
# original Phase 12 model wins, reuse its immutable prior Official artifact.
official_source="new_winner_only_eval"
if [[ "$winner_name" == "baseline_cf20_rpg_e10" ]]; then
  official_output="$baseline_root/official_winner_exact"
  official_source="reused_frozen_phase12"
  [[ -f "$official_output/ranked_metrics.json" && -f "$official_output/eval_manifest.json" ]] || {
    echo "missing frozen Phase 12 Official artifact" >&2
    exit 1
  }
else
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
    LETTER_OPQ_MASTER_PORT=29831 \
    LETTER_OPQ_EVAL_ROWS="$official_rows" \
    LETTER_OPQ_EVAL_SPLIT=official_core75 \
    LETTER_OPQ_EVAL_SEMANTIC_IDS="$winner_semantic_ids" \
    LETTER_OPQ_EVAL_CHECKPOINT="$winner_checkpoint" \
    LETTER_OPQ_EVAL_OUTPUT="$official_output" \
    LETTER_OPQ_CFS_AGENT_ROOT="$cfs_agent_root" \
    RIPOR_PROJECT_ROOT="$project_root" \
    bash "$eval_runner" 2>&1 | tee -a "$log_root/official_winner_exact.log"
fi

"${RIPOR_PYTHON_BIN:-python3}" - \
  "$output_root/completion.json" "$selection" "$official_output" \
  "$winner_name" "$winner_checkpoint" "$official_source" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

completion_path = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
official_root = Path(sys.argv[3])
winner_name = sys.argv[4]
winner_checkpoint = Path(sys.argv[5])
official_source = sys.argv[6]
metrics_path = official_root / "ranked_metrics.json"
manifest_path = official_root / "eval_manifest.json"
for required in (selection_path, metrics_path, manifest_path, winner_checkpoint):
    if not required.is_file():
        raise SystemExit(f"cooldown run is incomplete: {required}")
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
    "study": "letter_opq_winner_cooldown_position_calibration",
    "winner": winner_name,
    "selection_split": "dev_rows_001",
    "official_scope": "official_core75_winner_only",
    "official_source": official_source,
    "official_macro_recall": {
        f"R@{k}": float(metrics["overall"][f"@{k}"]["macro_recall"])
        for k in (1, 5, 10, 50)
    },
    "artifacts": {
        "selection": {"path": str(selection_path), "sha256": sha256(selection_path)},
        "official_metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
        "official_eval_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "winner_checkpoint": {
            "path": str(winner_checkpoint),
            "sha256": sha256(winner_checkpoint),
        },
    },
}
completion_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

trap - INT TERM HUP
echo "letter_opq_cooldown_complete=$(date -Iseconds) winner=$winner_name"
