#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKILLGR_CODE_ROOT:-/workspace/SkillSearch-data-synthesis}"
PHYSICAL_OUTPUT_ROOT="${SKILLGR_PHYSICAL_OUTPUT_ROOT:-/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260808_v3}"
ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh

if [[ -e "${PHYSICAL_OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite experiment root: ${PHYSICAL_OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${PHYSICAL_OUTPUT_ROOT}/logs"
exec > >(tee -a "${PHYSICAL_OUTPUT_ROOT}/logs/node-8gpu.log") 2>&1

bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1
git config --global --add safe.directory "${ROOT_DIR}"
git config --global --add safe.directory "$(readlink -f "${ROOT_DIR}")"
cd "${ROOT_DIR}"

CONFIG_PATH="${SKILLGR_CONFIG:-configs/legacy_catalog_final_effect_20260808.env}"
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH}"
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "catalog sweep requires a clean immutable checkout" >&2
  git status --short
  exit 1
fi
git rev-parse HEAD

set -a
source "${CONFIG_PATH}"
set +a
OUTPUT_ALIAS_PROBE="${PHYSICAL_OUTPUT_ROOT}/.output-root-alias-probe"
touch "${OUTPUT_ALIAS_PROBE}"
if [[ ! -f "${SKILLGR_OUTPUT_ROOT}/.output-root-alias-probe" ]]; then
  echo "logical output root does not point to the physical experiment root" >&2
  exit 1
fi
unlink "${OUTPUT_ALIAS_PROBE}"
LEGACY_SOURCE_ROOT="$(mktemp -d /tmp/skillgr-legacy-15f6c51.XXXXXX)"
git archive "${SKILLGR_LEGACY_COMMIT}" | tar -xf - -C "${LEGACY_SOURCE_ROOT}"
export SKILLGR_LEGACY_SOURCE_ROOT="${LEGACY_SOURCE_ROOT}"

observed_pipeline_sha256="$(sha256sum "${LEGACY_SOURCE_ROOT}/scripts/skill_gr_pipeline.py" | awk '{print $1}')"
if [[ "${observed_pipeline_sha256}" != "${SKILLGR_LEGACY_PIPELINE_SHA256}" ]]; then
  echo "archived pipeline SHA-256 mismatch" >&2
  exit 1
fi
python3 -m py_compile \
  "${LEGACY_SOURCE_ROOT}/scripts/skill_gr_pipeline.py" \
  scripts/skill_gr_pipeline.py
python3 -m unittest discover -s tests -p test_legacy_baseline_parity_probe.py

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ -n "${SKILLGR_ARCHIVED_RUN:-}" ]]; then
  ARCHIVED_PARITY_DIR="${SKILLGR_OUTPUT_ROOT}/archived_base_final_parity"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/skill_gr_pipeline.py eval-single-sid-decoder \
    --rows "${SKILLGR_OFFICIAL_ROWS}" \
    --model-dir "${SKILLGR_ARCHIVED_RUN}/final" \
    --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
    --output-dir "${ARCHIVED_PARITY_DIR}" \
    --split-name official_core75 \
    --max-source-length "${SKILLGR_MAX_SOURCE_LENGTH}" \
    --num-beams "${SKILLGR_NUM_BEAMS}" \
    --rank-aggregation beam_order \
    --decoder-head ce \
    --score-transform logprob \
    --k-values 1,10,50 \
    --log-every 25

  for subset in all single multi; do
    python3 scripts/skill_gr_pipeline.py rescore-ranked \
      --predictions "${ARCHIVED_PARITY_DIR}/predictions.jsonl" \
      --subset "${subset}" \
      --k-values 1,10,50 \
      --output "${ARCHIVED_PARITY_DIR}/ranked_metrics_${subset}.json"
  done

  python3 - \
    "${SKILLGR_ARCHIVED_OFFICIAL}" \
    "${ARCHIVED_PARITY_DIR}/predictions.jsonl" \
    "${ARCHIVED_PARITY_DIR}/parity.json" <<'PY'
import json
import sys
from pathlib import Path

archive_path = Path(sys.argv[1])
current_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

archive = load(archive_path)
current = load(current_path)
if len(archive) != 75 or len(current) != 75:
    raise SystemExit(f"archived parity row mismatch: archive={len(archive)}, current={len(current)}")
exact_rows = sum(
    left.get("ranked_skill_ids") == right.get("ranked_skill_ids")
    for left, right in zip(archive, current)
)
result = {"rows": len(current), "top50_exact_rows": exact_rows}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if exact_rows != len(current):
    raise SystemExit(f"archived base evaluation parity failed: {result}")
print(json.dumps(result, sort_keys=True))
PY
fi

declare -a PIDS=()
declare -A LABEL_BY_PID=()

terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
}
trap 'terminate_children; exit 130' INT TERM

launch_arm() {
  local gpu="$1"
  local catalog_epochs="$2"
  local seed="$3"
  local label="c${catalog_epochs}-seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    bash scripts/run_legacy_catalog_final_effect_20260808.sh "${catalog_epochs}" "${seed}" &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}"
  echo "launched ${label} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_arm 0 0 20260626
launch_arm 1 3 20260626
launch_arm 2 6 20260626
launch_arm 3 12 20260626
launch_arm 4 0 20260627
launch_arm 5 3 20260627
launch_arm 6 6 20260627
launch_arm 7 12 20260627

remaining=${#PIDS[@]}
failures=0
while (( remaining > 0 )); do
  finished_pid=""
  if wait -n -p finished_pid; then
    echo "completed ${LABEL_BY_PID[${finished_pid}]}"
  else
    status=$?
    echo "failed ${LABEL_BY_PID[${finished_pid}]:-${finished_pid:-unknown}} with status ${status}" >&2
    failures=$((failures + 1))
  fi
  if [[ -n "${finished_pid}" ]]; then
    unset 'LABEL_BY_PID[$finished_pid]'
  fi
  remaining=$((remaining - 1))
done

trap - INT TERM
if (( failures > 0 )); then
  exit 1
fi

python3 - "${SKILLGR_OUTPUT_ROOT}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = (20260626, 20260627)
epochs = (0, 3, 6, 12)
arms = []
for catalog_epochs in epochs:
    for seed in seeds:
        path = root / f"legacy-c{catalog_epochs}-q5-seed{seed}" / "result_summary.json"
        arms.append(json.loads(path.read_text(encoding="utf-8")))

averages = {}
for catalog_epochs in epochs:
    selected = [row for row in arms if row["catalog_epochs"] == catalog_epochs]
    averages[str(catalog_epochs)] = {
        subset: {
            f"recall@{cutoff}": statistics.fmean(
                float(row[subset][f"@{cutoff}"]["recall"])
                for row in selected
            )
            for cutoff in (1, 10, 50)
        }
        for subset in ("overall", "single", "multi")
    }

summary = {
    "artifact_format": "legacy_catalog_final_effect_sweep_v1",
    "evaluation": "Official core75, beam=50, beam_order, CE log-prob",
    "source_commit": arms[0]["source_commit"],
    "arms": arms,
    "mean_across_seeds": averages,
}
(root / "sweep_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY

touch "${SKILLGR_OUTPUT_ROOT}/sweep.done"
