#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(pwd)"
PHYSICAL_REPO=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis
PHYSICAL_OUTPUT_BASE=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis
PY310_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1
PY310_RUNTIME_ARCHIVE="${PY310_RUNTIME_ROOT}/runtime.tar.zst"
PY310_RUNTIME_ARCHIVE_SHA256=6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72
PY311_RUNTIME_ROOT=/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1
PY311_RUNTIME_ARCHIVE="${PY311_RUNTIME_ROOT}/site.tar.zst"
PY311_RUNTIME_ARCHIVE_SHA256=c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2
PY311_COMPAT_ARCHIVE="${PY311_RUNTIME_ROOT}/compat-overlay.tar.zst"
PY311_COMPAT_ARCHIVE_SHA256=a6b2f56bb9b4496caa4be225b1c2630270cef1b9a23f82013eea72632bda6b20
BASE_MODEL=/models_rw/SkillSearch-data-synthesis/base_models/t5-base-a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
LEGACY_COMMIT=15f6c51131f3b6f3e8580aa4564b17cbe13a9790
LEGACY_PIPELINE_SHA256=5d60624b704e3563f1893b021e15c2c6da3cdb28a18a1ac3018ca3884de04f67
OUTPUT_ROOT="${PHYSICAL_OUTPUT_BASE}/experiments/legacy_python_abi_probe_20260809_v1"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite Python-ABI output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/arms"
exec > >(tee "${OUTPUT_ROOT}/logs/node.log") 2>&1

if [[ ! -d "${PHYSICAL_REPO}/.git" ]]; then
  echo "physical repository is unavailable: ${PHYSICAL_REPO}" >&2
  exit 1
fi
git -C "${PHYSICAL_REPO}" cat-file -e "${LEGACY_COMMIT}^{commit}"
echo "launch_commit=${SKILLGR_LAUNCH_COMMIT:-unknown}"

TEMP_ROOT="$(mktemp -d /tmp/skillgr-python-abi.XXXXXX)"
cleanup() {
  case "${TEMP_ROOT}" in
    /tmp/skillgr-python-abi.*) rm -rf -- "${TEMP_ROOT}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2 ;;
  esac
}
trap cleanup EXIT

check_sha256() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || {
    echo "SHA-256 mismatch for ${path}: expected=${expected}, observed=${observed}" >&2
    exit 1
  }
}

check_sha256 "${PY310_RUNTIME_ARCHIVE}" "${PY310_RUNTIME_ARCHIVE_SHA256}"
check_sha256 "${PY311_RUNTIME_ARCHIVE}" "${PY311_RUNTIME_ARCHIVE_SHA256}"
check_sha256 "${PY311_COMPAT_ARCHIVE}" "${PY311_COMPAT_ARCHIVE_SHA256}"

mkdir -p "${TEMP_ROOT}/py310" "${TEMP_ROOT}/py311" "${TEMP_ROOT}/compat311"
tar --zstd -xf "${PY310_RUNTIME_ARCHIVE}" -C "${TEMP_ROOT}/py310"
tar --zstd -xf "${PY311_RUNTIME_ARCHIVE}" -C "${TEMP_ROOT}/py311"
tar --zstd -xf "${PY311_COMPAT_ARCHIVE}" -C "${TEMP_ROOT}/compat311"

PY310_PYTHON="${TEMP_ROOT}/py310/python/bin/python3"
PY310_SITE="${TEMP_ROOT}/py310/site"
PY311_PYTHON=python3
PY311_SITE="${TEMP_ROOT}/py311/site"
PY311_COMPAT_SITE="${TEMP_ROOT}/compat311"
NVIDIA_LIBRARY_PATHS="$(find "${PY311_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
if [[ -z "${NVIDIA_LIBRARY_PATHS}" ]]; then
  echo "fixed NVIDIA library directories are missing from ${PY311_SITE}" >&2
  exit 1
fi
FIXED_LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

runtime_preflight() {
  local label="$1" python_executable="$2" python_path="$3" expected_python_prefix="$4"
  local expected_networkx="$5" expected_torchvision="$6"
  EXPECTED_LABEL="${label}" EXPECTED_PYTHON_PREFIX="${expected_python_prefix}" \
    EXPECTED_NETWORKX="${expected_networkx}" EXPECTED_TORCHVISION="${expected_torchvision}" \
    EXPECTED_NVIDIA_ROOT="${PY311_SITE}/nvidia" \
    LD_LIBRARY_PATH="${FIXED_LD_LIBRARY_PATH}" PYTHONPATH="${python_path}" \
    "${python_executable}" - <<'PY'
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import huggingface_hub
import networkx
import safetensors
import tokenizers
import torch
import torchvision
import transformers
from transformers import T5ForConditionalGeneration

expected = {
    "python_prefix": os.environ["EXPECTED_PYTHON_PREFIX"],
    "torch": "2.10.0+cu128",
    "cuda": "12.8",
    "cudnn": 91002,
    "transformers": "5.3.0",
    "tokenizers": "0.22.2",
    "huggingface_hub": "1.24.0",
    "safetensors": "0.8.0",
    "networkx": os.environ["EXPECTED_NETWORKX"],
    "torchvision": os.environ["EXPECTED_TORCHVISION"],
    "cublas_package": "12.8.4.1",
    "cuda_runtime_package": "12.8.90",
    "cudnn_package": "9.10.2.21",
    "gpu_count": 8,
}
actual = {
    "python_prefix": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "safetensors": safetensors.__version__,
    "networkx": networkx.__version__,
    "torchvision": torchvision.__version__,
    "cublas_package": importlib.metadata.version("nvidia-cublas-cu12"),
    "cuda_runtime_package": importlib.metadata.version("nvidia-cuda-runtime-cu12"),
    "cudnn_package": importlib.metadata.version("nvidia-cudnn-cu12"),
    "gpu_count": torch.cuda.device_count(),
}
matrix = torch.ones((8, 8), dtype=torch.bfloat16, device="cuda")
torch.matmul(matrix, matrix)
torch.cuda.synchronize()
library_names = ("libcublas.so", "libcublasLt.so", "libcudnn.so")
loaded_cuda_libraries = sorted({
    fields[-1]
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    if len(fields := line.split()) >= 6
    and any(name in fields[-1] for name in library_names)
})
expected_nvidia_root = Path(os.environ["EXPECTED_NVIDIA_ROOT"]).resolve()
for library_name in library_names:
    matching = [Path(path).resolve() for path in loaded_cuda_libraries if library_name in path]
    if not matching or any(not path.is_relative_to(expected_nvidia_root) for path in matching):
        raise SystemExit(
            f"{os.environ['EXPECTED_LABEL']} loaded {library_name} outside fixed runtime: "
            f"expected_root={expected_nvidia_root}, loaded={matching}"
        )
python_version = actual.pop("python_prefix")
expected_python = expected.pop("python_prefix")
if not python_version.startswith(expected_python) or actual != expected:
    raise SystemExit(
        f"{os.environ['EXPECTED_LABEL']} runtime mismatch: "
        f"expected_python={expected_python}, actual_python={python_version}, "
        f"expected={expected}, actual={actual}"
    )
print(json.dumps({
    "label": os.environ["EXPECTED_LABEL"],
    "python": python_version,
    **actual,
    "loaded_cuda_libraries": loaded_cuda_libraries,
    "t5_class": T5ForConditionalGeneration.__name__,
}, sort_keys=True))
PY
}

runtime_preflight \
  cp310 "${PY310_PYTHON}" "${PY310_SITE}:${PY311_SITE}" 3.10.12 3.4.2 0.25.0+cu128
runtime_preflight \
  cp311 "${PY311_PYTHON}" "${PY311_COMPAT_SITE}:${PY311_SITE}" 3.11 3.6.1 0.25.0+cu128

MODEL_SAFE="${TEMP_ROOT}/model-safetensors"
mkdir -p "${MODEL_SAFE}"
for filename in config.json generation_config.json tokenizer.json spiece.model model.safetensors; do
  ln -s "${BASE_MODEL}/${filename}" "${MODEL_SAFE}/${filename}"
done

LEGACY_SOURCE="${TEMP_ROOT}/legacy-source"
mkdir -p "${LEGACY_SOURCE}"
git -C "${PHYSICAL_REPO}" archive --format=tar "${LEGACY_COMMIT}" | tar -xf - -C "${LEGACY_SOURCE}"
observed_pipeline_sha256="$(sha256sum "${LEGACY_SOURCE}/scripts/skill_gr_pipeline.py" | awk '{print $1}')"
if [[ "${observed_pipeline_sha256}" != "${LEGACY_PIPELINE_SHA256}" ]]; then
  echo "legacy pipeline mismatch: expected=${LEGACY_PIPELINE_SHA256}, observed=${observed_pipeline_sha256}" >&2
  exit 1
fi

set -a
export SKILLGR_EXACT_MODEL="${MODEL_SAFE}"
source "${CODE_ROOT}/configs/legacy_catalog_base_exact_runtime_20260808.env"
set +a
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

for required in "${SKILLGR_ROWS}" "${SKILLGR_DEV_ROWS}" "${SKILLGR_SKILLS}" "${SKILLGR_SEMANTIC_IDS}" "${BASE_MODEL}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done
check_sha256 "${SKILLGR_ROWS}" "${SKILLGR_ROWS_SHA256}"
check_sha256 "${SKILLGR_DEV_ROWS}" "${SKILLGR_DEV_ROWS_SHA256}"
check_sha256 "${SKILLGR_SKILLS}" "${SKILLGR_SKILLS_SHA256}"
check_sha256 "${SKILLGR_SEMANTIC_IDS}" "${SKILLGR_SEMANTIC_IDS_SHA256}"

"${PY310_PYTHON}" -m py_compile "${CODE_ROOT}/scripts/legacy_runtime_matrix_probe.py"
"${PY311_PYTHON}" -m py_compile "${CODE_ROOT}/scripts/legacy_runtime_matrix_probe.py"

declare -a PIDS=()
declare -A LABEL_BY_PID=()
terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait || true
}
trap 'terminate_children; cleanup; exit 130' INT TERM

launch_arm() {
  local gpu="$1" label="$2" python_abi="$3" python_executable="$4" python_path="$5"
  local output="${OUTPUT_ROOT}/arms/${label}"
  (
    exec > >(tee "${OUTPUT_ROOT}/logs/${label}.log") 2>&1
    CUDA_VISIBLE_DEVICES="${gpu}" LD_LIBRARY_PATH="${FIXED_LD_LIBRARY_PATH}" \
      PYTHONPATH="${CODE_ROOT}:${python_path}" \
      "${python_executable}" "${CODE_ROOT}/scripts/legacy_runtime_matrix_probe.py" \
        --arm "${label}" --legacy-source "${LEGACY_SOURCE}" --legacy-commit "${LEGACY_COMMIT}" \
        --rows "${SKILLGR_ROWS}" --dev-rows "${SKILLGR_DEV_ROWS}" \
        --skills-jsonl "${SKILLGR_SKILLS}" --semantic-ids "${SKILLGR_SEMANTIC_IDS}" \
        --model "${MODEL_SAFE}" --output-dir "${output}" --steps 50 --seed 20260626 \
        --precision bf16 --bf16-reduction auto --tf32-matmul auto --tf32-cudnn auto \
        --float32-matmul-precision auto --foreach auto --preferred-blas auto \
        --deterministic-algorithms auto --cublas-workspace-config auto \
        --cuda-module-loading auto
    echo "arm_python_abi=${python_abi}"
  ) &
  local pid=$!
  PIDS+=("${pid}")
  LABEL_BY_PID["${pid}"]="${label}"
  echo "launched ${label} on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pid}"
}

launch_arm 0 cp310_r0 cp310 "${PY310_PYTHON}" "${PY310_SITE}:${PY311_SITE}"
launch_arm 1 cp310_r1 cp310 "${PY310_PYTHON}" "${PY310_SITE}:${PY311_SITE}"
launch_arm 2 cp310_r2 cp310 "${PY310_PYTHON}" "${PY310_SITE}:${PY311_SITE}"
launch_arm 3 cp310_r3 cp310 "${PY310_PYTHON}" "${PY310_SITE}:${PY311_SITE}"
launch_arm 4 cp311_r0 cp311 "${PY311_PYTHON}" "${PY311_COMPAT_SITE}:${PY311_SITE}"
launch_arm 5 cp311_r1 cp311 "${PY311_PYTHON}" "${PY311_COMPAT_SITE}:${PY311_SITE}"
launch_arm 6 cp311_r2 cp311 "${PY311_PYTHON}" "${PY311_COMPAT_SITE}:${PY311_SITE}"
launch_arm 7 cp311_r3 cp311 "${PY311_PYTHON}" "${PY311_COMPAT_SITE}:${PY311_SITE}"

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
  if [[ -n "${finished_pid}" ]]; then unset 'LABEL_BY_PID[$finished_pid]'; fi
  remaining=$((remaining - 1))
done

trap - INT TERM
python3 - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = []
for path in sorted((root / "arms").glob("*/runtime_probe.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    arm = payload["arm"]
    python_abi = arm.split("_", 1)[0]
    arms.append({
        "arm": arm,
        "python_abi": python_abi,
        "runtime": payload["runtime"],
        "condition": payload["condition"],
        "initialization": payload["initialization"],
        "first_catalog_batch": payload["first_catalog_batch"],
        "losses": payload["losses"],
    })
if len(arms) != 8:
    raise SystemExit(f"expected 8 completed Python-ABI arms, found {len(arms)}")

expected_batch = "721a0d93aaef2f9d0104adae8501fcfabcefcd6bd9f23c1e4a1aae701710d443"
bad_batch = [a["arm"] for a in arms if a["first_catalog_batch"]["row_ids_sha256"] != expected_batch]
if bad_batch:
    raise SystemExit(f"catalog batch-order mismatch in arms: {bad_batch}")

conditions = {}
for python_abi in ("cp310", "cp311"):
    replicas = [arm for arm in arms if arm["python_abi"] == python_abi]
    if len(replicas) != 4:
        raise SystemExit(f"expected 4 {python_abi} replicas, found {len(replicas)}")
    exact_fields = ["initialization", "first_catalog_batch", "losses"]
    mismatched = [
        field
        for field in exact_fields
        if len({json.dumps(replica[field], sort_keys=True) for replica in replicas}) != 1
    ]
    if mismatched:
        raise SystemExit(f"{python_abi} replicas disagree in fields: {mismatched}")
    reference = replicas[0]
    conditions[python_abi] = {
        "replicas": 4,
        "replica_exact": True,
        "runtime": reference["runtime"],
        "initialization": reference["initialization"],
        "first_catalog_batch": reference["first_catalog_batch"],
        "losses": reference["losses"],
    }

archived_rtx = {"step_1": 4.528710842132568, "step_50": 4.203788757324219}
expected_cp311 = {"step_1": 4.546923637390137, "step_50": 4.184666633605957}
for condition in conditions.values():
    condition["delta_from_archived_rtx"] = {
        key: condition["losses"][key] - archived_rtx[key]
        for key in archived_rtx
    }

summary = {
    "artifact_format": "legacy_python_abi_matrix_v1",
    "archived_rtx_reference": {
        **archived_rtx,
        "python": "3.10.12",
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "first_catalog_batch_sha256": expected_batch,
    },
    "conditions": conditions,
    "cross_abi": {
        "initialization_equal": (
            conditions["cp310"]["initialization"] == conditions["cp311"]["initialization"]
        ),
        "losses_equal": conditions["cp310"]["losses"] == conditions["cp311"]["losses"],
    },
    "arms": arms,
}
(root / "matrix_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "arms": len(arms),
    "batch_order": "matched",
    "cp310_losses": conditions["cp310"]["losses"],
    "cp311_losses": conditions["cp311"]["losses"],
    "cross_abi": summary["cross_abi"],
}, sort_keys=True))
if conditions["cp311"]["losses"] != expected_cp311:
    raise SystemExit(
        "cp311 control did not reproduce the prior A800 reference: "
        f"expected={expected_cp311}, actual={conditions['cp311']['losses']}"
    )
PY

if (( failures > 0 )); then exit 1; fi
touch "${OUTPUT_ROOT}/matrix.done"
