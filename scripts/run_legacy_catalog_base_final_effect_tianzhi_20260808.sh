#!/usr/bin/env bash
set -euo pipefail

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
TORCH_WHEEL=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillDelta/pip_cache/torch-2.3.1+cu121-cp311-cp311-linux_x86_64.whl
NODE_LAUNCHER=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis/scripts/run_legacy_catalog_base_final_effect_node_20260808.sh
BOOTSTRAP_LOG=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/bootstrap_logs/legacy_catalog_base_sweep_20260808_v3.log
SITE_PACKAGES=/usr/local/lib/python3.11/site-packages

mkdir -p "$(dirname "${BOOTSTRAP_LOG}")"
exec > >(tee -a "${BOOTSTRAP_LOG}") 2>&1

bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1
cd /mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis

test -d "${SITE_PACKAGES}/nvidia" || { echo "missing pinned CUDA wheel packages" >&2; exit 2; }
CUDA_WHEEL_LD_PATH="$(find "${SITE_PACKAGES}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | sort | paste -sd ':' -)"
test -n "${CUDA_WHEEL_LD_PATH}" || { echo "no CUDA wheel library directories found" >&2; exit 2; }
export LD_LIBRARY_PATH="${CUDA_WHEEL_LD_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for library in \
  "${SITE_PACKAGES}/nvidia/cuda_runtime/lib/libcudart.so.12" \
  "${SITE_PACKAGES}/nvidia/cublas/lib/libcublas.so.12" \
  "${SITE_PACKAGES}/nvidia/cublas/lib/libcublasLt.so.12" \
  "${SITE_PACKAGES}/nvidia/cudnn/lib/libcudnn.so.8"; do
  test -e "${library}" || { echo "missing CUDA runtime library: ${library}" >&2; exit 2; }
done

rm -rf \
  "${SITE_PACKAGES}/torch" \
  "${SITE_PACKAGES}/torch.libs" \
  "${SITE_PACKAGES}"/torch-*.dist-info \
  "${SITE_PACKAGES}/torchgen" \
  "${SITE_PACKAGES}/functorch"
python3 -m pip install "${TORCH_WHEEL}" --no-deps -q

python3 - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable after Torch repair")
tensor = torch.ones((2, 2), device="cuda:0")
result = tensor @ tensor
torch.cuda.synchronize()
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"device={torch.cuda.get_device_name(0)} smoke_sum={float(result.sum().item())}",
    flush=True,
)
PY

exec bash "${NODE_LAUNCHER}"
