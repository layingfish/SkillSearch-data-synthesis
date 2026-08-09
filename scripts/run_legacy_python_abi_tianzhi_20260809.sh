#!/usr/bin/env bash
set -euo pipefail

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
PHYSICAL_REPO=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis
BOOTSTRAP_LOG=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/bootstrap_logs/legacy_python_abi_probe_20260809_v1.log
TORCH_BOOTSTRAP_WHEEL=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillDelta/pip_cache/torch-2.3.1+cu121-cp311-cp311-linux_x86_64.whl
LAUNCH_COMMIT="${SKILLGR_LAUNCH_COMMIT:?set SKILLGR_LAUNCH_COMMIT to the verified experiment commit}"

cd /tmp
mkdir -p "$(dirname "${BOOTSTRAP_LOG}")"
exec > >(tee -a "${BOOTSTRAP_LOG}") 2>&1
bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1

rm -rf /usr/local/lib/python3.11/site-packages/torch*
rm -rf /usr/local/lib/python3.11/site-packages/nvidia*
python3 -m pip install "${TORCH_BOOTSTRAP_WHEEL}" --no-deps -q

git config --global --add safe.directory "${PHYSICAL_REPO}"
git -C "${PHYSICAL_REPO}" cat-file -e "${LAUNCH_COMMIT}^{commit}"
resolved_commit="$(git -C "${PHYSICAL_REPO}" rev-parse "${LAUNCH_COMMIT}^{commit}")"
echo "launch_commit=${resolved_commit}"
export SKILLGR_LAUNCH_COMMIT="${resolved_commit}"

TEMP_CHECKOUT="$(mktemp -d /tmp/skillgr-python-abi-launch.XXXXXX)"
cleanup() {
  case "${TEMP_CHECKOUT}" in
    /tmp/skillgr-python-abi-launch.*) rm -rf -- "${TEMP_CHECKOUT}" ;;
    *) echo "refusing to remove unexpected temporary checkout: ${TEMP_CHECKOUT}" >&2 ;;
  esac
}
trap cleanup EXIT

git -C "${PHYSICAL_REPO}" archive --format=tar "${resolved_commit}" | tar -xf - -C "${TEMP_CHECKOUT}"
cd "${TEMP_CHECKOUT}"
bash scripts/run_legacy_python_abi_matrix_20260809.sh
