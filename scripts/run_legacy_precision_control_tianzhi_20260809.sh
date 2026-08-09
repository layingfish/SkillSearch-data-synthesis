#!/usr/bin/env bash
set -euo pipefail

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
PHYSICAL_REPO=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis
BOOTSTRAP_LOG=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/bootstrap_logs/legacy_precision_control_probe_20260809_v1.log
LAUNCH_COMMIT="${SKILLGR_LAUNCH_COMMIT:?set SKILLGR_LAUNCH_COMMIT to the verified experiment commit}"

mkdir -p "$(dirname "${BOOTSTRAP_LOG}")"
exec > >(tee -a "${BOOTSTRAP_LOG}") 2>&1
bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1

git config --global --add safe.directory "${PHYSICAL_REPO}"
git -C "${PHYSICAL_REPO}" cat-file -e "${LAUNCH_COMMIT}^{commit}"
resolved_commit="$(git -C "${PHYSICAL_REPO}" rev-parse "${LAUNCH_COMMIT}^{commit}")"
echo "launch_commit=${resolved_commit}"
export SKILLGR_LAUNCH_COMMIT="${resolved_commit}"

TEMP_CHECKOUT="$(mktemp -d /tmp/skillgr-precision-launch.XXXXXX)"
cleanup() {
  case "${TEMP_CHECKOUT}" in
    /tmp/skillgr-precision-launch.*) rm -rf -- "${TEMP_CHECKOUT}" ;;
    *) echo "refusing to remove unexpected temporary checkout: ${TEMP_CHECKOUT}" >&2 ;;
  esac
}
trap cleanup EXIT

git -C "${PHYSICAL_REPO}" archive --format=tar "${resolved_commit}" | tar -xf - -C "${TEMP_CHECKOUT}"
cd "${TEMP_CHECKOUT}"
bash scripts/run_legacy_precision_control_matrix_20260809.sh
