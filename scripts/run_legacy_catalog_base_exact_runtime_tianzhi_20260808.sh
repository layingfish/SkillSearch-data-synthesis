#!/usr/bin/env bash
set -euo pipefail

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
ROOT_DIR=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis
BOOTSTRAP_LOG=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/bootstrap_logs/legacy_catalog_base_exact_runtime_20260808_v1.log

mkdir -p "$(dirname "${BOOTSTRAP_LOG}")"
exec > >(tee -a "${BOOTSTRAP_LOG}") 2>&1
bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1
cd "${ROOT_DIR}"
exec bash scripts/run_legacy_catalog_base_exact_runtime_node_20260808.sh
