#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export SKILLGR_CODE_ROOT="${ROOT_DIR}"
export SKILLGR_CONFIG="${ROOT_DIR}/configs/legacy_catalog_base_final_effect_20260808.env"
export SKILLGR_PHYSICAL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/chenyufei03/SkillSearch-data-synthesis/experiments/legacy_catalog_base_sweep_20260808_v1

exec bash "${SCRIPT_DIR}/run_legacy_catalog_final_effect_node_20260808.sh"

