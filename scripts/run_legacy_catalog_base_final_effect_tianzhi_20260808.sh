#!/usr/bin/env bash
set -euo pipefail

ALIAS_HELPER=/mnt/cfs_algo_bj/workspace/chenyufei03/skillact/scripts/setup_cfs_aliases.sh
TORCH_WHEEL=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillDelta/pip_cache/torch-2.3.1+cu121-cp311-cp311-linux_x86_64.whl
NODE_LAUNCHER=/mnt/cfs_algo_bj/workspace/chenyufei03/SkillSearch-data-synthesis/scripts/run_legacy_catalog_base_final_effect_node_20260808.sh

bash "${ALIAS_HELPER}" --with-dataset-rw || exit 1
rm -rf /usr/local/lib/python3.11/site-packages/torch*
rm -rf /usr/local/lib/python3.11/site-packages/nvidia*
python3 -m pip install "${TORCH_WHEEL}" --no-deps -q

exec bash "${NODE_LAUNCHER}"

