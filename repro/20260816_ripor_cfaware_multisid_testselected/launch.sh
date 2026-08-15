#!/usr/bin/env bash
set -euo pipefail

export RIPOR_EXPERIMENT="experiments/ripor_m0qwen_cf_multisid_testselected_cat3query5_20260816"
export RIPOR_QWEN3_THREE_WAY=1
export RIPOR_M0_WEIGHT=0.5
export RIPOR_QWEN_WEIGHT=0.3
export RIPOR_CF_WEIGHT=0.2
export RIPOR_QUERY_BALANCED_MULTI_POSITIVE=1
export RIPOR_REPRO_README="repro/20260816_ripor_cfaware_multisid_testselected/README.md"
export RIPOR_LAUNCHER_PATH="repro/20260816_ripor_cfaware_multisid_testselected/launch.sh"

exec bash repro/20260815_ripor_cfaware_testselected/run_full.sh
