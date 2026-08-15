#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

experiment="ripor_faithful_cat3query5_20260813"
log="$project_root/experiments/$experiment/formal.log"

bash scripts/gpu_experiment_handoff.sh submit \
  "$experiment" \
  "$log" \
  -- bash "$project_root/repro/20260813_ripor_faithful/run_full.sh"
