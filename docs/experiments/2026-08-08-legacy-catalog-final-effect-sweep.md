# Experiment: Legacy catalog final retrieval effect sweep
Status: failed (wrong target: 100,078-row enhanced data)
Risk Tier: 2

## Goal / Hypothesis / Change Scope

This run accidentally used the 100,078-row enhanced dataset and the enhanced archived checkpoint. It does
not evaluate the 94,912-row `LETTER base single` target. Measure the retrieval effect of catalog pretraining with the archived single-process trainer. Run catalog
epochs `0/3/6/12` at seeds `20260626/20260627`, followed by five query epochs. The archived trainer from
commit `15f6c51131f3b6f3e8580aa4564b17cbe13a9790` preserves the original continuous shuffle RNG. The current
pipeline is used only for one consistent Official core75 evaluation.

## Commands

Train and evaluate eight independent arms on one Tianzhi node:

```text
bash scripts/run_legacy_catalog_final_effect_node_20260808.sh
```

Configuration: T5-base, per-arm global batch `16`, `lr=3e-4`, constant learning rate, weight decay `0`,
BF16, max source length `2048`, query epochs `5`, dev evaluation every `1,000` query steps, and dev-best
selection. Evaluation uses Official core75 with beam `50`, beam-order aggregation, CE log-prob scoring,
and Overall `R@1/R@10/R@50`.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260808_v3/legacy-c{0,3,6,12}-q5-seed{20260626,20260627}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260808_v3/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260808_v3/sweep_summary.json`

Job: `tide-chenyufei03-10506-kiheo` (`rudder-exper`, one 8xA800 node, launch commit
`44cf0eea862003a194943ae5c6741d1cf447bd6d`)

## Verification

- [x] Minimal local check: shell syntax, focused RNG test, and launch matrix audit
- [x] Archived trainer source commit and SHA-256 verified before launch
- [x] Eight unique GPU assignments and output directories verified
- [x] Remote preflight passed and all eight single-GPU training processes started
- [x] Each arm produces 75 Official core75 rankings and a validated training manifest
- [x] Reviewer required? no; focused local semantic review completed

## Result / Notes

This sweep did not reproduce the archived `c3`, seed `20260626` baseline under the Official core75
protocol below. Its cross-arm catalog deltas are therefore not valid evidence about catalog exposure.
The raw results remain recorded for diagnosis only. See
`docs/experiments/2026-08-08-legacy-baseline-last-checkpoint-parity.md`.

All eight arms completed training and Official core75 evaluation. The final artifact is
`/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260808_v3/sweep_summary.json`.
Every arm contains 75 unique query rankings and reports 200 total gold-skill mentions.

Official core75 Overall macro recall, mean across seeds `20260626/20260627` (75 queries, T5-base,
archived trainer commit `15f6c51`, per-arm global batch `16`, query epochs `5`, dev-best checkpoint,
beam `50`, beam-order aggregation, CE log-prob scoring):

| Catalog epochs | R@1 (%) | R@10 (%) | R@50 (%) | Delta vs c0 R@1 | Delta vs c0 R@10 | Delta vs c0 R@50 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 26.85 | 67.98 | 75.67 | 0.00 | 0.00 | 0.00 |
| 3 | 26.98 | 69.10 | 75.61 | +0.13 | +1.13 | -0.06 |
| 6 | 25.53 | 67.07 | 75.67 | -1.31 | -0.90 | 0.00 |
| 12 | 24.59 | 64.90 | 73.78 | -2.26 | -3.07 | -1.89 |

Per-seed Official core75 Overall macro recall under the same condition:

| Catalog epochs | Seed | R@1 (%) | R@10 (%) | R@50 (%) |
|---:|---:|---:|---:|---:|
| 0 | 20260626 | 27.26 | 68.14 | 78.44 |
| 0 | 20260627 | 26.43 | 67.81 | 72.89 |
| 3 | 20260626 | 27.92 | 67.65 | 72.56 |
| 3 | 20260627 | 26.03 | 70.56 | 78.67 |
| 6 | 20260626 | 24.37 | 63.37 | 74.33 |
| 6 | 20260627 | 26.70 | 70.78 | 77.00 |
| 12 | 20260626 | 24.92 | 62.48 | 71.11 |
| 12 | 20260627 | 24.26 | 67.33 | 76.44 |

The archived Notion predictions score `29.32/69.04/78.56` for Overall `R@1/R@10/R@50` under the
same Official core75 rescorer (75 queries, beam `50`, archived prediction set). The new `c3`, seed
`20260626` arm scores `27.92/67.65/72.56` under the A800 run and has `0/75` exact Top-50 ranking rows
against that archived prediction set. This checkpoint mismatch invalidates the intended historical
baseline comparison.

No catalog-effect conclusion is drawn from these arms because the archived baseline parity check failed.

The launch-commit aggregator used an obsolete flat metric key and exited after all per-arm evaluations
were complete. Commit `8b294c1` fixes the nested metric lookup. The validated summary was regenerated
from the eight immutable per-arm `result_summary.json` files; checkpoints and predictions were unchanged.
