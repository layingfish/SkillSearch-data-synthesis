# Experiment: Legacy catalog final retrieval effect sweep
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Measure the retrieval effect of catalog pretraining with the archived single-process trainer. Run catalog
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

Job: pending submission to `rudder-exper`, one 8xA800 node

## Verification

- [x] Minimal local check: shell syntax, focused RNG test, and launch matrix audit
- [x] Archived trainer source commit and SHA-256 verified before launch
- [x] Eight unique GPU assignments and output directories verified
- [ ] Each arm produces 75 Official core75 rankings and a validated training manifest
- [ ] Reviewer required? no; focused local semantic review required before submission

## Result / Notes

Pending.
