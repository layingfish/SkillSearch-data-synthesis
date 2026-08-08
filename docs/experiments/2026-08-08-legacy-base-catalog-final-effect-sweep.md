# Experiment: Legacy base catalog final retrieval effect sweep
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Reproduce the archived `LETTER base single` result on the 94,912-row task5 base dataset, excluding the
5,166 mode3 augment rows. Then measure catalog epochs `0/3/6/12` at seeds `20260626/20260627`, followed
by five query epochs. All arms use the archived single-process trainer from commit `15f6c51131f3b6f3e8580aa4564b17cbe13a9790`.

## Commands

Train and evaluate eight independent arms on one 8xA800 node:

```text
bash scripts/run_legacy_catalog_base_final_effect_tianzhi_20260808.sh
```

Configuration: task5 base source rows `94,912`, expanded query rows `124,175`, T5-base, per-arm global
batch `16`, `lr=3e-4`, constant learning rate, weight decay `0`, BF16, max source length `2048`, query
epochs `5`, eval batch `16`, dev evaluation every `200` query steps, and dev-best selection. Evaluation uses Official
core75 with `75` queries and `200` gold mentions, beam `50`, beam-order aggregation, and CE log-prob.
The node first reevaluates the archived base `final` checkpoint and requires all `75/75` Top-50 rankings
to match the archived prediction file exactly.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_sweep_20260808_v2/legacy-c{0,3,6,12}-q5-seed{20260626,20260627}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_sweep_20260808_v2/logs/`

Bootstrap log: `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/legacy_catalog_base_sweep_20260808_v2.log`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_sweep_20260808_v2/sweep_summary.json`

## Verification

- [x] Minimal local check: shell syntax, 74 focused pipeline tests, immutable artifact hashes
- [ ] Archived base `final` reproduces `75/75` Top-50 rankings before training starts
- [ ] Training manifests report source rows `94,912`, expanded query rows `124,175`, query steps `38,805`, eval batch `16`
- [ ] Query dev evaluation occurs every `200` steps and at terminal step `38,805`
- [ ] Each arm produces `75` unique Official core75 rankings
- [ ] Overall, single-skill, and multi-skill metrics are rescored
- [ ] Reviewer required? no; focused local semantic review

## Result / Notes

The archived target is `LETTER base single/final`, with catalog epochs `3`, query epochs `5`, seed
`20260626`, and dev-best query step `38,800`. Archived Official core75 macro recall is
`R@1/R@10/R@50 = 28.37/69.00/75.00` overall, `41.67/45.83/50.00` on 24 single-skill queries, and
`22.11/79.90/86.76` on 51 multi-skill queries under beam `50` and beam-order aggregation.

The superseded enhanced-data launcher also used `eval_batch=128` and `eval_steps=1,000`. The archived
base run used `eval_batch=16` and `eval_steps=200`; its dev-best checkpoint is query step `38,800`, which
the `1,000`-step cadence could not select. The base launcher now validates both values and the complete
query-eval cadence after every arm.
