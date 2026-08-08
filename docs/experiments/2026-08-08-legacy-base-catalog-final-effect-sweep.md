# Experiment: Legacy base catalog final retrieval effect sweep
Status: done (baseline parity failed; runtime-mismatch diagnostic only)
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

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_sweep_20260808_v3/legacy-c{0,3,6,12}-q5-seed{20260626,20260627}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_sweep_20260808_v3/logs/`

Bootstrap log: `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/legacy_catalog_base_sweep_20260808_v3.log`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_sweep_20260808_v3/sweep_summary.json`

Job: `tide-chenyufei03-10545-jumcn` (`rudder-exper`, one 8xA800 node, launch commit `8c72056`)

## Verification

- [x] Minimal local check: shell syntax, 74 focused pipeline tests, immutable artifact hashes
- [x] Archived base `final` reproduces `75/75` Top-50 rankings before training starts
- [x] Training manifests report source rows `94,912`, expanded query rows `124,175`, query steps `38,805`, eval batch `16`
- [x] Query dev evaluation occurs every `200` steps and at terminal step `38,805`
- [x] Each arm produces `75` Official core75 rankings; `600` rows total
- [x] Overall, single-skill, and multi-skill metrics are rescored
- [x] Reviewer required? no; focused local semantic review

## Result / Notes

The archived target is `LETTER base single/final`, with catalog epochs `3`, query epochs `5`, seed
`20260626`, and dev-best query step `38,800`. Archived Official core75 macro recall is
`R@1/R@10/R@50 = 28.37/69.00/75.00` overall, `41.67/45.83/50.00` on 24 single-skill queries, and
`22.11/79.90/86.76` on 51 multi-skill queries under beam `50` and beam-order aggregation.

The superseded enhanced-data launcher also used `eval_batch=128` and `eval_steps=1,000`. The archived
base run used `eval_batch=16` and `eval_steps=200`; its dev-best checkpoint is query step `38,800`, which
the `1,000`-step cadence could not select. The base launcher now validates both values and the complete
query-eval cadence after every arm.

The run completed all eight arms, but the archived `catalog=3`, seed `20260626` checkpoint was not
reproduced: none of the `75` Top-50 rankings matched. The archived query dev-best loss is `0.782136` at
step `38,800`; the new arm reaches `0.993299` at the same step. Data, semantic IDs, model revision,
shuffle order, optimizer settings, eval cadence, and evaluator inputs are byte- or manifest-matched.
The remaining confirmed mismatch is the runtime: this job used Python `3.11.11`, torch `2.3.1+cu121`,
Transformers `4.55.4`, cuDNN `8.9.2`, and A800; the archived record used Python `3.10.12`, torch
`2.10.0+cu128`, Transformers `5.3.0`, and RTX PRO 6000 Blackwell. The closest available runtime archive
also supplies cuDNN `9.10.2`.

Official core75 macro recall in percent, averaged across seeds `20260626/20260627` under base 94,912,
query epochs `5`, batch `16`, dev-best with eval every `200` steps, and beam `50`:

| Catalog epochs | Overall R@1/10/50 | Single R@1/10/50 | Multi R@1/10/50 |
|---:|---:|---:|---:|
| 0 | 23.59 / 67.51 / 76.78 | 31.25 / 47.92 / 54.17 | 19.99 / 76.73 / 87.42 |
| 3 | 14.51 / 48.50 / 61.78 | 8.33 / 16.67 / 27.08 | 17.42 / 63.48 / 78.10 |
| 6 | 16.90 / 55.03 / 65.11 | 16.67 / 31.25 / 37.50 | 17.01 / 66.22 / 78.10 |
| 12 | 18.32 / 54.48 / 68.11 | 20.83 / 33.33 / 43.75 | 17.14 / 64.44 / 79.58 |

These values must not be used to estimate catalog effects against the archived Notion baseline. The
archived-package-runtime rerun is tracked separately in
`docs/experiments/2026-08-08-legacy-base-exact-runtime-catalog-sweep.md`.
