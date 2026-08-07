# Experiment: Legacy catalog parity and exposure sweep
Status: failed
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Reproduce the archived mode3 single-SID LETTER baseline before interpreting catalog exposure. Run
catalog epochs `0/3/6/12` at seeds `20260626/20260627`, followed by five query epochs. All eight arms run
as independent single-GPU processes on one 8xA800 node. The `c3`, seed `20260626` arm is a strict parity
check against the archived Notion baseline. RIPOR heads remain frozen and their allocation uses an
isolated CPU RNG scope so auxiliary parameters cannot change the CE control-head seed.

## Commands

Train and eval all arms in one Tianzhi node:

```text
bash scripts/run_legacy_catalog_sweep_node_20260807.sh
```

Common configuration: T5-base, global batch `16`, `lr=3e-4`, constant scheduler, warmup `0`, weight decay
`0`, BF16, max source length `2048`, query epochs `5`, dev eval every `1,000` query steps, dev-best
selection, Official/private evaluation with beam `50` and beam-order aggregation. GPUs `0-3` run seed
`20260626`; GPUs `4-7` run seed `20260627`.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260807_v2/legacy-c{0,3,6,12}-q5-seed{20260626,20260627}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260807_v2/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260807_v2/eval/`

Job: `tide-chenyufei03-10489-pwucc` (`rudder`, one 8xA800 node, commit `1268740e4ff2164c3d9a61d3d4d44a957613de62`)

## Verification

- [x] Minimal local check: shell syntax, Python compile, input-path audit, and local unit suite
- [x] RIPOR auxiliary initialization preserves legacy CE RNG and parameters
- [x] Each arm uses one process, global batch 16, and exact catalog/query exposure
- [x] `c3`, seed `20260626` parity checked: 0 of 75 archived Official Top-50 rankings matched
- [x] Reviewer required? no; focused local semantic review completed before submission

## Result / Notes

Local unit suite: 71 passed and 13 torch-only tests skipped in the CPU workspace. Every remote arm
runs the skipped RNG parity test on torch 2.8.0 before training.

Submitted on `2026-08-07`. Remote smoke test and torch RNG parity test passed. All eight workers
entered training with GPUs `0-7` reporting active compute.

During evaluation, the `c3`, seed `20260626` arm recorded a `0/75` exact Top-50 match against the
archived predictions. The original launcher returned status 1 for that scientific result and attempted
fail-fast cleanup. The remaining training processes survived as reparented processes and continued.
The launcher was corrected so parity is recorded in the summary without failing the arm, and future
per-arm failures are aggregated only after every arm exits.

All eight training manifests passed: `ddp_world_size=1`, global batch `16`, query steps `40,420`,
catalog steps `0/2,730/5,460/10,920`, frozen RIPOR heads, and RNG-preserving auxiliary initialization.
Interrupted post-training evaluation was completed with the idempotent arm finalizer; no checkpoint was
retrained or modified.

Official core75 macro recall, averaged across seeds `20260626/20260627` (75 queries, 24 single-skill,
51 multi-skill, beam `50`, beam-order aggregation, dev-best checkpoint):

| Catalog epochs | All R@1 | All R@10 | All R@50 | Single R@1 | Single R@10 | Single R@50 | Multi R@1 | Multi R@10 | Multi R@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27.98 | 65.68 | 71.72 | 39.58 | 50.00 | 52.08 | 22.52 | 73.06 | 80.96 |
| 3 | 20.55 | 59.94 | 70.17 | 27.08 | 45.83 | 52.08 | 17.48 | 66.57 | 78.68 |
| 6 | 22.98 | 58.92 | 71.50 | 33.33 | 43.75 | 52.08 | 18.11 | 66.06 | 80.64 |
| 12 | 18.35 | 57.18 | 68.61 | 27.08 | 45.83 | 52.08 | 14.23 | 62.52 | 76.39 |

Per-seed Official core75 all-query macro recall under the same evaluation condition:

| Catalog epochs | Seed | R@1 | R@10 | R@50 |
|---:|---:|---:|---:|---:|
| 0 | 20260626 | 26.26 | 68.11 | 73.67 |
| 0 | 20260627 | 29.70 | 63.25 | 69.78 |
| 3 | 20260626 | 19.73 | 58.17 | 69.56 |
| 3 | 20260627 | 21.37 | 61.70 | 70.78 |
| 6 | 20260626 | 20.37 | 55.59 | 69.89 |
| 6 | 20260627 | 25.59 | 62.25 | 73.11 |
| 12 | 20260626 | 15.10 | 55.56 | 68.00 |
| 12 | 20260627 | 21.59 | 58.81 | 69.22 |

The archived Notion predictions score `29.32/69.04/78.56` on all-query R@1/R@10/R@50 under the
same current rescorer. The strict `c3`, seed `20260626` arm scores `19.73/58.17/69.56`, lower by
`9.59/10.87/9.00` points. Archived single-skill scores are `54.17/58.33/62.50`; current strict `c3`
scores are `25.00/45.83/54.17`. Archived multi-skill scores are `17.63/74.08/86.11`; current strict
`c3` scores are `17.25/63.98/76.80`.

The first confirmed protocol mismatch is epoch shuffle state. The archived single-process code used one
persistent `random.Random(seed)` for the initial catalog shuffle and every later catalog/query epoch. The
current DDP-capable loop independently seeds catalog epochs with `seed + 10,000 + epoch` and query epochs
with `seed + epoch`, so its first catalog batch already differs. The archived loop also kept the final
incomplete batch; `--no-padding-distributed-sampler` was therefore correct for single-GPU tail handling.
A short continuous-RNG probe is required before treating Notion parity as closed.
