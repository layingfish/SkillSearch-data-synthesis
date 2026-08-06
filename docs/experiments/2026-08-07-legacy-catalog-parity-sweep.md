# Experiment: Legacy catalog parity and exposure sweep
Status: running
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
- [ ] Each arm uses one process, global batch 16, and exact catalog/query exposure
- [ ] `c3`, seed `20260626` reproduces all 75 archived Official top-50 rankings
- [x] Reviewer required? no; focused local semantic review completed before submission

## Result / Notes

Local unit suite: 71 passed and 13 torch-only tests skipped in the CPU workspace. Every remote arm
runs the skipped RNG parity test on torch 2.8.0 before training.

Submitted on `2026-08-07`. Remote smoke test and torch RNG parity test passed. All eight workers
entered training with GPUs `0-7` reporting active compute.
