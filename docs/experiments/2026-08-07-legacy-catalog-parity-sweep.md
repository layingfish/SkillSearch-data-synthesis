# Experiment: Legacy catalog parity and exposure sweep
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Reproduce the archived mode3 single-SID LETTER baseline before interpreting catalog exposure. Run four
single-process continuous arms with catalog epochs `0/3/6/12`, followed by five query epochs. The `c3`
arm is a strict parity check against the archived Notion baseline. RIPOR heads remain frozen and their
allocation uses an isolated CPU RNG scope so auxiliary parameters cannot change the CE control-head seed.

## Commands

Train and eval, one arm per one-node Tianzhi job with one visible A800 process:

```text
bash scripts/run_legacy_catalog_sweep_20260807.sh {0,3,6,12}
```

Common configuration: T5-base, global batch `16`, `lr=3e-4`, constant scheduler, warmup `0`, weight decay
`0`, BF16, seed `20260626`, max source length `2048`, query epochs `5`, dev eval every `1,000` query
steps, dev-best selection, Official/private evaluation with beam `50` and beam-order aggregation.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260807_v1/legacy-c{0,3,6,12}-q5/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260807_v1/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260807_v1/eval/`

## Verification

- [x] Minimal local check: shell syntax, Python compile, input-path audit, and local unit suite
- [ ] RIPOR auxiliary initialization preserves legacy CE RNG and parameters
- [ ] Each arm uses one process, global batch 16, and exact catalog/query exposure
- [ ] `c3` reproduces all 75 archived Official top-50 rankings
- [x] Reviewer required? no; focused local semantic review completed before submission

## Result / Notes

Local unit suite: 71 passed and 13 torch-only tests skipped in the CPU workspace. Every remote arm
runs the skipped RNG parity test on torch 2.8.0 before training.

Pending submission.
