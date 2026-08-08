# Experiment: Legacy base exact-runtime catalog sweep
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Reproduce the archived `SkillGR base+single-SID/LETTER` training runtime before interpreting catalog
exposure. Use the frozen 94,912-row task5 base dataset without the 5,166-row mode3 augmentation. Run
catalog epochs `0/3/6/12` at seeds `20260626/20260627`, followed by five query epochs. The only intended
change from the active `v3` sweep is the Python package runtime: torch `2.10.0+cu128`, Transformers
`5.3.0`, Tokenizers `0.22.2`, CUDA `12.8`, and cuDNN `9.10.2` on the available A800 hardware.

## Commands

Train:

```text
bash scripts/run_legacy_catalog_base_exact_runtime_tianzhi_20260808.sh
```

Eval: each arm runs Official core75 with `75` queries, beam `50`, beam-order aggregation, CE log-prob,
and subset rescoring for Overall, Single, and Multi.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_exact_runtime_20260808_v1/legacy-c{0,3,6,12}-q5-seed{20260626,20260627}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_exact_runtime_20260808_v1/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_exact_runtime_20260808_v1/sweep_summary.json`

## Verification

- [ ] Runtime and eight-GPU preflight
- [ ] Immutable input, SID, model, and archived-prediction hashes
- [ ] Eight training manifests and exact eval cadence
- [ ] Eight unique 75-row Official core75 rankings
- [ ] Archived `catalog=3`, seed `20260626` comparison
- [ ] Reviewer required? no; parity-sensitive diagnostic with existing invariant tests

## Result / Notes

Pending.
