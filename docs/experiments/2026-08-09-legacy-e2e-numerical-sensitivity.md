# Experiment: Legacy end-to-end numerical sensitivity
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Test whether small CUDA numerical/backend changes on A800 are amplified by the
complete archived training trajectory enough to explain the retrieval gap from
the RTX PRO 6000 checkpoint. All arms use the archived 94,912-source-row base
dataset, catalog epochs `3`, query epochs `5`, seed `20260626`, batch `16`,
constant learning rate `3e-4`, weight decay `0`, and dev-best selection every
`200` query steps.

One `8xA800` node runs one single-GPU arm per GPU:

1. `default_bf16_r0`
2. `default_bf16_r1`
3. `bf16_reduction_off`
4. `fp32`
5. `cublaslt`
6. `cublaslt_bf16_reduction_off`
7. `workspace_16`
8. `foreach_false`

The two default arms use identical data, seed, and settings. They measure
within-node replication. Evaluation is identical across arms: Official core75,
75 queries, beam `50`, beam-order aggregation, and CE log-prob.

## Commands

Train and evaluate:

```text
bash scripts/run_legacy_e2e_numerical_sensitivity_tianzhi_20260809.sh
```

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_e2e_numerical_sensitivity_20260809_v1/arms/<arm>/train/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_e2e_numerical_sensitivity_20260809_v1/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_e2e_numerical_sensitivity_20260809_v1/matrix_summary.json`

Job: pending (`rudder-r1`, one `8xA800` node)

## Verification

- [x] Shell syntax, Python compile, and control-wrapper unit test
- [x] Archived checkpoint previously reproduced all `75/75` Official core75 Top-50 rankings under the fixed CUDA package runtime
- [ ] Eight distinct arms bind to eight visible GPUs
- [ ] Training manifests match catalog `2,730` steps and query `38,805` steps
- [ ] Every arm produces Overall, Single, and Multi Official core75 metrics
- [ ] Reviewer required? no; focused local semantic review

## Result / Notes

Pending.
