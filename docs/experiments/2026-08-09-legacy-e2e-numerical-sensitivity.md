# Experiment: Legacy end-to-end numerical sensitivity
Status: running
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

Job: `tide-chenyufei03-10576-epnfx` (`rudder-r1`, one `8xA800` node, launch commit `5989029`)

## Verification

- [x] Shell syntax, Python compile, and control-wrapper unit test
- [x] Archived checkpoint previously reproduced all `75/75` Official core75 Top-50 rankings under the fixed CUDA package runtime
- [x] Eight distinct arms bind to GPUs `0-7`; all eight entered training
- [ ] Training manifests match catalog `2,730` steps and query `38,805` steps
- [ ] Every arm produces Overall, Single, and Multi Official core75 metrics
- [ ] Reviewer required? no; focused local semantic review

## Result / Notes

Pending.

The first submission, `tide-chenyufei03-10575-kpcpi`, failed in bootstrap
before creating the output root because its command contained an invalid full
commit hash. Tianzhi removed that pod. The corrected job uses commit
`5989029c7a2bf37e0cc6e440d9865bab3abe449e`.

The corrected job passed the Python `3.10.12`, torch `2.10.0+cu128`, CUDA
`12.8`, cuDNN `9.10.2`, and eight-A800 preflight. All eight control audit files
match their assigned GPU and requested backend settings. Their observed
catalog step-50 losses reproduce the prior short-probe values exactly for the
seven BF16 arms; the FP32 arm was still before its first 50-step log point at
the time of this update.
