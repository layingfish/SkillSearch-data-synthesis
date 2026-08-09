# Experiment: Legacy end-to-end numerical sensitivity
Status: done
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

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_e2e_numerical_sensitivity_20260809_v1/arms/<arm>/result_summary.json`

Job: `tide-chenyufei03-10576-epnfx` (`rudder-r1`, one `8xA800` node, launch commit `5989029`)

## Verification

- [x] Shell syntax, Python compile, and control-wrapper unit test
- [x] Archived checkpoint previously reproduced all `75/75` Official core75 Top-50 rankings under the fixed CUDA package runtime
- [x] Eight distinct arms bind to GPUs `0-7`; all eight entered training
- [x] Seven BF16 training manifests match 94,912 source rows, 124,175 expanded query rows, catalog `2,730` steps, query `38,805` steps, batch `16`, lr `3e-4`, weight decay `0`, seed `20260626`, and the same semantic-ID SHA-256
- [x] Seven BF16 arms produce Overall, Single, and Multi Official core75 metrics; the FP32 arm has a recorded OOM and no retrieval result
- [x] The two default replicas have byte-identical final checkpoints and predictions
- [x] Reviewer required? no; focused local semantic review

## Result / Notes

All recall values below are percentages under the same evaluation condition:
Official core75, 75 queries, beam `50`, beam-order aggregation, CE log-prob.
All new arms use the fixed training condition in the goal section; only the
named numerical/backend control differs.

| Arm | Overall R@1/10/50 | Single R@1/10/50 | Multi R@1/10/50 | Dev-best step | Dev loss |
|---|---:|---:|---:|---:|---:|
| Archived RTX PRO 6000 reference | 28.3683 / 69.0000 / 75.0000 | 41.6667 / 45.8333 / 50.0000 | 22.1102 / 79.9020 / 86.7647 | 38,800 | unavailable here |
| `default_bf16_r0` | 12.7238 / 49.5429 / 65.8889 | 0.0000 / 20.8333 / 45.8333 | 18.7115 / 63.0532 / 75.3268 | 38,800 | 0.870400 |
| `default_bf16_r1` | 12.7238 / 49.5429 / 65.8889 | 0.0000 / 20.8333 / 45.8333 | 18.7115 / 63.0532 / 75.3268 | 38,800 | 0.870400 |
| `bf16_reduction_off` | 18.5460 / 52.6540 / 64.7778 | 12.5000 / 16.6667 / 25.0000 | 21.3912 / 69.5892 / 83.4967 | 38,805 | 0.836602 |
| `cublaslt` | 19.5905 / 66.0952 / 74.1746 | 20.8333 / 41.6667 / 50.0000 | 19.0056 / 77.5910 / 85.5509 | 38,805 | 0.792116 |
| `cublaslt_bf16_reduction_off` | 19.2571 / 61.2889 / 70.1111 | 25.0000 / 45.8333 / 45.8333 | 16.5546 / 68.5621 / 81.5359 | 38,805 | 0.799968 |
| `workspace_16` | 20.9238 / 69.5556 / 75.3333 | 25.0000 / 50.0000 / 50.0000 | 19.0056 / 78.7582 / 87.2549 | 38,200 | 0.770134 |
| `foreach_false` | 21.1016 / 64.2667 / 71.2222 | 29.1667 / 37.5000 / 41.6667 | 17.3063 / 76.8627 / 85.1307 | 38,400 | 0.779752 |

Across the seven BF16 arms under the fixed training and evaluation conditions,
the observed metric ranges are:

| Subset | R@1 range | R@10 range | R@50 range | Archived RTX R@1/10/50 |
|---|---:|---:|---:|---:|
| Overall | 12.7238-21.1016 | 49.5429-69.5556 | 64.7778-75.3333 | 28.3683 / 69.0000 / 75.0000 |
| Single | 0.0000-29.1667 | 16.6667-50.0000 | 25.0000-50.0000 | 41.6667 / 45.8333 / 50.0000 |
| Multi | 16.5546-21.3912 | 63.0532-78.7582 | 75.3268-87.2549 | 22.1102 / 79.9020 / 86.7647 |

The two default replicas are exactly reproducible on the same A800 runtime.
Their `structured_decoder.pt` SHA-256 is
`2aad4869b56a6d7d3e8c0ca6c66b3d8f6e6066609ea19eac8541f135c30e5c4a`;
their prediction SHA-256 is
`54b6c131c36703c43ca3210bed5aab4b5f0c5d6abed1221a1d19e4ee567c4daa`.
Every new arm has `0/75` rows whose complete Top-50 ordering exactly matches
the archived RTX predictions.

The backend controls cause large end-to-end changes despite identical data,
initialization, seed, and explicit training hyperparameters. The largest
within-A800 spans are 8.3778 points for Overall R@1, 20.0127 points for Overall
R@10, and 29.1667 points for Single R@1. This confirms that small numerical
trajectory changes can be amplified into large retrieval differences.

`workspace_16` recovers the archived deep-recall level: its Overall R@10/50 is
69.5556/75.3333 versus the archived 69.0000/75.0000, and its Multi R@10/50 is
78.7582/87.2549 versus 79.9020/86.7647. Its Overall R@1 remains 20.9238 versus
28.3683, and its Single R@1 remains 25.0000 versus 41.6667. None of the tested
A800 controls reproduces the archived Top-1 behavior. The best observed
Overall, Single, and Multi R@1 values are respectively 7.2667, 12.5000, and
0.7190 points below the archived values.

The experiment therefore supports numerical amplification as an explanation
for the large checkpoint gap, while leaving the exact archived trajectory
unreproduced. Combined with the preceding 50-step probes, the unresolved
coupled variable remains RTX PRO 6000 Blackwell plus driver `580.126.09` versus
A800 Ampere plus driver `535.161.07`; the evidence does not isolate architecture
from driver behavior.

The node launcher did not create `matrix_summary.json` because the FP32 child
failed. The seven per-arm `result_summary.json` files are the authoritative
metrics. The Tianzhi pod was removed after the launcher terminated.

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

The FP32 arm later failed with CUDA OOM at catalog step `650/2,730`. Under
batch `16` on one 79.33-GiB A800, the process held 78.50 GiB and the next
attention operation requested another 2.10 GiB with 838 MiB free. The seven
BF16 arms continue under their registered settings. FP32 therefore has no
comparable end-to-end retrieval result in this job.
