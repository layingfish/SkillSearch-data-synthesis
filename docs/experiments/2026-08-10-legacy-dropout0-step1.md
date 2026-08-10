# Experiment: Legacy BF16 Dropout-Zero Step-1 Probe
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Compare the first catalog update on A800 with the completed RTX PRO 6000 probe
after setting every instantiated T5 `Dropout.p` to `0.0`. Keep the archived
single-GPU protocol fixed: BF16 autocast, batch size `16`, seed `20260626`,
learning rate `3e-4`, weight decay `0`, default BF16 reduced-precision setting,
default TF32/cuDNN flags, default cuBLAS preference, and AdamW `foreach=None`
(effective `foreach=True`). Run exactly one catalog update.

## Commands

Train and audit:

```text
bash scripts/run_legacy_dropout0_probe_tianzhi_20260810.sh
```

## Outputs

Output root:

```text
/models_rw/SkillSearch-data-synthesis/experiments/legacy_dropout0_probe_20260810_v3
```

## Verification

- [x] Script and input SHA-256 checks pass
- [x] Initial model, CUDA RNG, and first catalog batch hashes match the archived contract
- [x] Child audit records `dropout_module_count` and only `p=0.0`
- [x] Exactly one catalog train event is present
- [x] A800 step-1 loss is compared with RTX `4.538916110992432` under the same dropout-zero protocol

## Result / Notes

A800 completed the probe in job `tide-chenyufei03-10588-tfdip` on queue
`rudder-exper`. The authoritative output is
`/models_rw/SkillSearch-data-synthesis/experiments/legacy_dropout0_probe_20260810_v3`.

Under BF16, batch `16`, seed `20260626`, lr `3e-4`, weight decay `0`, default
BF16 reduction, default TF32/cuDNN flags, and AdamW `foreach=None` (runtime
inference `foreach=True`), the step-1 losses were:

```text
RTX PRO 6000, dropout=0.0: 4.538916110992432
A800,          dropout=0.0: 4.554369926452637
delta:                       0.015453815460205078
```

The original dropout `0.1` comparison under the same probe contract was
`4.528710842132568` on RTX and `4.546923637390137` on A800. Setting dropout to
zero reduced the cross-GPU delta by only `0.002758979797` loss units. The probe
therefore does not support dropout masks as the main cause of the divergence.
