# Experiment: Legacy BF16 Dropout-Zero Step-1 Probe
Status: planned
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

- [ ] Script and input SHA-256 checks pass
- [ ] Initial model, CUDA RNG, and first catalog batch hashes match the archived contract
- [ ] Child audit records `dropout_module_count` and only `p=0.0`
- [ ] Exactly one catalog train event is present
- [ ] A800 step-1 loss is compared with RTX `4.538916110992432` under the same dropout-zero protocol

## Result / Notes

Pending GPU execution.
