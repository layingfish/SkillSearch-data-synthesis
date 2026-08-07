# Experiment: Legacy baseline parity probe
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Reproduce the archived `2026-07-14` single-SID baseline before rerunning the catalog sweep. The first
confirmed divergence is shuffle state: the archived single-process loop used one persistent
`random.Random(20260626)` across the initial catalog shuffle and every catalog/query epoch. The current
DDP-capable loop independently seeds each epoch and therefore changes the first catalog batch.

The probe restores only the archived single-process shuffle contract. It uses T5-base revision
`a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`, batch `16`, BF16, `lr=3e-4`, weight decay `0`, catalog
epochs `3`, and stops after catalog step `950`. It records runtime versions, input/tokenizer/model hashes,
initial trainable parameter hashes, ordered row IDs, RNG hashes, and step `1/50/900/950` losses. Step
`950` crosses the 7-row tail batch at step `910` and the second epoch shuffle at step `911`.

## Commands

Train:

```text
bash scripts/run_legacy_baseline_parity_probe_20260807.sh
```

Eval: none; the probe gate requires six-decimal training-loss parity at archived steps `50/900/950`:
`4.203789/4.091409/4.072384` under the archived configuration above.

## Outputs

Checkpoint: none

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_baseline_parity_probe_20260807_v1/probe.log`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_baseline_parity_probe_20260807_v1/parity_probe.json`

## Verification

- [x] Minimal local check: shell syntax, Python compile, and 135-test suite (17 torch-only skips)
- [x] Batch-order and incomplete-tail invariant tests
- [ ] Initial CE parameter parity test
- [ ] Single-GPU step `1/50/900/950` probe
- [x] Reviewer required? yes; no remaining CRITICAL/HIGH findings

## Result / Notes

Frozen input byte hashes already match the published handoff: enhanced query rows
`55b6539c3379523f78c2326580fa690500f9b213d1d620b250ba333b65bcba08`, catalog
`1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3`, and semantic IDs
`084cccd5033d0aee8fb7bfbf05d55508744376c50a3c9bef95803ee69b41fa61`.
