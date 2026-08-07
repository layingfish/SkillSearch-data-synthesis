# Experiment: Legacy baseline parity probe
Status: failed
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

Job: `tide-chenyufei03-10493-gekph` (`rudder`, one 8xA800 node, one GPU used for archived
single-process parity, commit `77d0b7c6e37d5b29808e5d131c51e275f143c194`)

## Verification

- [x] Minimal local check: shell syntax, Python compile, and 135-test suite (17 torch-only skips)
- [x] Batch-order and incomplete-tail invariant tests
- [ ] Initial CE parameter parity test (the historical run did not persist its initial state)
- [x] Single-GPU step `1/50/900/950` probe
- [x] Reviewer required? yes; no remaining CRITICAL/HIGH findings

## Result / Notes

Frozen input byte hashes already match the published handoff: enhanced query rows
`55b6539c3379523f78c2326580fa690500f9b213d1d620b250ba333b65bcba08`, catalog
`1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3`, and semantic IDs
`084cccd5033d0aee8fb7bfbf05d55508744376c50a3c9bef95803ee69b41fa61`.

Runtime preflight passed with Python `3.11.13`, torch `2.8.0+cu128`, CUDA `12.8`, cuDNN `91002`,
Transformers `4.57.6`, Tokenizers `0.22.2`, and one NVIDIA A800-SXM4-80GB selected from the node.

The probe completed all `950` catalog steps but failed the six-decimal parity gate:

```text
step  observed   archived   delta
50    4.188124   4.203789   -0.015665
900   4.090215   4.091409   -0.001194
950   4.084387   4.072384   +0.012003
```

The archived incomplete-batch boundary was reproduced without padding: step `910` used the final `7`
catalog rows and had loss `4.100275`; step `911` used the first batch after the second in-place shuffle
and had loss `4.114972`. The next probe must isolate the remaining model-initialization/runtime drift.
The strongest recorded historical environment candidate is Python `3.10.12`, torch `2.10.0+cu128`,
Transformers `5.3.0`, Tokenizers version still to recover, and an NVIDIA RTX PRO 6000 Blackwell GPU.
The archived log's `Loading weights: 257 tensors` path is also absent under the current Transformers
`4.57.6` runtime, so Transformers `5.3.0` model construction is the first variable to test.
