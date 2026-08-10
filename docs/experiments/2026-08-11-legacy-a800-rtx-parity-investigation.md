# Experiment: Legacy A800/RTX PRO 6000 parity investigation
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Explain why the archived single-SID checkpoint trained on one RTX PRO 6000
Blackwell does not reproduce on an A800, despite matching the recorded data,
model, seed, optimizer, and evaluation protocol. This investigation is
diagnostic. It does not change the CE target, multi-skill expansion, catalog
schedule, or evaluation semantics.

## Fixed archived training contract

The comparison refers to the archived single-SID run with `94,912` source rows,
`124,175` expanded single-positive rows, catalog epochs `3`, query epochs `5`,
batch size `16`, seed `20260626`, BF16 autocast, AdamW lr `3e-4`, weight decay
`0`, no scheduler, no warmup, and the default AdamW backend (`foreach=None`,
runtime inference `foreach=True`). The catalog phase has `2,730` updates and
the query phase has `38,805` updates under this fixed row count and batch size.

The train source, original trainer, catalog, semantic-ID file, and T5-base
weights match the archived SHA-256 contract. The source file contains the base
rows and task-style pseudo-query rows; it excludes the later `5,166` mode3
augmentation rows. The archived final checkpoint is `final`, which is tensor-
identical to the dev-best checkpoint.

## Evidence that input and evaluation are aligned

- Loading the archived RTX checkpoint on A800 reproduces the archived inference
  ordering and Official core75 metrics.
- The archived predictions close the checkpoint-to-evaluation loop: Official
  core75 uses `75` queries, beam `50`, beam-order aggregation, CE log-prob, and
  macro Recall over Overall, Single, and Multi subsets.
- The original model fingerprint, CUDA RNG hash, and first catalog batch hash
  match between the archived RTX probe and the A800 probe.
- The archived source order, incomplete tail batch, and persistent
  `random.Random(20260626)` shuffle behavior were reproduced on A800.

This rules out checkpoint corruption, SemID mismatch, prediction-directory
mix-up, evaluation implementation mismatch, and the known data-row mismatch.

## Runtime comparison

The final A800 probe used Python `3.10.12`, torch `2.10.0+cu128`, Transformers
`5.3.0`, Tokenizers `0.22.2`, CUDA `12.8`, and cuDNN `9.10.2`. The RTX audit
reports the same torch build, CUDA package family, and cuBLAS `12.8.4.1`.
The remaining coupled hardware variables are:

| Runtime | GPU | Driver |
|---|---|---|
| Archived reference | RTX PRO 6000 Blackwell, compute capability `12.0` | `580.126.09` |
| Reproduction | NVIDIA A800-SXM4-80GB, Ampere | `535.161.07` |

The library audit found no actionable user-space cuBLAS mismatch. The evidence
does not separate GPU architecture from driver behavior because they changed
together.

## Controlled step-1 probes

All values in this section use the fixed archived catalog probe: one catalog
update, batch `16`, seed `20260626`, BF16, lr `3e-4`, weight decay `0`, default
TF32/cuDNN flags, and AdamW `foreach=None` unless stated otherwise.

### Original dropout `0.1`

```text
RTX PRO 6000: 4.528710842132568
A800:         4.546923637390137
delta:        0.018212795257568
```

### Dropout forced to `0.0`

The A800 probe set all `86` instantiated T5 Dropout modules to `p=0.0` and
recorded that audit before training.

```text
RTX PRO 6000: 4.538916110992432
A800:         4.554369926452637
delta:        0.015453815460205
```

Removing dropout reduced the cross-GPU loss gap by only
`0.002758979797` under this step-1 protocol. Dropout mask mapping is therefore
not the dominant source of the split.

## Runtime/backend controls already tested

The following A800 catalog-only controls used the same `50`-step protocol,
input hashes, initialization, seed, batch `16`, BF16, lr `3e-4`, and weight
decay `0`:

```text
default BF16                         step-1 4.546923637  step-50 4.184666634
BF16 reduced-precision reduction off step-1 4.546850681  step-50 4.178029537
explicit cuBLAS                     step-1 4.546923637  step-50 4.184666634
cuBLASLt                            step-1 4.546899319  step-50 4.164062500
cuBLASLt + BF16 reduction off       step-1 4.546899319  step-50 4.169090271
CUBLAS_WORKSPACE_CONFIG=:16:8       step-1 4.546948433  step-50 4.171227932
foreach=False                       step-1 4.546923637  step-50 4.164748192
FP32 autocast off                   step-1 4.536283493  step-50 4.166709423
```

None of these A800 controls reproduced the RTX step-1/50 pair
`4.528710842/4.203788757` under the same protocol. Python `3.10.12` versus
`3.11.13`, Transformers `4.57.6` versus `5.3.0`, and the compatible torch
wheel variants also failed to explain the split; replicas were exact within
each tested A800 condition.

## End-to-end amplification

The complete `catalog=3 + query=5` A800 default-BF16 run produced Overall
R@1/10/50 `12.7238/49.5429/65.8889`, Single
`0.0000/20.8333/45.8333`, and Multi `18.7115/63.0532/75.3268` under Official
core75, beam `50`, and beam-order aggregation. The archived RTX checkpoint
produced Overall `28.3683/69.0000/75.0000`, Single
`41.6667/45.8333/50.0000`, and Multi `22.1102/79.9020/86.7647` under the same
evaluation condition.

Within A800, backend controls changed final Overall R@1 from `12.7238` to
`21.1016` and Overall R@10 from `49.5429` to `69.5556` under the same data and
training schedule. The two default replicas were byte-identical. This shows
that small early numerical differences can be amplified by the
`2,730`-update catalog phase, the shared AdamW state, and the subsequent
`38,805`-update query phase.

## Current conclusion

The first divergence occurs in GPU numerical execution during the first
forward/backward update. Dropout, input order, model initialization, Python
version, Transformers version, torch wheel choice, checkpoint loading, and
evaluation protocol have been tested or matched and do not explain the full
gap. The leading unresolved cause is the BF16/GEMM execution path on different
GPU architectures and drivers, with optimization amplifying the resulting
trajectory difference.

This conclusion is causal at the level of the coupled hardware/runtime
condition. It does not identify whether the remaining contribution comes from
the Blackwell versus Ampere architecture, driver `580.126.09` versus
`535.161.07`, or their interaction.

## State and next decision

The full recovery training proposed earlier remains paused. The dropout-zero
probe is complete and archived at
`/models_rw/SkillSearch-data-synthesis/experiments/legacy_dropout0_probe_20260810_v3`.
The next high-value step is a controlled full training on a matching RTX
environment, or a precision/backend bridge that can be validated first on the
one-step probe. Do not interpret the existing A800 catalog sweep as evidence
about the intrinsic value of catalog epochs until this hardware/runtime gap is
closed.
