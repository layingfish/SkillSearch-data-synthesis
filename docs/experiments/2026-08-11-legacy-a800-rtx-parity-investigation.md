# Experiment: Legacy A800/RTX PRO 6000 parity investigation
Status: running
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

The 2026-08-11 fixed-pool diagnostic localizes the final Top-1 failure more
precisely. For every Official core75 query, it exhaustively rescored the union
of the RTX/control/warmup/clip beam-order Top50 lists plus every gold skill.
All four checkpoints retained exactly the same R@1 and R@5 as their raw beam
outputs. The RTX-correct/A800-wrong cases recovered `0/9`, `0/9`, and `0/8`
Top-1 queries for workspace control, warmup=`500`, and clip=`1.0`, respectively;
conditions are Official core75=`75` queries, four learned-code full-vocabulary
log-probs plus stop control=`1`, and exhaustive per-query pools.

The final R@1 gap is therefore a checkpoint-scoring gap under the legacy CE
score. Search coverage still affects deeper recall: fixed-pool R@50 rose from
`75.0000/75.3333/73.0000/73.3333` to
`78.0000/78.3333/76.6667/75.6667` for RTX/control/warmup/clip under the same
fixed-pool condition. It produced no Top-1 or Top-5 recovery.

This conclusion is causal at the level of the coupled hardware/runtime
condition. It does not identify whether the remaining contribution comes from
the Blackwell versus Ampere architecture, driver `580.126.09` versus
`535.161.07`, or their interaction.

The position-score decomposition then tested all `3,061` normalized global
weightings over code positions `0/1/2/3` and stop control=`1`. No A800 weighting
passed R@1/R@10/R@50=`27.3683/69.0000/75.0000`; conditions are Official
core75=`75` queries and the same exhaustive fixed pool. Workspace control had
`668/3,061` deep-gate configurations, but their best R@1 was only `21.5905`.
Warmup=`500` and clip=`1.0` had `0/3,061` deep-gate configurations. Five-fold
heldout R@1 was `20.9238/23.8794/21.2127` for control/warmup/clip, compared with
uniform `20.9238/23.9905/23.8794` under the same fixed-pool condition.

In RTX-correct/A800-wrong cases, the mean A800 gold-minus-best-nongold margins
were most negative at code positions `0` and `1`: control=`-1.6596/-1.2475`
over `9` queries, warmup=`-0.8922/-0.9212` over `9` queries, and
clip=`-2.0248/-1.1925` over `8` queries. The stop-control margin was effectively
zero. The preferred global weight direction changed across checkpoints and
folds, so a single position calibration does not explain or repair the gap.

The prefix/component attribution then localized the stable aggregate gap to the
second learned-code decision. Prefix R@1 at depth `1` was
`50.1302/50.4635/48.3524/51.1302` for RTX/control/warmup/clip; at depth `2` it
was `47.7968/39.7968/44.7968/43.0190`. Conditions are Official core75=`75`
queries, `200` gold mentions, and the same fixed pool. Exact Shapley attribution
over all `32` RTX-to-A800 component masks assigns c1
`45.9%/52.9%/48.5%` of the control/warmup/clip endpoint R@1 gap. c1 is the
largest component in all three comparisons, while stop contributes exactly
zero.

No partial learned-code component subset passed the fixed-pool
R@1/R@10/R@50=`27.3683/69.0000/75.0000` gate. Only swapping all four learned
code components reproduced RTX=`28.3683/69.0000/78.0000`; conditions are the
same `75` queries and exhaustive fixed pool. A c1-only swap raised A800 R@1 but
reduced R@10, so the repair target must preserve cumulative sequence ranking
rather than reweighting a single head.

## State and next decision

The matching RTX environment remains unavailable. Global backend controls,
optimizer reset/lr, query warmup=`500`, and query global L2 clip=`1.0` have not
met the recovery gate. Fixed-pool search and global position weighting are now
also excluded as explanations for the Top-1 gap. No position-weight candidate
qualifies for the original full-catalog beam evaluator.

The bounded training candidate retained CE and added query-conditioned hard
negatives at c1-diverging prefixes plus a cumulative-prefix or sequence-level
ranking objective across c0--c3. The strict-parity control remained the archived
base schedule: source rows=`94,912`, expanded rows=`124,175`, catalog
updates=`2,730`, query updates=`38,805`, batch=`16`, seed=`20260626`, optimizer
carry, BF16, and `CUBLAS_WORKSPACE_CONFIG=:16:8`. This keeps the training
question aligned with the RTX baseline.

The base candidate audit freezes summary SHA-256
`72b0f525d6c3d9feabc7770c054ee70d8478f1b280d1a100793da8d5b6c6dd6d`
and manifest SHA-256
`8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d`.
Teacher top-50 supplies c1-diverging negatives for
`86,305/124,175=69.5027%` of expanded base rows after sibling-GT exclusion.

All eight arms completed under this contract. Both disabled controls reproduced
the historical A800 model hash, predictions, best step, dev loss, and metrics
exactly. No ranking-loss arm passed the joint gate. Sequence-only lambda=`0.10`
reached Overall R@1/R@10/R@50=`28.7683/70.5873/73.8889`, Single R@1=`50.0000`,
and Multi R@1=`18.7768`; cumulative lambda=`0.02` reached
Overall=`26.3683/69.9556/77.1111`, Single R@1=`33.3333`, and Multi
R@1=`23.0906`. Conditions are dev-best checkpoint, Official core75=`75`
queries, beam=`50`, and beam-order. The former recovers Top-1 through Single
queries while losing Multi/deep recall; the latter preserves Multi/deep recall
but misses the Overall and Single Top-1 gates. No arm enters multi-seed locking.

The next bounded step is a paired per-query error-overlap analysis of the
sequence-only, cumulative lambda=`0.02`, and exact control predictions before
designing another GPU sweep. The target is to preserve the complementary
Single/Multi gains instead of increasing a global ranking weight.
The enhanced query-only result R@1/R@10/R@50=`27.26/69.33/76.78` remains a
separate method reference under source rows=`100,078`, global batch=`16`
(`8` ranks x `2`), lr=`3e-4`, query epochs=`5`, catalog passes=`0`, Official
core75=`75` queries, beam=`50`, and beam-order aggregation.
