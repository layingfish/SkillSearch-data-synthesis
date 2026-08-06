# Experiment: Classic document augmentation catalog exposure
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Replace the previous sequential catalog-only warm-start comparison with a unified
document-augmentation stream. Each logical query epoch visits every expanded
query row once. Catalog document-to-SID rows are interleaved in the same stream
and revisited until the requested catalog exposure is reached.

The query target remains expanded single-SID CE. The catalog/query model uses
fresh AdamW, a fresh constant scheduler, and global step `0`; no optimizer state
is carried from a prior stage. The frozen candidate pool, LETTER semantic IDs,
data splits, beam evaluation, and loss implementation are unchanged.

Dataset counts for this experiment are `14,551` catalog rows and `129,341`
expanded query rows. Query exposure is fixed at `5` logical epochs. The four
catalog arms use `0`, `3`, `6`, and `12` effective catalog passes. Their exact
per-query-epoch schedules are:

| Arm | Catalog pass schedule over query epochs 1-5 | Catalog row visits | Catalog/query sample ratio | CE tokens | Optimizer steps |
|---|---|---:|---:|---:|---:|
| `query-only` | `[0, 0, 0, 0, 0]` | `0` | `0` | `3,233,525` | `40,420` |
| `mixed-c3` | `[1, 0, 1, 0, 1]` | `43,653` | `0.0675` | `3,451,790` | `43,150` |
| `mixed-c6` | `[1, 1, 2, 1, 1]` | `87,306` | `0.1350` | `3,670,055` | `45,879` |
| `mixed-c12` | `[2, 3, 2, 3, 2]` | `174,612` | `0.2700` | `4,106,585` | `51,335` |

Counts assume `5` target tokens per expanded single-SID row and global batch
`16` (`8` ranks x `2` rows). Each arm initializes T5-base from the same base
model and creates fresh AdamW and scheduler state at global step `0`.

## Literature reference

- DSI reports sequential indexing then retrieval as mediocre and finds a
  co-training indexing:retrieval sample ratio near `32:1` on NQ320K; it trains
  up to `1,000,000` updates with global batch `128`.
- NCI concatenates ground-truth, DocT5Query, and document-as-query examples;
  its official code defaults to `500` epochs. On NQ320K it uses `15` DocT5Query
  queries plus `11` document-as-query views per document, about `19.3:1`
  augmented:labeled rows before any repeated training passes.
- MINDER uses one stream with labeled query:substring:pseudo-query `3:10:5`
  and `800,000` updates; there is no standalone catalog epoch.
- DSI-QG uses `50` selected generated queries per NQ document and trains the
  DSI indexing input for up to `1,000,000` updates with global batch `256`;
  generated queries replace document indexing inputs rather than defining a
  separate catalog epoch.
- RIPOR/PAG use `10` pseudo queries per document for seq2seq pretraining;
  RIPOR's pretraining is `250,000` updates and its later rank-oriented stage
  is commonly `120` epochs. Those ranking epochs are not catalog epochs.

## Commands

Train: four one-node jobs, global batch `16` (`8` ranks x `2`), T5-base,
constant lr `3e-4`, weight decay `0`, warmup ratio `0`, bf16, max source length
`2048`, seed `20260626`, query logical epochs `5`, catalog passes `0/3/6/12`.

Eval: private `1,000` rows and Official core75 `75` rows, beam `50`,
`beam_order`, macro `R@1/R@10/R@50`; recompute overall/single/multi from the
same predictions.

Rudder jobs (`1` node and `8` A800 GPUs per arm):

- `mixed-c12`: `tide-chenyufei03-10468-kjmrb` on `rudder-exper`
- `mixed-c6`: `tide-chenyufei03-10469-cinsp` on `rudder-exper`
- `mixed-c3`: `tide-chenyufei03-10475-ujgsu` on `rudder` (migrated from pending `tide-chenyufei03-10470-mbmpm`; the only free `rudder-r2` node was marked `GPUUnhealthy`)
- `query-only`: `tide-chenyufei03-10471-fvsel` on `rudder`

## Outputs

Checkpoint root: `/models_rw/SkillSearch-data-synthesis/experiments/document_augmentation_exposure_20260806_v1/`

## Verification

- [x] Local sampler invariant: each query row is visited `5` times and each
  catalog row is visited exactly `0/3/6/12` times per arm.
- [x] Local parser/sampler smoke and targeted unit tests (`80` passed and `12`
  environment-dependent tests skipped on the CPU workspace).
- [x] Remote launch preflight and checkpoint manifest audit.
- [x] Retrieval metrics with subset recomputation.

## Result / Notes

All four arms completed their registered optimizer-step counts, exposure checks,
and evaluation. The table below reports macro recall in percent on Official
core75 only: `75` rows total (`24` single-skill and `51` multi-skill), beam
`50`, `beam_order`, and the same predictions for `all/single/multi` rescoring.

| Arm | Catalog passes | All R@1 | All R@10 | All R@50 | Single R@1 | Single R@10 | Single R@50 | Multi R@1 | Multi R@10 | Multi R@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `query-only` | 0 | 27.26 | 69.33 | 76.78 | 41.67 | 54.17 | 54.17 | 20.48 | 76.47 | 87.42 |
| `mixed-c3` | 3 | 18.58 | 59.73 | 69.33 | 25.00 | 37.50 | 50.00 | 15.56 | 70.19 | 78.43 |
| `mixed-c6` | 6 | 22.88 | 65.23 | 75.22 | 33.33 | 50.00 | 58.33 | 17.96 | 72.40 | 83.17 |
| `mixed-c12` | 12 | 27.59 | 65.89 | 76.00 | 41.67 | 58.33 | 58.33 | 20.97 | 69.44 | 84.31 |

Against `query-only`, `mixed-c12` changes all-row R@1/R@10/R@50 by
`+0.33/-3.44/-0.78` points. Its single-skill R@10 and R@50 each improve by
`4.17` points, while multi-skill R@10 and R@50 fall by `7.03` and `3.10`
points. `mixed-c3` and `mixed-c6` are below `query-only` on all-row recall at
all three depths. Catalog exposure is therefore non-monotonic and does not
provide a broad retrieval gain under this protocol; the largest exposure
mainly trades multi-skill deep recall for single-skill deep recall and a small
all-row R@1 change.

The private `1,000`-row split was evaluated by the registered launcher but is
not used in this conclusion.
