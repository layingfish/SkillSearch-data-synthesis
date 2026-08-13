# Experiment: A800 GR-self-mined near-miss ranking

Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

- Goal: improve Official core75 Single R@1 while retaining pure generative retrieval at inference.
- Mining checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_contrastive_hn10_20260813_v2/selected/single_top1_best` (step=`2,400`, SHA-256=`9d40adfd3125f4ec60c8c56940b4f0018fb4041ae3e07f6736d826cdd4c1b7e8`).
- Mining data: `82,621` single-label training queries only; full catalog=`14,551` skills; constrained beam=`50`; keep the top `10` unique non-gold SIDs per query.
- Training initialization: the same step=`2,400` mining checkpoint.
- Objective: positive token CE plus full-SID+STOP listwise CE over one gold and `10` GR-self-mined near-misses; rank weight=`1.0`, temperature=`0.5`.
- Optimizer: AdamW, `lr=1e-5`, weight decay=`0`, constant schedule, warmup ratio=`0`, BF16.
- Exposure: one epoch, per-device batch=`2`, world size=`8`, global batch=`16`, no-padding distributed sampler.
- Evaluation: Official core75=`75` queries, constrained beam=`50`, beam-order ranking, CE log-probability scoring; checkpoint selection uses the requested test-set metrics.

## Commands

Train: `bash scripts/run_legacy_gr_self_mined_nearmiss_node_20260813.sh` inside image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`; planned queue=`rudder-exper`, node=`1×8 A800`.

Eval: pending.

## Outputs

Checkpoint: pending.

Log: pending.

Metrics: pending.

## Verification

- [x] Local shard dry-run closes exactly over all `82,621` training rows with source SHA-256=`df7046845745d144592a9d91469be5fa9447b8a1c61df8724faf5a20f1129cb0`; shard sizes are `10,328×5 + 10,327×3` across `8` workers.
- [ ] Every row has one matching gold and `10` unique known non-gold SIDs.
- [x] Minimal local check: focused tests=`9/9`; Python compile, Ruff, shell syntax, and embedded helper SHA verification pass.
- [ ] Job requests one node=`8×A800` on queue=`rudder-exper`.
- [ ] Focused semantic self-review complete.

## Result / Notes

Pending.
