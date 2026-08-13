# Experiment: A800 GR-self-mined near-miss ranking

Status: done
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

Train: active job=`tide-chenyufei03-10684-iskwv`, launch commit=`b33f573`; `bash scripts/run_legacy_gr_self_mined_nearmiss_node_20260813.sh` inside image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`, queue=`rudder-exper`, node=`1×8 A800`.

Eval: the launcher evaluated the frozen `cum_002@28,400` anchor, the mining checkpoint, periodic step=`{200,400,...,5,000}`, and final step=`5,164` across all `8×A800`; every evaluation used Official core75=`75`, beam=`50`, beam-order, CE log-prob.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_self_mined_nearmiss_20260813_v1`.

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_self_mined_nearmiss_20260813_v1/logs/node.log`.

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_self_mined_nearmiss_20260813_v1/sequence_contrastive_checkpoint_sweep_summary.json`.

## Verification

- [x] Local shard dry-run closes exactly over all `82,621` training rows with source SHA-256=`df7046845745d144592a9d91469be5fa9447b8a1c61df8724faf5a20f1129cb0`; shard sizes are `10,328×5 + 10,327×3` across `8` workers.
- [x] Candidate artifact closes over `82,621/82,621` rows; every row has one matching gold and `10` unique known non-gold SIDs. SHA-256=`7eca29a2e31a9be194a148cbce7fe551c77590a1c13e73054c24ecb8c0a45c8a`.
- [x] Minimal local check: focused tests=`9/9`; Python compile, Ruff, shell syntax, and embedded helper SHA verification pass.
- [x] Tianzhi dry-run and submitted job request one node=`8×A800` on queue=`rudder-exper`.
- [x] Focused semantic self-review complete: mining uses training queries only; query SHA, gold ID, catalog membership, exact cardinality, checkpoint SHA, and evaluation protocol are fail-closed.

## Result / Notes

Job=`tide-chenyufei03-10684-iskwv` completed with pod restarts=`0`. Mining found gold at rank=`1` for `65,268/82,621` training queries and within top-50 for `81,568/82,621` training queries (mining checkpoint step=`2,400`, beam=`50`, beam-order, CE log-prob).

The Single-top1 selection chooses final step=`5,164`, checkpoint SHA-256=`b55092f2be32ff063bff3f4387f154343dcd5f652573e4d044274df33d39cf1d`:

- Single R@1/R@10/R@50=`45.8333/54.1667/54.1667%` (Official core75 single subset=`24`, beam=`50`, beam-order, CE log-prob), versus mining checkpoint=`41.6667/54.1667/58.3333%` under the same protocol.
- Overall R@1/R@10/R@50=`29.5905/70.9206/73.7778%` (Official core75=`75`, beam=`50`, beam-order, CE log-prob), versus mining checkpoint=`28.2571/71.1429/75.5556%` under the same protocol.
- Multi R@1/R@10/R@50=`21.9468/78.8049/83.0065%` (Official core75 multi subset=`51`, beam=`50`, beam-order, CE log-prob), versus mining checkpoint=`21.9468/79.1317/83.6601%` under the same protocol.

Compared with the mining checkpoint, final changes Single top-1 for `6/24` queries under the same Official protocol. It corrects `1` error and introduces `0` top-1 regressions. The corrected skill is `gt/manufacturing-failure-reason-codebook-normalization`, whose gold rank moves from `6` to `1`. Single R@50 loses one different query whose gold rank moves from `44` to outside top-50.

Conclusion: GR-self-mined near-misses produce the first measured Single top-1 gain in this line, from `10/24` to `11/24` under the requested checkpoint-selection protocol. The gain arrives only at final step=`5,164`; late rank loss remains active with mean=`1.3224` over logged steps=`4,200–5,164` (temperature=`0.5`). A short continuation from final is warranted without repeating mining.
