# Experiment: A800 GR hard-focus continuation

Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

- Goal: improve Single R@1 beyond `11/24 = 45.8333%` (Official core75=`75`, beam=`50`, beam-order, CE log-prob) by concentrating updates on training-query ranking boundaries.
- Initialization: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_self_mined_nearmiss_20260813_v1/selected/single_top1_best` (step=`5,164`, SHA-256=`b55092f2be32ff063bff3f4387f154343dcd5f652573e4d044274df33d39cf1d`).
- Data: all `17,353` training queries whose gold is not top-1 under the frozen step=`2,400` mining run, plus `17,353` top-1-correct queries with the smallest gold-versus-best-negative score margins; total=`34,706` rows.
- Candidates: one gold plus the existing top `10` GR-self-mined non-gold SIDs for every selected row.
- Objective: positive token CE plus full-SID+STOP listwise CE; rank weight=`1.0`, temperature=`0.5`.
- Optimizer: fresh AdamW, `lr=5e-6`, weight decay=`0`, constant schedule, warmup ratio=`0`, BF16.
- Exposure: one focused epoch, per-device batch=`2`, world size=`8`, global batch=`16`, expected optimizer steps=`2,170`.
- Evaluation: Official core75=`75`, beam=`50`, beam-order, CE log-prob; save/evaluate every `100` steps plus final.

## Commands

Train: job=`tide-chenyufei03-10685-tadnk`; `SKILLGR_LAUNCH_COMMIT=453682a bash scripts/run_legacy_gr_hard_focus_node_20260814.sh` (one node=`8×A800`, queue=`rudder-exper`, image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`).

Eval: `24` model points (`cum_082`, prior self-mined best, `21` periodic checkpoints, and final) on Official core75=`75`, beam=`50`, beam-order, CE log-prob.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_hard_focus_20260814_v1/selected/single_top1_best`, resolving to the unchanged prior self-mined final (SHA-256=`b55092f2be32ff063bff3f4387f154343dcd5f652573e4d044274df33d39cf1d`).

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_hard_focus_20260814_v1/logs/node.log`.

Metrics:

- Prior self-mined best: Single R@1/R@10/R@50=`45.8333/54.1667/54.1667%`; overall R@1/R@10/R@50=`29.5905/70.9206/73.7778%` (Official core75=`75`, beam=`50`, beam-order, CE log-prob).
- Every trained point from step=`100` through final step=`2,170`: Single R@1/R@10/R@50=`45.8333/54.1667/54.1667%`; overall R@1/R@10=`29.5905/70.9206%` under the same protocol. No point improves top-1.
- Overall R@50 remains `73.7778%` for `9/22` trained points and falls to `73.3333%` for `13/22` trained points under the same protocol.

## Verification

- [x] Focused rows and candidates form an exact one-to-one query-bound closure (`34,706` rows, output hashes reproduced byte-for-byte from the frozen `82,621`-row source and eight mining prediction shards).
- [x] Selection contains all `17,353` mining top-1 errors plus `17,353` lowest-margin correct rows (step=`2,400` mining checkpoint, beam=`50`, training split=`82,621` single-GT queries).
- [x] Local checks pass (`3/3` focus-builder tests; `2/2` analyzer tests; four torch-dependent loss tests deferred to the fixed GPU runtime).
- [x] Job requests one node=`8×A800` on queue=`rudder-exper` (job=`tide-chenyufei03-10685-tadnk`).
- [x] Focused semantic self-review complete: row indices are reindexed, positive SID and query SHA-256 bindings are exact, each row has `10` unique non-gold candidates, and the training loader repeats the same closure audit.
- [x] Node checks pass (`9/9` tests under Python=`3.10.12`, torch=`2.10.0+cu128`, transformers=`5.3.0`, GPU count=`8`).
- [x] Training completes at step=`2,170`, all `24/24` evaluations complete, pod restarts=`0`.

## Result / Notes

Focused artifact:

- Rows: `/dataset_rw/SkillSearch-data-synthesis/processed/gr_hard_focus_20260814_v1/rows.jsonl`, SHA-256=`2892c779c481b741921f7b6ac8205f6acdab2e8905aaff5cc2ab041318f0f3e3` (`34,706` selected training queries).
- Candidates: `/dataset_rw/SkillSearch-data-synthesis/processed/gr_hard_focus_20260814_v1/candidates.jsonl`, SHA-256=`1ab3265ade090d0b1bcc062934eabd9b71c683e41de1db495c1cf1c2f5fda1ae` (`10` self-mined non-gold SIDs per selected query).

The focused continuation changes the full beam ordering on every Official core75 query by final step=`2,170`, but changes only `3/75` top-1 predictions; all three remain incorrect. It does not cross a relevant top-1 boundary. Keep the prior self-mined final as the A800 Single top-1 checkpoint and do not spend more GPU time extending this stale-negative focused trajectory.
