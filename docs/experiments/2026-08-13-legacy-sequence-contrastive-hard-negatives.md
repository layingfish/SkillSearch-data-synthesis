# Experiment: A800 sequence-level hard-negative contrastive fine-tuning

Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

- Goal: improve Official core75 Single top-1 from the `cum_002@28,400` A800 baseline while keeping pure generator retrieval at inference.
- Hypothesis: full-SID sequence supervision over confusable skills corrects generator ordering errors that positive-only token CE leaves unresolved.
- Initialization: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v5/selected/joint_best`.
- Training data: `82,621` single-label query rows; each row uses one gold SID and exactly `10` static SkillsRouter hard negatives.
- Objective: positive token CE plus listwise CE over `11` complete paths. Each path score sums full-vocabulary log-probabilities for `c0+c1+c2+c3+STOP`; `lambda_rank=1.0`, temperature=`1.0`.
- Optimizer: AdamW, `lr=1e-5`, weight decay=`0`, constant schedule, warmup ratio=`0`, BF16.
- Exposure: one epoch, per-device batch=`2`, world size=`8`, global batch=`16`, no-padding distributed sampler.
- Evaluation: Official core75=`75` queries, constrained beam=`50`, beam-order ranking, CE log-probability scoring. Checkpoint selection uses Official core75 retrieval metrics as requested.

## Commands

Train: job=`tide-chenyufei03-10680-pglnl`, launch commit=`b8719de2f1baedc3a45d0fe5cc0d16e958367226`; `bash scripts/run_legacy_sequence_contrastive_hn10_node_20260813.sh` inside image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`, queue=`rudder-exper`, node=`1×8 A800`.

Eval: the launcher first anchors `cum_002@28,400`, then evaluates periodic step=`{200,400,...,5,000}` and final step=`5,164` in parallel over all `8×A800`; each evaluation uses Official core75=`75`, beam=`50`, beam-order, CE log-prob.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_contrastive_hn10_20260813_v1`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_contrastive_hn10_20260813_v1/logs/node.log`

Metrics: pending.

## Verification

- [x] Candidate data closes over all `82,621` rows with one gold and `10` unique non-gold skills. Compact artifact SHA-256=`c60b53655bf389340de1a791061c264d4dc3c65feb4390f20c62f1720cbad330`.
- [x] Disabled sequence loss preserves the original forward output exactly.
- [x] Batched path scores equal stepwise decoder log-probability sums including STOP (FP32 allclose, atol=`1e-6`, rtol=`0`).
- [x] Minimal local check: pipeline tests=`74/74`; focused new tests=`6/6`; Python compile、Ruff（忽略项目既有的 `2` 个 F401）和 shell syntax 通过。
- [x] Launch dry-run requests node=`1` on queue=`rudder-exper`; launcher asserts GPU count=`8` and starts DDP world size=`8`.
- [ ] Reviewer required? no; focused local semantic review and invariant tests required.

## Result / Notes

Pending.
