# Experiment: A800 sequence-level hard-negative contrastive fine-tuning

Status: done
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

Train: active job=`tide-chenyufei03-10681-njpcv`, launch commit=`2a48268`; `bash scripts/run_legacy_sequence_contrastive_hn10_node_20260813.sh` inside image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`, queue=`rudder-exper`, node=`1×8 A800`.

Eval: the launcher first anchors `cum_002@28,400`, then evaluates periodic step=`{200,400,...,5,000}` and final step=`5,164` in parallel over all `8×A800`; each evaluation uses Official core75=`75`, beam=`50`, beam-order, CE log-prob.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_contrastive_hn10_20260813_v2`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_contrastive_hn10_20260813_v2/logs/node.log`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_contrastive_hn10_20260813_v2/sequence_contrastive_checkpoint_sweep_summary.json`.

## Verification

- [x] Candidate data closes over all `82,621` rows with one gold and `10` unique non-gold skills. Compact artifact SHA-256=`c60b53655bf389340de1a791061c264d4dc3c65feb4390f20c62f1720cbad330`.
- [x] Disabled sequence loss preserves the original forward output exactly.
- [x] Batched path scores equal stepwise decoder log-probability sums including STOP (FP32 allclose, atol=`1e-6`, rtol=`0`).
- [x] Minimal local check: pipeline tests=`74/74`; focused new tests=`6/6`; Python compile、Ruff（忽略项目既有的 `2` 个 F401）和 shell syntax 通过。
- [x] Launch dry-run requests node=`1` on queue=`rudder-exper`; launcher asserts GPU count=`8` and starts DDP world size=`8`.
- [x] Focused local semantic review and invariant tests passed; independent reviewer not required.

## Result / Notes

Training and all-checkpoint evaluation completed successfully in job=`tide-chenyufei03-10681-njpcv` with pod restarts=`0`. The launcher verified query steps=`5,164` (training rows=`82,621`, global batch=`16`, one epoch), periodic checkpoints=`25` (save interval=`200`), and evaluations=`27` (init + `25` periodic + final).

Both selection objectives choose checkpoint step=`2,400`, SHA-256=`9d40adfd3125f4ec60c8c56940b4f0018fb4041ae3e07f6736d826cdd4c1b7e8`:

- Overall R@1/R@10/R@50=`28.2571/71.1429/75.5556%` (Official core75=`75` queries, beam=`50`, beam-order, CE log-prob), versus init=`28.2571/70.0000/75.1111%` under the same protocol.
- Single R@1/R@10/R@50=`41.6667/54.1667/58.3333%` (Official core75 single subset=`24` queries, beam=`50`, beam-order, CE log-prob), versus init=`41.6667/50.0000/58.3333%` under the same protocol.
- Multi R@1/R@10/R@50=`21.9468/79.1317/83.6601%` (Official core75 multi subset=`51` queries, beam=`50`, beam-order, CE log-prob), versus init=`21.9468/79.4118/83.0065%` under the same protocol.

The step=`2,400` ranking changes top-1 for `10/75` queries under the same Official core75 protocol: `3/24` single and `7/51` multi. Corrected top-1 queries=`0`; regressed top-1 queries=`0`. The multi changes swap one gold skill for another gold skill, while the single changes remain wrong-to-wrong. Top-10 order changes for `71/75` queries and top-50 order changes for `75/75` queries.

Conclusion: full-SID listwise training with `10` static SkillsRouter negatives changes rankings and improves deep recall, but this configuration does not improve Single top-1. The next top-1 experiment should mine generator-native near-miss negatives from the current checkpoint and optimize the gold-versus-current-top1 margin; the static external negatives are insufficiently aligned with the observed top-1 errors.

首次 job=`tide-chenyufei03-10680-pglnl` 命中固定 runtime 和 GPU count=`8`，随后在输入检查阶段失败：该 pod 没有 `/dataset_rw` alias。失败发生在模型加载、锚点评测和 optimizer step=`0` 之前。v2 launcher 改用 `/mnt/cfs_algo_bj/workspace/chenyufei03/datasets` 物理 CFS 路径。

v2 job=`tide-chenyufei03-10681-njpcv` 的固定 runtime 与节点测试=`6/6` 通过。训练前锚点评测精确恢复 `cum_002@28,400`：Overall R@1/R@10/R@50=`28.2571/70.0000/75.1111%`，Single R@1=`41.6667%`，Multi R@1=`21.9468%`（Official core75=`75` queries、beam=`50`、beam-order、CE log-prob）。随后启动 DDP world size=`8` 的正式训练。
