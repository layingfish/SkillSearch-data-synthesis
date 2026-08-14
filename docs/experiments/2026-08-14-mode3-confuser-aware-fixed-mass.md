# Experiment: Confuser-aware single-query synthesis with fixed per-skill mass

Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

- Goal: improve Single R@1=`45.8333%` while retaining Multi R@1=`21.9468%` under Official core75=`75`, single/multi=`24/51`, beam=`50`, beam-order, CE log-prob, test-selected cum_082 checkpoint.
- Hypothesis: the remaining Single misses have weak positive-view coverage, while appending more equal-weight mode3 rows would again shift SID supervision mass. Query diversity and per-skill update mass must be separated.
- Targets: all `1,722` canonical catalog skills satisfying `multi_row_frequency(skill)=0` on training rows only.
- Mining: run cum_082 constrained decoding on exactly `8` existing pseudo views per target skill (`5` task5 + `3` mode3), then aggregate the top `3` unique non-gold confusers per target skill.
- Synthesis: generate one contrastive task query per `(target skill, confuser)` pair, for exactly `5,166` new rows. Prompts use catalog documents and training-side model errors only.
- Training data: each target skill has an `11`-view pool (`5` task5 + `3` mode3 + `3` contrastive). The builder materializes two exposure cycles. Each cycle contains exactly `5` pseudo views per catalog skill plus all `2,030` original single rows.
- Initialization: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_self_mined_nearmiss_20260813_v1/selected/single_top1_best`, checkpoint SHA-256=`b55092f2be32ff063bff3f4387f154343dcd5f652573e4d044274df33d39cf1d`.
- Training contract: initialize from cum_082; AdamW lr=`1e-5`, weight decay=`0`, constant schedule, BF16, DDP world size=`8`, batch=`2/GPU` and global batch=`16`. Use positive expanded CE plus full-SID+STOP listwise CE with weight=`1.0`, temperature=`0.5`, and at most `10` mined negatives per row. Run trainer epochs=`1` over the two-cycle materialization (`149,570` rows, expected optimizer steps=`9,349`). Save every `200` optimizer steps.

## Commands

Mine: job=`tide-chenyufei03-10706-ldvau`; `CONFUSER_MINING_LAUNCH_COMMIT=9395ec4 bash scripts/run_confuser_aware_mining_node_20260814.sh` on one node=`8×A800`, queue=`rudder-exper`, image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`.

Train: pending.

Eval: Official core75=`75`, single/multi=`24/51`, beam=`50`, beam-order, CE log-prob; checkpoint selection uses this requested test protocol.

## Outputs

Checkpoint: pending.

Log: pending.

Metrics: pending.

## Verification

- [x] Minimal local check (`4/4` focused unit tests, Python compile, Ruff, shell syntax)
- [x] Every mining row is one of the expected `13,776 = 1,722 × 8` target pseudo rows
- [ ] Every generated row has one known target and one distinct known confuser
- [x] Official rows are absent from mining and synthesis inputs; target selection reads training co-occurrence only
- [x] Fixed-mass builder test verifies exactly `5` pseudo rows per catalog skill in each of the two materialized exposure cycles
- [ ] Reviewer required? no; perform focused local semantic review before launch

## Result / Notes

Pending.

Mining input: `/dataset_rw/SkillSearch-data-synthesis/processed/confuser_aware_mode4_20260814_v1/mining_input/mining_rows.jsonl`, SHA-256=`5b9002c69052717acdf6098bfb93300cb9923ff732601eb553e6d4bf9db8f4d5`. Canonical-owner selection discarded `20` alias-derived task5 rows: `4` canonical targets had an additional `5` mapped task5 rows each in the historical augmented file. The resulting input has exactly `5` canonical task5 plus `3` mode3 views for every target skill.

Mining job=`tide-chenyufei03-10706-ldvau` completed with pod restart=`0` in about `9` minutes (one node=`8×A800`). cum_082 retrieves the gold at top-1 for `12,221/13,776 = 88.7123%` target pseudo views and within beam=`50` for `13,475/13,776 = 97.8150%` views under beam-order CE log-prob. Candidate output contains exactly `10` unique non-gold SIDs for all `13,776` views; SHA-256=`a4aaa8bc6f81c7c2606340584f3e23104e14cbf59a572bf8df0a08d836fbb53c`.

Top-1-aware aggregation gives `5,166 = 1,722 × 3` pairs, SHA-256=`678c4b0b2805de6d43dfa24171cf1dfcccb212b1c4faaf90c6f3480d0a55588b`. It prioritizes SIDs that actually occupied top-1 on a failed training view, then fills remaining slots by absolute reciprocal rank. `709/1,722` targets have at least one top-1 failure across their `8` pseudo views; `1,136/5,166` selected pair slots have direct top-1-error support. The other slots are current runner-up boundaries for targets with fewer than `3` distinct top-1 errors.

The initial launcher omitted `--with-dataset-rw`; the lightweight pair output therefore landed in pod-local storage. GPU predictions and candidates were already durable under `/models_rw`, so the pair file was rebuilt locally without rerunning mining. The launcher now initializes `/dataset_rw` explicitly and uses a physical CFS output path.

Fixed-mass pre-audit: all `14,551` canonical skills have exactly `5` canonical-owner task5 rows. The historical single-row file also contains `2,030` original single queries and `2,670` alias-derived task5 rows that would duplicate canonical supervision. The planned two-cycle materialization contains `149,570 = 2 × (2,030 + 14,551 × 5)` rows; each canonical skill contributes exactly `10` pseudo exposures over both cycles.
