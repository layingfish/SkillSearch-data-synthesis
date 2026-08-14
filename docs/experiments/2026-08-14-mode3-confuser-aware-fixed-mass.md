# Experiment: Confuser-aware single-query synthesis with fixed per-skill mass

Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

- Goal: improve Single R@1=`45.8333%` while retaining Multi R@1=`21.9468%` under Official core75=`75`, single/multi=`24/51`, beam=`50`, beam-order, CE log-prob, test-selected cum_082 checkpoint.
- Hypothesis: the remaining Single misses have weak positive-view coverage, while appending more equal-weight mode3 rows would again shift SID supervision mass. Query diversity and per-skill update mass must be separated.
- Targets: all `1,722` canonical catalog skills satisfying `multi_row_frequency(skill)=0` on training rows only.
- Mining: run cum_082 constrained decoding on exactly `8` existing pseudo views per target skill (`5` task5 + `3` mode3), then aggregate the top `3` unique non-gold confusers per target skill.
- Synthesis: generate one contrastive task query per `(target skill, confuser)` pair, for exactly `5,166` new rows. Prompts use catalog documents and training-side model errors only.
- Training data: each target skill has an `11`-view pool (`5` task5 + `3` mode3 + `3` contrastive); each query epoch selects exactly `5` pseudo views per catalog skill. Original non-pseudo training rows remain unchanged.
- Initialization: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_gr_self_mined_nearmiss_20260813_v1/selected/single_top1_best`, checkpoint SHA-256=`b55092f2be32ff063bff3f4387f154343dcd5f652573e4d044274df33d39cf1d`.
- Optimizer proposal: AdamW, lr=`1e-5`, weight decay=`0`, BF16, DDP world size=`8`, query epochs=`2`; final values remain pending until the fixed-mass row audit closes.

## Commands

Mine: `bash scripts/run_confuser_aware_mining_node_20260814.sh` on one node=`8×A800`, queue=`rudder-exper`; pending submission.

Train: pending.

Eval: Official core75=`75`, single/multi=`24/51`, beam=`50`, beam-order, CE log-prob; checkpoint selection uses this requested test protocol.

## Outputs

Checkpoint: pending.

Log: pending.

Metrics: pending.

## Verification

- [x] Minimal local check (`3/3` focused unit tests, Python compile, Ruff, shell syntax)
- [x] Every mining row is one of the expected `13,776 = 1,722 × 8` target pseudo rows
- [ ] Every generated row has one known target and one distinct known confuser
- [ ] Official query text and Official GT labels are absent from mining and synthesis inputs
- [ ] Every catalog skill contributes exactly `5` pseudo rows per query epoch
- [ ] Reviewer required? no; perform focused local semantic review before launch

## Result / Notes

Pending.

Mining input: `/dataset_rw/SkillSearch-data-synthesis/processed/confuser_aware_mode4_20260814_v1/mining_input/mining_rows.jsonl`, SHA-256=`5b9002c69052717acdf6098bfb93300cb9923ff732601eb553e6d4bf9db8f4d5`. Canonical-owner selection discarded `20` alias-derived task5 rows: `4` canonical targets had an additional `5` mapped task5 rows each in the historical augmented file. The resulting input has exactly `5` canonical task5 plus `3` mode3 views for every target skill.
