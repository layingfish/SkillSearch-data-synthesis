# Experiment: Diverse training-side query coverage with GR error filtering

Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

- Goal: improve Single R@1 beyond `45.8333%` while retaining Multi R@1 near `21.9468%` under Official core75=`75`, single/multi=`24/51`, beam=`50`, beam-order, CE log-prob, test-selected cum_082 checkpoint.
- Evidence: only `1/13` cum_082 Single-miss top-1 confusers under Official core75 appears among the top `3` confusers mined from `8` existing training-side pseudo views per corresponding gold skill.
- Hypothesis: query-scenario coverage is the active bottleneck. Generate exactly `6` new views for each of the `1,722` zero-multi target skills, using six fixed diversity modes and the existing `8` views as exclusions.
- Generation budget: `10,332 = 1,722 × 6` selected rows from `1,722` API requests. Modes are `implicit_outcome`, `artifact_pipeline`, `workflow_stage`, `failure_recovery`, `verification_first`, and `terse_natural`.
- Filtering: decode all generated rows with cum_082 on one node=`8×A800`, beam=`50`; retain only rows whose gold is not top-1. Candidate negatives remain bound to the exact generated query.
- Training mass: pending mining statistics. The builder will preserve a fixed number of pseudo exposures per catalog skill and will not append the new rows on top of the catalog-balanced schedule.

## Commands

Generate: `uv run python scripts/build_diverse_coverage_queries.py generate` with target skills=`1,722`, views/target=`6`, workers=`8`, temperature=`0.9`, top_p=`0.95`, max_tokens=`4,096`, max_query_chars=`1,500`, existing/pairwise token-Jaccard limits=`0.78/0.72`, and timeout=`240s`. The initial run allowed `3` attempts/target; the final `6` unresolved targets were resumed with a ceiling of `5` attempts/target. Full output: `/dataset_rw/SkillSearch-data-synthesis/processed/diverse_coverage_20260814_v1/full_v1`.

Prepare mining: `uv run python scripts/build_diverse_coverage_queries.py prepare-mining` with expected targets/views=`1,722/6`. Output: `/dataset_rw/SkillSearch-data-synthesis/processed/diverse_coverage_20260814_v1/mining_input_cum082_v1/mining_rows.jsonl`.

Mine: `scripts/run_diverse_coverage_mining_node_20260814.sh`; one node=`8×A800`, queue=`rudder-exper`, image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`, cum_082 checkpoint SHA-256=`b55092f2be32ff063bff3f4387f154343dcd5f652573e4d044274df33d39cf1d`, beam=`50`, negatives/query=`10`.

Train: pending; one node=`8×A800`, queue=`rudder-exper`, image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`.

Eval: Official core75=`75`, single/multi=`24/51`, beam=`50`, beam-order, CE log-prob; checkpoint selection uses this requested test protocol.

## Outputs

Checkpoint: pending.

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_diverse_coverage_mining_20260814_v1/logs/node.log` after submission.

Metrics: pending.

## Verification

- [x] Minimal local check (`7/7` focused unit tests, Python compile, Ruff, actual-data prompt preview)
- [x] Exactly `6` unique modes and queries per target skill
- [x] Query text is distinct from all `8` existing views for the same target
- [ ] Generated rows and mined candidates have exact row/hash closure
- [ ] Fixed per-skill training mass and optimizer-step closure
- [ ] Reviewer required? no; perform focused local semantic review before GPU training

## Result / Notes

Generation completed with `10,332/10,332 = 1,722 × 6` selected rows and `0` failed targets under workers=`8`, temperature=`0.9`, top_p=`0.95`, max_tokens=`4,096`, max_query_chars=`1,500`, and final attempt ceiling=`5`. The selected file SHA-256 is `1240ca2f91524fb0dc7e623a5ccbd3d930c9e87df0b30b52dfc6499ffdd86940`; its generation manifest SHA-256 is `0631ce689ae838223ebfdb154ea8548965d47693b024a88aff3ced941e2a9fc1`.

Under the same full-generation output, all `10,332` indices and normalized query texts are unique. Query length is `139–1,301` characters (mean=`606.2997`). Maximum token Jaccard against the same target's `8` existing views is `0.5354` with limit=`0.78`; maximum within-new-set token Jaccard is `0.5678` with limit=`0.72`. Each mode contributes exactly `1,722` rows.

Mining preparation completed with `10,332` contiguous rows and exact `1,722 × 6` target/mode closure. Mining-row SHA-256 is `9a442c06ee2793fa3bf792cc0373c59bff8f3f1b95bb3877f717eb517025f86e`; preparation-manifest SHA-256 is `c9ecac076b3399c41912c9c588087c3230f6f3ca5040bb11aa162adb4af87b7e`.
