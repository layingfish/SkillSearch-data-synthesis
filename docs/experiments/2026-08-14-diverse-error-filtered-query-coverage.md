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

Generate: `uv run python scripts/build_diverse_coverage_queries.py generate` with target skills=`1,722`, views/target=`6`, workers=`8`, temperature=`0.9`, top_p=`0.95`, max_tokens=`4,096`, max_query_chars=`1,500`, existing/pairwise token-Jaccard limits=`0.78/0.72`, timeout=`240s`, and maximum attempts=`3`. Full output: `/dataset_rw/SkillSearch-data-synthesis/processed/diverse_coverage_20260814_v1/full_v1`.

Mine: pending; one node=`8×A800`, queue=`rudder-exper`, image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`.

Train: pending; one node=`8×A800`, queue=`rudder-exper`, image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`.

Eval: Official core75=`75`, single/multi=`24/51`, beam=`50`, beam-order, CE log-prob; checkpoint selection uses this requested test protocol.

## Outputs

Checkpoint: pending.

Log: pending.

Metrics: pending.

## Verification

- [x] Minimal local check (`4/4` focused unit tests, Python compile, Ruff, actual-data prompt preview)
- [ ] Exactly `6` unique modes and queries per target skill
- [ ] Query text is distinct from all `8` existing views for the same target
- [ ] Generated rows and mined candidates have exact row/hash closure
- [ ] Fixed per-skill training mass and optimizer-step closure
- [ ] Reviewer required? no; perform focused local semantic review before GPU training

## Result / Notes

Pending.

Generation smoke completed for the first `2` targets with `12/12 = 2 × 6` selected rows, no retry, and no rejected mode. Under workers=`2`, temperature=`0.9`, top_p=`0.95`, max_tokens=`4,096`, and max_query_chars=`1,500`, selected query lengths were `197–854` characters (mean=`703.08`), maximum token Jaccard against the same target's `8` existing views was `0.4016`, and maximum within-new-set token Jaccard was `0.3739`. The six prescribed modes all passed their layout constraints.
