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

Mine: `scripts/run_diverse_coverage_mining_node_20260814.sh`; job=`tide-chenyufei03-10713-kbwck`, one node=`8×A800`, queue=`rudder-exper`, image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`, cum_082 checkpoint SHA-256=`b55092f2be32ff063bff3f4387f154343dcd5f652573e4d044274df33d39cf1d`, beam=`50`, negatives/query=`10`.

Train: `scripts/run_diverse_error_fixed_mass_training_node_20260814.sh`; one node=`8×A800`, queue=`rudder-exper`, image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`. Initialization and optimizer settings match the prior fixed-mass run: cum_082 initialization, AdamW lr=`1e-5`, weight decay=`0`, constant schedule, BF16, batch=`2/GPU`, global batch=`16`, listwise loss weight=`1.0`, temperature=`0.5`, negatives/query=`10`, optimizer steps=`9,349`, and checkpoint interval=`200` steps.

Eval: Official core75=`75`, single/multi=`24/51`, beam=`50`, beam-order, CE log-prob; checkpoint selection uses this requested test protocol.

## Outputs

Checkpoint: pending.

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_diverse_coverage_mining_20260814_v1/logs/node.log` after submission.

Metrics: cum_082 reaches generated-query R@1/R@10/R@50=`53.6489/74.6612/80.3039%` before filtering under `10,332` training-side rows, beam=`50`, beam-order, and CE log-prob.

## Verification

- [x] Minimal local check (`7/7` focused unit tests, Python compile, Ruff, actual-data prompt preview)
- [x] Exactly `6` unique modes and queries per target skill
- [x] Query text is distinct from all `8` existing views for the same target
- [x] Generated rows and mined candidates have exact row/hash closure
- [x] Fixed per-skill training mass and candidate/query-hash closure
- [ ] Optimizer-step closure after training
- [ ] Reviewer required? no; perform focused local semantic review before GPU training

## Result / Notes

Generation completed with `10,332/10,332 = 1,722 × 6` selected rows and `0` failed targets under workers=`8`, temperature=`0.9`, top_p=`0.95`, max_tokens=`4,096`, max_query_chars=`1,500`, and final attempt ceiling=`5`. The selected file SHA-256 is `1240ca2f91524fb0dc7e623a5ccbd3d930c9e87df0b30b52dfc6499ffdd86940`; its generation manifest SHA-256 is `0631ce689ae838223ebfdb154ea8548965d47693b024a88aff3ced941e2a9fc1`.

Under the same full-generation output, all `10,332` indices and normalized query texts are unique. Query length is `139–1,301` characters (mean=`606.2997`). Maximum token Jaccard against the same target's `8` existing views is `0.5354` with limit=`0.78`; maximum within-new-set token Jaccard is `0.5678` with limit=`0.72`. Each mode contributes exactly `1,722` rows.

Mining preparation completed with `10,332` contiguous rows and exact `1,722 × 6` target/mode closure. Mining-row SHA-256 is `9a442c06ee2793fa3bf792cc0373c59bff8f3f1b95bb3877f717eb517025f86e`; preparation-manifest SHA-256 is `c9ecac076b3399c41912c9c588087c3230f6f3ca5040bb11aa162adb4af87b7e`.

Mining job=`tide-chenyufei03-10713-kbwck` completed with pod restart=`0` in about `6` minutes on one node=`8×A800`. cum_082 retrieves gold at top-1 for `5,543/10,332 = 53.6489%` generated rows and within beam=`50` for `8,297/10,332 = 80.3039%` rows under beam-order CE log-prob. All shards have exactly `50` ranked skills per query; all `10,332` queries have known gold and no source truncation under max_source_length=`2,048`.

Error filtering retains `4,789/10,332 = 46.3511%` rows across `1,364/1,722` target skills. It yields `4,050` unique `(gold, top-1 confuser)` pairs and exactly `10` query-bound negatives per retained row. Error counts by mode under `1,722` rows/mode are: implicit outcome=`866`, terse natural=`851`, failure recovery=`804`, workflow stage=`798`, verification first=`746`, and artifact pipeline=`724`.

Filtered rows SHA-256=`ec020458c2fb71c75ee610c1b27b7181842f4f47ff522da93e93c93342ee92de`; filtered candidates SHA-256=`8069a12876d2a2c42fe150e8a89ccf2c5668c0dcb12e105d888f04503599e14f`; filter manifest SHA-256=`ae710c7cfe7c4f33e356c12e5c129551fa8bfddea081dad75bb8c9c82e4f0c86`. Candidate compaction SHA-256=`eebcd0d528e753135bd3ae0c8f68f53d18d5359b94a54357931f43365c20a9a1`.

For the `13` cum_082 Single misses under Official core75=`24` single queries and beam=`50`, `12/13` gold skills have at least one retained generated-query failure. The exact test-time wrong top-1 skill reappears for `1/13` golds as a generated-query top-1 confuser and for `2/13` golds among the generated-query top-10 candidates. The fixed-mass schedule therefore trains on the failed query itself and its bound candidates; it does not rely on reproducing the exact Official query/confuser pair.

Fixed-mass materialization completed with `149,570 = 2 × (2,030 + 14,551 × 5)` rows and the same number of candidate records. Each catalog skill contributes exactly `5` pseudo views/cycle and `10` over two cycles. Every retained error query appears exactly once; the remaining target slots use `12,431` cum_082-bound task5/mode3 views, while original and non-target rows use `132,350` step2400-bound candidate records. Training rows SHA-256=`dacdf84aa57fc24e10b9c61590cd1e1c450e3f7797c173e90dd1177f0863e249`; candidates SHA-256=`1bf33c59cfbf7cdbe6a0f51d5b1efe0828b5deef2a7bd0c7a52e100f61828f84`; manifest SHA-256=`2c4f7b9dcae23fba97dbc0b1ee562e1ca7a3d8327f48589b52582e1cc15921e6`.

The training loader independently validated all `149,570` row/candidate joins under learned SID codes: exactly `10` unique non-gold negatives/row, SHA-256 query binding for every row, and exclusion of all source-selected skills from the negative set.
