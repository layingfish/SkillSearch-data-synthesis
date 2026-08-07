# Experiment: Legacy baseline last-checkpoint parity
Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Determine why the new catalog sweep does not reproduce the archived baseline. Evaluate the new `c3`,
seed `20260626` last checkpoint at query step `40420`, matching the archived checkpoint's terminal query
step. Compare the archived and new saved tokenizers on every expanded query-train row, query-dev row,
catalog row, and Official core75 row. The launcher validates immutable artifact hashes, checkpoint
metadata, training completion, and both evaluation manifests before reporting parity. Do not use catalog
sweep deltas until this check is resolved.

## Commands

Eval and tokenizer audit:

```text
bash scripts/run_legacy_last_checkpoint_parity_20260808.sh
```

Evaluation: T5-base, catalog epochs `3`, query epochs `5`, seed `20260626`, last query step `40420`,
Official core75 with `75` queries, beam `50`, beam-order aggregation, CE log-prob scoring, and distributed
evaluation on one 8xA800 node. Tokenizer audit uses max source length `2048`, train batch `16`, dev batch
`128`, and retrieval batch `1` under the same training image. It compares padded `input_ids` and
`attention_mask` in addition to unpadded token sequences.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260808_v3/legacy-c3-q5-seed20260626/last`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_last_parity_20260808_v1/logs/job.log`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_last_parity_20260808_v1/summary.json`

Job: `tide-chenyufei03-10529-cwknk` (`rudder-exper`, one 8xA800 node, launch commit `9d9f6e6`)

## Verification

- [x] Minimal local check: Python compile, shell syntax, 71 pipeline tests, padded-tokenizer test
- [x] Immutable artifact and protocol preflight passed locally
- [ ] Official core75 produces 75 unique rankings
- [ ] Overall, single-skill, and multi-skill metrics are rescored for both checkpoints
- [ ] Token IDs are compared over all train, dev, catalog, and Official core75 inputs
- [x] Focused reviewer found no remaining critical or high findings

## Result / Notes

Pending.
