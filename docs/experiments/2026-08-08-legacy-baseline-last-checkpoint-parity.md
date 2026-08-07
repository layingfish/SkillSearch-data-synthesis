# Experiment: Legacy baseline last-checkpoint parity
Status: done
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
`128`, and retrieval batch `1` with matching Transformers and Tokenizers versions. It compares padded
`input_ids` and `attention_mask` in addition to unpadded token sequences.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_sweep_20260808_v3/legacy-c3-q5-seed20260626/last`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_last_parity_20260808_v1/logs/job.log`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_last_parity_20260808_v1/summary.json`

Job: `tide-chenyufei03-10529-cwknk` (`rudder-exper`, one 8xA800 node, launch commit `b5eb09f`)

## Verification

- [x] Minimal local check: Python compile, shell syntax, 71 pipeline tests, padded-tokenizer test
- [x] Immutable artifact and protocol preflight passed locally
- [x] Official core75 produces 75 unique rankings
- [x] Overall, single-skill, and multi-skill metrics are rescored for both checkpoints
- [x] Token IDs are compared over all train, dev, catalog, and Official core75 inputs
- [x] Focused reviewer found no remaining critical or high findings

## Result / Notes

The new last checkpoint still does not reproduce the archived baseline. Results below are macro recall in
percent under one shared protocol: Official core75 with `75` queries and `200` gold mentions, beam `50`,
beam-order aggregation, CE log-prob scoring, max source length `2048`, and 8xA800 distributed evaluation.
The new arm is catalog epochs `3`, query epochs `5`, seed `20260626`, terminal query step `40420`.

| Subset | Rows / gold | Checkpoint | R@1 (%) | R@10 (%) | R@50 (%) |
|---|---:|---|---:|---:|---:|
| Overall | 75 / 200 | Archived final | 29.32 | 69.04 | 78.56 |
| Overall | 75 / 200 | New last, step 40420 | 26.48 | 67.25 | 75.33 |
| Single-skill | 24 / 24 | Archived final | 54.17 | 58.33 | 62.50 |
| Single-skill | 24 / 24 | New last, step 40420 | 37.50 | 50.00 | 54.17 |
| Multi-skill | 51 / 176 | Archived final | 17.63 | 74.08 | 86.11 |
| Multi-skill | 51 / 176 | New last, step 40420 | 21.29 | 75.37 | 85.29 |

The new last checkpoint has `0/75` exact Top-50 ranking rows against the archive under that protocol. The
single-skill gap is `-16.67/-8.33/-8.33` points at R@1/R@10/R@50. The multi-skill delta is
`+3.66/+1.29/-0.82` points under the same cutoffs. Selecting terminal step `40420` recovers part of the
new dev-best checkpoint's R@50 loss, but it does not restore historical parity.

Tokenizer serialization is not the cause. The archived and new tokenizers produce identical unpadded
tokens, padded `input_ids`, and `attention_mask` on all `146468` audited rows: expanded train `129341`
at batch `16`, dev `2501` at batch `128`, catalog `14551` at batch `16`, and Official core75 `75` at
batch `1`; every mismatch count is zero. Vocab, special tokens, pad ID, padding side, and truncation side
also match. The audit used Python `3.11`, Transformers `4.57.6`, Tokenizers `0.22.2`, SentencePiece
`0.2.1`, and CPU tensor construction.

The cluster job completed the 8xA800 retrieval, then exited during the first tokenizer audit because the
archived tokenizer config stores `extra_special_tokens` as a list. The evaluator already had a compatible
loader; the audit now uses the same fallback. Tokenizer audit, subset rescoring, and summary validation
were completed from the immutable cluster predictions. Final summary:
`/models_rw/SkillSearch-data-synthesis/experiments/legacy_last_parity_20260808_v1/summary.json`.

Checkpoint selection and tokenizer differences are excluded. The remaining mismatch is in the trained
weights and is consistent with the previously observed A800-versus-archived training trajectory drift.
The catalog sweep remains invalid for comparison with the historical baseline.
