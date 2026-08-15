# M0/Qwen3/CF-aware RIPOR with query-balanced multi-positive ranking

This run keeps the existing SkillRouter teacher and reuses the completed faithful
M0. RQ assignment uses row-normalized M0, base Qwen3-Embedding-0.6B, and LightGCN
CF vectors with weights `0.5/0.3/0.2`. Qwen and CF vectors are mapped to the M0
768-d space by semi-orthogonal Procrustes. Fixed assignments are followed by
codebook refitting in the original M0 space.

M2, M3_prefix4, M3_full, and M4 batch source queries. Every distinct gold active
prefix is ranked against a sampled non-gold prefix with the original signed RIPOR
MarginMSE. Pair losses are averaged within each query, then across source queries.
Single-positive queries are mathematically identical to the original objective.

M1 is retrained for the new Semantic IDs. All stage checkpoints are selected only
on Official core75 test with beam=50 using the existing lexicographic rule.
Training hyperparameters, candidates, teacher, beam settings, and checkpoint order
are unchanged.

For the exact server environment, asset contract, execution modes, and
cross-server comparison procedure, see
[`COMPANY_SERVER_REPRODUCTION.md`](COMPANY_SERVER_REPRODUCTION.md).
