# CF-aware RIPOR, Official-test checkpoint selection

This run reuses the completed faithful M0 dense encoder and all immutable dense
candidate/teacher artifacts from `ripor_faithful_cat3query5_20260813`.

The only method change before M1 is the Semantic-ID assignment. Normalized
LightGCN CF embeddings (`dim=32`, `layers=3`, `epochs=500`, `seed=42`) are mapped
into the normalized M0 space by semi-orthogonal Procrustes and fused at
`cf_weight=0.5`. Faiss RQ uses the fused vectors for the fixed `depth=8`,
`V=256` assignments. The codebooks used to initialize M1 are then refitted in
the original 768-dimensional M0 space under those fixed assignments.

M1, M2, M3-prefix4, M3-full, and M4 each run for five epochs with the unchanged
faithful losses and effective batch sizes. Every epoch is evaluated only on the
fixed Official core75 test (`75` queries, Single/Multi=`24/51`, `beam=50`). The
selection key is Overall R@1, balanced Single/Multi R@1, Multi R@50, Overall
R@10, then the earlier epoch. No dev split participates in checkpoint selection.

Beam mining retains Top-100 SIDs, `query_batch_size=4`, and
`decoder_batch_size=1024`. Trie transitions are materialized once on the GPU;
the established batch boundary remains unchanged.
