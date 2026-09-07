# LETTER OPQ replacement validation

## Question

Would replacing LETTER's residual quantizer with standard OPQ improve the SkillGR semantic IDs?

The word "replacement" hides an important method change. The formal project tokenizer is a nonlinear
RQ-VAE trained with semantic reconstruction, LightGCN collaborative alignment, and code diversity.
Standard OPQ learns a rotation and then applies independent product quantizers; it has no direct analogue
of LETTER's collaborative loss, and its code positions are not intrinsically coarse-to-fine.

This creates two separate hypotheses:

- **H1, minimal replacement:** D4 OPQ IDs are emitted by the existing autoregressive T5/trie decoder.
- **H2, architecture-aligned OPQ:** longer OPQ IDs are predicted with position-specific parallel heads and
  items are ranked by gathering and summing per-position scores over the catalog.

The current probe screens H1. H2 is a separate model branch, not a tokenizer-only ablation.

## Frozen offline protocol

- Catalog: the same 14,551 canonical skills and row order as the formal LETTER run.
- Semantic input: the same 1,024-d Qwen embeddings, with no additional normalization by default.
- CF audit input: the same row-aligned, 32-d LightGCN embeddings.
- Identifier budget: four positions, 256 values per position (32 bits total).
- Seed: 42.
- No core75 query, label, or metric may be used to fit a quantizer or select an OPQ code order.
- Full-tuple collisions use the same family of minimum-cost unique-last-code repair as learned-only LETTER.

Primary quantizer groups:

1. `pq`: plain parallel PQ, needed to verify that the OPQ rotation itself helps PQ.
2. `opq`: standard OPQ plus parallel PQ; this is the proposed replacement.
3. `rq`: Faiss full-vector residual quantization on the exact same semantic vectors; this isolates the
   linear quantizer comparison.
4. `letter_rqvae_cf`: the existing formal LETTER IDs; this is the actual system baseline.

Run two input panels when the tokenizer checkpoint is available:

- **Panel A (linear geometry):** fit PQ/OPQ/RQ directly on the same 1,024-d Qwen vectors.
- **Panel B (closest quantizer replacement):** freeze the trained LETTER encoder, export its 32-d latents,
  fit PQ/OPQ/RQ on those latents, and still compute prefix cohesion against the original 1,024-d semantic
  vectors plus 32-d CF vectors. Panel B is closer to changing only the discrete quantizer, although the
  frozen encoder was jointly learned with LETTER RQ and therefore remains mildly RQ-favored.

Optional `opq_rq` is only an optimization diagnostic. An orthogonal rotation preserves L2 geometry, so an
ideal full-vector RQ is rotation-equivariant; this group should not be described as standard OPQ.

For PQ and OPQ, all 24 permutations of the four parallel subquantizer IDs are audited. The selected order
lexicographically maximizes catalog-only semantic-neighbor prefix recall at depths 1, 2, and 3 (k=10),
with a lexicographic permutation tie-break. Reconstruction and full-tuple collisions are invariant to this
ordering; prefix search is not.

## Metrics and decision gate

The probe reports:

- raw and collision-repaired reconstruction distortion;
- semantic reconstruction-neighborhood overlap;
- per-position utilization and effective code count;
- unique/effective prefixes, maximum group size, and collision statistics at every depth;
- semantic and CF within-prefix variance;
- semantic and CF nearest-neighbor prefix recall at k=10 and k=50;
- native/selected/worst OPQ prefix behavior across all 24 code orders.

OPQ advances to a from-scratch decoder A/B only if it improves reconstruction over same-input RQ and its
semantic prefix metrics are non-inferior within the pre-registered absolute tolerance. When formal LETTER
IDs and CF embeddings are available, it must also be non-inferior to LETTER's semantic and CF prefix
metrics. Passing this gate is not an end-to-end win; final judgment still requires a fresh decoder with the
same data, initialization, training schedule, constrained beam, and core75/private-test evaluation.

Recent evidence sharpens this distinction. ACERec (arXiv:2602.13573v2, August 2026) uses OPQ successfully,
but with long IDs, parallel multi-token prediction, and holistic catalog scoring rather than autoregressive
prefix beam search. PRQ-KMeans (arXiv:2608.24207v1) instead retains residual hierarchy and reports gains from
improving residual construction and prefix utilization. Both are recent preprints and do not decide this
project's result, but they make it methodologically unsafe to infer H1 from an H2-style OPQ system.

## Command

```bash
python3 scripts/probe_letter_opq.py \
  --embeddings <canonical_skill_embeddings.pt> \
  --cf-embeddings <lightgcn_cf_embeddings.pt> \
  --letter-semantic-ids <formal_letter_semantic_ids.jsonl> \
  --output-dir <new_empty_output_dir> \
  --methods pq,opq,rq \
  --depth 4 --codebook-size 256 \
  --seed 42 --normalization none
```

For Panel B, replace `--embeddings` with the frozen latent matrix and add:

```bash
  --quantizer-space letter_latent \
  --semantic-audit-embeddings <canonical_skill_embeddings.pt>
```

The implementation requires the historically used `faiss-cpu==1.13.2`. It refuses to overwrite a
non-empty output directory. Generated manifests use `method=letter_opq_identifier_probe` and deliberately
do not impersonate the fixed `official_letter_rqvae_adapter` provenance contract.

## Current execution status

The private assets were restored and verified on 2026-09-01. All four catalog ID sources contain exactly
14,551 unique skills in the same row order; the known semantic/catalog/LETTER hashes match the handoff, and
the formal best RQ-VAE checkpoint loads strictly. The official LETTER source was sparse-checked-out at the
pinned commit `8d0154e28de37dbb6e24871c508ad8ddb1921cda` to export the frozen 32-d encoder output.

## Real-catalog result

Lower variance-normalized SSE is better. The table reports distortion after enforcing a collision-free
D4 identifier, because that is the artifact a decoder could actually consume.

| Panel | PQ | OPQ | same-input RQ | OPQ/RQ |
|---|---:|---:|---:|---:|
| A: 1,024-d semantic, seed 42 | 0.657485 | 0.967039 | 0.442951 | 2.18x |
| B: 32-d LETTER latent, seed 42 | 0.082913 | 0.096772 | 0.013904 | 6.96x |
| B: 32-d LETTER latent, seed 44 | — | 0.104230 | 0.013990 | 7.45x |

Panel B is the decisive quantizer-only approximation. Before collision repair, OPQ slightly improves PQ
(`0.029973` versus `0.030619`), which confirms that the OPQ rotation is functioning, but same-input RQ is
still much better (`0.013795`). OPQ produces only 7,743 raw unique tuples and requires repairing 9,545
items; RQ produces 13,866 unique tuples and repairs 1,221 items. The repair burden widens the distortion
gap rather than creating it.

The system-level prefix gate also fails. In Panel B, OPQ minus formal LETTER semantic-neighbor recall@10
at depths 1/2/3 is `[-0.0593, +0.0017, +0.0086]`; the CF deltas are
`[-0.0671, -0.0160, +0.0185]`. Both violate the pre-registered 0.01 non-inferiority tolerance. Seed 44
repeats the same decision. Seed 43 is worse operationally: its selected OPQ ordering contains a 284-item
three-position prefix, so a 256-way final code cannot make the catalog collision-free under the frozen
policy.

Therefore **H1 does not advance to decoder A/B**. Replacing the four LETTER residual codes with four
standard OPQ/PQ codes is not supported by either distortion, capacity, or prefix evidence. This does not
reject H2: a longer OPQ identifier with parallel heads and holistic catalog scoring remains a distinct
architecture-level research branch.

H2 was activated on 2026-09-02 after explicit user direction.  Its frozen OPQ-32/64 parallel-MTP protocol,
implementation, Tianzhi job, and Official-only result paths are recorded in
`docs/experiments/2026-09-02-letter-opq-parallel-mtp.md`; H2 does not reuse the autoregressive H1 decoder.

Reproducible reports:

- Panel A: `.tmp/letter_opq_panel_a_semantic_d4_c256_seed42_20260901/comparison.json`, SHA-256
  `ce0de3207c9535b3ef7da823251e609c2c42a78e27e6e8896e17584a6253f89c`.
- Panel B seed 42: `.tmp/letter_opq_panel_b_latent_d4_c256_seed42_20260901/comparison.json`, SHA-256
  `d99f241e2592a3231b7328d4a43757ddf128f993a44b5c879bd44d891a879d19`.
- Panel B seed 44: `.tmp/letter_opq_panel_b_latent_d4_c256_seed44_20260901/comparison.json`, SHA-256
  `c6718eb2205aef5ca2cc5662cb85c599cf8d27c55677523013887c3c7cf763fc`.
