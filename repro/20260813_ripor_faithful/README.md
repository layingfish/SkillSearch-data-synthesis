# Faithful RIPOR adaptation for SkillGR

This reproduction ports the complete RIPOR training chain to the fixed
cat3/query5 SkillGR data.  It deliberately does not reuse the earlier
`ripor-style` continuation checkpoints, because those checkpoints retained the
old LETTER identifiers and skipped RIPOR's relevance-trained identifier
initialization.

## Frozen sources

- Paper: Zeng et al., *Relevance-based Multi-granularity Learning for
  Generative Retrieval*, arXiv:2311.09134.
- Official implementation: `HansiZeng/RIPOR` commit
  `03fd0cbfafdeeb9cdd866e0c44727fec39a96a52` (2024-05-07).
- Official T5-base decoder-start embedding SHA256:
  `fe928c4929126abb23e1b22532d21aec65c9a723f2e5fb1f035cfea7da211f02`.
- SkillGR training rows SHA256:
  `ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5`
  (`94,912` source queries and `124,175` query/positive pairs under the
  cat3/query5 training split).
- Canonical catalog SHA256:
  `1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3`
  (`14,551` canonical skills in the full catalog).
- Official evaluation rows SHA256:
  `4adcd50e359e806d824ffa6f5ee29b4573321033e47ffa2a31780e65ff3cb8ba`
  (SkillsBench v1.1 core75: `75` queries, `24` single-skill queries and
  `51` multi-skill queries).

## Paper-to-code mapping

| RIPOR component | This implementation | Invariant checked before the next phase |
|---|---|---|
| M0 stage 1 | `build-bm25-candidates`, then `train-m0` from T5-base with teacher MarginMSE | BM25 Top-100, every GT excluded from negatives, one uniformly sampled negative per positive pair |
| M0 stage 2 | `mine-dense-candidates`, then a second `train-m0` continuation | M0 dense Top-100, signed teacher margins retained |
| Relevance SID | `build-rq` over final M0 skill embeddings with Faiss `IndexResidualQuantizer`, inner-product metric | raw residual-quantizer codes and centroids; collisions use official one-SID-to-many-documents expansion; no LETTER blend, RQ-VAE, dedup suffix, centroid normalization, or RMS rescaling |
| M1 | `train-m1` with token CE from a fresh RIPOR decoder initialized by M0 and raw RQ centroids | exactly the fixed RQ code sequence; code tokens only, no STOP loss |
| M2 | `train-rank` on final-M0 Top-100 candidates | full-sequence conditional-logit MarginMSE only; no generation CE |
| M3 | beam Top-100 mining from M2, union with final-M0 candidates, then progressive `train-rank` at prefix 4 and full depth | stage 4 optimizes prefix 4; full-depth stage jointly retains prefix-4 and full losses |
| M4 | beam Top-100 mining from M3, union with the M3 pool, then final `train-rank` | self-negative full-depth training retains every previous prefix objective |
| Retrieval | trie-constrained beam search with the sum of raw conditional logits | pure GR; no dense score, reranker, or teacher at inference |

## Deliberate dataset adaptations

RIPOR's MS MARCO setup uses a cross-encoder teacher and doc2query pseudo
queries.  The fixed SkillGR experiment makes only these two substitutions:

1. `pipizhao/SkillRouter-Embedding-0.6B` supplies training-only relevance
   scores.  Its normalized query/skill dot product is used without sign
   clipping, margin clipping, or a hand-tuned scale.  It is absent at
   inference.
2. The existing cat3/query5 task queries play the same M1 query-to-document
   supervision role as doc2query.  No new query data is introduced.

For a query with multiple GT skills, the source row expands to one pair per GT.
Every GT remains a positive.  Negatives exclude the complete GT set.  This
keeps RIPOR's pairwise MarginMSE unchanged and avoids inventing a multi-positive
loss.

## Identifier depth

The launch configuration uses `depth=8` and `codebook_size=256`.  Depth 8 is
the smallest official RIPOR depth that contains a genuine earlier optimization
target (`prefix_length=4`) before the full identifier.  Under the paper's
`beta=2` schedule, the target factors are `alpha_4=2/3` and `alpha_8=1` when
the final SID depth is 8.  These relative factors also match the released
`decay=2` preprocessing for an 8-code stage.

The RQ phase records full-SID collisions without changing their codes.  During
mining and evaluation, each generated SID expands to all assigned skills in
catalog order, matching the official `smtid_to_docids` behavior.  No custom
suffix is appended.

## Checkpoint and test policy

- Every phase consumes a completed manifest whose file hashes, row
  fingerprints, teacher identity, SID identity, and parent checkpoint hash
  match its command line.
- Training uses fixed epoch counts and saves the final checkpoint.  No private
  split or Official core75 metric selects a checkpoint.
- Official core75 is evaluated once, after M4, with trie-constrained
  conditional-logit beam search (`beam=50`, full `14,551`-skill catalog,
  code-only scoring).
- Unit tests compare the dense score, CE, conditional-logit prefix score,
  MarginMSE, progressive objective set, candidate exclusion, RQ initialization,
  and trie decoding against direct reference calculations.

## Frozen single-GPU schedule

The paper used eight 40GB A100 GPUs and a much larger MS MARCO training corpus.
The first SkillGR run keeps every method phase and freezes a smaller schedule
before looking at Official core75:

| Phase | Fixed training condition |
|---|---|
| M0 BM25 | `epochs=5`, effective `batch=16`, `microbatch=8`, `lr=1e-4`, `warmup=4.5%`, `max_length=2048`, BF16 |
| M0 dense-negative | `epochs=5`, effective `batch=16`, `microbatch=8`, `lr=1e-4`, `warmup=4.5%`, `max_length=2048`, BF16 |
| M1 | `epochs=5`, `batch=32`, `lr=1e-3`, `warmup=4.5%`, `max_length=768`, BF16 |
| M2 | `epochs=5`, effective `batch=64`, `microbatch=32`, `lr=1e-4`, `warmup=4.5%`, `max_length=768`, BF16 |
| M3 prefix 4 | `epochs=5`, effective `batch=64`, `microbatch=32`, `lr=1e-4`, `beta=2`, `alpha_4=2/3`, `max_length=768`, BF16 |
| M3 full depth 8 | `epochs=5`, effective `batch=64`, `microbatch=32`, `lr=1e-4`, objectives `{4,8}`, `alpha={2/3,1}`, `max_length=768`, BF16 |
| M4 | `epochs=5`, effective `batch=64`, `microbatch=32`, `lr=1e-4`, objectives `{4,8}`, `alpha={2/3,1}`, `max_length=768`, BF16 |

All phases use `seed=42`, AdamW with `weight_decay=0`, gradient clipping at
`1.0`, one uniformly sampled active negative per query/GT pair, and final
checkpoints.  Candidate mining uses Top-100.  The schedule is a compute
adaptation, so results must be described as a faithful method port under this
fixed single-GPU budget, not as a reproduction of RIPOR's published MS MARCO
numbers.

M0 caches the immutable token IDs once and accumulates two size-8 microbatches
into each effective size-16 update.  The two microbatch means are weighted by
their example counts before one gradient clip and optimizer step.  Sorting is
restricted to the examples already assigned to the same effective batch.  This
keeps its examples, sampled negatives, batch mean, update count, and LR schedule
unchanged while bounding the long-document attention peak.

The source-length preflight established `max_query=747 tokens` on the `94,912`
training rows, `max_query=678 tokens` on Official core75, and
`max_skill=1,715 tokens` on the `14,551`-skill catalog (`t5-base` tokenizer).
The selected limits therefore introduce no source truncation on these frozen
files.

`submit.sh` is the formal entry point.  It invokes `run_full.sh` through the
repository GPU handoff and writes every artifact under
`experiments/ripor_faithful_cat3query5_20260813`.
