# RIPOR faithfulness audit

Audited against the paper source for arXiv:2311.09134 and the official
repository at commit `03fd0cbfafdeeb9cdd866e0c44727fec39a96a52`.

## Preserved definitions

| Definition | Official source | Local implementation and check |
|---|---|---|
| M0 representation is the first decoder output conditioned on one learned start embedding and the text encoder output | Paper, “Generative retrieval model as dense encoder”; official `T5DocIDGenEncoder.encode` | `RIPORT5DenseEncoder.encode`; real T5-base forward/backward smoke |
| M0 score is an unnormalized query/document dot product | Official `T5SeqPretrainEncoder.get_rank_loss` | `command_train_m0`; signed MarginMSE unit test |
| M0 uses BM25 Top-100, then M0 dense Top-100 negatives | Paper, “DocID Initialization” | `build-bm25-candidates`, `mine-dense-candidates`; candidate-source manifests |
| One negative is sampled uniformly from candidate columns after the positive | Official `MarginMSEforPretrainDataset.__getitem__` | seeded uniform sampling in `command_train_m0`; every GT is excluded from negatives |
| RQ uses Faiss `IndexResidualQuantizer` with inner-product metric | Official `AddictvieQuantizeIndexer.index` | `command_build_rq`; Faiss API smoke at `depth=8`, `V=256`, `14,551` vectors |
| Raw RQ centroids initialize both decoder-input and decoder-output embeddings | Official `T5SeqAQEncoder.assign_output_embeds(..., shared_output_input_embeds=False)` | `initialize_ripor_structured_model`; exact tensor-equality test and `init_scale=1.0` |
| M1 predicts only the fixed RQ codes with token CE | Official `T5SeqAQEncoderForSeq2Seq` | `ripor_code_only_batch`, existing structured decoder CE; no STOP/control target test |
| Sequence relevance is the sum of raw conditional logits | Paper scoring definition; official `T5SeqAQEncoderForMarginMSE.forward` | `sequence_conditional_logit_prefix_scores`; one-pass direct dot-product equality test |
| M2 uses full-sequence MarginMSE without generation CE | Paper, “Initial Fine-tuning” | `train-rank --stage-name M2`; stage metadata asserts `generation_ce=false` |
| M3 progresses from the shortest representative prefix and retains every earlier objective | Paper Eq. for multi-objective progressive learning; official `T5SeqAQEncoderForLngKnpMarginMSE` | `M3_prefix4` then `M3_full`; progressive-loss sum equality test |
| M4 mines from M3 and trains the full setting with the same multi-objective loss | Paper, “Self-Negative Fine-tuning”; official self-negative merge preprocessing | `M4` after M3 beam mining; objectives remain `{4,8}` |
| Retrieval uses constrained beam search on raw conditional-logit sums and expands one SID to every assigned document | Paper scoring/retrieval definition; official `smtid_to_docids` | `topk_single_sid_scores_from_encoder(..., score_mode="conditional_logit", include_stop=False)` followed by catalog-order skill expansion; Official eval manifest |

## Explicit adaptations

1. The teacher is `pipizhao/SkillRouter-Embedding-0.6B` instead of the MS MARCO
   cross-encoder.  The training target is the raw dot-product margin between
   its L2-normalized query and skill embeddings.  Signed margins are retained.
2. Existing cat3/query5 task queries replace doc2query outputs in M1.  This
   introduces no additional query data and keeps the frozen experiment data.
3. Each multi-GT row expands to one pair per GT.  The complete GT set is
   removed from the negative pool.  RIPOR's pairwise loss itself is unchanged.
4. The SID uses `depth=8`, `V=256`.  This is the smallest official-style depth
   with a real progressive transition from prefix 4 to full depth.  Full-SID
   collisions retain RIPOR's official `smtid_to_docids` one-to-many semantics.
5. Source limits are increased to `2048 tokens` for M0 query/skill encoding and
   `768 tokens` for M1/rank/eval, because SkillGR inputs are much longer than
   MS MARCO queries.  The frozen files have zero truncation under these limits.
6. Epoch counts are reduced for a single GPU and frozen before Official test.
   The method graph, candidate stages, loss definitions, and checkpoint order
   remain intact.  Rank stages realize effective `batch=64` as two
   `microbatch=32` forwards with size-weighted loss accumulation, followed by
   one gradient clip, optimizer step, and scheduler step.

These adaptations affect the experimental condition and must accompany every
reported number.  They do not add an inference-time dense model or reranker.

## Paper/released-code discrepancy

The paper defines `alpha_i=(1-beta/i)/(1-beta/L)` with `beta=2`.  Released
preprocessing uses cumulative decay-2 factors `0.5, 0.75, 0.875, 1.0` for
lengths `4,8,16,32`, then normalizes earlier factors by the active stage's
factor.  For the selected final depth 8, both yield the same relative targets:
`alpha_4=0.5/0.75=2/3` and `alpha_8=1`.  No undocumented choice is needed.

The paper describes M4 as self-negative training.  The released
`merge_self_neg_sym_neg_for_jsonl_example.py` combines self negatives with the
existing candidate source.  The launcher follows this released behavior by
using the ordered union of the M3 pool and M3 beam Top-100.

## Prohibited shortcuts enforced by code

- A checkpoint lacking `checkpoint_format=ripor_structured_decoder_v1`, the
  audited official commit, code-only SID metadata, conditional-logit scoring,
  zero output bias, and the exact RQ semantic-ID hash is rejected.
- LETTER, RQ-VAE, relevance blends, dedup suffixes, centroid normalization,
  RMS centroid rescaling, STOP loss, generation CE during ranking, clipped
  teacher margins, teacher-best-GT filtering, and Official-selected
  checkpoints do not appear in the formal path.
- Full-SID collisions are retained and expanded to every assigned skill in
  catalog order.  The implementation never repairs them with a custom suffix.
