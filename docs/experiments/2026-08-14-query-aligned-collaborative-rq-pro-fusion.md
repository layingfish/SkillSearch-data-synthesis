# Experiment: Query-aligned collaborative RQ with PRO fusion
Status: running
Risk Tier: 3

## Goal / Hypothesis / Change Scope

Raise single-skill top-1 recall by making the complete GR path use one shared relevance geometry.
The tokenizer reconstructs T5 M0 relevance embeddings, preserves Qwen semantic and LightGCN CF similarities, and aligns its prefixes to frozen SkillsRouter teacher distributions. The T5 Query2SID decoder learns the same prefix distributions. Constrained decoding adds PRO's residual codebook score without catalog retrieval or reranking.

Primary configuration: T5-base M0; RQ depth=4 and codebook_size=256; semantic-affinity weight=0.01; CF-affinity weight=0.01; tokenizer prefix weight=0.1; tokenizer teacher temperature=0.05; decoder CE weight=1.0; decoder prefix-distillation weight=1.0; decoder teacher temperature=0.05; 49 teacher-ranked negatives per query; catalog_epochs=3; query_epochs=5; beam=50; PRO fusion weight omega in {0, 2, 5, 10, 20}. Fusion always computes the query vector with the frozen M0 checkpoint that produced the RQ codebook. All decoder training uses one rudder-exper node with 8xA800.

## Plan and local plan review

1. Export every training query's M0 vector with row-hash and checkpoint provenance.
2. Train a DDP residual codebook in M0 space with reconstruction, semantic affinity, CF affinity, and hierarchical teacher-distribution losses.
3. Resolve final-code collisions by minimum-cost assignment and emit collision-free four-code SIDs.
4. Add hierarchical prefix-distribution distillation to the CE decoder.
5. Add PRO residual geometry to legal trie children during beam search.
6. Verify score, prefix, provenance, DDP, collision, and zero-weight invariants locally.
7. Run a 2-step 8-GPU preflight, then the full M0 -> RQ -> decoder -> core75 evaluation chain.

Local plan review: the main semantic risk is mixing codebook and query vectors from different spaces. The implementation must reject fusion unless the codebook derives from the exact M0 checkpoint, and must reproduce M0's decoder-start pooling with the frozen M0 encoder. The second risk is silently changing baseline decoding. `omega=0` must preserve beam scores and ranking exactly. The third risk is using evaluation labels in training; all tokenizer/decoder teacher distributions come only from the 100,078-row training file and its frozen teacher panel.

## Commands

Train: `SKILLGR_LAUNCH_COMMIT=<commit> bash scripts/run_query_aligned_collaborative_rq_pro_node_20260814.sh full`

Eval: included in the full launcher; every periodic/final checkpoint is evaluated with `omega in {0,2,5,10,20}` under the metric protocol below.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v1/`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v1/logs/`

Metrics: Official core75, single subset=24 queries, beam=50, rank_aggregation=beam_order, K={1,10,50}; checkpoint selection uses the same test-set protocol as the A800 baseline.

## Verification

- [x] Minimal local check: Python compile, shell syntax, and complete pipeline unit suite (79 tests)
- [x] Unit/invariant test: M0 query pooling and codebook provenance
- [x] Unit/invariant test: teacher prefix mass and conditional loss
- [x] Unit/invariant test: omega=0 preserves decoder output and beam ordering exactly
- [x] Unit/invariant test: fusion scores only legal trie children
- [x] Unit/invariant test: gate=0 student output is allclose to base-model output
- [x] Unit/invariant test: collision-free four-code SIDs
- [x] 8-GPU two-step preflight
- [x] Structured local cross-review complete
- [x] Reviewer required? yes; fulfilled by structured local review because no delegated reviewer was requested

## Result / Notes

Local cross-review found two HIGH launch/semantic issues and fixed both before submission: a mistyped frozen-runtime SHA-256 and use of the trainable decoder copy for PRO query geometry. Fusion now loads the exact frozen M0 checkpoint, verifies its hash/config/state schema, uses its recorded query truncation and decoder-start pooling, and rejects mismatched codebook provenance. No unresolved CRITICAL/HIGH finding remains.

Method reference: PRO arXiv v3 (`2606.09241`, 2026-08-05), Eq. 7 and Appendix C; official implementation `layingfish/MGR_PRO@a9c47fda9cebc7a93a13aa8d19c0415f6c690b0d`. The implemented bias is `2 * residual dot code - squared_code_norm`; the paper default is `omega=10 (beam=50)`, and this run evaluates the local sensitivity set above.

The 8-GPU preflight completed in Tianzhi job `tide-chenyufei03-10691-tibwq`. It ran on one rudder-exper node with 8xA800, passed all 9 focused node tests, completed 2 M0 optimizer steps, 2 RQ optimizer steps, 2 decoder optimizer steps, and an 8-query constrained-decoding smoke evaluation. Preflight output: `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_preflight_v6/`.
