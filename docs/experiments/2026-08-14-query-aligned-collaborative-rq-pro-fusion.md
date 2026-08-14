# Experiment: Query-aligned collaborative RQ with PRO fusion
Status: running
Risk Tier: 3

## Goal / Hypothesis / Change Scope

Raise single-skill top-1 recall by making the complete GR path use one shared relevance geometry.
The tokenizer reconstructs T5 M0 relevance embeddings, preserves Qwen semantic and LightGCN CF similarities, and aligns its prefixes to frozen SkillsRouter teacher distributions. The T5 Query2SID decoder learns the same prefix distributions. Constrained decoding adds PRO's residual codebook score without catalog retrieval or reranking.

Primary configuration: T5-base M0; RQ depth=4 and codebook_size=256; semantic-affinity weight=0.01; CF-affinity weight=0.01; tokenizer prefix weight=0.1; tokenizer teacher temperature=0.05; decoder CE weight=1.0; decoder prefix-distillation weight=1.0; decoder teacher temperature=0.05; at most 49 frozen-teacher-ranked negatives per query; catalog_epochs=3; query_epochs=5; beam=50; PRO fusion weight omega in {0, 2, 5, 10, 20}. Fusion always computes the query vector with the frozen M0 checkpoint that produced the RQ codebook. All decoder training uses one rudder-exper node with 8xA800.

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

Train: initial run used `... full`; RQ recovery used `SKILLGR_LAUNCH_COMMIT=<commit> bash scripts/run_query_aligned_collaborative_rq_pro_node_20260814.sh resume-rq`; decoder recovery uses the same launcher with `resume-decoder`. The recovery stages reuse artifacts only after exact SHA-256 checks.

Eval: included in the full launcher; every periodic/final checkpoint is evaluated with `omega in {0,2,5,10,20}` under the metric protocol below.

## Outputs

Checkpoint: M0 source `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v1/`; completed RQ source `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v2/rq/`; decoder recovery output `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v3/`.

Log: RQ recovery log `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v2/logs/`; decoder recovery log `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_v3/logs/`.

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
- [x] Unit/invariant test: overloaded prefix falls back before the final code
- [x] 8-GPU recovery preflight on all 14,551 M0 skill vectors
- [x] Structured local cross-review complete
- [x] Reviewer required? yes; fulfilled by structured local review because no delegated reviewer was requested
- [x] Full strict-teacher coverage audit over 129,341 expanded query rows
- [x] Local pipeline suite: 88 tests passed; 27 torch-dependent tests skipped in the CPU-only local environment
- [ ] On-node variable-candidate tensor invariant

## Result / Notes

Local cross-review found two HIGH launch/semantic issues and fixed both before submission: a mistyped frozen-runtime SHA-256 and use of the trainable decoder copy for PRO query geometry. Fusion now loads the exact frozen M0 checkpoint, verifies its hash/config/state schema, uses its recorded query truncation and decoder-start pooling, and rejects mismatched codebook provenance. No unresolved CRITICAL/HIGH finding remains.

Method reference: PRO arXiv v3 (`2606.09241`, 2026-08-05), Eq. 7 and Appendix C; official implementation `layingfish/MGR_PRO@a9c47fda9cebc7a93a13aa8d19c0415f6c690b0d`. The implemented bias is `2 * residual dot code - squared_code_norm`; the paper default is `omega=10 (beam=50)`, and this run evaluates the local sensitivity set above.

The 8-GPU preflight completed in Tianzhi job `tide-chenyufei03-10691-tibwq`. It ran on one rudder-exper node with 8xA800, passed all 9 focused node tests, completed 2 M0 optimizer steps, 2 RQ optimizer steps, 2 decoder optimizer steps, and an 8-query constrained-decoding smoke evaluation. Preflight output: `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_preflight_v6/`.

The full run is Tianzhi job `tide-chenyufei03-10692-iblld`, launched from commit `f0d14d46af644794853d46526b9a1f09e44d291f` on one rudder-exper node with 8xA800. It writes to the checkpoint and log paths recorded above.

The first full run failed after completing all 4,692 RQ optimizer steps. The learned nearest-code assignment placed 10,223 skills under prefix `(43, 184, 90)`, which exceeds the 256 available final codes. The original postprocessor only reassigned the final code and therefore raised before writing the RQ artifact. M0 completed successfully and remains reusable: 129,341 MarginMSE pairs, 14,551 skill vectors, 100,078 query vectors, embedding width 768, and relevance checkpoint SHA-256 `c08d6ce0d3f7ca26b20938e37ff72aac828f2541ecfc50a047b709696ca0d519`.

Remediation: derive collision-free hierarchical assignments from the initial M0 residual KMeans solution with an explicit capacity of `codebook_size ** remaining_depth` at every prefix, then keep those discrete assignments fixed while optimizing codebook geometry. This removes the degenerate route where prefix cross-entropy can reduce its own branching domain by collapsing assignments. The final resolver must fall back to earlier levels when a parent exceeds the next level's suffix capacity; tests cover an overloaded prefix that cannot be repaired at the final level alone. The recovery run starts from the preserved M0 artifacts and reruns RQ, decoder training, and the full checkpoint-by-fusion-weight evaluation.

Recovery preflight job `tide-chenyufei03-10693-mblii` validated the v2 RQ artifact on all 14,551 skills: the initial assignment had 410 colliding full-code groups; hierarchical assignment made 520 final-level reassignments across 403 parent prefixes, produced 14,551 unique four-code SIDs, and left zero overloaded prefixes at every depth. Decoder initialization then rejected a configuration hash mismatch. The only differing field was `_name_or_path`: the preserved M0 recorded the v1 control-directory path while the recovery decoder loaded the byte-identical model through its v2 control-directory path. The check now verifies the original full config against its stored hash, then compares all model-behavior fields while excluding only `_name_or_path`.

Recovery preflight job `tide-chenyufei03-10694-trbne` passed from commit `46c9740ba6b7abe46c994448cfd8967c43977706`. Conditions: one rudder-exper node with 8xA800; all 14,551 M0 skill vectors; 4 levels with 256 codes per level; 25 KMeans iterations; 2 RQ optimizer steps; full 129,341-row query-source length audit; 2 decoder optimizer steps; reused M0 checkpoint SHA-256 `c08d6ce0d3f7ca26b20938e37ff72aac828f2541ecfc50a047b709696ca0d519`. Output: `/models_rw/SkillSearch-data-synthesis/experiments/query_aligned_collaborative_rq_pro_20260814_recovery_preflight_v2/`.

Formal RQ recovery job `tide-chenyufei03-10695-olgan` completed all 4,692 optimizer steps on one rudder-exper node with 8xA800. The capacity-safe assignment produced 14,551 unique four-code SIDs and zero overloaded prefix groups at every depth. Artifact hashes: `rqvae.pt=14b23af2101a1c4aa38416d6d984a1f8f3ab683ffceb2cab3c9b4cdac3ad875c`, `codebook_embeddings.pt=d6eb64f9d888f40db2334f53906fd5b19dc035cc82b4af4d935fe2cd895fa0a3`, `semantic_ids.jsonl=8c0e95b30940669b3e74271e6a6d667036568a0b998a5069be49cab4ee96a451`, and `rqvae_manifest.json=eda8b99dbeeb9e35e94c4aa49d9bedef240c35579735ca81b3bf6ae54d0e0828`.

The same job completed all 1,365 catalog optimizer steps (`catalog_epochs=3`, global batch=32) and then failed at the first query batch. The frozen teacher panel has fewer than 49 valid negatives after excluding every GT sibling for some multi-GT queries. Full expanded-row audit (`query_train_rows=129,341`, requested maximum=49) found 46 negatives for 8 rows, 47 for 633 rows, 48 for 7,505 rows, and 49 for 121,195 rows. Recovery keeps `max_negatives=49`, samples only candidates with finite frozen teacher scores, accepts the audited per-row range 46--49, and records the distribution in checkpoint provenance. No weak or fabricated teacher score is introduced.
