# Experiment: Frozen-LETTER RIPOR with baseline anchoring
Status: done
Risk Tier: 3

## Goal / Hypothesis / Change Scope

Preserve the established mode3 single-SID CE/LETTER retriever as an immutable log-probability branch,
while reproducing RIPOR's decoder-side raw conditional-logit prefix ranking on the same fixed four-level
LETTER identifiers.  The original baseline checkpoint, tokenizer, trie, beam configuration and predictions
must remain independently executable.  RIPOR training never mutates the baseline artifact.

The method layer and integration layer are reported separately:

1. `B`: frozen historical CE/LETTER checkpoint with log-softmax beam accumulation.
2. `R_raw`: independent RIPOR checkpoint with a bias-free position-specific ranking head and raw prefix
   accumulation.
3. `B_union_R_anchor`: deterministic candidate union and preregistered anchored scoring over outputs from
   the two independent branches.

RIPOR decoder-side stages are `M2 full-SID → M3 prefix 1/2/3/4 with multi-objective retention → frozen
M3 raw-score self-negative mining → M4 full-SID`.  Every mined negative must carry a numeric score from the
frozen SkillRouter teacher panel; missing scores are skipped and audited, never synthesized.  All source
GT siblings are excluded before pairs are formed.

## Commands

Train:

```bash
bash scripts/run_letter_ripor_anchored_20260805.sh all
```

The script is intended for one `rudder-r1` node with 8×A800.  It runs fail-closed baseline/scoring/DDP
preflight before M2.  Locked first-run budgets are M2=5,000 steps, each M3 prefix stage=2,500 steps, and
M4=5,000 steps, with per-device batch 8, LR 1e-4, weight decay 0, BF16, one dynamically sampled scored
negative per expanded positive, and a fresh optimizer/scheduler per stage.  These are a method-complete
first run on the project corpus, not a claim that the paper's 120-epoch MS MARCO compute scale is matched.

Eval:

```bash
bash scripts/run_letter_ripor_anchored_20260805.sh eval
```

Evaluation writes `B`, `R_raw`, the diagnostic `R_logprob`, and `B_union_R_anchor` on canonical dev,
Official core75 and private test.  The anchor weight is fixed before evaluation; Official/private are not
used to tune it.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_anchored_20260805_v7/{m2,m3_prefix1,m3_prefix2,m3_prefix3,m3_prefix4,m4}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_anchored_20260805_v7/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_anchored_20260805_v7/eval/`

## Verification

- [x] Historical `B` path reproduces archived prediction rankings exactly on the fixed panel
- [x] `gate=0` bypasses RIPOR generation/union/reordering and equals `B`
- [x] Real-model teacher-forced raw cumulative scores equal constrained-prefix scores
- [x] Bias-free RIPOR head initialization, mapping, trainable parameters and SHA are recorded
- [x] Dynamic negative sampling changes across presentations and excludes every sibling GT
- [x] M3 mining covers each unexpanded source query exactly once and seals checkpoint/artifact SHA
- [x] Single-process global-batch and 8-rank RIPOR loss/gradient invariants pass
- [x] Resume restores model/optimizer/scheduler/RNG/sampler position
- [x] Reviewer required? yes — independent plan review and final focused cross-review completed; no unresolved CRITICAL/HIGH
- [x] Minimal local check — shell syntax, Python compilation, and 62 unit tests pass (10 torch-only skips)

## Result / Notes

No GPU job has been submitted yet.  Independent review found no unconditional CRITICAL issue.  It requires
the baseline branch to remain a separate frozen execution path; `B`, `R_raw`, and anchored union to be
reported separately; self-negatives to come only from the frozen M3 raw decoder and be teacher-scored;
real-model train/decode score parity; deterministic no-padding DDP mining and sealed provenance; and a
fully preregistered anchor formula/tie-break.  Formal training is blocked until these checks pass.

The cross-component fixes make every RIPOR epoch no-repeat under DDP (zero-weight graph-connected dummies
on empty tail ranks), freeze RIPOR-only parameters during CE-only training, bind every stage checkpoint to
the exact direct upstream checkpoint SHA, and require M4 to re-hash and path-check the complete
M3P4→miner→sealed-panel chain.  The setting contract is accepted only from a clean committed worktree and
revalidates all immutable input hashes at every formal stage.

Final focused cross-review found no remaining CRITICAL/HIGH after changing the CE-only smoke to the legal
`custom` setting, graph-connecting trainable RIPOR heads during catalog CE, adding an 8-rank
catalog→RIPOR smoke, and switching unit invocation to discovery so an unrelated installed `tests` package
cannot shadow the repository tests.  The catalog→RIPOR smoke is an explicitly labeled weak-teacher
`custom` plumbing artifact; the later `train_ripor` smoke independently exercises the exact fixed-LETTER
`ripor` arm used by formal M2, which correctly reuses the already catalog-indexed shared baseline.

Two earlier routing attempts (`tide-chenyufei03-10290-gaxan` and the zero-work probe
`tide-chenyufei03-10291-dstlh`) were cancelled before scheduling; neither started a Pod or wrote an
experiment artifact.  The replacement run is explicitly bound to `rudder-r1`.

The first `rudder-r1` launch (`tide-chenyufei03-10297-asokx`) reached a node but stopped before contract
creation because Git rejected the CFS repository ownership.  It performed no preflight or optimizer step
and wrote no checkpoint.  The launcher now resolves the physical repository path and registers that exact
path as Git `safe.directory` before the clean-tree/commit provenance checks.

The replacement `rudder-r1` launch (`tide-chenyufei03-10298-dbfqv`) created and verified the setting
contract, passed 60+2 unit tests, the 8-rank loss/gradient invariant, and the two-step CE DDP smoke.  It then
failed closed before formal M2 in the catalog→RIPOR smoke because that smoke was mislabeled `custom`, which
correctly demands official M0/RQ-centroid provenance unless explicitly marked as an ablation.  The failed
`_v1` artifacts are retained read-only for diagnosis.

The next `rudder-r1` launch (`tide-chenyufei03-10299-kvkhe`) repeated the successful checks above and then
failed closed because the isolated `ripor` setting correctly prohibits catalog CE: the shared LETTER
baseline already owns catalog indexing.  The catalog→RIPOR graph-connectivity check is therefore restored
to `custom` with the explicit weak-teacher ablation flag, while the separate exact-M2 smoke remains
`--method-setting ripor --catalog-epochs 0`.  The two failed roots remain intact; the corrected run writes
to a fresh `_v3` root so no partial artifact can be mistaken for a resumable formal stage.

The `_v3` `rudder-r1` launch (`tide-chenyufei03-10300-jbrnj`) passed the custom catalog→RIPOR DDP smoke,
including one catalog and one ranking optimizer step, then failed before the real-model score comparison.
The historical checkpoint stores T5 `extra_special_tokens` in the older list form, which Transformers 4.57
cannot consume directly.  The preflight now uses the same guarded metadata-only compatibility fallback as
structured evaluation (`extra_special_tokens={}` only when that exact legacy list/error pair is observed);
weights, tokenizer vocabulary and LETTER IDs are unchanged.  `_v3` remains intact and the corrected run
writes to fresh `_v4`.

The `_v4` `rudder-r1` launch (`tide-chenyufei03-10301-exsxl`) passed the real-model raw-score invariant
(teacher-forced versus constrained-prefix maximum error `3.81e-6`) and exact single-GPU/8-rank evaluation
parity.  Its Official baseline reproduction has identical ranking-bearing fields on all 75 rows, including
the full top-50 order.  The archived predictions contain `ranked_skill_scores: null` on all rows, whereas
the current evaluator adds a 50-candidate score map; the old comparator incorrectly treated that enrichment
as a domain mismatch.  Best-beam scores differ by at most `4.17e-6`.  The comparator now requires exact
rankings, compares candidate scores whenever the archive actually supplies them, records missing archived
maps, and uses the real-model parity tolerance (`1e-5`) for floating scores.  `_v4` remains intact and the
corrected run writes to fresh `_v5`.

The `_v5` `rudder-r1` launch (`tide-chenyufei03-10302-epnhl`) passed historical baseline and gate-zero
parity, then failed in the final resume check before formal M2.  The metadata-only checkpoint reload used
`map_location=device`, which converted saved CPU RNG ByteTensors to CUDA tensors; `torch.set_rng_state`
requires a CPU ByteTensor.  That reload now stays on CPU (model/optimizer restoration remains on the rank's
device through the normal checkpoint loader), preserving the exact per-rank Python/CPU/CUDA RNG payloads.
`_v5` remains intact and the corrected run writes to fresh `_v6`.

The `_v6` `rudder-r1` launch (`tide-chenyufei03-10303-nkbkt`) restored every rank's RNG state and completed
the resumed update.  The uninterrupted and resumed step-2 loss histories are byte-identical; final FP32
parameters differ by at most `5.96e-8`, below one FP32 epsilon at unit scale, but the preregistered `1e-8`
absolute threshold was unnecessarily tighter than a single representable FP32 rounding step across a new
DDP process.  Resume verification now retains exact loss-history equality and records changed tensor/value
counts, with a still-strict `1e-7` absolute parameter bound.  `_v6` remains intact and the corrected run
writes to fresh `_v7`.

The formal `_v7` run (`tide-chenyufei03-10304-sugoo`, queue `rudder-r1`, commit `7e2385f`) completed on one
8×A800 node.  Preflight passed before the first formal optimizer step, including 62 tests, real-model
teacher-forced/constrained raw-score parity (`3.81e-6` maximum error), exact single-GPU/8-rank ranking
parity, exact historical top-50 reproduction on all 75 Official rows, gate-zero SHA equality, 8-rank
loss/gradient invariants, and resume parity (`5.96e-8` maximum parameter error with byte-identical loss
history).  M2, all four M3 prefix stages, and M4 completed their locked `5,000 + 4×2,500 + 5,000` steps;
every final checkpoint passed direct-upstream provenance verification.

M3 raw-score mining covered all `100,078` unexpanded source queries exactly once.  The sealed M4 input has
`58,481` eligible queries and `130,373` teacher-scored self-negatives; `41,597` queries without an available
teacher-scored negative were skipped, and `68,327` sibling-GT candidate mentions were excluded.  The
checkpoint, predictions, teacher panel, eligible rows, and output panel hashes are recorded in
`m3_self_negative_mining/self_negatives.manifest.json` and were re-verified before M4.

Overall macro Recall@K:

| Split | Arm | R@1 | R@10 | R@50 |
|---|---|---:|---:|---:|
| dev | B | 0.2189 | 0.7538 | 0.8930 |
| dev | R_raw | 0.0353 | 0.1387 | 0.2097 |
| dev | R_logprob | 0.0334 | 0.1360 | 0.2087 |
| dev | B_union_R_anchor | 0.1480 | 0.6403 | 0.8580 |
| Official core75 | B | 0.2932 | 0.6904 | 0.7856 |
| Official core75 | R_raw | 0.1932 | 0.4400 | 0.5178 |
| Official core75 | R_logprob | 0.1932 | 0.4489 | 0.5044 |
| Official core75 | B_union_R_anchor | 0.2566 | 0.6356 | 0.7522 |
| private test | B | 0.2233 | 0.7584 | 0.8909 |
| private test | R_raw | 0.0436 | 0.1512 | 0.2374 |
| private test | R_logprob | 0.0465 | 0.1519 | 0.2337 |
| private test | B_union_R_anchor | 0.1553 | 0.6206 | 0.8517 |

Single-skill versus multi-skill macro Recall@K (the multi value aggregates target counts 2–4 by row):

| Split | Arm | single R@1/R@10/R@50 | multi R@1/R@10/R@50 |
|---|---|---|---|
| dev | B | 0.2500 / 0.7019 / 0.8077 | 0.2153 / 0.7598 / 0.9029 |
| dev | R_raw | 0.1154 / 0.2692 / 0.3365 | 0.0259 / 0.1236 / 0.1949 |
| dev | B_union_R_anchor | 0.2500 / 0.6731 / 0.7981 | 0.1362 / 0.6365 / 0.8650 |
| Official core75 | B | 0.5417 / 0.5833 / 0.6250 | 0.1763 / 0.7408 / 0.8611 |
| Official core75 | R_raw | 0.2917 / 0.3333 / 0.3750 | 0.1469 / 0.4902 / 0.5850 |
| Official core75 | B_union_R_anchor | 0.4167 / 0.5417 / 0.5833 | 0.1812 / 0.6797 / 0.8317 |
| private test | B | 0.3365 / 0.7308 / 0.8077 | 0.2102 / 0.7616 / 0.9006 |
| private test | R_raw | 0.1923 / 0.3558 / 0.3750 | 0.0263 / 0.1274 / 0.2214 |
| private test | B_union_R_anchor | 0.2885 / 0.6538 / 0.7885 | 0.1399 / 0.6167 / 0.8591 |

The preregistered `lambda=1` anchor does not preserve the baseline advantage.  RIPOR's top-50 candidate
union adds only `+0.00075` dev, `+0.00444` Official, and `+0.00175` private macro recall over B, while its
scores substantially disturb B's ordering.  Therefore this run rejects the hypothesis that the current
RIPOR schedule and M4 self-negatives strengthen the fixed-LETTER retriever: the main failure is weak RIPOR
retrieval/ranking signal, not merely score calibration.  Gate-zero outputs remain byte-identical to B, so
the historical branch itself is preserved and the regression is isolated to the RIPOR contribution and
nonzero fusion.
