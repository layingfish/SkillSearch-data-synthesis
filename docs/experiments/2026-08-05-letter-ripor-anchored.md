# Experiment: Frozen-LETTER RIPOR with baseline anchoring
Status: planned
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

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_anchored_20260805_v2/{m2,m3_prefix1,m3_prefix2,m3_prefix3,m3_prefix4,m4}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_anchored_20260805_v2/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_anchored_20260805_v2/eval/`

## Verification

- [ ] Historical `B` path reproduces archived predictions exactly on the fixed panel
- [ ] `gate=0` bypasses RIPOR generation/union/reordering and equals `B`
- [ ] Real-model teacher-forced raw cumulative scores equal constrained-prefix scores
- [ ] Bias-free RIPOR head initialization, mapping, trainable parameters and SHA are recorded
- [ ] Dynamic negative sampling changes across presentations and excludes every sibling GT
- [ ] M3 mining covers each unexpanded source query exactly once and seals checkpoint/artifact SHA
- [ ] Single-process global-batch and 8-rank RIPOR loss/gradient invariants pass
- [ ] Resume restores model/optimizer/scheduler/RNG/sampler position
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
cannot shadow the repository tests.  The catalog→RIPOR smoke itself uses the exact `ripor` arm and fixed
LETTER manifest used by M2; `custom` is reserved for the full-paper M0/RQ path and for the CE-only control.

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
correctly demands official M0/RQ-centroid provenance.  The smoke now selects `--method-setting ripor` and
the locked LETTER manifest, matching the formal M2 command without weakening the strict `custom` checks.
The failed `_v1` artifacts are retained read-only for diagnosis; the corrected run writes to a fresh `_v2`
root so no partial artifact can be mistaken for a resumable formal stage.
