# Experiment: Fixed-LETTER RIPOR/PAG component settings
Status: running
Risk Tier: 3

## Goal / Hypothesis / Change Scope

Measure the contribution of RIPOR decoder ranking and PAG lexical Set IDs while keeping the existing
Qwen + LightGCN + LETTER Semantic IDs immutable. The three requested final settings are:

1. `ripor`: fixed LETTER SID + prefix-oriented MarginMSE.
2. `pag`: fixed LETTER SID CE baseline + PAG set loss/planning-ahead.
3. `ripor_pag`: fixed LETTER SID + prefix-oriented MarginMSE + PAG set loss/planning-ahead.

All settings use the same single-SID target expansion, canonical train/dev/private-test rows, Official
core75 main-eval rows, canonical catalog, LETTER SID
hash, CE+LETTER base decoder, numeric teacher panel, seed, per-device batch, 8-GPU world size, source
policy, and final-stage step budget. PAG Msp/Mset and RIPOR Mseq are each trained once and shared.
PAG-only merges Mset with the untouched CE+LETTER base; RIPOR+PAG merges the same Mset with the shared
fixed-LETTER RIPOR Mseq. Sequential SID/code/control heads always come bitwise from the corresponding
sequential model.

## Commands

Train:

```bash
bash scripts/run_letter_ripor_pag_settings.sh <stage>
```

The launcher sources `configs/letter_ripor_pag_20260804.env`. The execution DAG is `teacher → preflight
→ {pag_shared,ripor_mseq} → {final_ripor,final_pag,final_ripor_pag} → four eval stages`; independent
stages are submitted as one-node 8-GPU jobs. The teacher stage writes `setting_contract.json` with
input SHA-256 hashes before method training.

Locked protocol: Msp 4700 steps with the official PAG sparse-encoder `max_length=128`, PAG decoder
query-prefix `max_length=64`, Mset 5000,
shared RIPOR Mseq 5000, each final stage 5000; per-device
batch 8 (global 64), AdamW with zero weight decay, LR 5e-4, 4.5% linear warmup followed by linear
decay, BF16, seed 20260626, PAG m=64/top-N=1000/weight=1, and beam=50.

Eval:

The launcher evaluates all three final checkpoints with 8-GPU single-SID constrained decoding. PAG
settings use the same lexical set manifest, planning weight, top-N candidate domain, beam count, and K.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_pag_single_sid_20260804_v1/{ripor_only,pag_only,ripor_pag}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_pag_single_sid_20260804_v1/logs/`

Metrics: each setting writes `official_core75` (main) and `private_test` (secondary) subdirectories.
PAG settings additionally write `{planning,weight0_diagnostic}/ranked_metrics.json` inside each split.

## Verification

- [x] Minimal local syntax/config check
- [x] LETTER manifest/SID fingerprint invariant
- [x] PAG merge preserves Mseq SID embedding/code/control heads bitwise
- [x] Focused loss-setting tests (56 passed in the training image)
- [x] DDP global sum/count invariant on 8 ranks (`max_gradient_error=5.96e-08`)
- [x] Reviewer required? yes — independent plan review completed; CRITICAL/HIGH items incorporated or explicitly resolved

## Result / Notes

Formal train/dev/catalog, fixed LETTER, base single-SID decoder, SkillRouter teacher, and pinned T5-base
assets are resolved on CFS. Independent cross-review found no remaining CRITICAL/HIGH findings. Teacher
job `tide-chenyufei03-10230-hlrjf` failed before training because `/dataset_rw` was not initialized by
the alias helper. The launcher now passes `--with-dataset-rw` and starts CFS logging before validation.

Retry `tide-chenyufei03-10231-rshfg` completed distributed encoding but failed before scoring because
query embeddings were on CPU while the gathered catalog matrix was on each rank's GPU. The query
matrix is now moved to the local rank device immediately before matmul; this is a device-only fix.

Teacher retry `tide-chenyufei03-10232-nrekv` succeeded and produced 100078 ranked rows plus the
SHA-bound setting contract. The teacher metrics are Hit@1=0.83119167 and Hit@50=0.99351506.

Preflight `tide-chenyufei03-10233-hsqmj` passed all 53 image tests and the 8-rank DDP invariant, then
failed in real-model PAG Msp with CUDA OOM. Root cause: the launcher incorrectly reused the GR decoder
`max_source_length=2048` for T5-SPLADE. The official PAG `t5_splade_train.sh` fixes sparse-encoder
`max_length=128`; the locked config, setting contract, and launcher now use 128 for Msp while retaining
2048 for Mset/RIPOR/joint stages. A fresh preflight is required before formal training.

Preflight retry `tide-chenyufei03-10234-qtunc` passed Msp and Mset, then failed while loading the
existing CE+LETTER base into RIPOR because commit `605da31` had changed the four position-specific
code heads from the established biased LETTER parameterization to bias-free heads. The experiment
requires fixed LETTER initialization, so the model now retains and strictly loads the baseline head
biases; RIPOR modifies the prefix-ranking objective only. A focused model invariant asserts all four
LETTER code-head biases exist.

Preflight retry `tide-chenyufei03-10235-xfeut` passed strict checkpoint loading, then exposed a RIPOR
progressive-prefix tensor-axis bug: loss values were truncated to the active prefix count while the
positive/negative divergence mask retained the full SID depth. The mask is now sliced to the identical
prefix axis, so a pair contributes exactly for active prefixes at or after its first differing code; a
numeric unit test fixes the expected pair count and loss for a depth-4/active-depth-2 example.

Preflight `tide-chenyufei03-10236-kjmnb` passed Msp, Mset, RIPOR Mseq, RIPOR-only continuation, and
PAG-only joint training. RIPOR+PAG then OOMed because the implementation repeated the full (up to 2048)
encoder source as a decoder query prefix for every ranking candidate. The official PAG unified-training
script uses `max_length=64`. The locked implementation now keeps the existing 2048-token encoder input
unchanged but limits only PAG's decoder query prefix to the first 64 tokens in Mset, joint training, and
inference. Inference also repeats this bounded prefix to the active beam batch explicitly. The limit is
stored in the setting contract and checkpoint provenance and is verified by a focused tensor test.

Preflight `tide-chenyufei03-10237-dajit` succeeded and wrote `preflight/PASSED`; all training and both
PAG decoding diagnostics executed. Its eval audit nevertheless showed one-token empty sources and zero
gold because the locked dev split stores query/labels under `generated.*`, while single-SID expansion and
decoder source extraction only read prepared top-level fields. Both paths now accept the canonical raw
schema as well as prepared rows, with a regression test covering nested query and two nested labels.
Formal training/evaluation remains gated on a refreshed preflight with non-zero dev rows and gold.

Refreshed preflight `tide-chenyufei03-10238-lsltr` succeeded. All 56 focused tests, the 8-rank DDP
objective invariant, every two-step real-model training stage, planning-ahead decoding, and the weight=0
diagnostic completed. The raw dev audit now reports eight non-empty queries (110--341 tokens) and eight
gold skill mentions with zero missing mentions, so the earlier schema defect is closed. The formal
shared stages were then submitted concurrently: PAG Msp/Mset as `tide-chenyufei03-10239-gwkec` and
RIPOR Mseq as `tide-chenyufei03-10240-hrdvg`, each on one complete 8xA800 node in `rudder-r2`.

The first formal RIPOR job exposed a protocol asset error before any optimizer step: the config pointed
at the pre-alias raw dev split, which contains 80 gold mentions outside the frozen 14,551-skill LETTER
pool (78 affected rows). The established canonical assets already solve this without runtime filtering:
canonical dev and private test each contain 1,000 rows with zero missing labels, while Official core75
contains 75 rows (24 single/51 multi, 200 gold mentions) with zero missing labels. The locked protocol
now uses canonical dev for the non-selecting training-stage audit, Official core75 as the main evaluation,
and canonical private test as secondary evaluation. The setting contract must be regenerated before the
shared stages are restarted; artifacts produced under the old contract are not eligible for final arms.
The corrected SHA-bound contract was generated from the existing unchanged teacher panel, and RIPOR Mseq
was resubmitted as `tide-chenyufei03-10241-trolo`. The old PAG process was allowed to terminate through
its fail-closed dev/contract checks and could not produce an eligible Mset checkpoint.

Old-contract PAG job `tide-chenyufei03-10239-gwkec` finished all 4,700 Msp steps and then failed before
Mset optimization while expanding the pre-alias dev labels. Its Msp/Mset directories were moved intact
to `${SKILLGR_OUTPUT_ROOT}/invalid_raw_dev_contract/` and are ineligible for downstream use. The complete
PAG shared path was resubmitted from Msp as `tide-chenyufei03-10242-tmick` under the canonical contract.

Canonical RIPOR Mseq job `tide-chenyufei03-10241-trolo` succeeded after 5,000 steps with final logged
loss 0.042466 and wrote the shared checkpoint at `ripor_mseq/final/structured_decoder.pt`. The matched
second 5,000-step RIPOR-only final continuation was submitted as `tide-chenyufei03-10245-pcgrm`.

A semantic self-review then found that the structured trainer defaults to restarting progressive prefix
expansion on every invocation. That is correct for the shared Mseq stage, but not for either final stage:
both already initialize from a full-depth Mseq and must train the full multi-prefix objective throughout.
The launcher now makes the schedule explicit (`progressive` for shared Mseq, `full` for RIPOR-only and
RIPOR+PAG final), and checkpoints record it. The currently running `tide-chenyufei03-10245-pcgrm` is
therefore an audit-only invalid run; its output will be quarantined and the final stage rerun.

The corrected full-prefix final was launched concurrently as `tide-chenyufei03-10247-ctwhg` into the
isolated `ripor_only_fullprefix` directory. At the same local training position, its first logged loss was
about 0.048 versus about 0.018 for the restarted-curriculum run, confirming that the two invocations were
supervising different active prefix sets rather than merely carrying different provenance labels.

Corrected RIPOR-only job `tide-chenyufei03-10247-ctwhg` succeeded with final loss 0.008334; its isolated
directory was promoted to the canonical `ripor_only/` path, while the audit-only checkpoint remains
recoverable under `invalid_restarted_prefix_curriculum/`. PAG Mset also completed 5,000 steps and saved
its checkpoint (final loss 0.046081); its outer shell later hit an EOF because the launcher file had been
edited while that long-lived Bash process was running. New downstream jobs strictly loaded the resulting
Mset provenance before optimization: PAG-only `tide-chenyufei03-10248-cpaxi` and RIPOR+PAG
`tide-chenyufei03-10249-uhccs`.

Both final jobs completed, but PAG-only planning evaluation exposed a protocol-critical collapse:
all 14,551 skills had the same unordered top-64 Set ID, so Official top-1000 candidate recall was zero.
The 186 unique Official gold skills were all present in training and the manifest, excluding label
coverage as the cause. The affected Msp/Mset/PAG-only/RIPOR+PAG artifacts and evaluations were moved
intact to `invalid_collapsed_pag_set_ids/`; they are not eligible for reporting.

Review against official PAG commit `49802454...` also found that Msp had omitted the official
`RegWeightScheduler`, which quadratically ramps query/doc FLOPS weights during the first third of
training. Commit `4318e3e` implements the exact schedule, records it in Msp provenance, makes Mset reject
nonconforming Msp checkpoints, and fails all ranks together if every skill receives the same Set ID.
Focused jobs `tide-chenyufei03-10255-hggkq` (100-step compressed ramp) and
`tide-chenyufei03-10256-chxsw` (100-step FLOPS=0) both still collapsed, so FLOPS was not treated as the
sole cause. The locked 4,700-step Msp→Mset retry is `tide-chenyufei03-10257-vkfsd`.

An additional same-protocol 1,000-step Msp diagnostic, `tide-chenyufei03-10258-djvkx`, is launched only
to observe when lexical Set-ID diversity appears; it does not replace the locked 4,700-step stage and
cannot enter downstream training.

Because the 1,000-step loss trajectory remains nearly unchanged across regularization schedules, a
read-only 8-rank representation audit compares the pinned untrained T5-base and the 1,000-step Msp
checkpoint on the same first 512 skills. It reports raw-logit positive coordinates, lexical non-zero
coordinates, activation magnitudes, and unordered top-64 diversity; no weights or artifacts are mutated.

Representation audit `tide-chenyufei03-10259-qxqxc` completed. Base T5 has only 0.021484 positive raw
coordinates per skill row on average (maximum 1), raw-logit maximum mean -1.8689, lexical maximum mean
0.00815, 9 unordered top-64 sets and 10 distinct top-1 IDs. The 1,000-step Msp has zero positive raw or
lexical coordinates for every audited row; its raw-logit maximum mean is -9.4857 and even its largest
maximum is -4.09375, leaving one top-64 set and one top-1 ID. Thus the collapsed manifest reflects an
actual all-zero PAG lexical representation, not a distributed merge or serialization defect.

The official `full_scripts/t5_splade_train.sh` uses 8 GPUs, per-device batch 32, 200,000 steps and
25,000-step checkpoints. The locked 4,700-step scheduler-corrected job remains the mandatory confirmation
before changing protocol. If its full-catalog audit also collapses, `task_plan.md` section 8.1 permits only
an official-scale Msp amendment, bound to the unchanged experiment contract through a separate PAG Msp
protocol artifact; no activation, backbone, Set-ID rule, downstream step budget, or evaluation setting may
change.
