# Experiment: Multi-GT loss and catalog convergence
Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

- Test whether catalog indexing has converged by epoch 3 under the original fixed-LR setup.
- Test whether sibling-masked CE improves multi-GT retrieval while preserving the existing expanded single-SID training mass.
- Keep the enhanced 100,078-source-query data, fixed 14,551-skill LETTER IDs, T5-base initialization, and evaluation splits unchanged.
- Compare expanded CE and sibling-masked CE from the same catalog checkpoint.
- Compare epoch-3 and epoch-6 catalog initialization under the same expanded-CE query training.

## Commands

Train:

```text
Catalog: 6 epochs, global batch 16 (8 ranks x 2), lr=3e-4, constant schedule, seed=20260626
Query arms: 5 epochs, global batch 16 (8 ranks x 2), lr=3e-4, constant schedule, seed=20260626
Arms: epoch-3 expanded CE; epoch-3 sibling-masked CE; epoch-6 expanded CE
Long catalog: 30 continuous epochs from T5-base, global batch 16 (8 ranks x 2), lr=3e-4, constant schedule, seed=20260626; save and teacher-force eval every 3 epochs
Catalog extension: resume the epoch-27 checkpoint to a 60-epoch total horizon with restored optimizer/scheduler/per-rank RNG/shuffle epoch, global batch 16 (8 ranks x 2), lr=3e-4, constant schedule, seed=20260626; require exact epoch-30 model-state parity before using later checkpoints
Selected long-catalog query: read the first converged epoch from `eval/catalog_long_convergence.json`, then run 5 query epochs with expanded CE, global batch 16 (8 ranks x 2), lr=3e-4, constant schedule, and seed=20260626
```

Eval:

```text
Catalog checkpoints: epoch 1/2/3/6 teacher-forced CE, per-level accuracy, exact SID accuracy
Catalog checkpoints: epoch 3/6 constrained greedy exact SID accuracy
Long catalog checkpoints: epoch 3/6/9/.../30 teacher-forced metrics; epoch 6/12/18/24/30 constrained greedy
Query checkpoints: private 1,000-query and Official core75, constrained beam=50, R@1/R@10/R@50
```

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/multigt_loss_catalog_convergence_20260805_v1/`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/multigt_loss_catalog_convergence_20260805_v1/logs/`

Metrics: pending

## Verification

- [x] Minimal local check
- [x] Unit/invariant test for sibling exclusion and single-GT equivalence
- [x] Catalog epoch means cover every real sample exactly once
- [x] Catalog-resume contract tests for stage, horizon, scheduler, warmup, batch, and world size
- [ ] Remote epoch-30 exact model-state parity after resuming from epoch 27
- [ ] Retrieval evaluation from the selected converged catalog checkpoint
- [x] Reviewer required? no; structured local semantic review completed

## Result / Notes

- `tide-chenyufei03-10331-grtsb`: submitted to `rudder-r2`; remained Pending because the queue was full.
- `tide-chenyufei03-10331-grtsb`: cancelled after the `rudder-exper` replacement entered its training loop.
- `tide-chenyufei03-10333-dtdes`: first `rudder-exper` preflight failed before Python because Git rejected the physical CFS checkout as unsafe.
- `tide-chenyufei03-10335-hortt`: corrected `rudder-exper` preflight passed the image smoke test and all remote tests, then failed during the first sibling-masked query batch. Python applied the head-width slice before the boolean token mask, so the 2-class control head truncated target length from 5 to 2.
- Fix: select `[batch, target_length]` tokens first, then slice the exclusion label width for the active head. Added a full forward/backward regression test for the 2-class control head.
- Preflight harness fix: validate exactly 1 catalog train event and 2 query train events for the short run, instead of applying the formal 5-epoch query token-count assertion.
- `tide-chenyufei03-10336-mugig`: submitted to `rudder-exper` for corrected `preflight_v3` using 1 node / 8 A800 GPUs and `chenyufei03/skillsearch-gr:cu128-torch28-v1`.
- `tide-chenyufei03-10336-mugig` result: all 64 tests passed on every rank. The real T5 run completed 1 catalog step and 2 sibling-masked query steps, and wrote the final checkpoint. The wrapper then failed because it looked for `preflight_v3/loss_history.jsonl`; the actual file is `preflight_v3/model/loss_history.jsonl`.
- The follow-up `preflight_eval` stage reuses the completed `preflight_v3` checkpoint, validates its train events, runs the catalog teacher-forcing evaluator, and writes `preflight.done` without repeating training.
- `tide-chenyufei03-10337-ahpwp`: completed on `rudder-exper`. The short-run event contract was exactly 1 catalog train event and 2 query train events. Teacher-forcing evaluation completed on 16 catalog rows with world size 8, per-rank batch size 1, bf16, and max source length 2048; CE was 4.555237 and SID exact accuracy was 0.0 after the intentionally minimal 1+2-step smoke run.
- `tide-chenyufei03-10339-chwak`: submitted to `rudder-exper` for the formal 6-epoch catalog run with 14,551 skills, global batch size 16 (8 ranks x 2), constant lr 3e-4, bf16, max source length 2048, and seed 20260626.
- `tide-chenyufei03-10339-chwak` completed all 6 catalog epochs and evaluations. Every epoch contained exactly 72,755 effective CE tokens (14,551 rows x 5 target tokens). Teacher-forced CE improved from 3.955837 at epoch 3 to 3.087213 at epoch 6; constrained-greedy exact accuracy improved from 0.001100 (16/14,551) to 0.027627 (402/14,551), under the same bf16, world-size-8, max-source-length-2048 catalog evaluation.
- Planned query jobs use the `rudder` queue. Each arm uses 1 node / 8 A800 GPUs, 5 query epochs, global batch size 16 (8 ranks x 2), constant lr 3e-4, bf16, max source length 2048, and seed 20260626.
- `tide-chenyufei03-10360-qlvvp`: completed artifacts on `rudder`; epoch-3 catalog initialization with expanded CE. Log: `logs/query_expanded_ce.log`.
- `tide-chenyufei03-10361-oqabj`: completed artifacts on `rudder`; epoch-3 catalog initialization with sibling-masked CE. Log: `logs/query_sibling_masked_ce.log`.
- `tide-chenyufei03-10362-aatho`: completed artifacts on `rudder`; epoch-6 catalog initialization with expanded CE. Log: `logs/query_expanded_ce_e6.log`.
- Long-catalog convergence is pre-registered as three consecutive 3-epoch intervals where teacher-forced CE relative change is below 0.01 and SID exact absolute change is below 0.005. The earliest checkpoint completing that window is selected. If no checkpoint qualifies by epoch 30, the run is extended instead of treating epoch 30 as converged.
- `tide-chenyufei03-10364-psbjx`: completed artifacts on `rudder-exper` for the continuous 30-epoch catalog run with 1 node / 8 A800 GPUs. All 30 epochs contained exactly 72,755 effective CE tokens, and all 14,551 catalog rows passed the `max_source_length=2048` audit. Log: `logs/catalog_long_e30.log`.
- Added the `query_expanded_ce_long_selected` launch stage. It requires a completed 30-epoch catalog run with `converged=true`, resolves the selected saved checkpoint, and preserves the same 5-epoch expanded-CE query/evaluation settings as the catalog epoch-3 and epoch-6 arms.
- All three query arms completed 5 query epochs with exactly 646,705 effective CE tokens per epoch. All retrieval cells below report macro R@1/R@10/R@50 from the final checkpoint with beam=50 and beam-order aggregation.

| Query arm | Private overall (1,000 queries) | Private multi (896 queries) | Official overall (75 queries) | Official multi (51 queries) |
|---|---:|---:|---:|---:|
| catalog e3 + expanded CE | 0.220417 / 0.743417 / 0.886750 | 0.204706 / 0.752697 / 0.897042 | 0.252889 / 0.675079 / 0.761746 | 0.195425 / 0.757470 / 0.865313 |
| catalog e3 + sibling-masked CE | 0.245167 / 0.756750 / 0.886167 | 0.235677 / 0.765346 / 0.900856 | 0.261778 / 0.625460 / 0.727143 | 0.208497 / 0.684500 / 0.814426 |
| catalog e6 + expanded CE | 0.234833 / 0.759417 / 0.888000 | 0.221912 / 0.766090 / 0.898438 | 0.208127 / 0.634317 / 0.723333 | 0.188422 / 0.756349 / 0.848039 |

- On the private 896-query multi-skill subset, sibling-masked CE versus epoch-3 expanded CE changed R@1/R@10/R@50 by +0.030971/+0.012649/+0.003813. Epoch-6 catalog initialization versus epoch-3 initialization under expanded CE changed them by +0.017206/+0.013393/+0.001395. Official core75 has only 51 multi-skill queries and does not reproduce the private R@10/R@50 gain for sibling-masked CE.
- Catalog teacher-forcing on each final query checkpoint used 14,551 skills, bf16, world size 8, per-rank batch size 2, and max source length 2048. CE / SID exact was 0.704985 / 0.401484 for epoch-3 expanded CE, 0.699209 / 0.403752 for epoch-3 sibling-masked CE, and 0.599337 / 0.452340 for epoch-6 expanded CE.
- Each query arm wrote its final checkpoint, both retrieval manifests/predictions/metrics, catalog teacher-forcing JSON, and done marker before the shared launch script reported an EOF parse error. The script was edited after these long-running shells started; their post-function file offsets no longer matched. The artifacts are complete, while the shell exit status is not a valid experiment-success signal for these three jobs.
- The 30-epoch catalog run did not converge under the registered criterion. Teacher-forced metrics below use 14,551 skills, bf16, world size 8, per-rank batch size 2, and max source length 2048. Greedy exact uses the same skills with constrained beam=1.

| Catalog epoch | Teacher-forced CE | Teacher-forced SID exact | Greedy exact |
|---:|---:|---:|---:|
| 3 | 3.955837 | 0.000137 | - |
| 6 | 3.087213 | 0.004192 | 0.027627 |
| 9 | 1.657648 | 0.118411 | - |
| 12 | 0.410351 | 0.697547 | 0.817744 |
| 15 | 0.088257 | 0.921243 | - |
| 18 | 0.066194 | 0.935812 | 0.940416 |
| 21 | 0.063209 | 0.941653 | - |
| 24 | 0.056059 | 0.950175 | 0.951756 |
| 27 | 0.053916 | 0.953680 | - |
| 30 | 0.049537 | 0.957529 | 0.959178 |

- The epoch-27 to epoch-30 interval changed teacher-forced CE by 0.081220 relative and SID exact by 0.003849 absolute. It therefore failed the CE threshold before any three-interval stable window could form; `eval/catalog_long_convergence.json` records `converged=false` and `selected_epoch=null`.
- The extension resumes from epoch 27 rather than initializing from the epoch-30 final weights. The saved checkpoint contains optimizer, scheduler, per-rank RNG, and `next_epoch=27`; rerunning epochs 28-30 must produce an exact model-state hash match with the uninterrupted epoch-30 final checkpoint. The extension then continues to epoch 60, evaluates every 3 epochs, and runs constrained greedy at epochs 36/42/48/54/60.
- `tide-chenyufei03-10372-hiloc`: submitted to `rudder-exper` for the epoch-27-to-60 catalog extension with 1 node / 8 A800 GPUs, global batch size 16 (8 ranks x 2), constant lr 3e-4, bf16, max source length 2048, and seed 20260626. The launcher uses the detached worktree at commit `586ee2addd52c8fcfa33f0554f917ccfea4d8c2d`.
- `tide-chenyufei03-10372-hiloc` startup validation passed: the epoch-27 checkpoint reported `completed_steps=24570`, `next_epoch=27`, optimizer/scheduler state, and 8 per-rank RNG states. The image smoke test passed at commit `586ee2addd52c8fcfa33f0554f917ccfea4d8c2d`; resumed catalog training reached step 24700 under the same global-batch-16 and constant-lr-3e-4 conditions.
- `tide-chenyufei03-10372-hiloc` completed catalog epochs 28-60 with exactly 72,755 effective CE tokens per epoch under global batch size 16 (8 ranks x 2), constant lr 3e-4, bf16, max source length 2048, and seed 20260626. The launcher then stopped before teacher-forced or greedy evaluation because the resumed epoch-30 model-state SHA-256 (`88e139c20fcf0bc2c20714eb686de93e948de25e046a58707e1439a5c5d0e67e`) did not exactly match the uninterrupted epoch-30 SHA-256 (`55cf8a82817851cc52c049542ea6f2a611c1ee8343ab505470cfcad9873580d8`). Existing epoch-60 checkpoints remain quarantined pending numerical-equivalence and repeat-resume diagnostics.
- The resume diagnostic reruns epoch 28-30 from the same epoch-27 checkpoint as `resume-B`, preserving 1 node / 8 A800 GPUs, global batch size 16 (8 ranks x 2), constant lr 3e-4, bf16, max source length 2048, and seed 20260626. It compares original/resume-A/resume-B model and optimizer pairwise drift, scheduler/RNG fingerprints, current input hashes, and teacher-forced CE/SID exact on 14,551 catalog rows. No acceptance threshold is chosen before observing the repeat-resume variation; historical input-byte equality remains unproven because the original checkpoint did not record those hashes.
- `tide-chenyufei03-10387-toima`: submitted to `rudder-exper` for the resume-B diagnostic using 1 node / 8 A800 GPUs and detached commit `c706cdc`. Log: `logs/catalog_resume_diagnostics.log`.
- `tide-chenyufei03-10387-toima` completed resume-B epochs 28-30 and both teacher-forced evaluations, then failed in the checkpoint comparator because optimizer `step` is a scalar tensor and the byte-view helper did not flatten it before changing dtype. The completed training/evaluation artifacts are reusable; the comparator fix adds a scalar-tensor regression test.
