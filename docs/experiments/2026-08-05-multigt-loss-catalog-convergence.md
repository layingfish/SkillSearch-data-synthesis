# Experiment: Multi-GT loss and catalog convergence
Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

- Test whether catalog indexing has converged by epoch 3 under the original fixed-LR setup.
- Test whether sibling-masked CE improves multi-GT retrieval while preserving the existing expanded single-SID training mass.
- Keep the enhanced 100,078-source-query data, fixed 14,551-skill LETTER IDs, T5-base initialization, and evaluation splits unchanged.
- Compare expanded CE and sibling-masked CE from the same catalog checkpoint.

## Commands

Train:

```text
Catalog: 6 epochs, global batch 16 (8 ranks x 2), lr=3e-4, constant schedule, seed=20260626
Query arms: 5 epochs, global batch 16 (8 ranks x 2), lr=3e-4, constant schedule, seed=20260626
```

Eval:

```text
Catalog checkpoints: epoch 1/2/3/6 teacher-forced CE, per-level accuracy, exact SID accuracy
Catalog checkpoints: epoch 3/6 constrained greedy exact SID accuracy
Query checkpoints: private 1,000-query and Official core75, constrained beam=50, R@1/R@10/R@50
```

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/multigt_loss_catalog_convergence_20260805_v1/`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/multigt_loss_catalog_convergence_20260805_v1/logs/`

Metrics: pending

## Verification

- [x] Minimal local check
- [x] Unit/invariant test for sibling exclusion and single-GT equivalence
- [ ] Catalog epoch means cover every real sample exactly once
- [x] Reviewer required? no; structured local semantic review completed

## Result / Notes

- `tide-chenyufei03-10331-grtsb`: submitted to `rudder-r2`; remained Pending because the queue was full.
- `tide-chenyufei03-10333-dtdes`: first `rudder-exper` preflight failed before Python because Git rejected the physical CFS checkout as unsafe.
- `tide-chenyufei03-10335-hortt`: corrected `rudder-exper` preflight passed the image smoke test and all remote tests, then failed during the first sibling-masked query batch. Python applied the head-width slice before the boolean token mask, so the 2-class control head truncated target length from 5 to 2.
- Fix: select `[batch, target_length]` tokens first, then slice the exclusion label width for the active head. Added a full forward/backward regression test for the 2-class control head.
- Preflight harness fix: validate exactly 1 catalog train event and 2 query train events for the short run, instead of applying the formal 5-epoch query token-count assertion.
- `tide-chenyufei03-10336-mugig`: submitted to `rudder-exper` for corrected `preflight_v3` using 1 node / 8 A800 GPUs and `chenyufei03/skillsearch-gr:cu128-torch28-v1`.
- `tide-chenyufei03-10336-mugig` result: all 64 tests passed on every rank. The real T5 run completed 1 catalog step and 2 sibling-masked query steps, and wrote the final checkpoint. The wrapper then failed because it looked for `preflight_v3/loss_history.jsonl`; the actual file is `preflight_v3/model/loss_history.jsonl`.
- The follow-up `preflight_eval` stage reuses the completed `preflight_v3` checkpoint, validates its train events, runs the catalog teacher-forcing evaluator, and writes `preflight.done` without repeating training.
- `tide-chenyufei03-10337-ahpwp`: completed on `rudder-exper`. The short-run event contract was exactly 1 catalog train event and 2 query train events. Teacher-forcing evaluation completed on 16 catalog rows with world size 8, per-rank batch size 1, bf16, and max source length 2048; CE was 4.555237 and SID exact accuracy was 0.0 after the intentionally minimal 1+2-step smoke run.
