# Experiment: Legacy base exact-runtime catalog sweep
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Reproduce the archived `SkillGR base+single-SID/LETTER` training runtime before interpreting catalog
exposure. Use the frozen 94,912-row task5 base dataset without the 5,166-row mode3 augmentation. Run
catalog epochs `0/3/6/12` at seeds `20260626/20260627`, followed by five query epochs. The only intended
change from the active `v3` sweep is the Python package runtime: torch `2.10.0+cu128`, Transformers
`5.3.0`, Tokenizers `0.22.2`, CUDA `12.8`, and cuDNN `9.10.2` on the available A800 hardware.

## Commands

Train:

```text
bash scripts/run_legacy_catalog_base_exact_runtime_tianzhi_20260808.sh
```

Eval: each arm runs Official core75 with `75` queries, beam `50`, beam-order aggregation, CE log-prob,
and subset rescoring for Overall, Single, and Multi.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_exact_runtime_20260808_v1/legacy-c{0,3,6,12}-q5-seed{20260626,20260627}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_exact_runtime_20260808_v1/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_exact_runtime_20260808_v1/sweep_summary.json`

Job: `tide-chenyufei03-10553-hpwia` (`rudder-exper`, one 8xA800 node, launch commit `5a1968f`)

## Verification

- [x] Runtime and eight-GPU preflight
- [x] Immutable input, SID, model, and archived-prediction hashes
- [x] Eight training manifests and exact eval cadence
- [x] Eight unique 75-row Official core75 rankings
- [x] Archived `catalog=3`, seed `20260626` comparison
- [x] Reviewer required? no; parity-sensitive diagnostic with existing invariant tests

## Result / Notes

Completed on `2026-08-08`. All eight arms finished and wrote `sweep.done` plus
`result_summary.json`.

Under the fixed protocol (94,912 base rows, no 5,166 augment rows, catalog
epochs `0/3/6/12`, query `5` epochs, batch `16`, dev-best checkpoint, Official
core75, beam `50`, beam-order aggregation, CE log-prob, two seeds), the
two-seed mean Overall recall@1/10/50 was respectively `25.92/64.98/73.28`,
`14.71/55.16/68.06`, `20.26/58.74/70.67`, and `19.27/56.59/68.50`.

The archived final checkpoint parity check matched all `75/75` Official
core75 top-50 rankings. The `catalog=3`, seed `20260626` arm still did not
reproduce the Notion archived baseline; its exact top-50 row match count was
`0/75`. Treat the sweep as a same-runtime relative comparison, pending a
separate data/config reconciliation with the archived run.

The original-run evidence package (private archive commit `cf5876b`) has now
closed that configuration reconciliation. The original untracked trainer has
SHA256 `7b883c...`; comparison with the `15f6c511` trainer used here finds only
the later `eval_batch_size` patch, and both runs use eval batch `16`. The
original RTX probe and the A800 runtime matrix share the post-initialization
CPU RNG hash `c7674001...` and pre-forward CUDA RNG hash `87e7d790...`, while
their first catalog losses differ (`4.528710842` on RTX versus `4.546923637`
on A800 under the 50-step probe contract). This places the first divergence in
GPU numerical execution; it is not an input, checkpoint-selection, or
evaluation-protocol mismatch.
