# Experiment: Legacy runtime matrix
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Isolate the remaining archived baseline drift after source, inputs, shuffle order, batch size, optimizer,
and incomplete-tail behavior were matched. Compare Transformers `4.57.6` and `5.3.0`, each loading
either the pinned T5 safetensors or PyTorch bin checkpoint. Run two deterministic replicas per condition
on one 8xA800 node. Every arm uses archived source commit `15f6c51`, seed `20260626`, batch `16`, BF16,
`lr=3e-4`, weight decay `0`, and exactly `50` catalog optimizer steps with no query training.

Transformers `5.3.0` is paired with the latest compatible package set available by the recorded
`2026-07-20` historical environment snapshot: huggingface-hub `1.24.0`, Tokenizers `0.22.2`, and
Safetensors `0.8.0`. Tokenizers `0.22.2` is the newest published version satisfying Transformers
`5.3.0`'s `>=0.22.0,<=0.23.0` constraint because `0.23.0` was not published. Torch remains
`2.8.0+cu128` in this matrix so that the loader change is isolated.

## Commands

Train:

```text
bash scripts/run_legacy_runtime_matrix_20260807.sh
```

Eval: none. Rank arms by reproducibility across replicas, initialization hashes, and distance from the
archived step-50 loss `4.203789` under the archived training configuration above.

## Outputs

Checkpoint: none; archived trainer checkpoints are written to arm-local `/tmp` directories and removed
after loss/manifest validation.

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_runtime_matrix_20260807_v1/logs/`

Metrics: one `runtime_probe.json` per arm and a root `matrix_summary.json`

## Verification

- [x] Shell syntax and focused unit tests (2 tests passed)
- [x] Eight unique GPU assignments (`CUDA_VISIBLE_DEVICES=0..7`)
- [ ] Two replicas per runtime/checkpoint condition agree
- [x] Reviewer required? no; diagnostic only, followed by a 950-step parity gate

## Result / Notes

Pending.
