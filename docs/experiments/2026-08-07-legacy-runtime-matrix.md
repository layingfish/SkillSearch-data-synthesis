# Experiment: Legacy runtime matrix
Status: done
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

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_runtime_matrix_20260807_v2/logs/`

Metrics: one `runtime_probe.json` per arm and a root `matrix_summary.json`

Job: `tide-chenyufei03-10495-bkhqg` (`rudder`, one 8xA800 node, eight single-GPU arms,
commit `7d78f6523427192ed201429122f7276233bd53eb`)

## Verification

- [x] Shell syntax and focused unit tests (2 tests passed)
- [x] Eight unique GPU assignments (`CUDA_VISIBLE_DEVICES=0..7`)
- [x] Two replicas per runtime/checkpoint condition agree exactly
- [x] Reviewer required? no; diagnostic only, followed by a 950-step parity gate

## Result / Notes

The first submission, `tide-chenyufei03-10494-hsosv`, exited before launching any GPU arm because the
experiment-note commit was still writing to CFS and the immutable-checkout preflight observed that staged
file. Its `v1` output is retained. The resubmission uses a fresh `v2` output root and job name.

The `v2` matrix completed all eight arms. Results below use archived source commit `15f6c51`, seed
`20260626`, batch `16`, BF16, `lr=3e-4`, weight decay `0`, `50` catalog steps, torch `2.8.0+cu128`,
and one A800 per arm:

```text
Transformers  checkpoint        replicas  step 1    step 50   delta from archived step 50
4.57.6        safetensors/bin   exact     4.547071  4.188124  -0.015665
5.3.0         safetensors/bin   exact     4.547071  4.186447  -0.017342
```

All eight arms produced identical initial SID embedding, code-head, control-head, backbone, and complete
model hashes. Their CPU/CUDA RNG hashes before and after construction also matched. Checkpoint format has
no effect. Transformers `5.3.0` reproduces the archived `Loading weights: 257 tensors` output but moves
step `50` farther from the historical `4.203789`. The remaining variables are torch `2.10.0+cu128` and
the historical RTX PRO 6000 Blackwell GPU; the next matrix isolates torch on the available A800 node.
