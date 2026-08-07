# Experiment: Legacy torch runtime matrix
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Test the remaining available historical runtime variable after source, inputs, shuffle, optimizer,
incomplete-tail behavior, model bytes, checkpoint format, and SID-head initialization were matched.
Run torch `2.10.0+cu128` with Transformers `4.57.6` and `5.3.0`, using four deterministic replicas per
condition on one 8xA800 node. Compare against the completed torch `2.8.0+cu128` matrix. Checkpoint format
is fixed to the pinned T5 safetensors. Every arm uses archived source commit `15f6c51`, seed `20260626`,
batch `16`, BF16, `lr=3e-4`, weight decay `0`, and exactly `50` catalog optimizer steps with no query
training.

Torch `2.10.0` and Transformers `5.3.0` are pre-materialized with `uv` in the unique CFS archive
`/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1/site.tar.zst`. Its size is
`3,995,807,897` bytes and SHA-256 is
`c6ac264c8519eec039614af6e5357b56434ca4d2981d34bfdbdde5fe9272d2b2`. The launcher verifies that hash,
extracts once to node-local `/tmp`, then overlays
`/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py311-v1/compat-overlay.tar.zst` with SHA-256
`a6b2f56bb9b4496caa4be225b1c2630270cef1b9a23f82013eea72632bda6b20`. The overlay pins torchvision
`0.25.0+cu128`, SciPy `1.15.3`, and scikit-learn `1.7.2`. The launcher prepends the combined site to
`PYTHONPATH` and imports the T5 model class before starting arms. No torch installation occurs in the GPU
job. Transformers `4.57.6` is the only arm-local Python overlay; it uses huggingface-hub `0.36.2`,
Tokenizers `0.22.2`, and Safetensors `0.8.0`.

## Commands

Train:

```text
bash scripts/run_legacy_torch_runtime_matrix_20260807.sh
```

Eval: none. Rank arms by replica parity, initialization hashes, step-1 loss, and distance from archived
step-50 loss `4.203789` under the archived training configuration above.

## Outputs

Checkpoint: none; archived trainer checkpoints remain in arm-local `/tmp` directories until validation
and are then removed.

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_torch_runtime_matrix_20260807_v2/logs/`

Metrics: one `runtime_probe.json` per arm and a root `matrix_summary.json`

First job: `tide-chenyufei03-10498-kbvoj` (`rudder`, one 8xA800 node, eight single-GPU arms,
commit `7ff6d3196b51e478073e25e921db6b6f26cad11c`), failed before model initialization.

Second job: `tide-chenyufei03-10500-kswqr` (`rudder`, one 8xA800 node) was cancelled after remaining
Pending for 5 hours 3 minutes; its container never started.

Completed job: `tide-chenyufei03-10505-plfwk` (`rudder-exper`, one 8xA800 node, eight single-GPU arms,
commit `ffd31ce961782d56989091dc9bb3f3c99ca105d3`), completed in 4 minutes 49 seconds.

## Verification

- [x] Shell syntax and focused unit tests (2 tests passed)
- [x] Torch `2.10.0+cu128` compatibility overlay imports locally and on the GPU node
- [x] CFS runtime and compatibility archives match their registered SHA-256 values
- [x] Eight unique GPU assignments (`CUDA_VISIBLE_DEVICES=0..7`)
- [x] Four replicas per Transformers condition agree exactly
- [x] Reviewer required? no; diagnostic only

## Result / Notes

An image build was attempted first. The company registry returned HTTP `502` for the existing project
base image. A local PyTorch base reached the final check but Docker's `/dev/shm` data root had only
`6.9 GiB` free and could not commit the `7.32 GB` prepared runtime layer. The CFS runtime site contains
the same resolved packages without requiring a launch-time install.

Job `tide-chenyufei03-10498-kbvoj` failed in all eight arms before model initialization. The base image's
torchvision `0.23` attempted to register `torchvision::nms` against the overlaid torch `2.10.0+cu128` and
raised `RuntimeError: operator torchvision::nms does not exist`. The replacement compatibility overlay
uses torchvision `0.25.0+cu128`; a local import check passed with the exact versions registered above.

All four replicas within each Transformers condition produced identical losses. All eight arms produced
identical initialization, CPU RNG, and CUDA RNG hashes. Results below use seed `20260626`, batch `16`,
BF16, `lr=3e-4`, weight decay `0`, pinned T5 safetensors, and 50 catalog-only optimizer steps on A800:

```text
torch       transformers  step_1   step_50  delta_from_archived_step_50
2.10.0      4.57.6        4.546924 4.162638 -0.041151
2.10.0      5.3.0         4.546924 4.184667 -0.019122
2.8.0 ref   4.57.6        4.547071 4.188124 -0.015665
```

Torch `2.10.0+cu128` moves both Transformers conditions farther from the archived step-50 loss
`4.203789` than the prior torch `2.8.0+cu128` plus Transformers `4.57.6` reference. No new 950-step gate
is warranted. That prior reference already failed the 950-step strict-parity probe. The historical RTX
PRO 6000 Blackwell versus available A800 remains an unmatched hardware difference.
