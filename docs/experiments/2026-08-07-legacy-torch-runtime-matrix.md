# Experiment: Legacy torch runtime matrix
Status: running
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
extracts once to node-local `/tmp`, prepends the site to `PYTHONPATH`, and fails before starting arms
unless torch reports `2.10.0+cu128`, CUDA `12.8`, and cuDNN `91002`. No torch installation occurs in the
GPU job. Transformers `4.57.6` is the only arm-local Python overlay; it uses huggingface-hub `0.36.2`,
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

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_torch_runtime_matrix_20260807_v1/logs/`

Metrics: one `runtime_probe.json` per arm and a root `matrix_summary.json`

Job: `tide-chenyufei03-10498-kbvoj` (`rudder`, one 8xA800 node, eight single-GPU arms,
commit `7ff6d3196b51e478073e25e921db6b6f26cad11c`)

## Verification

- [x] Shell syntax and focused unit tests (2 tests passed)
- [ ] Torch `2.10.0+cu128` overlay import and CUDA/cuDNN preflight
- [x] CFS runtime archive matches the registered SHA-256
- [x] Eight unique GPU assignments (`CUDA_VISIBLE_DEVICES=0..7`)
- [ ] Four replicas per Transformers condition agree
- [x] Reviewer required? no; diagnostic only, followed by a 950-step parity gate

## Result / Notes

An image build was attempted first. The company registry returned HTTP `502` for the existing project
base image. A local PyTorch base reached the final check but Docker's `/dev/shm` data root had only
`6.9 GiB` free and could not commit the `7.32 GB` prepared runtime layer. The CFS runtime site contains
the same resolved packages without requiring a launch-time install.
