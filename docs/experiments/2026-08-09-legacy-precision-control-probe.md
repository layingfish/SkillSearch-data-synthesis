# Experiment: Legacy precision-control probe
Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

定位归档 RTX PRO 6000 与当前 A800 在同一 legacy catalog probe 上的首步数值分叉。
输入、SemID、T5 revision、source commit、shuffle、batch、optimizer 默认参数保持固定。
只改变 CUDA precision/backend controls；每个 arm 使用一张 GPU。

固定条件：source commit `15f6c51131f3b6f3e8580aa4564b17cbe13a9790`，seed `20260626`，
catalog batch `16`，lr `3e-4`，weight decay `0`，50 catalog optimizer steps，
T5-base revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`，torch `2.10.0+cu128`，
Transformers `5.3.0`，CUDA `12.8`，BF16 或显式 FP32。

PyTorch 官方文档说明 BF16 GEMM 默认允许 reduced-precision reduction，并允许通过
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False` 关闭：
<https://pytorch.org/docs/stable/notes/cuda.html#reduced-precision-reduction-in-bf16-gemms>。
跨平台不保证浮点结果逐 bit 一致：
<https://pytorch.org/docs/stable/notes/randomness.html>。

## Commands

Train:

```text
bash scripts/run_legacy_precision_control_matrix_20260809.sh
```

Eval: none. Compare step 1/50 losses, initialization hashes, RNG hashes, and runtime flags
against the archived RTX reference (`4.528710842/4.203788757`).

## Outputs

Checkpoint: none; each arm uses an ephemeral catalog-only checkpoint and removes it after validation.

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_precision_control_probe_20260809_v1/logs/`

Metrics phase 1: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_precision_control_probe_20260809_v1/matrix_summary.json`.

Metrics phase 2: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_blas_control_probe_20260809_v2/matrix_summary.json`.

Phase-1 job: `tide-chenyufei03-10562-ptvcn` (`rudder-exper`, one 8xA800 node,
eight single-GPU arms, launch commit `1b24d27fd9f97e615a600f70b1d2f2f1e8a392b1`).

Phase-2 job: pending; one `rudder-exper` 8xA800 node, eight single-GPU BLAS/kernel arms.

## Verification

- [x] Shell syntax, Python compile, and focused unit tests (3 tests passed)
- [x] Eight unique GPU assignments in phase 1
- [x] Input/order and RNG hashes match the archived contract in phase 1
- [x] Precision controls are recorded by the training child before the first forward
- [ ] Reviewer required? no; diagnostic probe only

## Result / Notes

Phase 1 completed in 4 minutes 48 seconds with no pod restart. Under the fixed 50-step catalog
protocol above, the A800 default BF16 arm reproduced step 1/50 losses
`4.546923637/4.184666634`; the archived RTX reference is `4.528710842/4.203788757`.

Closing BF16 reduced-precision reduction changed A800 step 1/50 to
`4.546850681/4.178029537`. TF32 matmul, cuDNN TF32, and explicit highest FP32 matmul precision
were bitwise identical to the default BF16 arm. Full FP32 without autocast produced
`4.536283493/4.166709423`; explicit `foreach=False` produced
`4.546923637/4.164748192`. None approached the archived RTX trajectory.

Phase 2 tests cuBLAS/cuBLASLt preference, deterministic algorithms, and
`CUBLAS_WORKSPACE_CONFIG={:4096:8,:16:8}` under the same fixed 50-step protocol.
