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

Metrics phase 3: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_python_abi_probe_20260809_v1/matrix_summary.json`.

Phase-3 cp310 runtime: `/models_rw/SkillSearch-data-synthesis/runtime/torch210-tf53-py310-v1/runtime.tar.zst`,
`1,019,567,918` bytes, SHA-256
`6102937effeaf983f44455c47adfce34ef2adc1afffcef6f5c6193288bb36b72`.

Phase-1 job: `tide-chenyufei03-10562-ptvcn` (`rudder-exper`, one 8xA800 node,
eight single-GPU arms, launch commit `1b24d27fd9f97e615a600f70b1d2f2f1e8a392b1`).

Phase-2 job: `tide-chenyufei03-10564-ekstg` (`rudder-exper`, one 8xA800 node,
eight single-GPU BLAS/kernel arms, launch commit `d7ae8686bc7a3ceff1c71b3f93528c0337698840`).

Phase-3 job: `tide-chenyufei03-10568-tartg` (`rudder-exper`, one 8xA800 node,
launch commit `46094cd0dd12d25ce57ab78ab2c61e8af8f2fb04`) failed before runtime extraction.
The pod started with cwd inside an empty `/workspace`; alias initialization replaced that directory,
so the subsequent pip bootstrap could not resolve `os.getcwd()`. The retry changes cwd to `/tmp`
before alias initialization. Four arms use Python `3.10.12`
with the official cp310 torch `2.10.0+cu128` wheel, and four controls use Python `3.11.11`
with the existing cp311 wheel. Every arm uses default BF16 controls, seed `20260626`, batch
`16`, lr `3e-4`, weight decay `0`, and exactly 50 catalog optimizer steps. The cp310 runtime
uses NetworkX `3.4.2`, the latest release compatible with Python `3.10`; the cp311 control uses
NetworkX `3.6.1`. CUDA package versions remain cuBLAS `12.8.4.1`, cuDNN `9.10.2.21`, and
CUDA runtime `12.8.90` in both conditions.

## Verification

- [x] Shell syntax, Python compile, and focused unit tests (3 tests passed)
- [x] Eight unique GPU assignments in phase 1
- [x] Input/order and RNG hashes match the archived contract in phase 1
- [x] Precision controls are recorded by the training child before the first forward
- [x] Phase-2 input/order and RNG hashes match the archived contract
- [x] Phase-3 cp310 archive hash, relocation, ABI tags, and package metadata verified locally
- [ ] Phase-3 cp310/cp311 imports and CUDA package versions pass GPU-node preflight
- [ ] Four replicas agree within each Python ABI condition in phase 3
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

Phase 2 completed with no arm failure. Under the fixed 50-step catalog protocol above,
default, explicit cuBLAS, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms plus
`:4096:8`, and `CUDA_MODULE_LOADING=EAGER` all reproduced A800 step-1/50 losses
`4.546923637/4.184666634` exactly. Explicit cuBLASLt produced
`4.546899319/4.164062500`; cuBLASLt with BF16 reduction disabled produced
`4.546899319/4.169090271`; and `CUBLAS_WORKSPACE_CONFIG=:16:8` produced
`4.546948433/4.171227932`. None reproduced the archived RTX step-1/50 reference
`4.528710842/4.203788757` under the same seed, batch, BF16, optimizer, and 50-step conditions.

Phase 3 isolates the remaining Python ABI and torch-wheel variable before attributing the
first-forward split to GPU architecture or driver behavior. The cp310 wheel SHA-256 is
`36368507b56eaa51acbd3c96ac8893bb9a86991ffcd0699fea3a1a74a2b8bdcb`.
