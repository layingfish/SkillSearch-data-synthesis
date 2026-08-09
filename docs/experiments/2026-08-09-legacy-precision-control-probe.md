# Experiment: Legacy precision-control probe
Status: planned
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

Metrics: one `runtime_probe.json` per arm and `matrix_summary.json` under the same root.

Job: pending submission to `rudder-exper`, one node, eight single-GPU arms.

## Verification

- [x] Shell syntax, Python compile, and focused unit tests (3 tests passed)
- [ ] Eight unique GPU assignments
- [ ] Input/order/model/RNG hashes match the archived contract
- [ ] Precision controls are recorded before the first forward
- [ ] Reviewer required? no; diagnostic probe only

## Result / Notes

Pending.
