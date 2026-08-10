# Experiment: Legacy A800 layerwise numerical trace
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

定位 A800 legacy BF16 训练中数值差异最早出现及明显放大的模块。比较 default BF16、`CUBLAS_WORKSPACE_CONFIG=:16:8` BF16、cuBLASLt BF16 与 FP32 reference。每种路径同时跑 `trace=on/off`，验证 instrumentation 不改变训练结果。

变更只新增独立诊断 wrapper、节点 launcher 和结果分析器。archived legacy 训练脚本保持不变。

## Commands

Train wrapper:

```text
bash scripts/run_legacy_layerwise_trace_tianzhi_20260811.sh
```

Tianzhi submission uses one `8×A800` node on `rudder-exper` with image `chenyufei03/skillsearch-gr:cu128-torch28-v1`. The bootstrap receives the verified launch commit through `SKILLGR_LAUNCH_COMMIT` and runs from a temporary archived checkout.

The node launcher starts eight single-GPU arms: default BF16, `CUBLAS_WORKSPACE_CONFIG=:16:8` BF16, cuBLASLt BF16, and FP32; each path has `trace=on/off`. Every arm uses catalog first batch, batch=16, seed=20260626, lr=3e-4, weight decay=0, dropout=0, updates=1.

Eval: 对四组 `trace=on/off` 配对检查 loss、optimizer 后参数与 provenance；对四个 traced arm 生成逐层差异报告。

## Outputs

Checkpoint: none；one-step model states are represented by region SHA-256 and raw parameter-update tensors, not retained training checkpoints.

Artifacts: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_layerwise_trace_20260811_v1/`

Log: `/models_rw/SkillSearch-data-synthesis/logs/legacy_a800_layerwise_trace_20260811_v1.log`

Metrics: pending

## Verification

- [x] Minimal local check：`py_compile`、`bash -n`、12 related unit tests passed using frozen Python 3.10.12 + torch 2.10.0 runtime
- [ ] `trace=off` wrapper 与原始无 wrapper 路径等价
- [ ] 四组 `trace=on/off` 的 step-1 loss 与 optimizer 后参数一致
- [ ] manifest、源码/data/model/runtime/首批 row/init/RNG/config hash 一致
- [x] Reviewer required? yes；当前执行策略不允许调用独立 sub-agent，已完成 structured local review，并把 trace/control 精确等价设为 GPU 硬门禁

## Result / Notes

- 用户指定队列为 `rudder-exper`。
- 单节点八张 A800 分配给八个独立单卡 arm；不使用 DDP，避免改变 archived 单卡训练语义。
- default BF16 control 必须精确复现 step-1 loss=`4.554369926452637`（A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、default backend）。
