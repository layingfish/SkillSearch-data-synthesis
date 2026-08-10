# Experiment: Legacy A800 layerwise numerical trace
Status: done
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

Submitted request:

```text
jobID=tide-chenyufei03-10589-irhff
queue=rudder-exper
nodes=1
image=chenyufei03/skillsearch-gr:cu128-torch28-v1
launch_commit=ed1527ad8e1893b41998638df9f471d8b7e2f2d2
```

The node launcher starts eight single-GPU arms: default BF16, `CUBLAS_WORKSPACE_CONFIG=:16:8` BF16, cuBLASLt BF16, and FP32; each path has `trace=on/off`. Every arm uses catalog first batch, batch=16, seed=20260626, lr=3e-4, weight decay=0, dropout=0, updates=1.

Eval: 对四组 `trace=on/off` 配对检查 loss、optimizer 后参数与 provenance；对四个 traced arm 生成逐层差异报告。

## Outputs

Checkpoint: none；one-step model states are represented by region SHA-256 and raw parameter-update tensors, not retained training checkpoints.

Artifacts (failed v1): `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_layerwise_trace_20260811_v1/`

Artifacts (v2 retry): `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_layerwise_trace_20260811_v2/`

Log: `/models_rw/SkillSearch-data-synthesis/logs/legacy_a800_layerwise_trace_20260811_v1.log`

Metrics: pending

## Verification

- [x] Minimal local check：`py_compile`、`bash -n`、12 related unit tests passed using frozen Python 3.10.12 + torch 2.10.0 runtime
- [x] `trace=off` wrapper 与原始无 wrapper 路径等价
- [x] 四组 `trace=on/off` 的 step-1 loss 与 optimizer 后参数一致
- [x] manifest、源码/data/model/runtime/首批 row/init/RNG/config hash 一致
- [x] Reviewer required? yes；当前执行策略不允许调用独立 sub-agent，已完成 structured local review，并把 trace/control 精确等价设为 GPU 硬门禁

## Result / Notes

- 用户指定队列为 `rudder-exper`。
- 单节点八张 A800 分配给八个独立单卡 arm；不使用 DDP，避免改变 archived 单卡训练语义。
- default BF16 control 必须精确复现 step-1 loss=`4.554369926452637`（A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、default backend）。
- `tide-chenyufei03-10589-irhff` 已提交，等待 pod 启动。
- v1 四个 control arms 均完成。四个 traced arms 均完成 one-step loss，但 post-run validation 一致报 `shared_embedding` boundary 缺失。
- v1 根因：hook 注册在 `backbone.shared`，实际 encoder 调用 `backbone.encoder.embed_tokens`。已修正并增加 regression test；v2 将使用全新输出目录和 job name。
- v2 retry：`tide-chenyufei03-10590-tdbdv`（`rudder-exper`、nodes=1、`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`、launch commit=`11fd64fa8952f0fda9dacbef5ca40bd159392ecc`）。
- v2 完成八个 arm；四组 trace/control 的 step-1 loss 与 final-state SHA-256 均精确一致，`pair_equivalence_passed=true`。每个 traced arm 保存 573 个 tensor artifacts；条件为 A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、updates=1。
- default BF16 对 workspace16/cuBLASLt BF16 的 shared embedding、encoder blocks 0–11、SID embedding 全部 exact；首个非零 activation boundary 都是 `decoder.block.0`。
- workspace16 相对 default 在 decoder block 0/1/2 的 relative L2=`1.811838e-05/4.396393e-04/1.063370e-03`；block 1 相对 block 0 放大 `24.2648×`，block 2 相对 block 1 放大 `2.4187×`。条件为上述 one-step trace，两条路径均为 BF16。
- cuBLASLt 相对 default 在 decoder block 0/1/2 的 relative L2=`0.001219331/0.003668086/0.006265732`；对应放大率为 `3.0083×/1.7082×`。条件同上，仅候选 backend 为 cuBLASLt。
- workspace16/cuBLASLt 的最大 activation relative L2=`0.004666657/0.009643726`，最大 gradient relative L2=`0.126297/0.428937`，最大 update relative L2=`0.269945/0.530078`；条件均为上述 one-step trace。
- authoritative outputs：`/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_layerwise_trace_20260811_v2/layerwise_comparison.{json,md}`。
