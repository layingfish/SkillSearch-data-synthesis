# Experiment: Legacy A800 local FP32 decoder refinement
Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Phase 2 中 decoder FP32 的 `D_param`/step-50 loss-gap reduction=`14.6651%/52.8662%`，heads+CE FP32=`4.9188%/95.8414%`；条件均为 A800、default 对 workspace16、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0。两段信号互补，本轮测试 decoder 半区和 decoder+heads/CE 联合边界。

archived pipeline SHA-256=`7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76` 保持不变。训练 wrapper 只新增四个可组合局部 FP32 region；Phase 2 BF16 control artifacts 作为固定 reference。

## Commands

Train: job=`tide-chenyufei03-10594-pemnv`，queue=`rudder-exper`，nodes=1，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`ec164e4ae1f9072ce84dc54c7208708a3835f449`。节点内运行四组 default/workspace16 pair：decoder blocks 0–5、decoder blocks 6–11、full decoder+heads/CE、decoder blocks 0–5+heads/CE，占用 `8×A800`。

每个 arm 固定 catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、外层 BF16 autocast、AdamW foreach=None。

Eval: 复用 Phase 2 BF16 control 的 `D_param=1.04082965952`、step-50 loss gap=`0.0297207832336`（A800、default 对 workspace16、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0）。候选两项都降低至少 `50%` 才通过。

## Outputs

Artifacts: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_local_fp32_refinement_20260811_v1/`

Node log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_local_fp32_refinement_20260811_v1/logs/node.log`

Bootstrap log: `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/legacy_a800_local_fp32_refinement_20260811_v1.log`

Metrics: pending

## Verification

- [x] Minimal local check：21 项相关测试在 Python 3.10.12、torch 2.10.0+cu128 的冻结 runtime 下通过；`py_compile`、`bash -n`、trailing-whitespace 与 `git diff --check` 通过
- [ ] 四个 region 的 module call count 与 output dtype audit 通过
- [ ] 八个 arms 的 source/data/model/runtime/first-batch/init/RNG hash 对齐
- [ ] Phase 2 BF16 control compatibility audit 通过
- [x] Reviewer required? yes；structured local review 未发现训练顺序、autocast 边界、module 选择、跨 job control 或八卡隔离的 CRITICAL/HIGH 问题；GPU provenance 与 dtype audit 仍是硬门禁

## Result / Notes

- Official core75 不在本阶段运行；本轮只做 50-step 局部精度定位。
