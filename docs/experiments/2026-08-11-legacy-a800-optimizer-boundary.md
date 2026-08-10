# Experiment: Legacy A800 catalog-to-query optimizer boundary
Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

测试 catalog 阶段累积的 AdamW moments 是否在 query 阶段放大 default/workspace16 数值路径差异。首轮同时测 query lr=`{2e-4, 3e-4}`，形成 backend × optimizer state × query lr 的 `2×2×2` factorial；同一 lr 内可独立估计 carry/reset 效果。

训练主体继续使用 archived single-SID pipeline SHA-256=`7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76`。wrapper 只在 query step 1 的 AdamW `step()` 前保留或清空 state，并设置 query lr；forward、backward、dropout RNG、batch shuffle 和参数顺序保持原样。

## Commands

Train: v1 job=`tide-chenyufei03-10595-comjs`，queue=`rudder-exper`，nodes=`1`，节点规格=`8×A800`，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`b0786627d891c22b1d516b1999553a0a0283429a`。八个 arm 完成训练后被 wrapper 的 catalog row-hash 口径错误拒绝；v2 待提交。

每个 arm 的 catalog 固定 2,730 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0.1、外层 BF16、AdamW foreach=None。query 固定前 50 updates、batch=16、dropout=0.1；只改变 optimizer state 和 query lr。

Eval: 在 CPU FP64 上计算 `D_update = ||delta_default - delta_workspace||_2 / ||delta_default||_2`，delta 为 query step 50 相对各自 catalog-final model 的参数变化。reset 相对同 lr carry 的 `D_update` 与 step-50 loss gap 都降低至少 `50%` 才进入完整 query 训练。

## Outputs

Artifacts: v2 `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_optimizer_boundary_20260811_v2/`

Node log: v2 `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_optimizer_boundary_20260811_v2/logs/node.log`

Bootstrap log: v2 `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/legacy_a800_optimizer_boundary_20260811_v2.log`

Metrics: pending

## Verification

- [x] Minimal local check：25 项相关测试在 Python 3.10.12、torch 2.10.0+cu128 的冻结 runtime 下通过；`py_compile`、`bash -n`、trailing-whitespace 与 `git diff --check` 通过
- [ ] default catalog step-1/50 loss 命中 frozen A800 reference
- [ ] catalog/query 首批 row hash 命中取证包 reference
- [ ] 每个 backend 内四个 catalog-final model/optimizer/RNG/gradient hash exact
- [ ] reset arm 在 query step 1 update 前 optimizer state entry count=`0`
- [ ] 八个 arms 的 source/data/model/runtime/init provenance 对齐
- [x] Reviewer required? yes；structured local review 未发现阶段边界、AdamW reset、RNG、batch、state artifact 或八卡隔离的 CRITICAL/HIGH 问题；GPU 边界 invariant 仍是硬门禁

## Result / Notes

- 本轮不跑 Official core75；50-step query probe 只判断 optimizer state 是否是主要放大器。
- post-run last/final checkpoint save 被 observer 抑制；每个 arm 只保存 catalog-final 与 query-step-50 trainable state，供 `D_update` 精确计算。
- v1 的八个 arm 均完成 catalog=`2,730` updates 与 query=`50` updates（batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`），随后统一失败于 post-run catalog row hash。根因是 wrapper 对 catalog row 误加了 `|positive_skill_id`；取证口径对 catalog 只 hash `index`，对 query 才 hash `index|positive_skill_id`。训练主体未失败，但 v1 未写出 state artifact，不能计算 `D_update`。
- v2 已把 catalog/query row-id 口径显式分开，并把 hash 门禁前移到对应首批，避免再到训练结束才报错。v1 日志完整保留。
