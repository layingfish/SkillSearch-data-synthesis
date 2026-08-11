# Experiment: Legacy A800 query stability ablation
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

在已验证的 carry/query lr=`3e-4` 配方上，分别测试 query `warmup_steps=500` 与 query `max_grad_norm=1.0` 能否恢复 Top-1，同时保持深召回。两项不组合。catalog 阶段保持 legacy 语义不变。

单节点运行八个独立单卡 arm：backend=`{default, workspace_16}` × intervention=`{control, warmup, clip}` 六路，加 `workspace_16/warmup` 与 `workspace_16/clip` 各一条 exact replica。所有 arm 固定 optimizer state=`carry`、catalog lr=`3e-4`、query base lr=`3e-4`。

warmup 仅作用于 query optimizer step。query step `s` 使用 lr=`3e-4 × min(s/500, 1)`，因此 query step=`1/500/501` 的 lr=`6e-7/3e-4/3e-4`。clip 仅作用于 query backward 后、AdamW step 前，global L2 `max_grad_norm=1.0`、`foreach=True`、非有限 norm 直接失败。

## Commands

Train: pending；queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`。

每个 arm：catalog=`2,730` updates、query=`38,805` updates、batch=`16`、eval batch=`16`、seed=`20260626`、catalog lr=`3e-4`、query base lr=`3e-4`、weight decay=`0`、betas=`(0.9,0.999)`、eps=`1e-8`、dropout=`0.1`、外层 BF16、每 `200` 个 query updates 评测 dev loss并选择 best。

Eval: Official core75=`75` queries、beam=`50`、beam-order aggregation、CE log-prob、K=`{1,10,50}`。两个 workspace intervention arms 参与选择；control、default pair 与 replica 仅做效果归因和审计。canonical private test 不参与本轮。

## Outputs

Artifacts: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_query_stability_20260811_v1/`

Node log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_query_stability_20260811_v1/logs/node.log`

Bootstrap log: `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/legacy_a800_query_stability_20260811_v1.log`

Metrics: pending

## Verification

- [x] Minimal local check：七个 legacy test modules 共 `40` 项在 Python=`3.10.12`、torch=`2.10.0+cu128` frozen runtime 下通过；`py_compile`、`bash -n` 与 `git diff --check` 通过
- [ ] archived source/data/model/runtime/init 与 catalog/query 首批 hash 全部命中
- [ ] 每个 backend 内的 catalog-final model/optimizer/RNG 与 query step-1 pre-intervention gradient exact
- [ ] warmup lr trace 命中 query step=`1/500/501` 的 `6e-7/3e-4/3e-4`
- [ ] clipping 调用次数=`38,805`，global L2 threshold=`1.0`，无 NaN/Inf
- [ ] workspace warmup 与 clip 的两个 exact replica invariant 通过
- [ ] default/workspace control 精确命中 2026-08-09 历史 anchor
- [ ] 八个 arm 全部生成 Official core75 指标
- [x] Reviewer required? yes；structured local review 已检查 query step 顺序、warmup lr、clip 时机、catalog invariant、checkpoint selection、历史 artifact hash、replica 和 GPU `0–7` 映射，未发现 CRITICAL/HIGH 问题

## Result / Notes

- 通过门槛：Overall R@1 `>=27.3683`、R@10 `>=69.0000`、R@50 `>=75.0000`（Official core75=`75` queries、beam=`50`、beam-order）；Single R@1 `>=41.6667`（Single=`24` queries）；Multi R@1 `>=21.1102`（Multi=`51` queries）。
- 若 warmup 与 clip 都通过，按 Overall R@1、Single R@1、Multi R@1、Overall R@10、较低 dev loss 的顺序选择。若均失败，报告各自相对 workspace control 与对应 default pair 的变化，再决定是否有依据测试组合或结束 Phase 5。
