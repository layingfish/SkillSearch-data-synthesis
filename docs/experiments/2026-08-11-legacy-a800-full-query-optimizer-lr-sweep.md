# Experiment: Legacy A800 full-query optimizer/lr sweep
Status: planned
Risk Tier: 2

## Goal / Hypothesis / Change Scope

在 `workspace_16` 上直接选择能恢复 Top-1 的 A800 query 优化配方。六个 selection arms 为 optimizer state=`{carry, reset}` × query lr=`{1e-4, 2e-4, 3e-4}`；catalog 固定 lr=`3e-4`。另外运行一个 `workspace_16/carry/query lr=3e-4` exact replica 和一个 `default/carry/query lr=3e-4` legacy anchor，占满单节点 `8×A800`。

训练主体使用 archived single-SID pipeline SHA-256=`7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76`。wrapper 只在 query step 1 的 AdamW `step()` 前设置 query lr，并按 arm 保留或清空 state。best/final checkpoint 只保存模型；last checkpoint 写盘被抑制。该存储策略不参与 forward、backward、optimizer step、dev checkpoint selection 或 RNG。

## Commands

Train: pending。目标 queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`。

每个 arm：catalog=`2,730` updates、query=`38,805` updates、batch=`16`、eval batch=`16`、seed=`20260626`、catalog lr=`3e-4`、query lr 按 arm、weight decay=`0`、betas=`(0.9,0.999)`、eps=`1e-8`、foreach=`None`、dropout=`0.1`、外层 BF16、每 `200` 个 query updates 评测 dev loss并选择 best。

Eval: Official core75=`75` queries、beam=`50`、beam-order aggregation、CE log-prob、K=`{1,10,50}`；六个 `workspace_16` selection arms 参与选择，两条锚点仅做审计。canonical private test 不参与本轮。

## Outputs

Artifacts: pending

Node log: pending

Bootstrap log: pending

Metrics: pending

## Verification

- [x] Minimal local check：六个 legacy probe test modules 共 `34` 项在 Python=`3.10.12`、torch=`2.10.0+cu128` 的 frozen runtime 下通过；`py_compile`、`bash -n` 与 `git diff --check` 通过；本机未安装 `shellcheck`
- [ ] archived source/data/model/runtime/init 与 catalog/query 首批 hash 全部命中
- [ ] 七个 `workspace_16` arms 的 catalog-final model/optimizer/RNG 与 query step-1 gradient/loss exact
- [ ] `workspace_16/carry/query lr=3e-4` 与 replica 的 final model、best step/dev loss、predictions exact
- [ ] `default/carry/query lr=3e-4` 与历史 A800 full-run 锚点对齐
- [ ] 六个 selection arms 全部生成 Official core75 指标
- [x] Reviewer required? yes；structured local review 已检查 query step-1 transition、AdamW reset、dev forward 与 train step 计数、best checkpoint 选择、模型-only checkpoint、八卡隔离和 selection/anchor 分离，未发现 CRITICAL/HIGH 问题；GPU provenance/invariant 仍是提交后的硬门禁

## Result / Notes

- 通过门槛：Overall R@1 `>=27.3683`、R@10 `>=69.0000`、R@50 `>=75.0000`（Official core75=`75` queries、beam=`50`、beam-order）；Single R@1 `>=41.6667`（Single=`24` queries）；Multi R@1 `>=21.1102`（Multi=`51` queries）。
- 若多个 arm 全部通过，按 Overall R@1、Single R@1、Multi R@1、Overall R@10、较低 dev loss 的顺序选择。若无人全部通过，先在 Overall R@10/R@50 达标的 arms 中按同一顺序选最佳；若深召回也无人达标，则直接按该顺序报告最佳 arm，供后续 warmup/clipping 单变量实验使用。
- wrapper 预期每个 arm 执行 optimizer step=`41,535`、query dev eval=`195` 次、每次 dev batch=`157`，因此 batch/forward call 总数必须为 `72,150`；条件为 catalog=`2,730` updates、query=`38,805` updates、eval steps=`200`、query-dev rows=`2,501`、batch=`16`。
- `workspace_16/carry/query lr=3e-4` 与 `default/carry/query lr=3e-4` 的历史锚点常量已用 2026-08-09 full-run artifact 做本地 exact check，best step/dev loss 与 Official core75 R@1/10/50 均匹配。
