# Experiment: Legacy core75 checkpoint oracle

Status: planned
Risk Tier: 3

## Goal / Hypothesis / Change Scope

诊断 strict-base A800 control 的低 R@1 是否主要来自 dev-loss 选模。GPU `0` 精确复现 auxiliary disabled 的 control 轨迹；在 query step=`{200,400,...,38,800,38,805}` 共 `195` 个时点保存临时 model-only checkpoint。GPU `1–7` 并行评测 Official core75=`75` queries、beam=`50`、beam-order、CE log-prob、K=`{1,10,50}`。

训练固定为 source rows=`94,912`、expanded rows=`124,175`、catalog/query updates=`2,730/38,805`、batch=`16`、seed=`20260626`、catalog/query lr=`3e-4`、dropout=`0.1`、BF16、`CUBLAS_WORKSPACE_CONFIG=:16:8`。test-best 按 Overall R@1、R@10、R@50、较早 step 依次排序。该选择复用 Official core75，只用于诊断，不作为可报告 test 指标。

## Plan

1. 在现有 full-query observer 增加默认关闭的 eval-step snapshot hook；关闭时保存策略与 artifact 保持不变。
2. snapshot 在 query eval 后、下一次 optimizer update 前生成；terminal step=`38,805` 在 best restore 前生成。
3. 用 node-local 临时队列承载 checkpoint；七个 evaluator worker 原子 claim，持久化每步 predictions、metrics 和 hash。
4. 每个 worker 只保留自身 R@1 最优临时权重；全部 `195` 步完成后复制全局 test-best 到 `/models_rw`。
5. 验证 step=`38,200` checkpoint、prediction 与历史 strict-base control hash 完全一致。
6. 验证 loss history 的 `195` 个 eval step 与 oracle results 一一对应，且全局排序可独立重算。
7. 若 max Overall R@1 `>=27.3683%`（Official core75=`75` queries、beam=`50`、beam-order），记为距离 RTX R@1=`28.3683%` 不超过 `1` 个点；否则训练轨迹未进入目标区间。

## Commands

Train: pending.

Eval: pending.

## Outputs

Checkpoint: pending.

Log: pending.

Metrics: pending.

## Verification

- [x] Minimal local check：`py_compile`、Ruff、两个 launcher 的 `bash -n` 均通过
- [x] Unit/invariant test：`19` tests passed（本地无 torch，tensor-only 的 `2` tests skipped；pod preflight 将用固定 CUDA runtime 补跑）
- [x] Reviewer required? yes；复核后 unresolved CRITICAL=`0`、HIGH=`0`
- [ ] snapshot timing：eval 后、下一 update 前；terminal snapshot 在 best restore 前
- [ ] eval steps=`195`，无缺失、重复或额外 step
- [ ] step=`38,200` 命中历史 strict-base model/prediction/metrics hash
- [ ] test-best 排序独立重算一致
- [ ] pod 使用 `1×8 A800`，训练 GPU=`1`、evaluator GPU=`7`

## Result / Notes

实现交叉审查初次结果为 CRITICAL=`0`、HIGH=`1`。HIGH 指向历史 control artifact 未固定 hash，已增加两份 JSON 的 SHA-256 检查，并直接固定 step=`38,200`、dev loss=`0.7701343703584546`、model-state hash 与 prediction hash。复核结果为 unresolved CRITICAL=`0`、HIGH=`0`。
