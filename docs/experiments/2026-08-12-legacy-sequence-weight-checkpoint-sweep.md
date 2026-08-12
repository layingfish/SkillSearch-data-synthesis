# Experiment: Legacy sequence-weight checkpoint sweep

Status: planned
Risk Tier: 3

## Goal / Hypothesis / Change Scope

在 strict-base A800 训练协议上，直接使用 Official core75 的检索结果选择 checkpoint，并 sweep sequence-only prefix-ranking loss。目标是恢复或超过 RTX baseline，同时保住深召回。

逻辑 matrix 包含已完成的 `control` oracle，以及新训练的 `cum_002` 和 `seq_{001,002,004,006,008,010}`。每个新 arm 固定 catalog/query updates=`2,730/38,805`、batch=`16`、lr=`3e-4`、seed=`20260626`、BF16、dropout=`0.1`、`CUBLAS_WORKSPACE_CONFIG=:16:8`。sequence-only lambda=`{0.01,0.02,0.04,0.06,0.08,0.10}`；cumulative control lambda=`0.02`。

每个新 arm 在 query step=`{200,400,...,38,800,38,805}` 共评测 `195` 个 checkpoint。评测固定为 Official core75=`75` queries、beam=`50`、beam-order、CE log-prob、K=`{1,10,50}`。

`top1-best` 按 Overall R@1、R@10、R@50、较早 step 依次选择。`joint-best` 先要求 Overall R@10 `>=69.0%` 且 R@50 `>=75.0%`，再使用同一排序。Official core75 同时用于选择和结果汇报，因此该 matrix 用于性能恢复和配方搜索，不估计无泄漏泛化性能。

## Plan

1. 复用已验证的 snapshot queue 和七路 evaluator，增加同时保留 `top1-best` 与 `joint-best` 的通用 arm 分析。
2. 用 `seq_010`、`cum_002` 的历史 dev-best checkpoint、prediction、metrics 和 loss history 做轨迹不变性锚点。
3. 本地验证选择规则、双 checkpoint 保留、完整 step coverage、queue drain 和 matrix 聚合。
4. 一个 `8×A800` 节点顺序运行七个新 arm；每个 arm 使用 GPU `0` 训练，GPU `1–7` 动态评测。
5. 汇总各 arm 的 `top1-best`、`joint-best`，并复制 matrix 全局最佳 checkpoint。

## Commands

Train: `bash scripts/run_legacy_sequence_weight_checkpoint_sweep_node_20260812.sh`。Tianzhi 使用 `1×8 A800`、queue=`rudder-exper`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`；正式 job 和 launch commit 待提交后记录。

Eval: 与训练并行消费 snapshot；Official core75=`75` queries、beam=`50`、beam-order、CE log-prob。

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v1/arms/<arm>/{top1_best,joint_best}/`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v1/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v1/sequence_weight_checkpoint_sweep_summary.json`

## Verification

- [x] Minimal local check：Ruff、Python compile、两个 launcher 的 `bash -n`
- [x] Unit/invariant test：本地相关测试=`29`，通过=`17`、torch/CUDA 环境跳过=`12`
- [ ] `when gate=0, student_forward output must be allclose to base-model output`
- [ ] `seq_010` 与 `cum_002` 历史轨迹锚点 exact
- [ ] 每个新 arm 覆盖 `195` 个 checkpoint，queue 无残留
- [x] Reviewer required? yes；独立 reviewer 未返回，已完成结构化本地复核

## Result / Notes

独立 reviewer 在计划和实现复核时均没有返回；两次中止后改做结构化本地复核。CRITICAL=`0`。初始 HIGH=`2`：同一 snapshot 同时更新两种 selection 时可能因 move 丢失一份权重；snapshot hook 可能静默改变 ranking arm 的训练轨迹。第一项已通过同盘 move + hard-link clone 和后续独立替换测试关闭。第二项已实现四类 fail-closed exact checks，需在 A800 正式运行命中后关闭。

实现自审覆盖 checkpoint 所有权转移、worker failure 传播、step/dev-loss 一一对齐、queue drain、matrix tie-break 和 control oracle 复用。历史 exact check 已成为正式运行前置门禁，任何不一致都会终止 matrix。当前 unresolved CRITICAL=`0`、HIGH=`0`。
