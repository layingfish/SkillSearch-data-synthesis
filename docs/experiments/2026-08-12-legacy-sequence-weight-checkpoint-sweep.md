# Experiment: Legacy sequence-weight checkpoint sweep

Status: running
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
4. 一个 `8×A800` 节点并行运行七个新 arm；GPU `0–6` 各训练一个 arm，GPU `7` 串行消费各 arm 的 snapshot。某个训练结束后，对应 GPU 转为该 arm 的第二 evaluator。
5. 汇总各 arm 的 `top1-best`、`joint-best`，并复制 matrix 全局最佳 checkpoint。

## Commands

Train: v4 job=`tide-chenyufei03-10665-lihis`，pod=`tide-chenyufei03-10665-lihis-worker-0`，launch commit=`14d320698591841181038d2ed2c8d1c3767348b0`，`1×8 A800`、queue=`rudder-exper`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`。节点入口为 `bash scripts/run_legacy_sequence_weight_checkpoint_sweep_node_20260812.sh`。

Eval: 与训练并行消费 snapshot；Official core75=`75` queries、beam=`50`、beam-order、CE log-prob。

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v4/arms/<arm>/{top1_best,joint_best}/`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v4/logs/`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v4/sequence_weight_checkpoint_sweep_summary.json`

## Verification

- [x] Minimal local check：Ruff、Python compile、两个 launcher 的 `bash -n`
- [x] Unit/invariant test：本地相关测试=`36`，通过=`24`、torch/CUDA 环境跳过=`12`
- [x] `when gate=0, student_forward output must be allclose to base-model output`；v4 节点测试=`36/36` 通过
- [ ] `seq_010` 与 `cum_002` 历史轨迹锚点 exact
- [ ] 每个新 arm 覆盖 `195` 个 checkpoint，queue 无残留
- [x] Reviewer required? yes；独立 reviewer 未返回，已完成结构化本地复核

## Result / Notes

独立 reviewer 在计划和实现复核时均没有返回；两次中止后改做结构化本地复核。CRITICAL=`0`。初始 HIGH=`2`：同一 snapshot 同时更新两种 selection 时可能因 move 丢失一份权重；snapshot hook 可能静默改变 ranking arm 的训练轨迹。第一项已通过同盘 move + hard-link clone 和后续独立替换测试关闭。第二项已实现四类 fail-closed exact checks，需在 A800 正式运行命中后关闭。

实现自审覆盖 checkpoint 所有权转移、worker failure 传播、step/dev-loss 一一对齐、queue drain、matrix tie-break 和 control oracle 复用。历史 exact check 已成为正式运行前置门禁，任何不一致都会终止 matrix。当前 unresolved CRITICAL=`0`、HIGH=`0`。

首次 job=`tide-chenyufei03-10662-bawkh` 使用 launch commit=`5b88e67`，节点识别 GPU count=`8`，固定 runtime 全部命中，节点测试=`29/29` 通过。随后在 optimizer step=`0` 前失败：worker 先创建了 `arms/cum_002`，训练器按 fail-closed overwrite guard 拒绝该目录。失败清理又因裸 `wait` 等待 node-level tee 而挂起；作业已取消，v1 日志保留，没有训练更新或 checkpoint。v2 改为训练进程先原子创建 arm root，再启动七个 evaluator；清理只等待显式 PID。

第二次 job=`tide-chenyufei03-10663-naxfb` 使用 launch commit=`31b5329`，固定 runtime 与节点测试=`31/31` 全部通过。训练器在 optimizer step=`0` 前执行 train/dev tokenizer audit；该 audit 覆盖 expanded train/dev rows=`124,175/2,501`，超过 launcher 等待 arm root 的 `60` 秒后被终止。v2 没有训练更新或 checkpoint。v3 将 evaluator 输出移到独立 `oracle/<arm>`，训练器独占 `arms/<arm>`，移除基于初始化耗时的同步假设。

v3 job=`tide-chenyufei03-10664-kmoen` 已完成调度 profiling。Pod ready=`1/1`、restart=`0`，识别 GPU count=`8`；Python=`3.10.12`、torch=`2.10.0+cu128`、CUDA package=`12.8`、cuDNN=`9.10.2`、Transformers=`5.3.0`、Tokenizers=`0.22.2` 全部命中。节点测试=`31/31` 通过。`cum_002` 的首个 query checkpoint 在 step=`200` 完成评测，Overall R@1/R@10/R@50=`1.61%/9.64%/15.22%`（Official core75=`75` queries、beam=`50`、beam-order、CE log-prob；训练 batch=`16`、lr=`3e-4`、seed=`20260626`、BF16、dropout=`0.1`）。单个 evaluator 约 `20–30` 秒完成一次评测，训练约 `60–90` 秒才产生一个 snapshot；原先 `1` trainer + `7` evaluator 的调度让多数 GPU 长时间空闲。

v4 调度固定 GPU `0–6` 同时训练七个 arm。GPU `7` 上的七个 arm worker 通过同一个 `fcntl` 文件锁串行评测，避免同卡并发占显存。每个 trainer 完成后，其 GPU 立即加入对应 arm 的 queue drain。训练协议、snapshot 内容、评测协议和 checkpoint 选择规则均未改变。本地相关测试=`36`，通过=`24`、因本机无 torch/CUDA 跳过=`12`；Ruff、Python compile、shell syntax、`wait -n -p` 行为和 git diff check 均通过。

v4 job=`tide-chenyufei03-10665-lihis` 已启动。Pod ready=`1/1`、restart=`0`；固定 runtime 全部命中，节点测试=`36/36` 通过。七个 trainer 分别绑定 GPU `0–6`，均到 catalog step=`50/2,730` 且 train loss=`4.171228`（各 arm：batch=`16`、lr=`3e-4`、seed=`20260626`、BF16、dropout=`0.1`、`CUBLAS_WORKSPACE_CONFIG=:16:8`）。采样时 GPU `0–6` utilization=`77%–100%`、显存约=`47.4 GiB/card`；GPU `7` 等待 query snapshot。
