# Experiment: Legacy core75 checkpoint oracle

Status: done
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

Train: job=`tide-chenyufei03-10655-wiaba`，queue=`rudder-exper`，node=`1×8 A800`，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`192cff306060b8ff8b0a82aa9357a59623b99596`。

Eval: GPU `1–7` 动态领取 GPU `0` 产生的 checkpoint；Official core75=`75` queries、beam=`50`、beam-order、CE log-prob、K=`{1,10,50}`。

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_core75_checkpoint_oracle_20260812_v1/test_best/structured_decoder.pt`，step=`37,000`，model-only，file SHA-256=`6b62f753db4189bd72d7cd27f777eb17e18d7b324ea6f2a3b1630ceec165d3c3`。

Log: `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/a800_legacy_core75_checkpoint_oracle_20260812_v1.log`；node log 位于 output root 的 `logs/node.log`。

Metrics: test-best Overall R@1/R@10/R@50=`25.4794/69.3556/76.3333`，Single=`37.5000/54.1667/54.1667`，Multi=`19.8226/76.5033/86.7647`；条件为 step=`37,000`、Official core75=`75` queries、beam=`50`、beam-order、CE log-prob。

## Verification

- [x] Minimal local check：`py_compile`、Ruff、两个 launcher 的 `bash -n` 均通过
- [x] Unit/invariant test：本地 `19` tests passed、`2` tensor-only tests skipped；固定 CUDA runtime 的 pod preflight `13/13` passed
- [x] Reviewer required? yes；复核后 unresolved CRITICAL=`0`、HIGH=`0`
- [x] snapshot timing：eval 后、下一 update 前；terminal step=`38,805` 的 reason=`before_terminal_best_restore`
- [x] eval steps=`195`，精确覆盖 `{200,400,...,38,800,38,805}`，无缺失、重复或额外 step
- [x] step=`38,200` 命中历史 strict-base dev loss、model-state、prediction、Overall/Single/Multi metrics
- [x] test-best 排序独立重算一致：R@1、R@10、R@50、较早 step 依次排序，selected step=`37,000`
- [x] pod 使用 `1×8 A800`：GPU `0` 训练，GPU `1–7` 为七个 evaluator；pod restart=`0`
- [x] 独立 artifact 审计：`195` 个 predictions hash 与 `585` 个 subset metrics 一致，共 `780` checks；七个 worker 唯一覆盖全部 `195` steps；failure marker=`0`

## Result / Notes

实现交叉审查初次结果为 CRITICAL=`0`、HIGH=`1`。HIGH 指向历史 control artifact 未固定 hash，已增加两份 JSON 的 SHA-256 检查，并直接固定 step=`38,200`、dev loss=`0.7701343703584546`、model-state hash 与 prediction hash。复核结果为 unresolved CRITICAL=`0`、HIGH=`0`。

全轨迹 oracle 的 test-best 是 step=`37,000`：Overall R@1/R@10/R@50=`25.4794/69.3556/76.3333`，dev loss=`0.797698909`；条件为 Official core75=`75` queries、beam=`50`、beam-order、CE log-prob。dev-best 是 step=`38,200`：Overall=`20.9238/69.5556/75.3333`，dev loss=`0.770134370`；条件相同。按 test R@1 选模提升 R@1=`4.5556` 个点，R@10 降低=`0.2000` 个点，R@50 提升=`1.0000` 个点。

相对 RTX R@1=`28.3683`（同 Official core75 协议），test-best 仍低 `2.8889` 个点，未达到 within-one-point 阈值 R@1=`27.3683`。原 dev-best 到 RTX 的 R@1 gap=`7.4444` 个点，因此 test-oracle 解释并回收其中 `61.1940%`，其余 `38.8060%` 在这条固定 A800 训练轨迹上无法通过 checkpoint 选择消除。

全 `195` 点的 dev loss 与 R@1 Pearson=`-0.9002`、Spearman=`-0.8942`；条件为同一 strict-base 轨迹与每 `200` updates 的 core75 评测。全局相关性强，但局部最小 dev loss 没有选中最佳 R@1。Official core75 被用于选模，因此该 test-best 仅是诊断上限，不能作为无泄漏的可报告 test 结果。

step=`38,200` 的 model-state full hash=`bbec7389f56b36a24db000111adf8ce99e3bf712609cfcd1562c64c9910172f8`、prediction hash=`3663a15ad9859abc76141d815f0da3c88bd9b3d5d9e914f09a3d8b0a3ed843f4` 与历史 control exact-match。完整 loss-history SHA-256=`1a8931699e8c2fcc0d31dcd0216d81108693e356cde0e6125669e8bd4ad594fe` 也与历史 control exact-match，证明 snapshot/evaluator 链路没有改变训练轨迹。
