# Experiment: Legacy core75 query-rank forensics
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

对齐 archived RTX、A800 `workspace_16` control、query warmup=`500` 与 query clip norm=`1.0` 的 Official core75 预测，定位 R@1 缺口来自 gold 的轻微错序还是候选召回丢失。

每个系统固定 Official core75=`75` queries、beam=`50`、beam-order、K=`{1,5,10,50}`。逐 query 记录最近 gold rank、macro recall contribution、RTX/A800 top-1 flip、Top-10/Top-50 candidate overlap、RTX top-1 在 A800 中的 rank，以及 A800 的 best-gold 对 best-nongold CE log-prob margin。

RTX 归档 prediction 没有 `ranked_skill_scores`。因此 RTX 只参与 rank、flip 和 candidate overlap；score margin 仅对三个 A800 checkpoint 计算。

## Commands

Analyze:

```bash
/home/claude/.local/bin/uv run python scripts/analyze_legacy_core75_query_ranks.py \
  --system rtx_archived=<archived RTX predictions.jsonl> \
  --system workspace_control=<workspace control predictions.jsonl> \
  --system workspace_warmup500=<workspace warmup predictions.jsonl> \
  --system workspace_clip1=<workspace clip predictions.jsonl> \
  --metrics rtx_archived=<archived ranked_metrics.json> \
  --metrics workspace_control=<workspace control ranked_metrics_all.json> \
  --metrics workspace_warmup500=<workspace warmup ranked_metrics_all.json> \
  --metrics workspace_clip1=<workspace clip ranked_metrics_all.json> \
  --reference-label rtx_archived --expected-rows 75 --expected-depth 50 \
  --output-dir /models_rw/SkillSearch-data-synthesis/experiments/legacy_core75_query_rank_forensics_20260811_v1
```

## Outputs

Output: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_core75_query_rank_forensics_20260811_v1/`

Inputs:

- RTX archived: `/models_rw/SkillSearch-data-synthesis/assets-models/outputs/eval/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714/core75_beam50/predictions.jsonl`
- workspace control/warmup/clip: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_query_stability_20260811_v1/arms/*/eval/official_core75/predictions.jsonl`

## Verification

- [x] 四个系统均为 `75` rows，index/source_index/query/gold exact 对齐
- [x] 每条 ranked list 长度=`50` 且 skill 去重
- [x] A800 score keys 覆盖 Top-50，ranked score 单调不增
- [x] 从 prediction 独立重算 R@1/R@10/R@50 并命中现有汇总
- [x] rank bucket、flip count 与 per-query contribution 守恒
- [x] 三项 focused unit tests 通过：Multi-GT macro recall、flip/margin 守恒、坏 score 与错位 query 拒绝
- [x] Reviewer required? yes；structured semantic review 已核对 beam completion score、Multi-GT contribution、rank bucket、margin 方向和 pairwise overlap，未发现 CRITICAL/HIGH 问题

## Result / Notes

- 输出：`query_rank_forensics.json`、`query_rank_forensics.md`、`per_query.jsonl`。所有文件位于上面的 output 目录。
- archived RTX、workspace control、warmup、clip 的 Overall R@1/R@10/R@50 分别为 `28.3683/69.0000/75.0000`、`20.9238/69.5556/75.3333`、`23.9905/64.8444/73.0000`、`23.8794/65.1778/73.3333`；条件均为 Official core75=`75` queries、beam=`50`、beam-order、macro recall。
- workspace control 相对 RTX 有 `9` 条 RTX top-1 正确 query 变错、`1` 条 RTX top-1 错误 query 变对；warmup 为 `9/4`，clip 为 `8/3`。条件为同一 `75` 条逐 query 配对。
- RTX 与 workspace control/warmup/clip 的平均 Top10 共享率=`34.00%/36.27%/36.27%`，平均 Top50 共享率=`24.00%/26.35%/27.01%`；条件为每条 query 对两个长度固定为 `10/50` 的候选集合求交集比例，再对 `75` 条 query 取均值。
- workspace control 与 warmup、control 与 clip、warmup 与 clip 的平均 Top50 共享率=`30.64%/30.53%/33.23%`；条件同上。训练干预后的候选轨迹仍大幅改写。
- workspace control 丢失的 `9` 条 RTX-correct query 中，最近 gold 位于 rank `2–5/6–10/11–50/Top50 外` 的数量为 `5/2/0/2`；warmup 为 `5/0/2/2`；clip 为 `3/1/2/2`。条件为每个系统自己的 beam-order Top50。
- index=`{8,22,35,54,66}` 在三个 A800 设置上都从 RTX top-1 正确变为错误。index=`8` 的 gold 在三个 A800 Top50 都缺失；index=`54/66` 的 gold 在三个设置中都存在但被错误候选压过。后两条提供稳定的 scoring failure 证据；index=`8` 提供稳定的 candidate-generation failure 证据。
- 下一步使用四个系统 Top50 与全部 gold 的逐 query union pool。四个 checkpoint 在同一候选集合上穷举 CE log-prob。该实验区分 beam pruning/search omission 与 checkpoint 本身的 scoring 差异。
