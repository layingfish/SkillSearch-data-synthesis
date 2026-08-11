# Experiment: CE-prefix paired query error overlap
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

比较 strict-base `control_a`、`seq_010` 和 `cum_002` 在同一 Official core75 上的逐 query 得失。目标是确认 sequence-only 的 Single Top-1 收益与 cumulative lambda=`0.02` 的 Multi/deep-recall 收益是否来自互补 query，并定位 R@10/R@50 回退对应的 gold rank 转移。

本实验只读取已完成 job=`tide-chenyufei03-10603-gigmv` 的 predictions，不训练模型。三路条件均为 base source queries=`94,912`、catalog updates=`2,730`、query updates=`38,805`、batch=`16`、seed=`20260626`、BF16、workspace=`:16:8`、dev-best checkpoint、Official core75=`75` queries、beam=`50`、beam-order。

## Plan

1. 强制校验三路 `75` 条 row 的 index、source index、query、gold 和 target count 对齐。
2. 独立重算 Overall/Single/Multi R@1/R@10/R@50，并与已有 result summary 做 exact-tolerance 对照。
3. 统计三路 Top-1 hit pattern、相对 control 的 gained/lost、逐 cutoff macro-recall delta 和候选集合重叠。
4. 统计每个 gold 在 Top-1/Top-10/Top-50 的进入、退出与降档；Top-50 之外只标记 omission，不推断搜索或 scoring 原因。
5. 报告 per-query oracle 上界，以及使用 gold cardinality 选择 Single→sequence、Multi→cumulative 的非部署诊断上界。
6. 增加合成测试，验证 pattern、oracle、subset 汇总与守恒。

## Commands

Analyze: `uv run python scripts/analyze_legacy_ce_prefix_error_overlap.py`，输入为 `control_a`、`seq_010`、`cum_002` 的 predictions/result summary，固定 expected rows=`75`、rank depth=`50`。

## Outputs

Summary: `/workspace/SkillSearch-data-synthesis/outputs/legacy_ce_prefix_error_overlap_20260812_v1/ce_prefix_error_overlap_summary.json`，SHA-256=`4460df1c571a04b94280cd0654ff74e0e4e9b538f3c80c06934ed31f3ade2e3d`

Per-query: `/workspace/SkillSearch-data-synthesis/outputs/legacy_ce_prefix_error_overlap_20260812_v1/per_query.jsonl`，SHA-256=`ed00c1ec5a4bb7c30fc289aab8674048263635e521ec9888ee51fd05ac2a03fe`

Report: `/workspace/SkillSearch-data-synthesis/outputs/legacy_ce_prefix_error_overlap_20260812_v1/ce_prefix_error_overlap.md`，SHA-256=`7f66c47ce75db0f2fd437af7cba06633faadac7801a0a866b361b9b8c4cbdb8f`

## Verification

- [x] 三路 row/gold alignment exact
- [x] 独立重算指标与已有 result summary 对齐，atol=`1e-12`
- [x] Top-1 pattern 总数对 Overall/Single/Multi 分别守恒
- [x] gained/lost 与 macro-recall delta 守恒
- [x] gold cutoff transition 与每路 R@10/R@50 差值守恒
- [x] oracle 不低于每个 constituent arm
- [x] 合成测试通过：新旧 analyzer 合计 `5` 项测试全部通过
- [x] Reviewer required? no

## Result / Notes

Top-1 hit pattern 的 bit 顺序为 `control/sequence/cumulative`。Single 子集没有 `001`、`101` 或 `100`：sequence 的 `12` 条 top-1 hit 是 control 的 `6` 条和 cumulative 的 `8` 条的超集。Multi 子集没有 `010`、`110` 或 `100`：cumulative 的 `37` 条 top-1 hit 是 control 的 `31` 条和 sequence 的 `30` 条的超集。具体互补项为：

- sequence-only 正确、另外两路错误：indices=`[3,8,35]`，全部是 Single。
- cumulative-only 正确、另外两路错误：indices=`[40,42,43,67]`，全部是 Multi。
- sequence/cumulative 都正确、control 错误：indices=`[0,34,39,45,66]`，其中 Single=`3`、Multi=`2`。
- control/cumulative 正确、sequence 错误：indices=`[29,55,58]`，全部是 Multi。
- control/sequence 正确、cumulative 错误：index=`[33]`，是 Single。

因此按 gold cardinality 诊断性地选择 Single→sequence、Multi→cumulative，可得到 Overall R@1/R@10/R@50=`31.7016/72.6222/75.7778%`（Official core75=`75` queries、beam=`50`、beam-order）。其 R@1 与逐 query oracle=`31.7016%` 完全相同；逐 query oracle 的 R@10/R@50=`75.5556/80.0000%`。该 router 使用 gold target count，不能作为推理策略。

这个结果可以转化为可部署训练目标：训练时 manifest 已包含同 query 的 `gold_skill_ids`，可对 Single rows 使用 sequence loss lambda=`0.10`，对 Multi rows 使用 cumulative loss lambda=`0.02`；推理时仍使用同一个模型和原 beam decoder。分 checkpoint 的互补性不保证条件 loss 会线性合并，因此下一轮只把它作为需要实证的主候选。

深召回仍需保护。Multi cumulative R@50=`85.9477%`（Multi=`51` queries、beam=`50`、beam-order），比 concurrent control=`87.2549%` 低 `1.3072` 个百分点，超过允许回退=`1.0` 个百分点约 `0.3072` 个点。下一轮应保留 exact CE，并把该 non-inferiority gate 继续设为硬门槛。

首次尝试写入 GPU job 的 `/models_rw/.../analysis/` 时，父目录属于 `root:root` 且 mode=`755`，两次都在创建输出目录前失败，没有部分 artifact。最终输出写入仓库已忽略的 `outputs/` 目录；冻结输入仍直接来自 CFS job artifacts。
