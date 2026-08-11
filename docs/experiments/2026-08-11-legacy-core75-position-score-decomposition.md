# Experiment: Legacy core75 position-score decomposition
Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

在已验证的 fixed pool 上保存每个 candidate 的五项独立 log-prob：code position=`0/1/2/3` 与 stop control=`1`。先比较 RTX 与三个 A800 checkpoint 在每个 position 上的候选排序，再测试受约束的 position weighting 能否恢复 A800 Top-1。

候选池、tokenizer、checkpoint、source text 和 forward 全部沿用 fixed-pool job=`tide-chenyufei03-10601-weesu`。每个 candidate 的 total score 必须等于五项 contribution 之和；uniform weights=`[1,1,1,1,1]` 必须精确重建现有 fixed-pool 排名和 R@1/R@5/R@10/R@50。

本轮另外冻结 model-input provenance。四个 checkpoint 的 `tokenizer.json` SHA-256 均为 `65430662114922dc57697bc782fc1122cd004547611abfcd520fe1a50f125bd9`；RTX 与 A800 的 `tokenizer_config.json` 仅允许已取证的 `is_local` 差异。每个 shard 还必须记录去掉 `is_local` 后的 semantic config hash，以及 Official core75=`75` queries 在 max source length=`2048` 条件下的实际 `input_ids`/`attention_mask` 全量 hash。base model 的 `config.json` 与 `model.safetensors` 也写入 manifest 并跨四路对齐。

权重网格由五个 raw weights 各自在 `{0.5,0.75,1.0,1.25,1.5}` 中取值，再除以五项均值，并按八位小数去重，共 `3,061` 组 normalized weights。每个 A800 checkpoint 独立选择：先要求 fixed-pool R@10 `>=69.0000`、R@50 `>=75.0000`，再依次最大化 R@1、R@10、R@50，最后最小化相对 uniform 的 L2 距离。完整恢复门槛为 R@1 `>=27.3683`、R@10 `>=69.0000`、R@50 `>=75.0000`；条件均为 Official core75=`75` queries 与本实验 fixed pool。

另外按 `row_position % 5` 做五折交叉验证。每折在 `60` 条 query 上按相同规则选择，在其余 `15` 条 query 上评测；只作为权重稳健性诊断。任何 fixed-pool 候选都必须再通过原始 full-catalog beam=`50` evaluator，才能成为 baseline 配置。

## Commands

Score: job=`tide-chenyufei03-10602-lbift`，queue=`rudder-exper`，nodes=`1`，节点规格=`8×A800`，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`0ef8258cca6f7fa0d2c6583b4fc98f5e17ee342f`。节点内运行四个 checkpoint × 两个 Official core75 row-position shards，共八个单卡 scorer。

Analyze: 节点内八个 scorer 成功后运行 `scripts/analyze_legacy_core75_position_scores.py`，输出 position summary、逐 query contribution 和完整 normalized weight grid。

## Outputs

Output: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_core75_position_score_decomposition_20260811_v1/`

## Verification

- [ ] 四个 checkpoint × 两个 shards 使用一个 `8×A800` 节点
- [ ] 每个 candidate 有五项 finite contributions，sum 与 total score allclose
- [ ] uniform weights 重建四路 fixed-pool ranking 与 R@1/R@5/R@10/R@50
- [ ] A800 archived Top50 score replay 和 Top-1/5/10/50 集合仍通过
- [ ] grid count=`3,061`，五折 split 各为 train=`60`、heldout=`15`
- [ ] weight selection、metric 重算与 query transition 守恒通过
- [ ] Reviewer required? yes；检查 contribution 与 decoder step 对齐、uniform replay、multi-GT macro recall、selection tie-break 和 full-catalog 限制
- [x] 冻结本地 runtime 单测通过：fixed-pool=`9/9`、position-score=`6/6`（Python=`3.10.12`、torch=`2.10.0+cu128`、NumPy 可用的冻结环境）
- [x] 旧 v1 fixed-pool 产物向后兼容：逐 query JSONL SHA-256 仍为 `2379a329f7832a39531427cba81c63bc63fa4ce1e67b3e8bb889ea8148c0ed8c`；summary 只新增 model-input alignment 验证字段

## Result / Notes

Job 已提交，等待 worker 创建。
