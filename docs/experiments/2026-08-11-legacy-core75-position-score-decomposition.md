# Experiment: Legacy core75 position-score decomposition
Status: done
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

- `position_score_summary.json` SHA-256=`4816f36bdb4fd144c5bed96b7765c6397b2099ae9dca417d3b5ef9e18eb63a64`
- `position_score_per_query.jsonl` SHA-256=`0f9b511dee4af2f7a8323cbce6ca4245849752e49dc763a71e5cdafd2d4506e3`（Official core75=`75` rows）
- `position_weight_grid.jsonl` SHA-256=`41c49c0c72c277daa356aed2631cd636283ce4ceb2596f01c8ef779535d92d89`（三个 A800 checkpoint × `3,061` weights=`9,183` rows）

## Verification

- [x] 四个 checkpoint × 两个 shards 使用一个 `8×A800` 节点
- [x] 每个 candidate 有五项 finite contributions，sum 与 total score allclose
- [x] uniform weights 重建四路 fixed-pool ranking 与 R@1/R@5/R@10/R@50
- [x] A800 archived Top50 score replay 和 Top-1/5/10/50 集合仍通过
- [x] grid count=`3,061`，五折 split 各为 train=`60`、heldout=`15`
- [x] weight selection、metric 重算与 query transition 守恒通过
- [x] Reviewer required? yes；独立复核无 CRITICAL/HIGH/MEDIUM
- [x] 冻结本地 runtime 单测通过：fixed-pool=`9/9`、position-score=`6/6`（Python=`3.10.12`、torch=`2.10.0+cu128`、NumPy 可用的冻结环境）
- [x] 旧 v1 fixed-pool 产物向后兼容：逐 query JSONL SHA-256 仍为 `2379a329f7832a39531427cba81c63bc63fa4ce1e67b3e8bb889ea8148c0ed8c`；summary 只新增 model-input alignment 验证字段

## Result / Notes

Job=`tide-chenyufei03-10602-lbift` 在 queue=`rudder-exper` 使用一个 `8×A800` 节点完成，pod runtime=`6m6s`、restart=`0`。四路各有 `10,708` 个 fixed-pool candidates；五项 contribution 的最大 sum error=`0`（FP64 analyzer），uniform ranking exact。三个 A800 checkpoint 的 archived Top50 score replay 最大绝对误差为 control=`2.455711e-05`、warmup=`1.239777e-05`、clip=`1.335144e-05`（Official core75=`75` queries、atol=`5e-5`），Top-1/5/10/50 集合全部保持。

下表条件统一为 Official core75=`75` queries、pool=four beam-order Top50 union plus all gold、五项分数为 code position=`0/1/2/3` 加 stop control=`1`、`3,061` 组 normalized weights。`CV` 是按 `row_position % 5` 的五折 heldout 汇总。

| system | uniform R@1/5/10/50 | full-set selected R@1/5/10/50 | CV R@1/5/10/50 | full gate |
|---|---:|---:|---:|---:|
| RTX archived | `28.3683/61.5968/69.0000/78.0000` | `28.7016/61.0190/69.7778/78.1111` | `25.7016/60.9746/68.0000/78.1111` | pass |
| workspace control | `20.9238/59.5302/70.8889/78.3333` | `21.5905/59.4190/69.8889/78.3333` | `20.9238/61.4190/69.1111/78.3333` | fail |
| workspace warmup=`500` | `23.9905/58.9302/64.8444/76.6667` | `24.7683/58.3746/66.4667/78.3333` | `23.8794/59.3746/64.9556/78.5556` | fail |
| workspace clip=`1.0` | `23.8794/58.3302/65.1778/75.6667` | `24.5460/58.2190/64.4667/74.7778` | `21.2127/57.7746/67.2222/74.3333` | fail |

三个 A800 grid 的 full-gate 通过数均为 `0/3,061`（R@1 `>=27.3683`、R@10 `>=69.0000`、R@50 `>=75.0000`，上述 fixed-pool 条件）。control 有 `668/3,061` 组通过 deep gate，其中最高 R@1=`21.5905`；warmup 与 clip 的 deep-gate 通过数均为 `0/3,061`。忽略 deep gate 后，control/warmup/clip 的 grid 最高 R@1=`22.9238/24.7683/24.5460`，仍低于 R@1 门槛=`27.3683`。

full-set weighting 只恢复 control index=`66`、warmup index=`48/55`、clip index=`54` 的 Top-1，A800 三路均无 Top-1 loss（Official core75=`75` queries、各自 full-set selected weights）。五折后 control R@1 回到 uniform=`20.9238`，warmup 从 uniform=`23.9905` 降到 `23.8794`，clip 从 uniform=`23.8794` 降到 `21.2127`；full-set 小增益没有跨 fold 稳定性。

RTX/A800 的 per-query candidate rank Spearman 在 code position=`0/1/2/3` 上分别为 control=`0.5021/0.4460/0.4613/0.4190`、warmup=`0.5421/0.4615/0.4690/0.4512`、clip=`0.5406/0.4639/0.4910/0.4439`（每项为 Official core75=`75` queries 的均值、同一 fixed pool）。stop control 的对应均值只有 `0.0082/0.0124/0.0388`；其 gold-vs-best-nongold margin 在 RTX-correct/A800-wrong cases 中近似为 `0`，对最终排序几乎没有贡献。

RTX-correct/A800-wrong cases 的 A800 mean gold-minus-best-nongold component margin 为：control=`[-1.6596,-1.2475,-0.0798,-0.4477,≈0]`（`9` queries），warmup=`[-0.8922,-0.9212,-0.2740,-0.5455,≈0]`（`9` queries），clip=`[-2.0248,-1.1925,-0.7115,-0.6658,≈0]`（`8` queries）；顺序为 `[c0,c1,c2,c3,stop]`，条件为各自 uniform checkpoint、同一 fixed pool。前两个 code positions 贡献了最大的平均负 margin，但最优权重方向在 checkpoint 与 fold 间不一致。

结论：固定的全局 position scaling 无法恢复 A800 R@1，也没有可进入 full-catalog beam=`50` evaluator 的候选。后续 prefix/component attribution 已完成：c1 占 control/warmup=`500`/clip=`1.0` endpoint R@1 gap 的 `45.9%/52.9%/48.5%`（Official core75=`75` queries、同一 fixed pool、全部 `32` 个 component-swap masks 的 exact Shapley），但只有同时替换 c0/c1/c2/c3 才能通过 R@1/R@10/R@50=`27.3683/69.0000/75.0000` 门槛。stop 的 Shapley contribution=`0` 。证据支持在 c1 分叉处引入 query-conditioned hard negatives，同时用 cumulative-prefix 或 sequence-level ranking loss 保护深召回；当前不再 sweep position weights。
