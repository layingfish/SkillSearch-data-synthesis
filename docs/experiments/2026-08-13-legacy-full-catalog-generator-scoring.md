# Experiment: Legacy full-catalog generator scoring
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

对 A800 基准 `cum_002@28,400` 做纯生成式完整 catalog 打分。每条 Official core75 query 都用同一生成模型的 CE log-prob 精确评分全部 LETTER SID，不引入外部候选或外部 score。

诊断区分两类 Top-1 损失：完整 catalog 的最优 SID 已是 gold、但 beam=`50` 没保住它，归入 beam pruning；完整 catalog 的最优 SID 仍非 gold，归入生成器排序。同步记录每个 gold 在 `c0/c1/c2/c3` 累计 prefix 上的全 catalog rank。

运行条件固定为 Official core75=`75` queries、checkpoint=`cum_002@28,400`、checkpoint SHA-256=`a2a26ab5f64fbbe5b0df875afccad945684a2b5e2166f71b94a3e299c36f03c1`、decoder head=`CE`、score=`full-vocabulary log-prob`、max source length=`2048`、beam baseline=`50`、rank aggregation=`beam_order`。完整 catalog scorer 使用一个 `8×A800` 节点，query 按 row position 分成 `8` 个 shard；前 `50` 个 frontier prefix 保持原 beam batch，剩余 prefix 使用 forward batch=`256`。

## Commands

Train: N/A

Eval: job=`tide-chenyufei03-10679-cvgac`，queue=`rudder-exper`，node=`1×8 A800`，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`7994dc5af377c38b1ac8698a3c0185516cbfa5ee`。节点入口为 `bash scripts/run_legacy_core75_full_catalog_node_20260813.sh`。复用同一 checkpoint 已保存且 SHA-256=`78386bffac0944fed624db2d9948023cc6407fedea32ec988df6d99f66a29a7b` 的 Official core75 beam=`50` predictions；随后运行 `scripts/score_legacy_core75_full_catalog.py score` 的 `8` 个 GPU shard，再运行 `analyze` 合并。

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_sequence_weight_checkpoint_sweep_20260812_v5/selected/joint_best/structured_decoder.pt`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_core75_full_catalog_generator_scoring_20260813_v1/logs/node.log`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_core75_full_catalog_generator_scoring_20260813_v1/full_catalog_summary.json`

- `full_catalog_summary.json` SHA-256=`754ef28e059f648f37fb426dc09d248583a9eef2f9e4a373aed7c766666c91eb`
- `full_catalog_per_query.jsonl` SHA-256=`56f2928fc8caa7d208a70c1a7537e873bb403a1a96b2cba893874cc0fefa709f`
- `full_catalog_report.md` SHA-256=`af685a8c54b7754f06798daa69663827aac364028dc85c4bb30a2fe18704551a`

## Verification

- [x] Minimal local check：相关单测通过=`11`、因本地无 torch 跳过=`2`；Ruff、`py_compile`、launcher `bash -n`、`git diff --check` 通过
- [x] Archived beam candidate score replay：最大绝对误差=`1.9073486328125e-05`（允许误差=`5e-5`），Top-1/5/10/50 集合全部保持
- [x] Full sequence score equals five position contributions within absolute tolerance=`1e-6`
- [x] All `8` shards are disjoint and cover Official core75=`75` queries
- [ ] Reviewer required? no; perform structured local semantic review

## Result / Notes

本地输入闭合检查通过：Official rows=`75` 与归档 predictions=`75` 逐行对齐，prediction SHA-256 命中选中 checkpoint 在 step=`28,400` 的记录。结构化自审确认 score 仍由 `c0/c1/c2/c3/stop=1` 五项 full-vocabulary log-prob 相加；完整 catalog 只移除 beam frontier 截断，没有改变模型、SID 或归一化域。

Job=`tide-chenyufei03-10679-cvgac` 正常完成，pod status=`Completed`、restart=`0`、wall time 约=`8m18s`（queue=`rudder-exper`、`1×8 A800`）。下表条件统一为 checkpoint=`cum_002@28,400`、Official core75=`75` queries、catalog=`14,551` frozen LETTER SIDs、CE full-vocabulary log-prob；beam 列使用 beam=`50`、beam-order，exhaustive 列完整展开全部 SID。

| group | exhaustive R@1/R@10/R@50 | beam R@1/R@10/R@50 |
|---|---:|---:|
| Overall=`75` | `28.2571/70.0000/76.6667%` | `28.2571/70.0000/75.1111%` |
| Single=`24` | `41.6667/50.0000/54.1667%` | `41.6667/50.0000/58.3333%` |
| Multi=`51` | `21.9468/79.4118/87.2549%` | `21.9468/79.4118/83.0065%` |

Top-1 结论：beam 与 exhaustive 的命中 query 完全相同。Overall 的 `29/75` 个 Top-1 miss、Single 的 `14/24` 个 Top-1 miss、Multi 的 `15/51` 个 Top-1 miss，在完整 catalog 排名下仍全部 miss；扩大 beam 无法恢复任何 Top-1。

Single 的 exhaustive gold rank 分布为 rank=`1` 共 `10/24`、rank=`2–10` 共 `2/24`、rank=`11–50` 共 `1/24`、rank=`>50` 共 `11/24`。beam=`50` 额外把 full-catalog rank=`140` 的 `gt/nlp-research-repo-package-installment` 提升到 beam rank=`44`，所以 Single exhaustive R@50 比 beam 低 `1/24=4.1667` 个百分点；这是搜索剪掉更高分 nongold 后的偶然提升。

一个明确的数据问题是 index=`12`：gold=`gt/timeseries-detrending` 的 exhaustive rank=`5`，Top-1=`other/timeseries-detrending`。两条 catalog body 去掉 `gt` 版本 YAML frontmatter 后完全相同，规范化文本 SHA-256 均为 `03274863ba9d516da03340c4e1fed9d61d7790c64a3589daa21b9093c0f5c97e`；当前 canonicalization 没把它们合成 alias。即使把这一条按语义正确计入，Single R@1 也只有 `11/24=45.8333%`（同一 exhaustive 条件）。

后续纯 GR 优先级因此确定：停止扩大 beam；用当前生成器的完整 catalog 高分 nongold 做 listwise hard-negative training，并继续使用 mode3 query augmentation 处理 Single OOD。推理仍只运行一个生成模型。
