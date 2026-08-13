# Experiment: Legacy full-catalog generator scoring
Status: running
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

## Verification

- [x] Minimal local check：相关单测通过=`11`、因本地无 torch 跳过=`2`；Ruff、`py_compile`、launcher `bash -n`、`git diff --check` 通过
- [ ] Archived beam candidate score replay at absolute tolerance=`5e-5`
- [ ] Full sequence score equals five position contributions within absolute tolerance=`1e-6`
- [ ] All `8` shards are disjoint and cover Official core75=`75` queries
- [ ] Reviewer required? no; perform structured local semantic review

## Result / Notes

Pending.

本地输入闭合检查通过：Official rows=`75` 与归档 predictions=`75` 逐行对齐，prediction SHA-256 命中选中 checkpoint 在 step=`28,400` 的记录。结构化自审确认 score 仍由 `c0/c1/c2/c3/stop=1` 五项 full-vocabulary log-prob 相加；完整 catalog 只移除 beam frontier 截断，没有改变模型、SID 或归一化域。
