# Experiment: Legacy core75 fixed-pool scoring
Status: running
Risk Tier: 2

## Goal / Hypothesis / Change Scope

把 archived RTX、A800 `workspace_16` control、query warmup=`500`、query global L2 clip=`1.0` 的逐 query beam-order Top50 取并集，并强制加入该 query 的全部 gold。四个 checkpoint 在完全相同的候选池上穷举 CE log-prob，区分 beam pruning/search omission 与 checkpoint scoring failure。

每条 candidate 固定为 learned code depth=`4` 加 stop control=`1`。score 为四个 position-specific code head 与 control head 的 full-vocabulary log-softmax 之和；不按 trie 合法 children 重新归一化。逐 query trie 只包含 union pool，beam capacity 等于 pool size，max skills=`1`，因此所有候选都必须被穷举且只出现一次。

一个节点运行四个 checkpoint × 两个 query shard，共八个独立单卡进程。每个 shard 处理 Official core75 中 `row_position % 2 == shard_id` 的 query；训练参数不改动。

## Commands

Score: job=`tide-chenyufei03-10601-weesu`，queue=`rudder-exper`，nodes=`1`，节点规格=`8×A800`，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`349505d88e246b6e45430189c59bb4858ea70386`。节点内运行四个 checkpoint × 两个 Official core75 row-position shards，共八个单卡 scorer。

Analyze: 八个 shard 全部成功后，同一作业内合并每个 checkpoint 的两个 shards，并输出 fixed-pool summary、逐 query 结果和验证状态。

## Outputs

Output: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_core75_fixed_pool_scoring_20260811_v1/`

Checkpoints:

- archived RTX final file SHA-256=`b92daf62e604f35dbecbb5c80897de9d212995d13c312f0f149f781caf5c0848`
- workspace control final file SHA-256=`2362f0567854c0134f748dd44de3d284224057a7c93751c31b66594ae2662b87`
- workspace warmup=`500` final file SHA-256=`b76389494d94c966d53f2e9701993ba295da32034f8bd23dcea46ba95109e74b`
- workspace clip=`1.0` final file SHA-256=`22728b8d401d0e56843f85faab98416b7f530c55d718585d0f2bafb75a7240c5`

## Verification

- [x] 四路输入 prediction 均为 `75` rows，query/gold/index exact 对齐，Top50 长度=`50` 且去重
- [x] 本地 pool 构造检查通过：每条 pool 等于四路 Top50 union 加全部 gold；pool size min/mean/max=`98/142.77/181`（Official core75=`75` queries）
- [ ] exhaustive scorer 对每条 query 返回 pool size 个唯一 candidate，gold coverage=`100%`
- [ ] A800 checkpoint 在自身原 Top50 上的重算 score 与归档 score 最大绝对误差 `<=5e-5`，且重排后的 Top-1/5/10/50 集合不变
- [ ] 四个 checkpoint × 两个 shards 合并为各 `75` 条，shard 无交叠、无缺失
- [x] scorer 的 code/control log-softmax 与逐 step constrained replay 在 fake model 上 allclose；冻结 Python=`3.10.12`、torch=`2.10.0+cu128` CPU runtime 的七项 focused tests 全部通过
- [x] analyzer 独立重算 fixed-pool R@1/R@5/R@10/R@50；逐 gold mention 统计必须覆盖 Official core75 的 `200` 个 gold mentions，并按 `1/target_skill_count` 反算 macro recall
- [x] 独立审查完成；stop-token、full-vocabulary normalization、pool exhaustive、shard provenance、Top-K replay 与 multi-GT 守恒问题均已检查并修正

## Result / Notes

Pending.

- `tide-chenyufei03-10601-weesu` 已提交；条件为 `rudder-exper`、nodes=`1`、节点规格=`8×A800`、Official core75=`75` queries、pool=four beam-order Top50 union plus all gold。

- 四个 checkpoint 的 metadata 均为 max source length=`2,048`、source text policy=`skillrouter_query_1500_chars_with_instruction_v1`、codebook size=`256`、depth=`4`、dedup size=`0`；条件为直接读取各自 `structured_decoder.pt`。
- 本地 source-only runtime 通过六项并跳过一项 torch fake；冻结 Python=`3.10.12`、torch=`2.10.0+cu128` CPU runtime 通过全部七项 focused tests。
- 本地 `py_compile`、两个 launcher 的 `bash -n` 与 `git diff --check` 通过；本机没有 `shellcheck`。
