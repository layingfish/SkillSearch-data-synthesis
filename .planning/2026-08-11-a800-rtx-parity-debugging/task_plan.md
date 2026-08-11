# A800 / RTX PRO 6000 训练差异排障计划

## 目标

在 RTX PRO 6000 Blackwell 不可用的条件下，定位 A800 上 legacy BF16 训练的主要数值敏感模块，并锁定一个强、稳定、可供后续 loss、catalog、RIPOR、PAG 实验共同使用的 A800 基线。

“A800 基线”指专门在当前 A800 Ampere + driver `535.161.07` 环境上验证并冻结的训练配方。它不要求逐 step 重现 Blackwell 轨迹，但必须恢复接近 archived RTX checkpoint 的检索能力，并通过多 seed 稳定性检查。

## 当前状态

- 状态：`in_progress`；Phase 2/3 已证明局部 FP32 无法让参数轨迹收敛。
- 当前阶段：Phase 5；optimizer-boundary 短 probe 已完成，正在进入完整 query 限定 sweep。
- 风险：训练与 checkpoint 语义属于 Tier 2；局部 autocast wrapper 必须通过关闭开关时的端到端等价检查。

## 固定条件

所有诊断默认继承 archived baseline：

- 原始训练脚本：`repro/20260714_letter_base_single_sid/skill_gr_pipeline.original_20260714.py`，SHA-256=`7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76`。
- T5-base revision：`a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`。
- 数据：94,912 source rows，展开后 124,175 single-SID rows；Semantic ID logical SHA-256=`da2fff997b1979f7d51996c5ddbe4593a18d23bdfd546c61dd61fb783567349c`。
- 单卡训练：batch=16、seed=20260626、AdamW lr=3e-4、weight decay=0、betas=(0.9,0.999)、eps=1e-8、foreach=None。
- 完整日程：catalog=3 epochs/2,730 updates，query=5 epochs/38,805 updates；无 scheduler、warmup、gradient clipping；optimizer 默认跨阶段继承。
- 最终评测：Official core75 75-query、beam=50、beam-order aggregation、K={1,10,50}。
- 任何 arm 都必须记录源码、数据、模型、runtime、首批 row、初始化、RNG 和配置 hash；不匹配时停止。

## 已停止的方向

以下全局变量已有充分结果，不再重复 sweep：TF32、cuDNN TF32、BF16 reduced-precision reduction、cuBLAS/cuBLASLt、`:4096:8`、`:16:8`、deterministic algorithms、CUDA module loading、AdamW foreach、Python ABI、Torch/Transformers 版本和 dropout。

`workspace_16` 保留为数值路径对照与临时强基线候选。它不是跨架构模拟开关。

## Phase 0：证据冻结

- [x] 锁定 archived RTX step-1/50 loss、训练 manifest、checkpoint 与 Official core75 指标。
- [x] 锁定 A800 default、`workspace_16`、cuBLASLt、FP32 和 dropout=0 对照。
- [x] 确认 A800 同配置 replica byte-identical，archived RTX checkpoint 在 A800 上推理一致。
- [x] 记录网络调研与官方 reproducibility 边界到 `findings.md`。
- **Status：done**

## Phase 1：建立不改变训练语义的逐层数值 trace

目标：找到差异最早出现的位置，以及差异在哪些 block 明显放大。

1. 新增独立 wrapper，不改 archived 脚本默认路径。记录以下边界的实际输入/输出 dtype、有限值比例、L2 norm、max/mean absolute value、固定坐标样本和 SHA-256：
   - shared/token embedding；
   - encoder 12 个 block；
   - decoder 12 个 block；
   - SID embedding；
   - 4 个 code heads、control head；
   - CE loss；
   - backward 后每个参数张量的 gradient norm/sample/hash；
   - AdamW step 后每个参数张量的 update norm/sample/hash。
2. 在 `rudder-exper` 的一个 `8×A800` 节点上跑八个单卡 one-step arm。四种数值路径各包含 `trace=on/off` 配对，条件均为 catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0：
   - A800 default BF16；
   - A800 `CUBLAS_WORKSPACE_CONFIG=:16:8` BF16；
   - A800 cuBLASLt BF16；
   - A800 FP32 reference。
3. instrumentation 门禁：同一 backend 下，带 trace 与无 trace 的 total loss 和 optimizer 后参数必须相同；若 trace 改变结果，缩小采样或改为独立 forward/backward replay。
4. 输出逐层相对 L2 error 与 error growth ratio。首个非零边界只用于定位；候选模块按“误差增长”和“梯度/update 分叉”共同排序。

产物计划：

- 实验记录：`docs/experiments/2026-08-11-legacy-a800-layerwise-numerical-trace.md`
- 输出：`/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_layerwise_trace_20260811_v1`

停止条件：得到可重复的逐层差异报告；如果 wrapper 会改变原始 step-1 结果且两种非侵入方案仍失败，不进入局部精度训练，先收缩到纯 forward probe。

- **Status：done**

## Phase 2：局部 FP32 高层二分

目标：判断数值敏感性主要来自 encoder、decoder 或输出/loss 路径。

第一轮在一个 8×A800 节点上跑 8 个单卡 arm。每个 arm 固定 catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0：

| 局部 FP32 区域 | A800 default | A800 `workspace_16` |
|---|---:|---:|
| 无，BF16 control | 1 arm | 1 arm |
| `backbone.encoder` | 1 arm | 1 arm |
| `backbone.decoder` | 1 arm | 1 arm |
| SID/code/control heads + CE | 1 arm | 1 arm |

实现规则：

- 只在指定模块内部关闭 autocast，并把浮点输入转换为 FP32；不手动回写输出 dtype，后续算子继续由外层 autocast 决定有效精度。
- 先记录有效 dtype；已经由 autocast 自动使用 FP32 的算子不重复测试，其 GPU slot 改测 Phase 1 中误差增长最大的半组 block。
- 强制 invariant：`precision_gate=off` 时，wrapper 输出必须与未包装 base-model output allclose；50-step loss、checkpoint tensor 和参数更新还必须与对应 control 一致。
- 记录峰值显存。发生 OOM 时缩小 FP32 区域，不通过改变 batch 或 gradient accumulation 掩盖问题。

定义参数轨迹差异：

```text
D_param = ||theta_default - theta_workspace||_2 / ||theta_default - theta_init||_2
```

在 CPU FP64 上按完整可训练参数计算。某区域同时满足以下条件时进入细分：

- step-50 `D_param` 比无局部 FP32 control 降低至少 50%；
- step-50 loss gap 比 control 降低至少 50%；
- 无 NaN/Inf，manifest/hash 全部一致。

如果多个区域通过，优先选择 FP32 参数量、峰值显存和运行开销更小的区域。

- **Status：done**

## Phase 3：敏感模块细分

只围绕 Phase 2 排名最高的 decoder 路径做最多两轮，避免无边界搜索。Phase 2 中 decoder FP32 的 `D_param` reduction=`14.6651%`、loss-gap reduction=`52.8662%`；heads+CE FP32 的对应值为 `4.9188%/95.8414%`。条件均为 A800、default 对 workspace16、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0。

第一轮在一个 `8×A800` 节点运行四组 default/workspace16 pair，每个 arm 沿用同一 50-step、dropout=0 protocol：

| 局部 FP32 区域 | 目的 |
|---|---|
| decoder blocks 0–5 | 验证首个分叉所在前半区 |
| decoder blocks 6–11 | 排除后半区独立放大 |
| full decoder + heads/CE | 验证 decoder 与输出/loss 路径的联合敏感性 |
| decoder blocks 0–5 + heads/CE | 测试更小的联合区域 |

Phase 2 的 BF16 control 作为固定 reference，不重复占用 GPU；新 analyzer 必须验证 source/data/model/runtime/first-batch/init/RNG 与该 control 对齐。

- decoder：先比较 blocks 0–5 与 6–11；若半区命中，再在第二轮比较 attention 与 FFN，必要时细到单个 block。
- 输出路径：分别比较 SID embedding、四个 code heads、control head 和 CE reduction。
- 每轮仍使用 default/`workspace_16` 成对的 50-step、dropout=0 protocol，并沿用 Phase 2 的 50% 双判据。
- 找到最小有效 FP32 区域后，用 dropout=0.1 重跑 50 steps；只有效果方向保持一致才进入完整训练。

停止条件：

- 找到一个最小局部 FP32 区域；或
- 两轮后没有区域达到 50% 双判据。后一情况说明敏感性分散在整个网络，转入 Phase 5 的优化稳定性路线。

- **Status：done**

## Phase 4：完整训练验证局部 FP32

短 probe 只用于定位，不能据此宣布 baseline 恢复。

1. 直接复用 archived RTX checkpoint 已锁定的 Official core75 reference：Overall R@1/10/50=`28.3683/69.0000/75.0000`（75 queries、beam=50、beam-order、K={1,10,50}）；Single R@1=`41.6667`（24 Single queries、beam=50、beam-order、K=1）；Multi R@1=`22.1102`（51 Multi queries、beam=50、beam-order、K=1）。
2. 对最多两个局部 FP32 候选跑完整 archived schedule：catalog=3 epochs、query=5 epochs、batch=16、seed=20260626、lr=3e-4、dropout=0.1、每 200 个 query updates 以同一 dev loss 规则选择 best。每个候选同时跑 default 与 `workspace_16`。
3. 直接使用 Official core75 75-query 选择候选。`workspace_16` 的 Overall R@10/R@50 已分别超过 RTX reference `0.5556/0.3333` 个点（catalog=3 epochs、query=5 epochs、batch=16、seed=20260626、lr=3e-4、75 queries、beam=50、beam-order），因此深召回只设不回退门槛。候选的 default 与 `workspace_16` 两条路径都需要：
   - Official Overall R@1 `>=27.3683`（75 queries、beam=50、beam-order），即相对 RTX reference `28.3683` 最多低 `1.0` 个点；高于 reference 允许通过；
   - Official Overall R@10 `>=69.0000`、R@50 `>=75.0000`（75 queries、beam=50、beam-order），不牺牲已经恢复的深召回；
   - Official Single R@1 `>=41.6667`（24 Single queries、beam=50、beam-order）。该子集每改变 1 个 query 的命中状态会改变 `4.1667` 个点，因此“最多低 1.0 个点”在这里等价于匹配或超过 RTX；
   - Official Multi R@1 `>=21.1102`（51 Multi queries、beam=50、beam-order），即相对 RTX reference `22.1102` 最多低 `1.0` 个点；
   - 同一候选的 default/`workspace_16` Overall R@1 相差不超过 `1.0` 个点（75 queries、beam=50、beam-order）；
   - 无 OOM/NaN，完整 manifest 与 checkpoint provenance 通过。
4. 未通过的 candidate 不跑 canonical private test 1,000-query；该集合只留给最终锁定配方做一次补充泛化检查。

通过后将该配方标记为“局部精度 A800 候选”，尚不直接用于后续方法实验。

- **Status：skipped**；Phase 2/3 没有局部 FP32 候选通过 50-step 双门禁。

## Phase 5：验证 catalog→query optimizer 放大器

仅在以下任一条件满足时进入：局部 FP32 没有通过 Phase 4；或局部 FP32 恢复深召回但 Top-1 仍明显偏低。

第一步使用完整 2×2 factorial 隔离 optimizer state 与 query lr；不加入 scheduler、warmup 或 clipping：

- 在一个 `8×A800` 节点运行 backend=`{default, workspace_16}` × optimizer state=`{carry, reset}` × query lr=`{2e-4, 3e-4}`，共八个独立单卡 arm。
- 八个 arm 的 catalog 阶段都固定为 2,730 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、betas=(0.9,0.999)、eps=1e-8、dropout=0.1。每个 backend 内四个 catalog-final model/optimizer hash 必须 exact，确保 query 分叉前状态一致。
- query 阶段只跑前 50 updates、batch=16、dropout=0.1。carry 保留 catalog AdamW moments；reset 从同一 catalog-final model 重建 AdamW。query lr 只取对应 factorial level，其余条件不变。
- 在 CPU FP64 上比较 default/workspace16 的 query update-vector 差异：`D_update = ||delta_default - delta_workspace||_2 / ||delta_default||_2`，其中 delta 是 query step 50 相对各自 catalog-final model 的参数变化。同时记录 step-1/50 loss gap 与 gradient/update trace。
- optimizer reset 若在同一 query lr 下把 `D_update` 与 step-50 loss gap 都降低至少 50%，再进入完整 query 训练；两个 lr 都报告，不用 Official core75 选择短 probe。

短 probe 已完成。reset 在 query lr=`2e-4` 时把 `D_update` 降低 `14.9632%`、step-50 loss gap 增加 `19.3827%`；在 query lr=`3e-4` 时把两项分别降低 `13.1786%/5.6570%`。条件均为 A800 default 对 `workspace_16`、catalog=`2,730` updates、query=`50` updates、batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`。没有组合通过双 `50%` 门禁。

因此执行一次限定的完整 query sweep：

- selection backend 固定为 `workspace_16`，因为它在 legacy full run 已达到 Overall R@10/R@50=`69.5556/75.3333`（catalog=`3` epochs、query=`5` epochs、batch=`16`、seed=`20260626`、lr=`3e-4`、Official core75=`75` queries、beam=`50`、beam-order），目前主要缺口是 Top-1；
- optimizer state=`{carry, reset}`，query lr=`{1e-4, 2e-4, 3e-4}`，共 `6` 个 selection arms；每个 arm 都跑 catalog=`2,730` updates 与 query=`38,805` updates，batch=`16`、seed=`20260626`、catalog lr=`3e-4`、weight decay=`0`、dropout=`0.1`、每 `200` 个 query updates 以 dev loss 选择 best；
- 同一 `8×A800` 节点剩余两张卡运行 `workspace_16/carry/query lr=3e-4` exact replica 与 `default/carry/query lr=3e-4` legacy anchor；两者使用同一完整日程，只用于确定性和历史结果对齐，不参与六选一；
- 八个 arm 都评测 Official core75=`75` queries、beam=`50`、beam-order、K=`{1,10,50}`；selection 只使用前六个 `workspace_16` arms，canonical private test 不参与选择。

完整 query sweep 已完成。六个 selection arms 均未达到 Phase 4 的全部 Official core75 标准；条件统一为 A800 `workspace_16`、catalog=`2,730` updates、query=`38,805` updates、batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`、BF16、dev-best checkpoint、Official core75=`75` queries、beam=`50`、beam-order。最高 Overall R@1=`25.4794` 来自 carry/query lr=`2e-4`，但其 R@10/R@50=`64.2889/68.7778`；carry/query lr=`3e-4` 是唯一达到 R@10/R@50=`69.5556/75.3333` 的组合，但 R@1=`20.9238`。七路 catalog invariant、query lr=`3e-4` exact replica 与两条历史 anchor 全部通过。

下一轮在 carry/query lr=`3e-4` 上分别测试 query `warmup_steps=500` 与 query `max_grad_norm=1.0`。两项不组合。单节点 `8×A800` 映射为：

- backend=`{default, workspace_16}` × intervention=`{control, warmup, clip}`，共六个完整单卡 arm；
- 另外运行 `workspace_16/warmup` 与 `workspace_16/clip` 各一个 exact replica，共两个完整单卡 arm；
- warmup 仅作用于 query optimizer step：第 `s` 步使用 `3e-4 × min(s/500, 1)`，因此 query step=`1/500/501` 的 lr=`6e-7/3e-4/3e-4`；
- clipping 仅作用于 query backward 后、AdamW step 前，global L2 `max_grad_norm=1.0`、`foreach=True`、非有限 norm 直接失败；
- 每个 backend 内所有 arm 的 catalog-final model/optimizer/RNG、首批与 query step-1 pre-intervention gradient 必须 exact；两个 replica 分别要求 final model、best step/dev loss、predictions 与 metrics exact；
- default/workspace control 必须精确命中 2026-08-09 历史 anchor；所有 arm 评测 Official core75=`75` queries、beam=`50`、beam-order，selection 只比较两个 workspace intervention arms。

query stability ablation 已完成。workspace control/warmup/clip 的 Overall R@1/R@10/R@50=`20.9238/69.5556/75.3333`、`23.9905/64.8444/73.0000`、`23.8794/65.1778/73.3333`；条件为 catalog=`2,730` updates、query=`38,805` updates、batch=`16`、seed=`20260626`、catalog/query base lr=`3e-4`、dropout=`0.1`、dev-best checkpoint、Official core75=`75` queries、beam=`50`、beam-order。两项 intervention 均未通过完整门槛。

逐 query 取证显示 RTX 与 workspace control/warmup/clip 的平均 Top50 共享率=`24.00%/26.35%/27.01%`；workspace 三路两两平均 Top50 共享率=`30.53%–33.23%`。条件为对每条 Official core75 query 的固定长度 Top50 集合求交集比例后取 `75` 条均值。下一步不再加入训练超参，先跑 fixed-pool scoring diagnostic：

- 每条 query 的候选池固定为四路 beam-order Top50 union 加全部 gold；四个 checkpoint 使用 exact 同一 pool；
- score 固定为四个 learned-code token 与 stop control=`1` 的 full-vocabulary CE log-prob 之和，不做 trie-child renormalization；逐 query beam capacity 等于 pool size，max skills=`1`，保证穷举；
- 一个 `8×A800` 节点运行四个 checkpoint × 两个 Official core75 query shard；每个 shard 处理 `row_position % 2 == shard_id`；
- 三个 A800 checkpoint 必须在自身 archived Top50 上满足 score replay 最大绝对误差 `<=5e-5`；否则停止解释 fixed-pool 结果；
- 若 raw beam top-1 错误而 fixed-pool top-1 正确，归为 search omission；若 fixed-pool 仍错误，归为 checkpoint scoring failure。两类同时存在时，后续 baseline 需要改训练目标或 decoder/search，而不是继续枚举 backend flag。

fixed-pool scoring diagnostic 已完成。四路 raw→fixed top-1 gained/lost 均为 `0/0`；RTX-correct/A800-wrong 的 control/warmup/clip=`9/9/8` 条 query fixed top-1 恢复均为 `0`。条件为 Official core75=`75` queries、pool=four beam-order Top50 union plus all gold、四个 learned-code full-vocabulary log-probs 加 stop control=`1`、逐 query穷举。四路 R@1/R@5 均与 raw 完全相同；search omission 只在 control R@10 和四路 R@50 产生增益。

position-score decomposition 已完成。control/warmup/clip 的 `3,061` 组 normalized weights 中，full gate 通过数均为 `0`；条件为 Official core75=`75` queries、fixed pool、R@1/R@10/R@50 门槛=`27.3683/69.0000/75.0000`。五折 heldout R@1 也没有超过各自 uniform baseline。全局 position scaling 停止，不进入 full-catalog beam evaluator。

在线训练前的 attribution 已完成。c1 占 control/warmup=`500`/clip=`1.0` endpoint R@1 gap 的 `45.9%/52.9%/48.5%`（Official core75=`75` queries、fixed pool、全部 `32` 个 component masks 的 exact Shapley），但通过 R@1/R@10/R@50=`27.3683/69.0000/75.0000` 门槛需要同时替换 c0/c1/c2/c3。当前先对冻结 SkillRouter teacher panel 与 LETTER IDs 做 hard-negative coverage audit，然后实现一个保留 CE 的限定 loss probe：c1 分叉 hard negative + c0--c3 cumulative-prefix/sequence ranking。control 固定为已有 A800 `query-only`（Official core75=`75` queries、beam=`50`、beam-order、global batch=`16`、lr=`3e-4`、query epochs=`5`时 R@1/R@10/R@50=`27.26/69.33/76.78`）。不再增加 backend、全局 position weight 或 catalog exposure sweep。

这一阶段得到的是新的 A800 稳定训练配方，报告中必须与“严格 legacy reproduction”分开命名。

- **Status：in_progress**

## Phase 6：多 seed 锁定与最终评测

1. 冻结候选代码、runtime、局部精度区域、backend、optimizer、lr 和 checkpoint selection。
2. 跑 seeds=`{20260626,20260627,20260628}` 的完整训练；每个 seed 使用相同 catalog/query 日程与数据。
3. 在 Official core75 上要求三个 seed 均满足：Overall R@1 `>=27.3683`、R@10 `>=69.0000`、R@50 `>=75.0000`（75 queries、beam=50、beam-order）；Single R@1 `>=41.6667`（24 Single queries、beam=50、beam-order）；Multi R@1 `>=21.1102`（51 Multi queries、beam=50、beam-order）。这把每个 seed 的 Overall/Multi R@1 下行幅度限制为相对 RTX 最多 `1.0` 个点，并要求深召回不低于 RTX。
4. Official core75 对三个 checkpoint 报告 mean、standard deviation、每个 seed 和 paired query-level 差异。配方锁定后，只对固定的 seed=20260626 checkpoint 跑一次 canonical private test 1,000-query，作为补充泛化检查，不据此回选配方。
5. 重新跑 seed=20260626 的 50-step replica，要求 checkpoint tensor byte-identical，确认新 A800 配方在同平台仍然确定。

如果三 seed 门禁失败，baseline 仍不适合承载后续方法结论；回到 Phase 5，但不再扩大 backend 搜索。

- **Status：pending**

## Phase 7：交付与后续实验解锁

- 更新对应 experiment note，记录命令、job ID、runtime、日志、checkpoint、metrics 和失败 arm。
- 生成一张统一表：archived RTX、A800 default、`workspace_16`、局部 FP32、最终 A800 配方；所有数字显式标注训练与评测条件。
- 固化最终 A800 baseline config 和 checkpoint provenance。
- 后续 loss、catalog、RIPOR、PAG 必须在同一 A800 配方下重跑 control 与 method，避免把平台差异计入方法收益。
- 完成 scoped git review、最小 invariant 检查和逻辑提交。

- **Status：pending**

## 决策树

```text
逐层 trace
  ├─ 找到集中敏感区 → 局部 FP32 二分 → 完整训练
  │                                  ├─ Official 恢复 → 3 seeds → 锁定 A800 baseline
  │                                  └─ Top-1/稳定性不足 → optimizer 放大器实验
  └─ 敏感性分散 ───────────────────────────────→ optimizer 放大器实验
                                                     ├─ Official 恢复 → 3 seeds
                                                     └─ 仍失败 → 报告 legacy 配方在 A800 不稳，重新设计 baseline
```

## 完成标准

- 明确到模块级的数值敏感性结论，或用两轮失败证据证明敏感性分散。
- 至少一个 A800 配方在三个 seed 上均通过 Official core75 门禁：Overall R@1 相对 RTX 下行不超过 `1.0` 个点（75 queries、beam=50、beam-order），Overall R@10/R@50 不低于 RTX（75 queries、beam=50、beam-order），Single/Multi R@1 通过各自门槛（24/51 queries、beam=50、beam-order）。
- canonical private test 只对锁定后的 seed=20260626 checkpoint 评测一次，不参与配方回选。
- 所有后续实验能复用同一 baseline config、checkpoint 选择和评测协议。
- 没有 unresolved HIGH 级训练语义或 provenance 问题。

## 已完成训练候选：strict-base CE-prefix ranking

- 固定 base source queries=`94,912`、expanded rows=`124,175`、catalog updates=`2,730`、query updates=`38,805`、batch=`16`、seed=`20260626`、optimizer carry、dev-best 与 `CUBLAS_WORKSPACE_CONFIG=:16:8`。
- candidate bundle 固定 summary SHA-256=`72b0f525d6c3d9feabc7770c054ee70d8478f1b280d1a100793da8d5b6c6dd6d`、manifest SHA-256=`8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d`；teacher top-50 条件下 c1 negative coverage=`86,305/124,175=69.5027%`。
- 一个 `8×A800` 节点跑两个 gate=`0` control、cumulative lambda=`{0.02,0.05,0.10,0.20}`、c1-only lambda=`0.10`、sequence-only lambda=`0.10`；每个 arm 独占一张 A800。
- positive CE forward 保留原 batch shape/order；auxiliary decoder 在 forked CPU/CUDA RNG state 内运行并恢复主 RNG。cumulative loss 先在 depth=`2/3/4` 分别计算 softplus，再对 row×depth 求平均。
- control 必须命中历史 model-state SHA-256=`bbec7389f56b36a24db000111adf8ce99e3bf712609cfcd1562c64c9910172f8`、prediction SHA-256=`3663a15ad9859abc76141d815f0da3c88bd9b3d5d9e914f09a3d8b0a3ed843f4`、best query step=`38,200` 与 dev loss=`0.7701343703584546`（上述 strict-base 条件）。
- 除 Overall absolute gate 外，Single/Multi R@10/R@50 相对 concurrent control 的允许回退均为 `1.0` 个百分点（Official core75=`75` queries、beam=`50`、beam-order）。
- 结果：六个非零 arm 的联合门槛通过数=`0/6`。`seq_010` 的 Overall R@1/R@10/R@50=`28.7683/70.5873/73.8889`、Single R@1=`50.0000`、Multi R@1=`18.7768`；`cum_002` 对应 Overall=`26.3683/69.9556/77.1111`、Single R@1=`33.3333`、Multi R@1=`23.0906`。条件为各自 dev-best checkpoint、Official core75=`75` queries、beam=`50`、beam-order。
- 决策：不进入多 seed。先在已保存 predictions 上做 control/`seq_010`/`cum_002` 的逐 query error-overlap，定位 Single/Multi 互补样本，再设计下一轮限定 loss。
- paired error-overlap 已完成。Single 的 sequence top-1 hit set与 Multi 的 cumulative top-1 hit set分别包含另外两路的全部命中；按 gold cardinality 组合的诊断 Overall R@1/R@10/R@50=`31.7016/72.6222/75.7778`（Official core75=`75` queries、beam=`50`、beam-order），R@1 与逐 query oracle相同。
- 下一训练候选：同一个 checkpoint 内保留 CE；Single rows 使用 sequence lambda=`0.10`，Multi rows 使用 cumulative lambda=`0.02`。query 类型只在训练时由 frozen manifest 的 `gold_skill_ids` 数量决定，推理协议不变。Multi R@50 继续使用相对 concurrent control 最多回退 `1.0` 个百分点的硬门槛。
- conditional hybrid 首轮固定一个 `8×A800` 节点：两个 disabled controls、fixed `seq_010/cum_002` anchors，以及 Single sequence lambda=`{0.05,0.10}` × Multi cumulative lambda=`{0.01,0.02}` 四个 selection arms。每路独占一张 A800；其余 strict-base 条件保持不变。
- conditional loss 对每个 eligible expanded row 先在 active depths 内取 mean，再乘组内 lambda，最后对 batch eligible rows 取 mean。frozen manifest 条件下，query epochs=`5` 的 Single/Multi presentations=`270,675/160,850`，active comparisons=`270,675/482,550`，合计=`753,225`。
- Official core75=`75` queries 已参与 scope/lambda 提议，本轮结果标记为探索性选择。没有 hybrid arm 通过原 absolute/deep-recall gate 时写 `selected_arm=null`；有 arm 通过后才进入 seeds=`{20260626,20260627,20260628}` 和一次不回选的 canonical private test=`1,000` queries。

## 错误与防重复

| 风险/错误 | 处理 |
|---|---|
| instrumentation 改变 CUDA allocator 或 kernel 选择 | 用带/不带 trace 的同 backend 等价门禁；失败后缩小采样或独立 replay |
| 只追逐 RTX step-1 loss，得到偶然接近的 arm | 以 A800 backend 差异收敛和最终 Official core75 恢复共同判断 |
| Official core75 参与精度/optimizer 配方选择 | 所有结果明确标注为 selection-set metrics；最终只用 canonical private test 做一次不回选的补充检查 |
| 全模型 FP32 再次 OOM | 不再重跑；只做局部 FP32，OOM 时缩小区域 |
| 短 probe loss 接近但最终检索仍差 | 任何候选必须通过完整训练与 retrieval 门禁 |
| 为了恢复指标同时改多个优化变量 | optimizer reset、lr、warmup、clipping 按阶段隔离 |
| 本地 `uv` 环境缺少 torch，tensor 单测首次启动失败 | 使用已冻结的 Python 3.10 + torch 2.10.0 runtime 做 CPU 单测；GPU hook/invariant 继续在 `8×A800` preflight 验证 |
| 首次解包本地 frozen runtime 的命令因包含临时目录递归清理而被执行器拒绝 | 改用唯一 `/tmp` 目录且不在命令内删除；依靠临时文件系统生命周期回收，不绕过安全策略 |
| Python 3.10 frozen runtime 单独加载 torch 时缺少 `libcudnn.so.9` | launcher 与本地测试叠加已校验的 Python 3.11 site archive，仅从中提供 NVIDIA shared libraries；Python 与 torch 仍使用冻结的 3.10 runtime |
| Tianzhi `submit.sh --help` 返回 unknown option | 直接读取 skill 自带脚本的 usage/参数解析，随后 dry-run 验证 resolved request，再正式提交 |
| Phase 1 v1 四个 traced arm 在训练后均报告缺失 `shared_embedding` boundary | hook 从未被调用的 `backbone.shared` 改到实际 encoder 路径 `backbone.encoder.embed_tokens`；新增回归测试，v1 结果保留并以 v2 新目录重跑 |
