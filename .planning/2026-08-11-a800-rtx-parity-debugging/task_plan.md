# A800 / RTX PRO 6000 训练差异排障计划

## 目标

在 RTX PRO 6000 Blackwell 不可用的条件下，定位 A800 上 legacy BF16 训练的主要数值敏感模块，并锁定一个强、稳定、可供后续 loss、catalog、RIPOR、PAG 实验共同使用的 A800 基线。

“A800 基线”指专门在当前 A800 Ampere + driver `535.161.07` 环境上验证并冻结的训练配方。它不要求逐 step 重现 Blackwell 轨迹，但必须恢复接近 archived RTX checkpoint 的检索能力，并通过多 seed 稳定性检查。

## 当前状态

- 状态：`in_progress`；Phase 2/3 已证明局部 FP32 无法让参数轨迹收敛。
- 当前阶段：Phase 5，catalog→query optimizer 稳定性。
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

若 reset 仍不足，执行一次限定 sweep：

- optimizer state=`{carry, reset}`；
- query lr 补充 `1e-4`，并与首轮已完成的 `{2e-4, 3e-4}` 合并分析；
- 其余条件保持不变，共 6 个完整 query arms；
- 使用 Official core75 75-query 选择；canonical private test 不参与选择。

只有上述 6 个 arm 全部不能达到 Phase 4 的 Official core75 标准，才在最佳组合上分别测试 `warmup_steps=500` 和 `max_grad_norm=1.0`。两项先单独测试，不在第一轮同时启用。

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
