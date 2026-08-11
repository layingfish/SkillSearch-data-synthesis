# Experiment: A800 legacy CE-prefix ranking loss probe
Status: running
Risk Tier: 3

## Goal / Hypothesis / Change Scope

在严格 legacy base 协议上测试累计 prefix ranking loss。目标是把 A800 Overall R@1 恢复到 RTX 基准 `28.3683%` 的 `1.0` 个百分点以内，同时保持 R@10/R@50。训练保留 catalog→query 顺序、同一个 AdamW state、dev-best 回载、LETTER IDs、CE decoder 和 beam-order 评测。

冻结输入为 base source queries=`94,912`、expanded query rows=`124,175`、catalog skills=`14,551`、dev expanded rows=`2,501`。base candidate manifest SHA-256=`8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d`；它在 teacher top-50 排除全部 sibling GT 的条件下，为 `86,305/124,175=69.5027%` 的 expanded rows 提供 same-c0/different-c1 negative。

每个 eligible row 使用一个冻结 negative。positive/negative score 使用 legacy `code_heads` 的 full-vocabulary log-softmax。depth=`d` 的 score 是 c0 到 c(d-1) learned-code log-prob 之和；stop control 不进入 ranking loss。

- `c1_only`：active depths=`{2}`。
- `sequence_only`：active depths=`{4}`。
- `cumulative`：active depths=`{2,3,4}`。

每个 active depth 单独计算 `softplus((S_neg_d-S_pos_d)/temperature)`，再对 eligible row × active depth 求平均；不先平均 margin。temperature=`1.0`。总 loss=`CE + lambda * prefix_rank_loss`；CE 仍按全部有效 target token 平均。

## Plan

1. 校验 archived pipeline、base rows、dev rows、catalog、LETTER IDs、teacher summary 与 candidate manifest 的 SHA-256。
2. 只在 query-train expanded row 上绑定 negative；catalog、dev 与无 candidate 的 query row 保持原 CE forward。
3. 增加 gate=`0` output/loss/gradient/update 等价、prefix score、loss normalization、sibling exclusion和 manifest 对齐测试。
4. 保持 catalog updates=`2,730`、query updates=`38,805`、batch=`16`、seed=`20260626` 与 optimizer carry exact。
5. 一个 `8×A800` 节点同时跑八个单卡 arm；两个 gate=`0` control 必须命中历史 workspace anchor 且彼此 exact。
6. 八路都跑 Official core75=`75` queries、beam=`50`、beam-order，并报告 Overall/Single/Multi 的 R@1/R@10/R@50。
7. 通过 absolute recovery gate 和 concurrent-control deep-recall gate 的 arm 才进入多 seed。

## Commands

Train: job=`tide-chenyufei03-10603-gigmv`，pod=`tide-chenyufei03-10603-gigmv-worker-0`，queue=`rudder-exper`，node=`1×8 A800`，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`1310c8d8cda9275e07a1dbb961068876ed796de6`。实际训练 runtime 从冻结 archive 加载：Python=`3.10.12`、torch=`2.10.0+cu128`、CUDA package=`12.8`、cuDNN=`9.10.2`、Transformers=`5.3.0`、Tokenizers=`0.22.2`。T5-base weight SHA-256=`a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4`；archived pipeline SHA-256=`7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76`。

每路 catalog epochs=`3`（updates=`2,730`）、query epochs=`5`（updates=`38,805`）、batch=`16`、seed=`20260626`、AdamW lr=`3e-4`、weight decay=`0`、`foreach=None`、optimizer carry、scheduler=none、warmup=`0`、gradient clipping=disabled、BF16、dropout=`0.1`、max source length=`2048`、dev eval interval=`200` query updates、dev-best 回载、`CUBLAS_WORKSPACE_CONFIG=:16:8`。catalog→query 不重建 optimizer，不重置 Python/CPU/CUDA RNG；query 沿用同一个 `random.Random(seed=20260626)` 与 CUDA dropout stream。

| GPU | arm | active depths | lambda |
|---:|---|---|---:|
| `0` | `control_a` | disabled | `0` |
| `1` | `control_b` | disabled | `0` |
| `2` | `cum_002` | `{2,3,4}` | `0.02` |
| `3` | `cum_005` | `{2,3,4}` | `0.05` |
| `4` | `cum_010` | `{2,3,4}` | `0.10` |
| `5` | `cum_020` | `{2,3,4}` | `0.20` |
| `6` | `c1_010` | `{2}` | `0.10` |
| `7` | `seq_010` | `{4}` | `0.10` |

每个 query epoch 的 eligible pair presentations=`86,305`。`c1_only/sequence_only` 每个 query epoch的 active comparisons=`86,305`；`cumulative` 每个 query epoch=`258,915`。query epochs=`5` 时，对应总数为 eligible pair presentations=`431,525`，单-depth active comparisons=`431,525`，cumulative active comparisons=`1,294,575`。

Eval: 八路 Official core75=`75` queries；K=`{1,10,50}`、beam=`50`、beam-order、max skills=`1`。

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_ce_prefix_ranking_loss_20260811_v1/arms/<arm>/train/final/`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_ce_prefix_ranking_loss_20260811_v1/logs/`

## Verification

- [x] `when gate=0, student_forward output must be allclose to base-model output`
- [x] 冻结 torch=`2.10.0+cu128` CPU runtime 下，gate=`0` 的 output、gradient、single update exact；gate-on 但 rank contribution=`0` 时，当前/下一 batch CE、主 RNG、gradient、update exact
- [x] GPU preflight 重复 gate-on/rank contribution=`0` 的主 CUDA RNG 与下一 batch CE invariant
- [ ] 两个 full control 的 final model/predictions exact
- [ ] 两个 control 命中 workspace anchor：model-state full SHA-256=`bbec7389f56b36a24db000111adf8ce99e3bf712609cfcd1562c64c9910172f8`、predictions SHA-256=`3663a15ad9859abc76141d815f0da3c88bd9b3d5d9e914f09a3d8b0a3ed843f4`、best query step=`38,200`、best dev loss=`0.7701343703584546`（base source queries=`94,912`、catalog updates=`2,730`、query updates=`38,805`、batch=`16`、seed=`20260626`、BF16、`:16:8`）
- [ ] candidate manifest/source rows/semantic IDs SHA-256 与 checkpoint provenance 精确
- [ ] 每个 negative 排除全部 sibling GT，且与 positive 在 c1 首分叉
- [ ] cumulative loss 是各 depth softplus 后的 mean
- [ ] 同 query 的 positive/negative 在 depth=`2/3/4` 上，full-sequence cumulative full-vocab logprob 与 legacy stepwise decoder allclose（rtol=`1e-5`、atol=`5e-5`、FP32 preflight）；encoder cross-attention context与原 decoder prefix position保持完整
- [ ] 每个 epoch 的 CE rows、eligible pairs、active comparisons 精确
- [ ] loss/gradient 无 NaN/Inf，记录 peak GPU memory
- [x] Reviewer required? yes

## Result / Notes

Absolute recovery gate：Overall R@1 `>=27.368253968253965%`、R@10 `>=69.0000%`、R@50 `>=75.0000%`（Official core75=`75` queries、beam=`50`、beam-order）；Single R@1 `>=41.66666666666667%`（Single=`24` queries、同一 decoding 条件）；Multi R@1 `>=21.110177404295052%`（Multi=`51` queries、同一 decoding 条件）。Overall/Multi R@1 门槛由对应 RTX anchor 精确减 `1.0` 个百分点。Single 子集只有 `24` 条 query，每个 hit 对应 `4.16666666666667` 个百分点，因此容忍 `1.0` 个百分点仍要求命中数与 RTX 相同。

Deep-recall gate：每个非零 arm 的 Single/Multi R@10/R@50 均不得低于两个 exact concurrent controls 的共同值超过 `1.0` 个百分点（Official core75 子集、beam=`50`、beam-order）。若多个 arm 通过，依次按 Overall R@1、Multi R@1、Single R@1、Overall R@10、较低 dev loss 选择。

启动前验证：冻结 Python=`3.10.12`、torch=`2.10.0+cu128` CPU runtime 下运行 `21` 项相关测试，`20` 项通过，`1` 项 CUDA-only 测试跳过；节点会在训练前补跑该 CUDA invariant。独立交叉审查初次发现 `2` 项 HIGH：R@1 gate 的显示值舍入，以及无过门 arm 时生成 fallback winner。两项已修复并加入回归测试。

集群启动验证：pod 在 `1/1 Running`、restart=`0` 条件下识别到 `8` 张 GPU；冻结 runtime 字段全部命中。节点上的 `21` 项测试全部通过，包括 CUDA RNG invariant。GPU `0–7` 已分别拉起八个独立 arm。八路已同步到 catalog step=`100/2,730`，train loss=`4.092995`（batch=`16`、seed=`20260626`、lr=`3e-4`、BF16、dropout=`0.1`）；八路数值完全相同，符合 query ranking loss 尚未启用的预期。
