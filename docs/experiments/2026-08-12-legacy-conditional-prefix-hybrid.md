# Experiment: Legacy conditional prefix-ranking hybrid

Status: planned
Risk Tier: 3

## Goal / Hypothesis / Change Scope

在 strict-base 单 checkpoint 训练中，按 frozen candidate manifest 的原始 gold 数量选择 auxiliary loss：Single row（`gold_count=1`）使用 sequence-only depth=`4`，Multi row（`gold_count>1`）使用 cumulative depths=`{2,3,4}`。推理保持 Official core75=`75` queries、beam=`50`、beam-order、K=`{1,10,50}`。

每个 eligible row 先对自身 active depths 的 `softplus(negative-positive)` 取均值，再乘组内 lambda；batch auxiliary loss 对全部 eligible rows 取均值。保留原 CE loss。固定 scope 路径保持原实现，用于复现上一轮 anchor。

训练条件固定为 source rows=`94,912`、expanded rows=`124,175`、catalog updates=`2,730`、query updates=`38,805`、batch=`16`、seed=`20260626`、catalog/query lr=`3e-4`、dropout=`0.1`、BF16、`CUBLAS_WORKSPACE_CONFIG=:16:8`、dev-best checkpoint。一个 `8×A800` 节点运行：

| GPU | arm | 条件 |
|---:|---|---|
| 0 | `control_a` | auxiliary disabled，lambda=`0` |
| 1 | `control_b` | auxiliary disabled，lambda=`0` |
| 2 | `seq_010` | fixed sequence-only，lambda=`0.10` |
| 3 | `cum_002` | fixed cumulative，lambda=`0.02` |
| 4 | `hybrid_s005_m001` | Single sequence lambda=`0.05`；Multi cumulative lambda=`0.01` |
| 5 | `hybrid_s005_m002` | Single sequence lambda=`0.05`；Multi cumulative lambda=`0.02` |
| 6 | `hybrid_s010_m001` | Single sequence lambda=`0.10`；Multi cumulative lambda=`0.01` |
| 7 | `hybrid_s010_m002` | Single sequence lambda=`0.10`；Multi cumulative lambda=`0.02` |

## Commands

Train: pending verified commit and Tianzhi job ID.

Eval: each arm uses Official core75=`75` queries、beam=`50`、beam-order、decoder head=`ce`、score transform=`logprob`、K=`{1,10,50}`。

本轮是探索性选择，因为同一个 Official core75=`75` queries 已用于提出 scope 和 lambda。只在四个 hybrid arms 中选择。门槛固定为 Overall R@1 `>=0.27368253968253965`、R@10 `>=0.69`、R@50 `>=0.75`；Single R@1 `>=10/24`；Multi R@1 `>=0.21110177404295052`；Single/Multi R@10/R@50 相对 concurrent control 的下降均不超过 `0.01`（dev-best、Official core75=`75` queries、beam=`50`、beam-order）。没有 arm 全部通过时写 `selected_arm=null`。

若有 arm 通过，只冻结排名最高的一路。随后按 seeds=`{20260626,20260627,20260628}`、其余 strict-base 条件不变做稳定性验证；最终只对锁定配方的 seed=`20260626` checkpoint 评测一次 canonical private test=`1,000` queries，不使用该结果回选 lambda。

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_conditional_prefix_hybrid_20260812_v1/arms/<arm>/train/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_conditional_prefix_hybrid_20260812_v1/logs`

Metrics: pending.

## Verification

- [x] Minimal local check
- [x] Unit/invariant test if needed
- [x] Reviewer required? yes
- [ ] `gate=0` forward、gradient、update、next CE 与 base exact
- [ ] `control_a/control_b` 命中 strict-base 历史 checkpoint、prediction、metrics hash
- [ ] `seq_010/cum_002` 命中上一轮对应 checkpoint、prediction、metrics hash
- [ ] gold cardinality 来自 hash-bound frozen manifest；expanded rows 与 manifest 闭合
- [ ] conditional group presentation、active comparison 与 trace 计数闭合
- [x] manifest SHA-256=`8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d` 下 source Single/Multi=`77,455/17,457`、expanded Single/Multi=`77,455/46,720`、每 epoch eligible Single/Multi=`54,135/32,170`
- [x] canonical `(candidate_key, gold_count)` mapping SHA-256=`016728eb7681486d41eb83a9de27fa32c56e70adce1a0b17d47d8631bff796aa`（上述 frozen manifest 条件）
- [ ] query epochs=`5` 时 eligible presentations Single/Multi=`270,675/160,850`，active comparisons Single/Multi=`270,675/482,550`、合计=`753,225`
- [ ] Single/Multi positive/negative score parity 各至少一条

## Result / Notes

计划依据：上一轮 strict-base dev-best checkpoint 的 paired analysis 中，Single 的 sequence-only lambda=`0.10` top-1 hit set包含其余两路命中；Multi 的 cumulative lambda=`0.02` top-1 hit set包含其余两路命中。gold-cardinality diagnostic router 的 Overall R@1/R@10/R@50=`31.7016/72.6222/75.7778%`（Official core75=`75` queries、beam=`50`、beam-order）；该结果只用于提出单 checkpoint 训练候选。

独立计划复核：无 CRITICAL。确认 per-row depth mean → group lambda → eligible-row mean 的归一化与现有 expanded-row CE 语义一致。首轮四个 hybrid arms 足以判断是否存在可用权重；若有 arm 过门，再补 `Single lambda=0/Multi lambda=0.02` 与 `Single lambda=0.10/Multi lambda=0` 两个单侧 ablation，验证两个分支是否都必要。

本地验证：冻结 Python=`3.10.12`、torch=`2.10.0+cu128` CPU runtime 下运行相关测试=`35` 项，结果为通过=`34`、CUDA-only 跳过=`1`；包含 mixed/Single-only/Multi-only 手算归一化、无 eligible row、Single 仅 depth=`4` gradient、Multi depths=`{2,3,4}` gradient、gold-count 传播与 conditional gate=`0` 的 forward/RNG/gradient/update/next-CE exact。节点 preflight 会在可见 GPU=`1` 条件下补跑 fixed/conditional CUDA RNG invariant。`ruff check`、`py_compile` 与两个 launcher 的 `bash -n` 已通过；本地未安装 `shellcheck`。

implementation cross-review 最终结论：CRITICAL=`0`、HIGH=`0`。复核确认 loss 归一化、fixed anchor 兼容、eligible/active comparison 计数、旧 gate、无 fallback 语义和 GPU `0–7` arm 映射成立。
