# Experiment: Legacy conditional prefix-ranking hybrid

Status: done
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

Train: job=`tide-chenyufei03-10607-crtcs`，queue=`rudder-exper`，node=`1×8 A800`，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`cad6fcf35d7e5353728f8a62bae1653dbbb59f21`。

Eval: each arm uses Official core75=`75` queries、beam=`50`、beam-order、decoder head=`ce`、score transform=`logprob`、K=`{1,10,50}`。

本轮是探索性选择，因为同一个 Official core75=`75` queries 已用于提出 scope 和 lambda。只在四个 hybrid arms 中选择。门槛固定为 Overall R@1 `>=0.27368253968253965`、R@10 `>=0.69`、R@50 `>=0.75`；Single R@1 `>=10/24`；Multi R@1 `>=0.21110177404295052`；Single/Multi R@10/R@50 相对 concurrent control 的下降均不超过 `0.01`（dev-best、Official core75=`75` queries、beam=`50`、beam-order）。没有 arm 全部通过时写 `selected_arm=null`。

若有 arm 通过，只冻结排名最高的一路。随后按 seeds=`{20260626,20260627,20260628}`、其余 strict-base 条件不变做稳定性验证；最终只对锁定配方的 seed=`20260626` checkpoint 评测一次 canonical private test=`1,000` queries，不使用该结果回选 lambda。

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_conditional_prefix_hybrid_20260812_v1/arms/<arm>/train/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_conditional_prefix_hybrid_20260812_v1/logs`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/a800_legacy_conditional_prefix_hybrid_20260812_v1/conditional_prefix_hybrid_summary.json`

下表统一使用 dev-best checkpoint、Official core75=`75` queries、beam=`50`、beam-order、CE log-prob。训练统一使用 source rows=`94,912`、expanded rows=`124,175`、catalog/query updates=`2,730/38,805`、batch=`16`、seed=`20260626`、lr=`3e-4`、dropout=`0.1`、BF16、`CUBLAS_WORKSPACE_CONFIG=:16:8`。

| arm | auxiliary 条件 | best step | dev loss（2,501 rows） | Overall R@1 | Overall R@10 | Overall R@50 | Single R@1（24 queries） | Multi R@1（51 queries） | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `control_a` | disabled | 38,200 | 0.770134 | 20.9238% | 69.5556% | 75.3333% | 25.0000% | 19.0056% | audit |
| `control_b` | disabled | 38,200 | 0.770134 | 20.9238% | 69.5556% | 75.3333% | 25.0000% | 19.0056% | audit |
| `seq_010` | fixed sequence lambda=`0.10` | 38,800 | 0.808841 | 28.7683% | 70.5873% | 73.8889% | 50.0000% | 18.7768% | audit |
| `cum_002` | fixed cumulative lambda=`0.02` | 38,800 | 0.792865 | 26.3683% | 69.9556% | 77.1111% | 33.3333% | 23.0906% | audit |
| `hybrid_s005_m001` | Single=`0.05`；Multi=`0.01` | 38,805 | 0.794514 | 24.4349% | 68.0000% | 74.2222% | 33.3333% | 20.2474% | fail |
| `hybrid_s005_m002` | Single=`0.05`；Multi=`0.02` | 38,805 | 0.794744 | 19.9238% | 65.5556% | 71.7778% | 25.0000% | 17.5350% | fail |
| `hybrid_s010_m001` | Single=`0.10`；Multi=`0.01` | 38,800 | 0.796435 | 24.1460% | 62.8095% | 70.5556% | 29.1667% | 21.7834% | fail |
| `hybrid_s010_m002` | Single=`0.10`；Multi=`0.02` | 38,805 | 0.787241 | 21.2571% | 63.9206% | 73.3333% | 20.8333% | 21.4566% | fail |

## Verification

- [x] Minimal local check
- [x] Unit/invariant test if needed
- [x] Reviewer required? yes
- [x] `gate=0` forward、gradient、update、next CE 与 base exact
- [x] `control_a/control_b` 命中 strict-base 历史 checkpoint、prediction、metrics hash
- [x] `seq_010/cum_002` 命中上一轮对应 checkpoint、prediction、metrics hash
- [x] gold cardinality 来自 hash-bound frozen manifest；expanded rows 与 manifest 闭合
- [x] conditional group presentation、active comparison 与 trace 计数闭合
- [x] manifest SHA-256=`8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d` 下 source Single/Multi=`77,455/17,457`、expanded Single/Multi=`77,455/46,720`、每 epoch eligible Single/Multi=`54,135/32,170`
- [x] canonical `(candidate_key, gold_count)` mapping SHA-256=`016728eb7681486d41eb83a9de27fa32c56e70adce1a0b17d47d8631bff796aa`（上述 frozen manifest 条件）
- [x] query epochs=`5` 时 eligible presentations Single/Multi=`270,675/160,850`，active comparisons Single/Multi=`270,675/482,550`、合计=`753,225`
- [x] Single/Multi positive/negative score parity 各至少一条；四个 hybrid arm 的 max absolute error=`0`（atol=`5e-5`、rtol=`1e-5`）
- [x] pod 终态=`Completed`、restart=`0`；日志未匹配 Traceback、OOM、NaN、RuntimeError、ValueError 或 ERROR

## Result / Notes

计划依据：上一轮 strict-base dev-best checkpoint 的 paired analysis 中，Single 的 sequence-only lambda=`0.10` top-1 hit set包含其余两路命中；Multi 的 cumulative lambda=`0.02` top-1 hit set包含其余两路命中。gold-cardinality diagnostic router 的 Overall R@1/R@10/R@50=`31.7016/72.6222/75.7778%`（Official core75=`75` queries、beam=`50`、beam-order）；该结果只用于提出单 checkpoint 训练候选。

独立计划复核：无 CRITICAL。确认 per-row depth mean → group lambda → eligible-row mean 的归一化与现有 expanded-row CE 语义一致。首轮四个 hybrid arms 足以判断是否存在可用权重；若有 arm 过门，再补 `Single lambda=0/Multi lambda=0.02` 与 `Single lambda=0.10/Multi lambda=0` 两个单侧 ablation，验证两个分支是否都必要。

本地验证：冻结 Python=`3.10.12`、torch=`2.10.0+cu128` CPU runtime 下运行相关测试=`35` 项，结果为通过=`34`、CUDA-only 跳过=`1`；包含 mixed/Single-only/Multi-only 手算归一化、无 eligible row、Single 仅 depth=`4` gradient、Multi depths=`{2,3,4}` gradient、gold-count 传播与 conditional gate=`0` 的 forward/RNG/gradient/update/next-CE exact。节点 preflight 会在可见 GPU=`1` 条件下补跑 fixed/conditional CUDA RNG invariant。`ruff check`、`py_compile` 与两个 launcher 的 `bash -n` 已通过；本地未安装 `shellcheck`。

implementation cross-review 最终结论：CRITICAL=`0`、HIGH=`0`。复核确认 loss 归一化、fixed anchor 兼容、eligible/active comparison 计数、旧 gate、无 fallback 语义和 GPU `0–7` arm 映射成立。

Tianzhi dry-run 与正式请求均确认 queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800` 和上述完整 launch commit。job=`tide-chenyufei03-10607-crtcs` 已提交。

节点启动验证：pod=`tide-chenyufei03-10607-crtcs-worker-0` 在 ready=`1/1`、restart=`0` 条件下识别 GPU count=`8`；冻结 runtime 全部命中。节点测试=`35/35` 通过，包括 fixed/conditional gate=`0` 的 CUDA RNG、gradient、update 与 next-CE invariant。GPU `0–7` 已分别拉起八个独立 arm。

训练已实际开始：八路均到 catalog step=`50/2,730`，train loss=`4.171228`（batch=`16`、seed=`20260626`、catalog lr=`3e-4`、BF16、dropout=`0.1`、`CUBLAS_WORKSPACE_CONFIG=:16:8`）。八路数值一致，符合 conditional auxiliary 仅在 query 阶段启用的预期；日志未发现 Traceback、OOM、NaN 或 CUDA error。

终态为 `selected_arm=null`，selection basis=`no_qualified_arm`。四个 hybrid arm 均未同时达到 Overall R@1/R@10/R@50=`27.3683%/69.0000%/75.0000%`、Single R@1=`41.6667%`、Multi R@1=`21.1102%` 及分组深召回 non-inferiority gate（Official core75=`75` queries、beam=`50`、beam-order、dev-best）。因此不启动 seeds=`{20260626,20260627,20260628}` 复验，也不评测 private test=`1,000` queries。

所有复现证据成立。`control_a/control_b` 的 model hash=`bbec7389...172f8`、prediction hash=`3663a15a...43f4` 与 strict-base 历史记录完全一致（auxiliary disabled、seed=`20260626`、上述 strict-base 条件）。`seq_010` 的 model/prediction hash=`f75c52ba...46d0`/`3d1c7ca5...c26e`，`cum_002` 的 model/prediction hash=`204b2512...543b`/`26972fee...3fe7`，均与上一轮对应 fixed anchor 完全一致（各自 auxiliary 条件、上述 strict-base 与 core75 条件）。catalog、control replica、fixed anchor、historical control invariant 全部通过。

结论：gold-cardinality conditional mixing 没有合并 fixed sequence 与 cumulative 的互补收益。四个 hybrid 的 Single R@1=`20.8333%–33.3333%`（24 queries、各自 dev-best、Official core75=`75` queries、beam=`50`、beam-order），都低于门槛=`41.6667%`；最低 dev loss=`0.787241`（`hybrid_s010_m002`、dev=`2,501` rows、step=`38,805`）对应 Overall R@1=`21.2571%`（同一 checkpoint 的 core75 条件），说明当前 auxiliary dev objective 与最终 top-1 检索排序没有可靠对齐。本轮是有效负结果，无技术故障，不重跑。
