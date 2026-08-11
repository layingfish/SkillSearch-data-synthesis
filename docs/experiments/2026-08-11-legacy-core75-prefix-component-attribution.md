# Experiment: Legacy core75 prefix and component-swap attribution
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

只读取 position-score job=`tide-chenyufei03-10602-lbift` 的 v2 shards，不做新 forward 或训练。目标是区分两件事：gold prefix 最早在哪个 learned-code depth 失去排序优势；把 RTX 的某些 position contribution 替换进 A800 total score 后，最多能恢复多少 R@1。

固定条件为 Official core75=`75` queries、pool=four beam-order Top50 union plus all gold、semantic depth=`4`、components=`[c0,c1,c2,c3,stop]`、四路各 `10,708` candidates。prefix 评测按每个 gold skill mention 计算 macro recall；同一 gold prefix 可以代表多个 gold mentions。每个 depth 的同-prefix cumulative score spread 必须 `<=1e-6`。

component swap 枚举五项 contribution 的全部 `2^5=32` 个 RTX→A800 masks。mask=`0` 必须精确重建 A800 fixed-pool ranking/metrics，mask=`31` 必须精确重建 RTX fixed-pool ranking/metrics。报告每个 mask 的 R@1/R@5/R@10/R@50、RTX-correct/A800-wrong Top-1 recovery，以及基于全部 `32` 个 masks 的 exact Shapley R@1 attribution。该分析只用于定位，hybrid score 不能作为可部署配置。

计划：

1. 复用 fixed-pool shard merge、alignment、provenance 与 multi-GT recall 实现。
2. 加载 frozen semantic IDs，并校验 file/logical SHA-256 与 shard manifest。
3. 在 code depth=`1/2/3/4` 计算 unique-prefix ranking、macro recall 和 gold-prefix margin。
4. 对三个 A800 checkpoint 各枚举 `32` 个 component-swap masks，并验证两个端点 exact。
5. 计算 R@1 Shapley contribution、最小 full-gate mask 和 critical-query recovery。
6. 跑 synthetic multi-GT/prefix-sharing、endpoint、Shapley conservation 与端到端测试。
7. 由独立 reviewer 检查 prefix 去重语义、stable tie order、multi-GT 守恒与 hybrid 限制。

## Commands

Analyze: local read-only analyzer `scripts/analyze_legacy_core75_prefix_component_attribution.py`，读取 job=`tide-chenyufei03-10602-lbift` 的八个 score shards；analysis runtime Python=`3.11.11`，analyzer SHA-256=`cbd506ec3c6ba8abdd6cf0f17700549d7815d4299388aa6024dc5a034b92757e`。

## Outputs

Output: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_core75_position_score_decomposition_20260811_v1/prefix_component_attribution/`

- `prefix_component_attribution_summary.json` SHA-256=`e2281f260916e971c5173f2b7b8c00a4711e870db0fc04ee555c2d58d41bcf8b`
- `prefix_component_attribution_per_query.jsonl` SHA-256=`fe56d95deed7f35238ce46802bf4db1a3405a00c0d3f24b64845cc371d025f12`（Official core75=`75` rows）
- `component_swap_grid.jsonl` SHA-256=`acd38f5d643352bf995bd3aa70a8e84d18edb47a7b3eeabc11019a753b1a57e3`（三个 A800 checkpoint × `32` masks=`96` rows）

## Verification

- [x] semantic IDs file/logical hash 与 score shards 一致
- [x] prefix score 在同一 prefix 内一致，maximum spread=`0`（四路各 `10,708` candidates）
- [x] prefix macro recall 保留 Official core75=`200` 个 gold mentions 的权重
- [x] 每个 A800 swap masks=`32`；A800/RTX 两个端点 ranking exact
- [x] Shapley contribution 之和重建 RTX−A800 metric delta；最大守恒误差=`6.94e-18`（R@1/5/10/50、三个 A800、每路 `32` masks）
- [x] `96/96` grid rows 都记录 `deployable=false` 与 `scope=counterfactual_attribution_only`
- [x] Reviewer required? yes；修复 gate 常量复用、mask verification 命名与 hybrid scope 后，无 CRITICAL/HIGH/MEDIUM
- [x] 冻结 Python=`3.10.12`、torch=`2.10.0+cu128` 环境相关测试=`20/20`（fixed-pool=`9`、position=`6`、attribution=`5`）；Ruff、`py_compile`、`git diff --check` 通过

## Result / Notes

所有数字的共同条件：Official core75=`75` queries、gold mentions=`200`、pool=four beam-order Top50 union plus all gold、semantic depth=`4`。prefix R@1 按 gold mention 计算 macro recall；hybrid 只做 counterfactual attribution，不能作为部署配置。

| system | code depth 1 R@1 | depth 2 | depth 3 | depth 4 |
|---|---:|---:|---:|---:|
| RTX archived | `50.1302` | `47.7968` | `48.6857` | `28.3683` |
| workspace control | `50.4635` | `39.7968` | `39.3524` | `20.9238` |
| workspace warmup=`500` | `48.3524` | `44.7968` | `46.5746` | `23.9905` |
| workspace clip=`1.0` | `51.1302` | `43.0190` | `43.9079` | `23.8794` |

depth=`1` 时 RTX 与 control/warmup/clip 的 R@1 差为 `-0.3333/+1.7778/-1.0000` 个点；到 depth=`2` 扩大为 `+8.0000/+3.0000/+4.7778` 个点（上述 prefix 条件）。总体差距从第二个 learned-code decision 开始稳定出现。对最终 RTX-correct/A800-wrong cases，最早出现“RTX prefix Top-1 正确、A800 prefix Top-1 错误”的 depth 分布为：control `{1:3,2:5,3:1}`（`9` queries）、warmup `{1:6,2:1,3:1,4:1}`（`9` queries）、clip `{1:5,2:2,3:1}`（`8` queries）。个别 query 已在 c0 分叉，但总体指标损失在 c1 后明显扩大。

R@1 exact Shapley contribution 如下，单位为百分点。每行五项之和等于 RTX fixed-pool R@1 减对应 A800 fixed-pool R@1；条件为每路全部 `32` 个 counterfactual masks。

| A800 system | c0 | c1 | c2 | c3 | stop | endpoint gap |
|---|---:|---:|---:|---:|---:|---:|
| workspace control | `1.9167` | `3.4167` | `1.3056` | `0.8056` | `0.0000` | `7.4444` |
| workspace warmup=`500` | `0.4889` | `2.3148` | `0.6111` | `0.9630` | `0.0000` | `4.3778` |
| workspace clip=`1.0` | `1.1981` | `2.1759` | `0.2315` | `0.8833` | `0.0000` | `4.4889` |

c1 占三路 endpoint R@1 gap 的约 `45.9%/52.9%/48.5%`（control/warmup/clip、上述 `32`-mask Shapley 条件），是唯一在三路都占最大份额的 component。单独替换 c1 后，control/warmup/clip R@1=`26.0349/25.9905/24.9905`，但 R@10=`66.1111/67.9556/63.9111`；条件为对应 A800 其余四项不变、同一 fixed pool。c1 swap 仍以深召回换 Top-1。

每路只有 mask=`15`（RTX c0/c1/c2/c3、A800 stop）与 mask=`31`（RTX 全五项）通过 fixed-pool R@1/R@10/R@50=`27.3683/69.0000/75.0000` 门槛；两者指标都等于 RTX=`28.3683/69.0000/78.0000`。stop swap 单独不改变任何 R@1/5/10/50。完整恢复需要四个 learned-code components，误差已经沿 prefix 累积，单独修一个 head 不够。

结论：最清晰的训练靶点是第二个 code decision 及其后续累计 prefix ranking。训练改造应同时约束整个 code sequence，优先增加 c1 分叉处的 query-conditioned hard negatives，再用 cumulative-prefix 或 sequence-level listwise loss 保持 R@10/R@50。仅提高 c1 CE 权重仍可能复现当前的 Top-1/深召回交换，因此不作为首选。
