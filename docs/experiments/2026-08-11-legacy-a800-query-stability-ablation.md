# Experiment: Legacy A800 query stability ablation
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

在已验证的 carry/query lr=`3e-4` 配方上，分别测试 query `warmup_steps=500` 与 query `max_grad_norm=1.0` 能否恢复 Top-1，同时保持深召回。两项不组合。catalog 阶段保持 legacy 语义不变。

单节点运行八个独立单卡 arm：backend=`{default, workspace_16}` × intervention=`{control, warmup, clip}` 六路，加 `workspace_16/warmup` 与 `workspace_16/clip` 各一条 exact replica。所有 arm 固定 optimizer state=`carry`、catalog lr=`3e-4`、query base lr=`3e-4`。

warmup 仅作用于 query optimizer step。query step `s` 使用 lr=`3e-4 × min(s/500, 1)`，因此 query step=`1/500/501` 的 lr=`6e-7/3e-4/3e-4`。clip 仅作用于 query backward 后、AdamW step 前，global L2 `max_grad_norm=1.0`、`foreach=True`、非有限 norm 直接失败。

## Commands

Train: job=`tide-chenyufei03-10599-dfvoq`、queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`、launch commit=`06f48f736a82029b922f6685b5ce2e20eb81f025`。

每个 arm：catalog=`2,730` updates、query=`38,805` updates、batch=`16`、eval batch=`16`、seed=`20260626`、catalog lr=`3e-4`、query base lr=`3e-4`、weight decay=`0`、betas=`(0.9,0.999)`、eps=`1e-8`、dropout=`0.1`、外层 BF16、每 `200` 个 query updates 评测 dev loss并选择 best。

Eval: Official core75=`75` queries、beam=`50`、beam-order aggregation、CE log-prob、K=`{1,10,50}`。两个 workspace intervention arms 参与选择；control、default pair 与 replica 仅做效果归因和审计。canonical private test 不参与本轮。

## Outputs

Artifacts: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_query_stability_20260811_v1/`

Node log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_query_stability_20260811_v1/logs/node.log`

Bootstrap log: `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/legacy_a800_query_stability_20260811_v1.log`

Metrics: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_query_stability_20260811_v1/query_stability_summary.json`

## Verification

- [x] Minimal local check：七个 legacy test modules 共 `40` 项在 Python=`3.10.12`、torch=`2.10.0+cu128` frozen runtime 下通过；`py_compile`、`bash -n` 与 `git diff --check` 通过
- [x] archived source/data/model/runtime/init 与 catalog/query 首批 hash 全部命中
- [x] 每个 backend 内的 catalog-final model/optimizer/RNG 与 query step-1 pre-intervention gradient exact
- [x] warmup lr trace 命中 query step=`1/500/501` 的 `6e-7/3e-4/3e-4`
- [x] clipping 调用次数=`38,805`，global L2 threshold=`1.0`，无 NaN/Inf
- [x] workspace warmup 与 clip 的两个 exact replica invariant 通过
- [x] default/workspace control 精确命中 2026-08-09 历史 anchor
- [x] 八个 arm 全部生成 Official core75 指标
- [x] Reviewer required? yes；structured local review 已检查 query step 顺序、warmup lr、clip 时机、catalog invariant、checkpoint selection、历史 artifact hash、replica 和 GPU `0–7` 映射，未发现 CRITICAL/HIGH 问题

## Result / Notes

- 通过门槛：Overall R@1 `>=27.3683`、R@10 `>=69.0000`、R@50 `>=75.0000`（Official core75=`75` queries、beam=`50`、beam-order）；Single R@1 `>=41.6667`（Single=`24` queries）；Multi R@1 `>=21.1102`（Multi=`51` queries）。
- 若 warmup 与 clip 都通过，按 Overall R@1、Single R@1、Multi R@1、Overall R@10、较低 dev loss 的顺序选择。若均失败，报告各自相对 workspace control 与对应 default pair 的变化，再决定是否有依据测试组合或结束 Phase 5。
- live runtime preflight 已命中 Python=`3.10.12`、torch=`2.10.0+cu128`、CUDA=`12.8`、cuDNN=`9.10.2`、Transformers=`5.3.0`、Tokenizers=`0.22.2`、GPU count=`8`；job=`tide-chenyufei03-10599-dfvoq`、restart=`0`。
- 八个 arm 已分别启动在 GPU `0–7`。catalog step=`50` loss 在三条 default arms 均为 `4.184667`，在五条 `workspace_16` arms 均为 `4.171228`；条件为 catalog step=`50/2,730`、batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`、BF16。日志没有训练错误。
- job=`tide-chenyufei03-10599-dfvoq` 已完成。八个 arm 全部生成 checkpoint、Official core75 预测和审计产物；pod restart=`0`。
- 条件统一为 catalog=`2,730` updates、query=`38,805` updates、batch=`16`、seed=`20260626`、catalog/query base lr=`3e-4`、dropout=`0.1`、dev-best checkpoint、Official core75=`75` queries、beam=`50`、beam-order。结果如下：

| arm | Overall R@1/R@10/R@50 | Single R@1 | Multi R@1 |
|---|---:|---:|---:|
| `default_control` | `12.7238/49.5429/65.8889` | `0.0000` | `18.7115` |
| `workspace_16_control` | `20.9238/69.5556/75.3333` | `25.0000` | `19.0056` |
| `default_warmup500` | `21.5905/57.8095/66.8889` | `20.8333` | `21.9468` |
| `workspace_16_warmup500` | `23.9905/64.8444/73.0000` | `33.3333` | `19.5938` |
| `default_clip1` | `16.1016/58.5873/69.0000` | `12.5000` | `17.7965` |
| `workspace_16_clip1` | `23.8794/65.1778/73.3333` | `33.3333` | `19.4304` |

- workspace warmup/clip 相对 workspace control 的 Overall R@1 分别增加 `3.0667/2.9556` 个点，同时 R@10 分别下降 `4.7111/4.3778` 个点、R@50 分别下降 `2.3333/2.0000` 个点；条件为上表统一训练与评测设置。两个 intervention 均未通过完整门槛。
- workspace warmup 与 clip replica 的 final model、best step/dev loss、prediction SHA-256、metrics 和 intervention trace 均 exact。default/workspace control 的 final model、prediction SHA-256、best step/dev loss 和 metrics 精确命中历史 anchor。
- clip 在 query step=`1/50/500/38,805` 的 pre-clip global L2 norm=`1.000335/1.144790/1.755364/2.277553`（`workspace_16_clip1`、query max grad norm=`1.0`、batch=`16`、seed=`20260626`）；四个采样点都实际触发 clipping。
