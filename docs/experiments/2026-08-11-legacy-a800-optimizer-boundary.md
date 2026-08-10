# Experiment: Legacy A800 catalog-to-query optimizer boundary
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

测试 catalog 阶段累积的 AdamW moments 是否在 query 阶段放大 default/workspace16 数值路径差异。首轮同时测 query lr=`{2e-4, 3e-4}`，形成 backend × optimizer state × query lr 的 `2×2×2` factorial；同一 lr 内可独立估计 carry/reset 效果。

训练主体继续使用 archived single-SID pipeline SHA-256=`7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76`。wrapper 只在 query step 1 的 AdamW `step()` 前保留或清空 state，并设置 query lr；forward、backward、dropout RNG、batch shuffle 和参数顺序保持原样。

## Commands

Train: v1 job=`tide-chenyufei03-10595-comjs` 因 wrapper 的 catalog row-hash 口径错误失败。v2 job=`tide-chenyufei03-10596-baniu`，launch commit=`f13c2de619824578661275ee7ad8152e466c6171`，因 wrapper 的 query-dev manifest 期望行数错误失败。v3 job=`tide-chenyufei03-10597-gxpdl`，queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`、launch commit=`7e046bf4b85e811c842f8b26960f63cf4b4b9059`。

每个 arm 的 catalog 固定 2,730 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0.1、外层 BF16、AdamW foreach=None。query 固定前 50 updates、batch=16、dropout=0.1；只改变 optimizer state 和 query lr。

Eval: 在 CPU FP64 上计算 `D_update = ||delta_default - delta_workspace||_2 / ||delta_default||_2`，delta 为 query step 50 相对各自 catalog-final model 的参数变化。reset 相对同 lr carry 的 `D_update` 与 step-50 loss gap 都降低至少 `50%` 才进入完整 query 训练。

## Outputs

Artifacts: v3 `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_optimizer_boundary_20260811_v3/`

Node log: v3 `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_optimizer_boundary_20260811_v3/logs/node.log`

Bootstrap log: v3 `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/legacy_a800_optimizer_boundary_20260811_v3.log`

Metrics:

| query lr | optimizer state | `D_update` | `D_update` reduction vs carry | step-50 loss gap | loss-gap reduction vs carry | 双 `50%` 门禁 |
|---:|---|---:|---:|---:|---:|---|
| `2e-4` | carry | `1.77980018117` | `0%` | `0.100789546967` | `0%` | fail |
| `2e-4` | reset | `1.51348570818` | `14.9632%` | `0.120325326920` | `-19.3827%` | fail |
| `3e-4` | carry | `1.74075836224` | `0%` | `0.120148658752` | `0%` | fail |
| `3e-4` | reset | `1.51135051239` | `13.1786%` | `0.113351821899` | `5.6570%` | fail |

表中条件统一为 A800 default 对 `workspace_16`、catalog=`2,730` updates、query=`50` updates、batch=`16`、seed=`20260626`、catalog lr=`3e-4`、query lr 与 optimizer state 按行、weight decay=`0`、dropout=`0.1`、外层 BF16；`D_update` 在 CPU FP64 上计算。

## Verification

- [x] Minimal local check：26 项相关测试在 Python 3.10.12、torch 2.10.0+cu128 的冻结 runtime 下通过；`py_compile`、`bash -n`、trailing-whitespace 与 `git diff --check` 通过
- [x] default catalog step-1/50 loss 命中 frozen A800 reference：`4.546923637390137/4.184666633605957`（A800 default BF16、catalog step=`1/50`、batch=`16`、seed=`20260626`、lr=`3e-4`、dropout=`0.1`）
- [x] catalog/query 首批 row hash 命中取证包 reference：`721a0d93.../56995788...`（catalog/query 各自首批、batch=`16`、seed=`20260626`）
- [x] 每个 backend 内四个 catalog-final model/optimizer/RNG/gradient hash exact（A800、catalog=`2,730` updates、batch=`16`、seed=`20260626`、lr=`3e-4`、dropout=`0.1`）
- [x] reset arm 在 query step 1 update 前 optimizer state entry count=`0`（A800、query step=`1`、reset arms）
- [x] 八个 arms 的 source/data/model/runtime/init provenance 对齐（Python=`3.10.12`、torch=`2.10.0+cu128`、CUDA=`12.8`、cuDNN=`9.10.2`、Transformers=`5.3.0`、Tokenizers=`0.22.2`）
- [x] Reviewer required? yes；structured local review 未发现阶段边界、AdamW reset、RNG、batch、state artifact 或八卡隔离的 CRITICAL/HIGH 问题；GPU 边界 invariant 仍是硬门禁

## Result / Notes

- 本轮不跑 Official core75；50-step query probe 只判断 optimizer state 是否是主要放大器。
- post-run last/final checkpoint save 被 observer 抑制；每个 arm 只保存 catalog-final 与 query-step-50 trainable state，供 `D_update` 精确计算。
- v1 的八个 arm 均完成 catalog=`2,730` updates 与 query=`50` updates（batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`），随后统一失败于 post-run catalog row hash。根因是 wrapper 对 catalog row 误加了 `|positive_skill_id`；取证口径对 catalog 只 hash `index`，对 query 才 hash `index|positive_skill_id`。训练主体未失败，但 v1 未写出 state artifact，不能计算 `D_update`。
- v2 已把 catalog/query row-id 口径显式分开，并把 hash 门禁前移到对应首批，避免再到训练结束才报错。v1 日志完整保留。
- v2 八个 arm 的 catalog/query 首批 hash、observer call counts、optimizer transition、dropout 和 default step-1/50 loss 门禁均通过；随后 manifest 校验把 archived query-dev rows=`2,501` 误写成 `48,750`。v2 未写 state artifact，日志完整保留。v3 将 row-count contract 直接与 archived training manifest 对照测试。
- v3 八个 arm 全部完成，job=`tide-chenyufei03-10597-gxpdl` 使用一个 `8×A800` 节点并行八个独立单卡进程，运行约 `29` 分钟、pod restart=`0`。matrix summary 与全部 state artifact 已写入 v3 输出目录。
- reset 在 query lr=`2e-4` 时只把 `D_update` 降低 `14.9632%`，step-50 loss gap 反而增加 `19.3827%`；在 query lr=`3e-4` 时把 `D_update`/loss gap 分别降低 `13.1786%/5.6570%`。条件均为 A800 default 对 `workspace_16`、catalog=`2,730` updates、query=`50` updates、batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`。
- catalog AdamW moments 只贡献小幅 backend 放大，不能解释当前最终 Top-1 差距。下一步固定 `workspace_16` 跑 optimizer state=`{carry, reset}` × query lr=`{1e-4, 2e-4, 3e-4}` 的六个完整 query arms，并直接用 Official core75 75-query 选择。
