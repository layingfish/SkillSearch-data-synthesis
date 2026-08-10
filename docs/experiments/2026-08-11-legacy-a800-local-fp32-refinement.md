# Experiment: Legacy A800 local FP32 decoder refinement
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

Phase 2 中 decoder FP32 的 `D_param`/step-50 loss-gap reduction=`14.6651%/52.8662%`，heads+CE FP32=`4.9188%/95.8414%`；条件均为 A800、default 对 workspace16、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0。两段信号互补，本轮测试 decoder 半区和 decoder+heads/CE 联合边界。

archived pipeline SHA-256=`7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76` 保持不变。训练 wrapper 只新增四个可组合局部 FP32 region；Phase 2 BF16 control artifacts 作为固定 reference。

## Commands

Train: job=`tide-chenyufei03-10594-pemnv`，queue=`rudder-exper`，nodes=1，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`ec164e4ae1f9072ce84dc54c7208708a3835f449`。节点内运行四组 default/workspace16 pair：decoder blocks 0–5、decoder blocks 6–11、full decoder+heads/CE、decoder blocks 0–5+heads/CE，占用 `8×A800`。

每个 arm 固定 catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、外层 BF16 autocast、AdamW foreach=None。

Eval: 复用 Phase 2 BF16 control 的 `D_param=1.04082965952`、step-50 loss gap=`0.0297207832336`（A800、default 对 workspace16、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0）。候选两项都降低至少 `50%` 才通过。

## Outputs

Artifacts: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_local_fp32_refinement_20260811_v1/`

Node log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_local_fp32_refinement_20260811_v1/logs/node.log`

Bootstrap log: `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/legacy_a800_local_fp32_refinement_20260811_v1.log`

Metrics: 以下结果均为 A800、default 对 workspace16、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、外层 BF16；reduction 相对 Phase 2 BF16 control 计算。

| Region | `D_param` | `D_param` reduction | Step-50 loss gap | Loss-gap reduction | 双门禁 |
|---|---:|---:|---:|---:|---:|
| decoder blocks 0–5 | 0.993864189690 | 0.045123 | 0.00408315658569 | 0.862616 | fail |
| decoder blocks 6–11 | 0.999720881664 | 0.039496 | 0.0393052101135 | -0.322482 | fail |
| full decoder+heads/CE | 1.09434129406 | -0.051412 | 0.0259284973145 | 0.127597 | fail |
| decoder blocks 0–5+heads/CE | 1.02160446214 | 0.018471 | 0.00432968139648 | 0.854321 | fail |

## Verification

- [x] Minimal local check：21 项相关测试在 Python 3.10.12、torch 2.10.0+cu128 的冻结 runtime 下通过；`py_compile`、`bash -n`、trailing-whitespace 与 `git diff --check` 通过
- [x] 四个 region 的 module call count 与 output dtype audit 通过
- [x] 八个 arms 的 source/data/model/runtime/first-batch/init/RNG hash 对齐；四组 pair audit 全部通过
- [x] Phase 2 BF16 control compatibility audit 通过；八个 candidate-to-control audit 全部通过
- [x] Reviewer required? yes；structured local review 未发现训练顺序、autocast 边界、module 选择、跨 job control 或八卡隔离的 CRITICAL/HIGH 问题；GPU provenance 与 dtype audit 仍是硬门禁

## Result / Notes

- Official core75 不在本阶段运行；本轮只做 50-step 局部精度定位。
- job `tide-chenyufei03-10594-pemnv` 在约 5 分 30 秒完成（`8×A800`、八个独立单卡 arm、0 restarts）。
- decoder blocks 0–5 把 step-50 loss gap 降低 `86.2616%`，但 `D_param` 只降低 `4.5123%`（上述固定训练条件）。
- full decoder+heads/CE 的 step-50 loss-gap reduction 只有 `12.7597%`，`D_param` 恶化 `5.1412%`（上述固定训练条件）。联合区域没有产生互补收敛。
- 四个区域都未达到 `D_param` 与 step-50 loss gap 各降低至少 `50%`（上述固定训练条件）。结合 Phase 2 整体 decoder FP32 的 `D_param` reduction=`14.6651%`（相同训练条件），继续细到 attention/FFN 的信息收益低；停止局部 FP32 搜索，转入 optimizer 稳定性实验。
