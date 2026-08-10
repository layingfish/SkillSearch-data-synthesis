# Experiment: Legacy A800 local FP32 region probe
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

验证 encoder、decoder、heads/CE 三个区域中，哪个局部 FP32 设置能压缩 A800 default 与 `CUBLAS_WORKSPACE_CONFIG=:16:8` 的 50-step 参数轨迹和 loss 差异。

Phase 1 已证明两条 BF16 路径的 embedding 与 encoder outputs tensor-exact，首个 activation 分叉位于 `decoder.block.0`。因此 decoder FP32 是主候选；encoder FP32 与 heads/CE FP32 保留为高层二分对照。Phase 1 dtype trace 显示 SID embedding 输出为 FP32，四个 code head 与 control head 输出为 BF16，CE 接收 BF16 logits 并输出 FP32；`heads_ce` arm 主要测试 head GEMM、logit 量化和 FP32 logits 上的 CE。

变更只新增独立 precision wrapper、matrix analyzer 和 launchers。archived `catalog_50_probe.py` 与原始 pipeline SHA-256=`7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76` 保持不变。

## Commands

Train: job=`tide-chenyufei03-10593-vgfrd`，queue=`rudder-exper`，nodes=1，image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`，launch commit=`544c44e6d44b5ebe4dbdfa46d161dfd791afd196`。节点内运行八个单卡 arm：default/workspace16 × `{BF16 control, encoder FP32, decoder FP32, heads+CE FP32}`，占用 `8×A800`。

每个 arm 固定 catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、外层 BF16 autocast、AdamW foreach=None。

Eval: 计算每个局部区域的 step-50 loss gap，以及 `D_param = ||theta_default - theta_workspace||_2 / ||theta_default - theta_init||_2`。两者相对 BF16 control 均降低至少 50% 才进入 Phase 3。

## Outputs

Artifacts: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_local_fp32_regions_20260811_v1/`

Node log: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_local_fp32_regions_20260811_v1/logs/node.log`

Bootstrap log: `/models_rw/SkillSearch-data-synthesis/bootstrap_logs/legacy_a800_local_fp32_regions_20260811_v1.log`

Metrics: 以下结果均为 A800、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、外层 BF16；每行只改变局部 FP32 区域，列内比较 default 与 `CUBLAS_WORKSPACE_CONFIG=:16:8`。

| Region | `D_param` | `D_param` reduction | Step-50 loss gap | Loss-gap reduction | 双门禁 |
|---|---:|---:|---:|---:|---:|
| none | 1.04082965952 | 0.000000 | 0.0297207832336 | 0.000000 | fail |
| encoder | 1.11710764746 | -0.073286 | 0.0211133956909 | 0.289608 | fail |
| decoder | 0.888190704721 | 0.146651 | 0.0140085220337 | 0.528662 | fail |
| heads+CE | 0.989632963297 | 0.049188 | 0.00123596191406 | 0.958414 | fail |

## Verification

- [x] Minimal local check：19 项相关测试在 Python 3.10.12、torch 2.10.0+cu128 的冻结 runtime 下通过；`py_compile`、`bash -n`、`git diff --check` 通过
- [x] precision gate off 的 CPU focused test 保持输出 exact；GPU default control 命中 archived step-1 loss
- [x] default BF16 control 精确复现 archived dropout=0 step-1 loss=`4.554369926452637`（A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、local FP32 off）
- [x] 八个 arms 的 source/data/model/runtime/first-batch/init/RNG hash 对齐；四组 pair audit 全部通过
- [x] 每个局部 FP32 arm 记录真实 input/output dtype 和 peak GPU memory；最大 `max_allocated_bytes=50,645,659,648`（A800 default、encoder FP32、catalog 前 50 updates、batch=16）
- [x] Reviewer required? yes；已完成 structured local review，未发现训练顺序、数据、autocast、optimizer 或参数距离语义偏移；GPU invariants 仍是硬门禁

## Result / Notes

- Official core75 不在本阶段运行；50-step probe 只用于局部精度定位。
- job `tide-chenyufei03-10593-vgfrd` 在约 5 分 56 秒完成（`8×A800`、八个独立单卡 arm、0 restarts）。
- decoder FP32 在 step 1 把 default/workspace16 loss gap 降为 `0`，到 step 50 仍只把 `D_param` 降低 `14.6651%`（上述固定训练条件）。
- heads+CE FP32 把 step-50 loss gap 降低 `95.8414%`，但 `D_param` 只降低 `4.9188%`（上述固定训练条件）。loss 接近没有转化为参数轨迹收敛。
- 没有区域同时达到 `D_param` 与 step-50 loss gap 各降低至少 `50%`（上述固定训练条件）。下一轮保留 decoder 为参数轨迹排名最高的区域，并增加 decoder+heads/CE 联合边界，检验敏感性是否跨两段分布。
