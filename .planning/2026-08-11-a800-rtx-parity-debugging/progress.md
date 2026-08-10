# A800 / RTX parity 排障进度

## 2026-08-11：计划建立

- **状态：** planned；未修改训练代码，未提交 GPU 作业。
- 已读取现有项目计划、核心 A800/RTX 调查、precision/backend probe、dropout=0 probe、端到端敏感性实验和 RTX 取证包。
- 已把网络调研证据写入独立 `findings.md`；外部资料只作为诊断依据。
- 已建立七阶段计划：证据冻结 → 逐层 trace → 局部 FP32 二分 → 完整训练 → optimizer 放大器 → 三 seed 锁定 → 交付。
- 已明确停止继续 sweep 的全局变量。
- 已设置局部 FP32、完整 Official core75 恢复和多 seed 稳定性门禁。
- 自检后修正局部 FP32 边界语义：模块输出不手动降回 BF16，由外层 autocast 决定后续算子精度。
- 明确 optimizer reset 对照在每个 backend 内从同一 catalog-final model 分叉，避免模型状态与 optimizer 状态同时变化。
- 原根目录 `task_plan.md`、`findings.md`、`progress.md` 属于 RIPOR/PAG 工作流，未修改。

## 文件

- `.planning/2026-08-11-a800-rtx-parity-debugging/task_plan.md`
- `.planning/2026-08-11-a800-rtx-parity-debugging/findings.md`
- `.planning/2026-08-11-a800-rtx-parity-debugging/progress.md`

## 检查

| 检查 | 结果 |
|---|---|
| 现有 worktree 起始状态 | clean |
| 原 RIPOR/PAG planning files | 已读取，未覆盖 |
| 新计划是否包含固定训练合同 | 是 |
| 新计划是否避免重复已完成 backend/runtime sweep | 是 |
| 新计划是否区分诊断、配方选择和最终 test | 是 |
| GPU job | 未启动 |
| `git diff --check` | 通过 |
| worktree 变化 | 仅新增 `.planning/` 独立计划目录 |
| plan attestation | SHA-256 前缀 `1456bbfa83db`，已包含 Phase 1 启动状态与当前错误记录 |

## 2026-08-11：评测选择策略调整

- 用户决定直接使用 Official core75 75-query 选择局部精度与 optimizer 配方，减少 full-run arm 的评测时间。
- Phase 4、Phase 5、Phase 6、完成标准和风险说明已同步修改。
- archived RTX Official reference 直接复用，不新增 canonical dev 1,000-query reference 评测。
- canonical private test 1,000-query 只对最终锁定的 seed=20260626 checkpoint 跑一次，不参与配方回选。
- Official core75 后续统一标记为 selection-set metrics。
- session catchup 只发现本轮计划读取与 attestation 操作，没有遗漏代码或实验状态。

## 下一动作

用户确认计划后，从 Phase 1 的逐层 one-step trace 开始；执行前创建对应 experiment note，并检查 `git status --short`。

## 2026-08-11：收紧最终指标门禁

- 已确认 `workspace_16` 的 Overall R@10/R@50=`69.5556/75.3333` 已超过 RTX reference=`69.0000/75.0000`；条件均为 catalog=3 epochs、query=5 epochs、batch=16、seed=20260626、lr=3e-4、Official core75 75-query、beam=50、beam-order。
- Phase 4 与 Phase 6 的 Overall R@1 门槛改为 `>=27.3683`（RTX=`28.3683`，Official core75 75-query、beam=50、beam-order），即每条候选路径、每个最终 seed 最多低 `1.0` 个点。
- Overall R@10/R@50 改为不回退门槛：分别 `>=69.0000/75.0000`（Official core75 75-query、beam=50、beam-order）。
- Single R@1 门槛设为 `>=41.6667`（24 Single queries、beam=50、beam-order）；该子集每个 query 对应 `4.1667` 个点，无法表达连续的 `1.0`-point 容差。
- Multi R@1 门槛设为 `>=21.1102`（RTX=`22.1102`，51 Multi queries、beam=50、beam-order）。
- 本次只修改排障计划，未修改训练代码，未提交 GPU 作业。
- task plan 已重新 attestation，SHA-256 前缀为 `3018beb045a3`（收紧后的 Phase 4/Phase 6 门禁）。

## 2026-08-11：开始 Phase 1

- 用户授权使用 `rudder-exper`，并要求一个节点上的 `8×A800` 全部投入实验。
- Phase 1 调度改为八个单卡进程：default BF16、`workspace_16` BF16、cuBLASLt BF16、FP32 四种数值路径各跑 `trace=on/off` 配对；所有 arm 使用 catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0。
- 已确认仓库尚无 layerwise trace 脚本；准备复用现有 legacy runtime probe 与 Tianzhi node launcher 的训练合同、hash 审计和单卡隔离方式。
- 已读 `legacy_runtime_matrix_probe.py` 与 `run_controlled_legacy_pipeline.py`。前者可复用 provenance 与固定数据合同，后者可复用 backend 控制；二者都缺少模型级 hook 和 optimizer 后参数捕获。
- 已定位 archived model factory 与单 SID 训练循环。实现方案是动态加载原脚本、包装全局 model factory 与 AdamW.step，再调用原 `main()`；control arm 继续使用现有未插桩 runtime probe。
- 已读现有 precision node/Tianzhi launchers。Phase 1 将复用经验证的 Python 3.11 + torch 2.10.0+cu128 + CUDA 12.8 + Transformers 5.3.0 runtime，以及八个 `CUDA_VISIBLE_DEVICES` 单卡子进程的整节点调度方式。
- 已确认可复用的 state/RNG/hash helpers，并找到现有 legacy probe 测试目录；下一步实现独立 trace wrapper、matrix analyzer、整节点 launcher 与 focused tests。
- 对比 launcher 后发现最新 dropout=0 probe 已切到原始 pipeline 与 Python 3.10 runtime；正在依据最新调查文档确认 Phase 1 的固定 runtime。
- Phase 1 runtime 已确认：Python 3.10.12、torch 2.10.0+cu128、Transformers 5.3.0、Tokenizers 0.22.2、CUDA 12.8、cuDNN 9.10.2，原始 pipeline SHA-256=`7b883cb7...`。
- 已新增 `scripts/legacy_layerwise_trace_probe.py`：包装 archived catalog probe，记录 module activation、CE、每参数 gradient/update、loss、最终 state hash 与完整 provenance。
- 已新增 `scripts/analyze_legacy_layerwise_trace.py`：验证四组 trace/control 的 loss 与最终参数逐 hash 等价，并计算逐层、gradient、update relative L2。
- 已新增 `scripts/run_legacy_layerwise_trace_node_20260811.sh` 与 Tianzhi bootstrap；一个 `8×A800` 节点运行八个独立单卡 arm，目标队列为 `rudder-exper`。
- 新增 Python 脚本的 `py_compile` 与两个 launcher 的 `bash -n` 均通过。
- 本地 `uv run python` 环境缺少 torch，首次 tensor 单测启动报 `ModuleNotFoundError: torch`；下一次尝试改用已冻结的 Python 3.10 runtime site，不安装或替换项目依赖。
- 使用 frozen runtime 的首次命令因带有 `/tmp` 递归清理而被执行器拒绝，尚未发生解包或文件删除；后续使用唯一临时目录且不在命令内清理。
- 第二次本地测试成功解包 Python 3.10 runtime，但因未叠加 NVIDIA libraries 报 `ImportError: libcudnn.so.9`；这同时发现 node launcher 的同类问题。
- node launcher 已改为叠加 SHA-256 固定的 Python 3.11 site archive，并从其 `site/nvidia/*/lib` 构造 `LD_LIBRARY_PATH`；训练解释器与 torch 包仍来自 Python 3.10 runtime。
- 修正后使用冻结的 Python 3.10.12 + torch 2.10.0 runtime 跑完 `tests.test_legacy_layerwise_trace` 与 `tests.test_legacy_runtime_matrix_probe`：共 8 项测试，全部通过。
- 本地测试目录为 `/tmp/skillgr-layer-trace-test.wZCAE5`；其中只包含 frozen runtime 临时解包，不包含数据或实验输出，后续由临时文件系统回收。
- 新增 BF16 tensor hash round-trip 测试及 default dropout=0 archived loss 门禁后，相关 8 项测试再次全部通过（冻结的 Python 3.10.12 + torch 2.10.0 runtime）。
- 已确认提交镜像为项目固定的 `chenyufei03/skillsearch-gr:cu128-torch28-v1`；GPU job 尚未提交。
- 提交前 focused local review 未发现训练顺序、autocast、dropout、optimizer 或数据顺序语义偏移。trace=off 保留 archived probe 路径；trace=on 只增加 forward hooks、CE observer 与 AdamW.step 前后快照。最终结论仍由 GPU 上的精确等价门禁决定。
- 提交前完整相关测试覆盖新 trace、runtime matrix、baseline parity 与 controlled wrapper，共 12 项测试在 frozen Python 3.10.12 + torch 2.10.0 runtime 下通过；`py_compile`、`bash -n`、可用时的 `shellcheck`、trailing-whitespace 检查和 `git diff --cached --check` 通过。
- staged scope 仅包含独立 A800/RTX 计划、Phase 1 experiment note、两个 Python 诊断脚本、两个 launcher 和一份 focused test；未包含数据、checkpoint、runtime archive 或本地日志。
- 尚未完成本地测试、focused review、commit 或 GPU 提交。
- 状态更新为 Phase 1 `in_progress`；GPU job 尚未提交。
