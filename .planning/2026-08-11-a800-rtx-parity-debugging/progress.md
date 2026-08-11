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
| plan attestation | SHA-256 前缀 `c0fcfc9e618d`，Phase 1 已完成并进入 Phase 2 |

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
- Phase 1 实现已提交为 commit `ed1527ad8e1893b41998638df9f471d8b7e2f2d2`；提交前 worktree 仅含上述 scoped files。
- `submit.sh` 不支持 `--help`，一次只读帮助尝试返回 unknown option；读取脚本 usage 后完成 dry-run，resolved request 为 queue=`rudder-exper`、nodes=1、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`。
- GPU job 已提交：`tide-chenyufei03-10589-irhff`，一个 `8×A800` 节点，launch commit=`ed1527ad8e1893b41998638df9f471d8b7e2f2d2`。状态待启动。
- `tide-chenyufei03-10589-irhff-worker-0` 在提交后约 34 秒进入 Running（1/1 ready、0 restarts）；bootstrap log 已在 CFS 创建，arm 输出目录尚未出现。
- 集群 runtime preflight 通过：Python 3.10.12、torch 2.10.0+cu128、CUDA 12.8、cuDNN 91002、Transformers 5.3.0、Tokenizers 0.22.2，节点可见 8 张 GPU。
- GPU 0–7 已分别启动 default BF16、workspace16 BF16、cuBLASLt BF16、FP32 的 control/trace 八个单卡进程；pod 保持 Running、0 restarts。
- 九份 node/arm log 已在 CFS 创建。首次 arm 日志检查仅见非阻塞 Hugging Face deprecation warning，尚未输出 probe header；进程仍在 archived 全量 source-length audit/模型初始化阶段。
- 运行约 2 分 41 秒时 pod 仍为 Running（1/1 ready、0 restarts）；node log 尚无 arm 完成事件。
- default BF16 control 已完成并精确复现 archived dropout=0 step-1 loss=`4.554369926452637`（A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、trace=off）；initial fingerprint 与 first-batch hash 同时通过。
- 同路径 traced arm 已完成 provenance/header 与 first-batch 阶段，正在执行 forward/backward 和 tensor artifact 写入。
- 四个 control arm 已全部完成；workspace16 BF16、cuBLASLt BF16、FP32 的 step-1 loss 分别为 `4.554419040679932/4.554467678070068/4.536993980407715`（A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、trace=off，各自 backend/precision 如名称）。
- 当前九份日志未发现 Traceback、OOM、磁盘不足或 arm failure；四个 traced arm 继续写 tensor artifacts。
- 运行约 5 分 12 秒时四个 control JSON 已落盘；四个 traced JSON 尚未落盘，pod 仍为 Running（1/1 ready、0 restarts），日志仍无错误。trace 进程在逐参数 hash/save 阶段不会持续打印进度。
- Phase 1 v1 四个 traced arm 在完成 step-1 后均失败于同一 post-run validation：`trace missed expected boundaries: ['shared_embedding']`。没有 OOM、NaN 或训练失败；各 traced arm 的 loss 已正常输出。
- 根因是 hook 注册在未被 encoder forward 调用的 `backbone.shared` module object；已改为 `backbone.encoder.embed_tokens`，并新增专门的 CPU 回归测试。
- retry 使用新的 output/bootstrap 目录 `legacy_a800_layerwise_trace_20260811_v2`，保留 v1 全部日志与四个 control JSON。
- hook 修复后的相关测试增至 13 项，并在 frozen Python 3.10.12 + torch 2.10.0 runtime 下全部通过；`py_compile`、`bash -n`、`git diff --check` 同时通过。
- v1 pod 已退出并被集群清理；无需取消。v2 尚未提交。
- hook 修复提交为 commit `11fd64fa8952f0fda9dacbef5ca40bd159392ecc`。
- v2 dry-run 通过后已提交：job=`tide-chenyufei03-10590-tdbdv`、queue=`rudder-exper`、nodes=1、`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`、launch commit=`11fd64fa8952f0fda9dacbef5ca40bd159392ecc`。
- v2 提交后前两次 pod 查询尚未返回 worker；当前处于 Tianzhi 创建/调度阶段，没有失败 pod。
- v2 worker 在约 32 秒内进入 Running（1/1 ready、0 restarts）；CFS aliases 与 launch commit=`11fd64fa8952f0fda9dacbef5ca40bd159392ecc` 校验通过，正在解包 frozen runtime。
- v2 runtime preflight 再次通过 Python 3.10.12、torch 2.10.0+cu128、CUDA 12.8、cuDNN 91002、Transformers 5.3.0、Tokenizers 0.22.2 与 node GPU count=8；正在完成数据/model hash preflight。
- v2 的八个单卡 arm 已全部启动在 GPU 0–7；pod 运行约 2 分 9 秒时为 Running（1/1 ready、0 restarts）。
- v2 运行约 3 分 13 秒时仍为 Running（1/1 ready、0 restarts）；各 arm 尚未完成 source-length audit/step-1，日志无 Traceback、OOM 或磁盘错误。
- v2 四个 control arms 已在约 4 分 15 秒内完成；四组 step-1 loss 与 final-state SHA-256 均和 v1 byte-identical（A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、trace=off，各自 backend/precision 如名称）。
- v2 运行约 5 分 25 秒时四个 traced arms 仍在逐参数 artifact 阶段；pod 无 restart，日志未出现新 Traceback/OOM/磁盘错误。
- v2 default traced arm 已输出 step-1 loss=`4.554369926452637`（A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、default backend、trace=on），与 control 精确一致；未再出现 missing-boundary error，当前在把 raw tensors 从 node `/tmp` 复制到 CFS。
- v2 八个 arm 全部完成；每个 traced arm 保存 573 个 artifacts，matrix analyzer 报 `pair_equivalence_passed=true`。job `tide-chenyufei03-10590-tdbdv` 在约 8 分 47 秒完成，0 restarts。
- Phase 1 结论：default/workspace16/cuBLASLt 的 embedding 与 encoder 输出 exact；首个 BF16 backend 分叉统一位于 `decoder.block.0`，并在 decoder blocks 1–2 放大。完整数值已写入 findings 与 experiment note。
- Phase 1 标记 done；Phase 2 标记 in_progress，继续实施 default/workspace16 × `{BF16 control, encoder FP32, decoder FP32, heads+CE FP32}` 的八个 50-step arm。
- 已创建 Phase 2 experiment note。
- 已实现 `legacy_local_fp32_probe.py`、matrix analyzer、八卡 node launcher、Tianzhi bootstrap 和 6 项 focused unit tests。
- Phase 2 相关 19 项测试在冻结的 Python 3.10.12 + torch 2.10.0 runtime 下全部通过；`py_compile`、`bash -n`、trailing-whitespace 检查和 `git diff --check` 通过。
- Phase 1 dtype audit 确认 `heads_ce` arm 仍覆盖 BF16 head outputs 与 BF16 CE logits；SID embedding 已是 FP32，因此该子模块的局部包装预期为 no-op。
- structured local review 已逐项检查 model factory 注入点、autocast 边界、CE patch 时机、50-step 调用次数、初始化/最终参数捕获、default/workspace provenance、CPU FP64 距离和八卡进程隔离。未发现 CRITICAL/HIGH 问题。
- analyzer 对零 control metric 改为明确失败，避免把未定义的 reduction 写成非标准 JSON 数值；新增正负边界断言后，19 项相关测试再次全部通过（冻结的 Python 3.10.12 + torch 2.10.0 runtime）。
- Phase 2 实现已提交为 commit `544c44e6d44b5ebe4dbdfa46d161dfd791afd196`；提交后 worktree clean，目标 v1 output 在提交作业前不存在。
- Tianzhi dry-run 确认 queue=`rudder-exper`、nodes=1、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`、launch commit=`544c44e6d44b5ebe4dbdfa46d161dfd791afd196`。
- Phase 2 GPU job 已提交：`tide-chenyufei03-10593-vgfrd`。节点内八个单卡 arm 映射到 GPU 0–7；状态待调度。
- worker `tide-chenyufei03-10593-vgfrd-worker-0` 已进入 Running（1/1 ready、0 restarts）；CFS aliases 与 launch commit=`544c44e6d44b5ebe4dbdfa46d161dfd791afd196` 校验通过，正在解包冻结 runtime。
- Phase 2 runtime preflight 通过：Python 3.10.12、torch 2.10.0+cu128、CUDA 12.8、cuDNN 91002、Transformers 5.3.0、Tokenizers 0.22.2、node GPU count=8。
- default/workspace16 × `{none, encoder, decoder, heads_ce}` 八个单卡 arm 已分别启动在 GPU 0–7；pod 保持 Running（1/1 ready、0 restarts）。
- 运行约 2 分 33 秒时九份 node/arm log 均已创建；尚未输出 probe header，说明各 arm 仍在 archived source-length audit/模型初始化阶段。日志未发现 Traceback、OOM、CUDA error 或 arm failure。
- 八个 arm 均已通过 probe header 与 first-batch 阶段并完成 step 1。default/workspace16 的 step-1 loss gap 为：BF16 control=`4.911423e-05`、encoder FP32=`7.295609e-05`、decoder FP32=`0`、heads+CE FP32=`4.339218e-05`；条件均为 A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0，仅局部 FP32 区域和 workspace path 按名称变化。
- decoder FP32 已在 step 1 消除两条 backend 路径的 loss gap；最终选择仍等待 step-50 loss gap 与 CPU FP64 `D_param` 双门禁。
- 八个 arm 均完成 50 updates 并保存 final trainable state、dtype audit、peak memory 与完整 provenance；matrix analyzer 成功生成 summary 和 `matrix.done`。
- Phase 2 双门禁无人通过：decoder FP32 的 `D_param`/loss-gap reduction=`14.6651%/52.8662%`，heads+CE FP32=`4.9188%/95.8414%`，encoder FP32=`-7.3286%/28.9608%`；条件均为 A800、default 对 workspace16、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0。
- job `tide-chenyufei03-10593-vgfrd` 在约 5 分 56 秒完成（1 个 `8×A800` 节点、八个独立单卡 arm、0 restarts）。Phase 2 标记 done，准备进入 Phase 3。
- Phase 3 第一轮固定四组 default/workspace16 pair：decoder blocks 0–5、decoder blocks 6–11、full decoder+heads/CE、decoder blocks 0–5+heads/CE；Phase 2 BF16 control 作为 reference，不重复占卡。
- task plan 已重新 attestation，SHA-256 前缀=`0b3a0b19122d`（Phase 2 done、Phase 3 in_progress）。
- 已扩展 local FP32 controller：支持 decoder block 半区与 heads/CE 组合，并把预期 module/CE 调用次数改为 region 驱动的统一验证。
- 已新增 Phase 3 analyzer、八卡 node launcher、Tianzhi bootstrap、experiment note 和 composed-region/cross-job-audit focused tests。
- Phase 3 相关 21 项测试在冻结的 Python 3.10.12 + torch 2.10.0 runtime 下全部通过；`py_compile`、`bash -n`、trailing-whitespace 与 `git diff --check` 通过。
- structured local review 已检查 block 选择、嵌套 autocast、CE patch、每步调用次数、跨 job stable-runtime/hash contract、Phase 2 control 重算和 GPU 0–7 映射。未发现 CRITICAL/HIGH 问题。
- Phase 3 实现提交为 commit `ec164e4ae1f9072ce84dc54c7208708a3835f449`；提交后 worktree clean，目标 v1 output 在作业提交前不存在。
- Tianzhi dry-run 通过后已提交 job=`tide-chenyufei03-10594-pemnv`（queue=`rudder-exper`、nodes=1、`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`、launch commit=`ec164e4ae1f9072ce84dc54c7208708a3835f449`）。
- worker `tide-chenyufei03-10594-pemnv-worker-0` 已进入 Running（1/1 ready、0 restarts）；CFS aliases、launch commit=`ec164e4ae1f9072ce84dc54c7208708a3835f449` 与 Phase 2 control root 校验通过，正在解包冻结 runtime。
- Phase 3 runtime preflight 通过：Python 3.10.12、torch 2.10.0+cu128、CUDA 12.8、cuDNN 91002、Transformers 5.3.0、Tokenizers 0.22.2、node GPU count=8。
- 四组 default/workspace16 pair 已分别启动在 GPU 0–7；pod 保持 Running（1/1 ready、0 restarts），当前日志只有非阻塞 Hugging Face deprecation warning。
- 运行约 2 分 50 秒时九份 node/arm log 均已创建；各 arm 仍在 archived source-length audit/模型初始化阶段，日志未发现 Traceback、OOM、CUDA error 或 arm failure。
- 八个 arm 均完成 step 50。decoder blocks 0–5、blocks 6–11、full decoder+heads/CE、blocks 0–5+heads/CE 的 step-50 loss gap 分别为 `0.00408315658569/0.0393052101135/0.0259284973145/0.00432968139648`；条件均为 A800、default 对 workspace16、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0。
- decoder blocks 0–5 的 loss-gap reduction 约为 `86.26%`（相对 Phase 2 BF16 control=`0.0297207832336`，上述固定条件），当前等待 final state 保存和 CPU FP64 `D_param` analyzer。
- Phase 3 analyzer 完成；四组 `D_param` reduction 分别为 `4.5123%/3.9496%/-5.1412%/1.8471%`，没有区域通过双门禁。条件均为 A800、default 对 workspace16、catalog 前 50 updates、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0。
- 四组 candidate pair audit、八个 Phase 2 control compatibility audit、module call-count 与 dtype audit 全部通过；没有 OOM/NaN/Inf。
- job `tide-chenyufei03-10594-pemnv` 在约 5 分 30 秒完成（1 个 `8×A800` 节点、八个独立单卡 arm、0 restarts）。局部 FP32 路线停止，下一步进入 catalog→query optimizer 稳定性实验。
- Phase 3 标记 done；Phase 4 因没有通过短 probe 的局部 FP32 候选而 skipped；Phase 5 标记 in_progress。
- Phase 5 首轮改为 backend × optimizer state × query lr 的 `2×2×2` factorial，共八个单卡 arm，占满一个 `8×A800` 节点；同一 lr 内仍可独立估计 carry/reset 效果。
- task plan 已重新 attestation，SHA-256 前缀=`f4f63191bcdb`（Phase 5 in_progress）。
- 已定位 archived optimizer 生命周期：单个 AdamW 跨 catalog/query 复用。Phase 5 准备通过 optimizer-step wrapper 在 query step 1 前捕获 catalog-final 状态，并对 reset arm 清空 moments；不复制 archived 训练循环。
- 已实现 optimizer-boundary probe、CPU FP64 `D_update` analyzer、八卡 node launcher、Tianzhi bootstrap、experiment note 和 focused tests；尚未完成本地验证、review、commit 或 GPU 提交。
- Phase 5 首次本地测试的 25 项中 24 项通过；optimizer state 的 scalar `step` tensor 在 byte-view hash 中触发 `RuntimeError`。hash helper 已先 reshape 为一维再 view，训练语义未受影响，准备重跑。
- scalar tensor hash 修复后，Phase 5 相关测试共 25 项并在冻结的 Python 3.10.12 + torch 2.10.0 runtime 下全部通过；`py_compile`、`bash -n`、trailing-whitespace 与 `git diff --check` 通过。
- structured local review 已检查 catalog/query call-count 边界、query 首批前 catalog capture、query step-1 backward 后 reset 时机、fresh-AdamW 等价、每 backend 四路 fork invariant、CPU FP64 `D_update`、checkpoint-save observer 与 GPU 0–7 映射。未发现 CRITICAL/HIGH 问题。
- Phase 5 实现提交为 commit `b0786627d891c22b1d516b1999553a0a0283429a`；提交后 worktree clean，目标 v1 output 在作业提交前不存在。
- Tianzhi dry-run 通过后已提交 job=`tide-chenyufei03-10595-comjs`（queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`、launch commit=`b0786627d891c22b1d516b1999553a0a0283429a`）。
- Phase 5 runtime preflight 通过：Python 3.10.12、torch 2.10.0+cu128、CUDA 12.8、cuDNN 9.10.2、Transformers 5.3.0、Tokenizers 0.22.2、node GPU count=`8`；八个单卡 arm 已分别启动在 GPU 0–7。
- 八个 arm 的 query train、query dev、catalog source-length audit 均通过；运行约 5 分 58 秒时同步完成 catalog step `250/2730`（batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`），pod 为 Running、0 restarts，日志未发现训练错误。
- v1 八个 arm 均完成 catalog=`2,730` updates 与 query=`50` updates（batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`），但在 post-run validation 统一失败于 catalog first-batch row hash。pod 全程 0 restarts，无 OOM/NaN/CUDA error。
- 根因是新 wrapper 的 `row_identifier()` 对实际 catalog row 中存在的 `positive_skill_id` 也拼接了 `|positive_skill_id`，得到 `95cb153d...`；取证包 catalog hash=`721a0d93...` 只使用 `index`。query reference 才使用 `index|positive_skill_id`。
- 修复已让 catalog/query 显式选择 row-id 口径，并把 hash 比较前移到对应首批。focused optimizer-boundary tests 共 5 项在冻结的 Python 3.10.12 + torch 2.10.0 runtime 下通过；v2 使用新 output/bootstrap 目录，保留 v1 日志。
- 修复后的五个 legacy probe test modules 共 25 项测试在冻结的 Python 3.10.12 + torch 2.10.0 runtime 下全部通过；`py_compile`、`bash -n` 与 `git diff --check` 通过。
- row-hash 口径修复与 v2 launcher 提交为 commit `f13c2de619824578661275ee7ad8152e466c6171`。Tianzhi dry-run 通过后已提交 v2 job=`tide-chenyufei03-10596-baniu`（queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`）。
- v2 worker 已进入 Running（1/1 ready、0 restarts）；冻结 runtime 与八卡 preflight 通过。八个 arm 的 catalog 首批 row hash 前置门禁全部通过，并同步完成 catalog step `200/2730`（batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`）。
- v2 四个 default arm 完成 catalog=`2,730` updates 与 query=`50` updates（batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`），并通过 catalog/query 首批 hash、observer、optimizer transition、dropout 和 frozen default loss 门禁；随后统一失败于 manifest 的 query-dev row count。
- 第二个根因是 wrapper 误写 expected query-dev rows=`48,750`；archived training manifest、exact-runtime dev JSONL 与 source-length audit 均为 `2,501` rows。训练主体未失败，但 v2 仍未落 state artifact。
- manifest row-count contract 已集中为常量并新增与 archived training manifest 的回归测试；v3 使用新 output/bootstrap 目录，保留 v1/v2 日志。
- v2 八个 arm 最终均只命中同一个 query-dev manifest 错误；job 全程 0 restarts，无训练异常。修复后的五个 legacy probe test modules 共 26 项测试在冻结的 Python 3.10.12 + torch 2.10.0 runtime 下全部通过；`py_compile`、`bash -n` 与 `git diff --check` 通过。
- manifest contract 修复与 v3 launcher 提交为 commit `7e046bf4b85e811c842f8b26960f63cf4b4b9059`。Tianzhi dry-run 通过后已提交 v3 job=`tide-chenyufei03-10597-gxpdl`（queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`）。

## 2026-08-11：Phase 5 optimizer-boundary 结果

- v3 八个 arm 全部完成；job=`tide-chenyufei03-10597-gxpdl` 使用一个 `8×A800` 节点并行八个独立单卡进程，运行约 `29` 分钟、pod restart=`0`。
- source/data/model/runtime/init、catalog-final model/optimizer/RNG、catalog/query 首批、query step-1 gradient/loss 与 reset transition 审计全部通过。default catalog step-1/50 loss=`4.546923637390137/4.184666633605957`（A800 default BF16、catalog step=`1/50`、batch=`16`、seed=`20260626`、lr=`3e-4`、dropout=`0.1`），精确命中 frozen reference。
- query lr=`2e-4` 时，reset 相对 carry 的 `D_update` reduction=`14.9632%`，step-50 backend loss-gap reduction=`-19.3827%`；query lr=`3e-4` 时，对应值为 `13.1786%/5.6570%`。条件均为 A800 default 对 `workspace_16`、catalog=`2,730` updates、query=`50` updates、batch=`16`、seed=`20260626`、catalog lr=`3e-4`、weight decay=`0`、dropout=`0.1`、外层 BF16。
- reset 没有通过双 `50%` 门禁。catalog AdamW moments 只贡献小幅 backend 放大，Phase 5 转入固定 `workspace_16` 的完整 query 限定 sweep。
- 完整 sweep 的六个 selection arms 为 optimizer state=`{carry, reset}` × query lr=`{1e-4,2e-4,3e-4}`；每个 arm 跑 catalog=`2,730` updates、query=`38,805` updates、batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`，并用 Official core75=`75` queries、beam=`50`、beam-order 选择。
- 同节点另外两张 A800 分别运行 `workspace_16/carry/query lr=3e-4` exact replica 与 `default/carry/query lr=3e-4` legacy anchor，只做确定性和历史结果对齐。
- task plan 已重新 attestation，SHA-256 前缀=`14ebdfad4683`（Phase 5 full-query sweep 设计冻结）。

## 2026-08-11：Phase 5 full-query sweep 实现

- 已新增 `legacy_full_query_optimizer_lr.py`。它动态加载 archived pipeline SHA-256=`7b883cb7...`，在 query step `1` 的 AdamW update 前设置 lr 并按 arm carry/reset state，不复制训练循环。
- 已新增 Official core75 arm/matrix analyzer。六个 selection arms 固定为 `workspace_16` × optimizer state=`{carry,reset}` × query lr=`{1e-4,2e-4,3e-4}`；另外两条为 workspace exact replica 与 default legacy anchor。
- 已新增八卡 node launcher 与 Tianzhi bootstrap。GPU `0–7` 各运行一个独立单卡进程；每个 arm 固定 catalog=`2,730` updates、query=`38,805` updates、batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`、eval steps=`200`。
- 为减少八个 full run 的 CFS 写入，best/final checkpoint 只保存模型，last checkpoint 写盘被抑制。训练中的 best selection、forward/backward、optimizer step、dev loss 与 RNG 不变；final state 必须与 best state exact。
- 模型 safetensors view 改放在持久化 experiment output 的 `controls/` 下，避免 checkpoint metadata 指向 job 结束后消失的 `/tmp` 路径。
- archived parser contract test 发现原始 single-SID 命令不接受 `--eval-batch-size`；该无效参数已在提交 GPU 作业前移除。dev evaluation 实际沿用训练 batch=`16`。
- 六个 legacy probe test modules 共 `34` 项将在 frozen Python=`3.10.12`、torch=`2.10.0+cu128` runtime 下做最终回归；focused test 当前 `8/8` 通过，`bash -n` 通过，本机没有 `shellcheck`。
- structured local review 已覆盖 query transition、checkpoint load/save、dev call cadence、catalog invariant、replica/historical anchor 和八卡映射，未发现 CRITICAL/HIGH 问题。
- full-query sweep 实现提交为 commit `e7f0f0f949c13d5ce122fd2a7cfc1e137b5cbe05`；提交后 worktree clean。
- Tianzhi dry-run 确认 queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1` 与完整 launch commit。
- 正式作业已提交：job=`tide-chenyufei03-10598-otasq`。输出目录为 `/models_rw/SkillSearch-data-synthesis/experiments/legacy_a800_full_query_optimizer_lr_20260811_v1/`，当前等待 worker 创建。

## 2026-08-11：Phase 5 full-query sweep 结果

- job=`tide-chenyufei03-10598-otasq` 在 queue=`rudder-exper` 使用一个 `8×A800` 节点完成，pod runtime=`3h10m`、restart=`0`；八个 arm 全部生成 checkpoint、Official core75 预测与结果摘要。
- 七路 `workspace_16` catalog invariant 全部通过；carry/query lr=`3e-4` 与 exact replica 的 final model hash、best step/dev loss、predictions hash、metrics 全部 exact；default/workspace 两条历史 anchor 的 best step/dev loss 与 R@1/10/50 全部 exact。
- 六个 selection arms 均未过门槛。条件统一为 A800 `workspace_16`、catalog=`2,730` updates、query=`38,805` updates、batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`、BF16、dev-best checkpoint、Official core75=`75` queries、beam=`50`、beam-order。
- carry/query lr=`2e-4` 得到最高 Overall R@1=`25.4794`，但 R@10/R@50=`64.2889/68.7778`；carry/query lr=`3e-4` 是唯一同时达到 R@10/R@50=`69.5556/75.3333` 的 arm，但 R@1=`20.9238`。
- reset/query lr=`{1e-4,2e-4,3e-4}` 的 Overall R@1=`22.5460/24.3238/21.9238`；相同 lr 的 carry 为 `17.8794/25.4794/20.9238`。reset 没有形成一致的 Top-1 改善方向；条件为上条统一条件。
- Phase 5 继续在 carry/query lr=`3e-4` 上分别测试 query `warmup_steps=500` 与 query `max_grad_norm=1.0`。下一节点使用 backend=`{default,workspace_16}` × intervention=`{control,warmup,clip}` 六路，加 workspace warmup/clip 各一个 exact replica，占满 `8×A800`。
- task plan 已重新 attestation，SHA-256 前缀=`0953d3bd5727`（Phase 5 stability ablation 设计冻结）。

## 2026-08-11：Phase 5 query stability ablation 实现

- full-query wrapper 已增加两个互斥的 query-only 控制：`warmup_steps=500` 使用 lr=`3e-4 × min(query_step/500,1)`；`max_grad_norm=1.0` 在 backward 后、AdamW step 前执行 global L2 clipping，固定 `foreach=True`、非有限 norm 直接失败。catalog optimizer step 不进入两条控制路径。
- clipping 每个 query update 调用一次，只在 query step=`{1,50,500,38805}` 把 pre-clip norm 同步到 CPU；条件为 query=`38,805` updates、batch=`16`。这保留审计证据，同时避免每步额外 CPU/GPU 同步。
- 八卡矩阵固定为 backend=`{default,workspace_16}` × intervention=`{control,warmup,clip}` 六路，加 workspace warmup/clip 各一个 exact replica；GPU `0–7` 各运行一个独立单卡进程。
- analyzer 将分别审计 default 三路和 workspace 五路 catalog invariant、两组 intervention replica、两条历史 final-model/predictions hash，并只用 workspace warmup/clip 做 Official core75 选择。
- 七个 legacy test modules 共 `40` 项在 Python=`3.10.12`、torch=`2.10.0+cu128` frozen runtime 下全部通过；`py_compile`、`bash -n` 与 `git diff --check` 通过。
- structured local review 已覆盖 query step-1 pre-intervention gradient、warmup step=`1/500/501` lr、global gradient 集合、clip 调用次数、load-best checkpoint、Official core75 评测和八卡映射，未发现 CRITICAL/HIGH 问题。
- task plan 已重新 attestation，SHA-256 前缀=`63b87ab19d4c`（clipping 固定为 `foreach=True`）。
- query stability 实现提交为 commit `06f48f736a82029b922f6685b5ce2e20eb81f025`；提交后 worktree clean。
- Tianzhi dry-run 确认 queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1` 与完整 launch commit。正式 job=`tide-chenyufei03-10599-dfvoq` 已提交，初始 pod 状态=`Init:0/1`、restart=`0`。
- live runtime preflight 命中 Python=`3.10.12`、torch=`2.10.0+cu128`、CUDA=`12.8`、cuDNN=`9.10.2`、Transformers=`5.3.0`、Tokenizers=`0.22.2`、GPU count=`8`；八个 arm 已分别在 GPU `0–7` 启动。
- 八路均到 catalog step=`50/2,730`（batch=`16`、seed=`20260626`、catalog lr=`3e-4`、dropout=`0.1`、BF16）。default 三路 loss 均为 `4.184667`，workspace 五路均为 `4.171228`；pod restart=`0`，日志无 Traceback/OOM/runtime error。

## 2026-08-11：Phase 5 query stability 与 query-rank 结果

- job=`tide-chenyufei03-10599-dfvoq` 已完成；八个 arm、两组 exact replica、两条历史 anchor、catalog/intervention audit 全部通过，pod restart=`0`。
- workspace control/warmup/clip 的 Overall R@1/R@10/R@50=`20.9238/69.5556/75.3333`、`23.9905/64.8444/73.0000`、`23.8794/65.1778/73.3333`；条件为 catalog=`2,730` updates、query=`38,805` updates、batch=`16`、seed=`20260626`、catalog/query base lr=`3e-4`、dropout=`0.1`、dev-best checkpoint、Official core75=`75` queries、beam=`50`、beam-order。两个 intervention 都未通过门槛。
- 新增 `scripts/analyze_legacy_core75_query_ranks.py` 与三项 focused unit tests。四个系统的 `75` 条 prediction exact 对齐；Top50 唯一性、A800 score 完整性/单调性、独立 metric 重算和所有守恒检查通过。
- RTX 与 workspace control/warmup/clip 的平均 Top50 共享率=`24.00%/26.35%/27.01%`；workspace 三路两两共享率=`30.53%–33.23%`。条件为逐 query 固定长度 Top50 集合交集比例后取 `75` 条均值。
- 三路 A800 共同丢失的 RTX-correct query index=`{8,22,35,54,66}`。index=`8` 在三路 A800 Top50 都没有 gold；index=`54/66` 在三路都有 gold，但 top-1 score margin 仍为错误方向。
- 下一动作：对四路 Top50 union 加全部 gold 构造逐 query fixed pool，在同一 pool 上穷举四个 checkpoint 的 CE log-prob。一个 `8×A800` 节点按四个 checkpoint × 两个 query shard 并行，区分 search omission 与 checkpoint scoring failure。
- fixed-pool protocol 已冻结：learned code depth=`4`、stop control=`1`、full-vocabulary log-softmax、max skills=`1`、beam capacity=逐 query pool size、score replay atol=`5e-5`。RTX final 与 workspace control/warmup/clip final checkpoint 文件 SHA-256 已写入 experiment note。
- 已实现 `score_legacy_core75_fixed_pool.py`、matrix analyzer、八卡 node launcher、Tianzhi bootstrap 和七项 focused tests。本地 source-only runtime 通过六项并跳过 torch fake；冻结的 Python=`3.10.12`、torch=`2.10.0+cu128` CPU runtime 跑完全部七项，exhaustive trie score 与逐 candidate step replay allclose。
- `py_compile` 与两个 launcher 的 `bash -n` 通过。GPU job 尚未提交。
- task plan 已重新 attestation，SHA-256 前缀=`61f0dca14238`（fixed-pool scoring protocol 冻结）。
- 独立审查的三项 HIGH 已修：launcher 注入 exact base model；analyzer 校验 shard/input/runtime/checkpoint provenance；multi-GT 改为逐 gold mention 统计，并用 Official core75 的 `200` 个 gold mentions 反算 macro recall。A800 archived Top50 replay 还要求 Top-1/5/10/50 集合不变。
- fixed-pool job=`tide-chenyufei03-10601-weesu` 已提交；条件为 `rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`、launch commit=`349505d88e246b6e45430189c59bb4858ea70386`。
- fixed-pool job=`tide-chenyufei03-10601-weesu` 已完成，pod restart=`0`。八个 shards、A800 score replay、Top-1/5/10/50 集合、shard provenance、pool/gold coverage 和 Official core75 的 `200` 个 gold mentions 守恒全部通过。
- 四路 raw→fixed 的 top-1 gained/lost 都是 `0/0`；RTX-correct/A800-wrong 的 control/warmup/clip=`9/9/8` 条 query fixed top-1 恢复均为 `0`。条件为 Official core75=`75` queries、pool=four beam-order Top50 union plus all gold、四个 learned-code full-vocabulary log-probs 加 stop control=`1`、逐 query 穷举。R@1 缺口已定位到 checkpoint learned scoring。
- task plan 已重新 attestation，SHA-256 前缀=`d288da586667`（fixed-pool 结果已写入，下一步冻结为 position-score decomposition）。

## 2026-08-11：position-score decomposition 实现

- fixed-pool scorer 已保存每个 candidate 的五项 contribution：code position=`0/1/2/3` 与 stop control=`1`。每行强制五项求和重建 total score，uniform weights=`[1,1,1,1,1]` 强制精确重建 fixed-pool 排名。
- position analyzer 固定搜索 `3,061` 组 normalized weights；条件为五个 raw weights 各取 `{0.5,0.75,1.0,1.25,1.5}`、按 tuple mean 归一化并按八位小数去重。每路 A800 先过 fixed-pool R@10/R@50=`69.0000/75.0000` 门槛，再按 R@1/R@10/R@50 与 uniform L2 distance 选择。
- 五折诊断固定按 `row_position % 5` 切分；每折 train=`60`、heldout=`15`（Official core75=`75` queries、同一 fixed pool）。无 deep-gate 候选时只报告 ineligible best-effort weighting。
- 新 manifest 记录 base-model `config.json`/`model.safetensors`、checkpoint tokenizer 原始文件、去掉 `is_local` 后的 semantic config，以及 Official core75=`75` queries、max source length=`2048` 条件下实际 `input_ids`/`attention_mask` 全量 hash。analyzer 强制跨系统对齐实际 tokenizer 输入。
- 独立 launcher 固定新 output/bootstrap log、`SKILLGR_POSITION_ANALYSIS=1`，并在启动时要求 node 可见 GPU count=`8` 且每张设备名称包含 `A800`。
- 冻结 Python=`3.10.12`、torch=`2.10.0+cu128`、NumPy 可用环境下，fixed-pool=`9/9`、position-score=`6/6` 测试通过；`py_compile`、三个 launcher 的 `bash -n`、Ruff 与 `git diff --check` 通过。
- 旧 v1 fixed-pool 产物重跑后逐 query JSONL SHA-256 保持 `2379a329f7832a39531427cba81c63bc63fa4ce1e67b3e8bb889ea8148c0ed8c`；summary 只新增 model-input alignment 验证字段。
- 提交前本地实现与验证已完成。
- 独立复核最终结论为无 CRITICAL/HIGH/MEDIUM 问题。实现提交为 commit `0ef8258cca6f7fa0d2c6583b4fc98f5e17ee342f`。
- Tianzhi dry-run 确认 queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1` 与完整 launch commit。正式 job=`tide-chenyufei03-10602-lbift` 已提交，等待 worker 创建。
- job=`tide-chenyufei03-10602-lbift` 已完成；条件为 queue=`rudder-exper`、一个 `8×A800` 节点、pod runtime=`6m6s`、restart=`0`。八个 shards、四路 model/tokenizer/input provenance、A800 archived replay、五项 contribution sum、uniform ranking、`3,061`-weight grid 与五折验证全部通过。
- control/warmup/clip 的 full-gate grid 数均为 `0/3,061`；条件为 Official core75=`75` queries、fixed pool、R@1/R@10/R@50 门槛=`27.3683/69.0000/75.0000`。control 有 `668/3,061` 组通过 deep gate，但最高 R@1 只有 `21.5905`；warmup/clip 无 deep-gate weighting。
- 五折 heldout R@1=`20.9238/23.8794/21.2127`，对应 uniform R@1=`20.9238/23.9905/23.8794`；条件为 control/warmup/clip、Official core75=`75` queries、同一 fixed pool、train=`60`/heldout=`15`。full-set 的小幅增益没有跨 fold 稳定性。
- RTX-correct/A800-wrong cases 的 mean gold-minus-best-nongold margin 主要损失在 c0/c1：control=`-1.6596/-1.2475`（`9` queries）、warmup=`-0.8922/-0.9212`（`9` queries）、clip=`-2.0248/-1.1925`（`8` queries）；stop margin 近似 `0`，条件为各自 uniform checkpoint 与同一 fixed pool。
- position weighting 路线停止。下一步直接在已保存 artifacts 上做 prefix-depth 与 RTX/A800 component-swap attribution，不提交新 GPU job。
- task plan 已重新 attestation，SHA-256 前缀=`83c3bd01c3b8`（position-score 结果与下一步 attribution 已写入）。

## 2026-08-11：prefix/component attribution 结果

- 本轮只读取 job=`tide-chenyufei03-10602-lbift` 的 score components，未启动新 GPU job。Official core75=`75` queries、gold mentions=`200`、semantic depth=`4` 与 fixed pool 全部保持不变。
- RTX/control/warmup=`500`/clip=`1.0` 的 prefix R@1 在 depth=`1` 为 `50.1302/50.4635/48.3524/51.1302`，在 depth=`2` 为 `47.7968/39.7968/44.7968/43.0190`；条件为上述 `200` 个 gold mentions 的 macro recall。总体差距从 c1 决策后明显扩大。
- 全部 `32` 个 RTX→A800 component masks 的 exact Shapley 显示，c1 占 control/warmup/clip endpoint R@1 gap 的 `45.9%/52.9%/48.5%`；stop contribution=`0`。
- 只有 mask=`15`（RTX c0/c1/c2/c3 + A800 stop）和 mask=`31`（全 RTX）通过 fixed-pool R@1/R@10/R@50=`27.3683/69.0000/75.0000` 门槛。单换 c1 虽提高 R@1，但 control/warmup/clip R@10 为 `66.1111/67.9556/63.9111`（各自其余 component 保持 A800），深召回交换仍在。
- attribution 相关 frozen-runtime tests=`20/20`（fixed-pool=`9`、position=`6`、attribution=`5`），endpoint、prefix-sharing、Shapley 守恒和 nondeployable marker 全部通过。独立复核修正三项 MEDIUM 后，无 CRITICAL/HIGH/MEDIUM。
- 已重读 Multi-GT loss/catalog convergence、classic document augmentation、RIPOR/PAG settings、anchored RIPOR 和单节点多卡说明。现有 A800 `query-only` 在 Official core75=`75` queries、beam=`50`、beam-order、global batch=`16` (`8` ranks x `2`)、lr=`3e-4`、query epochs=`5` 时 R@1/R@10/R@50=`27.26/69.33/76.78`，是下一 loss probe 的 control。
- 下一动作：先对冻结 SkillRouter teacher panel 和 LETTER IDs 做 c1-diverging hard-negative coverage audit；覆盖充足后，实现保留 CE 的 cumulative-prefix/sequence ranking 限定 probe。

## 2026-08-11：strict-base CE-prefix ranking 实现

- 发现 enhanced audit 与 RTX base 训练目标混用；已核对 base 的 `94,912` 行与 enhanced 前缀逐 JSON object 一致，teacher 前缀在 index/query/gold 上逐行对齐，尾部恰为 enhanced-only `5,166` 行。
- base candidate bundle 已生成：summary SHA-256=`72b0f525d6c3d9feabc7770c054ee70d8478f1b280d1a100793da8d5b6c6dd6d`、manifest SHA-256=`8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d`；source queries=`94,912`、gold mentions=`124,175`、teacher top-50 c1 candidates=`86,305`。
- 已实现 CE+cumulative-prefix loss、candidate bundle fail-closed loader、严格 legacy wrapper、Official core75 matrix analyzer、8卡 node launcher 与 Tianzhi bootstrap。
- auxiliary decoder 使用 `torch.random.fork_rng` 保存并恢复主 CPU/CUDA RNG；positive CE forward 保持原 batch shape/order。训练第一批还会对 positive/negative depth=`2/3/4` full-sequence score 与 legacy stepwise decoder做 allclose。
- frozen Python=`3.10.12`、torch=`2.10.0+cu128` CPU runtime 跑完 `27` 项 focused tests，结果 `26` pass、`1` CUDA-only skip；source-only Ruff、bash syntax 与 diff check 通过。GPU preflight 会补跑 CUDA RNG invariant。
- implementation cross-review 初次发现 `2` 项 HIGH：R@1 gate 使用显示值舍入、无合格 arm 时仍生成 fallback winner。两项已修复；复审结果为 CRITICAL=`0`、HIGH=`0`。
- 实现已提交为 commit=`1310c8d8cda9275e07a1dbb961068876ed796de6`。Tianzhi dry-run 确认 queue=`rudder-exper`、nodes=`1`、节点规格=`8×A800`、image=`chenyufei03/skillsearch-gr:cu128-torch28-v1`。
- GPU job=`tide-chenyufei03-10603-gigmv` 已提交；worker 在 `1/1 Running`、restart=`0` 条件下通过固定 runtime、GPU count=`8` 与全部 `21` 项 node tests，包括 CUDA RNG invariant。
- 八个独立单卡 arm 已分别启动在 GPU `0–7`；source-length audit 和模型初始化均已通过。八路已同步到 catalog step=`100/2,730`，train loss=`4.092995`（batch=`16`、seed=`20260626`、lr=`3e-4`、BF16、dropout=`0.1`），日志未发现 Traceback、OOM 或 CUDA error。
- 两个 disabled control 已完成 query updates=`38,805/38,805` 与 Official core75=`75` queries 评测；final model-state hash、best query step=`38,200`、best dev loss=`0.7701343703584546`、predictions 和指标均与历史 A800 workspace anchor exact（batch=`16`、seed=`20260626`、BF16、workspace=`:16:8`、beam=`50`、beam-order）。
- 六个 prefix-loss arm 当前位于 query step=`31,000–32,050/38,805`；GPU `2–7` 继续训练，pod=`1/1 Running`、restart=`0`。按最近约 `30` 秒的日志推进速度估计，训练主体还需约 `20–30` 分钟，之后执行各路 Official core75=`75` queries、beam=`50` 评测与 matrix gate。

## 2026-08-12：strict-base CE-prefix ranking 结果

- job=`tide-chenyufei03-10603-gigmv` 的八个 arm 全部完成；worker 已清理。catalog、control replica、历史 anchor、candidate、score parity、执行次数和 finite audit 全部通过。
- 两个 disabled control 的 model hash、predictions、best query step=`38,200`、best dev loss=`0.7701343703584546` 和 Official core75 指标均 exact（base source queries=`94,912`、catalog updates=`2,730`、query updates=`38,805`、batch=`16`、seed=`20260626`、BF16、`:16:8`、beam=`50`、beam-order）。
- `seq_010` 的 Overall R@1/R@10/R@50=`28.7683/70.5873/73.8889`、Single R@1=`50.0000`、Multi R@1=`18.7768`；`cum_002` 对应 Overall=`26.3683/69.9556/77.1111`、Single R@1=`33.3333`、Multi R@1=`23.0906`。条件为各自 dev-best checkpoint、Official core75=`75` queries、beam=`50`、beam-order。
- 六个非零 arm 均未通过联合门槛，`passing_arms=[]`、`selected_arm=null`，不进入多 seed。下一步先做 control/`seq_010`/`cum_002` 的 paired per-query error-overlap analysis，再决定如何合并 sequence Top-1 与 cumulative Multi/deep-recall 信号。

## 2026-08-12：paired error-overlap 结果

- 新增 `analyze_legacy_ce_prefix_error_overlap.py` 与合成测试；新旧 query-rank analyzer 合计 `5` 项测试、Ruff、`py_compile` 和 diff check 全部通过。
- control/`seq_010`/`cum_002` 的 `75` 条 Official core75 predictions 在 index、source index、query、gold、target count 上 exact 对齐；独立重算 R@1/R@10/R@50 与已有 result summary 在 atol=`1e-12` 内一致。
- Single 子集的 sequence top-1 hit set包含 control/cumulative 的全部命中；Multi 子集的 cumulative top-1 hit set包含 control/sequence 的全部命中。按 gold cardinality 诊断性组合的 Overall R@1/R@10/R@50=`31.7016/72.6222/75.7778`（Official core75=`75` queries、beam=`50`、beam-order），其中 R@1 与逐 query oracle相同。
- 输出 summary/per-query/report SHA-256 分别为 `4460df1c571a04b94280cd0654ff74e0e4e9b538f3c80c06934ed31f3ade2e3d`、`ed00c1ec5a4bb7c30fc289aab8674048263635e521ec9888ee51fd05ac2a03fe`、`7f66c47ce75db0f2fd437af7cba06633faadac7801a0a866b361b9b8c4cbdb8f`；路径为 ignored `outputs/legacy_ce_prefix_error_overlap_20260812_v1/`。
- 下一训练候选收敛为 conditional hybrid：Single rows 用 sequence lambda=`0.10`，Multi rows 用 cumulative lambda=`0.02`，原 CE 保留；推理仍为单 checkpoint 与原 beam decoder。

## 2026-08-12：conditional prefix hybrid 实现

- 独立计划复核无 CRITICAL；三项 HIGH 已处理：Official core75=`75` queries 的结果标记为探索性、原 performance gate 与 `selected_arm=null` 写死、单侧 ablation 延后到首轮有 arm 过门后。
- 已实现 manifest-bound gold cardinality、conditional loss、双组 score parity、group presentation/comparison audit 和 fixed-scope 兼容路径。canonical `(candidate_key,gold_count)` mapping SHA-256=`016728eb7681486d41eb83a9de27fa32c56e70adce1a0b17d47d8631bff796aa`（manifest SHA-256=`8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d`）。
- 首轮一个 `8×A800` 节点运行两个 disabled controls、fixed `seq_010/cum_002` anchors 与四个 hybrid arms；hybrid grid 为 Single sequence lambda=`{0.05,0.10}` × Multi cumulative lambda=`{0.01,0.02}`，每路各占一张 A800。
- frozen Python=`3.10.12`、torch=`2.10.0+cu128` CPU runtime 相关测试=`35` 项，结果为通过=`34`、CUDA-only 跳过=`1`；Ruff、`py_compile`、launcher `bash -n` 通过。本地缺少 `shellcheck`；节点 preflight 将补跑 fixed/conditional CUDA RNG invariant。
- implementation cross-review 最终结论为 CRITICAL=`0`、HIGH=`0`；loss、fixed anchors、counts/gates 与 GPU `0–7` arm 映射均通过复核。
