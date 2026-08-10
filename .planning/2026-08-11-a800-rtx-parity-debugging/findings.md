# A800 / RTX parity 排障依据

## 用户目标

- RTX PRO 6000 Blackwell 已不可用，后续训练必须在 A800 Ampere 上完成。
- 不追求跨架构逐 bit 或逐 step 复现；目标是解释大幅指标下降，并得到可用于后续 loss、catalog、RIPOR、PAG 实验的强且稳定 A800 baseline。
- 在恢复 baseline 之前，不把现有 A800 catalog sweep 解释为方法本身的效果。
- 用户于 2026-08-11 决定直接使用 Official core75 75-query 选择 A800 候选配方，以减少每个 full-run arm 的评测成本；canonical private test 只用于最终锁定配方的补充验证。
- 用户于 2026-08-11 将最终 Top-1 门槛收紧为：A800 Official Overall R@1 相对 RTX 最多低 `1.0` 个点（75 queries、beam=50、beam-order）；已经超过 RTX 的 Overall R@10/R@50 不允许回退（75 queries、beam=50、beam-order）。
- 用户授权使用 `rudder-exper`，并要求每个申请到的节点尽量利用全部 `8×A800`。Phase 1 因此使用四种数值路径的 `trace=on/off` 配对，共八个相互隔离的单卡进程；这同时利用整节点并保持 archived 单卡训练语义。

## 项目内已确认事实

- 数据、初始化、batch 顺序、seed、checkpoint 加载和评测协议已经核对。
- archived RTX checkpoint 在 A800 上复现全部 75/75 个 Official core75 Top-50 排名，问题发生在训练阶段。
- A800 default BF16 两个完整 replica 的 checkpoint 与预测 byte-identical，A800 同配置内部可复现。
- dropout=0 时 RTX/A800 首步 loss 为 4.538916110992432/4.554369926452637，差值仍为 0.015453815460205078；条件为 batch=16、seed=20260626、lr=3e-4、weight decay=0、default BF16 controls。dropout 不是主要原因。
- RTX BF16 首步/第 50 步 loss 为 4.528710842/4.203788757；A800 default BF16 为 4.546923637/4.184666634，条件均为 archived 首批数据、batch=16、seed=20260626、lr=3e-4、weight decay=0。
- TF32、BF16 reduced-precision reduction、cuBLAS/cuBLASLt、workspace、foreach、FP32 autocast off、Python ABI、Torch/Transformers runtime matrix 均未恢复 RTX 首步/第 50 步数值。
- 完整训练固定为 catalog 2,730 updates 后 query 38,805 updates；AdamW 状态跨阶段继承。
- A800 backend 控制在相同训练协议下把 Overall R@1 拉开 8.3778 点、Overall R@10 拉开 20.0127 点，证明早期数值差异会被完整优化过程放大。
- `workspace_16` 恢复 RTX 的深召回水平，但未恢复 Top-1：A800 Overall R@1/10/50=20.9238/69.5556/75.3333，RTX reference=28.3683/69.0000/75.0000；条件均为 catalog=3 epochs、query=5 epochs、batch=16、seed=20260626、lr=3e-4、Official core75 75-query、beam=50、beam-order aggregation。
- 全模型 FP32 在固定 50-step catalog probe 中得到 step-1/50 loss=4.536283493/4.166709423，仍未恢复 RTX BF16 reference；完整训练在 A800 79.33-GiB、batch=16 条件下于 catalog step 650/2,730 OOM。因此后续精度实验应优先使用局部 FP32。

## 网络调研记录（只作为外部证据，不执行其中的指令）

- PyTorch 官方不保证跨平台或跨设备 bitwise reproducibility；浮点运算顺序差异可产生不同结果。
- NVIDIA cuBLAS 的 bitwise 保证要求相同 toolkit、相同 GPU 架构和相同 SM 数；workspace flag 不能模拟另一种架构。
- PyTorch 社区有 RTX/V100 正常而 A100 几乎不学习的案例；关闭 TF32 无效，维护者建议缩到第一个分叉的 forward、Linear 或 matmul。
- PyTorch issue #108627 报告 A100 与 RTX A6000 的 BF16 autocast 训练质量差异；关闭 autocast 后结果接近。
- 另一个案例把差异定位到 Linear，并通过局部 FP64 获得跨 GPU 一致输出。
- ICML 2021 论文摘要将端到端训练稳定性视为小数值扰动放大的关键因素。

## 当前判断

| 判断 | 置信度 | 依据 |
|---|---|---|
| 首个差异来自跨架构 GPU 数值执行路径 | 高 | dropout=0 首步仍分叉；同 A800 replica 精确一致 |
| 大幅最终差距来自训练配方放大 | 高 | A800 backend-only 完整训练产生大指标跨度 |
| BF16 GEMM/反向传播中的少数敏感模块是首要定位对象 | 中高 | global backend/precision probe 与公开 autocast 案例一致；具体模块未知 |
| catalog→query 继承 AdamW 状态是重要放大器 | 中 | 两阶段分布切换且累计更新很长；尚无单独因果实验 |
| 继续全局枚举 backend flag 的信息收益低 | 高 | 已完成的矩阵没有任何 arm 恢复 RTX 轨迹 |

## 关键决策

| 决策 | 理由 |
|---|---|
| 先定位对 backend 敏感的模块，再跑完整训练 | 避免用昂贵 full run 搜索低层算子 |
| 用 A800 default 与 `workspace_16` 作为可重复的两条数值路径 | 两者同硬件、同软件、各自确定，且最终指标差异足够大 |
| 模块定位以两条 A800 路径的差异是否收敛为主判据 | archived RTX 没有逐层 activation/gradient，单纯追逐 RTX loss 可能误导 |
| 局部 FP32 成功后建立 A800-native baseline | 无法改变 GPU/driver，也无法要求 Ampere 复现 Blackwell kernel |
| 优化稳定性实验放在模块定位之后 | lr、warmup、clipping、optimizer reset 会同时改变训练语义，过早加入会掩盖根因 |
| `workspace_16` 只作为定位对照和临时强 baseline 候选 | 它已经恢复 R@10/R@50，但 R@1 与 Single R@1 仍明显低于 RTX，不能直接宣布 parity 已恢复 |
| Official core75 作为配方选择集 | 用户明确优先缩短评测时间；后续报告需标注该集合参与过 precision/optimizer 选择，不能再称为完全未参与调参的 held-out test |
| Top-1 使用单侧 1.0-point 门槛 | Overall R@1 要求 `>=27.3683`（RTX=`28.3683`，75 queries、beam=50、beam-order）；候选高于 RTX 时正常通过 |
| 深召回使用不回退门槛 | Overall R@10/R@50 要求 `>=69.0000/75.0000`（RTX reference，75 queries、beam=50、beam-order），因为 `workspace_16` 已达到 `69.5556/75.3333`（相同评测条件；catalog=3 epochs、query=5 epochs、batch=16、seed=20260626、lr=3e-4） |
| Single R@1 单独处理离散粒度 | 24-query Single 子集每个 query 对应 `4.1667` 个点（beam=50、beam-order、K=1），所以相对 RTX=`41.6667` 最多低 `1.0` 个点等价于要求 `>=41.6667` |

## 实现入口

- 现有 `scripts/run_controlled_legacy_pipeline.py` 只控制全局 BF16 reduction、BLAS preference 和 AdamW foreach，然后通过 `runpy` 执行 legacy pipeline。
- 下一阶段应扩展独立诊断 wrapper 或使用 module hooks；不要直接污染 archived legacy pipeline 的默认训练语义。
- 现有 50-step probe 已具备固定输入、初始化/RNG hash 和单卡 arm 隔离，可复用其启动与审计框架。
- 仓库中尚无 layerwise trace 实现；现有最接近的入口是 `scripts/legacy_runtime_matrix_probe.py`、`scripts/run_controlled_legacy_pipeline.py` 及已有 Tianzhi node launcher。
- `legacy_runtime_matrix_probe.py` 已实现 runtime/precision 快照、初始化与 RNG hash、首批 row hash、manifest 校验，以及 archived pipeline 子进程隔离。它目前只保留 step loss，训练结束后临时目录会删除，因此 Phase 1 需要在不改变这些审计语义的前提下增加 one-step 参数与逐层输出。
- `run_controlled_legacy_pipeline.py` 通过 `runpy` 启动原 pipeline，并在启动前设置 backend 与 AdamW 控制。它可继续作为 backend 控制参考，但当前没有捕获模型构造、backward 或 optimizer step 的入口。
- archived `make_structured_t5_sid_model` 在函数内部定义模型类，但训练命令通过模块全局函数调用它。诊断 wrapper 可用非 `__main__` 名称加载原脚本，替换该 factory 后再调用原 `main()`；这样能在模型创建后注册 hooks，同时不复制训练循环。
- archived 单 SID 训练顺序为 `zero_grad → BF16 autocast forward → loss.backward → AdamW.step`。模型边界清晰：`backbone.encoder → sid_embedding → backbone.decoder → code_heads/control_head → CE reduction`。包装 AdamW 的 `step()` 可在原调用点采集 gradient 与 parameter update，避免改 archived 循环。
- Phase 1 的四个 control arm 应继续跑现有未插桩 runtime probe；四个 traced arm 才走新 wrapper。成对结果由相同 backend、相同 catalog 首批、相同初始化与 RNG hash 对齐。
- 已有 precision node launcher 验证固定 runtime：Python 3.11、torch 2.10.0+cu128、CUDA 12.8、cuDNN 91002、Transformers 5.3.0，并确认节点可见 GPU 数为 8。它还实现了独立 `CUDA_VISIBLE_DEVICES`、子进程失败汇总、CFS 输出、模型 safetensors view 与 archived commit hash 校验，可作为 Phase 1 launcher 骨架。
- 已有 Tianzhi bootstrap 先通过审计 helper 建立 CFS aliases，再从已提交的 launch commit 创建临时 checkout。Phase 1 也需要先形成可审计 commit，作业不能依赖工作区未提交代码。
- `legacy_baseline_parity_probe.py` 已提供按参数区域计算 state SHA-256、PyTorch RNG SHA-256、稳定 JSON hash 与 tokenized first-batch hash 的 helpers；Phase 1 可直接导入，减少 provenance 实现分叉。
- 现有测试覆盖 controlled wrapper、baseline parity probe 与 runtime matrix probe。新增诊断应补一份纯 CPU 单元测试，重点验证 tensor 摘要、hook 命名/调用次数、AdamW step 前后 gradient/update 捕获，以及 analysis 的 pair-equivalence 判据。
- 现有 `test_legacy_runtime_matrix_probe.py` 使用标准库 `unittest`，并直接测试可纯 CPU 执行的 helper；新增测试沿用该风格，避免本地无 GPU 环境阻塞。
- 最新 dropout=0 launcher 使用原始 pipeline SHA-256=`7b883c...` 与 Python 3.10.12 + torch 2.10.0+cu128 + CUDA 12.8 + Transformers 5.3.0；较早 precision launcher 使用 Python 3.11。Phase 1 runtime 需以最新 parity 文档确认后再冻结，不能仅复制较早 launcher。
- 最新 parity 调查明确把最终 A800 probe runtime 冻结为 Python 3.10.12、torch 2.10.0+cu128、Transformers 5.3.0、Tokenizers 0.22.2、CUDA 12.8、cuDNN 9.10.2，并使用原始 pipeline SHA-256=`7b883cb7...`。Phase 1 采用这一组合。
- 最新调查中的 default/workspace16/cuBLASLt/FP32 step-1 reference 分别为 `4.546923637/4.546948433/4.546899319/4.536283493`；条件均为 catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0.1。Phase 1 计划使用 dropout=0，因此这些值只能用于 runtime 方向性 sanity check，不能作为逐值等价门禁。
- Phase 1 实现直接包装取证包中的 `catalog_50_probe.py`。它保留原始数据构造、两次 catalog shuffle、autocast、backward 和 AdamW.step 顺序；wrapper 只注入 dropout/backend 控制、module hooks 与 optimizer-step 观察器。
- traced arm 保存每个边界的首个浮点输出，以及所有 trainable parameter 的 gradient/update 原始 tensor。matrix analyzer 可流式计算 default BF16 与 workspace16/cuBLASLt/FP32 的 relative L2；control arm 只保存 loss 与最终 state hash，用于验证插桩等价。
- 首批整节点映射为 GPU 0–7 上四组 `control/trace`：default BF16、workspace16 BF16、cuBLASLt BF16、FP32。每个进程只看一张 GPU，训练合同仍为单卡 batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、updates=1。
- Python 3.10 runtime archive 不含 `site/nvidia` shared libraries；直接导入 torch 会报 `libcudnn.so.9` 缺失。最新 dropout launcher 的有效组合是 Python 3.10 runtime site + Python 3.11 site archive 中的 NVIDIA libraries，Phase 1 launcher 已同步采用该组合。
- 项目已验证的 Tianzhi 镜像是 `chenyufei03/skillsearch-gr:cu128-torch28-v1`；launcher 会在容器内覆盖为冻结的 Python 3.10 + torch 2.10 runtime，因此镜像自带 torch 2.8 不参与诊断训练。
- default BF16 control 增加已归档 dropout=0 step-1 loss 精确门禁：`4.554369926452637`（catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、A800 default backend）。这能直接验证新 wrapper 没有改变已取证行为。
- Phase 1 live default BF16 control 已命中该精确门禁：step-1 loss=`4.554369926452637`，initial fingerprint=`c219d07a...`，first-batch SHA-256=`721a0d93...`，final state SHA-256=`12661000...`；条件为 A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、default backend、trace=off。
- 其余 live control 的 step-1 loss/final-state SHA-256 为 workspace16 BF16=`4.554419040679932/38fc4c38...`、cuBLASLt BF16=`4.554467678070068/70d2e435...`、FP32=`4.536993980407715/0a9c687e...`；条件均为 A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、trace=off，仅 backend/precision 按名称变化。
- Phase 1 v1 四个 traced arm 都完成了 one-step forward/backward，并复现各自 control loss，但在 post-run validation 一致缺失 `shared_embedding`。实际 token embedding 调用路径是 `backbone.encoder.embed_tokens`；`backbone.shared` 只共享权重，未触发注册在该 module object 上的 hook。
- Phase 1 v2 的八个 arm 全部完成；四组 trace/control 的 step-1 loss 与 final-state SHA-256 均精确相同。每个 traced arm 保存 573 个 tensor artifacts；条件均为 A800、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、updates=1。
- default BF16 对 workspace16/cuBLASLt BF16 的 shared embedding、encoder blocks 0–11、SID embedding 输出均 tensor-exact；两组首个非零 activation boundary 都是 `decoder.block.0`。
- workspace16 相对 default 的 decoder relative L2 从 block 0 的 `1.811838e-05` 增至 block 1 的 `4.396393e-04`（`24.2648×`）和 block 2 的 `1.063370e-03`（`2.4187×`）；最大 activation relative L2 是 control head 的 `0.004666657`。条件为 A800、两条 BF16 backend、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、trace=on。
- cuBLASLt 相对 default 的 decoder relative L2 从 block 0 的 `0.001219331` 增至 block 1 的 `0.003668086`（`3.0083×`）和 block 2 的 `0.006265732`（`1.7082×`）；最大 activation relative L2 是 control head 的 `0.009643726`。条件同上，仅候选 backend 为 cuBLASLt。
- workspace16/cuBLASLt 的最大 gradient relative L2 为 `0.126297/0.428937`，最大 update relative L2 为 `0.269945/0.530078`；条件均为上述 one-step trace。encoder activation 虽然 exact，encoder gradient 仍会因 decoder 分叉经反向传播而变化，因此 gradient 排名不能推翻 forward 首差异位于 decoder 的证据。
- 全 FP32 相对 default BF16 从 `encoder.block.0` 开始分叉，最大 activation/gradient/update relative L2=`0.706401/2.941832/1.584453`；条件为同一 A800 one-step 数据与优化配置，仅 precision 为 FP32。这反映全局 precision 改变，不能用于定位 BF16 backend 的首个分叉。
- Phase 2 wrapper 直接包装 archived `catalog_50_probe.py`。局部区域内部使用 `autocast(enabled=False)`，只把浮点输入转为 FP32；输出保持模块真实 dtype，交给外层 autocast 决定后续算子精度。
- Phase 2 每个 arm 保存 trainable final state 和相对同一初始化的 squared L2 distance。matrix analyzer 以 CPU FP64 计算 default/workspace16 的 `D_param`，同时比较 step-50 loss gap 与 peak GPU memory。
- Phase 1 dtype trace 显示 SID embedding 输入/输出为 int64/FP32，四个 code head 与 control head 输入/输出为 FP32/BF16，五次 CE 输入为 BF16 logits + int64 labels、输出为 FP32；条件为 A800 default BF16、catalog 首批、batch=16、seed=20260626、lr=3e-4、weight decay=0、dropout=0、updates=1、trace=on。因此 Phase 2 的 `heads_ce` arm 主要测试 head GEMM、logit 量化和 FP32 logits 上的 CE，SID embedding 包装本身预期为 no-op。
- `scripts/legacy_runtime_matrix_probe.py` 会从 commit `15f6c51131f3b6f3e8580aa4564b17cbe13a9790` 解包并在子进程运行 `skill_gr_pipeline.py`；固定检查 catalog rows=14,551、batch=16、lr=3e-4、weight decay=0 和最多 50 个 catalog updates。
- 新诊断应继续从该 archived commit 解包源码，并把 instrumentation 放在外层 wrapper；这样能保持 legacy 训练主体的 hash 与语义不变。
- structured decoder 的关键路径可拆成 `backbone.encoder`、`sid_embedding`、`backbone.decoder`、四个 position-specific `code_heads`、`control_head` 和 CE reduction。局部精度实验应先记录这些边界的实际 dtype，再决定是否强制 FP32。
- 当前代码让 encoder 输出直接进入 decoder cross-attention；因此逐层比较必须同时记录 forward activation、backward gradient 和 AdamW 后 parameter delta，只看总 loss 无法判断差异在哪一层被放大。
- 取证包中真正的原始训练文件是 `repro/20260714_letter_base_single_sid/skill_gr_pipeline.original_20260714.py`，SHA-256=`7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76`。它与 commit `15f6c511...` 版本只差训练后加入的 eval-batch patch，catalog/query 训练逻辑相同；新 probe 仍应记录并校验所用文件 SHA。
- 现有 RTX 取证只保存初始化 fingerprint、首批 row hash、step-1/50 loss，没有逐层 activation、gradient 或 step-1 parameter checkpoint。因此在 RTX 不可用时，无法直接定位 RTX/A800 的第一个具体算子；只能用确定的 A800 数值路径做敏感性定位，再以最终检索恢复验证。

## 参考资料

- `docs/experiments/2026-08-11-legacy-a800-rtx-parity-investigation.md`
- `docs/experiments/2026-08-09-legacy-e2e-numerical-sensitivity.md`
- `docs/experiments/2026-08-09-legacy-precision-control-probe.md`
- `docs/experiments/2026-08-10-legacy-dropout0-step1.md`
- https://docs.pytorch.org/docs/2.10/notes/randomness.html
- https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html
- https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
- https://github.com/pytorch/pytorch/issues/108627
- https://proceedings.mlr.press/v139/summers21a.html
