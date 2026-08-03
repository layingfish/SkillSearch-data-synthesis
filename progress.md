# RIPOR / PAG 三组对照进度

## 2026-08-04：计划固化

- **当前状态：** 计划确认中，暂停继续开发和 GPU 提交。
- 已读取并遵循 `planning-with-files`，创建 `task_plan.md`、`findings.md` 和本进度文件。
- 已把用户边界、论文复现范围、三组设置、开发阶段、训练协议、最小验证和测评矩阵写入 `task_plan.md`。
- 已记录本轮之前存在的未提交实现，后续必须按计划逐项审计，不能直接沿用。

## 2026-08-04：计划获批并恢复执行

- **当前状态：** Phase 6 最小验证；独立审查后提交 teacher 与 8 卡 preflight。
- 用户确认按 `task_plan.md` 继续，并授权在 rudder-r2 按需使用 4–5 个节点。
- 调度原则已写回计划：每个任务必须完整利用节点的 8×A800；共享依赖完成后并行三组训练与四组评测。
- 已重新读取 `task_plan.md`、`findings.md`、`progress.md`，session catchup 未报告遗漏上下文。

## 此前已完成但尚未按本计划最终验收的工作

- 添加过固定 LETTER manifest/SID provenance 校验和三组 method contract。
- 添加过 RIPOR prefix MarginMSE、PAG Msp/Mset/joint/planning 相关实现。
- 添加过 8 卡 teacher evaluation、Set ID 构建、训练和评测路径。
- 创建过三组启动器 `scripts/run_letter_ripor_pag_settings.sh`。
- 做过 Python compile、shell syntax 和少量无 torch 契约测试。
- 这些修改尚未提交，且其中至少有两项需要按新计划修正：
  - `ripor_pag` 必须恢复使用共享 RIPOR Mseq，而不是直接使用 CE base。
  - single-SID ranking positive 必须是当前 target，siblings 只做负例排除，避免 `K²` 权重。

## 测试记录

| 检查 | 结果 | 备注 |
|---|---|---|
| `py_compile`：GR pipeline、SkillRouter、focused tests | 通过 | 仅语法级 |
| `bash -n scripts/run_letter_ripor_pag_settings.sh` | 通过 | 仅 shell 语法级 |
| 三组 method contract / LETTER manifest / scoreless teacher focused tests | 通过 | tensor merge test 因本地无 torch 跳过 |
| focused `tests.test_skill_gr_pipeline` | 通过 | 53 passed，7 个 optional-dependency skips |
| 锁定 stage launcher `bash -n` | 通过 | 每个 GPU stage 固定 `torchrun --nproc_per_node=8` |
| 8-rank DDP objective invariant | 通过 | `max_gradient_error=5.96e-08`，global count=20 |

## 2026-08-04：实现按计划收敛

- 修正 expanded single-SID 排序语义：当前 target 是唯一 positive，同一 source query 的全部
  ground truth 只作为 negative exclusions；focused invariant 覆盖 sibling exclusion 与无 K² weighting。
- 恢复受控联合初始化：`ripor_pag` 必须合并共享 PAG Mset 与 `ripor` 使用的同一 fixed-LETTER
  RIPOR Mseq；错误初始化 fail closed。
- 增加 PAG 同 checkpoint 的 weight=0 诊断，并保持联合模型训练得到的 decoder query prefix。
- 给 PAG Msp 和 structured method stages 增加 linear warmup/decay，并将 scheduler 写入 provenance。
- 把串行启动器改成可提交到独立 8×A800 节点的 stage DAG，所有资产和超参数锁在
  `configs/letter_ripor_pag_20260804.env`。
- pinned `t5-base` revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` 已下载到共享 CFS。
- 资产覆盖审计：100078 raw rows 展开为 129341 single-SID examples，17457 个 multi-GT raw
  queries；14551 个 semantic IDs 与 14551 个 canonical skills 完全对齐，无 missing/empty GT。
- Tier 3 独立 cross-review 未发现剩余 CRITICAL/HIGH；审查提出的 weak-teacher、contract SHA、
  Msp max-length 和跨 artifact contract 绑定问题均已补强。
- 代码已提交为 `f6ebacf`（`feat: add fixed-LETTER RIPOR PAG experiment DAG`）。
- teacher 8 卡任务已提交：`tide-chenyufei03-10230-hlrjf`；日志固定写入
  `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_pag_single_sid_20260804_v1/logs/teacher-0804a.log`。
- 首个 teacher job 在训练前失败：launcher 调用 alias helper 时未传 `--with-dataset-rw`，因此
  `/dataset_rw` 未创建，输入检查以 exit code 1 结束；该问题已定位并修复，日志初始化也前移到输入检查之前。
- 第二个 teacher job `tide-chenyufei03-10231-rshfg` 完成了 8 卡 skill/query encoding，但在相似度
  矩阵乘法处失败：query embeddings 留在 CPU、catalog embeddings 已在各 rank GPU。已在 matmul 前
  显式把 query embeddings 搬到当前 rank device；不改变分数或分片语义。
- 第三个 teacher job `tide-chenyufei03-10232-nrekv` 成功，生成 100078 行 teacher panel 和 SHA-bound
  `setting_contract.json`；teacher Hit@1=0.83119167，Hit@50=0.99351506。
- 首个 preflight `tide-chenyufei03-10233-hsqmj` 的 53 个镜像内单测和 8-rank DDP invariant 均通过，
  真实 PAG Msp 烟测因错误复用 GR `max_source_length=2048` 而 OOM。已按 PAG 官方
  `t5_splade_train.sh` 将 Msp 独立固定为 `max_length=128`，并同步锁定配置和 setting contract；
  Mset/RIPOR/joint 仍保持 2048。

## 错误与处理

| 错误 | 次数 | 处理 |
|---|---:|---|
| 错把 FastCtx 工具当成 `functions.exec` 内嵌工具调用 | 1 | 后续直接调用 FastCtx namespace |
| focused unittest 使用了两个不存在的测试方法名 | 1 | 查询真实方法名后重跑，相关测试通过 |
| 对整个 CFS 做 `**/config.json` 递归扫描耗时过长并被用户中止 | 1 | 不再重复；后续只读取明确资产路径 |

## 下一步

1. 清理失败 preflight 的临时子集并用官方 Msp 长度重新提交 8 卡 preflight。
2. preflight 通过后并发提交 `pag_shared` 与 `ripor_mseq`。
3. 共享 checkpoint 完成后并发提交三组 final，并运行 baseline/三组统一评测。
4. 持续回填 job ID、日志、checkpoint 和 metrics。

## 5 问恢复检查

| 问题 | 答案 |
|---|---|
| 当前在哪？ | Phase 6 最小验证与 cross-review |
| 接下来去哪？ | teacher → preflight → 正式 stage DAG |
| 目标是什么？ | 固定 LETTER，测量 RIPOR、PAG 和联合方法在 single/multi subsets 上的贡献 |
| 已学到什么？ | 见 `findings.md` |
| 已做什么？ | 见本文件此前工作和测试记录 |
