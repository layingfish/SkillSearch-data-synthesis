# RIPOR / PAG 三组对照进度

## 2026-08-04：计划固化

- **当前状态：** 计划确认中，暂停继续开发和 GPU 提交。
- 已读取并遵循 `planning-with-files`，创建 `task_plan.md`、`findings.md` 和本进度文件。
- 已把用户边界、论文复现范围、三组设置、开发阶段、训练协议、最小验证和测评矩阵写入 `task_plan.md`。
- 已记录本轮之前存在的未提交实现，后续必须按计划逐项审计，不能直接沿用。

## 2026-08-04：计划获批并恢复执行

- **当前状态：** Phase 7 正式训练；刷新后的 8 卡 preflight 已通过。
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
| 刷新后的全链路 8 卡 preflight | 通过 | 56 tests；所有两步训练、planning/weight0 解码成功；dev 8 条、gold 8 条 |

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
- 第二个 preflight `tide-chenyufei03-10234-qtunc` 已通过 Msp 与 Mset，随后在加载原 CE+LETTER
  checkpoint 时发现 4 个 `code_heads.*.bias` 与当前 bias-free 构造不兼容。固定 LETTER 要求完整保留
  基线参数，因此恢复 position heads 的 bias 并继续严格加载；RIPOR 只调整 prefix ranking objective。
- 第三个 preflight `tide-chenyufei03-10235-xfeut` 通过 checkpoint 严格加载后，在 RIPOR progressive
  prefix loss 发现 active prefix=2 与 divergence mask depth=4 的轴不一致。现将 divergence mask
  投影到同一 active-prefix 轴，并用 depth=4/active=2 的精确 loss 与 pair-count 单测锁定语义。
- 第四个 preflight `tide-chenyufei03-10236-kjmnb` 已通过 Msp、Mset、RIPOR Mseq、RIPOR-only 和
  PAG-only；RIPOR+PAG 因将最长 2048 的完整 encoder source 为每个 candidate 重复到 decoder prefix
  而 OOM。PAG 官方 unified 脚本固定 `max_length=64`，因此现在只把 PAG decoder query prefix 限为
  64 token，encoder 仍完整保留 2048；训练、推理、setting contract 和 checkpoint provenance 同步。
- 第五个 preflight `tide-chenyufei03-10237-dajit` 已成功并写出 `preflight/PASSED`，但 eval audit 暴露
  dev 原始 schema 使用 `generated.query/selected_skill_ids`，而 expansion/source 只读 top-level，导致
  smoke eval 输入 1 token、gold=0。现统一兼容 raw 与 prepared schema，并增加 nested query/labels 回归测试；
  需刷新 preflight，确认 dev rows/gold 非零后才启动正式训练。
- 刷新 preflight `tide-chenyufei03-10238-lsltr` 已成功：K8S `Succeeded`，56 个镜像内 focused tests、
  8-rank DDP invariant、Msp/Mset/RIPOR/PAG/联合两步训练以及 planning/weight0 两次解码均通过；
  dev 8 条 query 的长度为 110--341 tokens，gold mentions=8、missing=0。正式门禁已解除。
- 第一波共享正式任务已并发提交：PAG Msp/Mset 为 `tide-chenyufei03-10239-gwkec`，RIPOR Mseq
  为 `tide-chenyufei03-10240-hrdvg`；两者均为 rudder-r2 上完整 8×A800 单节点任务。
- 首个 RIPOR 正式任务在 optimizer step 之前发现协议资产错误：配置误指向 alias 归一化前的 raw dev，
  其中 78/1000 rows 含 80 个固定 LETTER pool 外的 GT mentions。既有 canonical dev/test 都是
  1000 rows、missing=0；Official core75 为 75 rows（single 24 / multi 51）、200 GT、missing=0。
  已把固定协议改为 canonical dev + Official core75 主评测 + canonical private test 辅助评测；旧
  setting contract 下的共享产物不得进入 final，需刷新 contract 后重新启动。
- canonical split 与 teacher panel 的 SHA-bound contract 已刷新；RIPOR Mseq 已重提为
  `tide-chenyufei03-10241-trolo`。旧 PAG 任务等待 contract mismatch 门禁退出后从 Msp 重跑。
- 旧 PAG `tide-chenyufei03-10239-gwkec` 在完成 Msp 4700 步后、进入 Mset optimizer 前因 raw dev
  label 越界失败；旧 Msp/Mset 目录已原样移动到 `invalid_raw_dev_contract/`，可恢复但禁止进入 final。
  canonical PAG 共享路径已从 Msp 重提为 `tide-chenyufei03-10242-tmick`。
- canonical RIPOR Mseq `tide-chenyufei03-10241-trolo` 已成功完成 5000 steps，末步 loss=0.042466，
  共享 checkpoint 已写出；RIPOR-only 对齐更新量的第二段 5000-step final 已提交为
  `tide-chenyufei03-10245-pcgrm`。
- 语义自检发现 structured trainer 每次调用默认重启 progressive prefix curriculum。shared Mseq
  应使用该 curriculum，但从完整 Mseq 初始化的 RIPOR-only / RIPOR+PAG final 应始终监督全部 prefix。
  启动器现已显式区分 `progressive` 与 `full`，checkpoint provenance 同步记录；正在运行的
  `tide-chenyufei03-10245-pcgrm` 标记为 audit-only，完成后隔离并重跑，不进入结果。
- 为避免等待无效 run 白白占用关键路径，launcher 允许仅覆盖 `SKILLGR_RIPOR_ONLY_DIR`；corrected
  full-prefix final 将写入独立目录并与 audit-only job 并行，完成后再原子切换 canonical 结果目录。
- corrected run 已提交为 `tide-chenyufei03-10247-ctwhg`；同一 local step 的首批 loss 约 0.048，
  audit-only restarted-curriculum run 约 0.018，证实二者实际监督的 prefix 集不同。
- corrected RIPOR-only `tide-chenyufei03-10247-ctwhg` 已成功，末步 loss=0.008334，并已切换到
  canonical `ripor_only/`；audit-only 目录保存在 `invalid_restarted_prefix_curriculum/`。
- PAG Mset 完成 5000 steps 并写出 checkpoint（末步 loss=0.046081）；外层 Bash 因运行期间 launcher
  文件被编辑而在写盘后 EOF。严格下游加载任务 PAG-only `tide-chenyufei03-10248-cpaxi` 与
  RIPOR+PAG `tide-chenyufei03-10249-uhccs` 已启动，用实际 load guard 验证该 checkpoint。
- PAG-only final `tide-chenyufei03-10248-cpaxi` 虽完成 5000 steps，但 Official planning eval 的
  candidate recall 为 0。全量资产审计证明 14,551 个 skill 的 top-64 Set ID 全部相同；Official
  186 个 unique gold 均见于训练且 manifest 无缺失，排除数据覆盖问题。
- 对照 PAG 官方 commit `49802454...` 后确认本实现遗漏 `RegWeightScheduler`：官方在前
  `max_steps/3` 对 query/doc FLOPS 权重做 quadratic ramp，本实现此前从第 1 步使用完整
  `0.01/0.008`。该偏差已修正并增加 Set ID collapse 门禁；但 100-step、FLOPS=0 的诊断任务
  `tide-chenyufei03-10256-chxsw` 仍然 collapse，所以不能把正则视为唯一根因。
- 失效的 `pag_msp/pag_mset/pag_only/ripor_pag` 已整体、可恢复地移动到
  `invalid_collapsed_pag_set_ids/`，没有删除 checkpoint。首次普通用户移动因 root-owned 目录权限不足
  未产生任何变更，随后在源/目标/mount 安全检查后使用 `sudo mv` 成功隔离。
- 1000-step 同协议诊断 `tide-chenyufei03-10258-djvkx` 仍得到 `unique_set_count=1/14551`。
  随后的 8-rank 表示审计 `tide-chenyufei03-10259-qxqxc` 对相同 512 个 skill 比较 base T5 与
  1000-step Msp：base 每行平均只有 0.021484 个正 raw coordinate、仍有 9 个 top-64 set；训练后
  raw-logit 最大值均小于 0，lexical representation 全零、top-64 set 只剩 1 个。短 Msp 把模型
  推入官方 ReLU lexical branch 的全零区，是 collapse 的直接机制。
- 官方 scheduler 下的锁定 4700-step 正式复核为 `tide-chenyufei03-10257-vkfsd`；
  `task_plan.md` 预注册了其失败分支：若全量 Set ID 仍 collapse，只把 PAG Msp 对齐到官方
  200000-step/每卡 batch32/save-25000 协议，并以独立 Msp contract 绑定原 setting contract，
  其余 RIPOR、LETTER、Mset/final 和评测均不改变。
- `tide-chenyufei03-10257-vkfsd` 已完成 4700 steps（末步 loss=0.026829），全量审计仍为
  `unique_set_count=1/14551`、最大 token DF ratio=1.0；8 个 rank 同步在 Mset 前 fail closed。
  无效 Msp 已完整移动到 `invalid_collapsed_pag_set_ids/pag_msp_scheduler4700_job10257/`，没有删除。
- 预注册恢复分支已实现：独立 `pag_msp_protocol.json` 绑定原 setting contract SHA，协议 SHA 为
  `0ef4ed4b0060b6b366377357b4bb9278b02ab5f2f5636711004937776a1fa1db`；Msp 支持每 25000 步
  原子保存 model/optimizer/scheduler、每 rank RNG 和精确 data position，Mset/final 会双重校验
  base contract 与 Msp amendment。
- 首个 resume 门禁 `tide-chenyufei03-10260-reiih` 证明每卡 batch32 无 OOM，连续与 2→4-step
  恢复路径末步 loss 相同，最大参数误差仅 `9.313225746154785e-10`；原 bitwise-zero 判据过严，
  已按 Tier-3 数值等价标准显式锁为 `allclose(atol=1e-8)`，需用新作业名复核成功退出。

## 错误与处理

| 错误 | 次数 | 处理 |
|---|---:|---|
| 错把 FastCtx 工具当成 `functions.exec` 内嵌工具调用 | 1 | 后续直接调用 FastCtx namespace |
| focused unittest 使用了两个不存在的测试方法名 | 1 | 查询真实方法名后重跑，相关测试通过 |
| 对整个 CFS 做 `**/config.json` 递归扫描耗时过长并被用户中止 | 1 | 不再重复；后续只读取明确资产路径 |

## 下一步

1. 用新作业名复核官方 batch32 显存和 2→4 step `allclose(atol=1e-8)` resume invariant。
2. 门禁通过后启动 200000-step 正式 Msp，并在每个 25000-step checkpoint 监控表示状态。
3. 有效 Mset 完成后并发运行 PAG-only 与 RIPOR+PAG final，再并发统一评测。
4. 汇总已完成 B/RIPOR 与待完成 PAG 两组的 subset 指标、delta 和交互项。

## 5 问恢复检查

| 问题 | 答案 |
|---|---|
| 当前在哪？ | Phase 7，PAG Msp 有效 Set ID 门禁；B/RIPOR 已完成 |
| 接下来去哪？ | 有效 Msp/Mset → PAG/PAG+RIPOR final → 统一评测与归因表 |
| 目标是什么？ | 固定 LETTER，测量 RIPOR、PAG 和联合方法在 single/multi subsets 上的贡献 |
| 已学到什么？ | 见 `findings.md` |
| 已做什么？ | 见本文件此前工作和测试记录 |
