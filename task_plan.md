# 固定 LETTER Semantic ID 的 RIPOR / PAG 三组对照计划

## 1. 目标

在不改变现有 `Qwen semantic embedding → LightGCN collaborative embedding → LETTER tokenizer → Semantic ID` 链路的前提下，把 RIPOR 的 prefix-oriented decoder ranking 和 PAG 的 lexical Set ID、联合训练及 planning-ahead decoding 接入当前 GR，训练并评测以下三组：

1. `ripor`：只引入 RIPOR。
2. `pag`：只引入 PAG。
3. `ripor_pag`：同时引入 RIPOR 和 PAG。

最终回答四个问题：

- RIPOR 是否缓解 single-skill subset 上的语义检索不足？
- PAG 的 Set ID 与 planning signal 是否改善 multi-skill subset？
- PAG 在已有 RIPOR sequential model 上是否仍有边际收益？
- RIPOR 与 PAG 是互补、冗余，还是存在负交互？

## 2. 当前阶段与执行门禁

- **当前阶段：Phase 6 最小验证；通过后进入 Phase 7 集群执行。**
- **状态：用户已于 2026-08-04 确认按本计划继续执行。**
- 开发和最小验证通过后可直接提交 GPU 任务。
- 后续每次改变方法定义、数据单位、初始化、loss、解码公式或主指标，都必须先更新本计划并向用户说明；不得在执行中静默改变。

## 3. 不可变边界

以下内容三组完全固定：

- Qwen 生成的语义 embedding。
- LightGCN 生成并融入 LETTER 的协同 embedding。
- LETTER tokenizer、LETTER codebook 和已经生成的 `semantic_ids.jsonl`。
- Semantic ID 深度、每层 codebook 范围、skill 到 SID 的映射。
- 基础 single-SID decoder checkpoint。
- train/dev/test 原始 query、canonical skill catalog、source text policy。
- teacher 模型、teacher candidate panel、随机种子、最终阶段更新步数、每卡 batch 和 8 卡 world size。
- 评测 query、beam size、K 值和指标实现。

禁止事项：

- 不更换 embedding 模型。
- 不重新训练 LightGCN 或 LETTER，不重新生成 Semantic ID。
- 不换成 Faiss RQ，不把 RIPOR 的 relevance-based RQ identifier 初始化混入本轮。
- 不把外部 dense/BM25 分数伪装成 PAG 推理分数。
- 不在本轮混入 multi-SID sequence generation；本轮只研究 single-SID GR 建模。
- 不因为中间结果不理想而临时改变 loss 权重、teacher、负例、beam 或数据范围。

## 4. 复现范围与论文差异

### 4.1 RIPOR

本轮实现并测量 RIPOR 的 decoder 训练核心：对固定 LETTER SID 做 position-specific raw conditional-logit scoring，并用 prefix-oriented MarginMSE 优化正负 SID 路径。

本轮有意不采用 RIPOR 的 relevance encoder → residual quantization → 新 SID 初始化，因为这会替换现有 LETTER SID，破坏对协同信号的控制变量。因此本轮结果表述为“RIPOR prefix-ranking 对固定 LETTER GR 的贡献”，不声称是替换 identifier 后的完整 RIPOR benchmark reproduction。

### 4.2 PAG

本轮实现 PAG 的以下核心机制：

- `Msp`：T5-SPLADE lexical representation，MarginMSE + FLOPS regularization。
- 从每个 skill document 的 `Msp` 表示选择固定 `m` 个 lexical Set ID。
- `Mset`：在固定 Set ID 上训练 simultaneous document score。
- `Mseq`：single-SID sequential decoder。
- 联合阶段：平均 `Mset` 与 `Mseq` 的共享 T5 参数；LETTER SID embedding、position-specific code heads 和 control heads 必须来自 `Mseq` 且 bitwise 不变。
- 联合 decoder 将 query token 同时作为 decoder prefix。
- 推理使用 PAG planning-ahead score，而不是后处理分数融合：

  `S'(prefix) = S_seq(prefix) + λ · max_{skill ∈ topN 且 descendant(prefix)} S_set(query, skill)`

  `S_set` 是当前 prefix 的绝对先验，每一步重新计算，不能沿 token 重复累加。

PAG 的 sequential identifier 继续使用 LETTER SID；Set ID 是独立的 lexical identifier 字段，不修改 LETTER tokenizer。

## 5. 三组设置的精确定义

现有 CE+LETTER single-SID checkpoint 记为 `B`，共享 PAG set model 记为 `Mset`。

| 设置 | Sequential 初始化 | Set 初始化 | 最终训练目标 | 推理解码 |
|---|---|---|---|---|
| `ripor` | `B → RIPOR Mseq` | 无 | `Lseq_prefix`；为匹配联合组总更新量，再执行同样的 final-stage RIPOR 更新 | constrained single-SID beam，只用 `S_seq` |
| `pag` | 共享 `B` | 共享 `Mset` | `CE_single_sid + Lset` | PAG planning-ahead beam |
| `ripor_pag` | 与 `ripor` 共享的 RIPOR Mseq | 与 `pag` 共享的 `Mset` | `Lseq_prefix + Lset` | PAG planning-ahead beam |

控制关系：

- `ripor - B`：RIPOR 总贡献。
- `pag - B`：没有 RIPOR 时的 PAG 总贡献。
- `ripor_pag - ripor`：已有 RIPOR 时 PAG 的边际贡献。
- `ripor_pag - pag`：已有 PAG 时 RIPOR 的边际贡献。
- 交互项：`(ripor_pag - ripor) - (pag - B)`。

`B` 复用现有 checkpoint 和已有/重新统一跑出的评测结果，不新增第四组训练。

## 6. 数据与多 ground-truth 语义

### 6.1 single-SID 训练单位

原始 query 若有 `K` 个 ground-truth skills，展开为 `K` 个 single-SID example。每个 example 只有一个当前正例 SID，保持现有 single-SID CE 建模方式。

### 6.2 排序负例规则

- 当前展开 example 的 target skill 是该 example 唯一的 ranking positive。
- 同一原始 query 的其他 ground-truth skills 只进入 exclusion mask，绝不能成为负例。
- 不在每个展开 example 中再次把全部 `K` 个 GT 都作为正例，否则一个 `K`-skill query 会产生 `K²` 正例权重。
- teacher top-ranked skills 中排除全部 GT 后再选 hard negatives。
- RIPOR 与 PAG 使用同一份冻结 numeric teacher score panel；score 缺失时 fail closed，不使用静默的 0/1 伪标签。

这保证 multi-skill query 在 single-SID 训练中的总权重仍为 `K`，同时消除 sibling false negatives。

## 7. 开发清单

### Phase 1：协议和资产锁定

- [x] 固化三组 `method_setting` 的 fail-closed 参数契约。
- [x] 校验 LETTER manifest、Semantic ID logical hash、LightGCN provenance。
- [ ] 记录 train/dev/test、catalog、base decoder、teacher checkpoint 的路径和 SHA-256。
- [x] 确认三组使用 single-SID target mode。
- **Status：in_progress（待 teacher stage 写入 input SHA-256）**

### Phase 2：single-SID 多 GT 语义修正

- [x] 分离“当前 target positive”和“同 query 全部 GT exclusion set”。
- [x] RIPOR candidate builder 不把 sibling GT 当负例，也不产生 `K²` 正例重复。
- [x] PAG Set loss candidate builder采用相同正例/排除语义。
- [x] manifest/setting contract 记录 source query 数、expanded example 数和 multi-GT 权重策略。
- **Status：done**

### Phase 3：RIPOR sequential 分支

- [x] 固定 LETTER SID，不进行 M0/RQ identifier 替换。
- [x] 使用 position-specific code heads 计算 raw conditional logits。
- [x] 实现 prefix cumulative score、prefix divergence mask、`alpha_i` 和 MarginMSE。
- [x] teacher score、hard negative、prefix objective 写入 checkpoint provenance。
- [ ] 生成可供 `ripor` 和 `ripor_pag` 共享的 RIPOR Mseq checkpoint。
- **Status：implemented，待正式 GPU stage 生成共享 checkpoint**

### Phase 4：PAG set 分支和联合模型

- [x] 实现 8 卡 `Msp`：T5-SPLADE log-saturation、max pooling、MarginMSE 和全局 FLOPS。
- [x] 实现 8 卡分片构建固定 `m=64` Set ID。
- [x] 实现从 `Msp` 初始化并训练共享 `Mset`。
- [x] 实现 `Mset + B` 和 `Mset + RIPOR Mseq` 两种受控合并。
- [x] 合并后验证 sequential SID embedding/code/control heads bitwise 等于对应 Mseq。
- [x] 联合阶段启用 query decoder prefix，同时训练 `Lset + CE` 或 `Lset + Lseq`。
- [x] 实现 top-N Set candidate domain 和 planning-ahead constrained beam。
- **Status：implemented，待 8 卡 preflight**

### Phase 5：8 卡 DDP 与启动器

- [x] teacher catalog/query encoding 使用 8 卡分片，按原始索引无重复合并。
- [x] Msp、Mset、RIPOR 和两种 joint training 使用 8 卡 DDP。
- [x] CE token、RIPOR pair、PAG pair/FLOPS 全部使用 global sum/count 归一化，避免各 rank 局部均值改变目标。
- [x] checkpoint、manifest、W&B、最终 JSONL 只由 rank 0 写入。
- [x] Set ID 构建和评测使用 8 卡无 padding shard。
- [x] 单一 stage 启动器生成三组目录和不可变 `setting_contract.json`；独立 stage 可并发使用 4–5 个节点。
- **Status：implemented，待 8 卡 preflight**

### Phase 6：最小正确性验证

只做会显著影响实验正确性或整批任务成功率的检查，不建立大规模测试流程：

- [ ] Python compile 和 shell syntax。
- [ ] 三组方法契约测试，错误组合必须 fail closed。
- [ ] LETTER SID hash 与 base/Mseq/Mset/final checkpoint 一致。
- [ ] single-SID sibling exclusion 与无 `K²` weighting 的小样本 invariant。
- [ ] PAG merge 后 sequential heads bitwise invariant。
- [ ] PAG planning prior 不随 token 重复累加的 beam invariant。
- [ ] DDP global loss 与单进程 global batch loss 的数值/梯度等价小样本检查。
- [ ] 8 卡 1–2 step preflight，确认 NCCL、显存、checkpoint 写入和数据路径；通过后直接进入正式训练。
- **Status：pending**

### Phase 7：正式训练、评测和归因报告

- [ ] 在 rudder-r2 按依赖关系申请最多 4–5 个节点；每个节点均使用完整 8×A800。
- [ ] teacher 完成后并行运行 PAG Msp/Mset 路径与 RIPOR Mseq；共享 checkpoint 完成后并行运行三组 final training；评测按 baseline/ripor/pag/ripor_pag 拆到最多 4 个 8 卡节点。
- [ ] 持续监控 pod、日志、checkpoint 和异常退出。
- [ ] 输出三组统一指标表和相对 `B` 的差值。
- [ ] 输出 RIPOR、PAG 的边际贡献与交互项，不用单个 overall 指标替代 subset 分析。
- **Status：pending**

## 8. 训练协议

- 硬件：rudder-r2 上按需使用 4–5 个单节点任务；每个任务为 8×A800，`torchrun --nproc_per_node=8`。
- 主干：现有 T5-base single-SID GR。
- 精度：BF16。
- 三组 final-stage：相同 global batch、seed、学习率、weight decay、warmup、更新步数。
- optimizer：每个方法阶段从相同规则重新初始化，不继承不对称 optimizer state。
- catalog CE：不在方法阶段重复训练；catalog knowledge 来自共享 base checkpoint。
- dev selector：不得使用只反映 CE 的 legacy dev loss 选择 RIPOR/PAG checkpoint；固定步数后取 last/final。
- PAG：`m=64`；planning candidate domain 初始固定为 top-1000；主解码权重在训练前固定，不看 test 调参。
- beam/K：三组使用相同 beam 和 `K={1,10,50}`；single-SID 评测只生成一个 SID path，但从 beams 汇总 skill ranking。

锁定数值（同时写入 `configs/letter_ripor_pag_20260804.env`）：

- 输出根目录：`/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_pag_single_sid_20260804_v1`。
- `Msp=4700` steps，`Mset=5000` steps，共享 RIPOR `Mseq=5000` steps。
- 三组 final-stage 各 `5000` steps；`ripor` 总计执行两段各 5000 步的 RIPOR 更新。
- 每卡 batch `8`，world size `8`，global batch `64`。
- AdamW（weight decay `0`），初始学习率 `5e-4`，linear warmup + linear decay，warmup ratio `4.5%`；每个 stage 重置 optimizer/scheduler。
- BF16，seed `20260626`，PAG `m=64`、top-N `1000`、planning weight `1.0`，beam `50`，`K={1,10,50}`。

## 9. 测评设计

### 9.1 数据切分

- 评测输入使用原始未展开 query，避免一个 multi-skill query 被重复计分。
- 固定报告 `overall`、`single-skill subset (|GT|=1)`、`multi-skill subset (|GT|>1)`。
- 同时保留按 `target_skill_count=1/2/3/...` 的细分结果。

### 9.2 主指标

- Single-skill：Hit/Recall@1、@10、@50。
- Multi-skill：Macro Recall@10/@50、Micro Recall@10/@50、Hit@10/@50、Full Coverage@50。
- Overall：同一套 Recall/Hit/Full Coverage 指标。
- PAG 额外报告 Set top-1000 的 gold candidate recall，区分“planner 召回不到”与“sequential ranking 失败”。
- 记录 invalid SID rate、重复预测、解码耗时和峰值显存，作为正确性与效率诊断，不替代质量主指标。

### 9.3 必须输出的对照表

| Setting | Single R@1 | Single R@10 | Multi Macro R@10 | Multi Macro R@50 | Multi Micro R@50 | Multi FC@50 | Overall R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B` | | | | | | | |
| `ripor` | | | | | | | |
| `pag` | | | | | | | |
| `ripor_pag` | | | | | | | |

另外输出四行 delta：`ripor-B`、`pag-B`、`ripor_pag-ripor`、`ripor_pag-pag`，以及交互项。

### 9.4 PAG 诊断评测

不新增训练组，只对 `pag` 与 `ripor_pag` checkpoint 额外跑一次 `set_weight=0` 的同 checkpoint 诊断，以拆分：

- joint training 带来的变化；
- planning-ahead decoding 的即时贡献。

该诊断不能代替三组主结果，也不使用外部 dense/BM25 分数。

## 10. 完成标准

只有同时满足以下条件，任务才算完成：

- 三组 checkpoint 均由同一份锁定协议产生，provenance 可追溯。
- 所有 invariant 通过，无 unresolved CRITICAL/HIGH 语义问题。
- 三组在同一评测集和同一指标实现上完成评测。
- single/multi subset 指标、边际贡献和交互项均已报告。
- 代码、启动命令、日志、checkpoint、metrics 路径写入 `progress.md` 和实验记录。
- 未违反第 3 节的不可变边界。

## 11. 已知错误与防重复记录

| 错误/风险 | 状态 | 处理规则 |
|---|---|---|
| 曾尝试递归扫描整个 CFS 寻找 T5 config，扫描过慢并被中止 | 已停止 | 后续只检查明确的资产路径，不再做宽泛递归 CFS 扫描 |
| 当前已有部分未提交实现早于本计划产生 | 待审查 | 恢复开发时逐项按本计划审计；不符合的代码修改或移除 |
| single-SID 展开后若每行把全部 GT 再作为 positive，会造成 `K²` 权重 | 已修正并有 invariant test | 当前 target 是唯一 positive，全部 sibling 只做 negative exclusion |
| 联合组若直接从 CE base 开始会遗漏 RIPOR Mseq 阶段 | 已修正并 fail closed | `ripor_pag` 强制使用与 `ripor` 共享的 RIPOR Mseq |
