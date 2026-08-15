# 深度调研：Semantic ID、协同信号与 GR decoder

Date: 2026-08-15
Status: done
Scope: SkillSearch 纯 generative retrieval；catalog=`14,551` skills；训练集=`94,912` query rows；Official core75=`75` queries（Single/Multi=`24/51`）。

## 结论

当前链路的核心问题是建模对象错位。

LETTER 把 SID 当作 skill embedding 的压缩码。LightGCN 把一次性 query row 当作 user。decoder 又把每条 query 当成单标签分类。最终评测衡量的是一个 query 对一组 skill 的相关性排序。这三层分别优化了不同对象。

建议的新主线暂称 **RF-SetGR（Relevance-Factorized Set Generative Retrieval，相关性分解式集合 GR）**。它把 SID 树定义为多标签相关性函数的概率分解：每个树节点预测“该子树是否含有相关 skill”，同一 parent 下允许多个 child 同时为正。ID、训练目标和解码分数由这一个定义统一起来。

这条主线保留纯 GR。推理只运行一个 T5 decoder 和合法树搜索，不使用 dense fusion、reranker 或外部候选分数。

## 当前链路审计

### 1. Semantic ID 与语义重建

- skill 文档使用 `Qwen/Qwen3-Embedding-0.6B` 生成 `1,024` 维 embedding（当前 canonical catalog=`14,551` skills，max_length=`512`）。
- LETTER 使用 RQ-VAE，配置为 depth=`4`、每层 codebook size=`256`、latent dim=`32`、epochs=`20,000`、alpha=`0.01`、beta=`0.0001`。
- 第一层只使用 `90/256` 个 code（当前 LETTER checkpoint）；后续三层均使用 `256/256` 个 code。
- 量化后有 `80` 个 collided items，分布在 `39` 个 collision groups；当前流水线通过最后一级 minimum-cost reassignment 修复到零 collision。
- tokenizer 目标以 `1,024` 维文档 embedding 重建为主。query 到 skill 的真实相关性无法直接更新冻结 SID。

这套 ID 保留了文档总体语义。它没有保证 query 能预测前缀，也没有保证相近 skill 在检索决策上可区分。D2-DocID、RIPOR、MERGE、DIGER、BLOGER 都从不同角度证明了“重建质量”和“可检索性”存在目标差异。

### 2. 当前所谓 CF embedding

LightGCN manifest 的条件为：query nodes=`94,912`、skills=`14,551`、edges=`124,175`、dim=`32`、layers=`3`、epochs=`500`、BPR batch=`4,096`、lr=`0.001`、reg=`0.0001`。

每条自然语言训练 row 被定义成一个独立 user node。它没有跨 row 的历史：

| query degree | rows | 占比（训练集=`94,912` rows） |
|---:|---:|---:|
| `1` | `77,455` | `81.6072%` |
| `2` | `7,761` | `8.1771%` |
| `3` | `7,586` | `7.9927%` |
| `4` | `2,110` | `2.2231%` |

degree=`1` 的 query node 无法提供 item-item 协同关系。真正连接多个 skill 的只有 degree=`2–4` 的 `17,457/94,912=18.3928%` rows。这些边表达“同一任务需要共同使用”，语义更接近 complementarity。

训练结果也符合这一判断：co-GT skill cosine mean=`0.867059`，random-pair cosine mean=`-0.000562`（当前 LightGCN manifest，co-GT pairs=`43,179`）。该 embedding 主要把共同需要的 skills 压到一起。它缺少推荐系统中稳定 user 历史所产生的 preference geometry。

因此，当前 CF regularization 会强化 Multi 的共同需求结构，同时模糊 Single 查询需要的细粒度边界。LETTER 原论文自己的 Instruments 实验也显示 collaborative regularization 会把第一层 code usage 从 `148` 降到 `76`（TIGER + collaborative regularization，first codebook=`256`）；diversity regularization 才把使用量拉回 `150`（LETTER，first codebook=`256`）。当前项目第一层 usage=`90/256`，方向一致。

### 3. decoder 与 loss

当前 `StructuredT5SIDModel` 使用：

- 独立的 `sid_embedding`；
- depth=`4` 个 position-specific `code_heads`；
- input embedding 与 output head 默认不 tied；
- 只有显式启用 RIPOR codebook initialization 时，RQ codebook 才会复制到两处参数。

`cum_082` 的训练条件为：query loss=`expanded_ce`、catalog epochs=`0`、lr=`1e-5`、BF16、DDP world size=`8`、batch=`2/GPU`、global batch=`16`、epochs=`1`、listwise weight=`1.0`、temperature=`0.5`、hard negatives≤`10/query`。

`expanded_ce` 会把一个 multi-GT row 展开为多个单目标 row。每个 token 仍用 categorical softmax：

```text
sum_over_legal_children P(child | query, prefix) = 1
```

当两个 gold skill 在同一 parent 下走向不同 child 时，两个 child 都应为正。softmax 强制它们争夺同一单位概率质量。独立展开只让两个方向轮流拉扯。

项目里的 `sibling_masked_ce` 仅在另一条 gold path 的首次分叉处，把该 token 从当前 CE denominator 删除。它没有在同一次 forward 中把所有 gold child 标成正例，也没有估计“子树含相关 skill”的边际概率。

RAD-DPO 已明确观察到 multi-label SID 的 probability squeezing。它的 Multi-label SFT 仍是多条正例 sequence log-likelihood 的平均，核心概率空间仍为 categorical sequence distribution。它没有改成独立 subtree event，也没有给出 leaf marginal 的一致估计。

### 4. 实验证据闭环

- `cum_082`：Overall R@1/R@10/R@50=`29.5905/70.9206/73.7778%`；Single=`45.8333/54.1667/54.1667%`；Multi=`21.9468/78.8049/83.0065%`（Official core75=`75`，Single/Multi=`24/51`，beam=`50`，beam-order，CE log-prob）。
- diverse-error step=`5,200`：Overall=`30.2571/71.5873/74.0000%`；Single=`45.8333/54.1667/54.1667%`；Multi=`22.9272/79.7852/83.3333%`（同一 Official core75 与解码条件）。
- `47` 个训练 checkpoint 中，Single 命中的 query 始终是相同的 `11/24`；`13` 个 miss 中有 `11` 个在所有 checkpoint 的 beam=`50` 里都没有 gold（Official Single=`24`）。
- `cum_002@28,400` exhaustive catalog scoring 的 Top-1 命中与 beam=`50` 完全相同（Official core75=`75`，catalog=`14,551`，CE full-vocabulary log-prob）。Single gold rank=`1/2–10/11–50/>50` 的数量为 `10/2/1/11`（Single=`24`）。
- mode3：Overall R@1/R@10/R@50=`29.32/69.04/78.56%`；Single=`54.1667/58.3333/62.5%`；Multi=`17.63/74.08/86.11%`（对应历史 mode3 训练与 Official core75 测试条件）。它提高个体区分，同时损失 Multi 排序。
- query-aligned RQ + prefix-mass distillation + PRO fusion 的最好结果为 Single R@1/R@10/R@50=`33.3333/41.6667/45.8333%`（Official Single=`24`，beam=`50`，best checkpoint=`17,200`，omega=`0`），低于 `cum_082` 的 `45.8333/54.1667/54.1667%`（同一 Official Single 与 beam 条件）。

这些结果共同指向 score model。扩大 beam、增加同类合成 query、继续 hard-negative reweighting 的上限已经很低。

## 全文精读范围

本轮完整阅读了本地目录 `/dataset_rw/SkillSearch-data-synthesis/literature/20260815-gr-method-review` 中 `46` 个 PDF；该集合已包含最后补入的 RAD-DPO。阅读范围包含正文、全部图表、公式、证明、appendix、implementation details、ablation、limitations 和 references。摘要与搜索 snippet 只用于筛选，没有支撑下述方法判断。

### Semantic ID 与 tokenizer

- DSI；NCI；GenRet；TIGER；SEATER；LETTER；GRID。
- D2-DocID；HHQ；MERGE；GR²；GRDR；MTGRec；PIT。
- ReSID；DIGER；BLOGER；UniGRec；DOS；DAS。
- Preserving Item Semantics for Free；Grounded Tokenization for Generative Retrieval；How Reliable Are Semantic-ID Tokenizer Comparisons；Semantic IDs at Snapchat。

### relevance、decoder 与 search

- RIPOR；PAG；PRO；APAO；SmartGR。
- DOGR；DDRO；Align³GR；Token-Level Credit Assignment Optimization。
- TCA4Rec；PrefixMem；LIGER。
- Generative Retrieval Overcomes Limitations of Dense Retrieval but Struggles with Identifier Ambiguity。
- How Well Does Generative Retrieval Generalize?；Understanding Semantic IDs；Expressiveness Limits of Fixed Semantic-ID Trees。
- RAD-DPO。

### 多标签与大标签空间理论

- Probabilistic Label Trees for Extreme Multi-label Classification。
- PECOS: Prediction for Enormous and Correlated Output Spaces。
- Label Disentanglement in Partition-based Extreme Multilabel Classification。
- LightGCN。

## 文献给出的明确边界

以下方向已有直接工作，不能作为本项目的核心创新：

| 方向 | 已有代表工作 | 对本项目的含义 |
|---|---|---|
| RQ-VAE / RQ-KMeans 语义 ID | TIGER、LETTER、GRID | 换 quantizer 或调 depth/codebook 只构成 tokenizer ablation。 |
| collaborative alignment | LETTER、DAS、UniGRec、DOS | 把一个 CF vector 对齐到 quantized embedding 已被充分覆盖。 |
| query/relevance-aware ID | RIPOR、MERGE、GR²、GRDR | 把 query embedding 混入 RQ 目标不足以形成新贡献。 |
| tokenizer 与 generator 联合优化 | GenRet、DIGER、BLOGER、UniGRec、PIT | end-to-end 或 Gumbel code assignment 本身已有工作。 |
| 多路径 / 多 ID | MINDER、MTGRec、GRDR、PIT、Label Disentanglement | 给每个 skill 复制多条 route 的新颖性很弱。 |
| prefix ranking / worst-prefix | RIPOR、PRO、APAO、SmartGR | prefix loss、beam-aware distillation 和几何 fusion 都已有直接方案。 |
| document-level preference optimization | DOGR、DDRO、Align³GR、RAD-DPO、Token Credit | listwise、DPO、RL、hard-negative 不能单独承担论文主线。 |
| prefix-conditioned SID representation | NCI PAWA、PrefixMem | prefix-aware input/output embedding 已有明确方案。 |

仍存在的空白是：现有 SID-GR 普遍把输出定义成一个 normalized sequence distribution；PLT/XMC 则为每个子树定义独立 Bernoulli event。本轮检索没有找到把后者完整移植到 neural autoregressive GR，并同时用于 set-aware SID learning、leaf marginal scoring 和 exact constrained top-k 的论文。

## 项目真正的问题

### 问题一：SID 在压缩“skill 是什么”

检索需要学习“给定 query，skill 是否相关”。同一个 skill 对不同 query 的可检索面可以很不一样。一个全局文档 embedding 的欧氏重建无法表达这种条件相关性。

当前第一层 `90/256` code usage 与后续完整使用形成明显 hourglass。宽 codebook 没有转化为有效的 coarse routing capacity。固定四级路径还累计了四次前缀决策误差。

### 问题二：CF 把 complementarity 压成 similarity

当前 graph 中 `81.6072%` query nodes 只有一条边（训练集=`94,912` rows）。它们不产生协同邻域。多边 query 把共同完成任务的 skills 拉近。结果是 co-GT cosine mean=`0.867059`（当前 LightGCN），远高于 random mean=`-0.000562`。

这恰好有利于 Multi 的粗召回，也会让近邻 skill 的 Single 决策边界变弱。

### 问题三：decoder 在解一个多分类问题

一个 query 的 gold set size 为 `1–4`（当前训练 rows）。如果两个相关 skill 分属不同子树，两个 subtree event 可以同时成立。categorical softmax 无法表示这个事实。

`expanded_ce` 造成正例间竞争。`sibling_masked_ce` 只消除一部分错误负梯度。hard-negative 与 listwise loss 在错误的概率空间内继续调权，因此主要改善 Multi 的平均覆盖，无法改变稳定的 Single top-1 决策。

### 问题四：三套几何没有闭合

RQ codebook、decoder input SID embedding、decoder output head 是三套参数。`cum_082` 没有使用 codebook-tied initialization。decoder 可以学会 token frequency 与路径模式，却没有持续接收 tokenizer 的语义几何。

这是次要问题。Preserving Item Semantics for Free 与 PrefixMem 已给出直接修法；它不能解释 set-valued relevance 的根本错位。

## 新主线：RF-SetGR

### 1. 概率定义

对 query `q` 的 gold set `G(q)`，为树中每个节点 `v` 定义：

```text
z_v(q) = 1  iff  descendants(v) 与 G(q) 的交集非空
```

若 parent `u` 已经为正，decoder 对每个合法 child `v` 独立输出：

```text
p_v(q) = sigmoid(f_theta(q, prefix(v)))
```

同一 parent 的多个 child 可以同时接近 `1`。一个 skill `s` 的边际相关性为：

```text
P(s relevant | q) = product of p_v(q) over nodes on path(s)
```

训练只展开正 parent，不展开 gold skill row。每个正 parent 使用完整 legal-child multi-hot target：

```text
L_set = sum over positive parents u
        sum over legal children v of u
        BCEWithLogits(logit_v, z_v)
```

使用未加 class weight 的 BCE，以保留 Bernoulli 概率语义。catalog=`14,551` 时，每个局部 child set 可完整计算，无需全 catalog negative mining。

### 2. 一致性依据

`z_skill=1` 必然蕴含路径上的 ancestor event 全部为 `1`。按概率链式法则，正确的条件 node probability 相乘就是 skill 的 marginal relevance。

BCE 是 strictly proper binary loss。节点估计一致时，leaf marginal 也一致。按 leaf marginal 排序对 expected Precision@K 是 Bayes-optimal；当 `|G(q)|` 对给定 query 固定时，同一排序也优化 expected Recall@K。

local softmax 额外施加 `sum(child)=1`。query 同时激活多个 child 时，该约束与真实事件不相容。PLT 的独立 sigmoid 去掉了这项限制，也去掉了 fixed SID softmax 导致的 sibling probability squeezing。

### 3. relevance-driven ID

当前 LightGCN embedding 删除。替代表示称为 **relevance profile**，指一个 skill 在训练 query 空间中的相关性原型：

```text
r_i = normalize(mean(Qwen3-Embedding(query)) for query where skill_i is gold)
s_i = normalize(Qwen3-Embedding(skill_document))
h_i = concat(r_i, s_i)
```

条件固定为当前 canonical training rows=`94,912`、skills=`14,551`、encoder=`Qwen3-Embedding-0.6B`、每侧 dim=`1,024`。`r_i` 是 PECOS/PIFA 类初始化，不作为独立 novelty claim。它直接聚合 query-skill relevance，避免把一次性 query node 伪装成 user。

第一版新树使用 depth=`2`：root branches=`128`，每个 root 下约 `113–114` 个 unique skill leaves（catalog=`14,551`）。容量 `128×128=16,384`，覆盖全部 catalog。leaf 使用 node-unique token，消除 collision 与“同一 code 在不同 prefix 下含义不同”的问题。

root assignment 采用 balanced Sinkhorn，保证 `128` 个 branch 全部使用。训练早期允许 soft assignment `a_i,c`。一个 gold set 对 branch `c` 的 soft union target 为：

```text
t_c(q) = 1 - product over i in G(q) of (1 - a_i,c)
```

当 assignment 变为 one-hot 时，它精确退化为“branch 中是否至少有一个 gold skill”。root BCE 的梯度可以直接更新 assignment，使 ID 按 set retrieval 难度学习。Gumbel temperature 初始方案为 `1.0→0.1`（前 `20%` tokenizer steps 线性退火）；balanced capacity 由 Sinkhorn 强制。

语义重建降为 leaf embedding anchor。初始 loss weight 待从 `{0, 0.01, 0.05}` 中选，条件为 fixed-ID Set-BCE 已通过最小验证。它承担 cold-start 与语义稳定，不再决定树的主拓扑。

### 4. decoder 参数化

- backbone 保持 T5-base。
- root head 输出 `128` 个独立 logits（new tree root=`128`）。
- leaf 使用 `14,551` 个 node-unique embeddings（catalog=`14,551`），只计算当前 root 的 `113–114` 个 legal leaves。
- leaf input embedding 与 output weight tied；初始化为 `h_i` 经线性投影后的向量。
- 一个原始 query row 只 forward 一次 encoder。decoder 分别计算所有 positive parent prefixes。
- Single 与 Multi 使用同一 `L_set`。区别只体现在 multi-hot target 的正 child 数量。

### 5. exact generative decoding

每条路径 score 是若干 `[0,1]` 条件概率的乘积，扩展路径后 score 不会上升。可用 priority queue 按 prefix score 展开，直到得到 top-`K` leaves。

Official 诊断先取 `K=50`（Official core75=`75`，与当前 R@50 对齐），不设 beam frontier。catalog=`14,551` 时也可直接 exhaustive score 验证 priority queue 的 exactness。该过程仍是合法 SID 生成。

## 为什么它可能同时改善 Single 和 Multi

Single query 的 gold set size=`1`。root 与 leaf 都得到完整 one-vs-rest relevance supervision，Top-1 直接由 marginal relevance 决定。

Multi query 的 gold set size=`2–4`。多个 root/leaf child 同时为正，不再互相挤压概率。模型无需通过“把 complement skills 编到相邻 SID”来绕过 loss 缺陷。

mode3 的历史结果提供了一个自然解释：更强的个体 query 监督把 Single R@1 推到 `54.1667%`（Official Single=`24`），同时把 Multi R@1 拉到 `17.63%`（Official Multi=`51`）。categorical distribution 中，增强一个 skill 会夺走其他合法 skill 的质量。独立 Bernoulli subtree events 允许个体区分和多 skill 覆盖共存。

## 与最近工作的差异

| 工作 | 它解决的内容 | RF-SetGR 新增的内容 |
|---|---|---|
| PLT | 固定 label tree 上的 Bernoulli node probability 与一致性理论 | 将它变成共享 T5 autoregressive decoder、semantic ID learning 与合法生成。 |
| PECOS | PIFA label representation、multi-label matcher、negative sampling | relevance profile 只作初始化；最终 ID 由 soft-union set loss 更新。 |
| SEATER | semantic tree、legal-child softmax、prefix ranking | child 由独立 sigmoid 建模；leaf score 是 marginal relevance。 |
| LETTER | semantic reconstruction、CF alignment、diversity | 删除伪 CF；主目标变成 set relevance factorization。 |
| RIPOR / PRO / APAO / SmartGR | prefix survival、ranking、distillation、fusion | 修正 node event 的概率语义；prefix 优化成为结果，不是附加 loss。 |
| DIGER / BLOGER / UniGRec | differentiable 或 joint tokenizer | assignment 接收的是 gold-set soft-union gradient。 |
| PIT / GRDR / Label Disentanglement | multi-route 或 overlapping assignment | 第一版坚持 one skill one path；创新不依赖 route duplication。 |
| RAD-DPO | DPO 的 prefix conflict、pseudo-negative 与 multi-positive coverage | 不使用 sequence-level normalized probability；直接估计多个 subtree events。 |
| Latte | 证明 fixed SID softmax 的表达约束，并用 latent root 缓解 | 独立 node events 从概率定义上移除 sibling sum-to-one 约束。 |

## 最小验证路径

### Stage A：只换概率语义

保持 current LETTER ID=`depth 4 × codebook 256`、backbone、`cum_082` initialization 和 Official evaluation。只改以下内容：

- 使用原始 rows=`94,912`，不做 multi-GT row expansion；
- 使用 node-level multi-hot BCE；
- 推理同时跑 exact priority queue top-`50` 与 exhaustive catalog score；
- 训练条件初始固定为 DDP world size=`8`、batch=`2/GPU`、global batch=`16`、lr=`1e-5`、epochs=`1`、BF16、catalog epochs=`0`。

进入 Stage B 的最低门槛：

- Single R@1 至少从 `11/24=45.8333%` 提升到 `12/24=50.0000%`（Official Single=`24`，exact leaf marginal）；
- Single gold rank≤`50` 至少从 `13/24` 提升到 `15/24`（Official Single=`24`，exhaustive catalog）；
- Multi R@1 不低于 `21.0%`（Official Multi=`51`，exact leaf marginal）；
- 至少一个历史稳定 miss 的 gold-minus-top1 marginal margin 翻正。

Stage A 没有满足前两项时，先停止新树训练。该结果会说明当前瓶颈更靠近 query encoder/OOD coverage，ID 重构的 GPU 投入暂时不划算。

### Stage B：relevance-driven two-level ID

使用 root=`128`、depth=`2`、unique leaves=`14,551` 的 balanced tree。先训练 soft-union root assignment，再 harden tree，随后训练完整 RF-SetGR。对照只保留 `cum_082` 与 Stage A；无需继续累加工程 loss。

### 必看诊断

- Official Single/Multi R@1/R@10/R@50（Official core75=`75`，Single/Multi=`24/51`）。
- exhaustive gold rank 与 exact queue rank。
- 每个 parent 下同时为正的 child probability。
- root branch usage（目标=`128/128`，new tree）。
- Single `13` 个历史 miss 的逐 query margin。
- mode3 与 base/task5 query 上的 leaf calibration，检查个体区分是否仍以 Multi 损失为代价。

## 风险判断

这条方法有高于继续合成数据和参数 sweep 的提升上限，因为它改变了训练目标、ID 定义和解码分数的共同概率语义。

它不能保证 Single R@1 大幅上涨。Official Single 只有 `24` 条 query，每翻转一条就是 `4.1667` 个点（Official Single=`24`）。如果 query encoder 对 OOD 意图完全没有表征，正确 loss 也无法凭空恢复语义。Stage A 正是低成本检验这一点。

方法层面的论文故事是完整的：**Semantic ID 应当分解 set-valued relevance，而非压缩 item embedding。** CF、ID、decoder 和 search 都由这个原则导出。

## 关键原始来源

- LETTER: https://arxiv.org/abs/2405.07314
- RIPOR: https://arxiv.org/abs/2311.09134
- PAG: https://arxiv.org/abs/2404.14600
- Probabilistic Label Trees: https://arxiv.org/abs/2009.11218
- PECOS: https://arxiv.org/abs/2010.05878
- Label Disentanglement: https://arxiv.org/abs/2106.12751
- DIGER: https://arxiv.org/abs/2601.19711
- PIT: https://arxiv.org/abs/2602.08530
- APAO: https://arxiv.org/abs/2603.02730
- Generalization analysis: https://arxiv.org/abs/2603.19809
- Fixed-tree expressiveness limits: https://arxiv.org/abs/2605.06331
- PrefixMem: https://arxiv.org/abs/2606.00324
- Understanding Semantic IDs: https://arxiv.org/abs/2607.24995
- SmartGR: https://arxiv.org/abs/2608.02048
- RAD-DPO: https://arxiv.org/abs/2602.23964
