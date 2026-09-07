# CF-aware OPQ DocID + 并行 MTP 解码器

Date: 2026-09-07
Status: current main line
Scope: SkillSearch 纯 generative retrieval；catalog=`14,551` canonical skills；Official SkillsBench v1.1 Core75=`75` queries

## 1. 一句话

把 skill 的 DocID 从 RQ-VAE 的 coarse-to-fine 残差路径换成 CF-aware OPQ 的 `64` 个并行子空间码，解码器相应地从自回归 trie beam 换成「一个共享表示 + 64 个位置独立分类头」，推理时对全部 `14,551` 个合法 catalog tuple 精确打分。不使用 trie、beam、prefix pruning、reranker、dense fusion 或任何外部候选分数。

## 2. 为什么必须换成并行解码

RQ-VAE / RQ-KMeans 的码是**残差序列**：第 `k` 层量化的是前 `k-1` 层的残差。所以任意前缀天然是一个 coarse-to-fine 近似，自回归生成加 trie 剪枝与这个结构是匹配的。

OPQ 不是。OPQ 学一个正交旋转 `R`，把旋转后的向量切成 `M` 个互不重叠的子空间，每个子空间独立做 k-means。**`M` 个码之间没有顺序，也没有残差依赖**——第 3 个码不是对第 1、2 个码的修正。由此：

- 自回归会把一个并不存在的层级强加到码上；任意前缀都不是语义粗粒度桶。
- trie/beam 剪枝按一个无意义的顺序提前砍掉候选，制造纯损失。
- 位置数不再等于解码步数。加长 DocID 的代价从「线性增加解码步」变成「增加并行头」，几乎免费。

本仓库对两种组合都做了实测：H1 = 长 OPQ + 原自回归 trie 解码器，H2 = 长 OPQ + 并行解码器。结果见 §9。

## 3. 系统组成

```text
Qwen3-Embedding-0.6B skill document embedding (1024d)
LightGCN skill CF embedding (32d)
        |
        |-- Stage 1  scripts/build_letter_opq_docids.py
        |     每个 view 逐行 L2 归一化 -> 按能量加权拼接 (semantic 0.8 / CF 0.2)
        |     -> 1056d -> 补零 1088d -> Faiss OPQMatrix + ProductQuantizer(M=64, 8bit)
        |     -> 每个 skill 一个 64 x [0,255] 的定长 DocID，零碰撞
        |
        |-- Stage 2/3  scripts/skill_gr_opq_parallel.py train
        |     T5-base encoder(query) -> decoder 单步(decoder_start) -> h
        |     -> 64 个位置头 -> [64, 256] logits
        |     -> mean-position cross-entropy (MTP)
        |
        +-- Stage 4  scripts/skill_gr_opq_parallel.py eval
              score(q,d) = sum_p w_p * log_softmax(logits_p)[code_p(d)]
              对全部 14,551 个合法 tuple 精确计算，取 top-K
```

## 4. Stage 1：CF-aware OPQ DocID

融合由 `weighted_semantic_cf_fusion` 实现：

1. semantic 与 CF 两个 view 各自**逐行 L2 归一化**；
2. 分别乘幅度 `sqrt(0.8)=0.894427` 与 `sqrt(0.2)=0.447214` 后拼接，得 `1056` 维——这样融合向量的平方 L2 能量里 CF 恰好占 `0.20`；
3. 补零到 `1088` 维，使其被 `M=64` 整除（子空间维 `17`）；
4. 断言融合后每行 L2 范数为 `1`，否则直接失败（fail-closed）。

CF 能量占比取 `0.20` 的依据是训练集中真正产生多 skill 协同边的 row 占 `17,457/94,912 = 18.39%`，向上取整。**该权重在读取任何测试指标之前就已冻结。**

量化用 Faiss `OPQMatrix` + `ProductQuantizer`：`M=64`、`8 bit/位`、`seed=42`、kmeans `25` iter、OPQ `20` iter、opq_pq `4` iter、单线程。位置顺序固定为 Faiss 原生顺序，不搜索排列（`position_order_policy = faiss_native_no_search`）——`64!` 的排列搜索既不可执行也会引入额外调参自由度。

冻结产物 CF20-OPQ64：

| 项 | 值 |
|---|---|
| skills | `14,551` |
| quantizer dim | `1088`（1024 semantic + 32 CF + 32 pad） |
| depth x codebook | `64 x 256` |
| subvector dim | `17` |
| bits / DocID | `512` |
| 原始碰撞组 | `0` |
| variance-normalized SSE | `0.288884` |
| energy-normalized SSE | `0.201544` |
| mean cosine | `0.895871` |
| p05 cosine | `0.861365` |
| logical semantic-ID SHA256 | `ba1bc612c278fb7c4da94a9004fb6d8a74c9d56170be627c3fad7638a19acc09` |

碰撞策略保留 LETTER 的 minimum-cost unique last code，但 CF20-OPQ64 的原始碰撞为 `0`，该修复分支未被触发。manifest 里同时记录 `codes.native.npy`（原始）与 `codes.resolved.npy`（去碰撞后），两者在本配置下相同。

## 5. Stage 2：解码器架构

共享表示：query 过 T5 encoder，decoder 只走**一步**（喂 `decoder_start_token_id`），取 `last_hidden_state[:, 0, :]` 作为 `h`。一条 query 只有一次前向、一个隐状态、没有序列。

三种头（`--head-type`）：

**`linear`**（Phase 11 基线）：`64` 个独立的 `nn.Linear(d_model, 256)`。

**`rpg_cosine_residual`**（当前 winner）：

- 每个位置一个 `ResidualProjection`：`h + SiLU(W h + b)`，其中 `W`、`b` 零初始化，因此初始状态是恒等映射；
- 可学习 code prototypes `P`，形状 `[64, 256, d_model]`；
- 投影结果与 prototype 都做 L2 归一化，`logits = cosine / temperature`，`temperature = 0.03` 固定。

**`rpg_cosine_residual_calibrated`**（已实现并验证，未胜出）：额外给 `64` 个位置各一个可学习 temperature（clamp 到 `[0.003, 0.3]`）与一组 softmax 位置权重；训练目标加 `0.1 x exact-catalog NLL + 0.01 x KL(weight || uniform)`，其中 catalog NLL 只用训练行的 gold skill 和完整合法 tuple，不接触 Dev/Official 标签。该分支在 Dev 上未胜过固定头，best checkpoint 落在 step 0。

## 6. Stage 3：训练

- 目标：`mean-position cross-entropy`——`64` 个位置各自 CE 后取平均，即 multi-token prediction。
- schedule：catalog `3` epochs（`skill document: ` 前缀的文档 -> 自身 DocID）而后 query `5` epochs。
- global batch `16`（`4` rank x `4`/卡）、AdamW、weight decay `0`、constant LR `3e-4`、BF16、seed `20260626`。
- checkpoint 以 expanded Dev mean-position CE 选 stage-best。
- 单节点 8xA800 拆成两个隔离的 4-rank DDP group（GPU `0-3` 与 `4-7`，各自独立 rendezvous port），每卡 batch 降到 `4` 以保持 global batch `16` 不变。之所以必须手动指定 rank 和 port，是因为托管 elastic launcher 会把两个 launcher 强制到同一个 rendezvous port，导致 4+4 无法隔离。

winner 的最后一段是 query-only cooldown：从上一阶段 winner 初始化，跳过 catalog stage，只再训 query `3` epochs，cosine LR `1e-4 -> 1e-5`，`max_grad_norm=1.0`，`--epoch-offset 10` 保证 shuffle 序列不与前 10 个 epoch 重复。

## 7. Stage 4：精确全库打分

`exact_catalog_log_scores`：

```python
log_probs = F.log_softmax(position_logits.float(), dim=-1)   # [B, 64, 256]
scores = 0
for p in range(64):
    scores += w[p] * log_probs[:, p, :].index_select(1, catalog_codes[:, p])
# scores: [B, 14551]
```

`w_p = 1/64` 固定（只有 calibrated 头会学 `w`）。对全部 `14,551` 个合法 catalog tuple 精确计算，再由 `stable_topk_catalog` 取 top-K。

**没有 trie、没有 beam、没有 prefix pruning、没有 reranker、没有外部候选分数、没有 dense fusion。** 候选生成的召回损失恒为 `0`，因此评测结果测量的是模型上限，而不是模型质量与近似搜索漏召回的混合。

## 8. 与 MIPS / PQ-ADC 的等价关系

每个位置 `log_softmax` 的归一化项与文档无关，所以文档间的排序等价于

```text
rank ∝ sum_p < normalize(proj_p(h)), normalize(P[p, code_p(d)]) >
```

即一个**分块结构的 MIPS**（结构化双塔）。等价地：先算一张 `64 x 256` 的 query-code 分数表，再对每个文档做 `64` 次查表求和——这正是 PQ 的 ADC（asymmetric distance computation）形式。两点含义：

- 训练与评测阶段用 exact 全库打分，测的是上限，不混入近似搜索误差；
- 若要上线，可直接替换为 IVF-PQ / ADC 查表，模型本身不需要改。

## 9. Official Core75 结果

Official SkillsBench v1.1 Core75：`75` 条 query（single `24` / multi `51`），`200` 个 GT mention。`R` 为 macro recall，所有指标逐 query 平均。

| 系统 | H@1 | H@10 | H@50 | R@1 | R@10 | R@50 | FC@10 | FC@50 |
|---|---|---|---|---|---|---|---|---|
| LETTER（RQ-VAE D4x256，AR+trie） | 0.6000 | 0.7467 | 0.7733 | 0.2837 | 0.6900 | 0.7500 | 0.6133 | 0.7333 |
| RIPOR | 0.7067 | 0.8667 | 0.8933 | 0.3702 | 0.7035 | 0.7429 | 0.5200 | 0.5733 |
| RIPOR+PAG | 0.7600 | 0.8933 | 0.9600 | 0.3896 | 0.7827 | 0.8800 | 0.6533 | 0.7867 |
| OPQ-64 + 自回归 trie（H1） | 0.6800 | 0.8000 | 0.8133 | 0.3059 | 0.7011 | 0.7233 | 0.6267 | 0.6533 |
| OPQ-64 + 并行（H2，semantic only） | 0.6667 | 0.8400 | 0.8800 | 0.3181 | 0.7644 | 0.8144 | 0.6933 | 0.7600 |
| **CF20-OPQ-64 + 并行 + cooldown** | **0.7333** | 0.8533 | 0.9200 | 0.3615 | **0.8256** | 0.8756 | **0.8133** | **0.8267** |

在同一个 OPQ-64 tokenizer 下，把自回归 trie 换成并行精确打分使 R@10 从 `0.7011` 升到 `0.7644`、R@50 从 `0.7233` 升到 `0.8144`、H@50 从 `0.8133` 升到 `0.8800`；再叠加 CF20 与 cooldown 到 R@10 `0.8256` / R@50 `0.8756` / FC@10 `0.8133`。

winner = `cf20_rpg_cooldown_e3`，checkpoint SHA256 `71bafc3c63dd41a7389a8ee35552545c06fe599adc9c91f55860e1e3f2443a7c`。

单/多子集拆分（`FC@1` 对 multi 按定义为 `0`，故 `FC@1 x 75` 恰为 single 子集的 top-1 命中数，配合 `H@1 x 75` 可反解）：

| 子集 | top-1 命中 |
|---|---|
| single（24） | `14/24` |
| multi（51） | `41/51` |

multi top-1 是所有已测系统中最好的；single top-1 与 BM25 持平，是当前唯一明显落后的格子。

## 10. 复现

依赖：`faiss-cpu==1.13.2`（见 `requirements-image.txt`）、torch、transformers。`PYTHONPATH` 需包含 `scripts/`，因为 `skill_gr_opq_parallel.py` 直接 import `skill_gr_pipeline` 与 `distributed_utils`。

Stage 1：

```bash
PYTHONPATH=scripts python3 scripts/build_letter_opq_docids.py \
  --embeddings   <skill_semantic_embeddings.npz> \
  --skills-jsonl <canonical_skills.jsonl> \
  --cf-embeddings   <lightgcn_skill_embeddings.npz> \
  --cf-skills-jsonl <canonical_skills.jsonl> \
  --cf-energy-fraction 0.2 \
  --output-dir <docid_out> \
  --depth 64 --seed 42
```

builder 会拒绝写入非空目录，并在 manifest 里落全部输入的 SHA256、逐位置 code usage/entropy、每个前缀长度的 trie 增长，以及 logical semantic-ID hash。

Stage 2/3/4 的完整命令见：

- `repro/20260902_letter_opq_32_64/` —— Phase 11，M32 vs M64、AR vs parallel；
- `repro/20260903_letter_opq_ceiling/` —— Phase 12，`{semantic, cf20} x {linear, rpg}` 四组强配置矩阵；
- `repro/20260903_letter_opq_cooldown_calibration/` —— Phase 13，winner cooldown 与 position calibration。

这些 runner 对每一个输入（训练/dev 行、catalog、DocID、manifest、base model 权重与 tokenizer、initializer checkpoint）做 SHA256 fail-closed 校验，把冻结协议写进 `experiment_contract.json`，并在训练后重新校验最终 checkpoint 的 architecture / head_type / semantic-ID hash / optimizer schedule / initializer provenance。**Official 评测由独立 runner 触发，候选 runner 只跑 Dev**，以保证配置选择在读取 Official 之前就已冻结。

## 11. 代码地图

| 文件 | 作用 |
|---|---|
| `scripts/probe_letter_opq.py` | OPQ/PQ/RQ 离线 identifier probe；同时提供被 builder 复用的量化拟合、碰撞统计、reconstruction 指标、SHA256/IO 原语 |
| `scripts/build_letter_opq_docids.py` | semantic-only 或 CF-aware OPQ DocID 构建 + 完整 manifest |
| `scripts/skill_gr_opq_parallel.py` | 并行 MTP 解码器，`train` / `eval` 两个子命令 |
| `scripts/select_letter_opq_ceiling.py` | 按冻结 Dev 规则从候选池选 winner |
| `tests/test_build_letter_opq_docids.py` | 融合能量配比、补零、范数断言、manifest 字段 |
| `tests/test_probe_letter_opq.py` | probe 的确定性与 provenance |
| `tests/test_skill_gr_opq_parallel.py` | SID 校验、MTP loss、exact catalog 打分、legacy checkpoint 兼容、tiny overfit |
| `tests/test_letter_opq_parallel_runner.py`、`test_letter_opq_split_runner.py` | 4+4 DDP 拆分与 runner 契约 |
| `tests/test_letter_opq_ceiling_runner.py`、`test_letter_opq_cooldown_runner.py` | 强配置矩阵与 cooldown/calibration 的冻结协议、Official gate |
| `tests/test_select_letter_opq_ceiling.py` | Dev 字典序选择规则 |
| `repro/20260902_letter_opq_32_64/` | Phase 11 runner |
| `repro/20260903_letter_opq_ceiling/` | Phase 12 runner |
| `repro/20260903_letter_opq_cooldown_calibration/` | Phase 13 runner |

仓库内已有的依赖模块：`scripts/skill_gr_pipeline.py`（数据与 semantic-ID 加载、评测口径）、`scripts/distributed_utils.py`（DDP 原语、无 padding 的 epoch 分片、global normalized loss）。

选择协议：候选按 Dev overall macro recall 的 `(R@10, R@5, R@1, R@50)` 字典序取最大，只有胜者跑一次 Official。

## 12. 已知问题与下一步

- `opq_docid_manifest.json` 的 `decoder_pipeline` 字段仍写着 `letter_sequential_autoregressive_trie`。这是 H1 阶段遗留的常量，与实际使用的并行解码器不符。只影响 manifest 可读性，不参与任何计算。
- Core75 只有 `75` 条 query，`1` 条 = `1.33pp`。cooldown 相对上一阶段的 `+0.22pp` 在这个分辨率下不可判定。配置选择应迁移到更大的 canonicalized 测试集。
- 相对最强 baseline（SkillRouter-0.6B + reranker，H@1 `0.8000`）的缺口全部集中在 single 子集：single top-1 `14/24` vs `20/24`，而 multi top-1 `41/51` 与 deep recall、FC 均已领先。本仓库五代 GR 系统在 single top-1 上一致不超过自己的输入 embedding（Dense Qwen3-0.6B 为 `15/24`），指向 `512 bit/skill` 的量化瓶颈与 query 侧表示容量，而非解码器。
- 训练目标（逐位置 CE）与评测目标（全库排序）不一致：CE 从不告诉模型「某个错误文档应该排更低」。`exact_catalog_log_scores` 已经可微且全库只有 `14,551` 项，可以对同一个分数直接加 listwise ranking loss，并用 multi-positive 形式表达集合目标。calibrated 分支虽然带了 `0.1x` catalog NLL，但与可学习 temperature、位置权重、KL 三项同时上线，是一个被混淆的实验，不能作为「排序损失无效」的证据。
- 并行解码器把 `64` 个位置建模为条件独立的边缘分布，再按等权乘积打分。加长 DocID（`M=128`）在这个架构下几乎免费，是尚未测过的容量轴；但每位置贡献从 `1/64` 降到 `1/128`，容量与可学性方向相反，只能实测。

