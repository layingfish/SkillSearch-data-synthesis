# GR Pipeline Experiment Summary 2026-06-27

本文档记录当前 2000 条合成数据上的生成式检索实验结果、消融、现象和初步解释。它只整理已经跑过的实验，不包含新的方法设计。

> 2026-07-11 指标更正：本文档原始表格中的 Recall 使用历史 micro 聚合。当前正式口径已对齐 SkillRouter，使用 per-query Recall 的 macro average；重算结果见 `outputs/analysis/skillgr_macro_recall_20260711/results.json`。Hit/FC 不受影响。

## 1. 实验设置

数据：

```text
dataset: outputs/eval/skillrouter_balanced_500_each_topup_v1/example_2000.jsonl
split: outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/decoder_split_rqvae_e100_c256_d3_seed20260627
train/dev/test: 1800 / 100 / 100
dev/test: k=1/2/3/4 each has 25 rows
unique selected skills: 3539
```

Skill encoding：

```text
skill text = name | description | body
embedding model = Qwen/Qwen3-Embedding-0.6B
```

RQ-VAE：

```text
output: outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/rqvae_e100_c256_d3_h512_z128_seed20260627
codebook_size: 256
depth: 3
semantic id: [c0, c1, c2, dedup]
unique skills: 3539
raw_unique_codes: 3388 / 3539
```

指标：

```text
Hit@K          = fraction of rows where top-K ranked skills contain at least one gold skill
Recall@K       = mean per-query recovered-gold fraction, macro averaged (current official metric)
micro_recall   = total_hits@K / total_gold (historical tables in this document)
FC@K           = Full Coverage@K, fraction of rows where all gold skills are covered by top-K
```

这里的 `@K` 是 ranked skill list 的截断深度，不是 beam size。生成式模型统一使用 `num_beams=50` 评估 `K={1,10,50}`。

## 2. 模型与消融

已经评估的系统：

```text
BM25:
lexical retrieval over 3539 selected-skill documents.

T5 dense:
vanilla t5-base encoder mean pooling + cosine retrieval. No finetuning.

Qwen3 dense:
Qwen/Qwen3-Embedding-0.6B last-token pooling + cosine retrieval. No finetuning.
Queries use the SkillRouter/Qwen instruction prefix stored in query_text.

multi-SID:
query -> multiple skill SIDs + stop.
This is the original structured multi-SID set generation route.

multi-SID + catalog:
stage 1: skill document -> single SID
stage 2: query -> multiple skill SIDs + stop
This is the correct "catalog-only" ablation for the original multi-SID route.

single-SID:
query -> one positive skill SID.
Multi-skill queries are expanded into multiple single-positive examples.
Inference uses beam top-K skills.

single-SID + catalog:
stage 1: skill document -> single SID
stage 2: query -> one positive skill SID
Inference uses beam top-K skills.
```

不要把误跑过的 `catalog zero-shot` 当作正式结果。它的含义是只训练 `skill document -> SID`、不做 query finetune；这不是当前表格里的 `multi-SID + catalog`。

## 3. Dev Results

Dev overall：

```text
model                  Hit@1  Recall@1  FC@1   Hit@10  Recall@10  FC@10  Hit@50  Recall@50  FC@50
BM25                   0.70   0.280     0.13   0.94    0.712      0.54   0.98    0.876      0.75
T5 dense               0.01   0.004     0.00   0.12    0.056      0.03   0.25    0.112      0.08
multi-SID              0.07   0.028     0.00   0.35    0.164      0.07   0.48    0.252      0.13
multi-SID + catalog    0.09   0.036     0.01   0.32    0.188      0.07   0.54    0.308      0.22
single-SID             0.08   0.032     0.00   0.27    0.168      0.06   0.61    0.436      0.25
single-SID + catalog   0.11   0.044     0.00   0.31    0.188      0.07   0.63    0.424      0.25
```

Dev `@50` summary：

```text
multi-SID              Recall@50 0.252, FC@50 0.13
multi-SID + catalog    Recall@50 0.308, FC@50 0.22
single-SID             Recall@50 0.436, FC@50 0.25
single-SID + catalog   Recall@50 0.424, FC@50 0.25
```

Dev seen/unseen `Recall@50`：

```text
dev gold assignments: 250
seen in train:   115 / 250 = 46.0%
unseen in train: 135 / 250 = 54.0%

BM25:                  seen 0.904, unseen 0.852
multi-SID:             seen 0.365, unseen 0.156
single-SID + catalog:  seen 0.643, unseen 0.237
```

## 4. Test Results

Test overall：

```text
model                  Hit@1  Recall@1  FC@1   Hit@10  Recall@10  FC@10  Hit@50  Recall@50  FC@50
BM25                   0.63   0.252     0.18   0.93    0.640      0.37   0.96    0.840      0.68
T5 dense               0.04   0.016     0.00   0.11    0.044      0.02   0.30    0.148      0.07
Qwen3 dense            0.58   0.232     0.19   0.95    0.636      0.42   1.00    0.844      0.70
multi-SID              0.02   0.008     0.00   0.26    0.116      0.04   0.43    0.208      0.13
multi-SID + catalog    0.07   0.028     0.01   0.44    0.224      0.07   0.57    0.312      0.20
single-SID             0.07   0.028     0.00   0.29    0.140      0.02   0.65    0.448      0.21
single-SID + catalog   0.09   0.036     0.00   0.32    0.164      0.02   0.70    0.444      0.24
```

Test `@50` by target skill count：

```text
model                  gold_k  Hit@50  Recall@50  FC@50
BM25                   1       0.84    0.840      0.84
BM25                   2       1.00    0.920      0.84
BM25                   3       1.00    0.773      0.48
BM25                   4       1.00    0.850      0.56

T5 dense               1       0.24    0.240      0.24
T5 dense               2       0.24    0.140      0.04
T5 dense               3       0.40    0.187      0.00
T5 dense               4       0.32    0.100      0.00

Qwen3 dense            1       1.00    1.000      1.00
Qwen3 dense            2       1.00    0.940      0.88
Qwen3 dense            3       1.00    0.853      0.64
Qwen3 dense            4       1.00    0.750      0.28

multi-SID              1       0.32    0.320      0.32
multi-SID              2       0.60    0.380      0.16
multi-SID              3       0.48    0.213      0.04
multi-SID              4       0.32    0.090      0.00

multi-SID + catalog    1       0.52    0.520      0.52
multi-SID + catalog    2       0.68    0.440      0.20
multi-SID + catalog    3       0.72    0.400      0.08
multi-SID + catalog    4       0.36    0.130      0.00

single-SID             1       0.28    0.280      0.28
single-SID             2       0.68    0.500      0.32
single-SID             3       0.84    0.533      0.20
single-SID             4       0.80    0.400      0.04

single-SID + catalog   1       0.36    0.360      0.36
single-SID + catalog   2       0.72    0.520      0.32
single-SID + catalog   3       0.88    0.520      0.24
single-SID + catalog   4       0.84    0.370      0.04
```

## 5. Seen / Unseen Analysis on Test

Train 中是否出现过 gold skill assignment 的比例：

```text
gold_k=1: seen  2 / 25  = 0.080
gold_k=2: seen 31 / 50  = 0.620
gold_k=3: seen 46 / 75  = 0.613
gold_k=4: seen 45 / 100 = 0.450
```

这说明 test 的 `k=1` 几乎是 cold-start skill 测试，因此不能简单把 `k=1` 结果解读为 single-skill query 本身更难。

Test `@50` seen/unseen recall by `gold_k`：

```text
BM25
k=1: seen 1.000, unseen 0.826
k=2: seen 0.935, unseen 0.895
k=3: seen 0.804, unseen 0.724
k=4: seen 0.889, unseen 0.818

multi-SID
k=1: seen 0.500, unseen 0.304
k=2: seen 0.419, unseen 0.316
k=3: seen 0.217, unseen 0.207
k=4: seen 0.178, unseen 0.018

multi-SID + catalog
k=1: seen 0.500, unseen 0.522
k=2: seen 0.581, unseen 0.211
k=3: seen 0.522, unseen 0.207
k=4: seen 0.200, unseen 0.073

single-SID
k=1: seen 0.000, unseen 0.304
k=2: seen 0.677, unseen 0.211
k=3: seen 0.630, unseen 0.379
k=4: seen 0.467, unseen 0.345

single-SID + catalog
k=1: seen 0.000, unseen 0.391
k=2: seen 0.677, unseen 0.263
k=3: seen 0.630, unseen 0.345
k=4: seen 0.511, unseen 0.255
```

## 6. Current Phenomena

### 6.1 数据是有效的

BM25 在 dev/test 上都很强：

```text
dev  Recall@50 = 0.876, FC@50 = 0.75
test Recall@50 = 0.840, FC@50 = 0.68
```

这说明 query 与 skill 文档之间有明显可检索信号，当前 2000 条数据不是随机噪声。

### 6.2 Dense retrieval strongly depends on the embedding model

未微调 `t5-base` encoder dense retrieval 很弱：

```text
dev  Recall@50 = 0.112
test Recall@50 = 0.148
```

但 `Qwen/Qwen3-Embedding-0.6B` dense retrieval 在 test 上已经和 BM25 同级：

```text
Qwen3 dense test Recall@50 = 0.844
Qwen3 dense test FC@50     = 0.70
BM25 test Recall@50        = 0.840
BM25 test FC@50            = 0.68
```

所以不能把 `T5 dense` 的弱结果解读成 dense retrieval 本身不适合该任务。更准确的结论是：vanilla T5 encoder 不适合作为 embedding baseline，而强 embedding model 进一步证明当前 query-skill semantic signal 很强。

### 6.3 生成式检索能学到东西，但仍显著弱于 BM25

所有生成式路线都高于 vanilla T5 dense，但都远低于 BM25：

```text
best generative test Recall@50: 0.448
BM25 test Recall@50:            0.840
Qwen3 dense test Recall@50:     0.844
```

这说明 GR pipeline 有学习信号，但当前瓶颈仍明显。

### 6.4 single-SID formulation 是主要收益来源

从 test 看：

```text
multi-SID  Recall@50 = 0.208
single-SID Recall@50 = 0.448
```

将多 skill query 拆成多个 `(query, one positive skill SID)`，再用 beam top-K 形成 ranked skill list，是当前最有效的 formulation 修正。

### 6.5 catalog indexing 对 multi-SID 有帮助，但不能救完整 set generation

从 test 看：

```text
multi-SID            Recall@50 = 0.208, FC@50 = 0.13
multi-SID + catalog  Recall@50 = 0.312, FC@50 = 0.20
```

catalog indexing 能帮助模型对齐 skill document 与 SID，但 multi-SID set generation 在 `k=4` 仍然明显崩溃：

```text
multi-SID + catalog, k=4:
Recall@50 = 0.130
FC@50     = 0.00
```

### 6.6 catalog indexing 对 single-SID 的收益不稳定

从 test 看：

```text
single-SID            Recall@50 = 0.448, FC@50 = 0.21
single-SID + catalog  Recall@50 = 0.444, FC@50 = 0.24
```

catalog 对 `single-SID` 的整体 `Recall@50` 没有提升，但略提高 `Hit@1/Hit@10/FC@50`。当前不能说 catalog 对 single-SID 稳定有效。

### 6.7 不同 k 上的现象不单调

`multi-SID` 在 `k=4` 明显崩溃，像是 sequence/set generation 的误差累积。`single-SID` 在 `k=2/3/4` 上 partial recall 更好，因为它只需要 ranked list 覆盖 gold skills 的一部分，不需要一次生成完整 set。

但 `FC@50` 在多 skill 场景仍低：

```text
single-SID + catalog FC@50:
k=2: 0.32
k=3: 0.24
k=4: 0.04
```

这说明 single-SID 能找出部分相关 skills，但完整覆盖多 skill target 仍很弱。

### 6.8 seen/unseen 必须成为固定报告项

test 的 `k=1` 只有 `2/25` gold assignments seen in train，而 `k=2/3` 的 seen 比例超过 0.6。不同 k 的表现受 cold-start 比例强烈影响。之后报告结果必须同时包含：

```text
overall
by gold_k
by seen/unseen
by gold_k × seen/unseen
```

否则容易误判不同 k 的真实难度。

## 7. Current Interpretation

目前比较稳的判断：

```text
1. 2000 条数据有效，BM25 和 Qwen3 dense 结果说明 query-skill lexical/semantic signal 都很强。
2. 当前 GR pipeline 有学习能力，但远未达到可用水平。
3. multi-SID set generation 不是当前最合适的主线，尤其 k=4 误差累积严重。
4. single-SID top-K retrieval 是当前更合理的主线。
5. catalog indexing 可以帮助 multi-SID，但对 single-SID 的收益不稳定。
6. 主要瓶颈大概率在 SID 表示质量和 decoder 对 unseen SID 的泛化。
```

## 8. Suggested Next Diagnostics

下一步不建议继续盲调 `lr/epoch/beam size`。更应该先诊断 SID 和 decoder 学习过程：

```text
1. SID locality:
   - sharing c0 / c0,c1 是否真的表示语义相近 skills
   - prefix fanout 是否过度不均衡
   - RQ-VAE collisions and dedup distribution

2. per-level teacher-forcing accuracy:
   - c0 / c1 / c2 / dedup 哪一层最容易错
   - seen vs unseen 分开看
   - gold_k 分开看

3. single-SID ranked errors:
   - top-50 miss 的 gold skills 是否 lexical BM25 可找回
   - miss 是否集中在特定 SID prefix

4. SID construction comparison:
   - 如果 RQ-VAE locality 差，最小对照 hierarchical k-means / balanced tree SID
```

当前最重要的判断边界是：如果 SID 本身像 hash code，而不是有可学习的语义结构，那么继续调 decoder 超参很难带来跃升。

## 9. Key Output Paths

```text
base:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803

BM25:
baselines/bm25_dev
baselines/bm25_test

T5 dense:
baselines/t5_base_dense_dev
baselines/t5_base_dense_test

Qwen3 dense:
baselines/qwen3_embedding_06b_test

multi-SID:
structured_t5_decoder_rqvae_e100_split_seed20260627_steps2000

multi-SID + catalog:
structured_catalog_t5_base_rqvae_e100_split_seed20260627_cat3_steps2000

single-SID:
single_sid_query_only_t5_base_rqvae_e100_split_seed20260627_q3000

single-SID + catalog:
single_sid_catalog_t5_base_rqvae_e100_split_seed20260627_cat3_q3000
```
