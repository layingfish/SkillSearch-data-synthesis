# SkillSearch Data Synthesis Handoff

更新时间：2026-07-15

本文档最前面的 **“当前接手状态”** 是最新事实源。后面保留了历史 handoff；接手 agent 应优先读当前状态，不要把旧的 Reference-400 replay 主线或旧的 micro Recall 表误认为当前主线。

## 0. 当前接手状态（2026-07-14）

### 0.0 Recall 指标更正（2026-07-11）

SkillRouter 官方口径是先对每条 query 计算 `|TopK(q) ∩ GT(q)| / |GT(q)|`，再对 query 等权平均，即 macro Recall。项目此前的 ranked evaluator 把 `total_hits / total_gold` 作为 `recall`，属于 micro Recall，未完全对齐官方聚合方式。

当前统一规则：

```text
Hit@K          = mean_q 1[TopK(q) contains any GT skill]
Recall@K       = mean_q (|TopK(q) ∩ GT(q)| / |GT(q)|), macro averaged
FC@K           = mean_q 1[all GT skills are covered by TopK(q)]
micro_recall   = sum_q hits_q / sum_q |GT(q)|, retained only as a diagnostic
```

代码与产物状态：

- `scripts/skill_gr_pipeline.py::compute_at_k_metrics` 的 `recall` 已改为 macro，并额外输出 `macro_recall`、`micro_recall` 和 `recall_aggregation=macro`。
- `scripts/build_skill_results_explorer.py` 的 run-level 公式和展示已同步为 macro。
- Notion `SkillGR` 页面中的 2k、20k、single/multi subset、pseudo-query 全部结果已重算。
- 44 个页面相关 artifact 已从原始 rankings 重算；审计报告位于 `outputs/analysis/skillgr_macro_recall_20260711/results.json`。
- 本文档后续章节中 2026-07-11 以前记录的 Recall 数值若未明确标为 macro，应视为历史 micro 结果；正式引用以审计报告和已更新的 Notion 页面为准。Hit/FC 不受本次更正影响。

### 0.0.1 GR 工程配置审计（2026-07-13）

在开始 full-catalog task-style pseudo-query 合成前，已系统检查 `scripts/skill_gr_pipeline.py` 中可能人为限制 GR 性能的设置。这里必须区分两层：

1. 数据级 canonical view：query `[:1500]`、description `[:300]`、skill body `[:2500]`。这是与 SkillsRouter 对齐的信息预算，必须保留。
2. 模型级 tokenizer 上限：structured/single T5 原来固定为 512 tokens，可能在 canonical view 上再次截断。这是 GR 自身的额外信息损失，应消除。

在保留 SkillsRouter canonical view 后，official core75 仍有 12/75 条 query 超过 512 T5 tokens，最长 678；catalog doc-aug 输入中有 13,029/15,083 超过 512，最长 1,718。因此真正需要修复的是第二层，而不是取消 1,500/300/2,500 的共同数据截断。

当前代码已调整为：

- GR 运行时从原始 `query` 稳定重建 `query[:1500]` + SkillRouter instruction，不依赖可能不一致的缓存字段。
- `prepare --query-max-chars` 默认保持 1,500；description/body 默认保持 300/2,500。
- main structured/single GR 的 `max_source_length` 统一为 2,048，确保完整读取上述 canonical view；legacy text decoder 与 T5-dense diagnostic 的旧默认不变。
- 训练、续训和评测前都会记录 source token-length audit；若 canonical input 仍超过上限，默认直接失败，只有显式传入 `--allow-source-truncation` 才允许截断。
- 新 checkpoint 保存 `max_source_length`、source-text policy、SemID 数量和 SemID SHA-256；续训/评测会校验，防止模型与错误 SemID 或预处理配置静默错配。
- legacy checkpoint 没有保存长度配置，评测时必须显式指定 `--max-source-length`，不再猜默认值。
- `num_beams=50`、multi `max_skills=4` 和 single `max_skills=1` 均保持原实验语义，不根据 official test 反向调参。
- fallback dev split 改为先按原始 query 切分、再展开 multi-positive rows，避免同一 query 同时进入 train/dev。
- eval 会报告 candidate/SemID pool 中不可达的 GT、rank capacity 和实际 ranked-list 长度。

使用当前 canonical view 和 `t5-base` tokenizer 的输入审计：

```text
train:    max=401,  over 2048=0/19,487
dev:      max=360,  over 2048=0/1,000
catalog:  max=1,718, over 2048=0/15,083
official: max=678,  over 2048=0/75
```

SemID 上游的 Qwen3 skill encoder 也已用其实际 tokenizer 复核：`skillrouter_text` 最长 1,742 tokens，`over 2048=0/15,083`，因此 encode-skills 阶段没有额外 token 截断。`skillrouter_text` 仍采用 `description[:300] + body[:2500]`，与 SkillsRouter 保持相同信息预算。

在对齐输入上进行的 legacy checkpoint 诊断：beam=50/100/200 的 macro `R@10` 都是 0.4582；beam=50 的 `R@50=0.5811`，beam=100/200 为 0.5744。因此保持原始 beam=50，不把 official test 当作 beam 调参集。旧 checkpoint 只用于工程诊断；下一轮模型必须从训练阶段就统一使用 canonical view + T5 2,048 上限。此前 full raw query 的诊断不符合对齐口径，禁止作为正式结果引用。

另一个已暴露但尚未解决的数据覆盖问题：当前 15,083 SemID pool 缺少 official GT `gt/mesh-analysis` 和 `gt/senior-data-engineer`，涉及 2/75 rows，且都属于 24 条 official single。下一轮 full-catalog 数据和 SemID 重建前必须先决定是否补齐完整 benchmark catalog，否则这两条在 GR 中永久不可达。

验证：远端含 PyTorch 环境中 `tests/test_skill_gr_pipeline.py` 共 28 项全部通过；本地 `compileall` 通过。

### 0.0.2 LLM 生成端点决策（2026-07-13）

OpenRouter 已从后续数据生成链路中废弃。历史 OpenRouter outputs、manifest、profile 名称和复现命令继续保留，且不得改写，以维持既有实验的可追溯性；任何新的 pseudo-query、single/multi replay 或 repair 任务统一使用：

```text
profiles/gpt4o_huayanapi.json
```

实测 `gpt4o_huayanapi` 支持 OpenAI-compatible `n` 参数，可在一次请求中返回多个独立 completion；pilot 返回 `gpt-4o-2024-11-20`，但 full run 对 `gpt-4o` alias 的实际路由包含 14,829 个 `gpt-4o-2024-11-20` 和 254 个 `gpt-4o-2024-08-06`，不能把 alias 当成冻结 snapshot。OpenRouter 端点会静默忽略 `n`，因此禁止用于后续 fixed-budget multi-sample query generation。API key 只保存在被 `.gitignore` 忽略的 profile 中，不得提交。

### 0.0.3 Full-catalog task-style pseudo-query 合成（2026-07-13）

当前 clean 主线已经从 official-single transductive 诊断切换为全训练 catalog 合成：输入只读取 `prepare_train/selected_skills.jsonl` 中的 15,083 个训练 skill，不读取 Official rows、GT 或 query。目标是给每个 skill 恰好生成 5 条 single-skill task-style query，共 75,415 条；与 19,487 条 base rows 合并后应为 94,902 条。

固定预算如下：

```text
每个 skill 1 次成功 API 请求
每次请求 n=8 independent completions
temperature=0.7, top_p=0.9, max_tokens=512
本地 leakage filter + exact dedup + MMR 选择 5 条
不做 data-quality retry；429/5xx 等 transport retry 不改变成功请求预算
query/description/body canonical view = 1500/300/2500 chars
```

实现入口：

```text
scripts/build_longtail_pseudo_queries.py generate-catalog
scripts/build_longtail_pseudo_queries.py augment-catalog
tests/test_longtail_pseudo_queries.py
```

clean 路径禁止 OpenRouter，并使用单个 user message 的冻结 task-spec prompt。生成记录按 skill 追加到 `catalog_generations.jsonl`，重启后跳过已有 `complete` / `insufficient_candidates` skill；最终物化会重新执行当前版本的本地选择，并强制校验每个 catalog skill 恰好 5 条、单 skill 标签、query 唯一、长度不超过 1,500、无内部 skill identifier 和无 Official/transductive metadata。

对于 body 中不可避免地包含内部同名命令或路径的 skill，例如 `/add-home-widget-selector`、`.claude/skills/ansible-fix/...`，选择前会把包含内部名称的整个 token 确定性替换为 `<PROJECT_COMMAND>`。训练行在 `pseudo_query.reference_sanitized` 中记录该事实；不会为此追加 LLM 请求。

16-skill 跨域 pilot：

```text
outputs/eval/full_catalog_task5_pilot_20260713
16/16 skills complete
80/80 selected rows
You need to / numbered requirements = 100% / 100%
pairwise token-Jaccard mean = 0.3724
source boundary-truncated = 1/80（完整句边界，非半句硬截断）
usage = 16,864 input + 33,480 output tokens
```

远端 full run 已完成：

```text
output dir: outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713
catalog_generations.jsonl: 15,088 records（15,083 初次结果 + 5 个 transport-failure 续跑结果）
selected_task5.jsonl: 75,415 rows
decoder_rows.base_plus_task5.jsonl: 94,902 rows
failures.jsonl: 0 rows
usage: 14,842,001 prompt + 30,992,568 completion = 45,834,569 tokens
workers: 4
```

5 个孤立 transport failures（2 timeout、3 HTTP 502）已通过同目录续跑恢复，没有 data-quality retry。当前 transport helper 覆盖 `TimeoutError`、retryable HTTP/URL errors，使用最多 6 次 capped exponential backoff + jitter，并遵循数值型 `Retry-After`。恢复时对同一 output directory 重跑完全相同的 `generate-catalog` 命令即可；`run_config.json` 会拒绝参数、catalog 顺序或模型配置发生变化的续跑。

最终 query 审计：

```text
15,083/15,083 skills，全部恰好 5 条
75,415/75,415 unique indices and globally unique queries
0 missing/extra skills
0 skill identifier leaks
0 JSON wrappers
0 bare-SQL pseudo queries
0 label/query_text contract errors
0 Official/transductive metadata
0 queries over 1,500 chars
mean chars/words = 1,129.21 / 168.52
You need to rate = 99.951%
numbered-line rate = 97.902%
pairwise token-Jaccard mean = 0.3301
source boundary-truncated rate = 3.201%
internal-reference sanitized rate = 2.599%
task-spec override = 10 rows（5 Galaxy meta-query + 5 Snowflake SQL input）
```

两个 skill 的 body 含会劫持生成器的内层输出指令：`other/optimizing-query-text` 会只返回 SQL，`other/galaxy-query-generation` 会直接返回 Galaxy tool-recommendation query。最终选择器对这 10 条做了 skill-specific、grounded、确定性的 task-spec 包装；没有增加 API 请求，并在 `pseudo_query.task_spec_override` 中留痕。截断 JSON completion 会恢复首个 `query` 字符串，不再把 JSON wrapper 写入训练数据。

关键 SHA-256：

```text
selected_task5.jsonl              7aae0b597298867a512bdeea82b899f6e129dfa2f83f8db53beb0c515bf79326
decoder_rows.base_plus_task5.jsonl 6ef8507e4f5d8c9b5fe3f5a58fcf1ee54575b9e71088cedd5964bf4509d9ae3a
manifest.json                     2b0ad4c992945fe20234121acb99ed57f1498d60b7c6f925338ee0da43ec0f95
augment_manifest.json             89c78e7351e8fb965ced7f550999fe8324e9a63d7146bf8e3f4543a8c2cb314f
```

验证：本地专项测试 20/20，远端完整测试 64/64。下一步不再合成同类数据，而是用 `decoder_rows.base_plus_task5.jsonl` 续训当前 single-SID GR checkpoint，并按冻结设置评测 own test 与 Official core75，重点报告 Official single。

### 0.0.4 Learned-only SemID 与 exact-document alias（2026-07-13）

此前 SemID 实际格式是 `[rq0, rq1, rq2, dedup_ordinal]`。最后一列只按同码 skill 的字典序编号，不是 RQ-VAE 学出的 code；当前 15,083-skill D3 产物中有 1,924 个 raw-code collision groups、4,799 个 collided items，2,875 个 skill 使用非零 dedup。为避免把任意编号当作语义码，现已增加纯 learned schema：

```text
legacy:        semantic_id_mode=rq_codes_with_dedup -> [rq0, rq1, rq2, dedup]
new mainline:  semantic_id_mode=learned_codes       -> [rq0, rq1, rq2, rq3]
```

兼容边界：

- 旧 semantic-ID 文件未声明 `semantic_id_mode` 时仍按 legacy 读取，旧 checkpoint 的 head、token offset 和解码状态机保持不变。
- 新 checkpoint 显式保存 schema；训练、续训和评测在 `load_state_dict` 前检查 schema。
- `learned_codes` decoder 只有 4 个 code heads 和 1 个 continue/stop control head，不创建 `dedup_head`。
- 因输出空间和模型参数已改变，legacy checkpoint 不能续训为 learned-only 模型；新 D4 decoder 必须从头训练。

在 tokenizer 前先按 whitespace-normalized `skillrouter_text` 合并完全相同的 retrieval document。canonical representative 固定取字典序最小 `skill_id`，不读取 Official query/GT。真实 catalog 审计：

```text
source skills:              15,083
canonical skills:           14,549
duplicate document groups:     441
redundant aliases removed:      534
max aliases in one group:         6
Official core75 GT remapped:       0
```

所有 94,902 条 base+task5 query 已原样保留并仅重映射 label；4,274 行命中 alias，合并后只去掉 10 个重复 label mention，无 unknown skill。canonical 后 14,549/14,549 skills 都有训练 query，最少 6 条，`<5` 为 0。Official 中原本缺失的 `gt/mesh-analysis` 与 `gt/senior-data-engineer` 仍缺失；clean training-only canonicalization 不会偷偷补入 test skill。

已物化产物：

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_learned_d4_inputs/
  canonical_skills.jsonl
  skill_aliases.jsonl
  skill_alias_map.json
  canonical_skill_embeddings.pt        # [14,549, 1,024]，与 canonical catalog 同序
  rows_000.decoder_rows.base_plus_task5.jsonl
  alias_manifest.json
```

LETTER D4 先执行官方 collision-time Sinkhorn pass。若不同 canonical documents 仍共享完整 code tuple，则对该 prefix 下全部 item 使用最后一层 residual-to-codebook squared distance 构造代价矩阵，并用 Hungarian minimum-cost assignment 联合分配互异末码；写盘前强制验证完整 tuple 零碰撞，容量不足直接失败，不再追加 dedup ordinal。

用现有 D4 checkpoint 对 canonical 14,549 skills 做容量审计：前三层共有 11,953 个 prefix，最大 prefix 仅 14 items；44 个 prefix 需要唯一化，待修复 prefix 最大也是 14，远低于末层 codebook 容量 256。

关键 SHA-256：

```text
canonical_skills.jsonl                         216315a154335f608566081e6bd9f921b0d13e34130f501db09d349f8f5f987a
canonical_skill_embeddings.pt                  08a9acf7115d3a748b197180d84f0c6e29e7eb73e969b7c846142a9df56f3920
rows_000.decoder_rows.base_plus_task5.jsonl     94627afd2befdc577d11ace3403584f603e4417848356d4dd163b32c02e02868
alias_manifest.json                             d8467ee8fa0d382ba624e8c49c1f92f3c070a1c811138fc8b222dc6d4e08eb0c
```

正式 learned-only tokenizer 命令：

```bash
python3 scripts/train_letter_rqvae_official.py \
  --letter-rqvae-dir external/LETTER/RQ-VAE \
  --embeddings outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_learned_d4_inputs/canonical_skill_embeddings.pt \
  --rows outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_learned_d4_inputs/rows_000.decoder_rows.base_plus_task5.jsonl \
  --output-dir outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/letter_d4_learned_codes \
  --semantic-id-mode learned_codes \
  --num-emb-list 256,256,256,256 \
  --e-dim 32 \
  --layers 2048,1024,512,256,128,64 \
  --batch-size 1024 \
  --epochs 100
```

验证：本地 `tests/test_skill_gr_pipeline.py` 39 项通过或按缺失依赖跳过；远端含 PyTorch/SciPy 环境 39/39 全通过。下一步实验顺序保持单变量：先完成 legacy SemID + pseudo-query continuation 以测数据效应，再从头训练 D4 learned-only tokenizer/decoder 以测方法效应。

### 0.0.5 SkillRouter core75 候选池补全（2026-07-13）

必须严格区分两个集合：SkillsBench v1.1 当前有 87 个任务，但 SkillRouter 论文从这 87 条中删除 12 条标签只含 `pdf`、`xlsx`、`pptx`、`docx` 等通用文件类型 skill 的 `generic-only` 样本，正式评测集是 **75 条 core queries（24 single + 51 multi）**。项目中的 `skillsbench_all87.rows.jsonl` 是筛选前辅助集合，不是 SkillRouter 论文的 Official test set，主结果禁止用 87 条口径报告。

论文 core75 在原 15,083-skill pool 中只缺以下两个 GT：

```text
gt/mesh-analysis
gt/senior-data-engineer
```

两个 skill document 均来自 `panel_classification_v2/easy_depth3_panel_classified.jsonl`。每个 skill 按冻结预算独立调用一次 GPT-4o，`n=8` 后本地选择 5 条，共新增 10 条 task-spec query；生成提示词和请求不读取 Official query。post-hoc 文本审计只在生成完成后比较对应 core75 query：最大 token-Jaccard 为 `0.2931/0.1525`，均无 exact/containment match，也无共享 5-gram。

这一版本使用 core75 GT 做候选缺口识别，因此属于 **benchmark-aware candidate-corpus completion**，不是 training-only clean 设置。实验必须分别报告：

```text
clean inductive:       15,083 source skills -> 14,549 canonical skills
core75-complete:       15,085 source skills -> 14,551 canonical skills
```

主实验产物：

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/core75_pool_completion_gt2/
  selected_skills.jsonl                         # 2 个新增 skill
  pseudo_task5/selected_task5.jsonl             # 10 条新增 query
  expanded_selected_skills.jsonl                # 15,085 source skills
  decoder_rows.base_plus_task5_gt2.jsonl         # 94,912 rows
  expanded_skill_embeddings.pt                  # [15,085, 1,024]
  candidate_coverage_audit.json
  official_text_similarity_audit.json

outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/
  canonical_skills.jsonl                         # 14,551 canonical skills
  canonical_skill_embeddings.pt                  # [14,551, 1,024]
  rows_000.decoder_rows.base_plus_task5_gt2.jsonl # 94,912 rows
  skill_alias_map.json
  alias_manifest.json
```

数据约束全部通过：15,085/15,085 source skills 各有 5 条 pseudo query；canonical 后每个 skill 至少 5 条训练 query，`<5` 为 0。core75 的 75/75 条 query、186/186 个唯一 canonical GT 均在新候选池中。旧 checkpoint 的 SID 空间没有这两个 skill，仍需基于 core75-complete pool 重新构造 SemID 并训练 decoder。core75 有 9 条样本包含 5--7 个 GT；若评测 structured set generation，冻结的 `max_skills=4` 无法完整生成这 9 条，single-SID ranked retrieval 不受此生成数量上限约束。

误建的 `full87_pool_completion_gt3/` 与 `canonical_alias_full87_learned_d4_inputs/` 保留为辅助审计产物，不用于论文主结果；其额外 `gt/docx` 只对应被 SkillRouter 排除的 `offer-letter-generator` generic-only 样本。

关键 SHA-256：

```text
core75_pool_completion_gt2/pseudo_task5/selected_task5.jsonl 9fb3783697d01969b310a46fe9692c9e1889a5f4a0878210c6606a7a940ae1e8
core75_pool_completion_gt2/decoder_rows.base_plus_task5_gt2.jsonl e5b5749a837124c3be1afc14eb8379dd562eb08690466fed74dc71e7c67fcea3
canonical_alias_core75_learned_d4_inputs/canonical_skills.jsonl 1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3
canonical_alias_core75_learned_d4_inputs/canonical_skill_embeddings.pt bb623a18cc0fc24d20b492db4730c484cfdd93422f49c72fead9b88fb3370f20
canonical_alias_core75_learned_d4_inputs/rows_000.decoder_rows.base_plus_task5_gt2.jsonl ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5
```

### 0.0.6 Full-catalog task5 + D4 learned-only 训练结果（2026-07-13）

core75-complete 主线已经完成从数据预检、D4 RQ-VAE、single-SID decoder 训练到冻结评测的全流程。训练前把 canonical train/dev/private test/Official core75 一次性物化为：

```text
rows_000.decoder_rows.base_plus_task5_gt2.jsonl  94,912 rows
rows_001.decoder_rows.base.jsonl                  1,000 private dev rows
rows_002.decoder_rows.base.jsonl                  1,000 private test rows
rows_003.skillsbench_core75.rows.jsonl               75 Official rows
```

四个 split 的 label 均在 14,551-skill canonical pool 中；train/dev/private test/core75 两两 exact-query overlap 均为 0。T5 token audit 中 train/dev/catalog/core75 最长分别为 747/360/1,718/678，均未超过 2,048。

D4 learned-only RQ-VAE 产物：

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/letter_d4_learned_codes_core75_gt2/
```

共 14,551 个唯一四码 SemID，写盘后 full-tuple collision 为 0；collision repair 修改 76 个 item。各层实际 code usage 为 `[74, 116, 123, 256]`，第一层只使用 74/256 个 code，最大一级 cluster 含 654 个 skill。前三层共有 12,460 个 prefix，最大 prefix 为 7。

single-SID decoder 产物：

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/
  single_sid_d4_task5_core75_gt2_t5base_cat3_querye5_b16/
```

冻结配置为 `t5-base`、catalog 3 epochs、query 5 epochs、batch 16、LR `3e-4`、BF16、`max_source_length=2048`。94,912 source rows 展开为 124,175 个 single-positive query rows，dev 展开为 2,501 rows；catalog/query 分别训练 2,730/38,805 steps。最佳 dev loss 为 `1.736187 @ step 38,800`，最终一步为 `1.737029`；`final` 已回载 dev-best，而不是 last。

冻结 `beam=50`、macro Recall 结果如下。所有模型使用相同的 14,551-skill canonical candidate pool；`SR official` 是 `pipizhao/SkillRouter-Embedding-0.6B`，`SR repro` 是项目现有的 1-epoch/10-negative 复现模型。

| Model | Official subset | R@1 | R@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| D4 single-SID GR | overall 75 | 0.0477 | 0.2707 | 0.3453 | 0.2667 |
| D4 single-SID GR | single 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| D4 single-SID GR | multi 51 | 0.0701 | 0.3980 | 0.5078 | 0.3922 |
| SR official | overall 75 | 0.4202 | 0.7717 | 0.8932 | 0.7467 |
| SR official | single 24 | 0.7917 | 0.9167 | 1.0000 | 1.0000 |
| SR official | multi 51 | 0.2454 | 0.7035 | 0.8430 | 0.6275 |
| SR repro | overall 75 | 0.3321 | 0.7768 | 0.8778 | 0.8133 |
| SR repro | single 24 | 0.5417 | 0.7917 | 0.8750 | 0.8750 |
| SR repro | multi 51 | 0.2335 | 0.7697 | 0.8791 | 0.7843 |

private test 上 D4 GR 的 overall `R@1/10/50=0.0209/0.1141/0.1554`、`FC@50=0.1220`；single 104 条的 `R@50=0.1923`，multi 896 条的 `R@50=0.1511`。因此失败不只发生在 Official query style 上。

必要诊断排除了三个表面原因：

1. **不是候选缺失或 alias 错配。** Official core75 的 200 个 GT mentions 全部可达；24 条 single 的每个 GT 都有 5 条 task-style pseudo query，其中 22 个还有 1 条 base query，新增两个各有 5 条。
2. **不只是 Official query OOD。** 对 23 个唯一 Official-single GT 的 115 条训练 pseudo query 原样回放，`beam=50` 的 `R@1/10/50=0/0/0.0348`，即模型连这些训练样本也几乎无法找回。
3. **不是 beam=50 的主要搜索剪枝。** 同一训练 pseudo 诊断集改为 `beam=256` 后 `R@50=0.0261`，没有改善；Official 主结果仍保持冻结的 beam=50。

teacher-forcing code-rank 进一步定位到粗粒度前两层。这里每一层都给定正确的此前 gold code：

| Rows | L0 Top-1 / mean rank | L1 Top-1 / mean rank | L2 Top-1 / mean rank | L3 Top-1 / mean rank | full SID Top-1 |
|---|---:|---:|---:|---:|---:|
| 115 training pseudo | 0.0261 / 15.01 | 0.0087 / 20.78 | 0.1913 / 3.99 | 0.8435 / 1.16 | 0.0000 |
| 24 Official single | 0.0000 / 14.75 | 0.0000 / 18.58 | 0.2083 / 4.50 | 0.8750 / 1.13 | 0.0000 |
| 104 private single | 0.1250 / 12.90 | 0.1635 / 16.49 | 0.4327 / 2.93 | 0.8846 / 1.21 | 0.0096 |

因此当前可证实的瓶颈是：**D4 learned-only SemID 的 level-0/1 code 严重不均衡且不可预测；后层在正确前缀下可以学习，但前层错误使完整 SID 检索归零。** “每个 skill 至少 5 条 query”只解决 label coverage，不保证参数化 Query2SID 能学会 14.5k-way hierarchy。当前 run 同时改变了 pseudo data 和 SemID schema，不能把回退单独归因于 pseudo query；能确定的是当前 `full-catalog task5 + D4 learned-only` 组合失败。

关键结果产物：

```text
outputs/eval/skill_gr_d4_task5_core75_gt2_20260713/core75_beam50/
outputs/eval/skill_gr_d4_task5_core75_gt2_20260713/private_test_beam50/
outputs/eval/skillrouter_samepool_core75_gt2_20260713/official_model_core75/
outputs/eval/skillrouter_samepool_core75_gt2_20260713/repro_model_core75/
outputs/analysis/skill_gr_d4_task5_core75_gt2_20260713/result_summary.json
outputs/analysis/skill_gr_d4_task5_core75_gt2_20260713/teacher_forced_code_rank.json
```

下一步不要继续增加同类 pseudo queries，也不要调大 Official beam。应先把 **SemID 前两层的 code utilization / cluster balance / Query2SID 可预测性** 作为单一变量解决，再用完全相同的数据重训 decoder；具体 SemID 改法应先做文献与现有 LETTER 实现核对，避免直接堆叠额外 loss。

### 0.0.7 LightGCN-d32 + LETTER 官方 tokenizer 训练（2026-07-13 23:07 CST 启动）

本轮已明确固定为 `LightGCN CF embedding + LETTER tokenizer + query2SID`，不使用内部 query-skill CF fallback。训练前审计结果：

```text
canonical skills: 14,551
skill embedding shape: [14,551, 1,024]
train rows / interactions: 94,912 / 124,175
single / multi rows: 77,455 / 17,457
covered / zero-degree skills: 14,551 / 0
unknown labels: 0
```

LightGCN 已完成：

```text
output: outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/
        lightgcn_cf_d32_l3_e500_seed42_core75_gt2/
config: dim=32, layers=3, epochs=500, batch=4096,
        lr=1e-3, reg=1e-4, seed=42
steps: 15,500
final BPR loss: 0.0002286781
embedding shape: [14,551, 32]
skill order exact match: true
finite / unit-normalized: true / true
co-occurrence cosine mean: 0.867059
random-pair cosine mean: -0.000562
```

LETTER D4 tokenizer 的原始 Python-loop run 已停止但保留：

```text
output: outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/
        letter_d4_lightgcn_d32_official_seed42_core75_gt2/
stopped after: 1,284 / 20,000 epochs
reason: superseded by bitwise-equivalent vectorized diversity loss
record: <output>/stopped_run.json
```

官方 `vq.py` 在 diversity loss 中对 batch 逐项执行 GPU `.item()` / scalar mask assignment，导致大量 CUDA synchronization。项目 adapter 现增加显式模式：

```text
--diversity-loss-impl vectorized_bitwise_v1
```

该实现只将逐项索引合并为一次 CPU list transfer，并将逐行 self-mask 改为一次 `scatter_`；保留完全相同的 Python `random.choice` 调用顺序。验证结果：

```text
single diversity loss: bitwise equal
full forward/backward loss: exact equal
one complete epoch: all losses exact equal
all parameters after one epoch: bitwise equal, max abs diff = 0
remote tests: 40/40 passed
controlled full-epoch benchmark: 4.122 s -> 0.545 s (7.56x)
```

使用完全相同超参的新正式 run 已于 2026-07-14 00:17:26 CST 从头启动：

```text
output: outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/
        letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/
PID at launch: 1167833
log: <output>/train.log
pid file: <output>/train.pid

semantic_id_mode=learned_codes
num_emb_list=[256,256,256,256]
e_dim=32
layers=[2048,1024,512,256,128,64]
alpha=0.01
beta=0.0001
quant_loss_weight=1.0
sk_epsilons=[0,0,0,0.003]
sk_iters=50
kmeans_iters=100
batch_size=1024
epochs=20000
eval_step=2000
optimizer=AdamW
lr=1e-3
weight_decay=1e-4
seed=42
external CF=the LightGCN-d32 artifact above
diversity_loss_impl=vectorized_bitwise_v1
```

这些是 LETTER 发布训练脚本对应的 tokenizer 设置；领域适配是用 LightGCN-d32 替代论文实验中的 SASRec-d32，并保留项目已确定的 D4 learned-only 零碰撞输出。新 run 真实稳定速度为最近 50 epochs 平均 `0.721 s/epoch`，对比旧 run 约 `3.21 s/epoch` 为 4.45x 端到端加速，总时长预计约 4 小时。新旧 run 的前 135 个 epoch 所有日志 loss tuples 逐 epoch 完全一致。训练完成前不得启动 query2SID decoder；完成后先审计四层 code usage、effective codes、最大 cluster、prefix 分布和 collision repair。

### 0.0.8 LightGCN + LETTER tokenizer、decoder 与冻结评测结果（2026-07-14）

上述正式 tokenizer run 已完成 20,000 epochs。新旧实现的完整重叠区间 `epoch 0--1283` 三项 loss 逐 epoch 全部一致，证明 vectorized diversity loss 没有改变已观测训练轨迹。最低 raw full-tuple collision 出现在 `epoch 17,999`：

```text
best collision rate: 0.0769706549
raw collision groups / items: 39 / 80
minimum-cost repair prefixes / items: 39 / 114
changed items: 99
post-resolution full-tuple collisions: 0
unique four-code SemIDs: 14,551 / 14,551
```

最终 SemID 与 canonical skill order 完全一致，四层 code 均在 `[0,255]`。后处理后层级统计如下：

| Level | Used / 256 | Effective codes | Max cluster |
|---|---:|---:|---:|
| L0 | 90 | 49.01 | 819 |
| L1 | 256 | 251.27 | 91 |
| L2 | 256 | 252.83 | 83 |
| L3 | 256 | 241.08 | 263 |

depth-1/2/3/4 unique prefixes 为 `[90, 7,174, 12,434, 14,551]`，最大 prefix group 为 `[819, 21, 10, 1]`。相较旧 D4 的 usage `[74,116,123,256]`，L1--L3 的 marginal utilization 已基本解决；L0 只从 74 提升到 90，仍明显不均衡。

冻结 single-SID decoder 产物：

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/
  single_sid_d4_lightgcn_d32_letterofficial_task5_core75_gt2_t5base_cat3_querye5_b16/
```

配置与旧 decoder 完全相同：`t5-base`、catalog 3 epochs、query 5 epochs、batch 16、LR `3e-4`、BF16、`max_source_length=2048`、seed `20260626`。catalog/query 分别为 2,730/38,805 steps；train/dev/catalog 的最大 source tokens 仍为 `747/360/1,718`，截断数均为 0。最佳 dev loss 为 `0.782136 @ step 38,800`，旧 SemID run 为 `1.736187`。`final` 与 `best` state fingerprint 完全相同，且不同于 `last`，确认最终加载的是 dev-best。

训练 query sanity：23 个唯一 Official-single GT 对应的 115 条 task-style pseudo query 原样回放，`R@1/10/50=0.9130/1.0000/1.0000`，旧模型同一诊断的 `R@50=0.0348`。因此 Query2SID 已从“训练样本也学不会”恢复为可学习任务。

冻结 `beam=50`、`beam_order`、macro Recall 结果：

| Model | Official subset | R@1 | R@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| New LightGCN+LETTER GR | overall 75 | 0.2837 | 0.6900 | 0.7500 | 0.7333 |
| New LightGCN+LETTER GR | single 24 | 0.4167 | 0.4583 | 0.5000 | 0.5000 |
| New LightGCN+LETTER GR | multi 51 | 0.2211 | 0.7990 | 0.8676 | 0.8431 |
| Old D4 GR | overall 75 | 0.0477 | 0.2707 | 0.3453 | 0.2667 |
| Old D4 GR | single 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Old D4 GR | multi 51 | 0.0701 | 0.3980 | 0.5078 | 0.3922 |
| SR official | overall 75 | 0.4202 | 0.7717 | 0.8932 | 0.7467 |
| SR official | single 24 | 0.7917 | 0.9167 | 1.0000 | 1.0000 |
| SR official | multi 51 | 0.2454 | 0.7035 | 0.8430 | 0.6275 |
| SR repro | overall 75 | 0.3321 | 0.7768 | 0.8778 | 0.8133 |
| SR repro | single 24 | 0.5417 | 0.7917 | 0.8750 | 0.8750 |
| SR repro | multi 51 | 0.2335 | 0.7697 | 0.8791 | 0.7843 |

New GR 在 Official multi 上已超过 SR official 的 `R@10/R@50/FC@50`，分别高 `+0.0955/+0.0247/+0.2157`；R@1 仍低 `0.0243`。overall 仍低 SR official `R@1/R@10/R@50=-0.1365/-0.0817/-0.1432`，主要由 single 的 12/24 个 GT 未进入 Top-50 导致。

private test 上 new GR 的 overall `R@1/10/50=0.2168/0.7541/0.8954`、`FC@50=0.8110`；single 104 条的 `R@50=0.8365`，multi 896 条的 `R@50=0.9023`。旧 GR overall `R@50/FC@50=0.1554/0.1220`，因此方法改动在非 Official 数据上同样大幅有效。

teacher-forcing gold-prefix 诊断：

| Rows | L0 Top-1 / mean rank | L1 Top-1 / mean rank | L2 Top-1 / mean rank | L3 Top-1 / mean rank | full SID Top-1 |
|---|---:|---:|---:|---:|---:|
| 115 training pseudo | 0.9304 / 1.12 | 0.9565 / 1.04 | 0.9739 / 1.03 | 1.0000 / 1.00 | 0.8870 |
| 24 Official single | 0.3750 / 11.17 | 0.4167 / 32.25 | 0.6250 / 11.33 | 0.7500 / 5.79 | 0.2500 |

这改变了当前诊断：主要瓶颈已不再是 decoder 无法学习 SemID，而是 **Official single 相对 task-style pseudo query 的跨层 query-side 泛化**；L0 不均衡仍可能放大该问题，但不能再用“所有层都塌缩”解释结果。若保持当前 multi 不变，要在 overall 上超过 SR official，single 约需达到 `R@1/R@10/R@50=0.84/0.71/0.95`，当前为 `0.42/0.46/0.50`。

关键产物：

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/single_sid_d4_lightgcn_d32_letterofficial_task5_core75_gt2_t5base_cat3_querye5_b16/
outputs/eval/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714/train_pseudo_single_beam50/
outputs/eval/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714/private_test_beam50/
outputs/eval/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714/core75_beam50/
outputs/analysis/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714/result_summary.json
outputs/analysis/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714/teacher_forced_code_rank.json
```

### 0.0.9 Zero-multi skills 三模式 query augmentation（2026-07-15）

为验证 Official single 的主要问题是否来自训练 query 模式覆盖不足，完成了一次 training-only 的定向数据实验。目标 skill 只根据 canonical 训练行统计选择：若一个 skill 从未出现在任何 multi-skill row 中，即 `multi_row_frequency == 0`，则纳入生成。选择过程不读取 Official rows、query 或 GT；但底层 14,551-skill pool 仍是 0.0.5 所述的 core75-complete benchmark-aware candidate pool，因此“clean”只指本次 query 选择与生成过程。

选择结果与生成预算：

```text
canonical skills:                    14,551
zero-multi-cooccurrence skills:       1,722
modes per skill:                          3
generated / selected rows:            5,166 / 5,166
API failures:                             0
GPT-4o total tokens:              5,635,164
```

三个固定 mode 是 `outcome_first`、`workflow_embedded`、`constraint_failure`。每个 `(skill, mode)` 独立调用一次，`n=1`、temperature `0.8`、top-p `0.95`、并发 `8`；不做质量重试。三条缺少英文 task heading 的输出通过确定性 layout normalization 修复，没有再次调用 API。最终 query 的 `You need to:` 和 numbered-requirement 覆盖率均为 100%；平均字符数 `937.88`，同 skill 三条 query 的 pairwise token-Jaccard 均值 `0.2901`，per-skill 最大 Jaccard 均值 `0.3286`，source truncation rate `0.00155`。全量验证同时检查 single label、skill-mode 覆盖、provenance、skill-name/id leak、index、query_text 和 base exact-query duplicate。

数据产物：

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/isolated_mode3_gpt4o_20260714/
  isolated_mode_generations.jsonl
  selected_mode3.jsonl
  manifest.json
  augment_manifest.json
  decoder_rows.base_plus_task5_gt2.plus_isolated_mode3.jsonl
```

增强前后 source rows 为 `94,912 -> 100,078`；single-positive 展开后为 `124,175 -> 129,341`。冻结同一个 LightGCN-d32 + LETTER D4 SemID，decoder 仍从 `t5-base` 重新初始化，保持 catalog 3 epochs、query 5 epochs、train batch 16、LR `3e-4`、BF16、seed `20260626` 和 `max_source_length=2048`。只把 dev eval batch 从 16 增到 128、eval interval 从 200 增到 1,000；这不改变训练样本、训练 batch、loss 或 epoch。catalog/query steps 为 `2,730/40,420`，train/dev/catalog 最大 token 为 `747/360/1,718`，截断数均为 0。catalog loss 与 baseline 对应 step 完全一致；最终 best dev loss 为 `0.794048 @ step 40,420`。

训练 wall clock 为 `2026-07-14 22:38:16` 到 `2026-07-15 00:51:13`，共 `7,977s = 132m57s`。baseline 是约 121 分钟。虽然单次 dev eval 的实测从 batch 16 的 `5.25s` 降到 batch 128 的 `3.91s`，且 eval/checkpoint 次数减少，但新增 4.2% query steps 后端到端并没有加速。结论是当前主要耗时在 T5 forward/backward，eval 不是主瓶颈；不要把这组 eval 参数宣传为训练加速结果。若要进一步提速，length-sorted batching、增大 train batch 或 fused/compiled 路径都会改变 batch composition 或数值轨迹，必须作为单独工程实验验证，不能静默用于主结果。

冻结 `beam=50`、`beam_order`、macro Recall 结果：

| Model | Official subset | R@1 | R@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| mode3 augmented GR | overall 75 | 0.2932 | 0.6904 | 0.7856 | 0.7467 |
| mode3 augmented GR | single 24 | 0.5417 | 0.5833 | 0.6250 | 0.6250 |
| mode3 augmented GR | multi 51 | 0.1763 | 0.7408 | 0.8611 | 0.8039 |
| task5 baseline GR | overall 75 | 0.2837 | 0.6900 | 0.7500 | 0.7333 |
| task5 baseline GR | single 24 | 0.4167 | 0.4583 | 0.5000 | 0.5000 |
| task5 baseline GR | multi 51 | 0.2211 | 0.7990 | 0.8676 | 0.8431 |
| SR official | overall 75 | 0.4202 | 0.7717 | 0.8932 | 0.7467 |
| SR repro | overall 75 | 0.3321 | 0.7768 | 0.8778 | 0.8133 |

相对 task5 baseline，Official overall 的 `R@1/R@10/R@50` 分别变化 `+0.0096/+0.0004/+0.0356`，single 三档均为 `+0.1250`，但 multi 为 `-0.0448/-0.0582/-0.0065`。single Top-50 有 4 条新召回、1 条旧召回丢失，净增 3/24。overall Top-50 的 total gold hits 仍是 168，micro Recall 仍为 0.84：macro R@50 的提升来自把 3 个 hit 从 multi rows 转移到 single rows，而不是总 gold hits 增加。multi query 的 Top-50 中 zero-multi target predictions 从平均 `2.78` 增到 `4.08`，同时 multi gold hits 从 156 降到 153，说明 targeted augmentation 强化了 isolated skills，但产生了候选挤占或共享参数干扰。该因果解释置信度为 medium：只有一个 seed，Official single 也只有 24 条。

private test 结果基本中性：overall `R@1/10/50` 从 `0.2168/0.7541/0.8954` 变为 `0.2233/0.7584/0.8909`；single 为 `0.3365/0.7308/0.8077`，相对 baseline `+0.0769/+0.0288/-0.0288`；multi 为 `0.2102/0.7616/0.9006`，与 baseline 基本持平。这进一步说明三模式数据更稳定地改善高位排序，而不是全面改善 Top-50 coverage。

结论：更广的任务模式覆盖是有效方向，但仅增加 isolated-skill pseudo queries 不能让 GR 超过 SkillsRouter，也不能视为无代价的数据改进。下一步不应继续叠加同类 pseudo query；应冻结这版数据与 SemID，优先处理“增强 single 时不干扰 multi”的排序目标或训练机制，再看是否同时提升 Official `R@1/R@10`。完整结果：

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/
  single_sid_d4_lightgcn_d32_letterofficial_task5_isolatedmode3_core75_gt2_t5base_cat3_querye5_b16/
outputs/eval/skill_gr_d4_lightgcn_letterofficial_task5_isolatedmode3_core75_gt2_20260714/
outputs/analysis/skill_gr_d4_lightgcn_letterofficial_task5_isolatedmode3_core75_gt2_20260715/result_summary.json
```

关键 SHA-256：

```text
selected_mode3.jsonl                                      87886fa794c68145bc6c3e725cc015ff2be9fc2e2c280fa632baf69c4a301237
decoder_rows.base_plus_task5_gt2.plus_isolated_mode3.jsonl 55b6539c3379523f78c2326580fa690500f9b213d1d620b250ba333b65bcba08
result_summary.json                                       152e69548e443f007149747a90fad71ad126a60d36c9786ab260fae10b7d515d
```

### 0.1 当前研究目标

项目现在已经从“只构造 SkillRouter-style query 数据”推进到完整实验主线：

```text
SkillSearch Data Synthesis
-> 21k synthetic skill-routing dataset
-> SkillGR / LETTER-style generative retrieval
-> SkillsRouter-style dense retrieval reproduction
-> private test vs official SkillsBench test 对比诊断
-> full-catalog task-style pseudo queries complete
-> LightGCN-d32 + LETTER D4 tokenizer 已完成
-> 相同数据的 single-SID decoder 与冻结评测已完成
-> zero-multi skills 三模式 augmentation 与冻结评测已完成
-> 下一步：保留 single 增益，同时消除 multi 干扰并改善 Top-10 排序
```

当前核心问题已经不是 query 数量、候选覆盖或 decoder 对训练 SID 的基本可学习性。三模式实验已经验证更广 query coverage 能把 Official single `R@50` 从 0.50 提到 0.625，但也把 multi `R@1/R@10` 从 `0.221/0.799` 降到 `0.176/0.741`。已确认的主要短板是：

1. `Official single query-side OOD` 仍未解决：mode3 后仍有 9/24 个 single GT 未进入 Top-50，且 `R@10=0.583` 远低于 SR repro 的 0.792。
2. `single/multi interference`：增强 isolated skills 后，multi Top-50 中此类预测平均增加 1.29 个，multi gold hits 净减 3。
3. `高位排序不足`：overall `R@10` 几乎没有变化，说明只增加正例 query 不能提供足够的相对排序信号。
4. `L0 仍不均衡`：L1--L3 已接近满利用，但 L0 只有 90/256 effective 49.01，仍可能放大 coarse routing 的错误。

因此后续重点应是：

```text
冻结 mode3 数据、候选池、SemID 与 beam
+ 不再增加同类 pseudo-query 数量
+ 下一次只改一个能表达相对排序或抑制 isolated-skill 干扰的训练机制
+ 同时检查 Official single 提升和 multi 不退化
```

不要调大 Official beam，也不要同时改 tokenizer、SemID 和 loss。下一轮仍应避免直接针对剩余 9 条 Official miss 做转述或生成；应使用训练/开发来源构造方法，并保持一个完整方法改动对应一组主实验。

### 0.2 工作区与远端

本地仓库：

```text
/Users/chenyufei/Documents/GR&Agentic/SkillSearch-data-synthesis
```

远端仓库：

```text
/root/workplace/SkillSearch-data-synthesis
```

SSH：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp
```

远端注意事项：

- 远端没有 `rg`，用 `find` / `grep` / Python 脚本代替。
- `outputs/` 和 `profiles/` 仍然通过 symlink 复用旧 workspace：

```text
/root/workplace/SkillSearch-data-synthesis/outputs  -> /root/workplace/SkillSearch/outputs
/root/workplace/SkillSearch-data-synthesis/profiles -> /root/workplace/SkillSearch/profiles
```

官方 SkillRouter repo：

```text
/root/workplace/SkillSearch/external/SkillRouter
```

官方 repo 当前公开内容主要是 eval / export 代码和 eval data，不包含完整训练脚本。训练细节需要结合论文 Appendix E/F 与我们自己的 `scripts/skillrouter_embedding_train.py` 复现实现判断。

### 0.3 当前代码状态

当前本地 git worktree 是 dirty，接手时不要随手回滚。最近有大量新增脚本还未提交：

```text
M  HANDOFF.md
M  README.md
M  scripts/replay_pilot_400_skillrouter_prompt.py
?? PLAN.md
?? docs/
?? scripts/build_balanced_topup_inputs.py
?? scripts/build_gt_multiskill_inputs.py
?? scripts/build_normal_skillrouter_inputs.py
?? scripts/skill_gr_pipeline.py
?? scripts/skillrouter_embedding_train.py
?? scripts/train_letter_rqvae_official.py
?? scripts/train_lightgcn_cf_embeddings.py
?? tests/test_normal_skillrouter_inputs.py
?? tests/test_skill_gr_pipeline.py
```

不要执行：

```bash
git reset --hard
git checkout -- .
```

除非用户明确要求。当前 handoff 只负责记录状态，不清理工作区。

### 0.4 最重要的行为约束

之前发生过一次严重问题：为了加速，我曾启动 `negatives_per_pair=4` 的 SkillsRouter 训练诊断，用户明确不接受这种未确认的实验语义变更。后续必须遵守：

- 不要静默改变 negatives 数量。
- 不要静默改变训练 epoch / max steps / effective batch size。
- 不要静默改变候选池、数据 split、eval set、max_skills、beam size、query truncation。
- 任何会改变实验语义的参数改动，必须先说明并得到用户认可。
- 非语义优化，例如 length sorting、日志格式、checkpoint 命名，可以做，但也要透明记录。

已经废弃且不要引用的 run：

```text
neg4 fast diagnostic
W&B run: 0inkgx5q
PID 曾为 979254，已停止
不要把它作为结果使用
```

当前远端没有必须继续跟踪的长跑训练进程。最近检查 `ps` 未发现 `skill_gr_pipeline` / `skillrouter_embedding_train` / `train_lightgcn` / `train_letter` 相关进程。

### 0.5 W&B

W&B project：

```text
https://wandb.ai/cyf04/SKillGR
```

不要把用户的 W&B API key 写进 repo 或 handoff。需要重新登录时让用户设置环境变量，或在 shell 里临时 export。

重要 run：

```text
SkillsRouter repro strict 10-neg:
https://wandb.ai/cyf04/SKillGR/runs/v9vvpy9y
```

这个 run 是有效结果：

- `negatives_per_pair=10`
- `batch_size=8`
- `grad_accum=4`
- `epochs=1`
- `lr=2e-5`
- `temperature=0.02`
- `max_length=2048`
- `bf16=true`
- gradient checkpointing true
- 1 epoch = 48,760 pairs
- 6,095 loader steps
- 1,524 optimizer updates
- elapsed 20,632s，约 5h44m
- final train loss 0.41247

输出目录：

```text
outputs/train/skillrouter_repro_20260707/sr_emb_qwen3_06b_1epoch_10neg
```

### 0.6 当前 21k 数据集

主数据根目录：

```text
outputs/train/skill_gr_21k_single_docaug_20260703
```

关键文件：

```text
prepare_train/decoder_rows.base.jsonl    19,487 rows
prepare_dev/decoder_rows.base.jsonl       1,000 rows
prepare_test/decoder_rows.base.jsonl      1,000 rows
prepare_train/selected_skills.jsonl      15,083 skills
```

当前 candidate pool：

```text
15,083 selected skills
```

训练集目标 skill mentions：

```text
train rows: 19,487
train total gold mentions: 48,760
k distribution:
  k=1: 2,030
  k=2: 7,761
  k=3: 7,576
  k=4: 2,120
```

dev/test 分布：

```text
dev rows: 1,000, total gold: 2,503
test rows: 1,000, total gold: 2,503
k distribution:
  k=1: 104
  k=2: 398
  k=3: 389
  k=4: 109
```

数据来源：

- 原始 2k 高质量数据保留。
- 18k 第一阶段 scaling 数据加入。
- 第二阶段 GT-skill multi 数据中 clean 的部分加入，weak 按人工审计结论选择性加入。
- 最后抽样 1,000 dev / 1,000 test，保证 test skills 不是 unseen。注意：official SkillsBench 中仍有个别 GT 不在当前 15k pool。

重要质量结论：

- 当前 private train/dev/test 是同一 synthetic pipeline 分布。
- private query 平均更短、更抽象，更像用户需求描述。
- official SkillsBench query 更像 benchmark task spec，包含文件路径、JSON schema、输出文件、数值约束、多步骤命令。

### 0.7 Official SkillsBench eval set

当前使用的 official subset：

```text
outputs/eval/skill_gr_baseline_eval_20260703/skillsbench_core75.rows.jsonl
```

统计：

```text
rows: 75
total gold skills: 200
unique gold skills: 186
k distribution:
  k=1: 24
  k=2: 13
  k=3: 17
  k=4: 12
  k=5: 4
  k=6: 4
  k=7: 1
```

注意：

- official test 有很多 5/6/7-skill query。
- 当前部分 GR eval 仍使用 `max_skills=4`，这对 official multi 不完全公平。
- BM25 / Qwen3 / SkillsRouter 是 ranked list，不需要显式决定生成几个 skill。
- GR multi-set decoding 需要同时解决生成 skill 数量、SemID、顺序、终止，因此 eval 复杂度更高。

### 0.8 Private vs Official query-side OOD 诊断

private test 文件：

```text
outputs/train/skill_gr_21k_single_docaug_20260703/prepare_test/decoder_rows.base.jsonl
```

official test 文件：

```text
outputs/eval/skill_gr_baseline_eval_20260703/skillsbench_core75.rows.jsonl
```

query 侧统计：

| 统计 | private test | official core75 |
|---|---:|---:|
| 平均词数 | 91.2 | 207.4 |
| 平均字符数 | 618.9 | 1391.3 |
| 平均行数 | 1.43 | 27.52 |
| code fence 比例 | 0.0% | 38.7% |
| JSON-like 比例 | 2.5% | 45.3% |
| 平均路径/文件名数 | 0.109 | 5.147 |
| 平均 numbered steps | 0.13 | 2.48 |
| 平均 constraint 词数 | 1.07 | 4.05 |
| query >1500 chars | 0.4% | 41.3% |

标签侧统计：

| 统计 | private test | official core75 |
|---|---:|---:|
| rows | 1000 | 75 |
| total gold skills | 2503 | 200 |
| unique gold skills | 1980 | 186 |
| GT 在训练集出现次数 median | 6 | 12 |
| GT 在训练集出现次数 mean | 11.4 | 24.9 |
| unique unseen vs private train | 0 | 2 |

结论：

- official 对 BM25 / Qwen3 / official SkillsRouter 反而更容易，因为 query 信息密度高，词面和语义锚点更多。
- 我们训练后的 SkillsRouter repro 在 private 上更高，说明它吃到了 synthetic private distribution。
- 这不是狭义 memorization overfitting，更准确是 `synthetic distribution / query-style overfitting`。
- official 对当前模型是 `query-side OOD`，不是 label-side OOD。official labels 反而更集中、更高频。

典型完整 query 对比已经在对话中输出过，源文件可用下面命令重新取：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp python3 <<'PY'
import json
from pathlib import Path
root=Path('/root/workplace/SkillSearch-data-synthesis')
priv=root/'outputs/train/skill_gr_21k_single_docaug_20260703/prepare_test/decoder_rows.base.jsonl'
off=root/'outputs/eval/skill_gr_baseline_eval_20260703/skillsbench_core75.rows.jsonl'
for p in [priv, off]:
    print(p)
    with open(p) as f:
        for i,line in zip(range(3), f):
            r=json.loads(line)
            print(r.get('index'), r.get('selected_skill_ids'), (r.get('query') or '')[:1000])
PY
```

### 0.9 当前核心实验结果

所有下列结果均为同一 15,083 candidate pool 口径，K values 为 `1,10,50`，metric 字段：

```text
H@K = hit，至少命中一个 gold
R@K = recall，命中 gold 数 / gold 数
FC@K = full coverage，所有 gold 都被 top-K 覆盖
```

#### 0.9.1 Baselines

BM25：

| Set | H@1 | R@10 | FC@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| private test | 0.360 | 0.409 | 0.190 | 0.604 | 0.369 |
| official core75 | 0.640 | 0.655 | 0.560 | 0.785 | 0.693 |

Qwen3-Embedding-0.6B base：

| Set | H@1 | R@10 | FC@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| private test | 0.386 | 0.449 | 0.231 | 0.665 | 0.444 |
| official core75 | 0.627 | 0.635 | 0.467 | 0.770 | 0.613 |

Official SkillRouter embedding-only ckpt，no reranker，controlled 15k pool：

| Set | H@1 | R@10 | FC@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| private test | 0.456 | 0.527 | 0.312 | 0.740 | 0.536 |
| official core75 | 0.747 | 0.710 | 0.560 | 0.850 | 0.720 |

注意：official SkillRouter paper headline 的 `74.0%` 是 `SR-Emb + SR-Rank` pipeline，不应直接和 embedding-only 或 GR decoder 混比。我们当前 controlled comparison 主要使用 embedding-only no-reranker 口径。

#### 0.9.2 我们复现的 SkillsRouter-style SR-Emb

输出：

```text
outputs/train/skillrouter_repro_20260707/sr_emb_qwen3_06b_1epoch_10neg
outputs/eval/skillrouter_repro_qwen3_06b_10neg_1epoch/own_test_final
outputs/eval/skillrouter_repro_qwen3_06b_10neg_1epoch/skillsbench_core75_final
```

结果：

| Set | H@1 | R@10 | FC@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| private test | 0.750 | 0.845 | 0.687 | 0.957 | 0.900 |
| official core75 | 0.653 | 0.780 | 0.613 | 0.880 | 0.773 |

single subset：

| Set | rows | H@1 | R@10 | FC@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|---:|
| private single | 104 | 0.769 | 0.933 | 0.933 | 0.981 | 0.981 |
| official single | 24 | 0.458 | 0.708 | 0.708 | 0.792 | 0.792 |

解释：

- 我们复现的 SR-Emb 对 private synthetic distribution 极强。
- 在 official 上 top-1 弱于 official ckpt，但 R@10/FC@10 不差。
- 差距主要是 query-style alignment 和 top-rank calibration，不是完全召回失败。

#### 0.9.3 GR single+doc-aug

输出：

```text
outputs/eval/skill_gr_baseline_eval_20260703/gr_single_docaug_e16_own_test_beam50
outputs/eval/skill_gr_baseline_eval_20260703/gr_single_docaug_e16_skillsbench_core75_beam50
```

结果：

| Set | H@1 | R@10 | FC@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| private test | 0.515 | 0.728 | 0.510 | 0.892 | 0.777 |
| official core75 | 0.347 | 0.600 | 0.347 | 0.730 | 0.520 |

single subset：

| Set | rows | H@1 | R@10 | FC@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|---:|
| private single | 104 | 0.202 | 0.519 | 0.519 | 0.721 | 0.721 |
| official single | 24 | 0.000 | 0.083 | 0.083 | 0.208 | 0.208 |

#### 0.9.4 LETTER / LightGCN / CF SemID 版本

相关脚本：

```text
scripts/train_lightgcn_cf_embeddings.py
scripts/train_letter_rqvae_official.py
```

LightGCN 用 skill co-occurrence 学 CF embeddings；`train_letter_rqvae_official.py` 复用 LETTER-style RQ-VAE 设计，把 semantic embedding 与 CF embedding 结合生成 SemID。

当前最相关结果：

LETTER single+doc-aug：

| Set | H@1 | R@10 | FC@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| private test | 0.546 | 0.780 | 0.588 | 0.912 | 0.819 |
| official core75 | 0.467 | 0.710 | 0.467 | 0.785 | 0.587 |

LETTER multi+doc-aug top-k marginal m4：

| Set | H@1 | R@10 | FC@10 | R@50 | FC@50 |
|---|---:|---:|---:|---:|---:|
| private test | 0.558 | 0.768 | 0.564 | 0.846 | 0.708 |
| official core75 | 0.507 | 0.670 | 0.413 | 0.760 | 0.533 |

single subset 仍然非常弱：

```text
LETTER single official single:
  H@1 = 0.000
  R@10 = 0.125
  R@50 = 0.250

LETTER multi official single:
  H@1 = 0.0417
  R@10 = 0.0833
  R@50 = 0.1667
```

结论：LETTER / LightGCN 当前没有解决 official single long-tail problem。它对整体指标有改善，但 CF SemID 还没有稳定提取出足够有效的协同结构，也没有解决冷启动 / 低频 skill 的 Query2SID 监督不足。

### 0.10 SkillsRouter 复现实现细节

脚本：

```text
scripts/skillrouter_embedding_train.py
```

三个子命令：

```text
mine
train
eval
```

核心输入：

```text
rows:         outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/decoder_rows.base.jsonl
skills_jsonl: outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/selected_skills.jsonl
```

训练 pair 输出：

```text
outputs/train/skillrouter_repro_20260707/train_pairs_hn.jsonl
outputs/train/skillrouter_repro_20260707/train_pairs_hn.manifest.json
```

manifest：

```json
{
  "pairs": 48760,
  "skills": 15083,
  "mine_top_k": 50,
  "negatives_per_pair": 10,
  "source_counts": {
    "semantic": 195040,
    "lexical": 146280,
    "taxonomy": 97445,
    "random": 48760,
    "fallback": 75
  },
  "filter_counts": {
    "same_name": 5103,
    "body_overlap": 2443,
    "embedding_similarity": 2194
  },
  "thresholds": {
    "embedding_similarity": 0.92,
    "trigram_jaccard": 0.6
  }
}
```

实现逻辑：

- encoder：`Qwen/Qwen3-Embedding-0.6B`
- pooling：last token pool
- normalize embeddings
- query format：

```text
Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task
Query:{query}
```

- skill text：

```text
{name} | {description[:300]} | {body[:2500]}
```

- query truncation：`query_max_chars=1500`
- max token length：`2048`
- hard negatives：
  - 4 semantic negatives
  - 3 BM25 lexical negatives
  - 2 taxonomy/category negatives
  - 1 random negative
- false negative filter：
  - same skill name
  - body char-trigram Jaccard > 0.6
  - Qwen embedding cosine > 0.92
- loss：in-batch InfoNCE with explicit negatives
- temperature：0.02

训练命令模板：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
python3 scripts/skillrouter_embedding_train.py mine \
  --rows outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/decoder_rows.base.jsonl \
  --skills-jsonl outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/selected_skills.jsonl \
  --output outputs/train/skillrouter_repro_20260707/train_pairs_hn.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B \
  --bf16

python3 scripts/skillrouter_embedding_train.py train \
  --pairs outputs/train/skillrouter_repro_20260707/train_pairs_hn.jsonl \
  --output-dir outputs/train/skillrouter_repro_20260707/sr_emb_qwen3_06b_1epoch_10neg \
  --model Qwen/Qwen3-Embedding-0.6B \
  --epochs 1 \
  --batch-size 8 \
  --grad-accum 4 \
  --negatives-per-pair 10 \
  --max-length 2048 \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --temperature 0.02 \
  --bf16 \
  --wandb-project SKillGR \
  --wandb-run-name sr_emb_qwen3_06b_1epoch_10neg
'
```

eval 命令模板：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
python3 scripts/skillrouter_embedding_train.py eval \
  --rows outputs/train/skill_gr_21k_single_docaug_20260703/prepare_test/decoder_rows.base.jsonl \
  --skills-jsonl outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/selected_skills.jsonl \
  --model outputs/train/skillrouter_repro_20260707/sr_emb_qwen3_06b_1epoch_10neg \
  --output-dir outputs/eval/skillrouter_repro_qwen3_06b_10neg_1epoch/own_test_final \
  --split-name own_test \
  --bf16

python3 scripts/skillrouter_embedding_train.py eval \
  --rows outputs/eval/skill_gr_baseline_eval_20260703/skillsbench_core75.rows.jsonl \
  --skills-jsonl outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/selected_skills.jsonl \
  --model outputs/train/skillrouter_repro_20260707/sr_emb_qwen3_06b_1epoch_10neg \
  --output-dir outputs/eval/skillrouter_repro_qwen3_06b_10neg_1epoch/skillsbench_core75_final \
  --split-name skillsbench_core75 \
  --bf16
'
```

### 0.11 GR pipeline 实现细节

主脚本：

```text
scripts/skill_gr_pipeline.py
```

主要子命令：

```text
prepare
encode-skills
train-rqvae
build-decoder-data
split-decoder-data
build-decoding-trie
train-decoder
train-structured-decoder
train-single-sid-decoder
eval-structured-decoder
eval-single-sid-decoder
eval-bm25
eval-t5-dense
eval-decoder
```

关键参数：

- skill encoder 默认：`Qwen/Qwen3-Embedding-0.6B`
- RQ-VAE 默认：
  - codebook size 256
  - depth 3
  - hidden dim 512
  - latent dim 128
  - epochs 100
- decoder 默认：`t5-base`
- structured decoder：
  - `catalog_epochs` 支持 doc-aug / catalog pretraining
  - `epochs` 支持 query training
  - `init_model_dir` / `step_offset` 支持续训
  - `load_best_model_at_end` 默认 true
- eval 默认 beam：
  - `num_beams=50`
  - `max_skills=4` for structured/multi
  - `max_skills=1` for single-sid

重要实现细节：

- `decoder_input` 不存在时 fallback 到 `query_text` / `query`，这是之前 query stage 崩溃后修的 schema 兼容。
- 续训 checkpoint 保存形式后来做过调整，目的是避免每次续训开新 W&B 曲线和覆盖 best/last。
- train loss 曲线曾经显示 single 震荡较大；dev loss 一直下降，但 dev split 和 test split 都是 synthetic uniform sampling，不能过度代表真实指标。
- 当前评测更依赖最终 ranked metrics，不再以 dev loss 判断方法有效性。

### 0.12 LETTER / LightGCN / CF SemID 实现细节

LightGCN wrapper：

```text
scripts/train_lightgcn_cf_embeddings.py
```

主要参数：

```text
--lightgcn-code-dir
--embeddings
--rows
--output-dir
--dim 64
--layers 3
--epochs 300
--batch-size 4096
--lr 1e-3
--reg 1e-4
```

作用：

- 从 train rows 的 skill co-occurrence 构造图。
- 用 LightGCN 学 skill CF embedding。
- 目标是复用推荐系统中 collaborative signal 的建模方式，而不是手写一个未经验证的 CF loss。

LETTER RQ-VAE wrapper：

```text
scripts/train_letter_rqvae_official.py
```

主要参数：

```text
--letter-rqvae-dir
--embeddings
--rows
--output-dir
--num-emb-list 256,256,256
--e-dim 128
--layers 512
--alpha 0.1
--beta 0.1
--quant-loss-weight 1.0
--cf-embeddings <external LightGCN embeddings>
```

当前判断：

- 官方 LETTER 的设计可复用，但最大瓶颈是 CF signal 是否能被有效提取。
- 目前结果没有证明 LETTER SemID 已经稳定捕捉协同信息。
- 用 LETTER 后 multi 整体指标尚可，但 single long-tail 仍然崩。

### 0.13 当前重要诊断结论

#### 0.13.1 为什么复现 SkillsRouter 比 GR 好

在当前口径下，复现的 SkillsRouter 比 GR 好是合理的，不是 bug：

1. SkillsRouter 的训练目标直接对齐 ranked retrieval：

```text
query embedding close to positive skill doc
far from hard negatives
```

2. 推理时 SkillsRouter 直接读取完整 skill 文档：

```text
skill name | desc | body
```

3. GR 把 skill 压缩成 SemID，推理时不再 query-doc matching。若 SemID 没保留足够语义和区分性，就会丢失信息。

4. SkillsRouter 有 explicit hard negatives；当前 GR teacher-forcing Query2SID 没有等价 contrastive signal。

5. multi-set GR decoding 需要同时解决生成几个 skill、生成哪个 SemID、顺序和终止；ranked retrieval 只需要排序。

6. 当前 GR 的核心优势，即“利用协同信号生成有效 multi set”，还没有真正兑现。

#### 0.13.2 为什么 GR official single subset 特别差

official single subset 上：

```text
GR single+doc-aug:
  H@1 = 0.000
  R@10 = 0.083
  R@50 = 0.208

LETTER single+doc-aug:
  H@1 = 0.000
  R@10 = 0.125
  R@50 = 0.250
```

核心原因不是 single 任务本身难，而是 official single 的 GT skill 在当前训练集中极端长尾：

```text
official single rows: 24
所有 GT skill 训练频次都 <= 1
其中 2 个训练频次为 0
median = 1
mean = 0.92
```

例子：

```text
0 gt/mesh-analysis
0 gt/senior-data-engineer
1 gt/academic-pdf-redaction
1 gt/azure-bgp
1 gt/citation-management
1 gt/d3-visualization
1 gt/dialogue_graph
1 gt/geospatial-analysis
1 gt/gh-cli
1 gt/jax-skills
1 gt/qutip
1 gt/pddl-skills
```

BM25 / Qwen / SkillsRouter 对这些 skill 仍能工作，因为推理时直接匹配文档；GR 是 parametric index，需要从训练中学到 `query -> SemID`，0/1 次监督基本不够。

结论：

```text
当前 SkillGR 对 frequent / seen / multi-composition skill 有一定能力；
但对 single long-tail skill 非常弱。
```

#### 0.13.3 GR 领域中的 pseudo query 与当前问题

用户明确指出的是 GR 领域里的 pseudo query，不是 dense retrieval 泛化概念。当前问题确实对应 GR 中常见的 pseudo query / synthetic query 需求：

```text
document/skill 没有足够 query supervision
-> 生成 pseudo queries
-> 训练 Query2ID / Query2SemID
-> 覆盖 long-tail documents / skills
```

相关文献口径：

- NCI / DSI 系列使用 query generation 改善 `query -> docid` 训练覆盖。
- DSI-QG 用 generated queries 缓解 indexing-retrieval mismatch。
- MINDER 把 pseudo queries 作为 synthetic identifiers / multi-view identifiers。

映射到当前项目：

```text
long-tail document in GR
≈ long-tail skill in SkillGR
≈ 训练中没有足够 query -> skill/SemID supervision
```

因此下一步最合理的是 GR-style pseudo query generation，而不是普通 paraphrase augmentation。

### 0.14 下一步建议路线

优先级最高：

```text
构造 task-style pseudo queries 覆盖 long-tail skills
```

具体要求：

1. 对所有低频 skill，至少对训练频次 `<=1/<=3` 的 skill 生成多条 pseudo queries。
2. 输入给 LLM 时不要只给 abstract metadata，要给真实 skill body 中的：
   - usage
   - examples
   - file names
   - command/API snippets
   - input/output artifacts
   - validation criteria
3. 生成 query 风格必须接近 official task spec：
   - concrete input files
   - output files
   - JSON/YAML/CSV/schema
   - numeric constraints
   - multi-step requirements
   - explicit evaluation criteria
4. 不要直接复制 official test query，也不要使用 official query wording。
5. 对 single-skill pseudo query 和 multi-skill collaborative query 分开建模：
   - single pseudo queries 解决 long-tail Query2SID coverage。
   - multi pseudo queries 解决协同信号和 set generation。

一个合理的下一轮实验设计：

```text
E1: Long-tail single pseudo query augmentation
    train only/mostly single Query2SID
    evaluate official single + private single

E2: Add task-style multi pseudo queries
    train multi decoder
    evaluate official multi + private multi

E3: Hybrid / rerank
    use GR generated candidates + query-doc dense rerank
```

不建议马上做大规模复杂消融。先做一个能完整验证核心想法的版本：

```text
skill body -> task-style pseudo queries -> Query2SID -> official single 是否恢复
```

如果 official single 从 `R@10=0.08/0.12` 明显恢复，说明 pseudo query 路线成立。

### 0.15 数据生成 prompt 的当前问题

虽然我们复用了 SkillRouter 论文中的核心 prompt shell：

```text
experienced user of AI assistants
concrete scenario with specific inputs/outputs
do not mention skill name
```

但当前生成分布仍然偏离 official，原因是 prompt input content 不同：

当前 single/multi 生成时 `Skill body` 实际是 metadata-as-body-preview：

```text
Operation: ...
Task objects: ...
Inputs: ...
Outputs: ...
```

而不是完整 skill body 中的真实 usage/examples/API/commands。LLM 因此更容易生成：

```text
I need help analyzing ...
provide a detailed report ...
validate the outputs ...
```

而不是 official 里的：

```text
/home/github/build/failed/<repo>/<id>/patch_{i}.diff
simulation_results.csv
vehicle_params.yaml
diarization.rttm
compression_report.json
```

所以后续不要只说“我们已经复用了 SkillRouter prompt”。准确说法是：

```text
我们复用了 prompt shell；
但没有复现官方 query generation 的输入条件和 task-style artifact density。
```

### 0.16 重要文件索引

数据：

```text
outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/decoder_rows.base.jsonl
outputs/train/skill_gr_21k_single_docaug_20260703/prepare_dev/decoder_rows.base.jsonl
outputs/train/skill_gr_21k_single_docaug_20260703/prepare_test/decoder_rows.base.jsonl
outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/selected_skills.jsonl
outputs/eval/skill_gr_baseline_eval_20260703/skillsbench_core75.rows.jsonl
```

GR / LETTER eval：

```text
outputs/eval/skill_gr_baseline_eval_20260703/gr_single_docaug_e16_own_test_beam50
outputs/eval/skill_gr_baseline_eval_20260703/gr_single_docaug_e16_skillsbench_core75_beam50
outputs/eval/letter_lightgcn_single_docaug_d3_a003_b001
outputs/eval/letter_lightgcn_multi_docaug_d3_a003_b001
outputs/eval/letter_lightgcn_multi_docaug_d3_a003_b001_topk
```

Baselines：

```text
outputs/eval/skill_gr_baseline_eval_20260703/bm25_own_test_train_vocab
outputs/eval/skill_gr_baseline_eval_20260703/bm25_skillsbench_core75_gr_vocab
outputs/eval/qwen3_embedding_06b_21k_current/own_test
outputs/eval/qwen3_embedding_06b_21k_current/skillsbench_core75
outputs/eval/skill_gr_baseline_eval_20260703/skillrouter_own_test_train_vocab_top50
outputs/eval/skill_gr_baseline_eval_20260703/skillrouter_skillsbench_core75_gr_vocab_top50
```

SkillsRouter repro：

```text
outputs/train/skillrouter_repro_20260707/train_pairs_hn.jsonl
outputs/train/skillrouter_repro_20260707/train_pairs_hn.manifest.json
outputs/train/skillrouter_repro_20260707/sr_emb_qwen3_06b_1epoch_10neg
outputs/eval/skillrouter_repro_qwen3_06b_10neg_1epoch/own_test_final
outputs/eval/skillrouter_repro_qwen3_06b_10neg_1epoch/skillsbench_core75_final
```

### 0.17 快速复核命令

统计数据行数：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
for f in \
  outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/decoder_rows.base.jsonl \
  outputs/train/skill_gr_21k_single_docaug_20260703/prepare_dev/decoder_rows.base.jsonl \
  outputs/train/skill_gr_21k_single_docaug_20260703/prepare_test/decoder_rows.base.jsonl \
  outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/selected_skills.jsonl \
  outputs/eval/skill_gr_baseline_eval_20260703/skillsbench_core75.rows.jsonl \
  outputs/train/skillrouter_repro_20260707/train_pairs_hn.jsonl
do
  printf "%s " "$f"; wc -l < "$f"
done
'
```

打印关键 ranked metrics：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp python3 <<'PY'
import json
from pathlib import Path
root=Path('/root/workplace/SkillSearch-data-synthesis')
paths=[
'outputs/eval/skillrouter_repro_qwen3_06b_10neg_1epoch/skillsbench_core75_final/ranked_metrics.json',
'outputs/eval/skill_gr_baseline_eval_20260703/gr_single_docaug_e16_skillsbench_core75_beam50/ranked_metrics.json',
'outputs/eval/letter_lightgcn_single_docaug_d3_a003_b001/skillsbench_core75_best/ranked_metrics.json',
'outputs/eval/qwen3_embedding_06b_21k_current/skillsbench_core75/ranked_metrics.json',
'outputs/eval/skill_gr_baseline_eval_20260703/bm25_skillsbench_core75_gr_vocab/ranked_metrics.json',
]
def overall(obj):
    if 'overall' in obj:
        return obj['overall']
    bs=obj['by_target_skill_count']
    rows=sum(v['rows'] for v in bs.values())
    out={'rows':rows,'total_gold':sum(v['total_gold'] for v in bs.values())}
    for k in ['@1','@10','@50']:
        out[k]={m:sum(v['rows']*v[k][m] for v in bs.values())/rows for m in ['hit','recall','fc']}
    return out
for rel in paths:
    o=overall(json.load(open(root/rel)))
    print(rel)
    print('H1',round(o['@1']['hit'],4),'R10',round(o['@10']['recall'],4),'FC10',round(o['@10']['fc'],4),'R50',round(o['@50']['recall'],4),'FC50',round(o['@50']['fc'],4))
PY
```

检查 official single GT 在训练集频次：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp python3 <<'PY'
import json, collections, statistics
from pathlib import Path
root=Path('/root/workplace/SkillSearch-data-synthesis')
train=root/'outputs/train/skill_gr_21k_single_docaug_20260703/prepare_train/decoder_rows.base.jsonl'
off=root/'outputs/eval/skill_gr_baseline_eval_20260703/skillsbench_core75.rows.jsonl'
freq=collections.Counter()
for line in open(train):
    freq.update(json.loads(line).get('selected_skill_ids') or [])
vals=[]
for line in open(off):
    r=json.loads(line); ids=r.get('selected_skill_ids') or []
    if len(ids)==1:
        sid=ids[0]; vals.append(freq.get(sid,0)); print(freq.get(sid,0), sid)
print('rows',len(vals),'mean',sum(vals)/len(vals),'median',statistics.median(vals))
PY
```

### 0.18 当前不要做的事

不要：

- 不经确认把 `negatives_per_pair=10` 改成 4 或其他值。
- 只为了加速改变 effective batch size 或 max length 后直接汇报为同口径结果。
- 把 official test query 风格直接复制到训练集。
- 继续只生成 generic “I need help...” 风格 query。
- 只看 dev loss 判断 GR 是否有效。
- 把 official SkillsRouter `SR-Emb + SR-Rank` headline 和我们 embedding-only / GR decoder 混比。
- 认为 LETTER 已经解决了 CF signal；当前证据还不足。

### 0.19 旧内容说明

下面从“更新时间：2026-06-29”开始的内容，是历史数据合成阶段 handoff。它仍包含 Reference-400、replay prompt、OpenRouter profile、repair/filter 等历史细节，可能对追溯 21k 数据来源有用，但不是当前实验主线。

更新时间：2026-06-29

本文档面向下一个接手 agent。目标是让接手者不依赖历史对话，也能直接理解当前系统目的、代码状态、远端路径、API 配置、数据输入输出、运行命令和已知边界。

## 1. 当前目标

我们要构造用于 **Skill generative retrieval** 的训练数据：

```text
query -> skill_id 或 skill_id list
```

其中：

- `k=1`：一个真实用户任务 query 应该检索到一个 skill。
- `k=2/3/4`：一个真实用户任务 query 应该检索到多个互补 skills。

这里的 multi-skill 目标不是严格证明 workflow 一定真实可执行，也不是泛化成“多 skill 推荐”。当前目标是生产合法、合理、能训练生成式检索模型学习 **协同信号** 的数据。合法样本可以是：

```text
sequential_workflow: 多个 skill 有明显前后依赖
capability_bundle: 多个互补能力共同支撑一个任务
```

不要把标准重新收紧回“必须是严格 DAG 可执行 workflow”。之前的实验表明，过强 DAG 约束会让系统低产且脆弱。

## 2. 当前冻结版本

本地仓库：

```text
/Users/chenyufei/Documents/GR&Agentic/SkillSearch-data-synthesis
```

远端仓库：

```text
/root/workplace/SkillSearch-data-synthesis
```

2026-06-26 命名变更：

```text
旧项目名：SkillSearch Query Generation
新项目名：SkillSearch Data Synthesis
旧路径：SkillSearch-query-generation
新路径：SkillSearch-data-synthesis
```

说明：

- 本地和远端实际代码入口都已改为 `SkillSearch-data-synthesis`。
- 为兼容历史命令、旧文档和外部脚本，本地和远端都保留了旧路径 symlink：

```text
/Users/chenyufei/Documents/GR&Agentic/SkillSearch-query-generation -> SkillSearch-data-synthesis
/root/workplace/SkillSearch-query-generation -> SkillSearch-data-synthesis
```

- 后续新文档、新命令和新输出目录命名应使用 `SkillSearch-data-synthesis` / `SkillSearch Data Synthesis`。
- 不要因为项目改名去重命名历史 `pilot_100_*`、`pilot_400_*` 数据文件；这些文件名是 replay 兼容边界和 provenance。

prompt 冻结基线 commit：

```text
58f284f chore: freeze skillrouter metadata replay pipeline
```

本文档本身会产生后续提交；接手时以 `git log --oneline -1` 看到的最新提交为准。

固定输入集合的正式名称：

```text
SkillRouter Reference-400 Candidate Set (2026-06-01 freeze)
简称：Reference-400
```

说明：

- `Reference-400` 指 2026-06-01 冻结的 400 条 replay 输入，每个 `k=1/2/3/4` 各 100 条。
- 历史文件名和旧输出中仍可能出现 `pilot_100_*`、`pilot_400_*`，这些是向后兼容路径名，不再作为正式叙述名称。
- 不要为了改名直接重命名历史数据文件；如需迁移文件名，应先建立 manifest / symlink / compatibility plan。

远端新仓库通过 symlink 复用旧数据和 profile：

```text
/root/workplace/SkillSearch-data-synthesis/outputs  -> /root/workplace/SkillSearch/outputs
/root/workplace/SkillSearch-data-synthesis/profiles -> /root/workplace/SkillSearch/profiles
```

旧远端 `/root/workplace/SkillSearch` 里已经删除旧 `scripts/docs/tests` 入口，只保留数据和配置目录，例如：

```text
outputs/
profiles/
data/
fixtures/
external/
checkpoints/
```

备份：

```text
本地完整备份:
/Users/chenyufei/Documents/GR_Agentic_full_backup_20260625_204425.tar.gz

远端旧代码备份:
/root/workplace/SkillSearch_code_backup_20260625_204558.tar.gz
/root/workplace/SkillSearch_old_code_misc_backup_20260625_204812.tar.gz
```

## 3. 为什么冻结这个版本

当前主线是：

```text
固定 SkillRouter Reference-400 candidate inputs
+ SkillRouter-style plain-text query prompt
+ 400 条版本的 metadata-as-body-preview input construction
+ 当前 selected-id repair / quality filter / reclassification
```

我们比较过两个版本：

```text
metadata-as-body-preview: 400 条数据使用的版本
raw body-prefix hybrid: 后续实验版，把原始 skill body prefix 也拼进 prompt
```

实验结论：

- raw body-prefix hybrid 的 prompt token 大约是 metadata-as-body-preview 的 `1.8x`。
- raw body-prefix hybrid 更容易出现 `selected_id_not_in_panel`。
- raw body-prefix hybrid 更容易把 skill body 里的专名、平台、实现细节带进 query。
- raw body-prefix hybrid 没有明显降低 `First... Then... Finally...` 这类 checklist-like 表达。

因此当前冻结版本不默认拼接原始 skill body。原始 body 后续可以用于 hard-case retry、audit 或 verifier，但不作为主 prompt 输入。

## 4. 当前 Pipeline

整体流程：

```text
classified skills
-> fixed Reference-400 sampled single skills / redundant multi-skill panels
-> GPT-4o selector / query generator
-> parse generated selected_skill_ids and query
-> deterministic selected-id repair
-> deterministic quality evaluation
-> workflow_quality_filter
-> accepted / weak / rejected
```

重要点：

- 当前 replay 脚本不重新构造 panels，而是读取 `Reference-400` 中已固定的 1/2/3/4 skill 输入。
- multi-skill 不是直接给 exactly k 个 skill，而是给一个 redundant candidate panel，让 GPT-4o 选择 exactly `k` 个互补 skills。
- 后处理会把 `candidate number -> full skill id`、`basename -> full skill id`、`skill name -> full skill id` 等常见错误做 deterministic repair。
- `query_too_long` 当前是 stat-only warning，不会降级为 weak。

## 5. 关键文件

```text
README.md
HANDOFF.md
tests/test_skillrouter_prompt_alignment.py
```

核心脚本：

```text
scripts/replay_pilot_400_skillrouter_prompt.py
scripts/generate_single_skill_probe.py
scripts/generate_multiskill_probe.py
scripts/evaluate_single_skill_probe.py
scripts/evaluate_multiskill_probe.py
```

辅助脚本：

```text
scripts/build_panel_classification.py
scripts/simulate_candidate_panels.py
scripts/rescore_multiskill_candidates.py
scripts/verify_multiskill_probe.py
scripts/workflow_role_taxonomy.py
```

当前最重要的是 `replay_pilot_400_skillrouter_prompt.py`。它负责：

- 读取固定 `Reference-400` inputs。
- 调用 single/multi prompt builder。
- 调 GPT-4o。
- 写 raw responses。
- parse / repair / score / filter。
- 追加写入 `pilot_400_skillrouter_prompt_replay.jsonl`、`accepted.jsonl`、`rejected.jsonl`。

## 6. 输入数据路径

远端当前通过 `outputs` symlink 使用这些文件。

skill lookup：

```text
outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl
```

行数：

```text
78,361
```

`Reference-400` 固定输入（历史文件名沿用 `pilot_100_*`）：

```text
outputs/eval/multiskill_probe_gpt4o_openrouter/pilot_100_1skill_combined_100.jsonl
outputs/eval/multiskill_probe_gpt4o_openrouter/pilot_100_2skills_combined_100_manual_clean.jsonl
outputs/eval/multiskill_probe_gpt4o_openrouter/pilot_100_3skills_combined_100_manual_clean.jsonl
outputs/eval/multiskill_probe_gpt4o_openrouter/pilot_100_4skills_combined_100_clean.jsonl
```

每个文件 100 行，总共 400 条 replay 输入。

历史 400 条原始 replay 输出：

```text
outputs/eval/multiskill_probe_gpt4o_openrouter_skillrouter_replay_20260605/full_400_t06_tp09_filter
```

历史 400 条 reclassified v2 输出：

```text
outputs/eval/multiskill_probe_gpt4o_openrouter_skillrouter_replay_20260605/full_400_t06_tp09_filter_reclassified_v2
```

历史 raw body-prefix smoke：

```text
outputs/eval/multiskill_probe_gpt4o_openrouter_skillrouter_replay_20260605/bodyprefix_smoke_10each_t06_tp09
```

## 7. LLM API 配置

> **历史配置，后续禁止使用。** 本节的 OpenRouter profile 仅用于复现 2026-06 的既有 outputs。当前和未来生成统一使用 `profiles/gpt4o_huayanapi.json`；参见 0.0.2。

历史远端 profile 路径：

```text
/root/workplace/SkillSearch-data-synthesis/profiles/gpt4o_openrouter.json
```

这个路径是 symlink 到旧 workspace：

```text
/root/workplace/SkillSearch/profiles/gpt4o_openrouter.json
```

profile 结构：

```json
{
  "name": "gpt4o_openrouter",
  "purpose": "skillrouter_query_realization",
  "base_url": "https://openrouter.ai/api/v1",
  "model": "openai/gpt-4o",
  "api_key": "<redacted>",
  "proxy_url": "http://v2raya_proxy:20171",
  "max_tokens": 1000,
  "temperature": 0.45,
  "extra_headers": {
    "HTTP-Referer": "https://localhost",
    "X-Title": "SkillSearch Smoke"
  }
}
```

实际 replay 命令通常覆盖：

```text
temperature: 0.6
top_p: 0.9
max_tokens: 1000
```

注意：

- `api_key` 不要提交到 git。
- `.gitignore` 已忽略 `profiles`。
- 脚本会读取 `proxy_url` 并安装 Python urllib proxy opener。
- OpenRouter 需要 `Authorization: Bearer <api_key>`。
- 额外 headers 会透传到 single/multi generator。

另有 DeepSeek profile，当前冻结 replay 主线不用它：

```text
profiles/deepseek_v32_windhub.json
base_url: https://windhub.cc/v1
model: deepseek-v3-2-251201
purpose: classification_card_dag_verifier
```

## 8. Prompt 细节

### 8.1 Single-Skill Prompt

System：

```text
You are an experienced user of AI assistants. You write clear, realistic task requests that describe what you need to accomplish.
```

User：

```text
Given the following skill specification, write a realistic task description that someone would ask an AI assistant to help with. The task should naturally require the capabilities described in this skill.

Skill name: {skill_name}
Category: {category}
Description: {description}
Skill body: Operation: {operation_role}/{operation_type}; Task objects: {task_object_types}; Inputs: {input_artifact_types}; Outputs: {output_artifact_types}

Requirements:
(1) Describe a concrete scenario with specific inputs/outputs.
(2) Include enough detail that the skill would be clearly useful.
(3) Do NOT mention the skill name "{skill_name}" anywhere in the task.

Output ONLY the task description.
```

注意：这里没有 `Skill id:`，也不要求 JSON 输出。模型只输出 task description，代码再包装成：

```json
{
  "query": "...",
  "target_skill_count": 1,
  "selected_skill_ids": ["..."],
  "selected_skill": {"skill_id": "..."}
}
```

### 8.2 Multi-Skill Prompt

System 同 single。

User：

```text
Given the following candidate skill specifications, first choose exactly {target_skill_count} skills that would be naturally useful together for one realistic task. Then write a realistic task description that someone would ask an AI assistant to help with. The task should naturally require the capabilities described in all selected skills.

Candidate skills:
{candidate_specs}

Requirements:
(1) Describe a concrete scenario with specific inputs/outputs.
(2) Include enough detail that every selected skill would be clearly useful.
(3) Select exactly {target_skill_count} skills, no fewer and no more.
(4) The selected skills should have complementary, non-overlapping roles in one plausible workflow.
(5) Do NOT mention any selected skill name anywhere in the task description.
(6) In the selected skill ids line, copy the exact full Skill id strings from the candidate list. Do not use candidate numbers, skill names, shortened ids, or ids without their namespace prefix.

Output ONLY the following two lines:
Selected skill ids: <comma-separated selected skill ids>
Task description: <task description>
```

每个 candidate spec：

```text
{number}. Skill id: {skill_id}
   Skill name: {skill_name}
   Category: {task_space_id}
   Description: {description}
   Skill body: Operation: {operation_role}/{operation_type}; Task objects: {task_object_types}; Inputs: {input_artifact_types}; Outputs: {output_artifact_types}; Modalities: {modality_types}
```

这里 `Skill body` 是 metadata-as-body-preview，不是 raw skill body。

## 9. 运行命令

### 9.1 本地测试

```bash
cd /Users/chenyufei/Documents/GR\&Agentic/SkillSearch-data-synthesis

python3 -m py_compile \
  scripts/generate_single_skill_probe.py \
  scripts/generate_multiskill_probe.py \
  scripts/replay_pilot_400_skillrouter_prompt.py \
  scripts/build_balanced_topup_inputs.py

python3 -m unittest discover -s tests -p 'test_skillrouter_prompt_alignment.py'
```

### 9.2 远端测试

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
python3 -m py_compile \
  scripts/generate_single_skill_probe.py \
  scripts/generate_multiskill_probe.py \
  scripts/replay_pilot_400_skillrouter_prompt.py \
  scripts/build_balanced_topup_inputs.py
python3 -m unittest discover -s tests -p "test_skillrouter_prompt_alignment.py"
'
```

### 9.3 远端 smoke run

建议先跑每个 `k` 各 2 条：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=outputs/eval/skillrouter_metadata_replay_smoke_2each_$(date +%Y%m%d_%H%M%S)
python3 scripts/replay_pilot_400_skillrouter_prompt.py \
  --profile profiles/gpt4o_openrouter.json \
  --output-dir "$out" \
  --k-values 1,2,3,4 \
  --limit-per-k 2 \
  --workers 4 \
  --timeout 180 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-tokens 1000
'
```

### 9.4 远端完整 400 replay

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=outputs/eval/skillrouter_metadata_replay_full_400_$(date +%Y%m%d_%H%M%S)
python3 scripts/replay_pilot_400_skillrouter_prompt.py \
  --profile profiles/gpt4o_openrouter.json \
  --output-dir "$out" \
  --k-values 1,2,3,4 \
  --workers 4 \
  --timeout 180 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-tokens 1000
'
```

默认参数已经是：

```text
desc_chars_single=420
desc_chars_multi=420
```

不要改成 `1600/600`，那是 body-prefix hybrid 实验版本。

### 9.5 Resume

如果中途失败：

```bash
python3 scripts/replay_pilot_400_skillrouter_prompt.py \
  --profile profiles/gpt4o_openrouter.json \
  --output-dir <same-output-dir> \
  --k-values 1,2,3,4 \
  --workers 4 \
  --timeout 180 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-tokens 1000 \
  --resume
```

`--resume` 会根据 `pilot_400_skillrouter_prompt_replay.jsonl` 里的 `(target_skill_count, index)` 跳过已完成行。

## 10. 输出文件

每个 run 的输出目录包含：

```text
run_config.json
pilot_400_skillrouter_prompt_replay.jsonl
accepted.jsonl
rejected.jsonl
raw_responses/k1/*.json
raw_responses/k2/*.json
raw_responses/k3/*.json
raw_responses/k4/*.json
```

每行主要字段：

```text
index
source_index
source_file
target_skill_count
task_space_id
generated.query
generated.selected_skill_ids
chosen_evaluation
workflow_quality_filter
accepted_for_training
parse_status
raw_response_path
usage
```

训练数据主文件不应该保留 API usage。若要导出干净数据：

```bash
DIR=<run-output-dir>
jq -c '{index, target_skill_count, source_index, source_file, raw_response_path, model, usage}' \
  "$DIR/pilot_400_skillrouter_prompt_replay.jsonl" \
  > "$DIR/pilot_400_usage.jsonl"

jq -c 'del(.usage)' "$DIR/pilot_400_skillrouter_prompt_replay.jsonl" \
  > "$DIR/pilot_400.clean.jsonl"
```

如果要分 high / medium：

```bash
DIR=<run-output-dir>
jq -c 'select(.workflow_quality_filter.status == "accepted") | del(.usage)' \
  "$DIR/pilot_400_skillrouter_prompt_replay.jsonl" \
  > "$DIR/high_confidence.jsonl"

jq -c 'select(.workflow_quality_filter.status == "weak") | del(.usage)' \
  "$DIR/pilot_400_skillrouter_prompt_replay.jsonl" \
  > "$DIR/medium_confidence.jsonl"

jq -c 'select(.workflow_quality_filter.status == "rejected") | del(.usage)' \
  "$DIR/pilot_400_skillrouter_prompt_replay.jsonl" \
  > "$DIR/rejected.clean.jsonl"

jq -c 'select(.accepted_for_training == true) | del(.usage)' \
  "$DIR/pilot_400_skillrouter_prompt_replay.jsonl" \
  > "$DIR/accepted.clean.jsonl"
```

## 11. 后处理技术细节

### 11.1 selected-id repair

函数：

```text
repair_selected_ids
```

支持：

```text
candidate number -> full skill id
case normalization
basename -> full skill id
skill name -> full skill id
```

典型修复：

```text
"1" -> candidate_ids[0]
"synchronize" -> "other/synchronize"
"Spec Writer" -> "dev/spec"
```

### 11.2 deterministic evaluation

single 使用：

```text
generate_single_skill_probe.score_generated
```

multi 使用：

```text
generate_multiskill_probe.score_generated
```

主要检查：

```text
wrong_target_skill_count
wrong_selected_count
duplicate_selected_ids
selected_id_not_in_panel
query_mentions_generation_artifact
query_mentions_skill_id
query_too_short
query_too_long
low_operation_role_diversity
duplicate_operation_type
near_duplicate_selected_skills
unsupported_named_context_in_query
selected_context_requires_query_context
```

### 11.3 workflow_quality_filter

函数：

```text
workflow_quality_filter
```

输出：

```json
{
  "accepted": true,
  "status": "accepted | weak | rejected",
  "reject_reasons": [],
  "weak_reasons": [],
  "ignored_warnings": [],
  "workflow_type": "single_skill | sequential_workflow | capability_bundle | weak_overlap | invalid"
}
```

规则口径：

- 有 deterministic hard issue 就 rejected。
- multi 选出的 skill 不在 panel metadata 中，rejected。
- near duplicate 且 operation type 没有足够差异，rejected。
- 同 role、重复 operation type、弱 shared context 通常 weak。
- 如果没有互补 role path 但有 shared task object，可以接受为 `capability_bundle`。
- `query_too_long` 是 `ignored_warnings`，不会让样本变 weak。

## 12. 400 条参考数据质量

历史 400 条 partner review 数据是：

```text
旧生成 prompt 生成 query
+ 当前后处理逻辑 reclassified
+ 清理 old_generated / usage 等无关字段
```

历史本地 clean partner 数据已不在当前工作目录里，但完整备份中有：

```text
/Users/chenyufei/Documents/GR_Agentic_full_backup_20260625_204425.tar.gz
```

原路径：

```text
/Users/chenyufei/Documents/GR&Agentic/partner_review/0605_skillrouter_replay_reclassified_v2/pilot_400.jsonl
```

历史统计：

```text
all rows: 400
accepted: 396
high_confidence: 357
medium_confidence: 39
rejected: 4
```

400 条真实 usage：

```text
input tokens: 653,324
output tokens: 67,241
total tokens: 720,565
estimated GPT-4o cost at input $2.5/M and output $10/M: $2.30572
```

按 `k`：

| k | rows | input tokens | output tokens | total tokens | estimated cost |
|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 23,853 | 18,661 | 42,514 | $0.24624 |
| 2 | 100 | 197,197 | 14,635 | 211,832 | $0.63934 |
| 3 | 100 | 197,403 | 16,079 | 213,482 | $0.65430 |
| 4 | 100 | 234,871 | 17,866 | 252,737 | $0.76584 |

## 13. 2026-06-26 当前 2000 条数据集状态

当前可用于 GR 训练的 2000 条数据集已经生成并只保留在远端。不要把 JSONL 数据集同步到本地工作目录；本地只保留代码和轻量文档。

远端 canonical clean 文件：

```text
/root/workplace/SkillSearch-data-synthesis/outputs/eval/skillrouter_balanced_500_each_topup_v1/balanced_500_each.final_high_quality.english_only_v2_1.clean.jsonl
```

远端对外命名副本：

```text
/root/workplace/SkillSearch-data-synthesis/outputs/eval/skillrouter_balanced_500_each_topup_v1/example_2000.jsonl
```

二者内容相同：

```text
rows: 2000
sha256: 3651dcfa890c747cc28dc39e2e97db72cfc204978fb4c2e3676b64cd7468faf8
size: 13,576,873 bytes
workflow_quality_filter.status: accepted x 2000
Chinese-character rows: 0
query duplicates: 0
selected skill set duplicates: 0
target_skill_count mismatches: 0
full skill_id leaks: 0
```

行序是按 `target_skill_count` 分块排列：

```text
1-500:     k=1
501-1000:  k=2
1001-1500: k=3
1501-2000: k=4
```

训练读取字段：

```text
query: generated.query
selected ids: generated.selected_skill_ids
k: target_skill_count 或 generated.target_skill_count
```

注意：如果 dataloader 不 shuffle，训练前应生成 shuffled copy，避免 batch 顺序先全是 k1、再全是 k2/k3/k4。

远端校验命令：

```bash
ssh gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
python3 - <<'"'"'PY'"'"'
import hashlib, json
from collections import Counter

p = "outputs/eval/skillrouter_balanced_500_each_topup_v1/example_2000.jsonl"
h = hashlib.sha256()
counts = Counter()
status = Counter()
zh_rows = 0
with open(p, "rb") as f:
    for raw in f:
        h.update(raw)
        row = json.loads(raw)
        counts[row["target_skill_count"]] += 1
        status[row["workflow_quality_filter"]["status"]] += 1
        q = row["generated"]["query"]
        if any("\u4e00" <= ch <= "\u9fff" for ch in q):
            zh_rows += 1
print("sha256", h.hexdigest())
print("by_k", dict(sorted(counts.items())))
print("status", dict(status))
print("zh_rows", zh_rows)
PY
'
```

Google Drive 副本：

```text
name: example_2000.jsonl
file id: 1W0rF16UHvpjdKesx50TyKUr00X5XBEQS
url: https://drive.google.com/open?id=1W0rF16UHvpjdKesx50TyKUr00X5XBEQS
```

Drive 侧回读结果：

```text
Size: 13576873
MD5: cfdf6fc42337f122b882cc9a07fea52b
SHA1: 3734d1971c39b35808fd0d0a1d7113d4ea937adc
SHA256: 3651dcfa890c747cc28dc39e2e97db72cfc204978fb4c2e3676b64cd7468faf8
```

公开访问曾用未登录态 HTTP 请求验证：

```text
HTTP 200
page contains example_2000.jsonl
no "You need access"
no "Request access"
no "Sign in"
```

Drive 预览注意事项：

- `.jsonl` 在 Google Drive 里通常不好预览。
- 上传时 Drive 识别为 `application/octet-stream`。
- 训练主文件保持 `.jsonl` 不改；如果需要人工预览，另传 `example_2000.preview.txt` 或抽关键列做 Google Sheet。

远端上传技术细节：

- 远端已安装 `rclone v1.74.3` 到 `/usr/local/bin/rclone`。
- Google Drive OAuth token 已在上传后删除，避免长期把 Drive 凭据留在远端。
- 远端到 Google API 直连不可达；必须走机内 v2ray 代理。
- 当前容器里可访问的代理地址是 `172.18.0.2:20171`，协议是 HTTP proxy。
- 代理账号密码不要写进 git 文档。使用时通过环境变量传入：

```bash
export PROXY_USER='<provided-out-of-band>'
export PROXY_PASS='<provided-out-of-band>'
export HTTPS_PROXY="http://${PROXY_USER}:${PROXY_PASS}@172.18.0.2:20171"
export HTTP_PROXY="$HTTPS_PROXY"
export ALL_PROXY="$HTTPS_PROXY"
```

如需重新授权远端 `rclone`：

```bash
# 1. 远端启动 headless Drive config，拿到 rclone authorize 命令提示
ssh gpu1_tailscale_jmp '
rclone config create gdrive drive scope drive config_is_local false --non-interactive
'

# 2. 在有浏览器的本机执行远端提示的 rclone authorize 命令。
#    只把 authorize result 写回远端 rclone config；不要上传或落盘数据集到本地。

# 3. 远端验证 Drive API 和代理链路
ssh gpu1_tailscale_jmp '
export PROXY_USER="<provided-out-of-band>"
export PROXY_PASS="<provided-out-of-band>"
export HTTPS_PROXY="http://${PROXY_USER}:${PROXY_PASS}@172.18.0.2:20171"
export HTTP_PROXY="$HTTPS_PROXY"
export ALL_PROXY="$HTTPS_PROXY"
rclone about gdrive:
'
```

如需重新上传或覆盖 Drive 副本：

```bash
ssh gpu1_tailscale_jmp '
set -euo pipefail
cd /root/workplace/SkillSearch-data-synthesis
SRC=outputs/eval/skillrouter_balanced_500_each_topup_v1/example_2000.jsonl
export PROXY_USER="<provided-out-of-band>"
export PROXY_PASS="<provided-out-of-band>"
export HTTPS_PROXY="http://${PROXY_USER}:${PROXY_PASS}@172.18.0.2:20171"
export HTTP_PROXY="$HTTPS_PROXY"
export ALL_PROXY="$HTTPS_PROXY"

rclone copyto "$SRC" "gdrive:example_2000.jsonl" \
  --drive-upload-cutoff 256M \
  --contimeout 10s \
  --timeout 120s

rclone lsjson --hash --metadata "gdrive:example_2000.jsonl"
rclone link "gdrive:example_2000.jsonl"
'
```

后续计划和数据 manifest：

```text
PLAN.md
docs/superpowers/dataset_remote_manifest.md
```

`PLAN.md` 记录长期 pipeline 改进方案；不要把它误认为本次 2000 条数据的生成脚本。本次 2000 条最终数据是经过人工/sub-agent 修复、英文清洗、二次审计和 top-up 后的结果。

当前新增的 sampler 原型脚本：

```text
scripts/build_balanced_topup_inputs.py
```

它只做 deterministic top-up input sampling，不调用模型。主要用途是后续改进 k4 top-up 的 panel 构造。使用前先看 `--help`：

```bash
python3 scripts/build_balanced_topup_inputs.py --help
```

远端示例：

```bash
ssh gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
mkdir -p outputs/eval/skillrouter_balanced_500_each_topup_v1/topup_input_smoke

python3 scripts/build_balanced_topup_inputs.py \
  --skills outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl \
  --output outputs/eval/skillrouter_balanced_500_each_topup_v1/topup_input_smoke/input_4skills.jsonl \
  --summary outputs/eval/skillrouter_balanced_500_each_topup_v1/topup_input_smoke/input_4skills.summary.json \
  --target-skill-count 4 \
  --limit 50 \
  --seed 20260626
'
```

如果要让 replay 脚本消费自定义 input，使用 `--input-1/2/3/4`。例如只跑一小批 k4 smoke：

```bash
ssh gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=outputs/eval/skillrouter_topup_k4_smoke_$(date +%Y%m%d_%H%M%S)
python3 scripts/replay_pilot_400_skillrouter_prompt.py \
  --profile profiles/gpt4o_huayanapi.json \
  --output-dir "$out" \
  --skills outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl \
  --input-4 outputs/eval/skillrouter_balanced_500_each_topup_v1/topup_input_smoke/input_4skills.jsonl \
  --k-values 4 \
  --limit-per-k 10 \
  --workers 4 \
  --timeout 180 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-tokens 1000
'
```

注意：`build_balanced_topup_inputs.py` 只产 candidate panels，不保证最终 query 质量；生成后仍要人工审计，只有 `manual_repair_review_status == "pass"` 或同等人工 pass 标记的数据才能进入最终训练集。

## 14. 2000 条生成式检索训练 Pipeline

当前新增脚本：

```text
scripts/skill_gr_pipeline.py
tests/test_skill_gr_pipeline.py
```

目标是用当前 2000 条数据先跑通最经典的 generative retrieval pipeline：

```text
selected skills in example_2000.jsonl
-> Qwen3-Embedding-0.6B encodes skill documents
-> RQ-VAE discretizes skill embeddings into semantic ids
-> T5-base decoder maps query text to one or more semantic ids
```

这条线只负责把训练 pipeline 跑通，不在这里展开指标、诊断或复杂 ablation。

### 14.0 当前接手快照（2026-06-29）

本节是当前 GR 实验的最新 handoff 摘要。后面的 14.x 小节保留了历史过程和 provenance；接手时先看本节，再按需要追后文细节。

#### 14.0.1 当前目标和判断边界

当前这一步主要用于验证两件事：

```text
1. 当前合成的 2000 条 query-skill 数据是否有质量、是否可检索。
2. 当前基础 GR pipeline 是否能学到信号，以及暴露了哪些后续 scaling / 方法改动仍需注意的问题。
```

不要把当前结果解读为“最终 GR 方法已经可用”。当前最重要的事实是：

```text
数据质量：支持有效。BM25 和 Qwen3 dense 在 test 上都很强。
GR 管线：能学到东西，但明显弱于强 retrieval baseline。
主要瓶颈：SID 表示结构、decoder 对 rare/unseen SID 的泛化、多 skill full coverage。
```

详细实验记录单独整理在：

```text
docs/gr_pipeline_experiment_summary_20260627.md
```

本地与远端都有该文件；远端路径：

```text
/root/workplace/SkillSearch-data-synthesis/docs/gr_pipeline_experiment_summary_20260627.md
```

#### 14.0.2 数据、split、候选集合

主数据：

```text
dataset:
outputs/eval/skillrouter_balanced_500_each_topup_v1/example_2000.jsonl

base output:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803

split:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/decoder_split_rqvae_e100_c256_d3_seed20260627

train/dev/test:
1800 / 100 / 100

dev/test stratification:
k=1/2/3/4 each has 25 rows
```

候选 skills：

```text
selected skills only, not full 78k catalog:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/selected_skills.jsonl

unique selected skills: 3539
skill text: name | description | body
```

注意：当前所有 retrieval / GR eval 都是在这 3539 个 selected skills 上做 candidate set。不要误用全量 78,361 skills。

#### 14.0.3 Embedding 和 RQ-VAE SID

Skill embedding：

```text
model: Qwen/Qwen3-Embedding-0.6B
pooling: last-token pooling
normalize: L2 normalize
skill input: skillrouter_text = name | description | body
```

RQ-VAE：

```text
output:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/rqvae_e100_c256_d3_h512_z128_seed20260627

codebook_size: 256
depth: 3
hidden_dim: 512
latent_dim: 128
semantic id: [c0, c1, c2, dedup]
skills: 3539
raw_unique_codes: 3388 / 3539
collision_groups: 144
max_collision: 4
```

当前 SID 的形式是：

```text
skill -> [RQ0 code, RQ1 code, RQ2 code, dedup column]
```

dedup column 用于处理多个 skills 获得同一组三层 RQ code 的 collision。

#### 14.0.4 主要命令入口

当前新增脚本：

```text
scripts/skill_gr_pipeline.py
tests/test_skill_gr_pipeline.py
```

重要命令：

```text
prepare                         生成 selected_skills.jsonl 和 decoder_rows.base.jsonl
encode-skills                   用 Qwen3-Embedding-0.6B 编码 selected skills
train-rqvae                     训练 RQ-VAE 并生成 semantic_ids.jsonl
train-structured-decoder        multi-SID set generation，可选 catalog pretrain
eval-structured-decoder         multi-SID beam eval，输出 ranked_metrics.json
train-single-sid-decoder        single-SID top-K route，可选 catalog pretrain
eval-single-sid-decoder         single-SID beam eval，输出 ranked_metrics.json
eval-bm25                       BM25 baseline
eval-t5-dense                   vanilla t5-base encoder mean-pooling dense baseline
```

Qwen3 dense baseline 目前是远端一次性脚本跑出的结果，还没有正式合并成 CLI 子命令。输出已保存：

```text
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/baselines/qwen3_embedding_06b_test
```

如需复现，参考 `docs/gr_pipeline_experiment_summary_20260627.md` 里的定义：`Qwen/Qwen3-Embedding-0.6B`、last-token pooling、query 用 `query_text` instruction、skill 用 `skillrouter_text`。

#### 14.0.5 multi-SID、single-SID、catalog indexing 的精确定义

`multi-SID`：

```text
input:  query
target: multiple skill SIDs + stop
```

具体 target 不是文本 `<SKILL_SEP>`，而是 structured decoder 的自定义 SID token 序列：

```text
skill_a: c0 c1 c2 dedup CONTINUE
skill_b: c0 c1 c2 dedup CONTINUE
...
skill_n: c0 c1 c2 dedup STOP
```

多个 skills 的顺序目前是：

```text
sorted(selected_skill_ids)
```

也就是按 `skill_id` 字典序做 canonical order。原因：skill set 本身无序，固定排序可以消除 permutation noise；缺点是字典序不表达 workflow / dependency / complementarity。

structured decoder 中没有使用旧 text route 的 `<SKILL_SEP>`。原因：当前 decoder 不再通过 HF tokenizer 生成 `<RQ0_...>` 文本 token，而是：

```text
T5 encoder encodes query
T5 decoder stack consumes custom SID embeddings
level heads predict c0/c1/c2/dedup
control head predicts continue/stop
```

自定义 SID token id 映射：

```text
RQ token id:      level * codebook_size + code
dedup token id:   depth * codebook_size + d
control token id: depth * codebook_size + dedup_size + {continue=0, stop=1}
BOS token id:     depth * codebook_size + dedup_size + 2
```

`single-SID`：

```text
input: query
target: one positive skill SID
```

一条 multi-skill query 会展开成多条 single-positive 样本：

```text
query -> skill_a SID
query -> skill_b SID
query -> skill_c SID
```

推理时不是只预测一个 skill，而是：

```text
query -> beam top-K SID -> ranked skill list
```

然后用 `Hit@K / Recall@K / FC@K` 评价是否覆盖多个 gold skills。

`catalog indexing` / `doc-aug`：

这里更准确叫 catalog indexing，不是伪查询。

```text
stage 1 input:  skill document = name | description | body
stage 1 target: that skill's SID
```

目的：让模型先学 skill 文档语义到 SID 的 grounding。

它不同于伪查询：

```text
catalog indexing: skill document -> SID
pseudo query:     synthetic query view -> SID
hard negatives:   distinguish positive skill from similar but wrong skills
```

#### 14.0.6 指标口径

统一报告：

```text
Hit@K          = fraction of rows where top-K ranked skills contain at least one gold skill
Recall@K       = mean per-query recovered-gold fraction, macro averaged (SkillRouter official)
micro_recall   = total_hits@K / total_gold, retained as a diagnostic
FC@K           = Full Coverage@K, fraction of rows where all gold skills are covered by top-K
```

默认：

```text
K = 1, 10, 50
generation beam size = 50
```

注意：

```text
@K 是最终 ranked skill list 截断深度。
beam size 是 constrained decoding 搜索宽度。
二者不是同一个概念。
```

后续必须同时报告：

```text
overall
by gold_k
seen vs unseen
by gold_k × seen/unseen
```

原因：当前 split 的 seen/unseen 分布很不均匀，尤其 test `k=1` 几乎是 cold-start skill：

```text
test gold_k=1: seen  2 / 25  = 0.080
test gold_k=2: seen 31 / 50  = 0.620
test gold_k=3: seen 46 / 75  = 0.613
test gold_k=4: seen 45 / 100 = 0.450
```

#### 14.0.7 当前 test 主结果

所有结果均为 test split，100 rows / 250 gold skill assignments。

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

`@50` 按 gold skill 数量分桶：

```text
model                  gold_k  Hit@50  Recall@50  FC@50
BM25                   1       0.84    0.840      0.84
BM25                   2       1.00    0.920      0.84
BM25                   3       1.00    0.773      0.48
BM25                   4       1.00    0.850      0.56

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

#### 14.0.8 当前现象和可延续问题

当前比较稳的判断：

```text
1. 2000 条数据有效。
   BM25 test Recall@50 = 0.840, FC@50 = 0.68。
   Qwen3 dense test Recall@50 = 0.844, FC@50 = 0.70。

2. vanilla T5 dense 很弱不代表 dense retrieval 不行。
   T5 dense test Recall@50 = 0.148。
   Qwen3 dense 已证明 semantic retrieval signal 很强。

3. 生成式检索能学到东西，但明显弱于强 retrieval baseline。
   best generative test Recall@50 = 0.448。

4. single-SID formulation 是主要收益来源。
   multi-SID test Recall@50 = 0.208。
   single-SID test Recall@50 = 0.448。

5. catalog indexing 对 multi-SID 有明显帮助。
   multi-SID -> multi-SID+catalog:
   Recall@50 0.208 -> 0.312
   FC@50     0.13  -> 0.20

6. catalog indexing 对 single-SID 的收益不稳定。
   single-SID Recall@50 = 0.448
   single-SID+catalog Recall@50 = 0.444
   但 FC@50 0.21 -> 0.24。

7. multi-skill 的核心难点是 full coverage，不只是命中一个相关 skill。
   single-SID+catalog k=4 FC@50 = 0.04。

8. multi-SID set generation 在 k=4 上明显崩溃。
   multi-SID k=4 Recall@50 = 0.090, FC@50 = 0.00。
   multi-SID+catalog k=4 Recall@50 = 0.130, FC@50 = 0.00。

9. seen/unseen 长尾强烈影响 by-k 解释，后续必须固定报告。
```

这轮实验对数据质量的结论是正向的：强 lexical / semantic retrieval baseline 都能很好地找回 gold skills。对 GR 方法的结论是：当前基础管线可学习，但 SID 表示和 decoder 泛化是瓶颈。

#### 14.0.9 后续方法方向和注意事项

用户当前倾向的后续方向：

```text
1. scaling 数据。
2. 让 SID 更结构化，适配 multi-skill 协同信号。
3. 引入 hard negatives。
4. 伪查询 / multi-view synthetic query augmentation 作为 rare/unseen skill 的 query-side supervision 补充。
```

需要注意的概念区分：

```text
catalog indexing:
  skill document -> SID
  学 skill catalog 到 SID 的 grounding。

pseudo query:
  synthetic query view -> SID
  补 low-frequency / unseen-positive skills 的 query-side supervision。

hard negatives:
  positive skill vs similar wrong skills
  解决 lexical shortcut 和 fine-grained discrimination。
```

当前建议优先诊断：

```text
1. SID locality:
   - sharing c0 / c0,c1 是否语义相近
   - prefix fanout 是否过度不均衡
   - RQ-VAE collisions / dedup 分布

2. per-level teacher-forcing accuracy:
   - c0 / c1 / c2 / dedup 哪一层最容易错
   - seen vs unseen
   - gold_k 分桶

3. single-SID top-K error analysis:
   - missed gold skills 是否被 BM25 / Qwen3 dense 找回
   - misses 是否集中在特定 SID prefix

4. SID construction 对照：
   - 若 RQ-VAE locality 差，尝试 hierarchical k-means / balanced tree SID。
```

### 14.1 参考依据和不能改的口径

优先参考这些公开实现/论文，而不是自行设计格式：

```text
SkillsRouter paper: https://arxiv.org/html/2603.22455v4
SkillsRouter code: https://github.com/zhengyanzhao1997/SkillRouter
Qwen3-Embedding-0.6B: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
SkillsRouter embedding model reference: https://huggingface.co/pipizhao/SkillRouter-Embedding-0.6B
RQ-VAE recommender reference: https://github.com/EdoardoBotta/RQ-VAE-Recommender
ResidualVQ library: https://github.com/lucidrains/vector-quantize-pytorch
Multi-label T5 reference: https://aclanthology.org/2023.findings-acl.360/
Multi-label T5 code: https://github.com/coastalcph/Multi-Label-Classification-T5
```

SkillRouter 相关固定口径：

- 当前先试 base embedding 模型 `Qwen/Qwen3-Embedding-0.6B`，不要默认使用 SkillRouter 发布的微调模型 `pipizhao/SkillRouter-Embedding-0.6B`。
- `pipizhao/SkillRouter-Embedding-0.6B` 是 SkillRouter 发布的 `SR-Emb-0.6B`，基于 `Qwen3-Embedding-0.6B` fine-tune；它只作为格式和后续对照参考。
- skill-side text 使用 SkillRouter Appendix/code 的形式：

```text
<name> | <description> | <body>
```

- 当前脚本默认截断：

```text
description: 300 chars
body: 2500 chars
embedding max_length: 2048 tokens
```

- query-side embedding instruction 使用 SkillRouter embedding model card 的 task instruction：

```text
Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task
Query:
```

不要再把 `operation_role`、`task_object_types`、`input_artifact_types` 等 metadata 拼进 GR 训练阶段的 skill encoding。那些字段只属于前面的 query/data synthesis prompt，不属于 SkillsRouter-style skill document embedding。

### 14.2 训练数据范围

GR 训练阶段只编码 `example_2000.jsonl` 中实际被选中的 skills，方便后续基于当前训练 vocabulary 构建 constrained decoding trie。

已核验：

```text
rows: 2000
unique selected skills: 3539
all selected skills found in easy_depth3_panel_classified.jsonl
```

输入路径：

```text
dataset: outputs/eval/skillrouter_balanced_500_each_topup_v1/example_2000.jsonl
skill lookup: outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl
```

`prepare` 阶段从 `generated.selected_skill_ids` 收集 skill id，只输出这 3539 个 skill 的 `selected_skills.jsonl`。skill 的完整 `name/description/body` 必须从 skill lookup 取，不能从 panel preview 或 query generation row 里补。

### 14.3 Decoder 怎么知道输出几个 skills

当前采用有公开参考的 seq2seq multi-label formulation，而不是手写一个无来源的 `<K>` 控制方案：

```text
decoder input:  skill routing query: <query>
decoder target: <semantic-id-of-skill-a> <SKILL_SEP> <semantic-id-of-skill-b> ...
```

每个 skill semantic id 是 RQ-VAE codes 加一个 collision dedup column：

```text
<RQ0_...> <RQ1_...> <RQ2_...> <DUP_...>
```

多 skill target 按 skill id 字典序排序后用 `<SKILL_SEP>` 连接。T5 作为 encoder-decoder 模型从训练 target 中学习输出几个 label/skill；生成到 EOS 停止。这对应 multi-label T5 的序列化标签思路：多标签作为一个目标文本序列生成，用 delimiter 分隔，而不是固定输出槽位。

RQ-VAE semantic id collision 处理参考 `RQ-VAE-Recommender`：如果多个 item/skill 得到同一组 RQ codes，就追加一个 deterministic dedup column，值为该 code tuple 之前出现过的次数。

### 14.4 远端依赖

远端训练需要：

```text
torch
transformers
accelerate
sentencepiece
vector-quantize-pytorch
```

2026-06-26 已检查并安装后，远端状态：

```text
torch: OK
transformers: OK
accelerate: OK
sentencepiece: OK
vector-quantize-pytorch: OK
```

如新环境缺依赖，安装缺失项：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
python3 -m pip install --user sentencepiece vector-quantize-pytorch
'
```

远端直连 `huggingface.co` 会出现 `Network is unreachable`。本次已验证 `hf-mirror.com` 可用，下载 Hugging Face 模型时在命令前加：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

### 14.5 远端 Smoke Run

同步脚本后在远端跑：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
python3 -m py_compile scripts/skill_gr_pipeline.py
python3 -m unittest discover -s tests -p "test_skill_gr_pipeline.py"
'
```

准备 2000 rows / 3539 selected skills：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=outputs/train/skill_gr_rqvae_t5_$(date +%Y%m%d_%H%M%S)
python3 scripts/skill_gr_pipeline.py prepare \
  --dataset outputs/eval/skillrouter_balanced_500_each_topup_v1/example_2000.jsonl \
  --skills outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl \
  --output-dir "$out"
'
```

编码 selected skills：

```bash
HF_ENDPOINT=https://hf-mirror.com python3 scripts/skill_gr_pipeline.py encode-skills \
  --skills-jsonl "$out/selected_skills.jsonl" \
  --output-dir "$out" \
  --model Qwen/Qwen3-Embedding-0.6B \
  --batch-size 4
```

先用短 RQ-VAE smoke 确认链路：

```bash
python3 scripts/skill_gr_pipeline.py train-rqvae \
  --embeddings "$out/skill_embeddings.pt" \
  --output-dir "$out/rqvae" \
  --epochs 2 \
  --batch-size 256
```

构造 decoder 数据：

```bash
python3 scripts/skill_gr_pipeline.py build-decoder-data \
  --rows "$out/decoder_rows.base.jsonl" \
  --semantic-ids "$out/rqvae/semantic_ids.jsonl" \
  --output-dir "$out/decoder_data"
```

T5-base 最小 smoke：

```bash
HF_ENDPOINT=https://hf-mirror.com python3 scripts/skill_gr_pipeline.py train-decoder \
  --decoder-rows "$out/decoder_data/decoder_rows.jsonl" \
  --special-tokens "$out/decoder_data/decoder_special_tokens.json" \
  --output-dir "$out/t5_smoke" \
  --model t5-base \
  --max-steps 5 \
  --batch-size 1
```

如果 GPU 不支持 bf16，改成 `--fp16` 或不传混精参数。

2026-06-27 已跑通的 base Qwen 0.6B 远端 smoke 输出：

```text
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803
```

结果：

```text
prepare: rows=2000, unique_selected_skills=3539
encode-skills: model=Qwen/Qwen3-Embedding-0.6B, batch_size=4, max_length=2048
train-rqvae smoke: epochs=2, codebook_size=256, depth=3, code_width=3, collision_groups=438
build-decoder-data: rows=2000, special_tokens=162
train-decoder smoke: model=t5-base, max_steps=5, train_rows=1800, dev_rows=200
```

2026-06-27 正式 RQ-VAE 训练输出：

```text
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/rqvae_e100_c256_d3_h512_z128_seed20260627
```

训练配置和结果：

```text
embedding: Qwen/Qwen3-Embedding-0.6B skill_embeddings.pt
skills: 3539
codebook_size: 256
depth: 3
hidden_dim: 512
latent_dim: 128
batch_size: 256
epochs: 100
lr: 1e-3
weight_decay: 1e-4
commit_weight: 1.0
seed: 20260627
final_loss: 0.0004080556
collision_groups: 144
unique_raw_codes: 3388
unique_semantic_ids after dedup: 3539
dedup_max: 3
```

过拟合诊断：

```text
diagnostic split: train=3185, heldout=354
same hyperparameters, diagnostic only, not saved as official model
epoch=1:   train_mse=0.00064254, heldout_mse=0.00064320
epoch=20:  train_mse=0.00043347, heldout_mse=0.00044419
epoch=60:  train_mse=0.00036666, heldout_mse=0.00039360
epoch=100: train_mse=0.00035852, heldout_mse=0.00039316
```

结论：heldout MSE 和 train MSE 同步下降，最终 gap 约 9.7%。没有看到明显训练集记忆导致 heldout 崩掉的过拟合形态。100 epoch 后收益很小，后续若只为节省时间，`60-80` epoch 已基本够用；当前正式产物保留 `100` epoch 版本。

对应 decoder 数据已经生成：

```text
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/decoder_data_rqvae_e100_c256_d3
rows: 2000
special_tokens: 696
```

### 14.7 Decoder 训练策略

参考依据：

```text
SGM: Sequence Generation Model for Multi-label Classification, COLING 2018
Deep Reinforced Sequence-to-Set Model for Multi-Label Classification, ACL 2019
TIGER / Recommender Systems with Generative Retrieval, NeurIPS 2023
DSI / Transformer Memory as a Differentiable Search Index, NeurIPS 2022
GENRE / Autoregressive Entity Retrieval, ICLR 2021
```

当前任务是多 skill set 生成，不是单一 docid 生成。采用保守的 seq2seq multi-label formulation：

```text
input:  skill routing query: <query>
target: <RQ0_xxx> <RQ1_xxx> <RQ2_xxx> <DUP_xxx> [<SKILL_SEP> ...] </s>
```

训练 loss：

```text
standard teacher-forcing token-level cross entropy
pad tokens ignored by DataCollatorForSeq2Seq
no BCE head
no extra count-classification loss
no hand-written <K> token
```

理由：

- SGM 将 multi-label classification 转成 label sequence generation，用 decoder 条件化前序 label 来建模 label dependency。
- TIGER/DSI 类 GR 工作把 retrieval target 表示成 token 序列，用 autoregressive CE 训练 decoder 生成 semantic/document identifiers。
- GENRE/DSI/NCI 一类 GR 推理通常配合 constrained decoding / trie，避免生成非法 identifier。
- Sequence-to-set 工作指出固定 label order 会引入 order sensitivity；当前先用 deterministic canonical ordering 降低监督噪声，不做 RL/dynamic order。若后续发现 order/count 是主要错误，再考虑 sequence-to-set 或 dynamic label-order。

数据 split 已固定：

```text
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/decoder_split_rqvae_e100_c256_d3_seed20260627
```

分层字段：

```text
target_skill_count
```

split：

```text
train.jsonl: 1800 rows, k=1/2/3/4 each 450
dev.jsonl:    100 rows, k=1/2/3/4 each 25
test.jsonl:   100 rows, k=1/2/3/4 each 25
seed: 20260627
```

命令：

```bash
python3 scripts/skill_gr_pipeline.py split-decoder-data \
  --rows outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/decoder_data_rqvae_e100_c256_d3/decoder_rows.jsonl \
  --output-dir outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/decoder_split_rqvae_e100_c256_d3_seed20260627 \
  --dev-ratio 0.05 \
  --test-ratio 0.05 \
  --seed 20260627 \
  --stratify-field target_skill_count
```

推荐第一轮 decoder 训练命令：

```bash
base=outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803
split=$base/decoder_split_rqvae_e100_c256_d3_seed20260627
data=$base/decoder_data_rqvae_e100_c256_d3

HF_ENDPOINT=https://hf-mirror.com python3 scripts/skill_gr_pipeline.py train-decoder \
  --decoder-rows "$split/train.jsonl" \
  --dev-rows "$split/dev.jsonl" \
  --special-tokens "$data/decoder_special_tokens.json" \
  --output-dir "$base/t5_base_decoder_rqvae_e100_split_seed20260627_best" \
  --model t5-base \
  --max-source-length 512 \
  --max-target-length 64 \
  --batch-size 8 \
  --epochs 20 \
  --lr 3e-4 \
  --logging-steps 25 \
  --eval-steps 50 \
  --save-steps 100 \
  --save-total-limit 3 \
  --seed 20260626 \
  --bf16
```

如果 bf16 不可用，去掉 `--bf16`。`train-decoder` 当前会用 `--seed` 对训练行做 shuffle；split seed 是 `20260627`，上面命令里的 `--seed 20260626` 是本次 decoder 训练 shuffle seed。脚本默认在有 dev set 时启用 `load_best_model_at_end=True`，并按 `eval_loss` 选择 best checkpoint；需要关闭时显式加 `--no-load-best-model-at-end`。调参阶段只看 dev；`test.jsonl` 只在模型和 decoding 方案固定后做一次客观读数。推理阶段应基于 `rqvae_e100.../semantic_ids.jsonl` 构建 trie，约束合法 semantic-id 序列；否则 T5 可能生成不存在的 code tuple。

2026-06-27 已完成的正式 decoder 训练输出：

```text
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/t5_base_decoder_rqvae_e100_split_seed20260627_best
```

训练配置：

```text
model: t5-base
train_rows/dev_rows/test_rows: 1800 / 100 / 100
batch_size: 8
epochs: 20
lr: 3e-4
bf16: true
max_source_length: 512
max_target_length: 64
eval_steps: 50
save_steps: 100
save_total_limit: 3
split seed: 20260627
decoder train shuffle seed: 20260626
load_best_model_at_end: true
```

结果摘要：

```text
train_runtime: 509.3s
train_loss: 4.767
best_model_checkpoint: checkpoint-3000
best_eval_loss: 3.769047737121582
best_epoch: 13.33
last_eval_loss: 3.892
last_epoch: 20
retained checkpoints: checkpoint-3000, checkpoint-4400, checkpoint-4500
```

解释：

- `checkpoint-3000` 是 dev loss 最优点；epoch 15 后 train loss 继续下降，但 dev loss 稳定回升，说明小数据设置下继续训练已经轻微过训练。
- `final/` 已由 trainer 加载 best checkpoint 后保存，不是最后一步权重。
- `final/` 可以被 `AutoTokenizer` / `AutoModelForSeq2SeqLM` 正常加载；tokenizer length 和 model embedding rows 均为 `32796`。
- special token 检查正常：`decoder_special_tokens.json` 里 696 个任务 token 已加入 tokenizer；`<SKILL_SEP>` 是单 token id `32100`，例如 `<RQ2_255>` 是单 token id `32795`。

不要把下面这个早先输出作为正式推理模型：

```text
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/t5_base_decoder_rqvae_e100_split_seed20260627
```

该目录用于第一轮曲线诊断：`save_total_limit=2` 只保留了最后两个 checkpoint，best checkpoint 已被清掉；其 `final/` 是最后一步权重，不是 best dev loss 权重。它的曲线仍有参考价值：best dev loss 约 `3.746` 出现在 epoch 14.67 左右，last dev loss 约 `3.844`。

已构建的 T5-base decoding trie：

```text
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/decoding_trie_rqvae_e100_t5_base
```

产物：

```text
semantic_id_trie.json
semantic_id_token_ids.jsonl
decoding_trie_manifest.json
```

manifest 摘要：

```text
skills: 3539
unique_sequences: 3539
sequence_length: 4
trie_nodes: 9894
root_branching: 256
separator_token: <SKILL_SEP>
separator_token_id: 32100
eos_token: </s>
eos_token_id: 1
max_skills: 4
```

注意：这里构建的是单个 skill semantic id 的合法前缀树，不是 2000 条 target set 的全文前缀树。推理时应重复使用该 trie：每生成完一个合法 semantic id 后，允许 EOS 或 `<SKILL_SEP>`；如果选择 `<SKILL_SEP>`，状态回到 trie root 生成下一个 skill。这样约束合法 skill id，同时不把 decoder 限制在训练集中见过的 skill set 组合上。解码时还应动态记录已输出 semantic ids，避免重复 skill。

2026-06-27 已完成 constrained decoding 评估：

主命令模板：

```bash
base=outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803

python3 scripts/skill_gr_pipeline.py eval-decoder \
  --rows "$base/decoder_split_rqvae_e100_c256_d3_seed20260627/dev.jsonl" \
  --model-dir "$base/t5_base_decoder_rqvae_e100_split_seed20260627_best/final" \
  --trie "$base/decoding_trie_rqvae_e100_t5_base/semantic_id_trie.json" \
  --semantic-id-token-ids "$base/decoding_trie_rqvae_e100_t5_base/semantic_id_token_ids.jsonl" \
  --output-dir "$base/t5_base_decoder_rqvae_e100_split_seed20260627_best/eval_dev_beam10_dedup" \
  --split-name dev \
  --batch-size 8 \
  --num-beams 10 \
  --max-skills 4 \
  --max-new-tokens 24
```

指标口径：

```text
当前不是 Recall@K / Hits@K。
num_beams=10 只是 constrained beam search 的搜索宽度；评估只取最高分的 1 条生成序列。
每条生成序列本身可以包含 1-4 个 predicted skills，所以指标是 top-1 generated set 上的 set metrics。

hit_count = |predicted_skill_ids ∩ gold_skill_ids| for the top-1 generated set
per-row recall = hit_count / |gold_skill_ids| for the top-1 generated set
micro_recall = total_hits / total_gold
micro_precision = total_hits / total_pred
hit_rate = fraction of rows with hit_count > 0 for the top-1 generated set
exact_match_rate / EM = fraction of rows where predicted skill set == gold skill set, order-insensitive
count_accuracy = fraction of rows where |predicted set| == |gold set|
invalid_rate = fraction of rows with invalid or unknown generated semantic ids
duplicate_predictions = duplicated predicted skill ids before set reduction
```

主结果使用 `checkpoint-3000` 加载后的 `final/`，constrained beam search `num_beams=10`，并在 trie 解码状态中动态屏蔽已生成 semantic id，避免重复 skill。当前应该同时报告 `total_hits` 和 `micro_recall`；`hit_rate`、`exact_match_rate`、`count_accuracy`、`invalid_rate` 作为辅助诊断。

dev 结果：

```text
output: t5_base_decoder_rqvae_e100_split_seed20260627_best/eval_dev_beam10_dedup
rows: 100
total_gold: 250
total_pred: 246
total_hits: 3
micro_recall: 0.012
macro_recall: 0.008333333333333333
micro_precision: 0.012195121951219513
macro_precision: 0.008333333333333333
hit_rate: 0.03
exact_match_rate: 0.0
count_accuracy: 0.74
invalid_rate: 0.0
duplicate_predictions: 0
```

dev 按目标 skill 数量分桶：

```text
k=1: rows=25, total_gold=25,  total_pred=29, total_hits=0, micro_recall=0.0,        hit_rate=0.0,  count_accuracy=0.88
k=2: rows=25, total_gold=50,  total_pred=50, total_hits=0, micro_recall=0.0,        hit_rate=0.0,  count_accuracy=0.76
k=3: rows=25, total_gold=75,  total_pred=77, total_hits=1, micro_recall=0.0133333,  hit_rate=0.04, count_accuracy=0.68
k=4: rows=25, total_gold=100, total_pred=90, total_hits=2, micro_recall=0.02,       hit_rate=0.08, count_accuracy=0.64
```

test 结果（在主配置固定后跑了一次，不再用于调参）：

```text
output: t5_base_decoder_rqvae_e100_split_seed20260627_best/eval_test_beam10_dedup
rows: 100
total_gold: 250
total_pred: 243
total_hits: 0
micro_recall: 0.0
macro_recall: 0.0
micro_precision: 0.0
macro_precision: 0.0
hit_rate: 0.0
exact_match_rate: 0.0
count_accuracy: 0.79
invalid_rate: 0.0
duplicate_predictions: 0
```

test 按目标 skill 数量分桶：

```text
k=1: rows=25, total_gold=25,  total_pred=26, total_hits=0, micro_recall=0.0, hit_rate=0.0, count_accuracy=0.96
k=2: rows=25, total_gold=50,  total_pred=52, total_hits=0, micro_recall=0.0, hit_rate=0.0, count_accuracy=0.80
k=3: rows=25, total_gold=75,  total_pred=72, total_hits=0, micro_recall=0.0, hit_rate=0.0, count_accuracy=0.68
k=4: rows=25, total_gold=100, total_pred=93, total_hits=0, micro_recall=0.0, hit_rate=0.0, count_accuracy=0.72
```

辅助检查：

```text
gold decoder_target -> tokenizer -> trie parser sanity check:
rows_checked: 100
bad: 0

dev beam=50 dedup:
total_hits: 1
micro_recall: 0.004
hit_rate: 0.01
count_accuracy: 0.74
invalid_rate: 0.0

train head100 beam=10 dedup:
total_hits: 1
micro_recall: 0.004166666666666667
hit_rate: 0.01
count_accuracy: 1.0
invalid_rate: 0.0

checkpoint-4500 dev beam=10 dedup:
total_hits: 0
micro_recall: 0.0
hit_rate: 0.0
count_accuracy: 0.74
invalid_rate: 0.0
```

结论：当前 target serialization、tokenizer special tokens、trie、semantic-id 映射是自洽的，约束解码也没有非法 id 或重复 id；但 decoder 基本没有学会 query -> skill semantic-id 的映射。dev/test 的 hits 与 recall 都接近 0，train head100 也只有 `1/240` 命中，因此这不是普通验证集泛化波动，而是当前 2000 条数据 + T5-base 直接生成 RQ-VAE semantic id 的训练目标本身没有跑出有效映射。

2026-06-27 纠正：上面这个 `train-decoder` / `eval-decoder` 路线是失败的 legacy text decoder 实现，不应作为后续主线。问题不是 RQ-VAE semantic id 自身不可用，而是把 RQ codes 作为 Hugging Face 新增 special tokens 交给 T5 LM head 少量 SFT，和 TIGER/RQ-VAE-Recommender 的 structured SID decoder 不对齐。

正确主线应走 TIGER-style structured decoder：

```text
query text -> T5 encoder
previous SID code tokens -> T5 decoder stack with SID embedding table
decoder hidden -> per-level classification heads

level0 head: 256-way CE for c0
level1 head: 256-way CE for c1
level2 head: 256-way CE for c2
dedup head:  dedup_size-way CE for collision column d
control head: continue/stop for multi-skill count
```

当前实现已经加入：

```text
scripts/skill_gr_pipeline.py train-structured-decoder
scripts/skill_gr_pipeline.py eval-structured-decoder
```

关键实现点：

```text
semantic id: [c0, c1, c2, d]
global SID token id for decoder input embedding:
  RQ level token: level * codebook_size + code
  dedup token:    depth * codebook_size + d
  control token:  depth * codebook_size + dedup_size + {continue=0, stop=1}
  BOS token:      depth * codebook_size + dedup_size + 2

training labels:
  per-step type controls which head is used
  loss = CE(RQ0) + CE(RQ1) + CE(RQ2) + CE(dedup) + CE(control), averaged over target steps

decoding:
  constrained beam search over semantic-id trie built from current 3539 skills
  control step emits continue/stop
  repeated completed semantic ids are blocked dynamically
```

注意：`build-decoder-data --target-mode native` 只保留为诊断选项，不是正式路线；`build-decoder-data` 的默认值已改回 legacy `special` 以复现旧失败结果。后续不要继续优化 legacy text decoder，除非专门做对照。

structured decoder overfit sanity：

```text
8-row overfit:
output: debug_sanity/structured_t5_overfit8_steps400
train_loss final: 0.001220
eval rows: 8
total_gold: 16
total_pred: 16
total_hits: 16
micro_recall: 1.0
exact_match_rate: 1.0
invalid_rate: 0.0

32-row overfit:
output: debug_sanity/structured_t5_overfit32_steps600
eval rows: 32
total_gold: 68
total_pred: 67
total_hits: 67
micro_recall: 0.9852941176470589
exact_match_rate: 0.96875
invalid_rate: 0.0
```

这两个 sanity check 证明 structured decoder 能记住小样本；旧 special-token text decoder 在同样 8/32 条 overfit 下分别只有 `2/16` 和 `1/68` hits，因此旧失败主要是 target/output-head 实现问题。

第一轮 full structured 训练：

```bash
base=outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803
out=$base/structured_t5_decoder_rqvae_e100_split_seed20260627_steps2000

python3 scripts/skill_gr_pipeline.py train-structured-decoder \
  --rows "$base/decoder_split_rqvae_e100_c256_d3_seed20260627/train.jsonl" \
  --dev-rows "$base/decoder_split_rqvae_e100_c256_d3_seed20260627/dev.jsonl" \
  --semantic-ids "$base/rqvae_e100_c256_d3_h512_z128_seed20260627/semantic_ids.jsonl" \
  --output-dir "$out" \
  --model "$base/t5_base_decoder_rqvae_e100_split_seed20260627_best/final" \
  --codebook-size 256 \
  --batch-size 8 \
  --max-steps 2000 \
  --lr 3e-4 \
  --weight-decay 0.0 \
  --logging-steps 50 \
  --eval-steps 200 \
  --seed 20260626 \
  --bf16 \
  --load-best-model-at-end
```

说明：这里 `--model` 暂时用了本地已有的 T5 checkpoint 路径以避免远端 Hugging Face 网络不稳定；它复用了 T5 encoder/decoder stack，不使用旧 LM head 或旧 special SID tokens。后续正式实验应优先尝试直接从 clean `t5-base` 初始化，如果远端缓存/网络恢复。

训练曲线摘要：

```text
step=200  dev_loss=2.867466
step=800  dev_loss=2.757435
step=1200 dev_loss=2.621613
step=1600 dev_loss=2.509118
step=2000 dev_loss=2.416690
```

dev constrained decoding 结果：

```text
output: structured_t5_decoder_rqvae_e100_split_seed20260627_steps2000/eval_dev_beam10
metric scope: top-1 generated set under beam=10 constrained decoding; not Recall@10
rows: 100
total_gold: 250
total_pred: 245
total_hits: 23
micro_recall: 0.092
micro_precision: 0.09387755102040816
macro_recall: 0.08416666666666664
hit_rate: 0.22
exact_match_rate: 0.0
count_accuracy: 0.73
invalid_rate: 0.0
duplicate_predictions: 0
```

dev 按目标 skill 数量分桶：

```text
k=1: rows=25, total_gold=25,  total_pred=31, total_hits=1, micro_recall=0.04,       hit_rate=0.04, count_accuracy=0.84
k=2: rows=25, total_gold=50,  total_pred=52, total_hits=5, micro_recall=0.10,       hit_rate=0.20, count_accuracy=0.84
k=3: rows=25, total_gold=75,  total_pred=75, total_hits=8, micro_recall=0.1066667,  hit_rate=0.28, count_accuracy=0.76
k=4: rows=25, total_gold=100, total_pred=87, total_hits=9, micro_recall=0.09,       hit_rate=0.36, count_accuracy=0.48
```

train head100 constrained decoding 结果：

```text
output: structured_t5_decoder_rqvae_e100_split_seed20260627_steps2000/eval_train_head100_beam10
metric scope: top-1 generated set under beam=10 constrained decoding; not Recall@10
rows: 100
total_gold: 240
total_pred: 238
total_hits: 34
micro_recall: 0.14166666666666666
micro_precision: 0.14285714285714285
hit_rate: 0.31
exact_match_rate: 0.04
count_accuracy: 0.98
invalid_rate: 0.0
duplicate_predictions: 0
```

解释：structured decoder 仍然远未达到可用质量，但已明显优于旧 special-token text decoder：dev 从 `3/250` hits 提升到 `23/250`，train head100 从 `1/240` 提升到 `34/240`，且 8/32 条 overfit sanity 基本通过。下一步应在这个 route 上继续调模型初始化、训练步数、learning rate、是否冻结 encoder、以及 split 中未见 skill 的评估口径；不要回到旧 text special-token decoder。

2026-06-27 新增统一 ranked 指标与 baseline：

后续报告 decoder / retrieval baseline 时统一使用 `Hit@K / Recall@K / FC@K`，默认 `K={1,10,50}`。

```text
Hit@K          = fraction of rows where top-K ranked skills contain at least one gold skill
Recall@K       = mean per-query recovered-gold fraction, macro averaged (SkillRouter official)
micro_recall   = total_hits@K / total_gold, retained as a diagnostic
FC@K           = Full Coverage@K, fraction of rows where all gold skills are covered by top-K
```

注意：`@K` 指最终 ranked skill list 的截断深度，不是 beam size。structured decoder 当前用 `num_beams=50` 生成若干合法 skill set，再按 beam score 汇聚成 ranked skill list；BM25 / T5 dense baseline 没有 beam search。下面结果只在 dev split 上跑，test split 仍应等模型和 decoding 方案固定后再一次性评估。

新增命令：

```text
scripts/skill_gr_pipeline.py eval-bm25
scripts/skill_gr_pipeline.py eval-t5-dense
scripts/skill_gr_pipeline.py eval-structured-decoder --k-values 1,10,50 --num-beams 50
scripts/skill_gr_pipeline.py train-single-sid-decoder
scripts/skill_gr_pipeline.py eval-single-sid-decoder --k-values 1,10,50 --num-beams 50
```

输出目录：

```text
BM25:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/baselines/bm25_dev

vanilla T5-base dense:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/baselines/t5_base_dense_dev

structured decoder, beam=50:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/structured_t5_decoder_rqvae_e100_split_seed20260627_steps2000/eval_dev_beam50_ranked

single-SID + catalog indexing, beam=50:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/single_sid_catalog_t5_base_rqvae_e100_split_seed20260627_cat3_q3000/eval_dev_beam50_ranked
```

dev overall:

```text
model                  Hit@1  Recall@1  FC@1   Hit@10  Recall@10  FC@10  Hit@50  Recall@50  FC@50
BM25                   0.70   0.280     0.13   0.94    0.712      0.54   0.98    0.876      0.75
vanilla T5-base dense  0.01   0.004     0.00   0.12    0.056      0.03   0.25    0.112      0.08
structured decoder     0.07   0.028     0.00   0.35    0.164      0.07   0.48    0.252      0.13
single-SID + catalog   0.11   0.044     0.00   0.31    0.188      0.07   0.63    0.424      0.25
```

dev `@50` 按 gold skill 数量分桶：

```text
model                  gold_k  Hit@50  Recall@50  FC@50
BM25                   1       0.92    0.920      0.92
BM25                   2       1.00    0.940      0.88
BM25                   3       1.00    0.867      0.68
BM25                   4       1.00    0.840      0.52
vanilla T5-base dense  1       0.32    0.320      0.32
vanilla T5-base dense  2       0.20    0.100      0.00
vanilla T5-base dense  3       0.28    0.107      0.00
vanilla T5-base dense  4       0.20    0.070      0.00
structured decoder     1       0.36    0.360      0.36
structured decoder     2       0.56    0.340      0.12
structured decoder     3       0.48    0.240      0.04
structured decoder     4       0.52    0.190      0.00
single-SID + catalog   1       0.48    0.480      0.48
single-SID + catalog   2       0.76    0.500      0.24
single-SID + catalog   3       0.68    0.467      0.28
single-SID + catalog   4       0.60    0.340      0.00
```

当前诊断：

- BM25 很强，说明 dev queries 对 gold skills 有明显 lexical signal；至少在这 2000 条数据上，简单 lexical retrieval 是一个必须超过的下限。
- vanilla `t5-base` dense retrieval 很弱，说明未微调 T5 encoder 的 dense space 不能直接用于这个 skill retrieval 任务。
- structured decoder 明显高于 vanilla T5 dense，但远低于 BM25，尤其多 skill 场景的 `FC@50` 很低；这表明当前生成模型确实学到了一些 query -> skill 信号，但还没有学到足够稳定的多 skill 覆盖能力。
- `single-SID + catalog indexing` 是当前更合理的主线：把多 skill query 拆成多个 `(query, one positive skill SID)` 训练样本，推理时用 beam top-K 作为 ranked skills；多 skill 预测通过 `Recall@K/FC@K` 评价，不再要求 decoder 一次生成完整 skill set。

2026-06-27 已跑 B 方案：single-SID top-K + catalog indexing，不做伪查询。

训练命令：

```bash
base=outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803
split=$base/decoder_split_rqvae_e100_c256_d3_seed20260627
out=$base/single_sid_catalog_t5_base_rqvae_e100_split_seed20260627_cat3_q3000

HF_ENDPOINT=https://hf-mirror.com python3 scripts/skill_gr_pipeline.py train-single-sid-decoder \
  --rows "$split/train.jsonl" \
  --dev-rows "$split/dev.jsonl" \
  --skills-jsonl "$base/selected_skills.jsonl" \
  --semantic-ids "$base/rqvae_e100_c256_d3_h512_z128_seed20260627/semantic_ids.jsonl" \
  --output-dir "$out" \
  --model t5-base \
  --batch-size 16 \
  --catalog-epochs 3 \
  --max-steps 3000 \
  --lr 3e-4 \
  --weight-decay 0.0 \
  --logging-steps 50 \
  --eval-steps 200 \
  --seed 20260626 \
  --bf16 \
  --load-best-model-at-end
```

训练摘要：

```text
catalog rows: 3539
catalog steps: 666
catalog last train loss: 2.544733
query train rows after single-SID expansion: 4500
query dev rows after single-SID expansion: 250
query steps: 3000
query best dev loss: 2.137574
query last train loss: 0.898004
load_best_model_at_end: true
```

曲线解释：

- query dev loss 在 step 1000 达到当前最好值约 `2.1376`。
- 之后 train loss 继续下降，但 dev loss 回升到 step 3000 的 `2.4101`，说明 query fine-tune 过久已明显过拟合。
- 当前 `final/` 保存的是 best dev loss 权重，不是最后一步权重。

eval 命令：

```bash
python3 scripts/skill_gr_pipeline.py eval-single-sid-decoder \
  --rows "$split/dev.jsonl" \
  --model-dir "$out/final" \
  --semantic-ids "$base/rqvae_e100_c256_d3_h512_z128_seed20260627/semantic_ids.jsonl" \
  --output-dir "$out/eval_dev_beam50_ranked" \
  --split-name dev \
  --num-beams 50 \
  --k-values 1,10,50 \
  --log-every 10
```

seen/unseen 诊断（train 中是否出现过 gold skill assignment）：

```text
dev gold assignments: 250
seen in train:   115 / 250 = 46.0%
unseen in train: 135 / 250 = 54.0%

@50 Recall:
BM25:                   seen 0.904, unseen 0.852
structured multi-SID:   seen 0.365, unseen 0.156
single-SID + catalog:   seen 0.643, unseen 0.237
```

结论：B 方案显著优于旧 multi-SID set generation，尤其 `Recall@50` 从 `0.252` 提到 `0.424`、`FC@50` 从 `0.13` 提到 `0.25`。但它仍远低于 BM25，尤其 unseen skills 仍只有 `0.237` Recall@50；这说明 catalog indexing 和 single-SID formulation 修正了一个重要建模问题，但 RQ-VAE SID locality / decoder 对未见 SID 的泛化仍是主要瓶颈。下一步优先做 SID locality 和 per-level teacher-forcing 诊断，而不是继续盲调 query fine-tune 超参。

2026-06-27 补充消融与 test split 结果：

注意这里的 `multi-SID+catalog` 才是正确的 “only catalog” 消融：保留原本 multi-SID set generation 训练目标，只在前面加 `skill document -> single SID` catalog indexing pretrain。不要把之前误跑的 catalog zero-shot 当作结果。

新增输出：

```text
single-SID query-only:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/single_sid_query_only_t5_base_rqvae_e100_split_seed20260627_q3000

multi-SID + catalog:
outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803/structured_catalog_t5_base_rqvae_e100_split_seed20260627_cat3_steps2000
```

multi-SID + catalog 训练命令：

```bash
base=outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803
split=$base/decoder_split_rqvae_e100_c256_d3_seed20260627
out=$base/structured_catalog_t5_base_rqvae_e100_split_seed20260627_cat3_steps2000

HF_ENDPOINT=https://hf-mirror.com python3 scripts/skill_gr_pipeline.py train-structured-decoder \
  --rows "$split/train.jsonl" \
  --dev-rows "$split/dev.jsonl" \
  --skills-jsonl "$base/selected_skills.jsonl" \
  --semantic-ids "$base/rqvae_e100_c256_d3_h512_z128_seed20260627/semantic_ids.jsonl" \
  --output-dir "$out" \
  --model t5-base \
  --codebook-size 256 \
  --batch-size 8 \
  --catalog-epochs 3 \
  --max-steps 2000 \
  --lr 3e-4 \
  --weight-decay 0.0 \
  --logging-steps 50 \
  --eval-steps 200 \
  --seed 20260626 \
  --bf16 \
  --load-best-model-at-end
```

multi-SID + catalog 训练摘要：

```text
catalog steps: 1329
catalog last train loss: 2.168904
multi-SID query best dev loss: 2.169928 at around step 1000
multi-SID query last dev loss: 2.314963
```

dev 消融摘要：

```text
model                 Hit@50  Recall@50  FC@50
multi-SID             0.48    0.252      0.13
multi-SID + catalog   0.54    0.308      0.22
single-SID            0.61    0.436      0.25
single-SID + catalog  0.63    0.424      0.25
```

test overall:

```text
model                 Hit@1  Recall@1  FC@1   Hit@10  Recall@10  FC@10  Hit@50  Recall@50  FC@50
BM25                  0.63   0.252     0.18   0.93    0.640      0.37   0.96    0.840      0.68
T5 dense              0.04   0.016     0.00   0.11    0.044      0.02   0.30    0.148      0.07
multi-SID             0.02   0.008     0.00   0.26    0.116      0.04   0.43    0.208      0.13
multi-SID + catalog   0.07   0.028     0.01   0.44    0.224      0.07   0.57    0.312      0.20
single-SID            0.07   0.028     0.00   0.29    0.140      0.02   0.65    0.448      0.21
single-SID + catalog  0.09   0.036     0.00   0.32    0.164      0.02   0.70    0.444      0.24
```

test `@50` 按 gold skill 数量分桶：

```text
model                 gold_k  Hit@50  Recall@50  FC@50
BM25                  1       0.84    0.840      0.84
BM25                  2       1.00    0.920      0.84
BM25                  3       1.00    0.773      0.48
BM25                  4       1.00    0.850      0.56
T5 dense              1       0.24    0.240      0.24
T5 dense              2       0.24    0.140      0.04
T5 dense              3       0.40    0.187      0.00
T5 dense              4       0.32    0.100      0.00
multi-SID             1       0.32    0.320      0.32
multi-SID             2       0.60    0.380      0.16
multi-SID             3       0.48    0.213      0.04
multi-SID             4       0.32    0.090      0.00
multi-SID + catalog   1       0.52    0.520      0.52
multi-SID + catalog   2       0.68    0.440      0.20
multi-SID + catalog   3       0.72    0.400      0.08
multi-SID + catalog   4       0.36    0.130      0.00
single-SID            1       0.28    0.280      0.28
single-SID            2       0.68    0.500      0.32
single-SID            3       0.84    0.533      0.20
single-SID            4       0.80    0.400      0.04
single-SID + catalog  1       0.36    0.360      0.36
single-SID + catalog  2       0.72    0.520      0.32
single-SID + catalog  3       0.88    0.520      0.24
single-SID + catalog  4       0.84    0.370      0.04
```

结论：

- `single-SID` formulation 是主要收益来源。test `Recall@50` 从 old `multi-SID` 的 `0.208` 提升到 `0.448`。
- catalog indexing 对 `multi-SID` 明显有效：test `Recall@50` 从 `0.208` 到 `0.312`，`FC@50` 从 `0.13` 到 `0.20`。
- catalog indexing 对 `single-SID` 的整体 @50 recall 没有带来提升：`0.448` vs `0.444`，但略提高 `Hit@1/Hit@10/FC@50`。
- BM25 仍显著强于所有生成式模型：test `Recall@50=0.840, FC@50=0.68`，说明当前主要瓶颈仍在 SID 表示/decoder 泛化，而不是数据完全无效。

2026-06-26 已跑通的远端 smoke 输出使用的是 SkillRouter 微调 embedding，作为历史 checkpoint 保留：

```text
outputs/train/skill_gr_rqvae_t5_smoke_20260626_223545
```

结果：

```text
prepare: rows=2000, unique_selected_skills=3539
encode-skills: model=pipizhao/SkillRouter-Embedding-0.6B, batch_size=4, max_length=2048
train-rqvae smoke: epochs=2, codebook_size=256, depth=3, code_width=3, collision_groups=411
build-decoder-data: rows=2000, special_tokens=101
train-decoder smoke: model=t5-base, max_steps=5, train_rows=1800, dev_rows=200
```

远端 transformers 是 `5.3.0`，`Seq2SeqTrainer` 使用 `processing_class` 参数；脚本里已经做了兼容，旧版 transformers 仍会走 `tokenizer` 参数。

后续正式试验应使用 `outputs/train/skill_gr_qwen06_rqvae_t5_smoke_20260627_000803` 作为 base Qwen 0.6B 的已验证链路参考；不要复用上面微调 embedding 的 RQ-VAE 输出。

### 14.6 输出文件

```text
selected_skills.jsonl
decoder_rows.base.jsonl
skill_embeddings.pt
embedding_manifest.json
rqvae/rqvae.pt
rqvae/semantic_ids.jsonl
rqvae/rqvae_manifest.json
decoder_data/decoder_rows.jsonl
decoder_data/decoder_special_tokens.json
t5_smoke/final/
```

后续 constrained decoding trie 应基于 `rqvae/semantic_ids.jsonl` 里的实际 token 序列构建；不要用全量 78,361 skills 建树。

## 15. 常见坑

1. 不要把当前版本误认为 raw SkillRouter body-preview 版本。

当前版本有 `Skill body:` 字段名，但内容是 metadata-as-body-preview。

2. 不要直接改回 raw body-prefix hybrid。

实验结果显示它更贵且质量没有更好。

3. 不要把 `usage` 放进训练数据主文件。

`usage` 应该放 sidecar，例如 `pilot_400_usage.jsonl`。

4. 不要把 `profiles`、`outputs` 提交到 git。

当前 `.gitignore` 已忽略：

```text
outputs
outputs/
profiles
profiles/
```

5. 不要把 old `/root/workplace/SkillSearch` 当代码入口。

远端代码入口是：

```text
/root/workplace/SkillSearch-data-synthesis
```

旧 `/root/workplace/SkillSearch` 现在只是数据/profile 容器。

6. full dataset 成本不能用 `258 panels` 估算上限。

之前的 `258 panels` 只是代表性 sampled panels，不是全量可生成 workflow 上限。若要估算大规模成本，应按目标 multiskill query 数量计算。

7. 不要把 Drive OAuth token 长期留在远端。

本次上传完成后已删除 `/root/.config/rclone/rclone.conf`。如果后续需要再次上传，重新授权，上传完成后再删除 token 配置。

## 16. 扩展方向

当前系统适合稳定复现 400 条 replay 和做小规模扩展。后续如果要扩大到 10k/50k/100k，多做这些改进：

```text
1. 更系统地扩展 candidate panel sampling
2. 对同一 panel 多 seed / 多组合采样
3. selected_id_not_in_panel retry
4. medium_confidence targeted rewrite
5. verifier/audit 只用于 hard cases，不默认放进生成 prompt
6. 数据导出脚本化，固定 clean dataset schema
```

## 17. 当前验收状态

本地和远端都已验证：

```bash
python3 -m py_compile scripts/generate_single_skill_probe.py scripts/generate_multiskill_probe.py scripts/replay_pilot_400_skillrouter_prompt.py scripts/build_balanced_topup_inputs.py
python3 -m unittest discover -s tests -p 'test_skillrouter_prompt_alignment.py'
```

测试结果：

```text
Ran 10 tests
OK
```

新增 GR pipeline 本地验证：

```bash
python3 -m py_compile scripts/skill_gr_pipeline.py
python3 -m unittest discover -s tests -p 'test_skill_gr_pipeline.py'
python3 -m unittest discover -s tests -p 'test*.py'
```

测试结果：

```text
Ran 16 tests
OK
```

prompt 冻结基线 commit：

```text
58f284f chore: freeze skillrouter metadata replay pipeline
```

接手时检查实际最新提交：

```bash
git log --oneline -1
```
