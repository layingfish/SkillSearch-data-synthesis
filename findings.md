# RIPOR / PAG 接入：事实、约束与决策依据

## 用户要求

- 固定现有 Qwen + LightGCN + LETTER 方法链，不更换 embedding，不破坏协同信号。
- RIPOR 和 PAG 要实现论文核心方法，不能用自定义近似替代后仍宣称复现。
- 三个设置：RIPOR-only、PAG-only、RIPOR+PAG。
- 训练和可并行评测尽可能使用单节点 8×A800 DDP。
- 开发优先，只运行会显著影响正确性或整体进度的最小测试。
- 先把计划固化到文件，防止执行过程漂移。

## 论文与官方实现事实

### RIPOR

- 核心 decoder 优化是 position-specific code score 上的 prefix-oriented MarginMSE。
- 原论文还包含 relevance model 和 residual quantization 构造 identifier；本项目固定 LETTER SID，因此本轮只转移 decoder ranking 核心，并明确这一差异。

### PAG

- 完整论文已读取 12 页主文、公式、算法、表格和参考文献。
- 官方代码仓库为 `https://github.com/HansiZeng/PAG`；项目镜像中的参考版本固定为 commit `49802454fb157fc659c1c6c42bb2d940a9f6bc1b`。
- Set ID 是固定大小的 lexical token set，与 sequential identifier 并存。
- `Msp` 用 T5-SPLADE + MarginMSE + FLOPS 构造 lexical representation；top-m token 成为 Set ID。
- 官方 `SpladeTrainer` 对 query/doc FLOPS 分别使用 `0.01/0.008`，并通过
  `RegWeightScheduler` 在前 `max_steps/3` 以平方曲线从 0 增长到目标权重；满权重不能从第 1 步施加。
- `Mset` 和 `Mseq` 分别训练，最终模型用两者参数平均初始化并联合优化 `Lset + Lseq`。
- 联合模型把 query token 同时送入 decoder，官方 collator 也将 query input IDs 放入 decoder prefix。
- planning-ahead beam score 使用当前 prefix 可到达文档的最大 simultaneous score 作为绝对先验，不是每个 token 累积一次 planner score。
- 论文默认 `m=64`、top-N=1000、beam=100；项目主评测 beam 是否沿用 100，在提交前按统一协议固定，不能按结果临时选择。

## 当前项目资产

- 固定 LETTER 目录：`/models_rw/SkillSearch-data-synthesis/assets-models/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/`
- LETTER logical SID hash：`da2fff997b1979f7d51996c5ddbe4593a18d23bdfd546c61dd61fb783567349c`。
- 基础 single-SID decoder：同一正式训练根目录下的 `single_sid_d4_lightgcn_d32_letterofficial_task5_isolatedmode3_core75_gt2_t5base_cat3_querye5_b16/final`。
- 正式训练数据：`/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/isolated_mode3_gpt4o_20260714/decoder_rows.base_plus_task5_gt2.plus_isolated_mode3.jsonl`。
- canonical skills：`/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/canonical_skills.jsonl`。
- dev：`/dataset_rw/SkillSearch-data-synthesis/assets-data/outputs/eval/skillrouter_train_dev_test_split_20260703/dev.jsonl`。
- 冻结 SkillRouter teacher：`/models_rw/SkillSearch-data-synthesis/assets-models/outputs/train/skillrouter_repro_task5_mode3_single82621_20260716/sr_emb_qwen3_06b_single_only_1epoch_10neg`。
- 运行镜像：`chenyufei03/skillsearch-gr:cu128-torch28-v1`。

## 技术决策

| 决策 | 理由 |
|---|---|
| 本轮以 single-SID GR 为唯一建模方式 | 隔离 RIPOR/PAG 贡献，避免同时混入 multi-SID serialization 的建模差异 |
| RIPOR 不替换 LETTER identifier | 保留 LightGCN 协同信息和现有 tokenizer 控制变量 |
| PAG Set ID 独立保存 | lexical reinforcement 与 LETTER semantic/collaborative SID 可同时存在 |
| `ripor_pag` 使用共享 RIPOR Mseq | 联合组必须真正包含 RIPOR 完整 sequential 组件，并符合 PAG 三阶段联合初始化 |
| 当前展开 target 是唯一 ranking positive，siblings 只排除负例 | 避免 false negative，也避免 `K²` multi-query 权重 |
| 三组 final checkpoint 固定步数取 final/last | legacy dev CE loss 不适合选择 ranking/set objective checkpoint |
| 只允许 PAG 自身 Set score进入 planning | 外部 dense/BM25 分数融合不是 PAG 核心方法 |

## 已发现的实现风险

- 当前未提交实现中，联合组曾被改为直接把 `Mset` 与 CE base 合并；这不符合本计划，恢复开发后需改回共享 RIPOR Mseq。
- 当前 candidate builder 曾让展开后的每个 example 使用全部 source positives，可能导致 multi-GT query 的 `K²` ranking 权重；必须按计划修正。
- PAG Set ID 构建原先是 CPU 单进程，已经开始改为 8 卡分片，但仍需语义审查。
- 首次正式 PAG Msp 生成的 14,551 份 top-64 Set ID 只有 1 个 unordered unique set；64 个
  token 的 document frequency 均为 14,551，导致 Official top-1000 candidate recall=0。当前实现
  确实遗漏官方 FLOPS quadratic scheduler，但 100-step、FLOPS=0 的诊断仍然 collapse，因此该偏差
  不是充分根因；还需用锁定的 4700-step 训练和表示审计判断是否为训练尺度/优化动态问题。全部
  相关 PAG checkpoint/eval 已隔离，不能进入结果。
- 8-rank 表示审计 `tide-chenyufei03-10259-qxqxc` 在相同 512 个 skill 上发现：pinned vanilla
  T5-base 每行平均只有 `0.021484` 个正 raw-logit coordinate（最多 1 个），但仍有 9 个 unordered
  top-64 set；1000-step Msp 的 raw-logit 最大值均为负（均值 `-9.4857`，最大 `-4.09375`），
  lexical positive coordinate 为 0，top-64 只剩 1 个 set。这证明短训练发生的是官方 ReLU lexical
  表示进入全零区，而不是 Set ID 写盘/合并错误。
- 官方 `t5_splade_train.sh` 使用 8 卡、每卡 batch 32、`max_steps=200000`、每 25000 步保存；
  当前 4700-step 预算远短于官方 Msp 训练尺度。若 4700-step 正式复核仍 collapse，计划只允许把
  Msp 对齐到该官方尺度，不允许换激活、换 backbone 或改 top-m 规则。
- teacher query 已分片；catalog encoding 也应分片后按原始 skill index 合并。
- 本地 `uv` 环境没有 torch，tensor invariant 要在已验证 GPU 镜像中运行；本地只做无 torch 的契约与语法检查。

## 外部资料

- PAG paper: `https://arxiv.org/abs/2404.14600`
- PAG official code: `https://github.com/HansiZeng/PAG`
- 项目接入说明：`docs/ripor-pag-ddp.md`
- 实验记录：`docs/experiments/2026-08-04-letter-ripor-pag-settings.md`
