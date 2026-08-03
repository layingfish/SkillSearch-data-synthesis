# RIPOR/PAG 复现接入与单节点 8 卡训练

这份说明只把论文核心机制称为复现；旧的 dense/BM25 外挂 planner 已被移出主路径，不能作为 PAG 结果。

## RIPOR

`train-single-sid-decoder` 的 `--ripor-loss-weight > 0` 使用 RIPOR 的 prefix-oriented MarginMSE：

- 先用 M0 relevance encoder 训练 query/document MarginMSE，再用 RQ centroids 初始化每个位置的 decoder input/output codebook；对每个候选 SemID teacher-force，计算官方的累计 raw conditional-logit 分数 `S_prefix_i`；dedup suffix 不进入排序。
- 对同一 query 的全部 ground-truth skill 作为正例，负例排除全部 GT skill；正负路径在首次分叉前不产生 prefix loss。
- 目标是 `(student_margin_i - alpha_i * teacher_margin)^2`，`alpha_i=(1-beta^i)/(1-beta^L)`，默认 `beta=0.5`（官方 decay=2 的等价写法）、`alpha_L=1`；所有已达到的 prefix objective 同时保留。
- 严格复现要求 `--ripor-hard-negatives` 提供冻结 teacher 的正负分数。缺分数的样本会直接报错；`--ripor-allow-weak-teacher` 只能用于单独标记的弱 teacher ablation。

RIPOR 的 relevance-based ID initialization 是独立阶段，不把现有 document-only D4/LETTER ID 叫作 RIPOR ID：

```bash
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-ripor-relevance-encoder \
  --rows train.jsonl --skills-jsonl skills.jsonl --hard-negatives teacher_ranked.jsonl \
  --output-dir ripor_relevance --model t5-base --batch-size 16 --bf16

uv run python scripts/skill_gr_pipeline.py train-rqvae \
  --embeddings ripor_relevance/skill_embeddings.pt --output-dir rqvae \
  --codebook-size 256 --depth 3 --ripor-identity
```

输出的 `ripor_relevance/skill_embeddings.pt` 才能作为 RQ/LETTER 量化的输入；`train-rqvae --ripor-identity` 直接在 M0/T5 hidden space 做 residual quantization，避免 learned bottleneck 改变 centroid 坐标系。它另外输出 `codebook_embeddings.pt`，训练 decoder 时必须传入 M0 与该 centroid 文件，才是 relevance-based ID initialization，而不是只换一份 SemID。

## PAG

PAG 按论文的阶段顺序执行：Msp 用官方 T5-SPLADE（`log(1+ReLU(logits))`、全局 FLOPS 正则、teacher MarginMSE）生成固定 m 个 lexical set ID；Mset 从 Msp 初始化，Mseq 从 RIPOR M0/centroid 初始化，最后逐参数合并两个有 provenance 的 checkpoint，并用 `Lset + Lseq` 联合微调。联合 decoder 的 sequential trajectory 采用 `[query tokens; SID prefix]`，与 set 分支共享 query-token context。

```bash
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-pag-sparse-encoder \
  --skills-jsonl skills.jsonl --rows train.jsonl --hard-negatives teacher_ranked.jsonl \
  --output-dir pag_sparse --model t5-base --batch-size 32 --bf16

python3 scripts/skill_gr_pipeline.py build-pag-set-ids \
  --skills-jsonl skills.jsonl --sparse-checkpoint pag_sparse --max-tokens 16 --output pag_set_ids.json

torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-structured-decoder \
  --rows multi_train.jsonl --dev-rows multi_dev.jsonl --semantic-ids semantic_ids.jsonl \
  --skills-jsonl skills.jsonl --pag-set-ids pag_set_ids.json \
  --pag-set-hard-negatives pag_teacher_ranked.jsonl --pag-set-loss-weight 1.0 \
  --pag-set-stage-only --pag-sparse-checkpoint pag_sparse --sequential-ce-weight 0 \
  --output-dir pag_mset --model t5-base --bf16

torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-structured-decoder \
  --rows multi_train.jsonl --dev-rows multi_dev.jsonl --semantic-ids semantic_ids.jsonl \
  --ripor-hard-negatives ripor_teacher_ranked.jsonl --ripor-loss-weight 1.0 \
  --ripor-relevance-checkpoint ripor_relevance/relevance_encoder.pt \
  --ripor-codebook-embeddings rqvae/codebook_embeddings.pt --pag-stage-role pag_mseq \
  --sequential-ce-weight 0 \
  --output-dir pag_mseq --model t5-base --bf16

torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-structured-decoder \
  --rows multi_train.jsonl --dev-rows multi_dev.jsonl --semantic-ids semantic_ids.jsonl \
  --skills-jsonl skills.jsonl --pag-set-ids pag_set_ids.json \
  --pag-set-hard-negatives pag_teacher_ranked.jsonl \
  --pag-set-loss-weight 1.0 --ripor-loss-weight 1.0 \
  --ripor-hard-negatives ripor_teacher_ranked.jsonl \
  --pag-set-init-model-dir pag_mset \
  --pag-seq-init-model-dir pag_mseq --pag-stage-role pag_joint \
  --sequential-ce-weight 0 --pag-set-top-n 100 \
  --output-dir pag_joint --model t5-base --bf16
```

严格 PAG 复现要求先得到 Msp→Mset 与 Mseq 两份带角色标记的 checkpoint，再按官方流程逐参数平均初始化联合模型；任意普通 checkpoint 不能冒充 Mset/Mseq。缺少任一阶段、teacher numeric score、M0 或 RQ centroid 时命令会直接失败；`--pag-set-allow-weak-teacher` 只允许显式标记的弱监督消融。严格联合目标将 CE 权重设为 0，CE 只能通过显式非零参数作为消融。

解码时，先从 lexical set branch 取固定的 top-N 候选域 `D`（示例为 100），再对每个 sequential prefix 使用
`S'(prefix)=S_seq(prefix)+weight*max_{skill descendant} S_set(query,skill)`，planner 分数按当前 prefix 重新计算，绝不沿 token 重复累加。`--pag-set-top-n` 非零时会剪枝到 set branch 的 top-N 候选，并在 `eval_manifest.json` 记录 gold candidate recall。`--pag-set-weight 0` 是同一 checkpoint 的 sequential baseline；不再接受外部 dense/BM25 分数作为 PAG 复现路径。

```bash
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py eval-structured-decoder \
  --rows multi_dev.jsonl --model-dir pag_joint/final --semantic-ids semantic_ids.jsonl \
  --pag-set-ids pag_set_ids.json --pag-set-top-n 100 --pag-set-weight 1.0 --output-dir pag_eval
```

## 8 卡

手写训练循环通过 `torchrun` 自动进入 DDP；`--batch-size` 是每卡 batch，manifest 记录全局 batch。structured decoder 的 CE、RIPOR pair 和 PAG set loss 都使用全局 sum/count 归一化；日志、W&B、checkpoint 只由 rank 0 写入。检索评测使用无 padding 分片并按原始行号合并。

```bash
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-single-sid-decoder ... --batch-size 16 --bf16
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-structured-decoder ... --batch-size 8 --bf16
torchrun --standalone --nproc_per_node=8 scripts/skillrouter_embedding_train.py train ... --training-mode standard --batch-size 8 --bf16
```

`train-rqvae`、LETTER 码本和 LightGCN 不做简单 DDP 包装：它们的码本/图传播需要专门的全局协议，复制到 8 卡会改变语义而不是单纯加速。
