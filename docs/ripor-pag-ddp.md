# RIPOR/PAG 与单节点 8 卡训练

## 方法接入

当前实现把两篇工作中适合 D4 Structured SID 的部分移植到现有管线，而不是把旧仓库的模型直接替换进来。

- RIPOR 位于 `train-single-sid-decoder`。模型对每个候选 SemID 做 teacher-forcing，计算每个语义 code level 的累计 log probability，并仅在正负路径首次分叉之后，将学生 prefix margin 对齐到冻结 teacher relevance margin。`source_selected_skill_ids` 中同一个 query 的全部技能都会被视为正例，避免 multi-skill query 在 single-SID 展开后互相构成 false negative。dedup suffix 只用于唯一化，不参与 relevance ranking。
- `--ripor-hard-negatives` 接收当前 GR 或 dense teacher 的 `predictions.jsonl`，从 `ranked_skill_ids` 与 `ranked_scores`/`ranked_skill_scores` 读取可刷新 hard-negative pool 和 teacher 分数；没有分数时显式使用 reciprocal-rank，未覆盖的槽位用正负 anchor 分数和确定性随机候选补齐。`--ripor-loss-weight 0` 保持原来的 CE-only 路径。
- PAG 适配为受限 Trie 内的 planner look-ahead：`eval-structured-decoder` 在每个 allowed child 上聚合冻结 dense/BM25 排名的后代 skill 分数，再与生成模型的 beam score 合并。planner 不读取 gold、不硬过滤 Trie 候选；`--pag-planner-weight 0` 逐 token 保持原 beam 行为。当前 multi-SID 仍使用原有词典序 set serialization，因此没有把 RIPOR 的完整序列排序损失误用于 multi。

示例：

```bash
python3 scripts/skill_gr_pipeline.py train-single-sid-decoder --rows train.jsonl --dev-rows dev.jsonl --semantic-ids semantic_ids.jsonl --skills-jsonl skills.jsonl --output-dir single-ripor --model t5-base --bf16 --ripor-loss-weight 0.25 --ripor-max-negatives 16 --ripor-hard-negatives dense-or-previous-gr/predictions.jsonl

python3 scripts/skill_gr_pipeline.py eval-single-sid-decoder --rows dev.jsonl --model-dir single-ripor/final --semantic-ids semantic_ids.jsonl --output-dir single-pag --pag-planner-predictions dense/predictions.jsonl --pag-planner-weight 0.15 --pag-planner-score-mode reciprocal_rank
```

## 8 卡启动

手写训练循环通过 `torchrun` 自动进入 DDP；普通单进程命令无需额外参数。`--batch-size` 表示每卡 batch，manifest 会同时记录 `global_batch_size = batch_size × WORLD_SIZE`。

```bash
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-single-sid-decoder ... --batch-size 16 --bf16
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-structured-decoder ... --batch-size 8 --bf16
torchrun --standalone --nproc_per_node=8 scripts/skillrouter_embedding_train.py train ... --training-mode standard --batch-size 8 --bf16
```

Structured decoder 使用全局 token-count 归一化，避免不同卡上 multi-SID target 长度不同造成 local-mean 偏差；日志、W&B、manifest、checkpoint 只由 rank 0 写入。SkillRouter standard 模式使用跨 rank differentiable all-gather，保留全局 in-batch negatives。`gradcache` 依赖手工重算梯度，仍保持单进程语义。

`train-decoder` 使用 Hugging Face `Seq2SeqTrainer`，以 `torchrun` 启动即可由 Trainer 原生使用 DDP，保存动作已保护为 world-process-zero。

PAG 的 `raw` planner 分数会按 query 平移到非负区间，使没有 teacher 记录的候选使用零基线，不会因为原始分数为负而反向获益。

## 有意不做普通 DDP 的阶段

`train-rqvae`、LETTER `train_letter_rqvae_official.py` 和 LightGCN 不做简单 DDP 包装：前两者含 k-means/Sinkhorn/EMA 码本与全局 collision assignment，跨卡复制会产生不一致 SemID；LightGCN 每个 BPR step 都重算完整图传播，复制 full graph 到 8 卡没有正向加速。它们应先保持单进程，后续若要并行需要专门的全局码本或图切分协议。

`encode-skills` 和 structured decoder eval 支持 `torchrun` 的无 padding 分片；rank 0 按原始行号合并输出，避免评测重复样本。
