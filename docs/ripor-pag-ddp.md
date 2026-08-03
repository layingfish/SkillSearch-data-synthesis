# 固定 LETTER Semantic ID 的 RIPOR/PAG 接入与单节点 8 卡训练

项目主线固定 `Qwen + LightGCN + LETTER` 已生成的 Semantic ID。RIPOR 只改变 sequential
decoder 的 prefix ranking 训练，PAG 通过独立 Set ID 分支提供词法补强；两者都不得重训
LETTER、替换 embedding 模型或重新生成 `semantic_ids.jsonl`。

三个最终设置由 `--method-setting` fail-closed 校验：

- `ripor`：LETTER SID + prefix-oriented MarginMSE，CE=0，PAG=0。
- `pag`：LETTER SID 的共享 CE baseline + PAG set loss/planning，RIPOR=0。
- `ripor_pag`：LETTER SID + prefix-oriented MarginMSE + PAG set loss/planning，CE=0。

三个设置都要求 `--query-target-mode single_sid`、`--letter-tokenizer-manifest`、`--reset-optimizer`、`--catalog-epochs 0`、
`--eval-steps 0 --no-load-best-model-at-end`。训练和评测的完整顺序在
`scripts/run_letter_ripor_pag_settings.sh`；输入文件 hash 会写入 `setting_contract.json`。

## RIPOR

`train-single-sid-decoder` 的 `--ripor-loss-weight > 0` 使用 RIPOR 的 prefix-oriented MarginMSE：

- 对固定 LETTER SemID teacher-force，计算累计 raw conditional-logit 分数 `S_prefix_i`；dedup suffix 不进入排序。
- 对同一 query 的全部 ground-truth skill 作为正例，负例排除全部 GT skill；正负路径在首次分叉前不产生 prefix loss。
- 目标是 `(student_margin_i - alpha_i * teacher_margin)^2`，`alpha_i=(1-beta^i)/(1-beta^L)`，默认 `beta=0.5`（官方 decay=2 的等价写法）、`alpha_L=1`；所有已达到的 prefix objective 同时保留。
- 项目设置要求 `--ripor-hard-negatives` 提供冻结 teacher 的正负分数。缺分数的样本会直接报错。

RIPOR 的 M0/RQ relevance-based ID initialization 与固定 LETTER 主线互斥，仅保留在
`--method-setting custom` 的论文参考路径中；不能把它混入三个项目设置：

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

PAG 的 Set ID 独立于 LETTER Semantic ID。Msp 用 T5-SPLADE
（`log(1+ReLU(logits))`、全局 FLOPS 正则、teacher MarginMSE）生成固定 m 个 lexical
Set ID，Mset 从 Msp 初始化；`pag` 与 `ripor_pag` 都把 Mset 与同一个共享 CE LETTER decoder
合并，方法差异只发生在最终阶段的 sequential loss。合并时共享 T5 参数取平均，但
`sid_embedding`、全部 code heads、dedup/control heads 必须与 Mseq bitwise 相同。

完整项目流程统一由一个启动器完成，避免三组手工参数漂移：

```bash
bash scripts/run_letter_ripor_pag_settings.sh
```

启动器先训练共享 Msp/Mset，再从同一个 CE+LETTER decoder 产生 `ripor_only`、`pag_only`、
`ripor_pag` 三个最终 checkpoint。三组都使用相同的 single-SID query 展开、相同 teacher panel、
相同 final-stage step budget；所有 optimizer 在方法阶段重新初始化。

解码时，先从 lexical set branch 取固定的 top-N 候选域 `D`（默认实验配置为 1000），再对每个 sequential prefix 使用
`S'(prefix)=S_seq(prefix)+weight*max_{skill descendant} S_set(query,skill)`，planner 分数按当前 prefix 重新计算，绝不沿 token 重复累加。`--pag-set-top-n` 非零时会剪枝到 set branch 的 top-N 候选，并在 `eval_manifest.json` 记录 gold candidate recall。`--pag-set-weight 0` 是同一 checkpoint 的 sequential baseline；不再接受外部 dense/BM25 分数作为 PAG 复现路径。

评测命令必须传 `--expected-method-setting`，防止 checkpoint 与方法标签错配。

## 8 卡

手写训练循环通过 `torchrun` 自动进入 DDP；`--batch-size` 是每卡 batch，manifest 记录全局 batch。structured decoder 的 CE、RIPOR pair 和 PAG set loss 都使用全局 sum/count 归一化；teacher 评分和 PAG Set ID 构建也按 rank 分片；日志、W&B、checkpoint 只由 rank 0 写入。检索评测使用无 padding 分片并按原始行号合并。

```bash
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-single-sid-decoder ... --batch-size 16 --bf16
torchrun --standalone --nproc_per_node=8 scripts/skill_gr_pipeline.py train-structured-decoder ... --batch-size 8 --bf16
torchrun --standalone --nproc_per_node=8 scripts/skillrouter_embedding_train.py train ... --training-mode standard --batch-size 8 --bf16
```

`train-rqvae`、LETTER 码本和 LightGCN 不做简单 DDP 包装：它们的码本/图传播需要专门的全局协议，复制到 8 卡会改变语义而不是单纯加速。
