# 2026-07-14 LETTER base single-SID 训练取证包

更新时间：2026-08-09（Asia/Shanghai）

本目录用于复现以下 run：

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/
single_sid_d4_lightgcn_d32_letterofficial_task5_core75_gt2_t5base_cat3_querye5_b16/
```

结论先行：原始训练使用 dirty worktree 中的 untracked `scripts/skill_gr_pipeline.py`。训练时 Git HEAD 为 `7a8f3340486beac94ff4e6f3b4c574362573c423`，不能用 `15f6c51131f3b6f3e8580aa4564b17cbe13a9790` 代表原始源码。所幸训练后保存了脚本 SHA256，且事后只有两次完整 patch 记录。反向重建得到的文件 SHA256 与原记录逐字节相同：

```text
7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76  skill_gr_pipeline.original_20260714.py
```

2026-08-08 在原 RTX PRO 6000 环境运行 50-step probe，得到：

```text
initial model fingerprint = c219d07a2ac425cffd677cefc0792a81f47aa3ad9d0fa76a0bfa37cdf9f5ebde
catalog first-batch hash  = 721a0d93aaef2f9d0104adae8501fcfabcefcd6bd9f23c1e4a1aae701710d443
catalog step 1 loss       = 4.528710842132568
catalog step 50 loss      = 4.203788757324219
```

上述 step 50 与 2026-07-14 归档 loss 逐 bit 相同。这证明原始代码、数据、T5 cache、RNG 顺序和原 RTX 环境已被重新闭环验证。

## 1. 本目录文件

| 文件 | 作用 |
|---|---|
| `skill_gr_pipeline.original_20260714.py` | 逐字节重建并由历史 SHA256 验证的原始训练脚本 |
| `run_repro_from_archive.sh` | 使用归档脚本重跑完整训练的安全启动脚本；输出目录存在时退出 |
| `catalog_50_probe.py` | 输出初始化模型 hash、RNG hash、首批 row IDs、step 1/50 loss 的 probe |
| `catalog_50_probe_reference.jsonl` | 原 RTX PRO 6000 probe 的参考输出及 `foreach=False` 对照 |
| `batch_order_reference.json` | catalog/query 首批 row IDs 与顺序 hash |
| `loss_reference.jsonl` | 原始 loss history 的关键行，保留完整浮点精度 |
| `requirements.nearest_20260716.txt` | 同一服务器在 2026-07-16 由 W&B 保存的最近环境快照；不冒充 2026-07-14 原始 `pip freeze` |
| `archive/train/single_sid_decoder_train_manifest.json` | 原 run 的完整训练 manifest |
| `archive/train/loss_history.jsonl` | 原 run 的完整 loss history |
| `archive/train/train.log` | 原 run 的完整 stdout/stderr 日志 |
| `archive/eval_core75_beam50/eval_manifest.json` | Official core75 最终评测的完整 manifest |
| `archive/eval_core75_beam50/eval_metrics.json` | 生成集合口径的原始评测指标 |
| `archive/eval_core75_beam50/predictions.jsonl` | Official core75 的 75 条原始 beam-order Top-50 排序 |
| `archive/eval_core75_beam50/ranked_metrics*.json` | Overall、Single、Multi 的原始 macro Recall@K |

## 2. 原始 Git 状态与源码

### 2.1 HEAD

训练时主仓库 HEAD：

```text
7a8f3340486beac94ff4e6f3b4c574362573c423  docs: add query generation handoff
```

`scripts/skill_gr_pipeline.py` 在该 commit 中不存在。训练时它是 untracked 文件。

训练结束后、后续 eval-batch patch 之前，最接近的 `git status --short` 快照记录于 `2026-07-14 21:19:35 CST`：

```text
 M HANDOFF.md
 M README.md
 M scripts/generate_single_skill_probe.py
 M scripts/replay_pilot_400_skillrouter_prompt.py
?? PLAN.md
?? docs/
?? scripts/build_balanced_topup_inputs.py
?? scripts/build_gt_multiskill_inputs.py
?? scripts/build_longtail_pseudo_queries.py
?? scripts/build_normal_skillrouter_inputs.py
?? scripts/build_skill_results_explorer.py
?? scripts/skill_gr_pipeline.py
?? scripts/skillrouter_embedding_train.py
?? scripts/train_letter_rqvae_official.py
?? scripts/train_lightgcn_cf_embeddings.py
?? tests/test_longtail_pseudo_queries.py
?? tests/test_normal_skillrouter_inputs.py
?? tests/test_skill_gr_pipeline.py
?? tests/test_skill_results_explorer.py
```

训练启动瞬间没有保存完整 `git status` 和 `git diff`。上面的 status 是现存最近证据。原训练脚本是 untracked 文件，因此 `git diff` 本来就不会包含它。tracked 文件的完整历史 diff 也没有归档，无法诚实补造。

### 2.2 原始脚本与当前脚本

```text
原始训练脚本 SHA256: 7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76
15f6c511 脚本 SHA256: 5d60624b704e3563f1893b021e15c2c6da3cdb28a18a1ac3018ca3884de04f67
5807397d 脚本 SHA256: 5d60624b704e3563f1893b021e15c2c6da3cdb28a18a1ac3018ca3884de04f67
```

两者只差训练完成后加入的两次 `eval_batch_size` patch：一次改 single-SID dev eval，一次改 multi-SID dev eval。single-SID patch 记录于 `2026-07-14 21:39 CST`，晚于该 run 在 `2026-07-14 06:36 CST` 完成。它没有改 catalog training、model init、shuffle、optimizer 或 query training batch。

因此，源码差异解释不了 catalog step 50 的分叉；归档原文件仍应作为完整复现输入。

## 3. 原始启动命令

第一次启动时间为 `2026-07-14 04:31:29 CST`，PID 为 `1531779`。命令没有设置 Hugging Face offline 环境变量，模型 metadata 请求失败，未进入训练。

成功启动时间为 `2026-07-14 04:35:06 CST`，PID 为 `1532068`。以下是成功 run 的原始命令：

```bash
cd /root/workplace/SkillSearch-data-synthesis
out=outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/single_sid_d4_lightgcn_d32_letterofficial_task5_core75_gt2_t5base_cat3_querye5_b16
mv "$out/train.log" "$out/startup_failure_online_hub.log"
nohup env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  python3 scripts/skill_gr_pipeline.py train-single-sid-decoder \
  --rows outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/rows_000.decoder_rows.base_plus_task5_gt2.jsonl \
  --dev-rows outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/rows_001.decoder_rows.base.jsonl \
  --semantic-ids outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/semantic_ids.jsonl \
  --skills-jsonl outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/canonical_skills.jsonl \
  --output-dir "$out" \
  --model t5-base \
  --catalog-epochs 3 \
  --epochs 5 \
  --batch-size 16 \
  --lr 3e-4 \
  --weight-decay 0 \
  --max-source-length 2048 \
  --logging-steps 50 \
  --eval-steps 200 \
  --seed 20260626 \
  --bf16 > "$out/train.log" 2>&1 &
echo $! > "$out/train.pid"
```

`run_repro_from_archive.sh` 保留相同参数，只把入口替换成本目录内已验证的原始脚本。

## 4. 原始数据、SemID 与 T5

### 4.1 私有仓库

```text
GitHub code:
  https://github.com/layingfish/SkillSearch-data-synthesis
  visibility: private
  baseline before adding the raw train/eval archive: 150a822ad91bf44b6c48468fafebf69b216bd5c7

ModelScope dataset:
  https://modelscope.ai/datasets/Yufie-Chen/SkillGR-Task5-Assets
  visibility: private
  frozen revision: 4198a928a08a022a0b923ffc1c00401d135642ba

ModelScope model:
  https://modelscope.ai/models/Yufie-Chen/SkillGR-Checkpoints
  visibility: private
  frozen revision: 8e5b73e59d617e1d9b5ee893efe1ef4c6da0ce6d
```

### 4.2 输入文件

所有相对路径均以 `/root/workplace/SkillSearch-data-synthesis` 为根目录。

| 输入 | 行数与展开条件 | 字节数 | SHA256 |
|---|---:|---:|---|
| `outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/rows_000.decoder_rows.base_plus_task5_gt2.jsonl` | 94,912 source rows；按每行排序后的 `selected_skill_ids` 展开为 124,175 single-SID rows | 263,376,702 | `ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5` |
| `outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/rows_001.decoder_rows.base.jsonl` | 1,000 source rows；展开为 2,501 dev rows | 1,569,173 | `a4d7b7c8223a79e1ac3157b2bc72121e64590383a52bb7bde1c540d51f2e7aed` |
| `outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs/canonical_skills.jsonl` | 14,551 catalog rows；训练前按 `skill_id` 排序 | 158,774,190 | `1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3` |
| `outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/semantic_ids.jsonl` | 14,551 learned four-code SIDs；codebook size=256、depth=4、dedup size=0 | 见 ModelScope LFS | `084cccd5033d0aee8fb7bfbf05d55508744376c50a3c9bef95803ee69b41fa61` |

checkpoint 中保存的 `semantic_ids_sha256=da2fff997b1979f7d51996c5ddbe4593a18d23bdfd546c61dd61fb783567349c` 是排序后逻辑映射的 fingerprint。它与文件字节 SHA256 的定义不同。

`rows_000.decoder_rows.base_plus_task5_gt2.jsonl` 的 94,912 条 source rows 由两部分组成：

| 组成 | source rows | 展开后的 single-positive rows | 条件 |
|---|---:|---:|---|
| 原 base 部分 | 19,487 | 48,750 | `source` 字段缺失；按每行去重、排序后的正例展开 |
| Task5 task-style pseudo query | 75,425 | 75,425 | `source=full_catalog_task_style_pseudo_query`；每行一个正例 |
| 合计 | 94,912 | 124,175 | 即实际传给 `--rows` 的完整文件 |

这里的“base 94,912 rows”已经包含 75,425 条 Task5 task-style pseudo query。它没有混入后续 5,166 条 mode3 augment，也没有混入其他 replay 文件。若加入 mode3，输入应为 100,078 source rows、展开后 129,341 rows；这与原 manifest 的 `source_train_rows=94912`、`query_train_rows=124175` 不符。

multi-skill source row 的处理规则固定：先对 `selected_skill_ids` 保序去重，再按 skill ID 排序，随后复制成每个正例一条 single-SID row。训练没有跨 source row 去重、sample、权重调整或每 epoch 重新展开；5 个 query epoch 只对同一份 124,175-row list 原地 shuffle。

### 4.3 query 输入字段与模板

94,912 条训练 source rows 均含 `query` 和 `query_text`，均不含 `decoder_input`；`query` 也没有预置 instruction。实际输入始终使用 `query`，先截取前 1,500 个字符，再拼接：

```text
Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task
Query:{query[:1500]}
```

随后 tokenizer 以 `max_source_length=2048` 做 token 截断。训练集审计最大为 747 tokens（124,175 expanded rows），所以实际没有发生 token 截断。没有额外字段拼接或 task description 格式转换。

### 4.4 T5-base cache

原始路径：

```text
/root/.cache/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
```

Hugging Face revision：`a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`。

| 文件 | SHA256 |
|---|---|
| `config.json` | `46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033` |
| `generation_config.json` | `f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a` |
| `model.safetensors` | `a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4` |
| `spiece.model` | `d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86` |
| `tokenizer.json` | `d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066` |

训练生成的 best/last/final tokenizer 文件 SHA256 均为 `65430662114922dc57697bc782fc1122cd004547611abfcd520fe1a50f125bd9`。

## 5. RNG 调用顺序

原始 single-SID 路径的完整顺序如下：

1. 读取 94,912 条 query source rows。
2. 创建唯一的 Python RNG：`rng = random.Random(20260626)`。
3. 因为显式传入 `--dev-rows`，train/dev 都直接展开。没有执行 query train/dev split shuffle。
4. 每条 source row 的 `selected_skill_ids` 去重后排序；source row 的文件顺序保持不变。
5. catalog 按 `skill_id` 排序后，执行一次 `rng.shuffle(catalog_rows)`。
6. 执行 `torch.manual_seed(20260626)` 和 `torch.cuda.manual_seed_all(20260626)`。
7. 加载 tokenizer，并依次做 query-train、query-dev、catalog token-length audit。项目代码在 audit 内不调用 RNG。
8. 创建 `T5ForConditionalGeneration`，再创建 SID embedding、4 个 code heads 和 1 个 control head。自定义层初始化消耗 torch CPU RNG。
9. catalog 的 3 个 epoch 各在 epoch 开头对同一个 list 执行一次 `rng.shuffle`。catalog 共发生 4 次 shuffle：训练前 1 次，加每个 epoch 1 次。
10. query 沿用同一个 Python RNG 对象。5 个 query epoch 各 shuffle 一次，不重置 seed。
11. torch CPU/CUDA RNG 在 catalog→query 边界也不重置。dropout 的 CUDA RNG 轨迹连续。

catalog step 1 的 16 条 row IDs 和 query step 1 的 16 条 row IDs 已保存在 `batch_order_reference.json`。对应 newline-terminated hash：

```text
catalog step 1: 721a0d93aaef2f9d0104adae8501fcfabcefcd6bd9f23c1e4a1aae701710d443
query step 1:   56995788f5f03f084be7ac3ea36848206aa09943ba41dbc6dbf68b0000fefb85
```

## 6. catalog→query 的 optimizer 生命周期

模型创建后只构造一次 optimizer：

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
```

catalog 和 query 调用同一个 `train_stage` closure，closure 捕获同一个 `model`、`opt` 和 `rng`。

准确行为：

- catalog 完成 2,730 次 update 后，query 直接继续；
- query 保留 catalog 产生的 AdamW first moment、second moment 和 per-parameter step；
- 边界处不重建 optimizer、不清空动量、不重载 catalog checkpoint、不重置任何 seed；
- 无 scheduler、无 warmup，lr 全程固定为 `3e-4`；
- 每个 batch 执行一次 `zero_grad()`、`backward()`、`step()`；
- gradient accumulation=1，无 gradient clipping；
- `zero_grad()` 使用 PyTorch 2.10 默认 `set_to_none=True`；
- `init_model_dir=null`，没有加载外部 optimizer state。

原训练对 AdamW 只显式传入 `lr` 和 `weight_decay`。实际参数为：

```text
betas=(0.9, 0.999)
eps=1e-8
amsgrad=False
foreach=None
fused=None
```

这里有一个关键纠正：`foreach=None` 在 CUDA 上自动选择 foreach 路径。原 RTX probe 输出 `inferred_foreach=true`、`inferred_fused=false`。如果复现代码显式设置 `foreach=False`，训练 setting 已经改变。PyTorch 2.10 的 optimizer 文档也将 AdamW 的默认 CUDA 实现列为 foreach：<https://docs.pytorch.org/docs/stable/optim>。

RTX PRO 6000 对照结果：

```text
default foreach auto: step 1=4.528710842132568, step 50=4.203788757324219
explicit foreach=False: step 1=4.528710842132568, step 50=4.228314399719238
```

该对照说明 optimizer backend 从第 2 次 update 起改变数值轨迹。公司复现若显式禁用了 foreach，应先恢复 `foreach=None`。

## 7. 进程、GPU 与环境

原始启动设置：

```text
GPU: 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB
CUDA_VISIBLE_DEVICES=0
processes=1
DDP=false
effective batch=16
gradient accumulation=1
torch.compile=false
fused AdamW 未启用
```

2026-07-14 原 run 没有保存完整 `pip freeze`。以下版本由同一服务器的运行记录和 2026-08-08 精确复现 probe 共同确认：

```text
Python 3.10.12
torch 2.10.0+cu128
Transformers 5.3.0
CUDA runtime 12.8
NVIDIA driver 580.126.09（driver 报告 CUDA 13.0）
cuDNN 9.10.2 / torch.backends.cudnn.version()=91002
Tokenizers 0.22.2
Safetensors 0.7.0
huggingface-hub 1.7.1
sentencepiece 0.2.1
```

`requirements.nearest_20260716.txt` 来自同一服务器两天后的 W&B requirements，SHA256 为 `d00e616021d2701f4b6844f34e00aa4ce7ae0ee350b588b65b027ad9b061c67d`。该文件含重复的 `numpy`、`matplotlib` 和 `antlr4-python3-runtime` 条目，只能作为最近环境证据。

原命令和脚本未设置 deterministic/TF32/cuDNN flag。2026-08-08 在原服务器精确复现 step 50 时，有效值为：

```text
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=False
torch.are_deterministic_algorithms_enabled()=False
T5 attention implementation=eager
```

这些值来自“同环境实机复验”，不冒充 2026-07-14 预先保存的环境快照。它们在原 RTX 上重新得到逐 bit 相同的 step 50，因此足以约束当前复现。

## 8. 原始 loss 与日志

原 run 没有传 `--wandb-project`，因此没有 W&B history。落盘证据为 `train.log` 和 `loss_history.jsonl`。

```text
train.log SHA256        = 368e96df25ba6975b277ccc96610347e20c908e8fb4c8aec67c79b8184599c20
loss_history SHA256     = a32848f173c3433159ef6ef287ca07074d7392f23b3d7e99d5c9ea41b6a10fad
train manifest SHA256   = f048b00924538145ec49ebd57c1610dd97791a64d072c46a6398b6ac77125c69
```

关键 loss 已逐行复制到 `loss_reference.jsonl`。摘要如下：

| stage | step | train loss | dev loss | 条件 |
|---|---:|---:|---:|---|
| catalog | 1 | `4.528710842132568` | — | 2026-08-08 原 RTX probe；原日志因 `logging_steps=50` 未记录 step 1 |
| catalog | 50 | `4.203788757324219` | — | 原归档与 2026-08-08 原 RTX probe 逐 bit 一致 |
| catalog | 900 | `4.091408729553223` | — | catalog epoch 1 尾部附近；batch=16、BF16 |
| catalog | 950 | `4.072384357452393` | — | catalog epoch 2 开始；batch=16、BF16 |
| catalog | 2730 | `4.044649600982666` | — | catalog 3 epochs 完成；batch=16、BF16 |
| query | 200 | `3.962428331375122` | `3.7965545721027385` | query epoch 1；eval interval=200 |
| query | 1000 | `3.7825965881347656` | `3.3352677041366072` | query epoch 1；eval interval=200 |
| query | 38800 | `0.6922861933708191` | `0.7821362744041177` | query epoch 5；dev-best |
| query | 38805 | `0.41363900899887085` | `0.7822972546096613` | query 5 epochs 最后一步；dev 略差 |

## 9. best、last、final

query 共 38,805 updates（124,175 expanded rows、batch=16、5 epochs）。每 200 steps 评测一次，并在最后 step 38,805 再评测。

dev-best 出现在 step 38,800：

```text
best dev loss = 0.7821362744041177
last dev loss = 0.7822972546096613 at step 38805
```

训练先保存 `last`，再将 `best` 的 model 与 optimizer state 回载到内存，最后保存 `final`。所以 `final` 的文件 SHA256 与 `best` 不同，但模型 tensor fingerprint 相同。

| checkpoint | 文件 SHA256 | 模型 state fingerprint | 含义 |
|---|---|---|---|
| best | `325a74b173b1a24fa81a58896eed9a1b3eec85921f570da51c32803f1e0184b4` | `fda812fd3a3b32646c0399479a5b874280664d0aa926ce747aa540aba8edb90c` | query step 38,800 |
| last | `b9b496e2890b5c268043f58adf72ac1b11a9c9d7ef886cf8a6d5c5e273d56509` | `696565a4f2f50512585158820de1f838d33cafa4278707f4735dad997121b899` | query step 38,805 |
| final | `b92daf62e604f35dbecbb5c80897de9d212995d13c312f0f149f781caf5c0848` | `fda812fd3a3b32646c0399479a5b874280664d0aa926ce747aa540aba8edb90c` | 回载 best 后重新保存 |

ModelScope model revision `8e5b73e59d617e1d9b5ee893efe1ef4c6da0ce6d` 内保留 best/last/final、manifest、loss history 和 train log。

## 10. 最终检索结果的原始证据链

Notion 中的结果对应以下唯一路径：

```text
checkpoint:
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/
single_sid_d4_lightgcn_d32_letterofficial_task5_core75_gt2_t5base_cat3_querye5_b16/final

evaluation directory:
outputs/eval/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714/core75_beam50

predictions:
outputs/eval/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714/
core75_beam50/predictions.jsonl
```

原始评测命令：

```bash
cd /root/workplace/SkillSearch-data-synthesis
root=outputs/eval/skill_gr_d4_lightgcn_letterofficial_task5_core75_gt2_20260714
model=outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/single_sid_d4_lightgcn_d32_letterofficial_task5_core75_gt2_t5base_cat3_querye5_b16/final
sem=outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/semantic_ids.jsonl
rows=outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/canonical_alias_core75_learned_d4_inputs
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python3 scripts/skill_gr_pipeline.py eval-single-sid-decoder \
  --rows "$rows/rows_003.skillsbench_core75.rows.jsonl" \
  --model-dir "$model" \
  --semantic-ids "$sem" \
  --output-dir "$root/core75_beam50" \
  --split-name skillsbench_core75 \
  --max-source-length 2048 \
  --num-beams 50 \
  --k-values 1,10,50 \
  --rank-aggregation beam_order \
  --log-every 10
```

原始 `eval_manifest.json` 记录 `model_dir=.../final`、`num_beams=50`、`rank_aggregation=beam_order`、`max_rank=50`、`max_skills=1`。Official core75 含 75 queries、200 个 gold mentions，其中 Single 24 queries、Multi 51 queries。

| subset | R@1 | R@10 | R@50 | 评测条件 |
|---|---:|---:|---:|---|
| Overall | 28.3683% | 69.0000% | 75.0000% | Official core75 75 queries；beam=50；beam-order；macro Recall |
| Single | 41.6667% | 45.8333% | 50.0000% | Official core75 24 single-skill queries；同上 |
| Multi | 22.1102% | 79.9020% | 86.7647% | Official core75 51 multi-skill queries；同上 |

这些数值直接来自归档的 `ranked_metrics.json`、`ranked_metrics.single.json` 和 `ranked_metrics.multi.json`，与 Notion 表完全一致。`predictions.jsonl` 含 75 行，并且公司侧已经确认归档 checkpoint 在当前代码下复现了全部 75 条 Top-50 排序。因此可以排除“Notion 指标抄自另一条 run”“评测误用了 best/last”“归档 checkpoint 与 predictions 不配套”三种可能。

原始评测文件 SHA256：

```text
1fe070112bbac91272fbbd8f2a5d7a5d62e9e5f206aaf7f08e759ed365b03703  eval_manifest.json
44cd1b3fe6fe0ac19fc154c21492a0fa620b29f0117312f9e66acad23492440d  eval_metrics.json
167426964ab2938889526128a8119f39043a3d788d590105cd4a3be6cc31cf88  predictions.jsonl
818e4f6df5a911943f003fc1d00911c0652e2cf4b8d28b67b912ca12dc24f6b0  ranked_metrics.json
ff439b28465167e6ec71b5c2013214d7b6a040ecebda2f718306fe44b2b1ca15  ranked_metrics.single.json
13bd5e8dc20fd7372d9953cb25a32d6c16ef632044333890d3779e8b999c6b9d  ranked_metrics.multi.json
```

## 11. 当前复现 manifest 的逐字段比较

公司 manifest 路径为：

```text
/models_rw/SkillSearch-data-synthesis/experiments/legacy_catalog_base_exact_runtime_20260808_v1/legacy-c3-q5-seed20260626/single_sid_decoder_train_manifest.json
```

该文件不在本地工作区，当前也无法从本机连接公司服务器，所以不能凭路径推测字段差异。原始 manifest 已放入 `archive/train/single_sid_decoder_train_manifest.json`。在公司服务器执行下面的只读比较，可只打印不同字段：

```bash
python3 - <<'PY'
import json
from pathlib import Path

old = json.loads(Path(
    "repro/20260714_letter_base_single_sid/archive/train/"
    "single_sid_decoder_train_manifest.json"
).read_text())
new = json.loads(Path(
    "/models_rw/SkillSearch-data-synthesis/experiments/"
    "legacy_catalog_base_exact_runtime_20260808_v1/"
    "legacy-c3-q5-seed20260626/single_sid_decoder_train_manifest.json"
).read_text())

def diff(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in a:
                print(path, "<missing>", "=>", repr(b[key]))
            elif key not in b:
                print(path, repr(a[key]), "=>", "<missing>")
            else:
                diff(a[key], b[key], path)
    elif a != b:
        print(prefix, repr(a), "=>", repr(b))

diff(old, new)
PY
```

## 12. 公司服务器的最短诊断路径

从私有 GitHub 拉取包含本目录的 commit，并从固定 ModelScope revision 恢复 assets。随后在 A800 环境运行：

```bash
cd /path/to/SkillSearch-data-synthesis
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python3 repro/20260714_letter_base_single_sid/catalog_50_probe.py \
  --repo-root "$PWD" \
  --model t5-base \
  --foreach auto
```

按以下顺序比较：

1. `source_sha256` 必须等于 `7b883c...6b76`。
2. `initial_model_fingerprint` 必须等于 `c219d0...ebde`。不一致时检查 T5 revision、Transformers 版本和模型初始化前的 torch CPU RNG。
3. `row_ids_sha256` 必须等于 `721a0d...443`。不一致时检查 input 文件、catalog sort 与两次 shuffle。
4. 比较 step 1。若模型 hash 和 batch hash 相同，但 A800 的 step 1 已不同，差异来自 GPU/BF16 forward 数值路径。
5. 若 step 1 相同而 step 50 分叉，重点检查 AdamW `foreach`、dropout CUDA RNG 和 GPU 数值轨迹。

公司复现当前报告的 step 50 loss 为 `4.184667`（A800、batch=16、BF16、seed=20260626）。原 RTX step 50 为 `4.203788757324219`（默认 `foreach=None`，有效 foreach=true）。先让公司 probe 返回 initial hash、batch hash、step 1、`optimizer_param_group` 和 `inferred_foreach`，即可把原因定位到初始化、顺序、首步 forward 或累积 update 四层中的一层。

当前归档已经闭环到原始 predictions。公司复现的问题是训练所得 checkpoint 的检索能力不同，评测口径和归档结果归属已被排除。原 RTX 50-step probe 仍可用于定位训练轨迹从哪里分叉，但 catalog loss 本身不能解释 Single R@1 的最终差距。
