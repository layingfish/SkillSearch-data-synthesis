# SkillSearch Query Generation Handoff

更新时间：2026-06-25

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
/Users/chenyufei/Documents/GR&Agentic/SkillSearch-query-generation
```

远端仓库：

```text
/root/workplace/SkillSearch-query-generation
```

prompt 冻结基线 commit：

```text
58f284f chore: freeze skillrouter metadata replay pipeline
```

本文档本身会产生后续提交；接手时以 `git log --oneline -1` 看到的最新提交为准。

远端新仓库通过 symlink 复用旧数据和 profile：

```text
/root/workplace/SkillSearch-query-generation/outputs  -> /root/workplace/SkillSearch/outputs
/root/workplace/SkillSearch-query-generation/profiles -> /root/workplace/SkillSearch/profiles
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
固定 0601 pilot candidate inputs
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
-> fixed 0601 sampled single skills / redundant multi-skill panels
-> GPT-4o selector / query generator
-> parse generated selected_skill_ids and query
-> deterministic selected-id repair
-> deterministic quality evaluation
-> workflow_quality_filter
-> accepted / weak / rejected
```

重要点：

- 当前 replay 脚本不重新构造 panels，而是读取 0601 已固定的 1/2/3/4 skill pilot 输入。
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

- 读取固定 pilot inputs。
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

固定 0601 pilot inputs：

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

远端 profile 路径：

```text
/root/workplace/SkillSearch-query-generation/profiles/gpt4o_openrouter.json
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
cd /Users/chenyufei/Documents/GR\&Agentic/SkillSearch-query-generation

python3 -m py_compile \
  scripts/generate_single_skill_probe.py \
  scripts/generate_multiskill_probe.py \
  scripts/replay_pilot_400_skillrouter_prompt.py

python3 -m unittest tests/test_skillrouter_prompt_alignment.py
```

### 9.2 远端测试

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-query-generation
python3 -m py_compile \
  scripts/generate_single_skill_probe.py \
  scripts/generate_multiskill_probe.py \
  scripts/replay_pilot_400_skillrouter_prompt.py
python3 -m unittest tests/test_skillrouter_prompt_alignment.py
'
```

### 9.3 远端 smoke run

建议先跑每个 `k` 各 2 条：

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-query-generation
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
cd /root/workplace/SkillSearch-query-generation
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

## 13. 常见坑

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
/root/workplace/SkillSearch-query-generation
```

旧 `/root/workplace/SkillSearch` 现在只是数据/profile 容器。

6. full dataset 成本不能用 `258 panels` 估算上限。

之前的 `258 panels` 只是代表性 sampled panels，不是全量可生成 workflow 上限。若要估算大规模成本，应按目标 multiskill query 数量计算。

## 14. 扩展方向

当前系统适合稳定复现 400 条 replay 和做小规模扩展。后续如果要扩大到 10k/50k/100k，多做这些改进：

```text
1. 更系统地扩展 candidate panel sampling
2. 对同一 panel 多 seed / 多组合采样
3. selected_id_not_in_panel retry
4. medium_confidence targeted rewrite
5. verifier/audit 只用于 hard cases，不默认放进生成 prompt
6. 数据导出脚本化，固定 clean dataset schema
```

## 15. 当前验收状态

本地和远端都已验证：

```bash
python3 -m py_compile scripts/generate_single_skill_probe.py scripts/generate_multiskill_probe.py scripts/replay_pilot_400_skillrouter_prompt.py
python3 -m unittest tests/test_skillrouter_prompt_alignment.py
```

测试结果：

```text
Ran 10 tests
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
