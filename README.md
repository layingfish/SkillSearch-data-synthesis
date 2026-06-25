# SkillSearch Query Generation

这是当前保留的 query-skill / query-multiskill 数据生成代码版本。

接手者请先读 [HANDOFF.md](HANDOFF.md)，那里包含远端路径、API profile、运行命令、数据口径和已知边界。

核心口径：

```text
固定 0601 pilot candidate inputs
+ SkillRouter-style plain-text query prompt
+ 400 条版本的 metadata-as-body-preview input construction
+ 当前 selected-id repair / quality filter / reclassification
```

这版代码对应当前给 partner 审阅的 400 条数据：

```text
/Users/chenyufei/Documents/GR&Agentic/partner_review/0605_skillrouter_replay_reclassified_v2/pilot_400.jsonl
```

## Pipeline

```text
classified skills
-> candidate panel construction
-> GPT-4o selector / query generator
-> parse generated selected_skill_ids and query
-> deterministic selected-id repair
-> deterministic quality evaluation
-> workflow_quality_filter
-> accepted / weak / rejected
-> high_confidence / medium_confidence / rejected
```

## Prompt Version

这版恢复到 400 条数据使用的 prompt input construction：

```text
Description: skill description prefix
Skill body: metadata-as-body-preview
```

这里的 `Skill body` 字段名保留 SkillRouter prompt 的形状，但内容不是完整原始 skill body 的截断，而是结构化 metadata preview。

### Single-Skill Prompt

System:

```text
You are an experienced user of AI assistants. You write clear, realistic task requests that describe what you need to accomplish.
```

User:

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

### Multi-Skill Prompt

System:

```text
You are an experienced user of AI assistants. You write clear, realistic task requests that describe what you need to accomplish.
```

User:

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

Candidate spec:

```text
{number}. Skill id: {skill_id}
   Skill name: {skill_name}
   Category: {task_space_id}
   Description: {description}
   Skill body: Operation: {operation_role}/{operation_type}; Task objects: {task_object_types}; Inputs: {input_artifact_types}; Outputs: {output_artifact_types}; Modalities: {modality_types}
```

## Why Not Raw Body Prefix

后续实验过 raw body-prefix hybrid：

```text
Skill body: raw skill body prefix
Workflow metadata: operation / objects / artifacts / modalities
```

40 条 smoke 的实验结果显示，它没有提升质量，反而带来这些问题：

```text
1. prompt token 约为 metadata-as-body-preview 的 1.8x
2. 更容易出现 selected_id_not_in_panel
3. query 更容易携带专名、平台和实现细节
4. checklist-like 表达没有明显下降
```

因此当前主线不默认拼接原始 skill body。原始 body 只适合作为 hard-case retry、audit 或 verifier 的补充信息。

## Postprocessing

当前保留后续改进过的后处理逻辑：

```text
candidate number -> full skill id repair
basename -> full skill id repair
case normalization repair
skill name -> full skill id repair
query_too_long as stat-only warning
workflow_type: single_skill / sequential_workflow / capability_bundle / weak_overlap / invalid
accepted / weak / rejected split
```

数据集分桶口径：

```text
high_confidence: accepted and no remaining weak warning
medium_confidence: accepted with weak warning
rejected: hard issue, such as selected_id_not_in_panel
```

## 400-Row Cost

当前 400 条 partner review 数据的真实 token usage 已单独保存到：

```text
/Users/chenyufei/Documents/GR&Agentic/partner_review/0605_skillrouter_replay_reclassified_v2/pilot_400_usage.jsonl
```

按 GPT-4o 价格：

```text
input:  $2.5 / 1M tokens
output: $10  / 1M tokens
```

400 条真实用量：

| k | rows | input tokens | output tokens | total tokens | estimated cost |
|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 23,853 | 18,661 | 42,514 | $0.24624 |
| 2 | 100 | 197,197 | 14,635 | 211,832 | $0.63934 |
| 3 | 100 | 197,403 | 16,079 | 213,482 | $0.65430 |
| 4 | 100 | 234,871 | 17,866 | 252,737 | $0.76584 |
| total | 400 | 653,324 | 67,241 | 720,565 | $2.30572 |

## Main Commands

Run the 400-row replay with the frozen prompt defaults:

```bash
python3 scripts/replay_pilot_400_skillrouter_prompt.py \
  --profile profiles/gpt4o_openrouter.json \
  --output-dir outputs/eval/skillrouter_metadata_replay \
  --k-values 1,2,3,4 \
  --workers 4 \
  --temperature 0.6 \
  --top-p 0.9
```

Run tests:

```bash
python3 -m py_compile scripts/generate_single_skill_probe.py scripts/generate_multiskill_probe.py scripts/replay_pilot_400_skillrouter_prompt.py
python3 -m unittest tests/test_skillrouter_prompt_alignment.py
```
