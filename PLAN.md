# SkillSearch 高质量合成 Pipeline 改进计划

## Summary

目标是把当前 “panel 召回后让 LLM 自由选 k 个 skill 并写 query” 改成：

```text
v2 coarse panel
-> composition card
-> compatibility sampler
-> replay-compatible input
-> existing SkillRouter-style query generation
-> card-aware verifier
-> manual audit smoke
```

核心决策：

- 保留现有 v2 classification 作为 coarse retrieval，不把它当 workflow composition 证据。
- 第一阶段不上全量 LLM composition planning，先做 deterministic/heuristic MVP，避免成本和状态复杂度。
- 不改 frozen prompt 主体；新逻辑通过新 input builder 和 verifier 接入。
- k4 不再硬凑；找不到强兼容组合就继续采样，不降级。

验收目标：

- k4 人工 pass rate 从 `45.5%` 提升到至少 `60%-65%`。
- 总体 manual pass rate 从 `72.15%` 提升到 `80%+`。
- `tech_stack_or_context_mismatch`、`redundant_skills`、`missing_selected_skill_signal` 明显下降。

## Key Changes

新增 `scripts/build_composition_cards.py`：

- 输入：`outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl`
- 输出：`composition_cards.jsonl`
- 每个 skill 生成最小 card：

```json
{
  "skill_id": "...",
  "task_space_id": "...",
  "source_task_space_id": "...",
  "primary_task_object_type": "...",
  "operation_role": "...",
  "operation_type": "...",
  "input_artifact_types": [],
  "output_artifact_types": [],
  "platform_vendor_terms": [],
  "project_context_terms": [],
  "required_query_terms": [],
  "incompatible_context_terms": [],
  "substitute_group": "...",
  "evidence_snippets": []
}
```

Card 抽取规则：

- `platform_vendor_terms` 只收高精度词，如 `react`, `vue`, `svelte`, `fastapi`, `django`, `rails`, `laravel`, `swiftui`, `expo`, `firebase`, `supabase`, `aws`, `azure`, `stripe`, `notion`, `slack`。
- `project_context_terms` 从 skill id/name/description 中抽明显专用项目、品牌、仓库、产品名。
- `required_query_terms` 只对 vendor/project/强技术栈启用，避免把 query 推成关键词堆砌。
- `substitute_group` 用 normalized capability signature 生成：`task_object + operation_type + main_platform/vendor`。
- `evidence_snippets` 保留 description/body 中支持这些 term 的短片段，供 verifier 解释。

新增 `scripts/build_skillrouter_composition_inputs.py`：

- 输入：v2 classified skills + composition cards。
- 输出 replay-compatible input files：`input_1skill.jsonl`、`input_2skills.jsonl`、`input_3skills.jsonl`、`input_4skills.jsonl`。
- 每行保留现有字段：`candidate_skill_ids`, `target_skill_count`, `task_space_id`, `anchor_task_object`, `panel_fingerprint`, `sampling`。
- 新增字段：`composition_constraints`，记录 shared context、role diversity、artifact bridge、blocked incompatible terms、substitute checks。

采样规则：

- 同 `source_task_space_id` 优先；不足时退到同 `task_space_id`。
- 至少共享一个非泛化 task object，或存在非泛化 input/output artifact bridge。
- 不允许同 `substitute_group` 重复。
- 不允许 incompatible tech/context pair，除非 card 或 query context 明确支持 migration/polyglot。
- k2 至少 2 个 operation roles；k3 至少 3 个 role 或 2 role + artifact bridge；k4 至少 3 个 role + shared context + no substitute collision。
- k4 找不到合格组合时继续采样，不降级。

新增 `scripts/verify_skillrouter_composition_rows.py`：

- 输入：generated rows + composition cards。
- 输出：`composition_verdict`，并可生成 pass/revise/drop-like machine filter。
- 检查：
  - query 覆盖 selected cards 的 `required_query_terms`；
  - query 中 vendor/project/tech term 必须被 selected card evidence 支持；
  - 每个 selected skill 至少有一个触发信号：role、artifact、task object、vendor/project、output；
  - selected set 不得违反 substitute/incompatible 规则；
  - query 词数建议 25-110，超过 140 直接 weak/revise；
  - 检测明显模板化短语和 `Inputs/Outputs` 堆叠。

现有 `scripts/replay_pilot_400_skillrouter_prompt.py` 只做小改：

- 支持读取新 input 文件中的 `composition_constraints` 并透传到输出。
- 调用 card-aware verifier，写入 `composition_verdict`。
- 默认行为保持兼容；不破坏当前 frozen 400 replay。

## Test Plan

单元测试新增覆盖：

- composition card 抽取：
  - FastAPI/Django/Rails/Laravel/React/Vue/Svelte/SwiftUI/Expo 等技术栈能正确进入 `platform_vendor_terms`。
  - 专用项目/品牌词进入 `project_context_terms` 和 `required_query_terms`。
  - 泛词如 `api`, `backend`, `tool`, `agent` 不进入强 required terms。

- compatibility sampler：
  - React + Svelte 同一 UI app 默认不兼容。
  - FastAPI + Django + TypeORM 默认不兼容。
  - `llamafile` + `localai` 被同 substitute group 拦截。
  - AWS Aurora + log analysis + AWS cost optimization 可通过。
  - Sphinx docs + technical writer + extra reference material 可通过。
  - k4 不满足规则时继续采样，而不是输出弱组合。

- verifier：
  - selected skill 的 vendor/project required term 未出现在 query 时判 weak/revise。
  - query 出现 unsupported vendor term 时判 weak/revise。
  - selected set 有 substitute collision 时 reject/drop。
  - query 过长或 `Inputs/Outputs` 模板化时降级。
  - 每个 selected skill 缺少触发信号时降级。

端到端 smoke：

- 先生成每个 k 各 50 条 input，不调用 LLM，只检查 sampler yield、skip reason、k 分布。
- 再生成 200-400 条真实样本，做人工审计。
- 只有当 manual pass rate 达到目标后，再扩到 2000+。

## Assumptions

- 第一阶段不追求严格 DAG，只要求 `sequential_workflow` 或高质量 `capability_bundle`。
- `revise` 样本先不直接训练；后续可作为 repair/top-up 池。
- `drop` 样本剔除，不作为正样本。
- k4 质量优先于一次性生成数量；不足时继续采样更严格的 k4，不降级。
- 不默认拼 raw body 到 query generation prompt；body/evidence 只用于 card 抽取和 verifier。
