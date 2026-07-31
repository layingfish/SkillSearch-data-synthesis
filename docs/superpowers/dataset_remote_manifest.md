# Dataset Remote Manifest

Generated datasets are stored only on the remote server.

```text
remote repo: gpu1_tailscale_jmp:/root/workplace/SkillSearch-data-synthesis
remote data root: /root/workplace/SkillSearch-data-synthesis/outputs/
```

## Canonical Training Dataset

Use this file for the current English-only high-quality GR training set:

```text
outputs/eval/skillrouter_balanced_500_each_topup_v1/balanced_500_each.final_high_quality.english_only_v2_1.clean.jsonl
rows: 2000
sha256: 3651dcfa890c747cc28dc39e2e97db72cfc204978fb4c2e3676b64cd7468faf8
```

Validation summary:

```text
k1/k2/k3/k4: 500/500/500/500
workflow_quality_filter.status: accepted x 2000
Chinese-character rows: 0
query duplicates: 0
selected skill set duplicates: 0
target_skill_count mismatches: 0
full skill_id leaks: 0
blocked internal phrase hits: 0
```

## Audit Files

```text
outputs/eval/skillrouter_balanced_500_each_topup_v1/english_only_repair_v2/balanced_500_each.final_high_quality.english_only_v2_1.with_language_repair_audit.jsonl
rows: 2000
sha256: 7e8eb37ff58627fdc44a31d04f84b5486e857a26f870fd6d4c663023ab6e9924

outputs/eval/skillrouter_balanced_500_each_topup_v1/english_only_repair_v2/english_only_v2_1_summary.json
rows: 54
sha256: cbc6c2c13dfc16bb189e73429b2f6e0d61609ce547c90f453e70c493b3815071
```

## Reference-400 Dataset

Official name:

```text
SkillRouter Reference-400 Candidate Set (2026-06-01 freeze)
```

The path below keeps its historical filename for compatibility.

```text
outputs/eval/reference_datasets/pilot_400.jsonl
rows: 400
sha256: 01d9d23124328e8bc8c2cc31c90ae6952403a05878e65788b8f459659c45604f
```

## Local Policy

- Keep code and lightweight documentation locally.
- Keep generated datasets, raw responses, audit JSONL files, and replay outputs on the remote server.
- `outputs/` is gitignored and should remain absent from the local working tree unless a small subset is temporarily fetched for inspection.
