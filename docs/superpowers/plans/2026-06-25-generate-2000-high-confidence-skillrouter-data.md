# Generate 2000 High-Confidence SkillRouter Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate at least 2000 high-confidence SkillRouter query training rows where `workflow_quality_filter.status == "accepted"`.

**Architecture:** Add a deterministic scalable input-builder that samples single-skill rows and multi-skill candidate panels from the existing v2 panel classification output, then reuse `scripts/replay_pilot_400_skillrouter_prompt.py` for LLM generation. Keep generation, replay, and export separate so the same sampler can scale from 2000 rows to tens of thousands with quota controls and deduplication.

**Tech Stack:** Python standard library, existing JSONL scripts, existing `generate_single_skill_probe.py`, `generate_multiskill_probe.py`, `replay_pilot_400_skillrouter_prompt.py`, Huayan OpenAI-compatible profile `profiles/gpt4o_huayanapi.json`, `python3 -m unittest`, `jq`.

---

## Target Definition

Use this quality definition:

```text
high_confidence = workflow_quality_filter.status == "accepted"
medium_confidence = workflow_quality_filter.status == "weak"
rejected = workflow_quality_filter.status == "rejected"
```

Do not count `weak` rows toward the 2000 high-confidence target.

Use these first-run raw quotas:

```text
k=1 raw rows: 525
k=2 raw rows: 625
k=3 raw rows: 550
k=4 raw rows: 625
total raw rows: 2325
```

Rationale from historical 400-row replay:

```text
k=1 accepted status: 96/100
k=2 accepted status: 84/100
k=3 accepted status: 92/100
k=4 accepted status: 85/100
overall high-confidence: 357/400 = 89.25%
```

Expected high-confidence rows from 2325 raw rows are about 2060 if rates hold. If smoke shows lower rates, increase raw quotas before the full run.

---

## File Structure

Create:

- `scripts/build_skillrouter_replay_inputs.py`
  - Builds replay-compatible input JSONL files for `k=1,2,3,4`.
  - Enumerates `(source_task_space_id, anchor_task_object)` buckets instead of only one panel per task space.
  - Applies high-precision and balanced sampling tiers.
  - Enforces panel fingerprint uniqueness and skill usage caps.
  - Writes `input_1skill.jsonl`, `input_2skills.jsonl`, `input_3skills.jsonl`, `input_4skills.jsonl`, and `manifest.json`.

- `scripts/export_skillrouter_high_confidence.py`
  - Splits replay output into `high_confidence.jsonl`, `medium_confidence.jsonl`, `rejected.clean.jsonl`, `accepted_for_training.clean.jsonl`, and `usage.jsonl`.
  - Removes `usage` from training files.
  - Counts status, target skill count, workflow type, duplicate selected sets, and task-space distribution.

- `tests/test_skillrouter_replay_input_builder.py`
  - Unit tests for filtering, bucket construction, panel sampling, quotas, schema compatibility, and determinism.

- `tests/test_skillrouter_high_confidence_export.py`
  - Unit tests for clean export and correct high/medium/rejected splits.

Modify:

- `scripts/replay_pilot_400_skillrouter_prompt.py`
  - Preserve optional sampling provenance from input rows in output rows:
    - `sampling`
    - `panel_fingerprint`
    - `panel_source_tier`
  - This is required for audit and top-up generation.

- `README.md`
  - Add a short section for 2000-row generation commands after implementation passes.

Do not modify:

- Prompt text in `generate_single_skill_probe.py` or `generate_multiskill_probe.py`.
- Quality semantics in `workflow_quality_filter`.
- Existing frozen 400-row replay inputs.

---

## Task 1: Add Input Builder Tests First

**Files:**
- Create: `tests/test_skillrouter_replay_input_builder.py`
- Create later: `scripts/build_skillrouter_replay_inputs.py`

- [ ] **Step 1: Write import-failing tests**

Create `tests/test_skillrouter_replay_input_builder.py` with these test cases:

```python
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_skillrouter_replay_inputs import (  # noqa: E402
    SamplingConfig,
    build_anchor_buckets,
    build_inputs,
    confidence,
    panel_fingerprint,
    usable_skill,
)


def row(
    sid: str,
    *,
    source_task_space_id: str = "development/testing_quality_assurance/test_automation_frameworks",
    task_space_id: str = "development/testing_quality_assurance/test_automation_frameworks",
    role: str = "validate",
    op_type: str = "unit_or_integration_test",
    objects: list[str] | None = None,
    flags: list[str] | None = None,
    task_object_confidence: float = 0.9,
) -> dict:
    return {
        "skill_id": sid,
        "id": sid,
        "skill_name": sid.rsplit("/", 1)[-1],
        "description": f"Useful skill {sid} for a concrete engineering task.",
        "source_task_space_id": source_task_space_id,
        "task_space_id": task_space_id,
        "operation_role": role,
        "operation_type": op_type,
        "task_object_types": objects or ["test_suite"],
        "input_artifact_types": ["source_code"],
        "output_artifact_types": ["test_results"],
        "modality_types": ["code"],
        "quality_flags": flags or [],
        "classification_confidence": {
            "operation": 0.9,
            "artifact": 0.8,
            "task_object": task_object_confidence,
            "task_space": 0.8,
        },
    }


class SkillRouterReplayInputBuilderTest(unittest.TestCase):
    def test_confidence_reads_missing_as_zero(self) -> None:
        self.assertEqual(confidence({}, "operation"), 0.0)
        self.assertEqual(confidence({"classification_confidence": {"operation": "0.75"}}, "operation"), 0.75)

    def test_usable_skill_filters_generic_and_excluded_flags(self) -> None:
        config = SamplingConfig(seed=7)
        self.assertTrue(usable_skill(row("testing/a"), config, tier="local"))
        self.assertFalse(usable_skill(row("testing/b", role="general_capability"), config, tier="local"))
        self.assertFalse(usable_skill(row("testing/c", flags=["broad_skill"]), config, tier="local"))
        self.assertFalse(usable_skill(row("testing/d", task_object_confidence=0.6), config, tier="local"))
        self.assertTrue(usable_skill(row("testing/e", task_object_confidence=0.6), config, tier="balanced"))

    def test_build_anchor_buckets_uses_source_task_space_and_non_generic_object(self) -> None:
        config = SamplingConfig(seed=7)
        rows = [
            row("testing/a", objects=["test_suite", "codebase"]),
            row("testing/b", objects=["test_suite"]),
            row("testing/c", objects=["unknown"]),
        ]
        buckets = build_anchor_buckets(rows, config, tier="local")
        self.assertIn(("development/testing_quality_assurance/test_automation_frameworks", "test_suite"), buckets)
        self.assertNotIn(("development/testing_quality_assurance/test_automation_frameworks", "codebase"), buckets)
        self.assertNotIn(("development/testing_quality_assurance/test_automation_frameworks", "unknown"), buckets)

    def test_panel_fingerprint_is_order_independent(self) -> None:
        self.assertEqual(panel_fingerprint(["b", "a", "c"]), panel_fingerprint(["c", "b", "a"]))

    def test_build_inputs_is_deterministic_and_replay_compatible(self) -> None:
        roles = ["load", "extract", "analyze", "implement", "validate", "generate"]
        types = [
            "document_file_load",
            "structured_data_parse",
            "comparison_or_impact_analysis",
            "backend_api_implementation",
            "unit_or_integration_test",
            "report_generation",
            "validation_check",
            "structured_output_generation",
        ]
        rows = []
        for i in range(80):
            rows.append(
                row(
                    f"testing/skill-{i:03d}",
                    role=roles[i % len(roles)],
                    op_type=types[i % len(types)],
                    objects=["test_suite"],
                )
            )
        config = SamplingConfig(
            seed=123,
            target_counts={1: 10, 2: 8, 3: 8, 4: 8},
            candidate_sizes={2: 12, 3: 12, 4: 15},
            max_skill_uses=6,
        )
        first = build_inputs(rows, config)
        second = build_inputs(rows, config)
        self.assertEqual(first, second)

        self.assertEqual(len(first[1]), 10)
        self.assertEqual(len(first[2]), 8)
        self.assertEqual(len(first[3]), 8)
        self.assertEqual(len(first[4]), 8)

        self.assertIn("skill_id", first[1][0])
        self.assertEqual(first[1][0]["target_skill_count"], 1)

        for target in [2, 3, 4]:
            sample = first[target][0]
            self.assertEqual(sample["target_skill_count"], target)
            self.assertIn("candidate_skill_ids", sample)
            self.assertGreaterEqual(len(sample["candidate_skill_ids"]), target + 4)
            self.assertIn("task_space_id", sample)
            self.assertIn("anchor_task_object", sample)
            self.assertIn("sampling", sample)
            self.assertIn("panel_fingerprint", sample)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test and verify it fails because the module does not exist**

Run:

```bash
python3 -m unittest tests/test_skillrouter_replay_input_builder.py
```

Expected:

```text
ModuleNotFoundError: No module named 'build_skillrouter_replay_inputs'
```

---

## Task 2: Implement Scalable Replay Input Builder

**Files:**
- Create: `scripts/build_skillrouter_replay_inputs.py`
- Test: `tests/test_skillrouter_replay_input_builder.py`

- [ ] **Step 1: Implement the builder module**

Create `scripts/build_skillrouter_replay_inputs.py` with these public interfaces:

```python
@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 20260625
    target_counts: dict[int, int] = field(default_factory=lambda: {1: 525, 2: 625, 3: 550, 4: 625})
    candidate_sizes: dict[int, int] = field(default_factory=lambda: {2: 12, 3: 12, 4: 15})
    tiers: tuple[str, ...] = ("local", "balanced")
    max_skill_uses: int = 8
    exclude_flags: tuple[str, ...] = ("generic_operation_label", "workflow_like_skill", "broad_skill")
    min_operation_confidence: float = 0.55
    min_artifact_confidence: float = 0.60
    local_min_task_object_confidence: float = 0.70
    balanced_min_task_object_confidence: float = 0.55
    min_bucket_size: int = 15
    min_roles: int = 4
    min_operation_types: int = 8
```

Required behavior:

```text
confidence(row, key):
  Return float(row["classification_confidence"][key]) or 0.0.

usable_skill(row, config, tier):
  Reject general_capability.
  Reject excluded quality flags.
  Require operation confidence >= 0.55.
  Require artifact confidence >= 0.60.
  Require task_object confidence >= 0.70 for local tier.
  Require task_object confidence >= 0.55 for balanced tier.

build_anchor_buckets(rows, config, tier):
  Group usable rows by (source_task_space_id, non-generic task_object).
  Exclude task objects "codebase" and "unknown".
  Keep only buckets with at least min_bucket_size, min_roles, and min_operation_types.

sample_panel(bucket_rows, target, config, rng, skill_use_counts):
  Use role/type stratified round-robin.
  Prefer rows with lower current skill use count.
  Return candidate_sizes[target] skill ids.
  Reject duplicate skill ids.
  Reject panels with fewer than target + 4 candidates.

build_inputs(rows, config):
  Build k=1 rows by stratified skill sampling across task_space_id.
  Build k=2/3/4 rows from local tier first, then balanced tier.
  Enforce panel_fingerprint uniqueness per target.
  Enforce max_skill_uses across multi-skill candidate panels.
  Return {1: [...], 2: [...], 3: [...], 4: [...]}.
```

Output row schema for `k=1`:

```json
{
  "index": 1,
  "skill_id": "namespace/name",
  "skill_name": "name",
  "task_space_id": "development/...",
  "operation_role": "implement",
  "operation_type": "backend_api_implementation",
  "target_skill_count": 1,
  "sampling": {
    "source": "skillrouter_2000_input_builder",
    "seed": 20260625,
    "tier": "single_skill_stratified"
  }
}
```

Output row schema for `k=2/3/4`:

```json
{
  "index": 1,
  "task_space_id": "development/...",
  "anchor_task_object": "test_suite",
  "candidate_skill_ids": ["namespace/a", "namespace/b"],
  "target_skill_count": 2,
  "panel_fingerprint": "sha256:...",
  "panel_source_tier": "local",
  "sampling": {
    "source": "skillrouter_2000_input_builder",
    "seed": 20260625,
    "tier": "local",
    "bucket_key": ["development/...", "test_suite"],
    "candidate_size": 12,
    "role_count": 5,
    "operation_type_count": 10
  }
}
```

- [ ] **Step 2: Implement CLI**

The CLI must support:

```bash
python3 scripts/build_skillrouter_replay_inputs.py \
  --skills outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl \
  --output-dir outputs/eval/skillrouter_2000_inputs_$(date +%Y%m%d_%H%M%S) \
  --target-counts 1:525,2:625,3:550,4:625 \
  --seed 20260625 \
  --max-skill-uses 8
```

CLI outputs:

```text
input_1skill.jsonl
input_2skills.jsonl
input_3skills.jsonl
input_4skills.jsonl
manifest.json
```

`manifest.json` must include:

```json
{
  "target_counts_requested": {"1": 525, "2": 625, "3": 550, "4": 625},
  "target_counts_written": {"1": 525, "2": 625, "3": 550, "4": 625},
  "seed": 20260625,
  "max_skill_uses": 8,
  "panel_fingerprint_duplicates": 0,
  "candidate_skill_use_max": 8,
  "tiers_used": {"local": 0, "balanced": 0}
}
```

- [ ] **Step 3: Run tests**

Run:

```bash
python3 -m unittest tests/test_skillrouter_replay_input_builder.py
```

Expected:

```text
Ran 5 tests
OK
```

- [ ] **Step 4: Run a tiny local fixture build**

Use the test fixture only if local `outputs/` is absent. On remote, use the real file:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=outputs/eval/skillrouter_2000_inputs_smoke_$(date +%Y%m%d_%H%M%S)
python3 scripts/build_skillrouter_replay_inputs.py \
  --skills outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl \
  --output-dir "$out" \
  --target-counts 1:25,2:25,3:25,4:25 \
  --seed 20260625 \
  --max-skill-uses 8
cat "$out/manifest.json"
'
```

Expected:

```text
target_counts_written == {"1": 25, "2": 25, "3": 25, "4": 25}
panel_fingerprint_duplicates == 0
```

---

## Task 3: Preserve Sampling Provenance in Replay Output

**Files:**
- Modify: `scripts/replay_pilot_400_skillrouter_prompt.py`
- Test: `tests/test_skillrouter_prompt_alignment.py`

- [ ] **Step 1: Add provenance preservation test**

Append a unit test to `tests/test_skillrouter_prompt_alignment.py` that verifies a helper copies optional fields. Add a helper function in the replay script named `sampling_metadata(row)`.

Test content:

```python
from replay_pilot_400_skillrouter_prompt import sampling_metadata


def test_sampling_metadata_preserves_optional_input_provenance(self) -> None:
    row = {
        "sampling": {"tier": "local", "seed": 20260625},
        "panel_fingerprint": "sha256:abc",
        "panel_source_tier": "local",
    }
    self.assertEqual(
        sampling_metadata(row),
        {
            "sampling": {"tier": "local", "seed": 20260625},
            "panel_fingerprint": "sha256:abc",
            "panel_source_tier": "local",
        },
    )
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_skillrouter_prompt_alignment.py'
```

Expected:

```text
ImportError: cannot import name 'sampling_metadata'
```

- [ ] **Step 3: Implement helper and merge it into `call_single` / `call_multi` output rows**

Add to `scripts/replay_pilot_400_skillrouter_prompt.py`:

```python
def sampling_metadata(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["sampling", "panel_fingerprint", "panel_source_tier"]:
        if key in row:
            out[key] = row[key]
    return out
```

In `call_single` and `call_multi`, merge the helper into the returned result:

```python
result = {
    ...
}
result.update(sampling_metadata(row))
return result, response
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_skillrouter_prompt_alignment.py'
```

Expected:

```text
Ran 11 tests
OK
```

---

## Task 4: Add High-Confidence Exporter

**Files:**
- Create: `scripts/export_skillrouter_high_confidence.py`
- Create: `tests/test_skillrouter_high_confidence_export.py`

- [ ] **Step 1: Write exporter tests**

Create `tests/test_skillrouter_high_confidence_export.py`:

```python
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from export_skillrouter_high_confidence import export_replay_outputs  # noqa: E402


class SkillRouterHighConfidenceExportTest(unittest.TestCase):
    def test_export_splits_status_and_removes_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "replay.jsonl"
            rows = [
                {
                    "index": 1,
                    "target_skill_count": 1,
                    "generated": {"query": "A", "selected_skill_ids": ["a"]},
                    "workflow_quality_filter": {"status": "accepted", "workflow_type": "single_skill"},
                    "accepted_for_training": True,
                    "usage": {"total_tokens": 10},
                },
                {
                    "index": 2,
                    "target_skill_count": 2,
                    "generated": {"query": "B", "selected_skill_ids": ["b", "c"]},
                    "workflow_quality_filter": {"status": "weak", "workflow_type": "capability_bundle"},
                    "accepted_for_training": True,
                    "usage": {"total_tokens": 20},
                },
                {
                    "index": 3,
                    "target_skill_count": 3,
                    "generated": {"query": "C", "selected_skill_ids": ["d", "e", "f"]},
                    "workflow_quality_filter": {"status": "rejected", "workflow_type": "invalid"},
                    "accepted_for_training": False,
                    "usage": {"total_tokens": 30},
                },
            ]
            source.write_text("".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8")
            summary = export_replay_outputs(source, root / "export")
            self.assertEqual(summary["high_confidence"], 1)
            self.assertEqual(summary["medium_confidence"], 1)
            self.assertEqual(summary["rejected"], 1)

            high = json.loads((root / "export" / "high_confidence.jsonl").read_text().splitlines()[0])
            self.assertNotIn("usage", high)
            usage = json.loads((root / "export" / "usage.jsonl").read_text().splitlines()[0])
            self.assertEqual(usage["usage"]["total_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement exporter**

Create `scripts/export_skillrouter_high_confidence.py` with:

```text
export_replay_outputs(input_path, output_dir):
  Read JSONL.
  Split rows by workflow_quality_filter.status.
  Write high_confidence.jsonl for status == accepted.
  Write medium_confidence.jsonl for status == weak.
  Write rejected.clean.jsonl for status == rejected.
  Write accepted_for_training.clean.jsonl for status in accepted/weak.
  Write usage.jsonl containing index, target_skill_count, model, raw_response_path, usage.
  Remove usage from all training JSONL files.
  Write summary.json.
```

CLI:

```bash
python3 scripts/export_skillrouter_high_confidence.py \
  --input <run-output-dir>/pilot_400_skillrouter_prompt_replay.jsonl \
  --output-dir <run-output-dir>/clean_export
```

- [ ] **Step 3: Run exporter tests**

Run:

```bash
python3 -m unittest tests/test_skillrouter_high_confidence_export.py
```

Expected:

```text
Ran 1 test
OK
```

---

## Task 5: Full Local Verification

**Files:**
- Existing and newly created tests.

- [ ] **Step 1: Compile scripts**

Run:

```bash
python3 -m py_compile \
  scripts/generate_single_skill_probe.py \
  scripts/generate_multiskill_probe.py \
  scripts/replay_pilot_400_skillrouter_prompt.py \
  scripts/build_skillrouter_replay_inputs.py \
  scripts/export_skillrouter_high_confidence.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run unit tests**

Run:

```bash
python3 -m unittest \
  tests/test_skillrouter_prompt_alignment.py \
  tests/test_skillrouter_replay_input_builder.py \
  tests/test_skillrouter_high_confidence_export.py
```

Expected:

```text
OK
```

- [ ] **Step 3: Commit implementation**

Run:

```bash
git status --short
git add scripts/build_skillrouter_replay_inputs.py \
  scripts/export_skillrouter_high_confidence.py \
  scripts/replay_pilot_400_skillrouter_prompt.py \
  tests/test_skillrouter_replay_input_builder.py \
  tests/test_skillrouter_high_confidence_export.py \
  tests/test_skillrouter_prompt_alignment.py \
  docs/superpowers/plans/2026-06-25-generate-2000-high-confidence-skillrouter-data.md
git commit -m "feat: add scalable skillrouter replay input builder"
```

Expected: one commit containing only implementation, tests, and this plan.

---

## Task 6: Build 100-Row Smoke Inputs on Remote

**Files:**
- Remote output directory under `outputs/eval/`.

- [ ] **Step 1: Copy or push implementation to remote**

Use the repository workflow in effect for this workspace. If no branch push is required, sync files directly:

```bash
rsync -av \
  scripts/build_skillrouter_replay_inputs.py \
  scripts/export_skillrouter_high_confidence.py \
  scripts/replay_pilot_400_skillrouter_prompt.py \
  tests/test_skillrouter_replay_input_builder.py \
  tests/test_skillrouter_high_confidence_export.py \
  tests/test_skillrouter_prompt_alignment.py \
  gpu1_tailscale_jmp:/root/workplace/SkillSearch-data-synthesis/
```

If using Git instead, run `git pull` on remote after pushing.

- [ ] **Step 2: Verify remote tests**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
python3 -m py_compile \
  scripts/generate_single_skill_probe.py \
  scripts/generate_multiskill_probe.py \
  scripts/replay_pilot_400_skillrouter_prompt.py \
  scripts/build_skillrouter_replay_inputs.py \
  scripts/export_skillrouter_high_confidence.py
python3 -m unittest \
  tests/test_skillrouter_prompt_alignment.py \
  tests/test_skillrouter_replay_input_builder.py \
  tests/test_skillrouter_high_confidence_export.py
'
```

Expected:

```text
OK
```

- [ ] **Step 3: Build smoke inputs**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
in_dir=outputs/eval/skillrouter_2000_inputs_smoke_$(date +%Y%m%d_%H%M%S)
python3 scripts/build_skillrouter_replay_inputs.py \
  --skills outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl \
  --output-dir "$in_dir" \
  --target-counts 1:25,2:25,3:25,4:25 \
  --seed 20260625 \
  --max-skill-uses 8
cat "$in_dir/manifest.json"
'
```

Expected:

```text
input_1skill.jsonl: 25
input_2skills.jsonl: 25
input_3skills.jsonl: 25
input_4skills.jsonl: 25
panel_fingerprint_duplicates: 0
```

---

## Task 7: Run 100-Row Huayan Smoke Replay

**Files:**
- Remote smoke input directory from Task 6.
- Remote replay output directory under `outputs/eval/`.

- [ ] **Step 1: Run smoke replay**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
in_dir=<SMOKE_INPUT_DIR>
out=outputs/eval/skillrouter_2000_huayan_smoke_$(date +%Y%m%d_%H%M%S)
python3 scripts/replay_pilot_400_skillrouter_prompt.py \
  --profile profiles/gpt4o_huayanapi.json \
  --output-dir "$out" \
  --input-1 "$in_dir/input_1skill.jsonl" \
  --input-2 "$in_dir/input_2skills.jsonl" \
  --input-3 "$in_dir/input_3skills.jsonl" \
  --input-4 "$in_dir/input_4skills.jsonl" \
  --k-values 1,2,3,4 \
  --workers 4 \
  --timeout 180 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-tokens 1000
echo "$out"
'
```

- [ ] **Step 2: Export smoke split**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=<SMOKE_REPLAY_DIR>
python3 scripts/export_skillrouter_high_confidence.py \
  --input "$out/pilot_400_skillrouter_prompt_replay.jsonl" \
  --output-dir "$out/clean_export"
cat "$out/clean_export/summary.json"
'
```

Expected minimum acceptance gate:

```text
high_confidence >= 85 / 100
rejected <= 5 / 100
selected_id_not_in_panel <= 2 / 100
```

- [ ] **Step 3: Inspect smoke examples**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=<SMOKE_REPLAY_DIR>
jq -r "
  select(.workflow_quality_filter.status != \"accepted\") |
  [.target_skill_count, .workflow_quality_filter.status, (.workflow_quality_filter.reject_reasons // [] | join(\";\")), (.workflow_quality_filter.weak_reasons // [] | join(\";\")), .generated.query] |
  @tsv
" "$out/pilot_400_skillrouter_prompt_replay.jsonl" | sed -n "1,40p"
'
```

Decision rule:

```text
If high_confidence >= 85%, proceed to full run.
If selected_id_not_in_panel repeats, inspect parser/repair before full run.
If weak mostly duplicate_operation_type or low_role_diversity, raise panel role/type diversity constraints.
If query style is too checklist-like, reduce candidate panel size for k=2/3 or add targeted retry later.
```

---

## Task 8: Build Full 2325 Raw Inputs

**Files:**
- Remote full input directory.

- [ ] **Step 1: Build full inputs**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
in_dir=outputs/eval/skillrouter_2000_inputs_full_$(date +%Y%m%d_%H%M%S)
python3 scripts/build_skillrouter_replay_inputs.py \
  --skills outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl \
  --output-dir "$in_dir" \
  --target-counts 1:525,2:625,3:550,4:625 \
  --seed 20260625 \
  --max-skill-uses 8
cat "$in_dir/manifest.json"
'
```

Expected:

```text
target_counts_written: {"1": 525, "2": 625, "3": 550, "4": 625}
panel_fingerprint_duplicates: 0
candidate_skill_use_max <= 8
```

- [ ] **Step 2: Preflight counts**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
in_dir=<FULL_INPUT_DIR>
wc -l "$in_dir"/input_*skill*.jsonl
'
```

Expected:

```text
525 input_1skill.jsonl
625 input_2skills.jsonl
550 input_3skills.jsonl
625 input_4skills.jsonl
2325 total
```

---

## Task 9: Run Full Huayan Replay

**Files:**
- Remote full replay output directory.

- [ ] **Step 1: Start full replay**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
in_dir=<FULL_INPUT_DIR>
out=outputs/eval/skillrouter_2000_huayan_full_$(date +%Y%m%d_%H%M%S)
python3 scripts/replay_pilot_400_skillrouter_prompt.py \
  --profile profiles/gpt4o_huayanapi.json \
  --output-dir "$out" \
  --input-1 "$in_dir/input_1skill.jsonl" \
  --input-2 "$in_dir/input_2skills.jsonl" \
  --input-3 "$in_dir/input_3skills.jsonl" \
  --input-4 "$in_dir/input_4skills.jsonl" \
  --k-values 1,2,3,4 \
  --workers 4 \
  --timeout 180 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-tokens 1000
echo "$out"
'
```

- [ ] **Step 2: Resume command if interrupted**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
in_dir=<FULL_INPUT_DIR>
out=<FULL_REPLAY_DIR>
python3 scripts/replay_pilot_400_skillrouter_prompt.py \
  --profile profiles/gpt4o_huayanapi.json \
  --output-dir "$out" \
  --input-1 "$in_dir/input_1skill.jsonl" \
  --input-2 "$in_dir/input_2skills.jsonl" \
  --input-3 "$in_dir/input_3skills.jsonl" \
  --input-4 "$in_dir/input_4skills.jsonl" \
  --k-values 1,2,3,4 \
  --workers 4 \
  --timeout 180 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-tokens 1000 \
  --resume
'
```

- [ ] **Step 3: Monitor status while running**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=<FULL_REPLAY_DIR>
wc -l "$out/pilot_400_skillrouter_prompt_replay.jsonl" "$out/accepted.jsonl" "$out/rejected.jsonl"
jq -r ".workflow_quality_filter.status" "$out/pilot_400_skillrouter_prompt_replay.jsonl" | sort | uniq -c
'
```

Expected after completion:

```text
2325 pilot_400_skillrouter_prompt_replay.jsonl
```

---

## Task 10: Export High-Confidence Dataset and Audit

**Files:**
- Remote full replay output directory.

- [ ] **Step 1: Export clean split**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=<FULL_REPLAY_DIR>
python3 scripts/export_skillrouter_high_confidence.py \
  --input "$out/pilot_400_skillrouter_prompt_replay.jsonl" \
  --output-dir "$out/clean_export"
cat "$out/clean_export/summary.json"
'
```

Expected:

```text
high_confidence >= 2000
medium_confidence tracked separately
rejected tracked separately
```

- [ ] **Step 2: Check duplicates**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=<FULL_REPLAY_DIR>
jq -r ".generated.selected_skill_ids | sort | join(\",\")" "$out/clean_export/high_confidence.jsonl" \
  | sort | uniq -d | sed -n "1,20p"
jq -r ".generated.query" "$out/clean_export/high_confidence.jsonl" \
  | sort | uniq -d | sed -n "1,20p"
'
```

Expected: no duplicate selected-skill sets and no exact duplicate queries. If duplicates exist, remove duplicates from high-confidence export and run top-up.

- [ ] **Step 3: Manual spot audit**

Sample 10 per k:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=<FULL_REPLAY_DIR>
for k in 1 2 3 4; do
  echo "===== k=$k"
  jq -c "select(.target_skill_count == $k)" "$out/clean_export/high_confidence.jsonl" \
    | shuf -n 10 \
    | jq -r "[.target_skill_count, (.generated.selected_skill_ids | join(\",\")), .generated.query] | @tsv"
done
'
```

Reject full run only if the audit shows a systemic issue such as many generated queries mentioning skill names, candidate panels, operation labels, or implausible mixed technology stacks.

---

## Task 11: Top-Up If High-Confidence < 2000

**Files:**
- Existing full input and output directories.
- New top-up input and output directories.

- [ ] **Step 1: Compute missing count**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
out=<FULL_REPLAY_DIR>
high=$(wc -l < "$out/clean_export/high_confidence.jsonl")
echo "high=$high missing=$((2000 - high))"
'
```

- [ ] **Step 2: Build top-up inputs**

If missing is positive, build a 1.25x top-up distributed toward the weak k values. Example for 150 missing:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
topup=outputs/eval/skillrouter_2000_inputs_topup_$(date +%Y%m%d_%H%M%S)
python3 scripts/build_skillrouter_replay_inputs.py \
  --skills outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl \
  --output-dir "$topup" \
  --target-counts 1:25,2:60,3:35,4:70 \
  --seed 20260626 \
  --max-skill-uses 8 \
  --exclude-panel-fingerprints <FULL_INPUT_DIR>/manifest.json
cat "$topup/manifest.json"
'
```

The builder must support `--exclude-panel-fingerprints` before this step is used. If not implemented in Task 2, add it before top-up.

- [ ] **Step 3: Replay top-up and merge high-confidence**

Run the same replay/export commands with the top-up inputs. Merge clean high-confidence files only after deduplication:

```bash
jq -s -c '
  flatten
  | unique_by((.generated.selected_skill_ids | sort | join(",")) + "\t" + .generated.query)
  | .[]
' <FULL_REPLAY_DIR>/clean_export/high_confidence.jsonl <TOPUP_REPLAY_DIR>/clean_export/high_confidence.jsonl \
  > <FULL_REPLAY_DIR>/clean_export/high_confidence_merged.jsonl
```

Expected:

```text
wc -l high_confidence_merged.jsonl >= 2000
```

---

## Task 12: Final Deliverables

**Files:**
- Remote final export directory.

- [ ] **Step 1: Produce final dataset files**

Final directory should contain:

```text
high_confidence.jsonl
medium_confidence.jsonl
rejected.clean.jsonl
accepted_for_training.clean.jsonl
usage.jsonl
summary.json
manifest.json
```

- [ ] **Step 2: Record final command log**

Add a `RUNBOOK.md` inside the output directory with:

```text
input builder command
replay command
resume command
export command
profile name
model
date
final counts by k and status
known limitations
```

- [ ] **Step 3: Final verification**

Run:

```bash
ssh -o ExitOnForwardFailure=no gpu1_tailscale_jmp '
cd /root/workplace/SkillSearch-data-synthesis
final=<FINAL_EXPORT_DIR>
wc -l "$final/high_confidence.jsonl" "$final/medium_confidence.jsonl" "$final/rejected.clean.jsonl" "$final/usage.jsonl"
jq -r ".workflow_quality_filter.status" "$final/high_confidence.jsonl" | sort | uniq -c
jq "del(.usage)" "$final/high_confidence.jsonl" >/tmp/high_confidence_parse_check.json
'
```

Expected:

```text
high_confidence.jsonl >= 2000
status distribution in high_confidence: accepted only
no JSON parse errors
```

---

## Self-Review

Spec coverage:

```text
2000 high-confidence data target: Task 8-12
Panel construction for scaling: Task 2
Quality split accepted/weak/rejected: Task 4 and Task 10
Reuse current frozen prompt and replay logic: Task 3 and Task 9
Huayan endpoint usage: Task 7 and Task 9
Future scaling path to tens of thousands: builder enumerates all anchor buckets and uses quotas/dedup
```

Known limitations:

```text
The first implementation uses deterministic lexical/panel fingerprint deduplication, not embedding-level semantic deduplication.
For tens of thousands of rows, add query embedding or MinHash near-duplicate filtering after the 2000-row run.
The builder should initially prefer local high-precision buckets; balanced buckets are fallback expansion, not the first quality tier.
```

Execution choice after this plan:

```text
1. Subagent-Driven: implement task-by-task with review checkpoints.
2. Inline Execution: implement in this session with checkpoints.
```
