from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_diverse_coverage_queries import (
    COVERAGE_MODES,
    build_coverage_messages,
    build_generation_record,
    collect_existing_views,
    materialize,
    normalize_coverage_query_layout,
)


def benchmark_query(theme: str) -> str:
    return (
        f"A {theme} workflow has an input artifact that must be processed into a reliable deliverable for an internal review. "
        f"The request uses the {theme} scenario to exercise a distinct supported operation while keeping all specialized values "
        "as user-provided placeholders.\n\n"
        "You need to:\n"
        f"1. Inspect the available input and identify the information required for the {theme} operation.\n"
        f"2. Perform the supported {theme} transformation without adding an unrelated specialized capability.\n"
        "3. Save the result in the requested location and check that the expected fields are present.\n\n"
        f"Deliverable: a completed {theme} result ready for review. Acceptance criterion: the output is internally consistent, "
        "contains the requested information, and passes the stated structural check."
    )


def generated_items() -> list[dict[str, str]]:
    themes = {
        "implicit_outcome": "forecast handoff",
        "artifact_pipeline": "record conversion",
        "workflow_stage": "release checkpoint",
        "failure_recovery": "damaged input recovery",
        "verification_first": "compliance validation",
    }
    items = [
        {"mode": mode, "query": benchmark_query(themes[mode])}
        for mode in COVERAGE_MODES
        if mode != "terse_natural"
    ]
    items.append(
        {
            "mode": "terse_natural",
            "query": (
                "Please process the supplied artifact into the requested review output and preserve the user-provided values. "
                "Save the result to the stated destination, then confirm that its required fields are present and consistent."
            ),
        }
    )
    return items


class DiverseCoverageQueriesTest(unittest.TestCase):
    def test_collect_existing_views_keeps_five_plus_three_and_drops_alias(self) -> None:
        target = "skill/a"
        rows = []
        for rank in range(5):
            rows.append(
                {
                    "index": f"pseudo:catalog_task:{target}:{rank}",
                    "query": f"catalog query {rank}",
                    "selected_skill_ids": [target],
                    "positive_skill_id": target,
                }
            )
        for mode in ("outcome_first", "workflow_embedded", "constraint_failure"):
            rows.append(
                {
                    "index": f"pseudo:zero_multi_mode:{mode}:{target}",
                    "query": f"mode query {mode}",
                    "selected_skill_ids": [target],
                    "positive_skill_id": target,
                }
            )
        rows.append(
            {
                "index": "pseudo:catalog_task:alias/a:0",
                "query": "alias query",
                "selected_skill_ids": [target],
                "positive_skill_id": target,
            }
        )
        queries, indices, aliases = collect_existing_views(
            rows,
            [target],
            task5_per_skill=5,
            mode3_per_skill=3,
        )
        self.assertEqual(len(queries[target]), 8)
        self.assertEqual(len(indices[target]), 8)
        self.assertEqual(aliases, 1)
        self.assertNotIn("alias query", queries[target])

    def test_prompt_contains_all_modes_and_existing_views(self) -> None:
        skill = {"skill_id": "skill/a", "name": "Private capability", "body": "Supported body."}
        messages = build_coverage_messages(skill, [f"existing query {index}" for index in range(8)], body_chars=2500)
        prompt = messages[0]["content"]
        for mode in COVERAGE_MODES:
            self.assertIn(f"`{mode}`", prompt)
        self.assertIn("Existing view 8: existing query 7", prompt)

    def test_layout_normalizer_inserts_marker_before_adjacent_numbered_list(self) -> None:
        query = "A concrete input must be processed into a verified output. To complete it: 1. Inspect the input. 2. Save the output."
        normalized = normalize_coverage_query_layout(query, "artifact_pipeline")
        self.assertIn("\n\nYou need to:\n1. Inspect the input.", normalized)
        self.assertIn("\n2. Save the output.", normalized)

    def test_generation_record_and_materialization_have_exact_mode_closure(self) -> None:
        target = "skill/a"
        skill = {"skill_id": target, "name": "Private capability", "body": "Process an input into an output."}
        response = {
            "model": "gpt-test",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "choices": [{"message": {"content": json.dumps(generated_items())}}],
        }
        existing = [f"unrelated existing training view {index}" for index in range(8)]
        record = build_generation_record(
            target=target,
            skill=skill,
            existing_queries=existing,
            response=response,
            max_query_chars=1500,
            existing_jaccard_threshold=1.0,
            pairwise_jaccard_threshold=1.0,
        )
        self.assertEqual(record["status"], "complete", record["reject_reasons"])
        self.assertEqual([row["mode"] for row in record["queries"]], list(COVERAGE_MODES))

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "coverage_generations.jsonl").write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            manifest = materialize(
                output_dir=output_dir,
                targets=[target],
                existing_indices={target: [f"source-{index}" for index in range(8)]},
                profile={"model": "gpt-test", "api_key": "secret"},
                run_config={"schema_version": 1},
                max_attempts=3,
            )
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["selected_rows"], 6)
            selected = [
                json.loads(line)
                for line in (output_dir / "selected_diverse_coverage.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["pseudo_query"]["mode"] for row in selected],
                list(COVERAGE_MODES),
            )
            self.assertNotIn("secret", (output_dir / "manifest.json").read_text(encoding="utf-8"))

    def test_generation_record_rejects_existing_query_overlap(self) -> None:
        target = "skill/a"
        skill = {"skill_id": target, "name": "Private capability", "body": "Process an input into an output."}
        items = generated_items()
        response = {"choices": [{"message": {"content": json.dumps(items)}}]}
        record = build_generation_record(
            target=target,
            skill=skill,
            existing_queries=[items[0]["query"]],
            response=response,
            max_query_chars=1500,
            existing_jaccard_threshold=0.78,
            pairwise_jaccard_threshold=1.0,
        )
        self.assertEqual(record["status"], "invalid_generation")
        self.assertTrue(any("existing_jaccard" in reason for reason in record["reject_reasons"]))


if __name__ == "__main__":
    unittest.main()
