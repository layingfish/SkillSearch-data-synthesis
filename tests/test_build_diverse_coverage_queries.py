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
    file_sha256,
    filter_mined_errors,
    materialize,
    normalize_coverage_query_layout,
    prepare_mining_rows,
    query_digest,
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

    def test_natural_non_terse_query_is_valid_without_numbered_layout(self) -> None:
        target = "skill/a"
        skill = {"skill_id": target, "name": "Private capability", "body": "Process an input into an output."}
        items = generated_items()
        items[0]["query"] = (
            "A review team has a supplied input artifact and needs a reliable processed result for a scheduled handoff. "
            "Inspect the available content, carry out the supported transformation, preserve every user-provided value, and save "
            "the result at the requested destination. Confirm that the output contains the expected information and remains "
            "internally consistent before reporting completion to the reviewer."
        )
        response = {"choices": [{"message": {"content": json.dumps(items)}}]}
        record = build_generation_record(
            target=target,
            skill=skill,
            existing_queries=["an unrelated training query"],
            response=response,
            max_query_chars=1500,
            existing_jaccard_threshold=1.0,
            pairwise_jaccard_threshold=1.0,
        )
        self.assertEqual(record["status"], "complete", record["reject_reasons"])
        self.assertFalse(record["queries"][0]["has_task_marker"])
        self.assertFalse(record["queries"][0]["has_numbered_steps"])

    def test_prepare_and_filter_mining_preserve_exact_query_binding(self) -> None:
        targets = ["skill/a", "skill/b"]
        selected = []
        for target in targets:
            for rank, mode in enumerate(COVERAGE_MODES):
                query = f"query for {target} in {mode} with unique ordinal {rank}"
                selected.append(
                    {
                        "index": f"pseudo:diverse_coverage:{target}:{mode}",
                        "source": "training_gr_diverse_coverage_pseudo_query",
                        "query": query,
                        "query_text": f"instruction\nQuery:{query}",
                        "selected_skill_ids": [target],
                        "positive_skill_id": target,
                        "pseudo_query": {"mode": mode},
                    }
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected_path = root / "selected.jsonl"
            selected_path.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected),
                encoding="utf-8",
            )
            generation_manifest_path = root / "generation_manifest.json"
            generation_manifest_path.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "target_skill_count": 2,
                        "selected_rows": 12,
                        "modes": list(COVERAGE_MODES),
                        "selected_rows_sha256": file_sha256(selected_path),
                    }
                ),
                encoding="utf-8",
            )
            mining_dir = root / "mining_input"
            mining_manifest = prepare_mining_rows(
                selected_rows_path=selected_path,
                generation_manifest_path=generation_manifest_path,
                output_dir=mining_dir,
                expected_targets=2,
                views_per_target=6,
            )
            mining_rows_path = Path(mining_manifest["output"])
            mining_rows = [json.loads(line) for line in mining_rows_path.read_text().splitlines()]
            self.assertEqual([row["index"] for row in mining_rows], list(range(12)))
            self.assertEqual(mining_rows[0]["source_index"], selected[0]["index"])

            candidates = []
            predictions = []
            for row in mining_rows:
                row_index = row["index"]
                positive = row["positive_skill_id"]
                negatives = [f"negative/{row_index}/{rank}" for rank in range(10)]
                ranked = [positive, *negatives] if row_index % 2 == 0 else [negatives[0], positive, *negatives[1:]]
                candidates.append(
                    {
                        "row_index": row_index,
                        "source_index": row["source_index"],
                        "query_sha256": query_digest(row),
                        "positive_skill_id": positive,
                        "negative_skill_ids": negatives,
                    }
                )
                predictions.append(
                    {
                        "index": row_index,
                        "query": row["query"],
                        "gold_skill_ids": [positive],
                        "ranked_skill_ids": ranked,
                        "invalid": False,
                    }
                )
            candidates_path = root / "candidates.jsonl"
            candidates_path.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in candidates),
                encoding="utf-8",
            )
            predictions_path = root / "predictions.jsonl"
            predictions_path.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in predictions),
                encoding="utf-8",
            )
            checkpoint_sha = "a" * 64
            candidate_manifest_path = root / "candidate_manifest.json"
            candidate_manifest_path.write_text(
                json.dumps(
                    {
                        "rows_sha256": file_sha256(mining_rows_path),
                        "output_sha256": file_sha256(candidates_path),
                        "mining_checkpoint_sha256": checkpoint_sha,
                        "mining_protocol": {"num_beams": 50},
                        "prediction_shards": [
                            {"predictions_sha256": file_sha256(predictions_path)}
                        ],
                        "audit": {
                            "rows": 12,
                            "training_rows": 12,
                            "negatives_per_row": 10,
                            "known_unique_non_gold_negatives": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            filtered_manifest = filter_mined_errors(
                mining_rows_path=mining_rows_path,
                candidates_path=candidates_path,
                candidate_manifest_path=candidate_manifest_path,
                prediction_paths=[predictions_path],
                output_dir=root / "filtered",
                expected_rows=12,
                expected_targets=2,
                views_per_target=6,
                expected_negatives=10,
                expected_beams=50,
                expected_checkpoint_sha256=checkpoint_sha,
            )
            self.assertEqual(filtered_manifest["error_rows"], 6)
            self.assertEqual(filtered_manifest["targets_with_errors"], 2)
            error_candidates = [
                json.loads(line)
                for line in Path(filtered_manifest["outputs"]["error_candidates"])
                .read_text()
                .splitlines()
            ]
            self.assertEqual([row["row_index"] for row in error_candidates], list(range(6)))
            self.assertTrue(
                all(row["top1_confuser_skill_id"] == row["negative_skill_ids"][0] for row in error_candidates)
            )


if __name__ == "__main__":
    unittest.main()
