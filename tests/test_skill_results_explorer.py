from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_skill_results_explorer import build_explorer_payload, render_html  # noqa: E402


class SkillResultsExplorerTest(unittest.TestCase):
    def test_build_payload_computes_cutoff_coverage_and_skill_metadata(self) -> None:
        rows = [
            {
                "index": 0,
                "query": "Need a mesh mass calculation tool.",
                "selected_skill_ids": ["gt/mesh", "gt/csv"],
                "target_skill_count": 2,
                "domain": "engineering",
                "difficulty": "hard",
            },
            {
                "index": 1,
                "query": "Need a missing catalog skill.",
                "selected_skill_ids": ["gt/missing"],
                "target_skill_count": 1,
                "domain": "data",
                "difficulty": "medium",
            },
        ]
        skills = [
            {
                "skill_id": "gt/mesh",
                "name": "mesh-short-name",
                "description": '["engineering"]',
                "body": (
                    "---\n"
                    "name: mesh-analysis\n"
                    "description: Analyze meshes and compute volumes.\n"
                    "---\n"
                    "# Mesh Analysis\n"
                    "Compute volume and connected components."
                ),
            },
            {
                "skill_id": "gt/csv",
                "name": "csv-short-name",
                "description": '["data"]',
                "body": (
                    "---\n"
                    "name: csv-processing\n"
                    "description: Read and write CSV files.\n"
                    "---\n"
                    "# CSV Processing\n"
                    "Normalize tabular files."
                ),
            },
        ]
        runs = [
            {
                "name": "baseline",
                "predictions": [
                    {"index": 0, "ranked_skill_ids": ["gt/mesh", "other/x", "gt/csv"]},
                    {"index": 1, "ranked_skill_ids": ["other/y"]},
                ],
            }
        ]

        payload = build_explorer_payload(rows, skills, runs, cutoffs=[1, 2])

        self.assertEqual(payload["summary"]["query_count"], 2)
        self.assertEqual([run["name"] for run in payload["runs"]], ["baseline"])

        first = payload["queries"][0]
        self.assertEqual(first["relevant_count"], 2)
        self.assertEqual(first["coverage_by_cutoff"]["1"]["covered_any"], 1)
        self.assertEqual(first["coverage_by_cutoff"]["1"]["missed_any"], 1)
        self.assertEqual(first["coverage_by_cutoff"]["2"]["covered_any"], 1)
        self.assertEqual(first["skills"][0]["name"], "mesh-analysis")
        self.assertEqual(first["skills"][0]["description"], "Analyze meshes and compute volumes.")
        self.assertEqual(first["skills"][0]["runs"]["baseline"]["rank"], 1)
        self.assertEqual(first["skills"][1]["runs"]["baseline"]["rank"], 3)
        self.assertIsNone(first["skills"][1]["runs"]["baseline"]["hit_at"]["2"])
        self.assertEqual(first["metrics_by_run"]["baseline"]["1"]["hit_count"], 1)
        self.assertEqual(first["metrics_by_run"]["baseline"]["1"]["hit"], 1)
        self.assertAlmostEqual(first["metrics_by_run"]["baseline"]["1"]["recall"], 0.5)
        self.assertEqual(first["metrics_by_run"]["baseline"]["1"]["full_coverage"], 0)

        second = payload["queries"][1]
        self.assertEqual(second["skills"][0]["name"], "gt/missing")
        self.assertEqual(second["skills"][0]["description"], "")
        self.assertEqual(second["coverage_by_cutoff"]["2"]["missed_any"], 1)

        run_metrics = payload["runs"][0]["metrics_by_cutoff"]["1"]
        self.assertEqual(run_metrics["hit_queries"], 1)
        self.assertEqual(run_metrics["full_coverage_queries"], 0)
        self.assertEqual(run_metrics["query_count"], 2)
        self.assertEqual(run_metrics["total_hits"], 1)
        self.assertEqual(run_metrics["total_gold"], 3)
        self.assertAlmostEqual(run_metrics["hit"], 0.5)
        self.assertAlmostEqual(run_metrics["full_coverage"], 0.0)
        self.assertAlmostEqual(run_metrics["recall"], 0.25)
        self.assertAlmostEqual(run_metrics["macro_recall"], 0.25)
        self.assertAlmostEqual(run_metrics["micro_recall"], 1 / 3)
        self.assertEqual(payload["summary"]["recall_aggregation"], "macro")

    def test_multiple_runs_are_kept_separate_and_switchable(self) -> None:
        payload = build_explorer_payload(
            rows=[{"index": 0, "query": "query", "selected_skill_ids": ["gt/a"]}],
            skills=[],
            runs=[
                {
                    "name": "first",
                    "predictions": [{"index": 0, "ranked_skill_ids": ["gt/a"]}],
                },
                {
                    "name": "second",
                    "predictions": [{"index": 0, "ranked_skill_ids": ["other/x"]}],
                },
            ],
            cutoffs=[1],
        )

        self.assertEqual(payload["runs"][0]["metrics_by_cutoff"]["1"]["hit"], 1.0)
        self.assertEqual(payload["runs"][1]["metrics_by_cutoff"]["1"]["hit"], 0.0)
        self.assertEqual(payload["runs"][0]["metrics_by_cutoff"]["1"]["full_coverage"], 1.0)
        self.assertEqual(payload["runs"][1]["metrics_by_cutoff"]["1"]["full_coverage"], 0.0)
        self.assertEqual(payload["queries"][0]["metrics_by_run"]["first"]["1"]["hit"], 1)
        self.assertEqual(payload["queries"][0]["metrics_by_run"]["second"]["1"]["hit"], 0)

        html = render_html(payload, default_cutoff=1)
        self.assertIn("run-tab-list", html)
        self.assertIn("grid-template-columns: repeat(auto-fit, minmax(160px, 1fr))", html)
        self.assertIn("activeRunId", html)
        self.assertIn("one active run at a time", html)
        self.assertIn("first", html)
        self.assertIn("macro averaged", html)
        self.assertIn('<math class="formula-math"', html)
        self.assertIn("Macro Recall gives every query equal weight", html)
        self.assertIn("<munderover><mo>&sum;</mo>", html)
        self.assertIn("second", html)
        self.assertNotIn("comparison column", html)
        self.assertNotIn("selectedRuns", html)

    def test_strict_metadata_rejects_incomplete_catalog(self) -> None:
        with self.assertRaisesRegex(ValueError, "pass the complete skill catalog"):
            build_explorer_payload(
                rows=[{"index": 0, "query": "query", "selected_skill_ids": ["gt/missing"]}],
                skills=[],
                runs=[
                    {
                        "name": "baseline",
                        "predictions": [{"index": 0, "ranked_skill_ids": ["gt/missing"]}],
                    }
                ],
                cutoffs=[1],
                strict_skill_metadata=True,
            )

    def test_rendered_skill_detail_shows_original_metadata_and_semantic_id(self) -> None:
        payload = build_explorer_payload(
            rows=[
                {
                    "index": 0,
                    "query": "Need a mesh mass calculation tool.\nPreserve this second line.",
                    "selected_skill_ids": ["gt/mesh"],
                }
            ],
            skills=[
                {
                    "skill_id": "gt/mesh",
                    "name": "top-level-category-name",
                    "description": '["engineering"]',
                    "body": (
                        "---\n"
                        "name: Original Mesh Name\n"
                        "description: Original mesh description.\n"
                        "---\n"
                        "# Body Title\n"
                        "Extra body text that should not be rendered."
                    ),
                }
            ],
            runs=[
                {
                    "name": "baseline",
                    "predictions": [{"index": 0, "ranked_skill_ids": ["gt/mesh"]}],
                }
            ],
            cutoffs=[1],
            semantic_ids=[
                {
                    "skill_id": "gt/mesh",
                    "semantic_id": [127, 41, 4, 0],
                    "tokens": ["<RQ0_127>", "<RQ1_041>", "<RQ2_004>", "<DUP_000>"],
                }
            ],
        )

        html = render_html(payload, default_cutoff=1)

        self.assertIn("Name</div>", html)
        self.assertIn("Description</div>", html)
        self.assertIn("RQ semantic codes</div>", html)
        self.assertIn("semanticIdData", html)
        self.assertIn("learnedSemanticComponents", html)
        self.assertIn("semantic-id-components", html)
        self.assertIn("RQ${index}=${value}", html)
        self.assertIn('!component.token.startsWith("<DUP_")', html)
        self.assertNotIn("values.slice(0, -1)", html)
        self.assertEqual(payload["semantic_ids"]["gt/mesh"]["values"], [127, 41, 4, 0])
        self.assertEqual(payload["semantic_ids"]["gt/mesh"]["tokens"][0], "<RQ0_127>")
        self.assertIn("Original Mesh Name", html)
        self.assertIn("Original mesh description.", html)
        self.assertNotIn("top-level-category-name", html)
        self.assertNotIn("[&quot;engineering&quot;]", html)
        self.assertNotIn("Skill ID</div>", html)
        self.assertNotIn("Body Preview", html)
        self.assertNotIn("missing from catalog", html)
        self.assertNotIn("Extra body text that should not be rendered.", html)
        self.assertIn("white-space: pre-wrap", html)
        self.assertIn("Preserve this second line.", html)

    def test_payload_and_html_include_top_k_ranked_results_with_gt_marker(self) -> None:
        payload = build_explorer_payload(
            rows=[
                {
                    "index": 0,
                    "query": "Need a mesh mass calculation tool.",
                    "selected_skill_ids": ["gt/mesh"],
                }
            ],
            skills=[
                {
                    "skill_id": "gt/mesh",
                    "body": (
                        "---\n"
                        "name: mesh-analysis\n"
                        "description: Analyze meshes and compute volumes.\n"
                        "---\n"
                    ),
                },
                {
                    "skill_id": "other/x",
                    "body": (
                        "---\n"
                        "name: other-skill\n"
                        "description: Not relevant.\n"
                        "---\n"
                    ),
                },
            ],
            runs=[
                {
                    "name": "baseline",
                    "predictions": [{"index": 0, "ranked_skill_ids": ["gt/mesh", "other/x"]}],
                }
            ],
            cutoffs=[1, 10, 50],
        )

        ranked = payload["queries"][0]["ranked_results"]["baseline"]

        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["skill_id"], "gt/mesh")
        self.assertTrue(ranked[0]["is_gt"])
        self.assertEqual(ranked[0]["name"], "mesh-analysis")
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertFalse(ranked[1]["is_gt"])

        html = render_html(payload, default_cutoff=50)

        self.assertEqual(payload["summary"]["cutoffs"], [1, 10, 50])
        self.assertIn("result-mode-buttons", html)
        self.assertIn("renderGroundTruthResults", html)
        self.assertIn("renderTopKResults", html)
        self.assertIn('item.is_gt ? "gt" : "non-gt"', html)
        self.assertIn(".result-row.gt", html)
        self.assertIn(".result-row.non-gt", html)
        self.assertIn("Hit, Recall, and Full Coverage formulas", html)
        self.assertIn("FC@k", html)
        self.assertIn("Entire run", html)
        self.assertNotIn("comparison column", html)
        self.assertNotIn('id="run-list"', html)
        self.assertNotIn("selectedRuns", html)


if __name__ == "__main__":
    unittest.main()
