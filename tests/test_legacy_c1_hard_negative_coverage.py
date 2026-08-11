import unittest

from scripts.analyze_legacy_c1_hard_negative_coverage import (
    analyze,
    analyze_teacher_prefix,
    first_divergence,
)


class LegacyC1HardNegativeCoverageTest(unittest.TestCase):
    def setUp(self):
        self.semantic_ids = {
            "gold-a": (1, 2, 3, 4),
            "gold-b": (1, 2, 8, 9),
            "neg-c0": (9, 0, 0, 0),
            "neg-c0b": (8, 0, 0, 0),
            "neg-c0c": (7, 0, 0, 0),
            "neg-c1": (1, 7, 0, 0),
            "neg-c2": (1, 2, 7, 0),
            "neg-c3": (1, 2, 3, 7),
            "collision": (1, 2, 3, 4),
        }
        self.source_rows = [
            {
                "index": 0,
                "query": "query zero",
                "selected_skill_ids": ["gold-a", "gold-b"],
            },
            {
                "index": 1,
                "query": "query one",
                "selected_skill_ids": ["gold-a"],
            },
        ]
        self.teacher_rows = [
            {
                "index": 0,
                "query": "query zero",
                "gold_skill_ids": ["gold-a", "gold-b"],
                "ranked_skill_ids": [
                    "gold-b",
                    "neg-c0",
                    "neg-c1",
                    "neg-c2",
                    "neg-c3",
                    "gold-a",
                ],
                "ranked_scores": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                "invalid": False,
            },
            {
                "index": 1,
                "query": "query one",
                "gold_skill_ids": ["gold-a"],
                "ranked_skill_ids": [
                    "neg-c0",
                    "collision",
                    "gold-a",
                    "neg-c0b",
                    "neg-c0c",
                ],
                "ranked_scores": [0.8, 0.7, 0.6, 0.5, 0.4],
                "invalid": False,
            },
        ]

    def test_first_divergence(self):
        self.assertEqual(first_divergence((1, 2, 3, 4), (1, 7, 3, 4)), 1)
        self.assertIsNone(first_divergence((1, 2), (1, 2)))

    def test_c1_coverage_excludes_sibling_gt(self):
        summary, manifest = analyze(
            self.source_rows,
            self.teacher_rows,
            self.semantic_ids,
            top_ks=(1, 3, 5),
            candidate_limit=5,
        )
        self.assertTrue(summary["verification"]["passed"])
        self.assertEqual(summary["counts"]["gold_mentions"], 3)
        self.assertEqual(summary["counts"]["selected_c1_gold_mentions"], 2)
        c1_at_3 = summary["coverage_by_top_k"]["3"]["first_divergence_c1"]
        self.assertEqual(c1_at_3["covered_gold_mentions"], 2)
        self.assertEqual(c1_at_3["rows_with_all_golds_covered"], 1)
        selected = manifest[0]["gold_candidates"][0]["selected_c1_negative"]
        self.assertEqual(selected["skill_id"], "neg-c1")
        self.assertNotEqual(selected["skill_id"], "gold-b")

    def test_structural_c1_exists_when_teacher_lacks_it(self):
        summary, manifest = analyze(
            self.source_rows,
            self.teacher_rows,
            self.semantic_ids,
            top_ks=(3,),
            candidate_limit=5,
        )
        row_one = manifest[1]["gold_candidates"][0]
        self.assertIsNone(row_one["selected_c1_negative"])
        self.assertGreater(row_one["structural_c1_negative_count"], 0)
        self.assertEqual(summary["counts"]["gold_mentions_with_teacher_top50_code_collision"], 1)

    def test_rejects_alignment_drift(self):
        bad = [dict(row) for row in self.teacher_rows]
        bad[1] = dict(bad[1], query="different")
        with self.assertRaisesRegex(ValueError, "query mismatch"):
            analyze(
                self.source_rows,
                bad,
                self.semantic_ids,
                top_ks=(1,),
                candidate_limit=5,
            )

    def test_rejects_non_monotonic_teacher_scores(self):
        bad = [dict(row) for row in self.teacher_rows]
        bad[0] = dict(bad[0], ranked_scores=[0.9, 0.8, 0.85, 0.6, 0.5, 0.4])
        with self.assertRaisesRegex(ValueError, "not non-increasing"):
            analyze(
                self.source_rows,
                bad,
                self.semantic_ids,
                top_ks=(1,),
                candidate_limit=5,
            )

    def test_allows_unsorted_appended_gold_scores(self):
        teacher = [dict(row) for row in self.teacher_rows]
        teacher[0] = dict(
            teacher[0],
            ranked_scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.85],
        )
        summary, _ = analyze(
            self.source_rows,
            teacher,
            self.semantic_ids,
            top_ks=(5,),
            candidate_limit=5,
        )
        self.assertTrue(summary["verification"]["teacher_ranking_valid"])

    def test_rejects_short_retrieval_domain(self):
        short = [dict(row) for row in self.teacher_rows]
        short[1] = dict(
            short[1],
            ranked_skill_ids=short[1]["ranked_skill_ids"][:4],
            ranked_scores=short[1]["ranked_scores"][:4],
        )
        with self.assertRaisesRegex(ValueError, "fewer than candidate_limit"):
            analyze(
                self.source_rows,
                short,
                self.semantic_ids,
                top_ks=(5,),
                candidate_limit=5,
            )

    def test_rejects_non_gold_suffix(self):
        bad = [dict(row) for row in self.teacher_rows]
        bad[0] = dict(
            bad[0],
            ranked_skill_ids=[*bad[0]["ranked_skill_ids"], "neg-c0b"],
            ranked_scores=[*bad[0]["ranked_scores"], 0.3],
        )
        with self.assertRaisesRegex(ValueError, "suffix is not exactly"):
            analyze(
                self.source_rows,
                bad,
                self.semantic_ids,
                top_ks=(5,),
                candidate_limit=5,
            )

    def test_accepts_string_indices(self):
        source = [dict(self.source_rows[0], index="pseudo:skill:0")]
        teacher = [dict(self.teacher_rows[0], index="pseudo:skill:0")]
        summary, manifest = analyze(
            source,
            teacher,
            self.semantic_ids,
            top_ks=(5,),
            candidate_limit=5,
        )
        self.assertEqual(summary["counts"]["rows"], 1)
        self.assertEqual(manifest[0]["index"], "pseudo:skill:0")

    def test_exact_teacher_prefix_allows_audited_trailing_rows(self):
        trailing = {"index": "enhanced-only"}
        summary, manifest = analyze_teacher_prefix(
            self.source_rows,
            [*self.teacher_rows, trailing],
            self.semantic_ids,
            expected_rows=2,
            expected_teacher_trailing_rows=1,
            top_ks=(5,),
            candidate_limit=5,
        )
        self.assertEqual(len(manifest), 2)
        self.assertEqual(summary["protocol"]["teacher_scope"], "exact_aligned_prefix")
        self.assertEqual(summary["verification"]["teacher_prefix_rows"], 2)
        self.assertEqual(summary["verification"]["teacher_trailing_rows"], 1)

    def test_exact_teacher_prefix_rejects_unexpected_trailing_rows(self):
        trailing = {"index": "enhanced-only"}
        with self.assertRaisesRegex(ValueError, "teacher trailing rows=1"):
            analyze_teacher_prefix(
                self.source_rows,
                [*self.teacher_rows, trailing],
                self.semantic_ids,
                expected_rows=2,
                expected_teacher_trailing_rows=0,
                top_ks=(5,),
                candidate_limit=5,
            )


if __name__ == "__main__":
    unittest.main()
