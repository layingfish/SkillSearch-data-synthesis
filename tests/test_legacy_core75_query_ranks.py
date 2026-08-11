import copy
import unittest

from scripts.analyze_legacy_core75_query_ranks import analyze, validate_and_align


def make_ranking(prefix: str, first: list[str]) -> list[str]:
    ranked = list(first)
    candidate = 0
    while len(ranked) < 50:
        skill_id = f"{prefix}/candidate-{candidate}"
        candidate += 1
        if skill_id not in ranked:
            ranked.append(skill_id)
    return ranked


def make_row(
    index: int,
    gold: list[str],
    ranked: list[str],
    *,
    with_scores: bool,
) -> dict:
    return {
        "index": index,
        "source_index": 100 + index,
        "query": f"query-{index}",
        "gold_skill_ids": gold,
        "target_skill_count": len(gold),
        "predicted_skill_ids": [ranked[0]],
        "ranked_skill_ids": ranked,
        "ranked_skill_scores": (
            {skill_id: -0.1 * rank for rank, skill_id in enumerate(ranked)}
            if with_scores
            else None
        ),
    }


class LegacyCore75QueryRankForensicsTest(unittest.TestCase):
    def setUp(self):
        reference_rankings = [
            make_ranking("r0", ["gold-0", "distractor-0"]),
            make_ranking("r1", ["distractor-1", "gold-1", "gold-2"]),
            make_ranking("r2", ["distractor-2"]),
        ]
        system_rankings = [
            make_ranking("s0", ["distractor-0", "gold-0"]),
            make_ranking("s1", ["gold-2", "distractor-1", "gold-1"]),
            make_ranking("s2", ["distractor-2"]),
        ]
        gold = [["gold-0"], ["gold-1", "gold-2"], ["missing-gold"]]
        self.systems = {
            "rtx": [
                make_row(index, gold[index], ranking, with_scores=False)
                for index, ranking in enumerate(reference_rankings)
            ],
            "a800": [
                make_row(index, gold[index], ranking, with_scores=True)
                for index, ranking in enumerate(system_rankings)
            ],
        }

    def test_multigt_recall_flip_and_margin_semantics(self):
        payload, per_query = analyze(
            self.systems,
            reference_label="rtx",
            expected_rows=3,
            expected_depth=50,
        )

        a800 = payload["systems"]["a800"]["subsets"]["overall"]
        self.assertAlmostEqual(a800["macro_recall"]["@1"], 1.0 / 6.0)
        self.assertEqual(a800["top1_hit_queries"], 1)
        self.assertEqual(
            a800["nearest_gold_rank_buckets"],
            {
                "rank1": 1,
                "rank2_5": 1,
                "rank6_10": 0,
                "rank11_50": 0,
                "missing_top50": 1,
            },
        )

        comparison = payload["comparisons_to_reference"]["a800"]
        self.assertEqual(
            comparison["top1_flip_counts"],
            {
                "both_correct": 0,
                "reference_correct_system_wrong": 1,
                "reference_wrong_system_correct": 1,
                "both_wrong": 1,
            },
        )
        self.assertAlmostEqual(
            per_query[0]["systems"]["a800"]["score_margins"][
                "top1_minus_best_gold"
            ],
            0.1,
        )
        self.assertEqual(
            comparison["critical_system_gold_rank_buckets"]["rank2_5"], 1
        )
        pairwise = payload["pairwise_rank_overlap"]["rtx__vs__a800"]
        self.assertEqual(pairwise["rows"], 3)
        self.assertEqual(pairwise["common_top50_score_pearson"]["count"], 0)
        self.assertEqual(payload["critical_overlap"]["lost_by_all_systems"], [0])
        self.assertTrue(payload["verification"]["passed"])

    def test_nonmonotonic_scores_are_rejected(self):
        systems = copy.deepcopy(self.systems)
        ranked = systems["a800"][0]["ranked_skill_ids"]
        systems["a800"][0]["ranked_skill_scores"][ranked[1]] = 0.5
        with self.assertRaisesRegex(ValueError, "not monotonic"):
            validate_and_align(
                systems,
                reference_label="rtx",
                expected_rows=3,
                expected_depth=50,
            )

    def test_alignment_mismatch_is_rejected(self):
        systems = copy.deepcopy(self.systems)
        systems["a800"][1]["query"] = "different query"
        with self.assertRaisesRegex(ValueError, "query differs"):
            validate_and_align(
                systems,
                reference_label="rtx",
                expected_rows=3,
                expected_depth=50,
            )


if __name__ == "__main__":
    unittest.main()
