import unittest

from scripts.analyze_legacy_core75_prefix_component_attribution import (
    ALL_RTX_MASK,
    COMPONENT_LABELS,
    component_swap_analysis,
    hybrid_rank,
    macro_metrics,
    prefix_query_payload,
    shapley_values,
)
from scripts.legacy_core75_score_protocol import R1_GATE, RTX_R1_REFERENCE


def attribution_row(label, index, ranked, gold, components):
    return {
        "label": label,
        "row_position": index,
        "index": index,
        "source_index": 100 + index,
        "gold_skill_ids": list(gold),
        "target_skill_count": len(gold),
        "pool_skill_ids": sorted(components),
        "fixed_ranked_skill_ids": list(ranked),
        "fixed_ranked_skill_scores": {
            skill_id: sum(float(value) for value in values)
            for skill_id, values in components.items()
        },
        "fixed_ranked_skill_score_components": components,
    }


class PrefixAttributionTest(unittest.TestCase):
    def test_r1_gate_reuses_exact_position_protocol(self):
        self.assertEqual(R1_GATE, RTX_R1_REFERENCE - 0.01)

    def test_prefix_macro_recall_preserves_shared_gold_mentions(self):
        semantic_ids = {
            "g1": [1, 1, 1, 1],
            "g2": [1, 2, 2, 2],
            "x": [2, 1, 1, 1],
        }
        row = attribution_row(
            "a800",
            0,
            ["g1", "g2", "x"],
            ["g1", "g2"],
            {
                "g1": [0.0, 0.0, 0.0, 0.0, 0.0],
                "g2": [0.0, -1.0, 0.0, 0.0, 0.0],
                "x": [-1.0, 0.0, 0.0, 0.0, 0.0],
            },
        )
        depth1 = prefix_query_payload(row, semantic_ids, 1)
        depth2 = prefix_query_payload(row, semantic_ids, 2)
        self.assertEqual(depth1["unique_gold_prefixes"], 1)
        self.assertEqual(depth1["recall"]["@1"], 1.0)
        self.assertEqual(depth2["unique_gold_prefixes"], 2)
        self.assertEqual(depth2["recall"]["@1"], 0.5)

    def test_same_prefix_score_spread_is_rejected(self):
        semantic_ids = {
            "g1": [1, 1, 1, 1],
            "g2": [1, 2, 2, 2],
            "x": [2, 1, 1, 1],
        }
        row = attribution_row(
            "a800",
            0,
            ["g2", "g1", "x"],
            ["g1"],
            {
                "g1": [0.0, 0.0, 0.0, 0.0, 0.0],
                "g2": [0.01, 0.0, 0.0, 0.0, 0.0],
                "x": [-1.0, 0.0, 0.0, 0.0, 0.0],
            },
        )
        with self.assertRaisesRegex(ValueError, "same-prefix score spread"):
            prefix_query_payload(row, semantic_ids, 1)

    def test_component_swap_endpoints_and_critical_recovery(self):
        reference = attribution_row(
            "rtx",
            0,
            ["g", "x"],
            ["g"],
            {
                "g": [2.0, 0.0, 0.0, 0.0, 0.0],
                "x": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
        )
        candidate = attribution_row(
            "a800",
            0,
            ["x", "g"],
            ["g"],
            {
                "g": [0.0, 0.0, 0.0, 0.0, 0.0],
                "x": [1.0, 0.0, 0.0, 0.0, 0.0],
            },
        )
        self.assertEqual(hybrid_rank(reference, candidate, 0)[0], ["x", "g"])
        self.assertEqual(
            hybrid_rank(reference, candidate, ALL_RTX_MASK)[0], ["g", "x"]
        )
        summary, grid = component_swap_analysis(
            {"rtx": [reference], "a800": [candidate]}, reference_label="rtx"
        )
        self.assertEqual(len(grid), 32)
        self.assertTrue(all(not row["deployable"] for row in grid))
        self.assertTrue(
            all(row["scope"] == "counterfactual_attribution_only" for row in grid)
        )
        self.assertEqual(summary["a800"]["minimal_full_gate"]["rtx_components"], [COMPONENT_LABELS[0]])
        self.assertEqual(
            summary["a800"]["single_component_swaps"][COMPONENT_LABELS[0]][
                "critical_top1_recovered"
            ],
            1,
        )
        self.assertAlmostEqual(
            summary["a800"]["shapley"]["@1"][COMPONENT_LABELS[0]], 1.0
        )

    def test_shapley_conservation_and_macro_metrics(self):
        coefficients = [0.5, -0.25, 0.125, 0.0, 0.25]
        values = {
            mask: sum(
                coefficient
                for position, coefficient in enumerate(coefficients)
                if mask & (1 << position)
            )
            for mask in range(32)
        }
        shapley = shapley_values(values)
        for label, expected in zip(COMPONENT_LABELS, coefficients):
            self.assertAlmostEqual(shapley[label], expected)

        rows = [{"gold_skill_ids": ["g1", "g2"]}]
        metrics = macro_metrics(rows, [["g1", "x", "g2"]], [["g1", "g2"]])
        self.assertEqual(metrics["@1"], 0.5)
        self.assertEqual(metrics["@5"], 1.0)


if __name__ == "__main__":
    unittest.main()
