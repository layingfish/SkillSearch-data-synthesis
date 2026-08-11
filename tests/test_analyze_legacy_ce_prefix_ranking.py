import unittest

from scripts.analyze_legacy_ce_prefix_ranking import (
    ABSOLUTE_THRESHOLDS,
    HISTORICAL_CONTROL,
    RTX_R1_ANCHORS,
    SELECTION_LABELS,
    choose_qualified_arm,
    ranking_key,
    selection_gate,
)


def summary(
    *,
    overall=(0.28, 0.70, 0.76),
    single=(0.42, 0.50, 0.50),
    multi=(0.22, 0.79, 0.87),
    dev=0.8,
):
    def subset(values):
        return {
            cutoff: {"recall": value}
            for cutoff, value in zip(("@1", "@10", "@50"), values)
        }

    return {
        "overall": subset(overall),
        "single": subset(single),
        "multi": subset(multi),
        "training": {"query_best_dev_loss": dev},
    }


class LegacyCePrefixRankingAnalyzerTest(unittest.TestCase):
    def test_exact_one_point_r1_boundaries_pass(self):
        control = HISTORICAL_CONTROL["recalls"]
        candidate = summary(
            overall=(RTX_R1_ANCHORS["overall"] - 0.01, 0.70, 0.76),
            single=(RTX_R1_ANCHORS["single"], 0.50, 0.50),
            multi=(RTX_R1_ANCHORS["multi"] - 0.01, 0.79, 0.87),
        )
        self.assertTrue(selection_gate(candidate, control)["passed"])
        self.assertEqual(
            ABSOLUTE_THRESHOLDS["overall_r1"], RTX_R1_ANCHORS["overall"] - 0.01
        )

    def test_no_gate_pass_has_no_fallback_winner(self):
        summaries = {label: summary() for label in SELECTION_LABELS}
        gates = {label: {"passed": False} for label in SELECTION_LABELS}
        passing, selected, basis = choose_qualified_arm(summaries, gates)
        self.assertEqual(passing, [])
        self.assertIsNone(selected)
        self.assertEqual(basis, "no_qualified_arm")

    def test_gate_requires_subset_deep_recall_noninferiority(self):
        control = HISTORICAL_CONTROL["recalls"]
        passing = summary()
        self.assertTrue(selection_gate(passing, control)["passed"])
        regressed = summary(multi=(0.22, 0.77, 0.87))
        gate = selection_gate(regressed, control)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["multi_r10_control_noninferiority"])

    def test_ranking_prioritizes_overall_then_multi_top1(self):
        higher_overall = summary(overall=(0.29, 0.70, 0.76), multi=(0.21, 0.79, 0.87))
        higher_multi = summary(overall=(0.28, 0.72, 0.77), multi=(0.30, 0.79, 0.87))
        self.assertGreater(ranking_key(higher_overall), ranking_key(higher_multi))


if __name__ == "__main__":
    unittest.main()
