import unittest

from scripts.analyze_legacy_conditional_prefix_hybrid import (
    EXPECTED_CONDITIONAL_COMPARISONS,
    EXPECTED_PRESENTATIONS,
    SELECTION_LABELS,
    choose_qualified_arm,
)


def summary(*, overall=0.28, multi=0.22, single=0.42, dev=0.8):
    def subset(r1):
        return {
            "@1": {"recall": r1},
            "@10": {"recall": 0.70},
            "@50": {"recall": 0.76},
        }

    return {
        "overall": subset(overall),
        "single": subset(single),
        "multi": subset(multi),
        "training": {"query_best_dev_loss": dev},
    }


class LegacyConditionalPrefixHybridAnalyzerTest(unittest.TestCase):
    def test_no_gate_pass_has_no_fallback_winner(self):
        summaries = {label: summary() for label in SELECTION_LABELS}
        gates = {label: {"passed": False} for label in SELECTION_LABELS}
        passing, selected, basis = choose_qualified_arm(summaries, gates)
        self.assertEqual(passing, [])
        self.assertIsNone(selected)
        self.assertEqual(basis, "no_qualified_arm")

    def test_only_qualified_hybrid_arms_participate_in_selection(self):
        self.assertEqual(
            SELECTION_LABELS,
            (
                "hybrid_s005_m001",
                "hybrid_s005_m002",
                "hybrid_s010_m001",
                "hybrid_s010_m002",
            ),
        )
        summaries = {label: summary() for label in SELECTION_LABELS}
        summaries["hybrid_s010_m002"] = summary(overall=0.30)
        gates = {label: {"passed": True} for label in SELECTION_LABELS}
        _, selected, basis = choose_qualified_arm(summaries, gates)
        self.assertEqual(selected, "hybrid_s010_m002")
        self.assertEqual(basis, "qualified_only")

    def test_expected_conditional_comparison_count_closes(self):
        observed = (
            EXPECTED_PRESENTATIONS["single"] + 3 * EXPECTED_PRESENTATIONS["multi"]
        )
        self.assertEqual(observed, EXPECTED_CONDITIONAL_COMPARISONS)


if __name__ == "__main__":
    unittest.main()
