from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_catalog_convergence import summarize_catalog_convergence  # noqa: E402


class CatalogConvergenceTest(unittest.TestCase):
    def test_selects_first_checkpoint_after_three_stable_intervals(self) -> None:
        checkpoints = [
            {"epoch": 3, "teacher_forced_ce": 2.0, "teacher_forced_sid_exact_accuracy": 0.10},
            {"epoch": 6, "teacher_forced_ce": 1.5, "teacher_forced_sid_exact_accuracy": 0.20},
            {"epoch": 9, "teacher_forced_ce": 1.495, "teacher_forced_sid_exact_accuracy": 0.202},
            {"epoch": 12, "teacher_forced_ce": 1.490, "teacher_forced_sid_exact_accuracy": 0.203},
            {"epoch": 15, "teacher_forced_ce": 1.485, "teacher_forced_sid_exact_accuracy": 0.204},
        ]

        result = summarize_catalog_convergence(
            checkpoints,
            checkpoint_interval=3,
            ce_relative_threshold=0.01,
            sid_exact_absolute_threshold=0.005,
            consecutive_stable_intervals=3,
        )

        self.assertTrue(result["converged"])
        self.assertEqual(result["selected_epoch"], 15)

    def test_reports_no_convergence_when_ce_keeps_improving(self) -> None:
        checkpoints = [
            {
                "epoch": epoch,
                "teacher_forced_ce": 3.0 - 0.1 * index,
                "teacher_forced_sid_exact_accuracy": 0.1 + 0.01 * index,
            }
            for index, epoch in enumerate(range(3, 31, 3))
        ]

        result = summarize_catalog_convergence(
            checkpoints,
            checkpoint_interval=3,
            ce_relative_threshold=0.01,
            sid_exact_absolute_threshold=0.005,
            consecutive_stable_intervals=3,
        )

        self.assertFalse(result["converged"])
        self.assertIsNone(result["selected_epoch"])


if __name__ == "__main__":
    unittest.main()
