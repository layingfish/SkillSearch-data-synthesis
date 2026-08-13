from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_sequence_contrastive_checkpoint_sweep import select_rows  # noqa: E402


def row(label: str, step: int, single_r1: float, all_r1: float, all_r10: float, all_r50: float):
    return {
        "label": label,
        "step": step,
        "metrics": {
            "all": {"r@1": all_r1, "r@10": all_r10, "r@50": all_r50},
            "single": {"r@1": single_r1, "r@10": 0.0, "r@50": 0.0},
            "multi": {"r@1": 0.0, "r@10": 0.0, "r@50": 0.0},
        },
    }


class SequenceContrastiveCheckpointSweepTest(unittest.TestCase):
    def test_selects_single_top1_and_separate_deep_recall_guard(self) -> None:
        baseline = row("init", 0, 0.42, 0.28, 0.70, 0.75)
        top1 = row("step_00000200", 200, 0.60, 0.31, 0.68, 0.74)
        guarded = row("step_00000400", 400, 0.55, 0.30, 0.71, 0.76)

        selected = select_rows([baseline, top1, guarded], baseline)

        self.assertIs(selected["single_top1_best"], top1)
        self.assertIs(selected["deep_recall_guarded_best"], guarded)

    def test_guard_falls_back_to_anchor_when_all_trained_checkpoints_regress(self) -> None:
        baseline = row("init", 0, 0.42, 0.28, 0.70, 0.75)
        regressed = row("step_00000200", 200, 0.60, 0.31, 0.69, 0.76)

        selected = select_rows([baseline, regressed], baseline)

        self.assertIs(selected["deep_recall_guarded_best"], baseline)


if __name__ == "__main__":
    unittest.main()
