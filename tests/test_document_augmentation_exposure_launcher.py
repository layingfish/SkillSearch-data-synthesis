from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_document_augmentation_exposure_20260806.sh"
CONFIG = ROOT / "configs" / "document_augmentation_exposure_20260806.env"


class DocumentAugmentationExposureLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = LAUNCHER.read_text(encoding="utf-8")
        cls.config = CONFIG.read_text(encoding="utf-8")

    def test_launcher_has_four_isolated_arms_and_exact_schedules(self) -> None:
        for arm, passes, schedule, steps in (
            ("query-only", 0, "0,0,0,0,0", 40420),
            ("mixed-c3", 3, "1,0,1,0,1", 43150),
            ("mixed-c6", 6, "1,1,2,1,1", 45879),
            ("mixed-c12", 12, "2,3,2,3,2", 51335),
        ):
            self.assertIn(f"{arm}) CATALOG_PASSES={passes}; EXPECTED_SCHEDULE={schedule}; EXPECTED_STEPS={steps}", self.script)
        self.assertIn(
            "document_augmentation_exposure_20260806_v1",
            self.script,
        )

    def test_launcher_uses_one_fresh_mixed_stream_and_last_checkpoint(self) -> None:
        for fragment in (
            "--catalog-epochs 0",
            "--catalog-max-steps 0",
            "--document-augmentation",
            "--epochs \"${SKILLGR_QUERY_EPOCHS}\"",
            "--query-loss expanded_ce",
            "--lr-scheduler constant",
            "--no-load-best-model-at-end",
        ):
            self.assertIn(fragment, self.script)
        self.assertNotIn("--init-model-dir", self.script)
        self.assertNotIn("--resume-from-checkpoint", self.script)
        self.assertNotIn("--no-reset-optimizer", self.script)

    def test_launcher_rescores_same_predictions_for_all_subsets(self) -> None:
        self.assertIn('for subset in all single multi; do', self.script)
        self.assertIn('"${output}/predictions.jsonl"', self.script)
        self.assertIn('ranked_metrics_${subset}.json', self.script)
        self.assertIn("single/multi subsets do not partition", self.script)

    def test_config_registers_fixed_dataset_and_training_contract(self) -> None:
        expected = {
            "SKILLGR_QUERY_EPOCHS=5",
            "SKILLGR_LR=0.0003",
            "SKILLGR_WEIGHT_DECAY=0.0",
            "SKILLGR_WARMUP_RATIO=0.0",
            "SKILLGR_MAX_SOURCE_LENGTH=2048",
            "SKILLGR_SEED=20260626",
            "SKILLGR_NUM_BEAMS=50",
            "SKILLGR_EXPECTED_CATALOG_ROWS=14551",
            "SKILLGR_EXPECTED_QUERY_ROWS=129341",
            "SKILLGR_TARGET_TOKENS_PER_ROW=5",
        }
        for line in expected:
            self.assertIn(line, self.config)

    def test_launcher_passes_shell_syntax_check(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
