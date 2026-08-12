import tempfile
import unittest
from pathlib import Path

from scripts.legacy_sequence_weight_checkpoint_sweep import (
    _matrix_winner,
    _replace_retained,
    joint_eligible,
    select_best,
)


def _result(step, r1, r10, r50):
    return {
        "step": step,
        "overall": {
            "@1": {"recall": r1},
            "@10": {"recall": r10},
            "@50": {"recall": r50},
        },
    }


class SequenceWeightSelectionTest(unittest.TestCase):
    def test_joint_gate_is_inclusive(self):
        self.assertTrue(
            joint_eligible(_result(200, 0.28, 0.69, 0.75), min_r10=0.69, min_r50=0.75)
        )
        self.assertFalse(
            joint_eligible(_result(200, 0.29, 0.689, 0.80), min_r10=0.69, min_r50=0.75)
        )

    def test_joint_selection_filters_before_ranking(self):
        top1_only = _result(400, 0.31, 0.68, 0.80)
        joint = _result(600, 0.29, 0.70, 0.76)
        results = [joint, top1_only]
        self.assertIs(select_best(results), top1_only)
        self.assertIs(select_best(results, min_r10=0.69, min_r50=0.75), joint)

    def test_matrix_tie_prefers_registered_arm_order(self):
        result = _result(200, 0.29, 0.70, 0.76)
        arms = [
            {"arm": "seq_001", "top1_best": result},
            {"arm": "seq_002", "top1_best": result},
        ]
        self.assertEqual(_matrix_winner(arms, "top1_best")["arm"], "seq_001")


class DualCheckpointRetentionTest(unittest.TestCase):
    @staticmethod
    def _candidate(root: Path, name: str, value: str) -> Path:
        candidate = root / name
        candidate.mkdir()
        (candidate / "structured_decoder.pt").write_text(value, encoding="utf-8")
        return candidate

    def test_dual_retention_then_independent_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            top1 = root / "worker_01_top1_best"
            joint = root / "worker_01_joint_best"

            first = self._candidate(root, "step_00000200", "first")
            _replace_retained(
                first,
                top1_path=top1,
                joint_path=joint,
                improve_top1=True,
                improve_joint=True,
                queue_root=root,
            )
            top1_file = top1 / "structured_decoder.pt"
            joint_file = joint / "structured_decoder.pt"
            self.assertEqual(top1_file.read_text(encoding="utf-8"), "first")
            self.assertEqual(top1_file.stat().st_ino, joint_file.stat().st_ino)

            second = self._candidate(root, "step_00000400", "second")
            _replace_retained(
                second,
                top1_path=top1,
                joint_path=joint,
                improve_top1=True,
                improve_joint=False,
                queue_root=root,
            )
            self.assertEqual(top1_file.read_text(encoding="utf-8"), "second")
            self.assertEqual(joint_file.read_text(encoding="utf-8"), "first")

            third = self._candidate(root, "step_00000600", "third")
            _replace_retained(
                third,
                top1_path=top1,
                joint_path=joint,
                improve_top1=False,
                improve_joint=True,
                queue_root=root,
            )
            self.assertEqual(top1_file.read_text(encoding="utf-8"), "second")
            self.assertEqual(joint_file.read_text(encoding="utf-8"), "third")


class SweepLauncherOrderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_legacy_sequence_weight_checkpoint_sweep_node_20260812.sh"
        ).read_text(encoding="utf-8")

    def test_trainer_owns_arm_root_before_evaluators_start(self):
        run_arm = self.launcher.index("run_arm() {")
        trainer_started = self.launcher.index('TRAINING_PID="$!"', run_arm)
        worker_loop = self.launcher.index(
            "for worker_id in 1 2 3 4 5 6 7; do", trainer_started
        )
        self.assertLess(trainer_started, worker_loop)
        self.assertNotIn('mkdir -p "${arm_root}"', self.launcher[run_arm:worker_loop])

    def test_failure_cleanup_waits_only_explicit_pids(self):
        cleanup_start = self.launcher.index("terminate_workers() {")
        cleanup_end = self.launcher.index("\n}", cleanup_start)
        cleanup = self.launcher[cleanup_start:cleanup_end]
        self.assertNotIn("wait || true", cleanup)
        self.assertIn('wait "${TRAINING_PID}"', cleanup)
        self.assertIn('wait "${pid}"', cleanup)


if __name__ == "__main__":
    unittest.main()
