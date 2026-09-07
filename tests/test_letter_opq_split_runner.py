import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOINT_RUNNER = (
    PROJECT_ROOT / "repro/20260902_letter_opq_32_64/run_rudder_1node_split_4x4.sh"
)


class LetterOpqSplitRunnerTest(unittest.TestCase):
    def test_assigns_disjoint_four_gpu_groups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            fake_project = temp_root / "project"
            fake_runner = (
                fake_project
                / "repro/20260902_letter_opq_32_64/run_rudder_8gpu_decoder.sh"
            )
            fake_runner.parent.mkdir(parents=True)
            fake_runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'mkdir -p "$RIPOR_EXPERIMENT"\n'
                "printf '%s\\t%s\\t%s\\t%s\\t%s\\n' "
                '"$LETTER_OPQ_DEPTH" "$CUDA_VISIBLE_DEVICES" '
                '"$LETTER_OPQ_WORLD_SIZE" '
                '"$LETTER_OPQ_PER_DEVICE_BATCH_SIZE" '
                '"$LETTER_OPQ_MASTER_PORT" >> "$LETTER_OPQ_TEST_RECORD"\n',
                encoding="utf-8",
            )
            record = temp_root / "assignments.tsv"
            env = os.environ.copy()
            env.update(
                {
                    "LETTER_OPQ_CFS_AGENT_ROOT": str(temp_root / "cfs-agent"),
                    "RIPOR_PROJECT_ROOT": str(fake_project),
                    "LETTER_OPQ_RUN_TAG": "unit-test",
                    "LETTER_OPQ_TEST_RECORD": str(record),
                }
            )

            result = subprocess.run(
                ["bash", str(JOINT_RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            assignments = sorted(record.read_text(encoding="utf-8").splitlines())
            self.assertEqual(
                assignments,
                [
                    "32\t0,1,2,3\t4\t4\t29532",
                    "64\t4,5,6,7\t4\t4\t29564",
                ],
            )
            self.assertIn("letter_opq_variant_exit depth=32 status=0", result.stdout)
            self.assertIn("letter_opq_variant_exit depth=64 status=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
