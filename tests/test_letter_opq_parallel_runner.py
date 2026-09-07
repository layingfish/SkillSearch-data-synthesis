import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_RUNNER = (
    PROJECT_ROOT
    / "repro/20260902_letter_opq_32_64/run_rudder_parallel_1node_split_4x4.sh"
)
VARIANT_RUNNER = (
    PROJECT_ROOT / "repro/20260902_letter_opq_32_64/run_rudder_parallel_mtp.sh"
)


class LetterOpqParallelRunnerTest(unittest.TestCase):
    def test_assigns_isolated_four_gpu_groups_and_ports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            fake_project = temp_root / "project"
            fake_runner = (
                fake_project
                / "repro/20260902_letter_opq_32_64/run_rudder_parallel_mtp.sh"
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
                    "LETTER_OPQ_RUN_TAG": "unit-test-parallel",
                    "LETTER_OPQ_TEST_RECORD": str(record),
                }
            )

            result = subprocess.run(
                ["bash", str(SPLIT_RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                sorted(record.read_text(encoding="utf-8").splitlines()),
                [
                    "32\t0,1,2,3\t4\t4\t29632",
                    "64\t4,5,6,7\t4\t4\t29664",
                ],
            )
            self.assertIn("letter_opq_parallel_exit depth=32 status=0", result.stdout)
            self.assertIn("letter_opq_parallel_exit depth=64 status=0", result.stdout)

    def test_variant_is_official_only_and_uses_explicit_rank_rendezvous(self):
        source = VARIANT_RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("torch.distributed.run", source)
        self.assertIn('MASTER_PORT="$master_port"', source)
        self.assertIn('RANK="$rank"', source)
        self.assertIn('LOCAL_RANK="$rank"', source)
        self.assertIn("-u TORCHELASTIC_RUN_ID", source)
        self.assertIn('"splits": ["official_core75"]', source)
        self.assertIn('"private_set": False', source)
        self.assertNotIn("private_test1000", source)
        self.assertNotIn("rows_002.decoder_rows", source)


if __name__ == "__main__":
    unittest.main()
