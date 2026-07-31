from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_normal_skillrouter_inputs import compute_cardinality_quotas
from build_normal_skillrouter_inputs import write_single_skill_inputs


class NormalSkillRouterInputsTest(unittest.TestCase):
    def test_truncated_normal_quota_is_bell_shaped_and_sums_to_total(self) -> None:
        quotas = compute_cardinality_quotas(total=1000, mu=2.5, sigma=0.9, min_k=1, max_k=4)

        self.assertEqual(sum(quotas.values()), 1000)
        self.assertEqual(set(quotas), {1, 2, 3, 4})
        self.assertGreater(quotas[2], quotas[1])
        self.assertGreater(quotas[3], quotas[4])
        self.assertEqual(quotas[2], quotas[3])
        self.assertEqual(quotas[1], quotas[4])

    def test_single_skill_input_writer_uses_replay_schema(self) -> None:
        rows = [
            {
                "skill_id": "dev/a",
                "skill_name": "A",
                "task_space_id": "space1",
                "description": "Implement a concrete code change from a user request with clear output artifacts.",
                "operation_role": "implement",
                "operation_type": "code_change",
            },
            {
                "skill_id": "dev/b",
                "skill_name": "B",
                "task_space_id": "space2",
                "description": "Analyze a technical artifact and return a concise inspection report for the user.",
                "operation_role": "analyze",
                "operation_type": "inspection",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "input_1skill.jsonl"
            written = write_single_skill_inputs(rows, output, limit=2, seed=7)

            self.assertEqual(written, 2)
            payloads = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["target_skill_count"] for row in payloads], [1, 1])
        self.assertEqual([row["index"] for row in payloads], [1, 2])
        self.assertEqual({row["skill_id"] for row in payloads}, {"dev/a", "dev/b"})
        self.assertIn("sampling", payloads[0])


if __name__ == "__main__":
    unittest.main()
