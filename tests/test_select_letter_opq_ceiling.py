import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from select_letter_opq_ceiling import load_dev_key, select_candidate  # noqa: E402


def write_metrics(path: Path, values: dict[int, float]) -> None:
    path.write_text(
        json.dumps(
            {
                "recall_aggregation": "macro",
                "overall": {
                    f"@{k}": {"macro_recall": value} for k, value in values.items()
                },
            }
        ),
        encoding="utf-8",
    )


class DevOnlySelectionTest(unittest.TestCase):
    def test_key_is_r10_r5_r1_r50(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "ranked_metrics.json"
            write_metrics(metrics, {1: 0.1, 5: 0.5, 10: 0.7, 50: 0.9})
            self.assertEqual(load_dev_key(metrics), (0.7, 0.5, 0.1, 0.9))

    def test_selection_prefers_r10_then_uses_input_order_for_exact_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            semantic_ids = root / "semantic_ids.jsonl"
            checkpoint.write_bytes(b"checkpoint")
            semantic_ids.write_text("{}\n", encoding="utf-8")
            candidates = []
            for name, values in (
                ("simple", {1: 0.4, 5: 0.6, 10: 0.7, 50: 0.9}),
                ("higher_r1", {1: 0.5, 5: 0.6, 10: 0.7, 50: 0.9}),
                ("higher_r10", {1: 0.1, 5: 0.1, 10: 0.8, 50: 0.1}),
            ):
                metrics = root / f"{name}.json"
                write_metrics(metrics, values)
                candidates.append(
                    {
                        "name": name,
                        "metrics_path": str(metrics),
                        "checkpoint_path": str(checkpoint),
                        "semantic_ids_path": str(semantic_ids),
                    }
                )
            selected = select_candidate(candidates)
            self.assertEqual(selected["winner"]["name"], "higher_r10")
            self.assertFalse(selected["official_metrics_read"])

            tied = select_candidate(candidates[:1] + [dict(candidates[0], name="tie")])
            self.assertEqual(tied["winner"]["name"], "simple")


if __name__ == "__main__":
    unittest.main()
