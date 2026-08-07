import json
import tempfile
import unittest
from pathlib import Path

from scripts.legacy_runtime_matrix_probe import catalog_loss_summary, selected_weight_file


class LegacyRuntimeMatrixProbeTest(unittest.TestCase):
    def test_selected_weight_file_requires_exactly_one_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                selected_weight_file(root)
            (root / "model.safetensors").write_bytes(b"safe")
            self.assertEqual(selected_weight_file(root).name, "model.safetensors")
            (root / "pytorch_model.bin").write_bytes(b"bin")
            with self.assertRaises(ValueError):
                selected_weight_file(root)

    def test_catalog_loss_summary_requires_contiguous_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loss_history.jsonl"
            rows = [
                {"stage": "catalog", "event": "train", "step": step, "train_loss": 5.0 - step}
                for step in range(1, 4)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(catalog_loss_summary(path, 3), {"step_1": 4.0, "step_3": 2.0})
            rows.pop(1)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                catalog_loss_summary(path, 3)


if __name__ == "__main__":
    unittest.main()
