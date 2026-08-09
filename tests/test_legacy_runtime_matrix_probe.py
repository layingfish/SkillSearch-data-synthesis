from argparse import Namespace
import json
import tempfile
import unittest
from pathlib import Path

from scripts.legacy_runtime_matrix_probe import (
    catalog_loss_summary,
    optional_bool,
    selected_weight_file,
    training_wrapper_source,
)


class LegacyRuntimeMatrixProbeTest(unittest.TestCase):
    def test_optional_bool_and_wrapper_controls(self):
        self.assertIsNone(optional_bool("auto"))
        self.assertIs(optional_bool("true"), True)
        self.assertIs(optional_bool("false"), False)
        with self.assertRaises(ValueError):
            optional_bool("invalid")

        source = training_wrapper_source(
            Namespace(
                bf16_reduction="false",
                tf32_matmul="true",
                tf32_cudnn="false",
                float32_matmul_precision="highest",
                foreach="false",
                preferred_blas="cublas",
                deterministic_algorithms="true",
            )
        )
        compile(source, "<training-wrapper>", "exec")
        self.assertIn("'bf16_reduction': False", source)
        self.assertIn("'foreach': False", source)

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
