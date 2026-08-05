from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_catalog_resume_checkpoints import (  # noqa: E402
    compare_tensor_mappings,
    file_sha256,
    parse_label_path,
    tensor_mapping_sha256,
)


class CatalogResumeCheckpointComparisonTest(unittest.TestCase):
    def test_parse_label_path(self) -> None:
        label, path = parse_label_path("resume_a=/tmp/checkpoint.pt")

        self.assertEqual(label, "resume_a")
        self.assertEqual(path, Path("/tmp/checkpoint.pt"))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_label_path("missing-separator")

    def test_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.bin"
            path.write_bytes(b"catalog-resume")

            observed = file_sha256(path, chunk_size=3)

        self.assertEqual(observed, hashlib.sha256(b"catalog-resume").hexdigest())

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "torch is available in the remote training environment",
    )
    def test_compare_tensor_mappings_reports_exact_and_numeric_drift(self) -> None:
        import torch

        left = {
            "weight": torch.tensor([1.0, 2.0, 3.0]),
            "step": torch.tensor(4),
        }
        right = {
            "weight": torch.tensor([1.0, 2.5, 3.0]),
            "step": torch.tensor(4),
        }

        result = compare_tensor_mappings(left, right, chunk_elements=2, top_k=1)

        self.assertEqual(result["total_numel"], 4)
        self.assertEqual(result["exact_numel"], 3)
        self.assertAlmostEqual(result["exact_fraction"], 0.75)
        self.assertAlmostEqual(result["max_abs_diff"], 0.5)
        self.assertTrue(result["schema_exact"])
        self.assertEqual(result["left_nonfinite_numel"], 0)
        self.assertEqual(result["right_nonfinite_numel"], 0)
        self.assertEqual(result["top_drift_tensors"][0]["key"], "weight")

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "torch is available in the remote training environment",
    )
    def test_tensor_mapping_sha256_accepts_scalar_tensors(self) -> None:
        import torch

        observed = tensor_mapping_sha256({"step": torch.tensor(4.0)})

        self.assertEqual(len(observed), 64)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "torch is available in the remote training environment",
    )
    def test_compare_tensor_mappings_reports_schema_and_nonfinite_values(self) -> None:
        import torch

        result = compare_tensor_mappings(
            {"shared": torch.tensor([1.0, float("nan")]), "left_only": torch.ones(1)},
            {"shared": torch.tensor([1.0, float("inf")]), "right_only": torch.ones(1)},
            chunk_elements=1,
        )

        self.assertFalse(result["schema_exact"])
        self.assertEqual(result["missing_from_left"], ["right_only"])
        self.assertEqual(result["missing_from_right"], ["left_only"])
        self.assertEqual(result["left_nonfinite_numel"], 1)
        self.assertEqual(result["right_nonfinite_numel"], 1)
        self.assertEqual(result["finite_numel"], 1)


if __name__ == "__main__":
    unittest.main()
