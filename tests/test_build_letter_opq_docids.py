#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_letter_opq_docids import (  # noqa: E402
    position_usage,
    prefix_summary,
    weighted_semantic_cf_fusion,
)
from probe_letter_opq import semantic_ids_sha256, write_semantic_ids  # noqa: E402


class LetterLongOPQDocIDTest(unittest.TestCase):
    def test_weighted_semantic_cf_fusion_has_declared_energy_and_padding(self) -> None:
        semantic = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
        collaborative = np.asarray([[0.0, 5.0], [12.0, 5.0]], dtype=np.float32)
        fused, metadata = weighted_semantic_cf_fusion(
            semantic,
            collaborative,
            cf_energy_fraction=0.2,
            dimension_multiple=3,
        )

        self.assertEqual(fused.shape, (2, 6))
        np.testing.assert_allclose(np.linalg.norm(fused, axis=1), 1.0, atol=2e-6)
        np.testing.assert_allclose(np.square(fused[:, :2]).sum(axis=1), 0.8, atol=2e-6)
        np.testing.assert_allclose(np.square(fused[:, 2:4]).sum(axis=1), 0.2, atol=2e-6)
        np.testing.assert_array_equal(fused[:, 4:], 0.0)
        self.assertEqual(metadata["padding_dim"], 2)
        self.assertEqual(metadata["quantizer_dim"], 6)

    def test_weighted_semantic_cf_fusion_rejects_invalid_contracts(self) -> None:
        semantic = np.ones((2, 4), dtype=np.float32)
        collaborative = np.ones((3, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "row mismatch"):
            weighted_semantic_cf_fusion(
                semantic,
                collaborative,
                cf_energy_fraction=0.2,
                dimension_multiple=4,
            )
        with self.assertRaisesRegex(ValueError, "strictly between"):
            weighted_semantic_cf_fusion(
                semantic,
                np.ones((2, 2), dtype=np.float32),
                cf_energy_fraction=1.0,
                dimension_multiple=4,
            )

    def test_long_docid_artifact_is_decoder_compatible(self) -> None:
        from skill_gr_pipeline import load_semantic_ids_with_mode  # noqa: PLC0415

        rng = np.random.default_rng(42)
        for depth in (32, 64):
            codes = rng.integers(0, 256, size=(17, depth), dtype=np.int64)
            skill_ids = [f"skill-{index:02d}" for index in range(len(codes))]
            with self.subTest(depth=depth), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "semantic_ids.jsonl"
                write_semantic_ids(path, skill_ids, codes, codes, "opq")
                loaded, mode = load_semantic_ids_with_mode(path)
                self.assertEqual(mode, "learned_codes")
                self.assertEqual({len(value) for value in loaded.values()}, {depth})
                self.assertEqual(
                    semantic_ids_sha256(skill_ids, codes),
                    __import__("skill_gr_pipeline").semantic_ids_sha256(loaded, mode),
                )
                first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(len(first["tokens"]), depth)
                self.assertEqual(first["tokens"][-1].split("_")[0], f"<RQ{depth - 1}")

    def test_usage_and_prefix_summaries_cover_every_position(self) -> None:
        codes = np.asarray([[0, 1, 2], [0, 1, 3], [4, 5, 6]], dtype=np.int64)
        usage = position_usage(codes, 256)
        prefixes = prefix_summary(codes)
        self.assertEqual(len(usage), 3)
        self.assertEqual([row["unique_prefixes"] for row in prefixes], [2, 2, 3])
        self.assertEqual([row["max_bucket_size"] for row in prefixes], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
