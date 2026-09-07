from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from probe_letter_opq import (  # noqa: E402
    apply_position_order,
    audit_code_matrix,
    collision_summary,
    minimum_cost_unique_last_codes,
    parallel_order_report,
    reconstruction_metrics,
    undo_position_order,
)


class LetterOpqProbeTest(unittest.TestCase):
    def test_position_order_round_trip(self) -> None:
        codes = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])
        permutation = [2, 0, 3, 1]

        ordered = apply_position_order(codes, permutation)

        np.testing.assert_array_equal(undo_position_order(ordered, permutation), codes)

    def test_minimum_cost_collision_repair_is_unique_and_joint(self) -> None:
        codes = np.asarray(
            [
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [9, 9, 9, 0],
            ]
        )
        costs = np.full((4, 4), 100.0)
        costs[0] = [8.0, 0.0, 5.0, 6.0]
        costs[1] = [7.0, 0.1, 0.0, 6.0]
        costs[2] = [7.0, 0.2, 0.3, 0.0]
        costs[3] = [0.0, 5.0, 5.0, 5.0]

        resolved, audit = minimum_cost_unique_last_codes(codes, costs)

        self.assertEqual(collision_summary(resolved)["collision_groups"], 0)
        self.assertEqual(set(resolved[:3, -1].tolist()), {1, 2, 3})
        self.assertEqual(audit["repaired_prefixes"], 1)
        self.assertEqual(audit["repaired_items"], 3)
        self.assertEqual(audit["changed_items"], 2)
        np.testing.assert_array_equal(resolved[3], codes[3])

    def test_audit_reports_prefix_hierarchy_and_collisions(self) -> None:
        codes = np.asarray(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 0],
            ]
        )
        vectors = np.asarray([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32)

        audit = audit_code_matrix(codes, codebook_size=2, semantic_vectors=vectors)

        self.assertEqual(audit["positions"][0]["used_codes"], 2)
        self.assertEqual(audit["prefixes"][0]["unique_prefixes"], 2)
        self.assertLess(audit["prefixes"][0]["semantic_within_group_sse_ratio"], 0.001)
        self.assertEqual(audit["full_tuple"]["collision_groups"], 1)
        self.assertEqual(audit["full_tuple"]["collided_items"], 2)

    def test_parallel_order_puts_semantic_partition_first(self) -> None:
        # Position 1 identifies the two semantic neighborhoods; position 0 is noise.
        codes = np.asarray(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )
        sample_indices = np.arange(4)
        semantic_neighbors = np.asarray([[1], [0], [3], [2]])

        report = parallel_order_report(
            codes, sample_indices, semantic_neighbors, order_k=1
        )

        self.assertEqual(report["selected"]["permutation"], [1, 0])
        self.assertEqual(
            report["selected"]["semantic_neighbor_prefix_recall_at_order_k"], [1.0]
        )
        self.assertEqual(
            report["native"]["semantic_neighbor_prefix_recall_at_order_k"], [0.0]
        )

    def test_reconstruction_metrics_are_zero_for_identity(self) -> None:
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        metrics = reconstruction_metrics(vectors, vectors.copy())

        self.assertEqual(metrics["mse_per_dimension"], 0.0)
        self.assertEqual(metrics["variance_normalized_sse"], 0.0)
        self.assertEqual(metrics["mean_cosine"], 1.0)


if __name__ == "__main__":
    unittest.main()
