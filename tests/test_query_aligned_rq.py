from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "torch is available in the remote training environment",
)
class QueryAlignedRQTest(unittest.TestCase):
    def test_one_batch_reconstruction_overfits(self) -> None:
        import torch

        from train_query_aligned_rq import QueryAlignedResidualCodebook

        initial = torch.tensor(
            [
                [[0.0, 0.0], [0.5, 0.5]],
                [[0.0, 0.0], [0.25, 0.25]],
            ]
        )
        model = QueryAlignedResidualCodebook(initial)
        embeddings = torch.tensor([[1.0, 1.0], [-0.5, -0.5]])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        losses = []
        for _ in range(30):
            components = model(
                embeddings,
                embeddings,
                embeddings,
                [],
                [],
                [],
                [{(): (0, 1)}, {(0,): (0, 1), (1,): (0, 1)}],
                semantic_weight=0.0,
                cf_weight=0.0,
                prefix_weight=0.0,
                teacher_temperature=1.0,
                student_temperature=1.0,
            )
            optimizer.zero_grad()
            components["loss"].backward()
            optimizer.step()
            losses.append(float(components["loss"].detach()))

        self.assertLess(losses[-1], losses[0] * 0.2)

    def test_residual_encode_uses_ordered_codebooks(self) -> None:
        import torch

        from train_query_aligned_rq import residual_encode

        codebooks = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.0], [0.0, 0.5]],
            ]
        )
        codes, reconstruction = residual_encode(
            torch.tensor([[1.4, 0.1], [0.1, 1.4]]), codebooks
        )

        self.assertEqual(codes.tolist(), [[0, 0], [1, 1]])
        self.assertTrue(
            torch.allclose(
                reconstruction,
                torch.tensor([[1.5, 0.0], [0.0, 1.5]]),
            )
        )

    def test_prefix_teacher_loss_uses_all_legal_children(self) -> None:
        import torch

        from train_query_aligned_rq import prefix_teacher_distribution_loss

        codebooks = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                [[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]],
            ],
            requires_grad=True,
        )
        codes = torch.tensor([[0, 0], [0, 1], [1, 0]])
        teacher_scores = torch.log(torch.tensor([0.2, 0.3, 0.5]))
        legal = [
            {(): (0, 1, 2)},
            {(0,): (0, 1, 2), (1,): (0,), (2,): (0,)},
        ]

        loss = prefix_teacher_distribution_loss(
            torch.tensor([1.0, 1.0]),
            codes,
            teacher_scores,
            codebooks,
            legal,
            teacher_temperature=1.0,
            student_temperature=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(codebooks.grad.abs().sum()), 0.0)

    def test_hierarchical_capacity_resolution_is_collision_free(self) -> None:
        import torch

        from train_query_aligned_rq import hierarchical_capacity_unique_codes

        codebooks = torch.tensor(
            [
                [[0.0, 0.0], [10.0, 10.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        )
        embeddings = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        collided = torch.tensor([[0, 0], [0, 0]])

        resolved, audit = hierarchical_capacity_unique_codes(
            collided, embeddings, codebooks
        )

        self.assertEqual(len({tuple(row) for row in resolved.tolist()}), 2)
        self.assertEqual(audit["post_collision_groups"], 0)
        self.assertTrue(any(row["repaired_items"] == 2 for row in audit["level_audit"]))

    def test_hierarchical_capacity_resolution_falls_back_before_last_level(self) -> None:
        import torch

        from train_query_aligned_rq import hierarchical_capacity_unique_codes

        codebooks = torch.tensor(
            [
                [[0.0], [10.0]],
                [[0.0], [1.0]],
                [[0.0], [0.1]],
            ]
        )
        embeddings = torch.tensor([[0.00], [0.02], [0.04], [0.06], [0.08]])
        collided = torch.zeros((5, 3), dtype=torch.long)

        resolved, audit = hierarchical_capacity_unique_codes(
            collided, embeddings, codebooks
        )

        self.assertEqual(len({tuple(row) for row in resolved.tolist()}), 5)
        self.assertGreater(audit["level_audit"][0]["capacity_reassignments"], 0)
        self.assertTrue(
            all(row["overloaded_prefix_groups"] == 0 for row in audit["post_capacity_audit"])
        )

    def test_fixed_skill_codes_control_reconstruction(self) -> None:
        import torch

        from train_query_aligned_rq import QueryAlignedResidualCodebook

        initial = torch.tensor(
            [
                [[0.0], [10.0]],
                [[0.0], [10.0]],
            ]
        )
        model = QueryAlignedResidualCodebook(initial)
        embedding = torch.tensor([[0.0]])
        components = model(
            embedding,
            embedding,
            embedding,
            [],
            [],
            [],
            [{(): (0, 1)}, {(0,): (0, 1), (1,): (0, 1)}],
            skill_codes=torch.tensor([[1, 1]]),
            semantic_weight=0.0,
            cf_weight=0.0,
            prefix_weight=0.0,
            teacher_temperature=1.0,
            student_temperature=1.0,
        )

        self.assertAlmostEqual(float(components["reconstruction_loss"]), 400.0)
        components["loss"].backward()
        self.assertEqual(float(model.codebooks.grad[:, 0].abs().sum()), 0.0)
        self.assertGreater(float(model.codebooks.grad[:, 1].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
