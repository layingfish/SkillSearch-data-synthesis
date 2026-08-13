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

    def test_minimum_cost_last_code_resolution_is_collision_free(self) -> None:
        import torch

        from train_query_aligned_rq import minimum_cost_unique_last_codes

        codebooks = torch.tensor(
            [
                [[0.0, 0.0], [10.0, 10.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        )
        embeddings = torch.tensor([[0.0, 0.0], [0.9, 0.0]])
        collided = torch.tensor([[0, 0], [0, 0]])

        resolved, audit = minimum_cost_unique_last_codes(
            collided, embeddings, codebooks
        )

        self.assertEqual(len({tuple(row) for row in resolved.tolist()}), 2)
        self.assertEqual(audit["post_collision_groups"], 0)
        self.assertEqual(audit["repaired_items"], 2)


if __name__ == "__main__":
    unittest.main()
