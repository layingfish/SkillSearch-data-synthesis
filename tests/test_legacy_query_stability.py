import unittest

try:
    import torch
except ModuleNotFoundError:  # Local source-only environments may omit torch.
    torch = None

from scripts.analyze_legacy_query_stability import (
    ARM_SPECS,
    SELECTION_LABELS,
    audit_intervention,
    expected_intervention,
    expected_lr_trace,
)
from scripts.legacy_full_query_optimizer_lr import (
    FullQueryObserver,
    clip_query_gradients,
    query_lr_at_step,
    validate_loss_history,
)


class LegacyQueryStabilityMetadataTest(unittest.TestCase):
    def test_arm_matrix_fills_eight_gpus(self):
        self.assertEqual(len(ARM_SPECS), 8)
        self.assertEqual(
            [label for label, role, _, _ in ARM_SPECS if role == "selection"],
            list(SELECTION_LABELS),
        )
        self.assertEqual(sum(role == "replica" for _, role, _, _ in ARM_SPECS), 2)
        self.assertEqual(sum(path == "default" for _, _, path, _ in ARM_SPECS), 3)
        self.assertEqual(sum(path == "workspace_16" for _, _, path, _ in ARM_SPECS), 5)

    def test_warmup_schedule(self):
        self.assertAlmostEqual(query_lr_at_step(3e-4, 1, 500), 6e-7)
        self.assertAlmostEqual(query_lr_at_step(3e-4, 250, 500), 1.5e-4)
        self.assertAlmostEqual(query_lr_at_step(3e-4, 500, 500), 3e-4)
        self.assertAlmostEqual(query_lr_at_step(3e-4, 501, 500), 3e-4)
        self.assertEqual(query_lr_at_step(3e-4, 1, 0), 3e-4)

    def test_warmup_loss_history_uses_logged_query_step(self):
        events = [
            {"stage": "catalog", "event": "train", "step": 50, "lr": 3e-4},
            {
                "stage": "query",
                "event": "train",
                "step": 50,
                "lr": query_lr_at_step(3e-4, 50, 500),
            },
            {"stage": "query", "event": "train", "step": 500, "lr": 3e-4},
            {"stage": "query", "event": "train", "step": 501, "lr": 3e-4},
            {"stage": "query", "event": "eval", "step": 500, "dev_loss": 1.0},
            {"stage": "query", "event": "eval", "step": 501, "dev_loss": 0.9},
        ]
        summary = validate_loss_history(
            events,
            query_lr=3e-4,
            query_steps=501,
            eval_steps=500,
            query_warmup_steps=500,
        )
        self.assertEqual(summary["best_step"], 501)
        self.assertEqual(summary["query_warmup_steps"], 500)

        events[1]["lr"] = 3e-4
        with self.assertRaisesRegex(ValueError, "query lr history mismatch"):
            validate_loss_history(
                events,
                query_lr=3e-4,
                query_steps=501,
                eval_steps=500,
                query_warmup_steps=500,
            )

    def test_interventions_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "tested separately"):
            FullQueryObserver(
                optimizer_mode="carry",
                query_lr=3e-4,
                catalog_steps=2730,
                query_steps=38805,
                eval_steps=200,
                batch_size=16,
                query_warmup_steps=500,
                query_max_grad_norm=1.0,
            )

    def test_intervention_audit_contract(self):
        for intervention in ("control", "warmup", "clip"):
            warmup_steps, max_grad_norm = expected_intervention(intervention)
            payload = {
                "condition": {
                    "query_warmup_steps": warmup_steps,
                    "query_max_grad_norm": max_grad_norm,
                },
                "query_stability": {
                    "lr_trace": expected_lr_trace(intervention),
                    "gradient_clipping": {
                        "max_grad_norm": max_grad_norm,
                        "norm_type": 2.0,
                        "foreach": True,
                        "calls": 38805 if intervention == "clip" else 0,
                        "sampled_clipped_calls": 4 if intervention == "clip" else 0,
                        "pre_clip_norm_samples": (
                            {key: 1.5 for key in ("1", "50", "500", "38805")}
                            if intervention == "clip"
                            else {}
                        ),
                    },
                },
            }
            self.assertTrue(audit_intervention(payload, intervention)["passed"])


@unittest.skipIf(torch is None, "torch is unavailable")
class LegacyQueryStabilityTensorTest(unittest.TestCase):
    def test_clip_query_gradients_uses_global_l2_norm(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[3.0, 4.0]])
        total_norm = float(clip_query_gradients(model, 1.0))
        self.assertAlmostEqual(total_norm, 5.0, places=6)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(model.weight.grad)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
