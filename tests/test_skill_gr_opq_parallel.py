import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import skill_gr_opq_parallel as opq_parallel  # noqa: E402


class ParallelMtpLossTest(unittest.TestCase):
    def test_depth_mean_keeps_duplicate_position_loss_scale(self):
        one_position = torch.tensor([[[2.0, 0.0, -1.0]]], requires_grad=True)
        one_label = torch.tensor([[0]])
        one_sum, one_count, _ = opq_parallel.parallel_mtp_loss_components(
            one_position, one_label
        )

        duplicated = one_position.detach().repeat(1, 4, 1).requires_grad_(True)
        duplicated_labels = one_label.repeat(1, 4)
        duplicate_sum, duplicate_count, _ = opq_parallel.parallel_mtp_loss_components(
            duplicated, duplicated_labels
        )

        self.assertEqual(float(one_count), 1.0)
        self.assertEqual(float(duplicate_count), 1.0)
        torch.testing.assert_close(one_sum, duplicate_sum)
        duplicate_sum.backward()
        self.assertTrue(torch.isfinite(duplicated.grad).all())

    def test_zero_weight_tail_is_graph_connected_but_not_counted(self):
        logits = torch.randn(1, 3, 5, requires_grad=True)
        labels = torch.tensor([[0, 1, 2]])
        loss_sum, count, _ = opq_parallel.parallel_mtp_loss_components(
            logits, labels, torch.zeros(1)
        )
        self.assertEqual(float(loss_sum), 0.0)
        self.assertEqual(float(count), 0.0)
        loss_sum.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)


class ParallelHeadContractTest(unittest.TestCase):
    def test_legacy_linear_architecture_name_is_preserved(self):
        self.assertEqual(opq_parallel.ARCHITECTURE, opq_parallel.LINEAR_ARCHITECTURE)
        self.assertEqual(
            opq_parallel.architecture_for_head(opq_parallel.LINEAR_HEAD),
            "t5_decoder_start_parallel_position_heads_v1",
        )

    def test_rpg_head_has_distinct_checkpoint_architecture(self):
        self.assertEqual(
            opq_parallel.architecture_for_head(opq_parallel.RPG_HEAD),
            "t5_decoder_start_rpg_cosine_residual_heads_v1",
        )
        with self.assertRaisesRegex(ValueError, "unknown OPQ parallel head"):
            opq_parallel.architecture_for_head("not-a-head")

    def test_calibrated_rpg_head_has_distinct_objective_and_scoring(self):
        head = opq_parallel.RPG_CALIBRATED_HEAD
        self.assertEqual(
            opq_parallel.architecture_for_head(head),
            "t5_decoder_start_rpg_cosine_residual_position_calibration_v1",
        )
        self.assertEqual(
            opq_parallel.training_objective_for_head(head),
            opq_parallel.CALIBRATED_TRAINING_OBJECTIVE,
        )
        self.assertEqual(
            opq_parallel.catalog_scoring_for_head(head),
            opq_parallel.CALIBRATED_CATALOG_SCORING,
        )

    def test_rpg_checkpoint_can_initialize_only_the_calibrated_upgrade(self):
        class ToyCalibrated(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = torch.nn.Parameter(torch.zeros(2))
                self.position_log_temperatures = torch.nn.Parameter(torch.zeros(2))
                self.position_weight_logits = torch.nn.Parameter(torch.zeros(2))

        checkpoint = {
            "head_type": opq_parallel.RPG_HEAD,
            "architecture": opq_parallel.RPG_ARCHITECTURE,
            "catalog_scoring": opq_parallel.CATALOG_SCORING,
            "base_model": "base",
            "depth": 2,
            "codebook_size": 3,
            "semantic_ids_sha256": "sid-sha",
            "checkpoint_role": "final",
            "best_query_step": 10,
            "model_state_dict": {"shared": torch.ones(2)},
        }
        model = ToyCalibrated()
        report = opq_parallel.load_initializer_checkpoint(
            model,
            checkpoint,
            target_head_type=opq_parallel.RPG_CALIBRATED_HEAD,
            base_model="base",
            depth=2,
            codebook_size=3,
            semantic_ids_sha256="sid-sha",
        )
        self.assertTrue(report["upgrade"])
        torch.testing.assert_close(model.shared, torch.ones(2))
        with self.assertRaisesRegex(
            ValueError, "unsupported initializer head transition"
        ):
            opq_parallel.load_initializer_checkpoint(
                model,
                checkpoint,
                target_head_type=opq_parallel.LINEAR_HEAD,
                base_model="base",
                depth=2,
                codebook_size=3,
                semantic_ids_sha256="sid-sha",
            )

    def test_tiny_t5_calibrated_upgrade_preserves_logits_and_backpropagates(self):
        from transformers import T5Config, T5ForConditionalGeneration

        config = T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dropout_rate=0.0,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )

        def tiny_backbone(_):
            return T5ForConditionalGeneration(config)

        with mock.patch(
            "transformers.AutoModelForSeq2SeqLM.from_pretrained",
            side_effect=tiny_backbone,
        ):
            source = opq_parallel.make_parallel_t5_model(
                "tiny", 2, 4, head_type=opq_parallel.RPG_HEAD
            )
            checkpoint = {
                "head_type": opq_parallel.RPG_HEAD,
                "architecture": opq_parallel.RPG_ARCHITECTURE,
                "catalog_scoring": opq_parallel.CATALOG_SCORING,
                "base_model": "tiny",
                "depth": 2,
                "codebook_size": 4,
                "semantic_ids_sha256": "sid-sha",
                "checkpoint_role": "final",
                "model_state_dict": source.state_dict(),
            }
            calibrated = opq_parallel.make_parallel_t5_model(
                "tiny", 2, 4, head_type=opq_parallel.RPG_CALIBRATED_HEAD
            )
        opq_parallel.load_initializer_checkpoint(
            calibrated,
            checkpoint,
            target_head_type=opq_parallel.RPG_CALIBRATED_HEAD,
            base_model="tiny",
            depth=2,
            codebook_size=4,
            semantic_ids_sha256="sid-sha",
        )
        source.eval()
        calibrated.eval()
        input_ids = torch.tensor([[3, 4, 1], [5, 6, 1]])
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            source_logits = source(input_ids, attention_mask)
            calibrated_logits, initial_weights = calibrated(
                input_ids, attention_mask, return_position_weights=True
            )
        torch.testing.assert_close(source_logits, calibrated_logits)
        torch.testing.assert_close(initial_weights, torch.full((2,), 0.5))

        logits, position_weights = calibrated(
            input_ids, attention_mask, return_position_weights=True
        )
        catalog = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
        catalog_sum, _, _ = opq_parallel.exact_catalog_nll_components(
            logits,
            catalog,
            torch.tensor([0, 1]),
            position_weights=position_weights,
        )
        token_sum, _, _ = opq_parallel.parallel_mtp_loss_components(
            logits, torch.tensor([[0, 0], [1, 1]])
        )
        (token_sum + 0.1 * catalog_sum).backward()
        self.assertTrue(torch.isfinite(calibrated.position_log_temperatures.grad).all())
        self.assertTrue(torch.isfinite(calibrated.position_weight_logits.grad).all())


class ExactCatalogScoringTest(unittest.TestCase):
    def test_scores_all_positions_and_ranks_with_stable_ties(self):
        logits = torch.tensor(
            [
                [
                    [5.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0],
                ]
            ]
        )
        catalog = torch.tensor(
            [
                [0, 1],
                [0, 0],
                [2, 1],
                [2, 2],
            ]
        )
        scores = opq_parallel.exact_catalog_log_scores(logits, catalog)[0]
        ranked, ranked_scores = opq_parallel.stable_topk_catalog(
            scores.tolist(), ["best", "tie-b", "tie-a", "worst"], 4
        )
        self.assertEqual(ranked[0], "best")
        self.assertEqual(ranked[-1], "worst")
        self.assertEqual(ranked[1:3], ["tie-a", "tie-b"])
        self.assertGreater(ranked_scores[0], ranked_scores[1])

    def test_simultaneous_position_permutation_is_invariant(self):
        generator = torch.Generator().manual_seed(7)
        logits = torch.randn(3, 5, 11, generator=generator)
        codes = torch.randint(0, 11, (17, 5), generator=generator)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        original = opq_parallel.exact_catalog_log_scores(logits, codes)
        permuted = opq_parallel.exact_catalog_log_scores(
            logits.index_select(1, permutation),
            codes.index_select(1, permutation),
        )
        torch.testing.assert_close(original, permuted)

    def test_nonuniform_position_weights_change_ranking_and_remain_differentiable(self):
        logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]], requires_grad=True)
        catalog = torch.tensor([[0, 0], [1, 1]])
        uniform_scores = opq_parallel.exact_catalog_log_scores(logits, catalog)[0]
        torch.testing.assert_close(uniform_scores[0], uniform_scores[1])

        weight_logits = torch.tensor([2.0, -2.0], requires_grad=True)
        weights = torch.softmax(weight_logits, dim=0)
        weighted_scores = opq_parallel.exact_catalog_log_scores(
            logits, catalog, position_weights=weights
        )[0]
        self.assertGreater(float(weighted_scores[0]), float(weighted_scores[1]))
        loss_sum, count, _ = opq_parallel.exact_catalog_nll_components(
            logits,
            catalog,
            torch.tensor([0]),
            position_weights=weights,
        )
        self.assertEqual(float(count), 1.0)
        (
            loss_sum + 0.01 * opq_parallel.position_weight_kl_to_uniform(weights)
        ).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(weight_logits.grad).all())
        self.assertGreater(float(weight_logits.grad.abs().sum()), 0.0)

    def test_position_weight_validation_rejects_invalid_shapes_and_values(self):
        logits = torch.zeros(1, 2, 2)
        catalog = torch.tensor([[0, 0]])
        with self.assertRaisesRegex(ValueError, "\[depth\]"):
            opq_parallel.exact_catalog_log_scores(
                logits, catalog, position_weights=torch.ones(1)
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            opq_parallel.exact_catalog_log_scores(
                logits, catalog, position_weights=torch.tensor([1.0, -1.0])
            )


class LearningRateScheduleTest(unittest.TestCase):
    def test_cosine_hits_frozen_endpoints_and_is_monotonic(self):
        values = [
            opq_parallel.scheduled_learning_rate(
                step=step,
                total_steps=5,
                peak_lr=1e-4,
                min_lr=1e-5,
                schedule="cosine",
            )
            for step in range(1, 6)
        ]
        self.assertAlmostEqual(values[0], 1e-4)
        self.assertAlmostEqual(values[-1], 1e-5)
        self.assertEqual(values, sorted(values, reverse=True))

    def test_constant_schedule_preserves_peak(self):
        self.assertEqual(
            opq_parallel.scheduled_learning_rate(
                step=3,
                total_steps=5,
                peak_lr=3e-4,
                min_lr=0.0,
                schedule="constant",
            ),
            3e-4,
        )


class SemanticIdContractTest(unittest.TestCase):
    def test_accepts_unique_learned_codes(self):
        audit = opq_parallel.validate_parallel_semantic_ids(
            {"a": [0, 1], "b": [1, 0]},
            "learned_codes",
            depth=2,
            codebook_size=2,
            expected_catalog_size=2,
        )
        self.assertEqual(audit["unique_full_tuples"], 2)

    def test_rejects_collisions_and_autoregressive_dedup_mode(self):
        with self.assertRaisesRegex(ValueError, "collision-free"):
            opq_parallel.validate_parallel_semantic_ids(
                {"a": [0, 1], "b": [0, 1]},
                "learned_codes",
                depth=2,
                codebook_size=2,
            )
        with self.assertRaisesRegex(ValueError, "learned_codes"):
            opq_parallel.validate_parallel_semantic_ids(
                {"a": [0, 1, 0]},
                "rq_codes_with_dedup",
                depth=2,
                codebook_size=2,
            )


if __name__ == "__main__":
    unittest.main()
