import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # Local source-only environments may omit NumPy.
    np = None

if np is not None:
    from scripts.analyze_legacy_core75_position_scores import (
        COMPONENT_LABELS,
        analyze_position_scores,
        baseline_metrics,
        cross_validate_weights,
        descending_ranks,
        grid_query_metrics,
        normalized_weight_grid,
        selected_weight_payload,
        validate_component_rows,
    )


def position_row(position, ranked, gold, components):
    scores = {
        skill_id: sum(float(value) for value in values)
        for skill_id, values in components.items()
    }
    return {
        "label": "a800",
        "row_position": position,
        "index": position,
        "gold_skill_ids": list(gold),
        "target_skill_count": len(gold),
        "pool_skill_ids": sorted(components),
        "pool_size": len(components),
        "fixed_ranked_skill_ids": list(ranked),
        "fixed_ranked_skill_scores": scores,
        "fixed_ranked_skill_score_components": components,
    }


@unittest.skipIf(np is None, "NumPy is unavailable")
class PositionScoreProtocolTest(unittest.TestCase):
    def test_normalized_grid_is_frozen_and_contains_uniform_once(self):
        grid = normalized_weight_grid()
        self.assertEqual(len(grid), 3061)
        self.assertEqual(grid.count((1.0, 1.0, 1.0, 1.0, 1.0)), 1)
        self.assertEqual(descending_ranks([2.0, 2.0, 1.0]), [1.5, 1.5, 3.0])

    def test_component_sum_and_uniform_rank_are_exact(self):
        row = position_row(
            0,
            ["a", "b", "c"],
            ["a"],
            {
                "a": [-0.1, -0.2, -0.3, -0.4, -0.5],
                "b": [-0.2, -0.3, -0.4, -0.5, -0.6],
                "c": [-0.3, -0.4, -0.5, -0.6, -0.7],
            },
        )
        audit = validate_component_rows("a800", [row])
        self.assertTrue(audit["uniform_rank_exact"])
        self.assertLessEqual(audit["maximum_sum_absolute_error"], 1e-6)

    def test_weight_selection_recovers_synthetic_top1_and_cv_splits(self):
        rows = [
            position_row(
                position,
                ["x", "g"],
                ["g"],
                {
                    "x": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "g": [1.0, 0.0, 0.0, 0.0, -2.0],
                },
            )
            for position in range(10)
        ]
        weight_rows = normalized_weight_grid()
        weights = np.asarray(weight_rows, dtype=np.float64)
        query_recall, query_correct = grid_query_metrics(rows, weights)
        uniform_index = weight_rows.index((1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(float(query_recall[:, uniform_index, 0].mean()), 0.0)
        selected, _ = selected_weight_payload(
            rows, weights, query_recall, query_correct
        )
        self.assertEqual(selected["metrics"]["@1"], 1.0)
        self.assertEqual(selected["selection_status"], "deep_gate_candidate")
        self.assertTrue(selected["eligible_candidate"])
        self.assertEqual(
            selected["query_top1_transitions"]["baseline_wrong_weighted_correct"],
            10,
        )
        self.assertTrue(selected["gold_mention_top1_transitions"]["passed"])
        cross_validation = cross_validate_weights(
            rows, weights, query_recall, folds=5
        )
        self.assertEqual(
            [(fold["train_queries"], fold["heldout_queries"]) for fold in cross_validation["folds"]],
            [(8, 2)] * 5,
        )
        self.assertEqual(cross_validation["aggregate_heldout_metrics"]["@1"], 1.0)

    def test_no_deep_gate_candidate_is_reported_ineligible(self):
        ranked = [f"x-{position}" for position in range(59)] + ["gold"]
        components = {
            skill_id: [-float(position)] * 5
            for position, skill_id in enumerate(ranked)
        }
        rows = [position_row(0, ranked, ["gold"], components)]
        weight_rows = normalized_weight_grid()
        weights = np.asarray(weight_rows, dtype=np.float64)
        query_recall, query_correct = grid_query_metrics(rows, weights)
        selected, _ = selected_weight_payload(
            rows, weights, query_recall, query_correct
        )
        self.assertEqual(
            selected["selection_status"],
            "best_effort_no_deep_gate_candidate",
        )
        self.assertFalse(selected["eligible_candidate"])

    def test_multi_gt_macro_recall_matches_uniform_grid(self):
        row = position_row(
            0,
            ["g1", "x", "g2"],
            ["g1", "g2"],
            {
                "g1": [0.0, 0.0, 0.0, 0.0, 0.0],
                "x": [-1.0, 0.0, 0.0, 0.0, 0.0],
                "g2": [-2.0, 0.0, 0.0, 0.0, 0.0],
            },
        )
        weights = np.ones((1, 5), dtype=np.float64)
        query_recall, _ = grid_query_metrics([row], weights)
        baseline = baseline_metrics([row])
        self.assertEqual(float(query_recall[0, 0, 0]), 0.5)
        for position, cutoff in enumerate((1, 5, 10, 50)):
            self.assertEqual(
                float(query_recall[0, 0, position]), baseline[f"@{cutoff}"]
            )

    def test_end_to_end_analysis_checks_folds_and_grid_rows(self):
        systems = {"rtx": [], "a800": []}
        for position in range(75):
            for label in systems:
                row = position_row(
                    position,
                    ["g", "x"],
                    ["g"],
                    {
                        "g": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "x": [-1.0, 0.0, 0.0, 0.0, 0.0],
                    },
                )
                row.update(
                    {
                        "label": label,
                        "shard_id": position % 2,
                        "source_index": 1000 + position,
                        "query": f"query-{position}",
                        "pool_sha256": f"pool-{position}",
                        "raw_ranked_skill_ids": ["g", "x"],
                        "score_replay_validation": {
                            "available": label != "rtx",
                            "rank_set_checks": (
                                {"@1": True, "@5": True, "@10": True, "@50": True}
                                if label != "rtx"
                                else {}
                            ),
                        },
                        "exhaustive_audit": {
                            "capacity": 2,
                            "frontier_sizes": [1, 2],
                        },
                    }
                )
                systems[label].append(row)
        common_manifest = {
            "artifact_format": "legacy_core75_fixed_pool_score_shard_v2",
            "candidate_pool": {"global_sha256": "same-pool"},
            "model_inputs": {
                "base_model": {
                    "path": "base",
                    "config_json_sha256": "base-config",
                    "model_safetensors_sha256": "base-weights",
                },
                "tokenizer": {
                    "path": "tokenizer",
                    "tokenizer_json_sha256": "tokenizer-json",
                    "tokenizer_config_json_sha256": "tokenizer-config",
                    "tokenizer_semantic_config_sha256": "tokenizer-semantic",
                    "tokenized_query_inputs_sha256": "tokenized-inputs",
                },
            },
            "scoring": {
                "component_labels": list(COMPONENT_LABELS),
                "score": "same",
            },
            "semantic_ids_file_sha256": "semantic-file",
            "semantic_ids_logical_sha256": "semantic-logical",
            "rows_sha256": "rows",
            "prediction_inputs": {"panel": "same"},
            "implementation": {"commit": "same"},
            "runtime": {"torch": "same"},
            "checkpoint_file_sha256": "checkpoint",
            "rows": 75,
        }
        rtx_manifest = {
            **common_manifest,
            "score_replay_validation": {
                "available": False,
                "queries": 0,
                "maximum_absolute_error": None,
                "atol": 5e-5,
                "rank_set_checks": {
                    "@1": None,
                    "@5": None,
                    "@10": None,
                    "@50": None,
                },
            },
        }
        a800_manifest = {
            **common_manifest,
            "score_replay_validation": {
                "available": True,
                "queries": 75,
                "maximum_absolute_error": 1e-6,
                "atol": 5e-5,
                "rank_set_checks": {
                    "@1": True,
                    "@5": True,
                    "@10": True,
                    "@50": True,
                },
            },
        }
        payload, per_query, grid_rows = analyze_position_scores(
            systems,
            {"rtx": [rtx_manifest], "a800": [a800_manifest]},
            reference_label="rtx",
        )
        self.assertTrue(payload["verification"]["passed"])
        self.assertEqual(len(per_query), 75)
        self.assertEqual(len(grid_rows), 3061)
        self.assertEqual(
            [(row["train"], row["heldout"]) for row in payload["verification"]["fold_sizes"]],
            [(60, 15)] * 5,
        )


if __name__ == "__main__":
    unittest.main()
