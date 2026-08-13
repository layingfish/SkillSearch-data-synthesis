import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Local source-only environments may omit torch.
    torch = None

from scripts.analyze_legacy_core75_fixed_pool import (
    analyze,
    load_system_shards,
    render_markdown,
)
from scripts.score_legacy_core75_fixed_pool import (
    aligned_prediction_panel,
    build_candidate_pools,
    exhaustive_score_candidate_pool,
    file_sha256,
    pipeline,
    score_single_candidate_stepwise,
    tokenized_query_inputs_sha256,
    tokenizer_semantic_config_sha256,
    validate_archived_scores,
)


def prediction(index, ranked, gold):
    return {
        "index": index,
        "source_index": 100 + index,
        "query": f"query-{index}",
        "gold_skill_ids": list(gold),
        "target_skill_count": len(gold),
        "ranked_skill_ids": list(ranked),
    }


class FixedPoolConstructionTest(unittest.TestCase):
    def test_node_launcher_maps_four_checkpoints_to_eight_gpus(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (
            root / "scripts" / "run_legacy_core75_fixed_pool_node_20260811.sh"
        ).read_text(encoding="utf-8")
        launch_lines = [
            line.split()
            for line in launcher.splitlines()
            if line.startswith("launch_shard ")
        ]
        self.assertEqual([int(parts[1]) for parts in launch_lines], list(range(8)))
        self.assertEqual(
            Counter(parts[2] for parts in launch_lines),
            {
                "rtx_archived": 2,
                "workspace_control": 2,
                "workspace_warmup500": 2,
                "workspace_clip1": 2,
            },
        )
        self.assertIn('if any("A800" not in name for name in gpu_names):', launcher)

    def test_position_launcher_binds_output_log_and_analysis(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (
            root
            / "scripts"
            / "run_legacy_core75_position_scores_tianzhi_20260811.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "SKILLGR_FIXED_POOL_OUTPUT_ROOT=/mnt/cfs_algo_bj/models/experiments/"
            "chenyufei03/SkillSearch-data-synthesis/experiments/"
            "legacy_core75_position_score_decomposition_20260811_v1",
            launcher,
        )
        self.assertIn(
            "SKILLGR_BOOTSTRAP_LOG=/mnt/cfs_algo_bj/models/experiments/"
            "chenyufei03/SkillSearch-data-synthesis/bootstrap_logs/"
            "legacy_core75_position_score_decomposition_20260811_v1.log",
            launcher,
        )
        self.assertIn("SKILLGR_POSITION_ANALYSIS=1", launcher)

    def test_union_pool_adds_gold_and_aligns_inputs(self):
        systems = {
            "rtx": [prediction(0, ["a", "b"], ["gold"])],
            "a800": [prediction(0, ["b", "c"], ["gold"])],
        }
        indices, indexed = aligned_prediction_panel(
            systems, expected_rows=1, expected_depth=2
        )
        pools = build_candidate_pools(
            indexed, indices, known_skill_ids={"a", "b", "c", "gold"}
        )
        self.assertEqual(pools[0], ["a", "b", "c", "gold"])

    def test_tokenizer_provenance_ignores_only_is_local_and_hashes_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.json"
            second = Path(tmpdir) / "second.json"
            first.write_text(
                json.dumps({"backend": "tokenizers", "is_local": False}),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"backend": "tokenizers", "is_local": True}),
                encoding="utf-8",
            )
            self.assertEqual(
                tokenizer_semantic_config_sha256(first),
                tokenizer_semantic_config_sha256(second),
            )

        class FakeTokenizer:
            def __call__(self, texts, **_kwargs):
                length = len(texts[0])
                return {"input_ids": [[length, 1]], "attention_mask": [[1, 1]]}

        rows = [{"index": 7, "query": "same query"}]
        first_hash = tokenized_query_inputs_sha256(
            FakeTokenizer(), rows, max_source_length=2048
        )
        second_hash = tokenized_query_inputs_sha256(
            FakeTokenizer(), rows, max_source_length=2048
        )
        self.assertEqual(first_hash, second_hash)

    def test_archived_score_replay_threshold_is_enforced(self):
        row = {
            "ranked_skill_ids": ["a", "b"],
            "ranked_skill_scores": {"a": -1.0, "b": -2.0},
        }
        audit = validate_archived_scores(
            {"a": -1.00001, "b": -2.00001}, row, atol=5e-5
        )
        self.assertTrue(audit["available"])
        self.assertTrue(all(audit["rank_set_checks"].values()))
        with self.assertRaisesRegex(ValueError, "exceeds atol"):
            validate_archived_scores(
                {"a": -1.001, "b": -2.0}, row, atol=5e-5
            )
        near_tie = {
            "ranked_skill_ids": ["a", "b"],
            "ranked_skill_scores": {"a": -1.0, "b": -1.00001},
        }
        with self.assertRaisesRegex(ValueError, "Top-K membership"):
            validate_archived_scores(
                {"a": -1.00002, "b": -0.99999}, near_tie, atol=5e-5
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class FixedPoolScoreSemanticsTest(unittest.TestCase):
    class FakeModel:
        def logits_for_next_type_from_encoder(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            target_type,
            ripor_score=False,
        ):
            del encoder_hidden_states, encoder_attention_mask, decoder_attention_mask
            assert not ripor_score
            width = 4 if target_type < 2 else 2
            base = torch.arange(width, dtype=torch.float32)[None, :] * 0.2
            prefix = decoder_input_ids.float().sum(dim=1, keepdim=True)
            coefficients = torch.linspace(0.01, 0.03, width)[None, :]
            return base + prefix * coefficients + 0.05 * target_type

    def test_exhaustive_scores_match_single_candidate_replay(self):
        semantic_ids = {"a": [0, 1], "b": [2, 3], "c": [0, 2]}
        encoder = torch.zeros((1, 1, 3))
        mask = torch.ones((1, 1), dtype=torch.long)
        model = self.FakeModel()
        ranked, scores, score_components, audit = exhaustive_score_candidate_pool(
            model,
            encoder,
            mask,
            semantic_ids,
            codebook_size=4,
            depth=2,
            dedup_size=0,
            semantic_id_mode=pipeline.SEMANTIC_ID_MODE_LEARNED,
        )
        self.assertEqual(set(ranked), set(semantic_ids))
        self.assertEqual(set(score_components), set(semantic_ids))
        self.assertEqual(audit["capacity"], 3)
        for skill_id, codes in semantic_ids.items():
            expected = score_single_candidate_stepwise(
                model,
                encoder,
                mask,
                codes,
                codebook_size=4,
                depth=2,
                dedup_size=0,
                semantic_id_mode=pipeline.SEMANTIC_ID_MODE_LEARNED,
            )
            self.assertAlmostEqual(scores[skill_id], expected, places=6)
            self.assertEqual(len(score_components[skill_id]), 3)
            self.assertAlmostEqual(
                sum(score_components[skill_id]),
                scores[skill_id],
                places=6,
            )

    def test_chunked_exhaustive_scoring_preserves_scores_and_ranking(self):
        semantic_ids = {
            "a": [0, 1],
            "b": [2, 3],
            "c": [0, 2],
            "d": [1, 0],
        }
        encoder = torch.zeros((1, 1, 3))
        mask = torch.ones((1, 1), dtype=torch.long)
        model = self.FakeModel()
        reference = exhaustive_score_candidate_pool(
            model,
            encoder,
            mask,
            semantic_ids,
            codebook_size=4,
            depth=2,
            dedup_size=0,
            semantic_id_mode=pipeline.SEMANTIC_ID_MODE_LEARNED,
        )
        chunked = exhaustive_score_candidate_pool(
            model,
            encoder,
            mask,
            semantic_ids,
            codebook_size=4,
            depth=2,
            dedup_size=0,
            semantic_id_mode=pipeline.SEMANTIC_ID_MODE_LEARNED,
            forward_batch_size=2,
            replay_prefix_size=1,
        )
        self.assertEqual(chunked[0], reference[0])
        self.assertEqual(chunked[1], reference[1])
        self.assertEqual(chunked[2], reference[2])
        self.assertEqual(chunked[3]["forward_batch_size"], 2)
        self.assertEqual(chunked[3]["replay_prefix_size"], 1)
        self.assertTrue(all(chunked[3]["forward_batch_sizes"]))


def fixed_row(label, position, raw, fixed, gold, scores):
    pool = sorted(scores)
    return {
        "label": label,
        "shard_id": position % 2,
        "row_position": position,
        "index": position,
        "source_index": 100 + position,
        "query": f"query-{position}",
        "gold_skill_ids": gold,
        "target_skill_count": len(gold),
        "pool_skill_ids": pool,
        "pool_sha256": f"pool-{position}",
        "pool_size": len(pool),
        "fixed_ranked_skill_ids": fixed,
        "fixed_ranked_skill_scores": scores,
        "raw_ranked_skill_ids": raw,
        "score_replay_validation": {
            "available": label != "rtx",
            "rank_set_checks": (
                {"@1": True, "@5": True, "@10": True, "@50": True}
                if label != "rtx"
                else {}
            ),
        },
        "exhaustive_audit": {"capacity": len(pool), "frontier_sizes": [1, len(pool)]},
    }


class FixedPoolAnalyzerTest(unittest.TestCase):
    def test_two_shards_merge_without_overlap(self):
        rows = [
            fixed_row(
                "a800",
                position,
                ["gold", "x"],
                ["gold", "x"],
                ["gold"],
                {"gold": -1.0, "x": -2.0},
            )
            for position in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dirs = []
            for shard_id, row in enumerate(rows):
                shard_dir = Path(tmpdir) / f"shard_{shard_id}"
                shard_dir.mkdir()
                score_path = shard_dir / "fixed_pool_scores.jsonl"
                score_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                manifest = {
                    "artifact_format": "legacy_core75_fixed_pool_score_shard_v1",
                    "label": "a800",
                    "shard_id": shard_id,
                    "num_shards": 2,
                    "rows": 1,
                    "expected_total_rows": 2,
                    "fixed_pool_scores_sha256": file_sha256(score_path),
                }
                (shard_dir / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                (shard_dir / "shard.done").touch()
                shard_dirs.append(shard_dir)
            merged, manifests = load_system_shards(
                "a800", shard_dirs, expected_rows=2, expected_shards=2
            )
        self.assertEqual([row["row_position"] for row in merged], [0, 1])
        self.assertEqual([row["shard_id"] for row in manifests], [0, 1])

    def test_critical_search_recovery_and_metrics(self):
        systems = {
            "rtx": [
                fixed_row(
                    "rtx",
                    0,
                    ["gold", "x", "y"],
                    ["gold", "x", "y"],
                    ["gold"],
                    {"gold": -1.0, "x": -2.0, "y": -3.0},
                ),
                fixed_row(
                    "rtx",
                    1,
                    ["x", "gold", "y"],
                    ["x", "gold", "y"],
                    ["gold"],
                    {"x": -1.0, "gold": -2.0, "y": -3.0},
                ),
            ],
            "a800": [
                fixed_row(
                    "a800",
                    0,
                    ["x", "y", "gold"],
                    ["gold", "x", "y"],
                    ["gold"],
                    {"gold": -1.0, "x": -2.0, "y": -3.0},
                ),
                fixed_row(
                    "a800",
                    1,
                    ["x", "gold", "y"],
                    ["x", "gold", "y"],
                    ["gold"],
                    {"x": -1.0, "gold": -2.0, "y": -3.0},
                ),
            ],
        }
        common = {
            "candidate_pool": {"global_sha256": "same"},
            "scoring": {"score": "same"},
            "semantic_ids_file_sha256": "semantic-file",
            "semantic_ids_logical_sha256": "semantic-logical",
            "rows_sha256": "rows",
            "prediction_inputs": {"panel": "same"},
            "implementation": {"commit": "same"},
            "runtime": {"torch": "same"},
            "checkpoint_file_sha256": "checkpoint",
            "rows": 2,
        }
        manifests = {
            "rtx": [
                {
                    **common,
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
            ],
            "a800": [
                {
                    **common,
                    "score_replay_validation": {
                        "available": True,
                        "queries": 2,
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
            ],
        }
        payload, per_query = analyze(
            systems,
            manifests,
            reference_label="rtx",
            cutoffs=(1, 5, 10, 50),
        )
        self.assertAlmostEqual(payload["systems"]["a800"]["raw_macro_recall"]["@1"], 0.0)
        self.assertAlmostEqual(payload["systems"]["a800"]["fixed_macro_recall"]["@1"], 0.5)
        critical = payload["reference_critical"]["a800"]
        self.assertEqual(critical["indices"], [0])
        self.assertEqual(critical["fixed_top1_recovered"], 1)
        self.assertEqual(len(per_query), 2)
        self.assertTrue(payload["verification"]["passed"])
        rendered = render_markdown(payload)
        self.assertIn("raw R@1/5/10/50", rendered)
        self.assertIn("Gold-mention accounting", rendered)

    def test_multi_gt_gold_mentions_reconstruct_macro_recall(self):
        systems = {
            "rtx": [
                fixed_row(
                    "rtx",
                    0,
                    ["g1", "x", "g2"],
                    ["g2", "x", "g1"],
                    ["g1", "g2"],
                    {"g2": -1.0, "x": -2.0, "g1": -3.0},
                )
            ]
        }
        manifest = {
            "candidate_pool": {"global_sha256": "same"},
            "scoring": {"score": "same"},
            "semantic_ids_file_sha256": "semantic-file",
            "semantic_ids_logical_sha256": "semantic-logical",
            "rows_sha256": "rows",
            "prediction_inputs": {"panel": "same"},
            "implementation": {"commit": "same"},
            "runtime": {"torch": "same"},
            "checkpoint_file_sha256": "checkpoint",
            "rows": 1,
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
        payload, per_query = analyze(
            systems,
            {"rtx": [manifest]},
            reference_label="rtx",
            cutoffs=(1, 5, 10, 50),
        )
        summary = payload["systems"]["rtx"]
        self.assertAlmostEqual(summary["raw_macro_recall"]["@1"], 0.5)
        self.assertAlmostEqual(summary["fixed_macro_recall"]["@1"], 0.5)
        mentions = summary["gold_mentions"]
        self.assertEqual(mentions["total"], 2)
        self.assertEqual(mentions["top1_transitions"]["raw_rank1_fixed_not"], 1)
        self.assertEqual(mentions["top1_transitions"]["raw_not_fixed_rank1"], 1)
        self.assertEqual(len(per_query[0]["systems"]["rtx"]["gold_mentions"]), 2)
        self.assertTrue(
            payload["verification"]["gold_mention_conservation"]["rtx"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
