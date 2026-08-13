import unittest
from pathlib import Path

from scripts.score_legacy_core75_full_catalog import (
    prefix_rank_diagnostics,
    summarize_rows,
)


def diagnostic_row(
    position,
    *,
    gold,
    full_ranked,
    beam_ranked,
    diagnosis,
    transition,
    full_rank,
    beam_rank,
    first_over=None,
):
    return {
        "row_position": position,
        "index": position,
        "gold_skill_ids": list(gold),
        "target_skill_count": len(gold),
        "catalog_size": 3,
        "full_catalog_top_skill_ids": list(full_ranked),
        "beam_ranked_skill_ids": list(beam_ranked),
        "top1_diagnosis": diagnosis,
        "gold_diagnostics": [
            {
                "skill_id": skill_id,
                "full_catalog_rank": full_rank,
                "beam_rank": beam_rank,
                "topk_transition": transition,
                "first_prefix_position_over_beam": first_over,
            }
            for skill_id in gold
        ],
        "score_replay_validation": {
            "max_absolute_error": 1e-6,
            "rank_set_checks": {
                "@1": True,
                "@5": True,
                "@10": True,
                "@50": True,
            },
        },
        "prefix_audit": {"maximum_shared_prefix_score_spread": 0.0},
    }


class FullCatalogDiagnosticTest(unittest.TestCase):
    def test_prefix_ranks_use_unique_prefixes_and_cumulative_scores(self):
        semantic_ids = {
            "gold": [0, 0],
            "sibling": [0, 1],
            "other": [1, 0],
        }
        components = {
            "gold": [-0.1, -0.1, 0.0],
            "sibling": [-0.1, -0.3, 0.0],
            "other": [-0.2, -0.1, 0.0],
        }
        ranks, audit = prefix_rank_diagnostics(
            semantic_ids,
            components,
            ["gold"],
            beam_width=1,
        )
        self.assertEqual(ranks["gold"], [1, 1])
        self.assertEqual(audit["unique_prefix_counts"], [2, 3])
        self.assertIsNone(
            audit["first_prefix_position_over_beam_by_gold"]["gold"]
        )
        self.assertEqual(audit["maximum_shared_prefix_score_spread"], 0.0)

    def test_summary_separates_beam_pruning_from_generator_scoring(self):
        rows = [
            diagnostic_row(
                0,
                gold=["gold"],
                full_ranked=["gold", "x", "y"],
                beam_ranked=["x", "y"],
                diagnosis="beam_pruning",
                transition="beam_pruned",
                full_rank=1,
                beam_rank=None,
                first_over=2,
            ),
            diagnostic_row(
                1,
                gold=["gold"],
                full_ranked=["x", "gold", "y"],
                beam_ranked=["x", "gold"],
                diagnosis="generator_scoring",
                transition="both_topk",
                full_rank=2,
                beam_rank=2,
            ),
            diagnostic_row(
                2,
                gold=["g1", "g2"],
                full_ranked=["g1", "x", "g2"],
                beam_ranked=["g1", "x"],
                diagnosis="both_hit",
                transition="both_topk",
                full_rank=1,
                beam_rank=1,
            ),
        ]
        summary = summarize_rows(rows, beam_width=2)
        self.assertEqual(
            summary["top1_diagnosis"]["single"],
            {"beam_pruning": 1, "generator_scoring": 1},
        )
        self.assertEqual(
            summary["gold_topk_transitions"]["single"]["at_beam_width"],
            {"beam_pruned": 1, "both_topk": 1},
        )
        self.assertEqual(
            summary["first_missing_gold_prefix_over_beam"]["single"],
            {"c1": 1},
        )
        self.assertAlmostEqual(
            summary["metrics"]["single"]["full_catalog_macro_recall"]["@1"],
            0.5,
        )
        self.assertTrue(summary["verification"]["passed"])

    def test_node_launcher_assigns_all_eight_a800_gpus(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_legacy_core75_full_catalog_node_20260813.sh"
        ).read_text(encoding="utf-8")
        launch_lines = [
            line for line in launcher.splitlines() if line.startswith("launch_shard ")
        ]
        self.assertEqual(
            launch_lines,
            [f"launch_shard {gpu}" for gpu in range(8)],
        )
        self.assertIn('if any("A800" not in name for name in gpu_names):', launcher)


if __name__ == "__main__":
    unittest.main()
