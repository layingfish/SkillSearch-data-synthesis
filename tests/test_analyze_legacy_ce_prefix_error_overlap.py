import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_legacy_ce_prefix_error_overlap import analyze_overlap


def ranking(prefix: str, first: list[str]) -> list[str]:
    result = list(first)
    candidate = 0
    while len(result) < 50:
        skill_id = f"{prefix}/candidate-{candidate}"
        candidate += 1
        if skill_id not in result:
            result.append(skill_id)
    return result


def row(index: int, gold: list[str], ranked: list[str]) -> dict:
    return {
        "index": index,
        "source_index": f"source-{index}",
        "query": f"query-{index}",
        "gold_skill_ids": gold,
        "target_skill_count": len(gold),
        "predicted_skill_ids": [ranked[0]],
        "ranked_skill_ids": ranked,
        "ranked_skill_scores": {
            skill_id: -float(rank) for rank, skill_id in enumerate(ranked)
        },
    }


class LegacyCePrefixErrorOverlapTest(unittest.TestCase):
    def setUp(self):
        gold = [["a"], ["b"], ["c", "d"], ["e", "f"]]
        self.systems = {
            "control": [
                row(0, gold[0], ranking("c0", ["a", "x0"])),
                row(1, gold[1], ranking("c1", ["x1", "b"])),
                row(2, gold[2], ranking("c2", ["c", "x2", "d"])),
                row(3, gold[3], ranking("c3", ["x3", "e", "f"])),
            ],
            "sequence": [
                row(0, gold[0], ranking("s0", ["a", "x0"])),
                row(1, gold[1], ranking("s1", ["b", "x1"])),
                row(2, gold[2], ranking("s2", ["x2", "c", "d"])),
                row(3, gold[3], ranking("s3", ["e", "f", "x3"])),
            ],
            "cumulative": [
                row(0, gold[0], ranking("u0", ["x0", "a"])),
                row(1, gold[1], ranking("u1", ["x1", "b"])),
                row(2, gold[2], ranking("u2", ["d", "c", "x2"])),
                row(3, gold[3], ranking("u3", ["f", "e", "x3"])),
            ],
        }

    def _write_inputs(self, root: Path):
        input_paths = {}
        metric_paths = {}
        expected_r1 = {"control": 0.375, "sequence": 0.625, "cumulative": 0.25}
        for label, rows in self.systems.items():
            prediction_path = root / f"{label}.jsonl"
            prediction_path.write_text(
                "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
            )
            summary_path = root / f"{label}.summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "overall": {
                            "@1": {"macro_recall": expected_r1[label]},
                            "@10": {"macro_recall": 1.0},
                            "@50": {"macro_recall": 1.0},
                        }
                    }
                ),
                encoding="utf-8",
            )
            input_paths[label] = prediction_path
            metric_paths[label] = summary_path
        return input_paths, metric_paths

    def test_patterns_oracles_and_transition_conservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_paths, metric_paths = self._write_inputs(Path(temporary))
            payload, per_query = analyze_overlap(
                self.systems,
                input_paths=input_paths,
                metric_paths=metric_paths,
                expected_rows=4,
                expected_depth=50,
            )

        overall = payload["subsets"]["overall"]
        self.assertEqual(
            overall["top1_hit_pattern_indices"],
            {"011": [3], "101": [2], "110": [0], "010": [1]},
        )
        self.assertAlmostEqual(overall["per_query_oracle"]["macro_recall"]["@1"], 0.75)
        self.assertAlmostEqual(
            overall["gold_cardinality_router"]["macro_recall"]["@1"], 0.75
        )
        control_to_sequence = overall["pairs"]["control_to_sequence"]
        self.assertEqual(
            control_to_sequence["cutoffs"]["@1"]["gained_query_indices"], [1, 3]
        )
        self.assertEqual(
            control_to_sequence["cutoffs"]["@1"]["lost_query_indices"], [2]
        )
        self.assertTrue(
            control_to_sequence["cutoffs"]["@1"]["transition_delta_matches"]
        )
        self.assertEqual(per_query[1]["top1_hit_pattern"], "010")
        self.assertTrue(payload["verification"]["passed"])

    def test_alignment_drift_fails_closed(self):
        self.systems["sequence"][1]["query"] = "drifted"
        with tempfile.TemporaryDirectory() as temporary:
            input_paths, metric_paths = self._write_inputs(Path(temporary))
            with self.assertRaisesRegex(ValueError, "query differs"):
                analyze_overlap(
                    self.systems,
                    input_paths=input_paths,
                    metric_paths=metric_paths,
                    expected_rows=4,
                    expected_depth=50,
                )


if __name__ == "__main__":
    unittest.main()
