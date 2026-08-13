from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_gr_self_mined_hard_negatives import (  # noqa: E402
    compact_self_mined_candidates,
    shard_training_rows,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class GrSelfMinedHardNegativesTest(unittest.TestCase):
    def training_rows(self) -> list[dict]:
        return [
            {
                "index": 100 + index,
                "source_index": 200 + index,
                "query": f"query {index}",
                "selected_skill_ids": [f"skill/{index % 3}"],
            }
            for index in range(5)
        ]

    def test_shards_by_training_ordinal_and_preserves_payload_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_path = root / "rows.jsonl"
            write_jsonl(rows_path, self.training_rows())

            manifest = shard_training_rows(rows_path, root / "shards", num_shards=2)

            shard_zero = [json.loads(line) for line in (root / "shards/shard_00.jsonl").read_text().splitlines()]
            shard_one = [json.loads(line) for line in (root / "shards/shard_01.jsonl").read_text().splitlines()]
            self.assertEqual([row["index"] for row in shard_zero], [0, 2, 4])
            self.assertEqual([row["index"] for row in shard_one], [1, 3])
            self.assertEqual([row["_self_mining_payload_index"] for row in shard_zero], [100, 102, 104])
            self.assertEqual(manifest["row_count"], 5)
            self.assertEqual([row["rows"] for row in manifest["shards"]], [3, 2])

    def test_compacts_ranked_gr_outputs_into_exact_non_gold_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = self.training_rows()[:3]
            rows_path = root / "rows.jsonl"
            write_jsonl(rows_path, rows)
            semantic_path = root / "semantic_ids.jsonl"
            write_jsonl(
                semantic_path,
                [
                    {
                        "skill_id": f"skill/{index}",
                        "semantic_id": [index, index + 1, index + 2, index + 3],
                        "semantic_id_mode": "learned_codes",
                    }
                    for index in range(4)
                ],
            )
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "structured_decoder.pt").write_bytes(b"checkpoint")

            prediction_paths = []
            for shard_index, row_indices in enumerate(([0, 2], [1])):
                eval_dir = root / f"eval_{shard_index}"
                prediction_path = eval_dir / "predictions.jsonl"
                prediction_paths.append(prediction_path)
                prediction_rows = []
                for row_index in row_indices:
                    positive = rows[row_index]["selected_skill_ids"][0]
                    ranked = ["skill/3", positive, "skill/2", "skill/1", "skill/0"]
                    prediction_rows.append(
                        {
                            "index": row_index,
                            "source_index": rows[row_index]["source_index"],
                            "query": rows[row_index]["query"],
                            "gold_skill_ids": [positive],
                            "ranked_skill_ids": ranked,
                        }
                    )
                write_jsonl(prediction_path, prediction_rows)
                (eval_dir / "eval_manifest.json").write_text(
                    json.dumps(
                        {
                            "model_dir": str(model_dir),
                            "num_beams": 3,
                            "max_skills": 1,
                            "rank_aggregation": "beam_order",
                            "decoder_head": "ce",
                            "score_transform": "logprob",
                            "distributed_world_size": 1,
                        }
                    ),
                    encoding="utf-8",
                )

            manifest = compact_self_mined_candidates(
                rows_path=rows_path,
                prediction_paths=prediction_paths,
                semantic_ids_path=semantic_path,
                model_dir=model_dir,
                output_dir=root / "candidates",
                expected_negatives=2,
                expected_beams=3,
            )

            candidates = [
                json.loads(line)
                for line in (root / "candidates/candidates.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["row_index"] for row in candidates], [0, 1, 2])
            for row in candidates:
                self.assertEqual(len(row["negative_skill_ids"]), 2)
                self.assertNotIn(row["positive_skill_id"], row["negative_skill_ids"])
            self.assertEqual(manifest["audit"]["rows"], 3)
            self.assertEqual(manifest["audit"]["gold_in_beam_rows"], 3)

    def test_compaction_fails_when_beam_cannot_supply_exact_negative_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_path = root / "rows.jsonl"
            write_jsonl(rows_path, self.training_rows()[:1])
            semantic_path = root / "semantic_ids.jsonl"
            write_jsonl(
                semantic_path,
                [
                    {
                        "skill_id": f"skill/{index}",
                        "semantic_id": [index, index, index, index],
                        "semantic_id_mode": "learned_codes",
                    }
                    for index in range(3)
                ],
            )
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "structured_decoder.pt").write_bytes(b"checkpoint")
            eval_dir = root / "eval"
            write_jsonl(
                eval_dir / "predictions.jsonl",
                [
                    {
                        "index": 0,
                        "query": "query 0",
                        "gold_skill_ids": ["skill/0"],
                        "ranked_skill_ids": ["skill/0", "skill/1"],
                    }
                ],
            )
            (eval_dir / "eval_manifest.json").write_text(
                json.dumps(
                    {
                        "model_dir": str(model_dir),
                        "num_beams": 3,
                        "max_skills": 1,
                        "rank_aggregation": "beam_order",
                        "decoder_head": "ce",
                        "score_transform": "logprob",
                        "distributed_world_size": 1,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "insufficient unique GR near-misses"):
                compact_self_mined_candidates(
                    rows_path=rows_path,
                    prediction_paths=[eval_dir / "predictions.jsonl"],
                    semantic_ids_path=semantic_path,
                    model_dir=model_dir,
                    output_dir=root / "candidates",
                    expected_negatives=2,
                    expected_beams=3,
                )


if __name__ == "__main__":
    unittest.main()
