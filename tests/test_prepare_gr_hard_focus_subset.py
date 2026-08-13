from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_gr_hard_focus_subset import build_focused_subset  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class GrHardFocusSubsetTest(unittest.TestCase):
    def test_includes_all_errors_and_lowest_margin_correct_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                {
                    "index": f"payload-{index}",
                    "source_index": f"source-{index}",
                    "query": f"query {index}",
                    "selected_skill_ids": [f"skill/{index}"],
                }
                for index in range(4)
            ]
            candidates = [
                {
                    "row_index": index,
                    "source_index": f"source-{index}",
                    "query_sha256": hashlib.sha256(f"query {index}".encode("utf-8")).hexdigest(),
                    "positive_skill_id": f"skill/{index}",
                    "negative_skill_ids": ["negative/a", "negative/b"],
                }
                for index in range(4)
            ]
            predictions = [
                {
                    "index": 0,
                    "query": "query 0",
                    "gold_skill_ids": ["skill/0"],
                    "ranked_skill_ids": ["negative/a", "skill/0", "negative/b"],
                    "ranked_skill_scores": {"negative/a": 2.0, "skill/0": 1.0, "negative/b": 0.0},
                },
                {
                    "index": 1,
                    "query": "query 1",
                    "gold_skill_ids": ["skill/1"],
                    "ranked_skill_ids": ["skill/1", "negative/a", "negative/b"],
                    "ranked_skill_scores": {"skill/1": 2.0, "negative/a": 1.9, "negative/b": 0.0},
                },
                {
                    "index": 2,
                    "query": "query 2",
                    "gold_skill_ids": ["skill/2"],
                    "ranked_skill_ids": ["skill/2", "negative/a", "negative/b"],
                    "ranked_skill_scores": {"skill/2": 4.0, "negative/a": 1.0, "negative/b": 0.0},
                },
                {
                    "index": 3,
                    "query": "query 3",
                    "gold_skill_ids": ["skill/3"],
                    "ranked_skill_ids": ["negative/b", "negative/a", "skill/3"],
                    "ranked_skill_scores": {"negative/b": 3.0, "negative/a": 2.0, "skill/3": 1.0},
                },
            ]
            rows_path = root / "rows.jsonl"
            candidates_path = root / "candidates.jsonl"
            predictions_path = root / "predictions.jsonl"
            write_jsonl(rows_path, rows)
            write_jsonl(candidates_path, candidates)
            write_jsonl(predictions_path, predictions)

            manifest = build_focused_subset(
                rows_path=rows_path,
                candidates_path=candidates_path,
                prediction_paths=[predictions_path],
                output_dir=root / "focused",
                expected_negatives=2,
                correct_to_error_ratio=0.5,
            )

            focused_rows = [
                json.loads(line) for line in (root / "focused/rows.jsonl").read_text().splitlines()
            ]
            focused_candidates = [
                json.loads(line)
                for line in (root / "focused/candidates.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["query"] for row in focused_rows], ["query 0", "query 1", "query 3"])
            self.assertEqual([row["row_index"] for row in focused_candidates], [0, 1, 2])
            self.assertEqual(
                [row["source_row_index"] for row in focused_candidates], [0, 1, 3]
            )
            self.assertEqual(
                [row["selection_reason"] for row in focused_candidates],
                ["top1_error", "lowest_margin_top1_correct", "top1_error"],
            )
            self.assertEqual(manifest["selection"]["top1_error_rows"], 2)
            self.assertEqual(manifest["selection"]["boundary_correct_rows"], 1)
            self.assertEqual(manifest["selection"]["focused_rows"], 3)

    def test_rejects_incomplete_prediction_closure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                {"query": "query 0", "selected_skill_ids": ["skill/0"]},
                {"query": "query 1", "selected_skill_ids": ["skill/1"]},
            ]
            candidates = [
                {
                    "row_index": index,
                    "query_sha256": hashlib.sha256(f"query {index}".encode("utf-8")).hexdigest(),
                    "positive_skill_id": f"skill/{index}",
                    "negative_skill_ids": ["negative/a"],
                }
                for index in range(2)
            ]
            predictions = [
                {
                    "index": 0,
                    "query": "query 0",
                    "gold_skill_ids": ["skill/0"],
                    "ranked_skill_ids": ["negative/a", "skill/0"],
                    "ranked_skill_scores": {"negative/a": 2.0, "skill/0": 1.0},
                }
            ]
            rows_path = root / "rows.jsonl"
            candidates_path = root / "candidates.jsonl"
            predictions_path = root / "predictions.jsonl"
            write_jsonl(rows_path, rows)
            write_jsonl(candidates_path, candidates)
            write_jsonl(predictions_path, predictions)

            with self.assertRaisesRegex(ValueError, "focused source closure failed"):
                build_focused_subset(
                    rows_path=rows_path,
                    candidates_path=candidates_path,
                    prediction_paths=[predictions_path],
                    output_dir=root / "focused",
                    expected_negatives=1,
                    correct_to_error_ratio=1.0,
                )

    def test_rejects_candidate_query_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [{"query": "query 0", "selected_skill_ids": ["skill/0"]}]
            candidates = [
                {
                    "row_index": 0,
                    "query_sha256": hashlib.sha256(b"different query").hexdigest(),
                    "positive_skill_id": "skill/0",
                    "negative_skill_ids": ["negative/a"],
                }
            ]
            predictions = [
                {
                    "index": 0,
                    "query": "query 0",
                    "gold_skill_ids": ["skill/0"],
                    "ranked_skill_ids": ["negative/a", "skill/0"],
                    "ranked_skill_scores": {"negative/a": 2.0, "skill/0": 1.0},
                }
            ]
            rows_path = root / "rows.jsonl"
            candidates_path = root / "candidates.jsonl"
            predictions_path = root / "predictions.jsonl"
            write_jsonl(rows_path, rows)
            write_jsonl(candidates_path, candidates)
            write_jsonl(predictions_path, predictions)

            with self.assertRaisesRegex(ValueError, "candidate query mismatch"):
                build_focused_subset(
                    rows_path=rows_path,
                    candidates_path=candidates_path,
                    prediction_paths=[predictions_path],
                    output_dir=root / "focused",
                    expected_negatives=1,
                    correct_to_error_ratio=0.0,
                )


if __name__ == "__main__":
    unittest.main()
