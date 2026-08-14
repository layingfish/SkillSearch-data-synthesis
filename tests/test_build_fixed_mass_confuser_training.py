from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_fixed_mass_confuser_training import build_fixed_mass_training, query_digest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class FixedMassConfuserTrainingTest(unittest.TestCase):
    def test_two_cycles_fix_pseudo_mass_and_bind_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = "skill/a"
            ordinary = "skill/b"
            negatives = ["skill/n1", "skill/n2", "skill/n3"]
            spare = "skill/n4"
            catalog_ids = [target, ordinary, *negatives, spare]
            catalog_rows = [
                {"skill_id": sid, "name": sid}
                for sid in catalog_ids
            ]
            write_jsonl(root / "catalog.jsonl", catalog_rows)
            write_jsonl(root / "targets.jsonl", [{"skill_id": target}])
            single_rows: list[dict] = [
                {
                    "index": "original-a",
                    "query": "original target query",
                    "selected_skill_ids": [target],
                    "positive_skill_id": target,
                }
            ]
            for sid in catalog_ids:
                for rank in range(5):
                    single_rows.append(
                        {
                            "index": f"pseudo:catalog_task:{sid}:{rank}",
                            "query": f"task {sid} {rank}",
                            "selected_skill_ids": [sid],
                            "positive_skill_id": sid,
                            "pseudo_query": {"mode": "task_spec", "rank": rank},
                        }
                    )
            for mode in ("outcome_first", "workflow_embedded", "constraint_failure"):
                single_rows.append(
                    {
                        "index": f"pseudo:zero_multi_mode:{mode}:{target}",
                        "query": f"mode {mode}",
                        "selected_skill_ids": [target],
                        "positive_skill_id": target,
                        "pseudo_query": {"mode": mode},
                    }
                )
            write_jsonl(root / "single.jsonl", single_rows)
            legacy_candidates = []
            for index, row in enumerate(single_rows):
                positive = row["positive_skill_id"]
                legacy_candidates.append(
                    {
                        "row_index": index,
                        "positive_skill_id": positive,
                        "query_sha256": query_digest(row),
                        "negative_skill_ids": [sid for sid in catalog_ids if sid != positive][:3],
                    }
                )
            write_jsonl(root / "legacy.jsonl", legacy_candidates)

            target_pseudo = [
                row
                for row in single_rows
                if row["positive_skill_id"] == target and str(row["index"]).startswith("pseudo:")
            ]
            mining_rows = [
                {**row, "index": index, "source_index": row["index"]}
                for index, row in enumerate(target_pseudo)
            ]
            current_candidates = []
            for index, row in enumerate(mining_rows):
                ranked = negatives if index < 5 else [negatives[1], negatives[0], negatives[2]]
                current_candidates.append(
                    {
                        "row_index": index,
                        "source_index": row["source_index"],
                        "positive_skill_id": target,
                        "query_sha256": query_digest(row),
                        "negative_skill_ids": ranked,
                    }
                )
            write_jsonl(root / "mining.jsonl", mining_rows)
            write_jsonl(root / "current.jsonl", current_candidates)
            # Reciprocal-rank aggregation puts n1, n2, n3 in this order.
            write_jsonl(
                root / "pairs.jsonl",
                [
                    {
                        "target_skill_id": target,
                        "confuser_skill_id": sid,
                        "confuser_rank": rank,
                    }
                    for rank, sid in enumerate(negatives, start=1)
                ],
            )
            new_rows = [
                {
                    "index": f"pseudo:confuser_contrastive:{target}:{rank - 1}",
                    "query": f"contrastive query {rank}",
                    "selected_skill_ids": [target],
                    "positive_skill_id": target,
                    "pseudo_query": {
                        "mode": "confuser_contrastive",
                        "rank": rank - 1,
                        "confuser_skill_id": sid,
                        "confuser_rank": rank,
                    },
                }
                for rank, sid in enumerate(negatives, start=1)
            ]
            write_jsonl(root / "new.jsonl", new_rows)

            manifest = build_fixed_mass_training(
                single_rows_path=root / "single.jsonl",
                new_rows_path=root / "new.jsonl",
                target_skills_path=root / "targets.jsonl",
                catalog_path=root / "catalog.jsonl",
                legacy_candidates_path=root / "legacy.jsonl",
                current_mining_rows_path=root / "mining.jsonl",
                current_candidates_path=root / "current.jsonl",
                confuser_pairs_path=root / "pairs.jsonl",
                output_dir=root / "output",
                expected_catalog_skills=6,
                expected_targets=1,
                expected_negatives=3,
                cycles=2,
                pseudo_views_per_skill_per_cycle=5,
            )
            output_rows = [
                json.loads(line)
                for line in (root / "output/train_rows.two_cycle_fixed_mass.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            output_candidates = [
                json.loads(line)
                for line in (root / "output/candidates.two_cycle_fixed_mass.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(manifest["output_rows"], 62)
            self.assertEqual(len(output_rows), len(output_candidates))
            self.assertEqual(manifest["contrastive_exposures_per_target"], 3)
            for cycle in range(2):
                cycle_pseudo = [
                    row
                    for row in output_rows
                    if row["fixed_mass"]["cycle"] == cycle
                    and row["fixed_mass"]["source_kind"] != "original"
                ]
                self.assertEqual(len(cycle_pseudo), 6 * 5)
                self.assertEqual(
                    {sid: sum(row["positive_skill_id"] == sid for row in cycle_pseudo) for sid in catalog_ids},
                    {sid: 5 for sid in catalog_ids},
                )
            target_pseudo_output = [
                row
                for row in output_rows
                if row["positive_skill_id"] == target
                and row["fixed_mass"]["source_kind"] != "original"
            ]
            self.assertEqual(len(target_pseudo_output), 10)
            self.assertEqual(
                sum(row["fixed_mass"]["source_kind"] == "contrastive" for row in target_pseudo_output),
                3,
            )
            for index, (row, candidate) in enumerate(zip(output_rows, output_candidates)):
                self.assertEqual(row["index"], index)
                self.assertEqual(candidate["row_index"], index)
                self.assertEqual(candidate["query_sha256"], query_digest(row))
                self.assertEqual(len(candidate["negative_skill_ids"]), 3)
                self.assertEqual(len(set(candidate["negative_skill_ids"])), 3)
                self.assertNotIn(row["positive_skill_id"], candidate["negative_skill_ids"])
                if row["fixed_mass"]["source_kind"] == "contrastive":
                    self.assertEqual(
                        candidate["negative_skill_ids"][0],
                        row["pseudo_query"]["confuser_skill_id"],
                    )


if __name__ == "__main__":
    unittest.main()
