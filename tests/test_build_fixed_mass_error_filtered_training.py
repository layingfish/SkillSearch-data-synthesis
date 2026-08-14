from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_confuser_aware_queries import MODE3_PREFIX, TASK5_PREFIX, file_sha256
from scripts.build_diverse_coverage_queries import COVERAGE_MODES, COVERAGE_PREFIX
from scripts.build_fixed_mass_confuser_training import query_digest
from scripts.build_fixed_mass_error_filtered_training import (
    ERROR_SOURCE_KIND,
    build_fixed_mass_error_filtered_training,
    target_cycle_views,
)


MODE3_MODES = ("outcome_first", "workflow_embedded", "constraint_failure")


def row(index: object, positive: str, *, mode: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "index": index,
        "query": f"query::{index}",
        "query_text": f"instruction\nQuery:query::{index}",
        "selected_skill_ids": [positive],
        "positive_skill_id": positive,
    }
    if mode is not None:
        payload["pseudo_query"] = {"mode": mode}
    return payload


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in rows),
        encoding="utf-8",
    )


class FixedMassErrorFilteredTrainingTest(unittest.TestCase):
    def test_target_cycle_views_places_zero_through_six_errors_without_within_cycle_duplicates(self) -> None:
        target = "skill/a"
        task5 = [row(f"{TASK5_PREFIX}{target}:{rank}", target) for rank in range(5)]
        mode3 = [
            row(f"{MODE3_PREFIX}{mode}:{target}", target, mode=mode)
            for mode in MODE3_MODES
        ]
        all_errors = [
            {
                **row(f"{COVERAGE_PREFIX}{target}:{mode}", target, mode=mode),
                "error_filtered_mining": {
                    "gold_rank": None if rank == 0 else rank + 1,
                    "top1_confuser_skill_id": f"skill/n{rank}",
                },
            }
            for rank, mode in enumerate(COVERAGE_MODES)
        ]
        for error_count in range(7):
            cycles = target_cycle_views(task5, mode3, all_errors[:error_count])
            self.assertEqual([len(cycle) for cycle in cycles], [5, 5])
            for cycle in cycles:
                indices = [str(item[0]["index"]) for item in cycle]
                self.assertEqual(len(indices), len(set(indices)))
            selected_errors = [
                str(item[0]["index"])
                for cycle in cycles
                for item in cycle
                if item[1] == ERROR_SOURCE_KIND
            ]
            self.assertCountEqual(
                selected_errors,
                [str(item["index"]) for item in all_errors[:error_count]],
            )

    def test_full_builder_preserves_mass_and_query_bound_candidates(self) -> None:
        catalog_ids = ["skill/a", "skill/b", *(f"skill/n{rank}" for rank in range(11))]
        targets = ["skill/a", "skill/b"]

        def negatives(positive: str) -> list[str]:
            return [skill for skill in catalog_ids if skill != positive][:10]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path = root / "catalog.jsonl"
            write_jsonl(catalog_path, [{"skill_id": skill} for skill in catalog_ids])
            targets_path = root / "targets.jsonl"
            write_jsonl(targets_path, [{"skill_id": skill} for skill in targets])

            single_rows = [row(0, "skill/a")]
            for skill in catalog_ids:
                single_rows.extend(
                    row(f"{TASK5_PREFIX}{skill}:{rank}", skill) for rank in range(5)
                )
            for skill in targets:
                single_rows.extend(
                    row(f"{MODE3_PREFIX}{mode}:{skill}", skill, mode=mode)
                    for mode in MODE3_MODES
                )
            single_rows_path = root / "single_rows.jsonl"
            write_jsonl(single_rows_path, single_rows)
            legacy_candidates = [
                {
                    "row_index": index,
                    "positive_skill_id": item["positive_skill_id"],
                    "negative_skill_ids": negatives(str(item["positive_skill_id"])),
                    "query_sha256": query_digest(item),
                }
                for index, item in enumerate(single_rows)
            ]
            legacy_candidates_path = root / "legacy_candidates.jsonl"
            write_jsonl(legacy_candidates_path, legacy_candidates)

            base_rows = []
            for skill in targets:
                source_rows = [
                    *(row(f"{TASK5_PREFIX}{skill}:{rank}", skill) for rank in range(5)),
                    *(
                        row(f"{MODE3_PREFIX}{mode}:{skill}", skill, mode=mode)
                        for mode in MODE3_MODES
                    ),
                ]
                for source in source_rows:
                    base_rows.append(
                        {
                            **source,
                            "index": len(base_rows),
                            "source_index": source["index"],
                        }
                    )
            base_rows_path = root / "base_rows.jsonl"
            write_jsonl(base_rows_path, base_rows)
            base_candidates = [
                {
                    "row_index": index,
                    "positive_skill_id": item["positive_skill_id"],
                    "negative_skill_ids": negatives(str(item["positive_skill_id"])),
                    "query_sha256": query_digest(item),
                }
                for index, item in enumerate(base_rows)
            ]
            base_candidates_path = root / "base_candidates.jsonl"
            write_jsonl(base_candidates_path, base_candidates)

            error_rows = []
            for rank, mode in enumerate(COVERAGE_MODES):
                source_index = f"{COVERAGE_PREFIX}skill/a:{mode}"
                positive = "skill/a"
                top1 = negatives(positive)[rank]
                error_rows.append(
                    {
                        **row(source_index, positive, mode=mode),
                        "source_index": source_index,
                        "error_filtered_mining": {
                            "gold_rank": None if rank == 0 else rank + 2,
                            "top1_confuser_skill_id": top1,
                        },
                    }
                )
            error_rows_path = root / "error_rows.jsonl"
            write_jsonl(error_rows_path, error_rows)
            error_candidates = []
            for index, item in enumerate(error_rows):
                candidate_negatives = negatives("skill/a")
                top1 = str(item["error_filtered_mining"]["top1_confuser_skill_id"])
                candidate_negatives = [top1, *(skill for skill in candidate_negatives if skill != top1)]
                error_candidates.append(
                    {
                        "row_index": index,
                        "positive_skill_id": "skill/a",
                        "negative_skill_ids": candidate_negatives,
                        "query_sha256": query_digest(item),
                    }
                )
            error_candidates_path = root / "error_candidates.jsonl"
            write_jsonl(error_candidates_path, error_candidates)
            checkpoint_sha = "b" * 64
            error_manifest_path = root / "error_manifest.json"
            error_manifest_path.write_text(
                json.dumps(
                    {
                        "format": "diverse_coverage_error_filtered_v1",
                        "target_skill_count": 2,
                        "views_per_target": 6,
                        "error_rows": 6,
                        "mining_checkpoint_sha256": checkpoint_sha,
                        "negatives_per_error_row": 10,
                        "error_count_per_target_histogram": {"0": 1, "6": 1},
                        "outputs": {
                            "error_rows_sha256": file_sha256(error_rows_path),
                            "error_candidates_sha256": file_sha256(error_candidates_path),
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_fixed_mass_error_filtered_training(
                single_rows_path=single_rows_path,
                target_skills_path=targets_path,
                catalog_path=catalog_path,
                legacy_candidates_path=legacy_candidates_path,
                base_mining_rows_path=base_rows_path,
                base_candidates_path=base_candidates_path,
                error_rows_path=error_rows_path,
                error_candidates_path=error_candidates_path,
                error_manifest_path=error_manifest_path,
                output_dir=root / "output",
                expected_catalog_skills=13,
                expected_targets=2,
                expected_negatives=10,
                expected_error_rows=6,
                expected_mining_checkpoint_sha256=checkpoint_sha,
                cycles=2,
                pseudo_views_per_skill_per_cycle=5,
            )
            self.assertEqual(manifest["output_rows"], 132)
            self.assertEqual(manifest["cycle_row_counts"], {0: 66, 1: 66})
            self.assertEqual(manifest["diverse_error_exposures"], 6)
            self.assertEqual(
                manifest["candidate_provenance_counts"],
                {
                    "cum082_base_query_bound": 14,
                    "cum082_diverse_error_query_bound": 6,
                    "step2400_query_bound": 112,
                },
            )

            output_rows = [
                json.loads(line)
                for line in Path(manifest["rows_output"]).read_text(encoding="utf-8").splitlines()
            ]
            output_candidates = [
                json.loads(line)
                for line in Path(manifest["candidates_output"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            pseudo_counts: Counter[str] = Counter()
            cycle_sources: dict[tuple[int, str], list[str]] = defaultdict(list)
            for item in output_rows:
                fixed = item["fixed_mass"]
                if fixed["source_kind"] == "original":
                    continue
                positive = item["positive_skill_id"]
                pseudo_counts[positive] += 1
                cycle_sources[(fixed["cycle"], positive)].append(fixed["original_index"])
            self.assertTrue(all(pseudo_counts[skill] == 10 for skill in catalog_ids))
            self.assertTrue(
                all(len(indices) == len(set(indices)) == 5 for indices in cycle_sources.values())
            )
            self.assertEqual([item["row_index"] for item in output_candidates], list(range(132)))
            self.assertTrue(
                all(item["query_sha256"] == query_digest(output_rows[index]) for index, item in enumerate(output_candidates))
            )


if __name__ == "__main__":
    unittest.main()
