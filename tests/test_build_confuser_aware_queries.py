from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_confuser_aware_queries import (
    aggregate_confusers,
    build_contrastive_messages,
    materialize_generation,
    prepare_mining_rows,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class ConfuserAwareQueriesTest(unittest.TestCase):
    def test_prepare_mining_keeps_canonical_five_and_three_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = root / "targets.jsonl"
            augmented = root / "augmented.jsonl"
            write_jsonl(targets, [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}])
            rows = []
            for sid in ("skill/a", "skill/b"):
                for rank in range(5):
                    rows.append(
                        {
                            "index": f"pseudo:catalog_task:{sid}:{rank}",
                            "query": f"task {sid} {rank}",
                            "selected_skill_ids": [sid],
                            "positive_skill_id": sid,
                            "pseudo_query": {"mode": "task_spec", "rank": rank},
                        }
                    )
                for mode in ("outcome_first", "workflow_embedded", "constraint_failure"):
                    rows.append(
                        {
                            "index": f"pseudo:zero_multi_mode:{mode}:{sid}",
                            "query": f"mode {sid} {mode}",
                            "selected_skill_ids": [sid],
                            "positive_skill_id": sid,
                            "pseudo_query": {"mode": mode},
                        }
                    )
            rows.append(
                {
                    "index": "pseudo:catalog_task:alias/a:0",
                    "query": "mapped alias row",
                    "selected_skill_ids": ["skill/a"],
                    "positive_skill_id": "skill/a",
                    "pseudo_query": {"mode": "task_spec", "rank": 0},
                }
            )
            write_jsonl(augmented, rows)

            manifest = prepare_mining_rows(
                augmented_rows_path=augmented,
                target_skills_path=targets,
                output_dir=root / "prepared",
                expected_targets=2,
                task5_per_skill=5,
                mode3_per_skill=3,
            )
            selected = [
                json.loads(line)
                for line in (root / "prepared/mining_rows.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(selected), 16)
            self.assertEqual(manifest["discarded_alias_task5_rows"], 1)
            self.assertTrue(manifest["all_skills_exact"])
            self.assertNotIn("pseudo:catalog_task:alias/a:0", {row["source_index"] for row in selected})

    def test_aggregate_uses_reciprocal_rank_over_all_target_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = root / "targets.jsonl"
            catalog = root / "skills.jsonl"
            mining = root / "mining.jsonl"
            candidates = root / "candidates.jsonl"
            candidate_manifest = root / "candidate_manifest.json"
            write_jsonl(targets, [{"skill_id": "skill/a"}])
            write_jsonl(
                catalog,
                [
                    {"skill_id": "skill/a", "name": "A"},
                    {"skill_id": "skill/n1", "name": "N1"},
                    {"skill_id": "skill/n2", "name": "N2"},
                    {"skill_id": "skill/n3", "name": "N3"},
                ],
            )
            mining_rows = [
                {
                    "index": index,
                    "source_index": f"source-{index}",
                    "query": f"query {index}",
                    "selected_skill_ids": ["skill/a"],
                    "positive_skill_id": "skill/a",
                }
                for index in range(8)
            ]
            write_jsonl(mining, mining_rows)
            candidate_rows = []
            for index in range(8):
                negatives = (
                    ["skill/n2", "skill/n1", "skill/n3"]
                    if index < 5
                    else ["skill/n1", "skill/n3", "skill/n2"]
                )
                candidate_rows.append(
                    {
                        "row_index": index,
                        "source_index": f"source-{index}",
                        "positive_skill_id": "skill/a",
                        "negative_skill_ids": negatives,
                    }
                )
            write_jsonl(candidates, candidate_rows)
            candidate_manifest.write_text(
                json.dumps(
                    {
                        "output": str(candidates),
                        "mining_checkpoint_sha256": "checkpoint-sha",
                    }
                ),
                encoding="utf-8",
            )

            manifest = aggregate_confusers(
                mining_rows_path=mining,
                candidates_path=candidates,
                candidate_manifest_path=candidate_manifest,
                catalog_path=catalog,
                target_skills_path=targets,
                output_dir=root / "aggregated",
                confusers_per_skill=2,
                expected_views_per_skill=8,
            )
            pairs = [
                json.loads(line)
                for line in (root / "aggregated/confuser_pairs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["confuser_skill_id"] for row in pairs], ["skill/n2", "skill/n1"])
            self.assertEqual(manifest["pair_count"], 2)
            self.assertEqual(manifest["mining_checkpoint_sha256"], "checkpoint-sha")

    def test_prompt_and_materialization_bind_the_confuser(self) -> None:
        target = {
            "skill_id": "skill/target",
            "name": "Target Capability",
            "description": "Transforms a structured artifact.",
            "body": "Read the artifact, apply the supported transformation, and verify the result.",
        }
        confuser = {
            "skill_id": "skill/confuser",
            "name": "Nearby Capability",
            "description": "Inspects a different artifact.",
            "body": "Inspect the artifact and report metadata.",
        }
        messages = build_contrastive_messages(
            target,
            confuser,
            target_body_chars=2200,
            confuser_body_chars=1200,
        )
        self.assertIn("Positive private document", messages[0]["content"])
        self.assertIn("Confuser private document", messages[0]["content"])
        self.assertIn("cannot be completed correctly", messages[0]["content"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = {
                "target_skill_id": "skill/target",
                "confuser_skill_id": "skill/confuser",
                "confuser_rank": 1,
            }
            write_jsonl(
                root / "contrastive_generations.jsonl",
                [
                    {
                        "work_key": "skill/target\tskill/confuser",
                        "status": "complete",
                        "query": "A concrete task.\n\nYou need to:\n1. Transform the supplied artifact.\n2. Verify the transformed output.\n\nDeliverable: the verified artifact.",
                        "response_model": "test-model",
                        "parse_mode": "json",
                        "usage": {},
                    }
                ],
            )
            manifest = materialize_generation(
                output_dir=root,
                pairs=[pair],
                profile={"model": "test-model", "api_key": "secret", "base_url": "https://example.invalid"},
                run_config={"pair_count": 1},
                max_attempts=3,
            )
            row = json.loads(
                (root / "selected_confuser_contrastive.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertTrue(manifest["complete"])
            self.assertEqual(row["positive_skill_id"], "skill/target")
            self.assertEqual(row["pseudo_query"]["confuser_skill_id"], "skill/confuser")
            self.assertEqual(manifest["profile"]["api_key"], "<redacted>")


if __name__ == "__main__":
    unittest.main()
