from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ripor_anchored_artifacts import (  # noqa: E402
    command_anchor,
    command_seal_self_negatives,
    command_verify_self_negatives,
)


class RiporAnchoredArtifactsTest(unittest.TestCase):
    def test_gate_zero_is_bitwise_baseline_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = root / "rows.jsonl"
            baseline = root / "baseline.jsonl"
            ripor = root / "intentionally_absent_ripor.jsonl"
            payload = {"index": 1, "query": "q", "selected_skill_ids": ["a"]}
            rows.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            prediction = {
                "index": 1,
                "query": "q",
                "gold_skill_ids": ["a"],
                "predicted_skill_ids": ["a"],
                "ranked_skill_ids": ["a"],
                "ranked_skill_scores": {"a": -0.1},
                "invalid": False,
            }
            baseline.write_text(json.dumps(prediction, sort_keys=True) + "\n", encoding="utf-8")
            output = root / "gate0"
            command_anchor(
                argparse.Namespace(
                    rows=rows,
                    baseline_predictions=baseline,
                    ripor_predictions=ripor,
                    output_dir=output,
                    anchor_weight=0.0,
                    max_rank=50,
                    k_values="1,10,50",
                )
            )
            self.assertEqual(baseline.read_bytes(), (output / "predictions.jsonl").read_bytes())

    def test_self_negative_seal_excludes_all_siblings_and_unscored_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = root / "rows.jsonl"
            predictions = root / "predictions.jsonl"
            eval_manifest = root / "eval_manifest.json"
            teacher = root / "teacher.jsonl"
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "structured_decoder.pt").write_bytes(b"checkpoint")
            row = {
                "index": 7,
                "query": "q",
                "selected_skill_ids": ["a"],
                "source_selected_skill_ids": ["a", "b"],
            }
            rows.write_text(json.dumps(row) + "\n", encoding="utf-8")
            predictions.write_text(
                json.dumps(
                    {
                        "index": 7,
                        "query": "q",
                        "gold_skill_ids": ["a"],
                        "ranked_skill_ids": ["b", "unscored", "neg", "a"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            eval_manifest.write_text(
                json.dumps(
                    {
                        "decoder_head": "ripor",
                        "score_transform": "raw",
                        "docid_control_score": "excluded",
                        "model_dir": str(checkpoint),
                    }
                ),
                encoding="utf-8",
            )
            teacher.write_text(
                json.dumps(
                    {
                        "index": 7,
                        "query": "q",
                        "ranked_skill_ids": ["a", "b", "neg"],
                        "ranked_scores": [1.0, 0.9, 0.2],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            panel = root / "selfneg.jsonl"
            eligible = root / "eligible.jsonl"
            manifest = root / "manifest.json"
            command_seal_self_negatives(
                argparse.Namespace(
                    rows=rows,
                    miner_predictions=predictions,
                    miner_eval_manifest=eval_manifest,
                    miner_checkpoint=checkpoint,
                    teacher_panel=teacher,
                    output_panel=panel,
                    output_rows=eligible,
                    output_manifest=manifest,
                    max_negatives=16,
                )
            )
            sealed = json.loads(panel.read_text(encoding="utf-8"))
            self.assertEqual(sealed["ranked_skill_ids"], ["neg", "a"])
            audit = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(audit["eligible_query_count"], 1)
            self.assertEqual(audit["total_self_negatives"], 1)
            verify_args = argparse.Namespace(
                manifest=manifest,
                rows=rows,
                miner_predictions=predictions,
                miner_eval_manifest=eval_manifest,
                miner_checkpoint=checkpoint,
                teacher_panel=teacher,
                output_panel=panel,
                output_rows=eligible,
            )
            command_verify_self_negatives(verify_args)
            teacher.write_text(teacher.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "teacher_panel_sha256 mismatch"):
                command_verify_self_negatives(verify_args)


if __name__ == "__main__":
    unittest.main()
