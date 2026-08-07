from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from legacy_baseline_parity_probe import (  # noqa: E402
    EXPECTED_ARCHIVED_MANIFEST,
    EXPECTED_PROBE_MANIFEST,
    command_finalize,
    file_sha256,
    legacy_shuffle_trace,
)


class LegacyBaselineParityProbeTest(unittest.TestCase):
    def test_persistent_rng_replays_archived_in_place_shuffle_sequence(self) -> None:
        trace = legacy_shuffle_trace(
            list("abcdefghij"),
            seed=17,
            epochs=2,
            batch_size=4,
            recorded_batches=2,
        )

        self.assertEqual(
            trace["initial_catalog_shuffle"]["first_batch"],
            ["h", "d", "a", "f"],
        )
        self.assertEqual(trace["epochs"][0]["first_batch"], ["d", "j", "h", "i"])
        self.assertEqual(trace["epochs"][1]["first_batch"], ["g", "d", "c", "j"])
        self.assertEqual(trace["final_order"], ["g", "d", "c", "j", "a", "e", "b", "f", "i", "h"])

    def test_trace_keeps_incomplete_tail_without_padding(self) -> None:
        trace = legacy_shuffle_trace(
            [str(index) for index in range(10)],
            seed=3,
            epochs=1,
            batch_size=4,
            recorded_batches=3,
        )

        self.assertEqual(trace["epochs"][0]["recorded_batch_count"], 3)
        self.assertEqual(trace["tail_batch_size"], 2)
        self.assertEqual(len(trace["epochs"][0]["last_batch"]), 2)
        self.assertEqual(len(trace["final_order"]), 10)

    def test_finalize_accepts_archived_losses_across_the_epoch_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            training = root / "training"
            training.mkdir()
            audit = {
                "historical_source_commit": "15f6c51",
                "configuration": {
                    "seed": 20260626,
                    "batch_size": 16,
                    "bf16": True,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                    "catalog_epochs": 3,
                    "max_source_length": 2048,
                    "sampler": "archived_single_process_persistent_rng_in_place_shuffle_no_tail_padding",
                },
            }
            (root / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
            events = [
                {
                    "stage": "catalog",
                    "event": "train",
                    "step": step,
                    "train_loss": {
                        50: 4.2037888,
                        900: 4.0914092,
                        950: 4.0723839,
                    }.get(step, 4.3),
                }
                for step in range(1, 951)
            ]
            (training / "loss_history.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            (training / "single_sid_decoder_train_manifest.json").write_text(
                json.dumps(EXPECTED_PROBE_MANIFEST),
                encoding="utf-8",
            )
            (root / "archived.log").write_text(
                "catalog_step=50 train_loss=4.203789\n"
                "catalog_step=900 train_loss=4.091409\n"
                "catalog_step=950 train_loss=4.072384\n",
                encoding="utf-8",
            )
            (root / "archived_manifest.json").write_text(
                json.dumps(EXPECTED_ARCHIVED_MANIFEST),
                encoding="utf-8",
            )

            command_finalize(
                argparse.Namespace(
                    audit=root / "audit.json",
                    training_output=training,
                    archived_train_log=root / "archived.log",
                    archived_train_log_sha256=file_sha256(root / "archived.log"),
                    archived_manifest=root / "archived_manifest.json",
                    archived_manifest_sha256=file_sha256(root / "archived_manifest.json"),
                    output=root / "result.json",
                )
            )

            result = json.loads((root / "result.json").read_text(encoding="utf-8"))
            self.assertTrue(result["aligned"])
            self.assertEqual(result["comparison"]["step_1_loss"], 4.3)
            self.assertEqual(result["comparison"]["boundary_losses"]["step_910_tail_batch"], 4.3)


if __name__ == "__main__":
    unittest.main()
