import json
import tempfile
import unittest
from pathlib import Path

from scripts.legacy_core75_checkpoint_oracle import (
    EvalCheckpointSnapshotQueue,
    oracle_eval_steps,
    oracle_selection_key,
)


class _FakeTokenizer:
    def save_pretrained(self, output_dir: str) -> None:
        root = Path(output_dir)
        (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (root / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


class _FakeState:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return _FakeState(self.value)


class _FakeRng:
    def __init__(self):
        self.value = 0

    def get_rng_state(self):
        return _FakeState(self.value)


class _FakeTorch:
    def __init__(self):
        self.random = _FakeRng()
        self.cuda = _FakeRng()

    @staticmethod
    def equal(left, right):
        return left.value == right.value


def _fake_saver(
    checkpoint_dir,
    model,
    tokenizer,
    base_model,
    codebook_size,
    depth,
    dedup_size,
    seed,
    training_formulation,
    extra=None,
    optimizer=None,
    semantic_id_mode="learned_codes",
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model,
        "model": str(base_model),
        "codebook_size": codebook_size,
        "depth": depth,
        "dedup_size": dedup_size,
        "seed": seed,
        "training_formulation": training_formulation,
        "semantic_id_mode": semantic_id_mode,
        **(extra or {}),
    }
    (checkpoint_dir / "structured_decoder.pt").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    tokenizer.save_pretrained(str(checkpoint_dir))


class Core75CheckpointOracleMetadataTest(unittest.TestCase):
    def test_eval_steps_cover_195_points(self):
        steps = oracle_eval_steps(38805, 200)
        self.assertEqual(len(steps), 195)
        self.assertEqual(steps[:2], [200, 400])
        self.assertEqual(steps[-3:], [38600, 38800, 38805])

    def test_selection_key_prioritizes_r1_then_deep_recall_then_earlier_step(self):
        def result(step, r1, r10, r50):
            return {
                "step": step,
                "overall": {
                    "@1": {"recall": r1},
                    "@10": {"recall": r10},
                    "@50": {"recall": r50},
                },
            }

        higher_r1 = result(400, 0.29, 0.60, 0.70)
        lower_r1 = result(200, 0.28, 0.80, 0.90)
        self.assertGreater(
            oracle_selection_key(higher_r1), oracle_selection_key(lower_r1)
        )
        early = result(200, 0.29, 0.70, 0.75)
        late = result(400, 0.29, 0.70, 0.75)
        self.assertGreater(oracle_selection_key(early), oracle_selection_key(late))


class Core75CheckpointSnapshotTest(unittest.TestCase):
    def _capture_template(self, queue):
        save_args = (
            Path("best"),
            {"weight": [1.0]},
            _FakeTokenizer(),
            "base-model",
            256,
            4,
            0,
            20260626,
            "single_sid_catalog_indexing",
            {"best_step": 200, "best_dev_loss": 1.0, "contract": "frozen"},
        )
        queue.capture_saver_template(
            _fake_saver,
            save_args,
            {"optimizer": object(), "semantic_id_mode": "learned_codes"},
        )

    def test_snapshot_is_atomic_model_only_and_rng_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = EvalCheckpointSnapshotQueue(
                root, expected_steps=[200, 205], max_queued=4
            )
            self._capture_template(queue)
            model = {"weight": [1.0, 2.0]}
            fake_torch = _FakeTorch()

            queue.observe_loss_event(
                {"stage": "query", "event": "eval", "step": 200, "dev_loss": 1.0}
            )
            self.assertTrue(
                queue.flush(model, fake_torch, reason="before_dev_best_save")
            )
            ready = json.loads(
                (root / "step_00000200.ready.json").read_text(encoding="utf-8")
            )
            checkpoint = json.loads(
                (root / "step_00000200" / "structured_decoder.pt").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(ready["step"], 200)
            self.assertEqual(checkpoint["oracle_step"], 200)
            self.assertEqual(checkpoint["contract"], "frozen")
            self.assertNotIn("optimizer_state_dict", checkpoint)

            queue.observe_loss_event(
                {"stage": "query", "event": "eval", "step": 205, "dev_loss": 0.9}
            )
            self.assertTrue(
                queue.flush(
                    model, fake_torch, reason="before_terminal_best_restore"
                )
            )
            artifact = queue.validate()
            self.assertEqual(artifact["steps"], [200, 205])
            self.assertTrue(artifact["rng_invariant_passed"])

    def test_snapshot_rejects_rng_mutation(self):
        fake_torch = _FakeTorch()

        def mutating_saver(*args, **kwargs):
            fake_torch.random.value += 1
            _fake_saver(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = EvalCheckpointSnapshotQueue(
                root, expected_steps=[200], max_queued=1
            )
            save_args = (
                Path("best"),
                {"weight": [1.0]},
                _FakeTokenizer(),
                "base-model",
                256,
                4,
                0,
                20260626,
                "single_sid_catalog_indexing",
                {"best_step": 200, "best_dev_loss": 1.0},
            )
            queue.capture_saver_template(
                _fake_saver,
                save_args,
                {"optimizer": None, "semantic_id_mode": "learned_codes"},
            )
            queue.original_saver = mutating_saver
            queue.observe_loss_event(
                {"stage": "query", "event": "eval", "step": 200, "dev_loss": 1.0}
            )
            with self.assertRaisesRegex(RuntimeError, "changed CPU RNG"):
                queue.flush(
                    {"weight": [1.0]},
                    fake_torch,
                    reason="before_dev_best_save",
                )

    def test_snapshot_fails_fast_when_an_evaluator_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = EvalCheckpointSnapshotQueue(
                root, expected_steps=[200], max_queued=4
            )
            self._capture_template(queue)
            queue.observe_loss_event(
                {"stage": "query", "event": "eval", "step": 200, "dev_loss": 1.0}
            )
            (root / "worker.failed.json").write_text(
                '{"worker_id": 1, "error": "synthetic"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "evaluator worker failed"):
                queue.flush(
                    {"weight": [1.0]},
                    _FakeTorch(),
                    reason="before_dev_best_save",
                )


if __name__ == "__main__":
    unittest.main()
