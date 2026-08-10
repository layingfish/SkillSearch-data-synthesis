import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Local source-only environments may omit torch.
    torch = None

from scripts.analyze_legacy_layerwise_trace import (
    compare_trace_control_pair,
    tensor_relative_l2,
)
from scripts.legacy_layerwise_trace_probe import (
    TensorArtifactWriter,
    TraceRecorder,
    parse_probe_events,
    tensor_summary,
)


@unittest.skipIf(torch is None, "torch is unavailable")
class LegacyLayerwiseTraceTensorTest(unittest.TestCase):
    def test_tensor_summary_and_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
            summary = tensor_summary(tensor)
            self.assertEqual(summary["shape"], [2, 2])
            self.assertEqual(summary["finite_fraction"], 1.0)
            self.assertAlmostEqual(summary["l2_norm"], math.sqrt(30.0))
            self.assertEqual(len(summary["sha256"]), 64)

            bf16_summary = tensor_summary(tensor.to(torch.bfloat16))
            self.assertEqual(bf16_summary["dtype"], "torch.bfloat16")
            self.assertEqual(len(bf16_summary["sha256"]), 64)

            writer = TensorArtifactWriter(root)
            record = writer.capture("activation_output", "linear", tensor, save=True)
            saved = torch.load(root / Path(record["artifact"]).name, weights_only=True)
            self.assertTrue(torch.equal(saved, tensor))

    def test_recorder_captures_activation_gradient_and_update(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = TensorArtifactWriter(Path(directory))
            recorder = TraceRecorder(writer, enabled=True)
            model = torch.nn.Linear(2, 1, bias=False)
            recorder.model = model
            recorder.register_module("linear", model)
            optimizer = recorder.wrap_optimizer(
                torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
            )

            before = model.weight.detach().clone()
            loss = model(torch.tensor([[1.0, 2.0]])).sum()
            loss.backward()
            optimizer.step()

            self.assertEqual(recorder.module_calls, {"linear": 1})
            self.assertEqual(recorder.optimizer_steps, 1)
            self.assertEqual(len(recorder.gradient_records), 1)
            self.assertEqual(len(recorder.update_records), 1)
            self.assertFalse(torch.equal(before, model.weight.detach()))
            self.assertTrue(recorder.activation_records[0]["outputs"][0]["artifact"])

    def test_relative_l2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.pt"
            candidate = root / "candidate.pt"
            torch.save(torch.tensor([3.0, 4.0]), reference)
            torch.save(torch.tensor([0.0, 4.0]), candidate)
            result = tensor_relative_l2(reference, candidate)
            self.assertAlmostEqual(result["reference_norm"], 5.0)
            self.assertAlmostEqual(result["difference_norm"], 3.0)
            self.assertAlmostEqual(result["relative_l2"], 0.6)
            self.assertFalse(result["exact"])


class LegacyLayerwiseTraceMetadataTest(unittest.TestCase):
    def test_parse_probe_events(self):
        rows = [
            {"event": "probe_header", "source_sha256": "source"},
            {"event": "first_batch", "row_ids_sha256": "batch"},
            {"event": "train", "step": 1, "train_loss": 4.5},
        ]
        stdout = "noise\n" + "\n".join(json.dumps(row) for row in rows) + "\n"
        parsed = parse_probe_events(stdout)
        self.assertEqual(parsed["train"]["train_loss"], 4.5)

    def test_trace_control_pair_requires_exact_final_state(self):
        base = {
            "condition": {"arm": "control", "trace": False, "path": "default_bf16"},
            "file_sha256": {"pipeline": "same"},
            "probe_header": {
                "initial_model_fingerprint": "init",
                "cpu_rng_state_sha256_after_model_init": "cpu",
                "cuda_rng_state_sha256_before_first_forward": "cuda",
            },
            "first_catalog_batch": {"row_ids_sha256": "batch"},
            "losses": {"step_1": 4.5},
            "final_state_sha256": {"full": "final"},
        }
        traced = json.loads(json.dumps(base))
        traced["condition"].update({"arm": "trace", "trace": True})
        self.assertTrue(compare_trace_control_pair(base, traced)["passed"])
        traced["final_state_sha256"]["full"] = "different"
        self.assertFalse(compare_trace_control_pair(base, traced)["passed"])


if __name__ == "__main__":
    unittest.main()
