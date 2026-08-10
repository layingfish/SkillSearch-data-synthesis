import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Local source-only environments may omit torch.
    torch = None

from scripts.analyze_legacy_local_fp32_matrix import reduction, state_distance_squared
from scripts.legacy_local_fp32_probe import (
    LocalFP32Controller,
    cast_floating_tensors,
    parse_probe_events,
)


@unittest.skipIf(torch is None, "torch is unavailable")
class LegacyLocalFP32TensorTest(unittest.TestCase):
    def test_cast_floating_tensors_preserves_integer_inputs(self):
        value = {
            "float": torch.tensor([1.0], dtype=torch.bfloat16),
            "integer": torch.tensor([1], dtype=torch.long),
            "nested": (torch.tensor([2.0], dtype=torch.float16),),
        }
        cast = cast_floating_tensors(value)
        self.assertEqual(cast["float"].dtype, torch.float32)
        self.assertEqual(cast["integer"].dtype, torch.int64)
        self.assertEqual(cast["nested"][0].dtype, torch.float32)
        self.assertEqual(value["float"].dtype, torch.bfloat16)

    def test_precision_gate_off_preserves_output(self):
        model = torch.nn.Linear(3, 2)
        inputs = torch.randn(4, 3)
        expected = model(inputs)
        controller = LocalFP32Controller("none", device_type="cpu")
        observed = controller.remember_model(model)(inputs)
        controller.validate(expected_steps=1)
        self.assertTrue(torch.equal(expected, observed))

    def test_local_fp32_module_disables_outer_cpu_autocast(self):
        module = torch.nn.Linear(3, 2)
        controller = LocalFP32Controller("encoder", device_type="cpu")
        controller.wrap_module("backbone.encoder", module)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = module(torch.randn(4, 3))
        self.assertEqual(output.dtype, torch.float32)
        controller.validate(expected_steps=1)

    def test_state_distance_squared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.pt"
            right = root / "right.pt"
            torch.save({"weight": torch.tensor([1.0, 2.0])}, left)
            torch.save({"weight": torch.tensor([4.0, 6.0])}, right)
            self.assertEqual(state_distance_squared(left, right), 25.0)


class LegacyLocalFP32MetadataTest(unittest.TestCase):
    def test_parse_probe_events_requires_step_1_and_50(self):
        stdout = "\n".join(
            [
                '{"event":"probe_header"}',
                '{"event":"first_batch"}',
                '{"event":"train","stage":"catalog","step":1,"train_loss":4.5}',
                '{"event":"train","stage":"catalog","step":50,"train_loss":4.0}',
            ]
        )
        parsed = parse_probe_events(stdout, 50)
        self.assertEqual(set(parsed["trains"]), {1, 50})

    def test_reduction(self):
        self.assertEqual(reduction(2.0, 1.0), 0.5)
        self.assertEqual(reduction(2.0, 0.5), 0.75)
        with self.assertRaisesRegex(ValueError, "control metric must be positive"):
            reduction(0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "candidate metric must be non-negative"):
            reduction(2.0, -1.0)


if __name__ == "__main__":
    unittest.main()
