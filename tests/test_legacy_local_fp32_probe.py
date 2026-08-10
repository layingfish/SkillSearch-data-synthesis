import copy
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Local source-only environments may omit torch.
    torch = None

from scripts.analyze_legacy_local_fp32_matrix import reduction, state_distance_squared
from scripts.analyze_legacy_local_fp32_refinement import control_compatibility
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

    def test_composed_decoder_heads_region_wraps_exact_boundaries(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.decoder = torch.nn.Module()
        model.backbone.decoder.block = torch.nn.ModuleList(
            [torch.nn.Linear(3, 3) for _ in range(12)]
        )
        model.sid_embedding = torch.nn.Embedding(8, 3)
        model.code_heads = torch.nn.ModuleList(
            [torch.nn.Linear(3, 2) for _ in range(4)]
        )
        model.dedup_head = None
        model.control_head = torch.nn.Linear(3, 2)

        controller = LocalFP32Controller(
            "decoder_blocks_0_5_heads_ce", device_type="cpu"
        )
        local_cross_entropy = controller.wrap_cross_entropy(
            torch.nn.functional.cross_entropy
        )
        controller.remember_model(model)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            hidden = torch.randn(4, 3)
            for block in model.backbone.decoder.block[:6]:
                hidden = block(hidden)
            model.sid_embedding(torch.ones((4, 1), dtype=torch.long))
            logits = [head(hidden) for head in model.code_heads]
            logits.append(model.control_head(hidden))
            labels = torch.zeros(4, dtype=torch.long)
            for output in logits:
                local_cross_entropy(output, labels)

        controller.validate(expected_steps=1)
        self.assertTrue(controller.includes_heads_ce)
        self.assertIn("backbone.decoder.block.0", controller.call_counts)
        self.assertIn("backbone.decoder.block.5", controller.call_counts)
        self.assertNotIn("backbone.decoder.block.6", controller.call_counts)
        self.assertEqual(controller.call_counts["cross_entropy"], 5)

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

    def test_cross_job_control_compatibility_ignores_probe_and_gpu_uuid(self):
        control = {
            "condition": {
                "arm": "default_none",
                "path": "default",
                "region": "none",
                "precision_gate": False,
                "steps": 50,
                "requested_controls": {"cublas_workspace_config": "auto"},
            },
            "file_sha256": {"probe": "old", "pipeline": "pipeline", "rows": "rows"},
            "probe_header": {"initial_model_fingerprint": "init"},
            "first_catalog_batch": {"row_ids_sha256": "batch"},
            "precision": {"preferred_blas_library": "cublas"},
            "dropout": {"requested_rate": 0.0},
            "runtime": {
                "torch": "2.10.0+cu128",
                "torch_file": "/tmp/old/torch/__init__.py",
                "torch_file_sha256": "torch",
                "nvidia_smi": "GPU-old-uuid",
            },
        }
        candidate = copy.deepcopy(control)
        candidate["condition"].update(
            {
                "arm": "default_decoder_blocks_0_5",
                "region": "decoder_blocks_0_5",
                "precision_gate": True,
            }
        )
        candidate["file_sha256"]["probe"] = "new"
        candidate["runtime"]["torch_file"] = "/tmp/new/torch/__init__.py"
        candidate["runtime"]["nvidia_smi"] = "GPU-new-uuid"

        audit = control_compatibility(candidate, control)
        self.assertTrue(audit["passed"])
        self.assertTrue(all(audit["checks"].values()))


if __name__ == "__main__":
    unittest.main()
