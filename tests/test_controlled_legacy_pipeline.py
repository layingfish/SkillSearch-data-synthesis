import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_controlled_legacy_pipeline as controlled


class _PreferredBlas:
    def __init__(self) -> None:
        self.value = "_BlasBackend.Cublas"

    def __call__(self, value=None):
        if value is not None:
            self.value = f"_BlasBackend.{value.capitalize()}"
        return self.value


class ControlledLegacyPipelineTest(unittest.TestCase):
    def test_controls_are_applied_before_pipeline_execution(self) -> None:
        preferred_blas = _PreferredBlas()
        original_calls = []

        def adamw(*args, **kwargs):
            original_calls.append((args, kwargs))
            return kwargs

        fake_torch = types.SimpleNamespace(
            __version__="2.10.0+cu128",
            version=types.SimpleNamespace(cuda="12.8"),
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda _: "NVIDIA A800 80GB PCIe",
            ),
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    matmul=types.SimpleNamespace(
                        allow_bf16_reduced_precision_reduction=True,
                        allow_tf32=False,
                    ),
                    preferred_blas_library=preferred_blas,
                ),
                cudnn=types.SimpleNamespace(allow_tf32=True, version=lambda: 91002),
            ),
            get_float32_matmul_precision=lambda: "highest",
            optim=types.SimpleNamespace(AdamW=adamw),
        )

        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.json"
            argv = [
                "run_controlled_legacy_pipeline.py",
                "--audit",
                str(audit),
                "--bf16-reduction",
                "false",
                "--foreach",
                "false",
                "--preferred-blas",
                "cublaslt",
                __file__,
                "train-single-sid-decoder",
                "--seed",
                "20260626",
            ]
            with (
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(controlled.runpy, "run_path") as run_path,
                mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3"}, clear=False),
            ):
                controlled.main()
                fake_torch.optim.AdamW("parameters", lr=3e-4)

            payload = json.loads(audit.read_text(encoding="utf-8"))

        self.assertFalse(fake_torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        self.assertEqual(preferred_blas.value, "_BlasBackend.Cublaslt")
        self.assertFalse(original_calls[0][1]["foreach"])
        self.assertEqual(payload["controls"]["foreach"], False)
        self.assertEqual(payload["environment"]["CUDA_VISIBLE_DEVICES"], "3")
        run_path.assert_called_once_with(__file__, run_name="__main__")


if __name__ == "__main__":
    unittest.main()
