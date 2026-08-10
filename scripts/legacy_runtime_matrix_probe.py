#!/usr/bin/env python3
"""Run and validate one archived baseline runtime-matrix arm."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from legacy_baseline_parity_probe import _hash_tensor_state, _torch_rng_sha256
except ModuleNotFoundError:
    from scripts.legacy_baseline_parity_probe import _hash_tensor_state, _torch_rng_sha256


EXPECTED_STEPS = 50
OPTIONAL_BOOL_CHOICES = ("auto", "true", "false")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def optional_bool(value: str) -> bool | None:
    if value == "auto":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"unsupported optional boolean: {value}")


def apply_precision_controls(torch, args: argparse.Namespace) -> dict[str, Any]:
    bf16_reduction = optional_bool(args.bf16_reduction)
    tf32_matmul = optional_bool(args.tf32_matmul)
    tf32_cudnn = optional_bool(args.tf32_cudnn)
    if bf16_reduction is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bf16_reduction
    if tf32_matmul is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
    if tf32_cudnn is not None:
        torch.backends.cudnn.allow_tf32 = tf32_cudnn
    if args.float32_matmul_precision != "auto":
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
    if args.preferred_blas != "auto":
        torch.backends.cuda.preferred_blas_library(args.preferred_blas)
    deterministic_algorithms = optional_bool(args.deterministic_algorithms)
    if deterministic_algorithms is not None:
        torch.use_deterministic_algorithms(deterministic_algorithms)
    return precision_snapshot(torch)


def precision_snapshot(torch) -> dict[str, Any]:
    matmul = torch.backends.cuda.matmul
    preferred_blas = getattr(torch.backends.cuda, "preferred_blas_library", None)
    return {
        "allow_bf16_reduced_precision_reduction": getattr(
            matmul, "allow_bf16_reduced_precision_reduction", None
        ),
        "allow_fp16_reduced_precision_reduction": getattr(
            matmul, "allow_fp16_reduced_precision_reduction", None
        ),
        "allow_fp16_accumulation": getattr(matmul, "allow_fp16_accumulation", None),
        "allow_tf32_matmul": matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "preferred_blas_library": str(preferred_blas()) if preferred_blas else None,
    }


def training_wrapper_source(args: argparse.Namespace) -> str:
    controls = {
        "bf16_reduction": optional_bool(args.bf16_reduction),
        "tf32_matmul": optional_bool(args.tf32_matmul),
        "tf32_cudnn": optional_bool(args.tf32_cudnn),
        "float32_matmul_precision": args.float32_matmul_precision,
        "foreach": optional_bool(args.foreach),
        "preferred_blas": args.preferred_blas,
        "deterministic_algorithms": optional_bool(args.deterministic_algorithms),
        "dropout_rate": getattr(args, "dropout_rate", None),
    }
    return """
import runpy
import json
import os
import sys
import torch

controls = __CONTROLS__
dropout_modules = []
if controls["dropout_rate"] is not None:
    requested_dropout_rate = float(controls["dropout_rate"])
    if not 0.0 <= requested_dropout_rate <= 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1], got {requested_dropout_rate}")
    original_dropout = torch.nn.Dropout
    class ControlledDropout(original_dropout):
        def __init__(self, p=0.5, inplace=False):
            super().__init__(p=requested_dropout_rate, inplace=inplace)
            dropout_modules.append(self)
    torch.nn.Dropout = ControlledDropout
if controls["bf16_reduction"] is not None:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = controls["bf16_reduction"]
if controls["tf32_matmul"] is not None:
    torch.backends.cuda.matmul.allow_tf32 = controls["tf32_matmul"]
if controls["tf32_cudnn"] is not None:
    torch.backends.cudnn.allow_tf32 = controls["tf32_cudnn"]
if controls["float32_matmul_precision"] != "auto":
    torch.set_float32_matmul_precision(controls["float32_matmul_precision"])
if controls["preferred_blas"] != "auto":
    torch.backends.cuda.preferred_blas_library(controls["preferred_blas"])
if controls["deterministic_algorithms"] is not None:
    torch.use_deterministic_algorithms(controls["deterministic_algorithms"])
if controls["foreach"] is not None:
    original_adamw = torch.optim.AdamW
    def controlled_adamw(*optimizer_args, **optimizer_kwargs):
        optimizer_kwargs["foreach"] = controls["foreach"]
        return original_adamw(*optimizer_args, **optimizer_kwargs)
    torch.optim.AdamW = controlled_adamw

script_path = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script_path, run_name="__main__")
audit_path = os.environ["SKILLGR_PRECISION_CONTROL_AUDIT"]
audit_payload = {
    "precision": {
        "allow_bf16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", None
        ),
        "allow_fp16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", None
        ),
        "allow_fp16_accumulation": getattr(
            torch.backends.cuda.matmul, "allow_fp16_accumulation", None
        ),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
    },
    "foreach_override": controls["foreach"],
    "dropout_rate_requested": controls["dropout_rate"],
    "dropout_module_count": len(dropout_modules),
    "dropout_module_rates": sorted({float(module.p) for module in dropout_modules}),
    "environment": {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING"),
    },
}
with open(audit_path, "w", encoding="utf-8") as handle:
    json.dump(audit_payload, handle, indent=2, sort_keys=True)
    handle.write("\\n")
""".replace("__CONTROLS__", repr(controls))


def runtime_snapshot(torch) -> dict[str, Any]:
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,uuid",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        smi = None
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_file_sha256": file_sha256(Path(torch.__file__).resolve()),
        "torch_config": torch.__config__.show(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": package_version("transformers"),
        "tokenizers": package_version("tokenizers"),
        "huggingface_hub": package_version("huggingface-hub"),
        "safetensors": package_version("safetensors"),
        "triton": package_version("triton"),
        "nvidia_cublas_cu12": package_version("nvidia-cublas-cu12"),
        "nvidia_cudnn_cu12": package_version("nvidia-cudnn-cu12"),
        "nvidia_cuda_runtime_cu12": package_version("nvidia-cuda-runtime-cu12"),
        "nvidia_cuda_nvrtc_cu12": package_version("nvidia-cuda-nvrtc-cu12"),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory": int(properties.total_memory),
        "nvidia_smi": smi,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_MODULE_LOADING",
                "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            )
        },
    }


def legacy_model_state_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.view(-1).cpu().numpy().tobytes())
    return digest.hexdigest()


def load_historical_pipeline(source_root: Path):
    scripts_dir = source_root / "scripts"
    pipeline_path = scripts_dir / "skill_gr_pipeline.py"
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"archived pipeline is missing: {pipeline_path}")
    sys.path.insert(0, str(scripts_dir))
    import skill_gr_pipeline  # type: ignore

    if Path(skill_gr_pipeline.__file__).resolve() != pipeline_path.resolve():
        raise RuntimeError(f"loaded the wrong archived pipeline: {skill_gr_pipeline.__file__}")
    return skill_gr_pipeline


def selected_weight_file(model_dir: Path) -> Path:
    candidates = [model_dir / "model.safetensors", model_dir / "pytorch_model.bin"]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError(
            f"runtime probe model view must expose exactly one weight file, found: {present}"
        )
    return present[0]


def catalog_loss_summary(loss_history: Path, expected_steps: int = EXPECTED_STEPS) -> dict[str, float]:
    events = [
        json.loads(line)
        for line in loss_history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog = [
        row for row in events if row.get("stage") == "catalog" and row.get("event") == "train"
    ]
    steps = [int(row["step"]) for row in catalog]
    if steps != list(range(1, expected_steps + 1)):
        raise ValueError(f"catalog steps are incomplete or reordered: {steps}")
    return {
        "step_1": float(catalog[0]["train_loss"]),
        f"step_{expected_steps}": float(catalog[-1]["train_loss"]),
    }


def command_run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"each runtime arm requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    configured_precision = apply_precision_controls(torch, args)
    historical = load_historical_pipeline(args.legacy_source)
    semantic_ids, semantic_id_mode = historical.load_semantic_ids_with_mode(args.semantic_ids)
    sid_config = historical.infer_structured_sid_config(
        semantic_ids,
        semantic_id_mode=semantic_id_mode,
    )
    catalog_rows = historical.build_single_sid_catalog_rows(
        historical.load_jsonl(args.skills_jsonl),
        semantic_ids,
        "skill document: ",
    )
    order_rng = random.Random(args.seed)
    order_rng.shuffle(catalog_rows)
    order_rng.shuffle(catalog_rows)
    first_batch_ids = [str(row["index"]) for row in catalog_rows[:16]]
    first_batch_sha256 = hashlib.sha256(
        "".join(f"{row_id}\n" for row_id in first_batch_ids).encode("utf-8")
    ).hexdigest()
    del catalog_rows
    weight_file = selected_weight_file(args.model)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch_rng_before = _torch_rng_sha256(torch.get_rng_state())
    cuda_rng_before = _torch_rng_sha256(torch.cuda.get_rng_state())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = historical.make_structured_t5_sid_model(
        str(args.model),
        int(sid_config["codebook_size"]),
        int(sid_config["depth"]),
        int(sid_config["dedup_size"]),
        semantic_id_mode,
    ).to(torch.device("cuda"))
    state_hashes = _hash_tensor_state(model.state_dict())
    legacy_state_fingerprint = legacy_model_state_fingerprint(model)
    torch_rng_after = _torch_rng_sha256(torch.get_rng_state())
    cuda_rng_after = _torch_rng_sha256(torch.cuda.get_rng_state())
    try:
        from torch.optim.optimizer import _default_to_fused_or_foreach

        inferred_fused, inferred_foreach = _default_to_fused_or_foreach(
            list(model.parameters()), differentiable=False, use_fused=False
        )
    except (ImportError, TypeError):
        inferred_fused, inferred_foreach = None, None
    del tokenizer, model
    gc.collect()
    torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory(prefix=f"skillgr-runtime-{args.arm}.", dir="/tmp") as directory:
        train_output = Path(directory) / "train"
        pipeline_path = args.legacy_source / "scripts" / "skill_gr_pipeline.py"
        pipeline_args = [
            str(pipeline_path),
            "train-single-sid-decoder",
            "--rows",
            str(args.rows),
            "--dev-rows",
            str(args.dev_rows),
            "--semantic-ids",
            str(args.semantic_ids),
            "--skills-jsonl",
            str(args.skills_jsonl),
            "--output-dir",
            str(train_output),
            "--model",
            str(args.model),
            "--catalog-epochs",
            "1",
            "--catalog-max-steps",
            str(args.steps),
            "--epochs",
            "0",
            "--max-source-length",
            "2048",
            "--batch-size",
            "16",
            "--eval-batch-size",
            "128",
            "--lr",
            "0.0003",
            "--weight-decay",
            "0",
            "--logging-steps",
            "1",
            "--eval-steps",
            "1000",
            "--seed",
            str(args.seed),
        ]
        if args.precision == "bf16":
            pipeline_args.append("--bf16")
        command = [sys.executable, "-c", training_wrapper_source(args), *pipeline_args]
        child_audit_path = Path(directory) / "child_precision.json"
        child_environment = os.environ.copy()
        child_environment["SKILLGR_PRECISION_CONTROL_AUDIT"] = str(child_audit_path)
        if args.cublas_workspace_config != "auto":
            child_environment["CUBLAS_WORKSPACE_CONFIG"] = args.cublas_workspace_config
        if args.cuda_module_loading != "auto":
            child_environment["CUDA_MODULE_LOADING"] = args.cuda_module_loading
        subprocess.run(command, check=True, env=child_environment)
        child_precision = json.loads(child_audit_path.read_text(encoding="utf-8"))
        if child_precision["precision"] != configured_precision:
            raise ValueError(
                "parent/child precision control mismatch: "
                f"parent={configured_precision}, child={child_precision['precision']}"
            )
        losses = catalog_loss_summary(train_output / "loss_history.jsonl", args.steps)
        manifest = json.loads(
            (train_output / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8")
        )
    expected_manifest = {
        "batch_size": 16,
        "catalog_rows": 14551,
        "catalog_steps": args.steps,
        "query_steps": 0,
        "seed": args.seed,
        "lr": 3e-4,
        "weight_decay": 0.0,
        "semantic_id_mode": "learned_codes",
        "semantic_ids_sha256": "da2fff997b1979f7d51996c5ddbe4593a18d23bdfd546c61dd61fb783567349c",
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"training manifest mismatch: {mismatches}")

    payload: dict[str, Any] = {
        "artifact_format": "legacy_runtime_matrix_arm_v2",
        "arm": args.arm,
        "historical_source_commit": args.legacy_commit,
        "runtime": runtime_snapshot(torch),
        "condition": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": 16,
            "precision": args.precision,
            "lr": 3e-4,
            "weight_decay": 0.0,
            "weight_format": weight_file.name,
            "weight_sha256": file_sha256(weight_file),
            "requested_controls": {
                "bf16_reduction": args.bf16_reduction,
                "tf32_matmul": args.tf32_matmul,
                "tf32_cudnn": args.tf32_cudnn,
                "float32_matmul_precision": args.float32_matmul_precision,
                "foreach": args.foreach,
                "preferred_blas": args.preferred_blas,
                "deterministic_algorithms": args.deterministic_algorithms,
                "cublas_workspace_config": args.cublas_workspace_config,
                "cuda_module_loading": args.cuda_module_loading,
            },
            "effective_precision": configured_precision,
            "child_effective_precision": child_precision,
            "inferred_fused": inferred_fused,
            "inferred_foreach": inferred_foreach,
            "effective_foreach": (
                optional_bool(args.foreach)
                if args.foreach != "auto"
                else inferred_foreach
            ),
        },
        "initialization": {
            "torch_rng_before_model_sha256": torch_rng_before,
            "torch_rng_after_model_sha256": torch_rng_after,
            "cuda_rng_before_model_sha256": cuda_rng_before,
            "cuda_rng_after_model_sha256": cuda_rng_after,
            "state_sha256": state_hashes,
            "legacy_state_fingerprint": legacy_state_fingerprint,
        },
        "first_catalog_batch": {
            "row_ids": first_batch_ids,
            "row_ids_sha256": first_batch_sha256,
        },
        "losses": losses,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "runtime_probe.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "runtime_probe.json")
    print(json.dumps(payload, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--legacy-commit", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--bf16-reduction",
        choices=OPTIONAL_BOOL_CHOICES,
        default="auto",
    )
    parser.add_argument("--tf32-matmul", choices=OPTIONAL_BOOL_CHOICES, default="auto")
    parser.add_argument("--tf32-cudnn", choices=OPTIONAL_BOOL_CHOICES, default="auto")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("auto", "highest", "high", "medium"),
        default="auto",
    )
    parser.add_argument("--foreach", choices=OPTIONAL_BOOL_CHOICES, default="auto")
    parser.add_argument(
        "--preferred-blas",
        choices=("auto", "default", "cublas", "cublaslt"),
        default="auto",
    )
    parser.add_argument(
        "--deterministic-algorithms",
        choices=OPTIONAL_BOOL_CHOICES,
        default="auto",
    )
    parser.add_argument(
        "--cublas-workspace-config",
        choices=("auto", ":4096:8", ":16:8"),
        default="auto",
    )
    parser.add_argument(
        "--cuda-module-loading",
        choices=("auto", "LAZY", "EAGER"),
        default="auto",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=None,
        help="override instantiated torch.nn.Dropout modules; omit to preserve the archived setting",
    )
    parser.set_defaults(func=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1:
        raise ValueError(f"steps must be positive, got {args.steps}")
    args.func(args)


if __name__ == "__main__":
    main()
