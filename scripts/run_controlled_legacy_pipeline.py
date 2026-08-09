#!/usr/bin/env python3
"""Run a legacy pipeline under explicit CUDA and optimizer controls."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any


OPTIONAL_BOOL = ("auto", "true", "false")


def optional_bool(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--bf16-reduction", choices=OPTIONAL_BOOL, default="auto")
    parser.add_argument("--foreach", choices=OPTIONAL_BOOL, default="auto")
    parser.add_argument(
        "--preferred-blas",
        choices=("auto", "cublas", "cublaslt"),
        default="auto",
    )
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    return parser


def precision_snapshot(torch: Any) -> dict[str, Any]:
    preferred_blas = getattr(torch.backends.cuda, "preferred_blas_library", None)
    return {
        "allow_bf16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul,
            "allow_bf16_reduced_precision_reduction",
            None,
        ),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "preferred_blas_library": str(preferred_blas()) if preferred_blas else None,
    }


def main() -> None:
    args = build_parser().parse_args()

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "a controlled arm requires exactly one visible CUDA GPU; "
            f"observed={torch.cuda.device_count()}"
        )
    if not args.pipeline.is_file():
        raise FileNotFoundError(args.pipeline)

    bf16_reduction = optional_bool(args.bf16_reduction)
    foreach = optional_bool(args.foreach)
    if bf16_reduction is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bf16_reduction
    if args.preferred_blas != "auto":
        torch.backends.cuda.preferred_blas_library(args.preferred_blas)
    if foreach is not None:
        original_adamw = torch.optim.AdamW

        def controlled_adamw(*optimizer_args: Any, **optimizer_kwargs: Any) -> Any:
            optimizer_kwargs["foreach"] = foreach
            return original_adamw(*optimizer_args, **optimizer_kwargs)

        torch.optim.AdamW = controlled_adamw

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "controls": {
            "bf16_reduction": bf16_reduction,
            "foreach": foreach,
            "preferred_blas": args.preferred_blas,
        },
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "precision": precision_snapshot(torch),
        "runtime": {
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sys.argv = [str(args.pipeline), *args.pipeline_args]
    runpy.run_path(str(args.pipeline), run_name="__main__")


if __name__ == "__main__":
    main()
