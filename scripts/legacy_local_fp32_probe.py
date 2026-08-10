#!/usr/bin/env python3
"""Run one archived 50-step arm with an optional local FP32 region."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

try:
    from legacy_baseline_parity_probe import _hash_tensor_state
    from legacy_layerwise_trace_probe import (
        EXPECTED_DROPOUT0_DEFAULT_BF16_STEP1_LOSS,
        EXPECTED_DROPOUT_MODULES,
        EXPECTED_FIRST_BATCH_SHA256,
        EXPECTED_INITIAL_MODEL_FINGERPRINT,
        EXPECTED_PIPELINE_SHA256,
        TeeCapture,
        load_module,
        stable_json_sha256,
    )
    from legacy_runtime_matrix_probe import (
        apply_precision_controls,
        file_sha256,
        precision_snapshot,
        runtime_snapshot,
        selected_weight_file,
    )
except ModuleNotFoundError:
    from scripts.legacy_baseline_parity_probe import _hash_tensor_state
    from scripts.legacy_layerwise_trace_probe import (
        EXPECTED_DROPOUT0_DEFAULT_BF16_STEP1_LOSS,
        EXPECTED_DROPOUT_MODULES,
        EXPECTED_FIRST_BATCH_SHA256,
        EXPECTED_INITIAL_MODEL_FINGERPRINT,
        EXPECTED_PIPELINE_SHA256,
        TeeCapture,
        load_module,
        stable_json_sha256,
    )
    from scripts.legacy_runtime_matrix_probe import (
        apply_precision_controls,
        file_sha256,
        precision_snapshot,
        runtime_snapshot,
        selected_weight_file,
    )


PHASE2_REGIONS = ("none", "encoder", "decoder", "heads_ce")
PHASE3_REGIONS = (
    "decoder_blocks_0_5",
    "decoder_blocks_6_11",
    "decoder_heads_ce",
    "decoder_blocks_0_5_heads_ce",
)
REGIONS = PHASE2_REGIONS + PHASE3_REGIONS
HEADS_CE_REGIONS = {
    "heads_ce",
    "decoder_heads_ce",
    "decoder_blocks_0_5_heads_ce",
}
DECODER_BLOCK_REGIONS = {
    "decoder_blocks_0_5": range(0, 6),
    "decoder_blocks_6_11": range(6, 12),
    "decoder_blocks_0_5_heads_ce": range(0, 6),
}
OPTIONAL_BOOL_CHOICES = ("auto", "true", "false")


def iter_tensors(value: Any, prefix: str = "value") -> Iterator[tuple[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from iter_tensors(value[key], f"{prefix}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_tensors(item, f"{prefix}.{index}")


def tensor_tree_metadata(value: Any) -> list[dict[str, Any]]:
    return [
        {"path": path, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        for path, tensor in iter_tensors(value)
    ]


def cast_floating_tensors(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: cast_floating_tensors(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cast_floating_tensors(item) for item in value)
    if isinstance(value, list):
        return [cast_floating_tensors(item) for item in value]
    return value


class LocalFP32Controller:
    def __init__(self, region: str, *, device_type: str = "cuda") -> None:
        if region not in REGIONS:
            raise ValueError(f"unsupported local FP32 region: {region}")
        self.region = region
        self.device_type = device_type
        self.model = None
        self.initial_parameters: dict[str, Any] = {}
        self.call_counts: dict[str, int] = {}
        self.dtype_audit: dict[str, dict[str, Any]] = {}
        self.expected_calls_per_step: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.region != "none"

    @property
    def includes_heads_ce(self) -> bool:
        return self.region in HEADS_CE_REGIONS

    def wrap_module(self, name: str, module: Any) -> None:
        import torch

        if name in self.expected_calls_per_step:
            raise ValueError(f"duplicate local FP32 module wrapper: {name}")
        self.expected_calls_per_step[name] = 1
        original_forward = module.forward

        def local_fp32_forward(*args: Any, **kwargs: Any) -> Any:
            cast_args = cast_floating_tensors(args)
            cast_kwargs = cast_floating_tensors(kwargs)
            with torch.autocast(device_type=self.device_type, enabled=False):
                output = original_forward(*cast_args, **cast_kwargs)
            call_index = self.call_counts.get(name, 0)
            self.call_counts[name] = call_index + 1
            if call_index == 0:
                self.dtype_audit[name] = {
                    "inputs_before": tensor_tree_metadata({"args": args, "kwargs": kwargs}),
                    "inputs_after": tensor_tree_metadata(
                        {"args": cast_args, "kwargs": cast_kwargs}
                    ),
                    "outputs": tensor_tree_metadata(output),
                }
            return output

        module.forward = local_fp32_forward

    def wrap_cross_entropy(self, original_cross_entropy: Any) -> Any:
        import torch

        name = "cross_entropy"
        if name in self.expected_calls_per_step:
            raise ValueError(f"duplicate local FP32 function wrapper: {name}")
        self.expected_calls_per_step[name] = 5

        def local_fp32_cross_entropy(
            logits: Any,
            labels: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            cast_logits = cast_floating_tensors(logits)
            with torch.autocast(device_type=self.device_type, enabled=False):
                output = original_cross_entropy(cast_logits, labels, *args, **kwargs)
            call_index = self.call_counts.get(name, 0)
            self.call_counts[name] = call_index + 1
            if call_index == 0:
                self.dtype_audit[name] = {
                    "inputs_before": tensor_tree_metadata(
                        {"logits": logits, "labels": labels}
                    ),
                    "inputs_after": tensor_tree_metadata(
                        {"logits": cast_logits, "labels": labels}
                    ),
                    "outputs": tensor_tree_metadata(output),
                }
            return output

        return local_fp32_cross_entropy

    def wrap_heads_ce_modules(self, model: Any) -> None:
        self.wrap_module("sid_embedding", model.sid_embedding)
        for index, head in enumerate(model.code_heads):
            self.wrap_module(f"code_head.{index}", head)
        if model.dedup_head is not None:
            self.wrap_module("dedup_head", model.dedup_head)
        self.wrap_module("control_head", model.control_head)

    def remember_model(self, model: Any) -> Any:
        self.model = model
        self.initial_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if self.region == "encoder":
            self.wrap_module("backbone.encoder", model.backbone.encoder)
        elif self.region in {"decoder", "decoder_heads_ce"}:
            self.wrap_module("backbone.decoder", model.backbone.decoder)
        elif self.region in DECODER_BLOCK_REGIONS:
            blocks = model.backbone.decoder.block
            if len(blocks) != 12:
                raise ValueError(f"expected 12 decoder blocks, found {len(blocks)}")
            for index in DECODER_BLOCK_REGIONS[self.region]:
                self.wrap_module(f"backbone.decoder.block.{index}", blocks[index])
        if self.includes_heads_ce:
            self.wrap_heads_ce_modules(model)
        return model

    def validate(self, expected_steps: int) -> None:
        if not self.enabled:
            if self.call_counts or self.dtype_audit:
                raise ValueError("precision gate is off but local FP32 calls were recorded")
            return
        expected = {
            name: expected_steps * calls_per_step
            for name, calls_per_step in self.expected_calls_per_step.items()
        }
        if self.call_counts != expected:
            raise ValueError(
                f"local FP32 call counts mismatch: expected={expected}, actual={self.call_counts}"
            )
        for name, audit in self.dtype_audit.items():
            output_float_dtypes = {
                row["dtype"]
                for row in audit["outputs"]
                if row["dtype"].startswith("torch.float")
                or row["dtype"] == "torch.bfloat16"
            }
            if output_float_dtypes != {"torch.float32"}:
                raise ValueError(
                    f"local FP32 output dtype mismatch for {name}: {output_float_dtypes}"
                )


def parse_probe_events(stdout: str, expected_steps: int) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and "event" in event:
            events.append(event)
    headers = [event for event in events if event.get("event") == "probe_header"]
    batches = [event for event in events if event.get("event") == "first_batch"]
    trains = {
        int(event["step"]): event
        for event in events
        if event.get("event") == "train" and event.get("stage") == "catalog"
    }
    expected_train_steps = {1, expected_steps}
    if len(headers) != 1 or len(batches) != 1 or set(trains) != expected_train_steps:
        raise ValueError(
            "unexpected archived probe events: "
            f"headers={len(headers)}, batches={len(batches)}, "
            f"train_steps={sorted(trains)}"
        )
    return {"header": headers[0], "first_batch": batches[0], "trains": trains}


def command_run(args: argparse.Namespace) -> None:
    if args.steps != 50:
        raise ValueError(f"local FP32 matrix requires steps=50, got {args.steps}")
    if args.cublas_workspace_config != "auto":
        observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if observed_workspace != args.cublas_workspace_config:
            raise ValueError(
                "CUBLAS_WORKSPACE_CONFIG must be set before Python starts: "
                f"expected={args.cublas_workspace_config}, observed={observed_workspace}"
            )

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"each local FP32 arm requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite local FP32 output: {args.output_dir}")
    if not args.probe.is_file():
        raise FileNotFoundError(args.probe)
    pipeline_path = args.probe.parent / "skill_gr_pipeline.original_20260714.py"
    if file_sha256(pipeline_path) != EXPECTED_PIPELINE_SHA256:
        raise ValueError(f"archived pipeline SHA-256 mismatch: {pipeline_path}")

    configured_precision = apply_precision_controls(torch, args)
    controller = LocalFP32Controller(args.region)
    dropout_modules: list[Any] = []
    original_dropout = torch.nn.Dropout

    class ControlledDropout(original_dropout):
        def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
            super().__init__(p=args.dropout_rate, inplace=inplace)
            dropout_modules.append(self)

    torch.nn.Dropout = ControlledDropout
    original_cross_entropy = functional.cross_entropy
    if controller.includes_heads_ce:
        functional.cross_entropy = controller.wrap_cross_entropy(original_cross_entropy)

    original_adamw = torch.optim.AdamW

    def observed_adamw(*optimizer_args: Any, **optimizer_kwargs: Any) -> Any:
        optimizer = original_adamw(*optimizer_args, **optimizer_kwargs)
        torch.cuda.reset_peak_memory_stats()
        return optimizer

    torch.optim.AdamW = observed_adamw

    probe = load_module(args.probe, f"catalog_local_fp32_{args.arm}")
    original_load_pipeline = probe.load_pipeline

    def controlled_load_pipeline(path: Path) -> Any:
        pipeline = original_load_pipeline(path)
        original_factory = pipeline.make_structured_t5_sid_model

        def controlled_factory(*factory_args: Any, **factory_kwargs: Any) -> Any:
            return controller.remember_model(original_factory(*factory_args, **factory_kwargs))

        pipeline.make_structured_t5_sid_model = controlled_factory
        return pipeline

    probe.load_pipeline = controlled_load_pipeline
    probe_argv = [
        str(args.probe),
        "--repo-root",
        str(args.repo_root),
        "--model",
        str(args.model),
        "--batch-size",
        str(args.batch_size),
        "--max-source-length",
        str(args.max_source_length),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--foreach",
        args.foreach,
    ]
    original_argv = sys.argv
    tee = TeeCapture(sys.stdout)
    try:
        sys.argv = probe_argv
        with contextlib.redirect_stdout(tee):
            probe.main()
    finally:
        sys.argv = original_argv
        torch.nn.Dropout = original_dropout
        functional.cross_entropy = original_cross_entropy
        torch.optim.AdamW = original_adamw

    if controller.model is None:
        raise RuntimeError("archived probe did not construct a model")
    controller.validate(args.steps)
    events = parse_probe_events(tee.captured(), args.steps)
    header = events["header"]
    first_batch = events["first_batch"]
    if header.get("source_sha256") != EXPECTED_PIPELINE_SHA256:
        raise ValueError(f"probe source mismatch: {header.get('source_sha256')}")
    if header.get("initial_model_fingerprint") != EXPECTED_INITIAL_MODEL_FINGERPRINT:
        raise ValueError(
            f"initial model fingerprint mismatch: {header.get('initial_model_fingerprint')}"
        )
    if first_batch.get("row_ids_sha256") != EXPECTED_FIRST_BATCH_SHA256:
        raise ValueError(f"first catalog batch mismatch: {first_batch.get('row_ids_sha256')}")
    step_1_loss = float(events["trains"][1]["train_loss"])
    if args.region == "none" and args.path == "default" and (
        step_1_loss != EXPECTED_DROPOUT0_DEFAULT_BF16_STEP1_LOSS
    ):
        raise ValueError(
            "default BF16 step-1 loss mismatch: "
            f"expected={EXPECTED_DROPOUT0_DEFAULT_BF16_STEP1_LOSS}, observed={step_1_loss}"
        )
    dropout_rates = sorted({float(module.p) for module in dropout_modules})
    if len(dropout_modules) != EXPECTED_DROPOUT_MODULES or dropout_rates != [args.dropout_rate]:
        raise ValueError(
            f"dropout override mismatch: count={len(dropout_modules)}, rates={dropout_rates}"
        )

    final_parameters = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in controller.model.named_parameters()
        if parameter.requires_grad
    }
    if set(final_parameters) != set(controller.initial_parameters):
        raise ValueError("trainable parameter names changed during probe")
    distance_from_initial_squared = 0.0
    for name, final in final_parameters.items():
        initial = controller.initial_parameters[name]
        distance_from_initial_squared += float(
            ((final.to(torch.float64) - initial.to(torch.float64)) ** 2).sum().item()
        )

    weight_file = selected_weight_file(args.model)
    condition = {
        "arm": args.arm,
        "path": args.path,
        "region": args.region,
        "precision_gate": controller.enabled,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_source_length": args.max_source_length,
        "seed": args.seed,
        "outer_precision": "bf16",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "dropout_rate": args.dropout_rate,
        "foreach": args.foreach,
        "requested_controls": {
            "bf16_reduction": args.bf16_reduction,
            "tf32_matmul": args.tf32_matmul,
            "tf32_cudnn": args.tf32_cudnn,
            "float32_matmul_precision": args.float32_matmul_precision,
            "preferred_blas": args.preferred_blas,
            "deterministic_algorithms": args.deterministic_algorithms,
            "cublas_workspace_config": args.cublas_workspace_config,
            "cuda_module_loading": args.cuda_module_loading,
        },
    }
    payload = {
        "artifact_format": "legacy_local_fp32_arm_v1",
        "condition": condition,
        "condition_sha256": stable_json_sha256(condition),
        "runtime": runtime_snapshot(torch),
        "precision": precision_snapshot(torch),
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "file_sha256": {
            "probe": file_sha256(args.probe),
            "pipeline": file_sha256(pipeline_path),
            "rows": file_sha256(args.rows),
            "dev_rows": file_sha256(args.dev_rows),
            "skills": file_sha256(args.skills_jsonl),
            "semantic_ids": file_sha256(args.semantic_ids),
            "model_weight": file_sha256(weight_file),
        },
        "probe_header": header,
        "first_catalog_batch": first_batch,
        "losses": {
            "step_1": step_1_loss,
            f"step_{args.steps}": float(events["trains"][args.steps]["train_loss"]),
        },
        "dropout": {
            "requested_rate": args.dropout_rate,
            "module_count": len(dropout_modules),
            "module_rates": dropout_rates,
        },
        "local_fp32": {
            "enabled": controller.enabled,
            "region": controller.region,
            "call_counts": controller.call_counts,
            "dtype_audit": controller.dtype_audit,
        },
        "trajectory": {
            "distance_from_initial_squared": distance_from_initial_squared,
            "final_state_sha256": _hash_tensor_state(controller.model.state_dict()),
            "trainable_state_artifact": "trainable_state.pt",
        },
        "memory": {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }

    with tempfile.TemporaryDirectory(prefix=f"skillgr-local-fp32-{args.arm}.", dir="/tmp") as directory:
        temporary_state = Path(directory) / "trainable_state.pt"
        torch.save(final_parameters, temporary_state)
        args.output_dir.mkdir(parents=True)
        shutil.copy2(temporary_state, args.output_dir / "trainable_state.pt")
    (args.output_dir / "probe_stdout.log").write_text(tee.captured(), encoding="utf-8")
    temporary_json = args.output_dir / "local_fp32_probe.json.tmp"
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, args.output_dir / "local_fp32_probe.json")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "region": args.region,
                "step_1": payload["losses"]["step_1"],
                "step_50": payload["losses"]["step_50"],
                "final_state_sha256": payload["trajectory"]["final_state_sha256"]["full"],
                "max_allocated_bytes": payload["memory"]["max_allocated_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--path", choices=("default", "workspace_16"), required=True)
    parser.add_argument("--region", choices=REGIONS, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--foreach", choices=OPTIONAL_BOOL_CHOICES, default="auto")
    parser.add_argument("--bf16-reduction", choices=OPTIONAL_BOOL_CHOICES, default="auto")
    parser.add_argument("--tf32-matmul", choices=OPTIONAL_BOOL_CHOICES, default="auto")
    parser.add_argument("--tf32-cudnn", choices=OPTIONAL_BOOL_CHOICES, default="auto")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("auto", "highest", "high", "medium"),
        default="auto",
    )
    parser.add_argument(
        "--preferred-blas",
        choices=("auto", "default", "cublas", "cublaslt"),
        default="auto",
    )
    parser.add_argument("--deterministic-algorithms", choices=OPTIONAL_BOOL_CHOICES, default="auto")
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
    parser.set_defaults(
        precision="bf16",
        func=command_run,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.dropout_rate <= 1.0:
        raise ValueError(f"dropout-rate must be in [0, 1], got {args.dropout_rate}")
    args.func(args)


if __name__ == "__main__":
    main()
