#!/usr/bin/env python3
"""Probe catalog-to-query AdamW carry/reset behavior in the archived trainer."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:
    from legacy_baseline_parity_probe import _hash_tensor_state
    from legacy_layerwise_trace_probe import (
        EXPECTED_DROPOUT_MODULES,
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
        EXPECTED_DROPOUT_MODULES,
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


CATALOG_STEPS = 2730
QUERY_STEPS = 50
EXPECTED_CATALOG_FIRST_BATCH_SHA256 = (
    "721a0d93aaef2f9d0104adae8501fcfabcefcd6bd9f23c1e4a1aae701710d443"
)
EXPECTED_QUERY_FIRST_BATCH_SHA256 = (
    "56995788f5f03f084be7ac3ea36848206aa09943ba41dbc6dbf68b0000fefb85"
)
EXPECTED_CPU_RNG_AFTER_MODEL_SHA256 = (
    "c7674001d52ec9a441a446aa5bcea21926392ee15eee3d5ed578decde6118178"
)
EXPECTED_CUDA_RNG_BEFORE_FORWARD_SHA256 = (
    "87e7d790d0d53018bfaa5c17183bf79642be40bb0bd607387e1e875a19234ac8"
)
EXPECTED_DEFAULT_CATALOG_LOSSES = {
    "catalog_step_1": 4.546923637390137,
    "catalog_step_50": 4.184666633605957,
}
OPTIMIZER_MODES = ("carry", "reset")
OPTIONAL_BOOL_CHOICES = ("auto", "true", "false")


def tensor_bytes_sha256(tensor: Any) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(-1).cpu().numpy().tobytes()
    ).hexdigest()


def legacy_model_state_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(-1).cpu().numpy().tobytes())
    return digest.hexdigest()


def _update_recursive_hash(digest: Any, value: Any) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_recursive_hash(digest, key)
            _update_recursive_hash(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        for item in value:
            _update_recursive_hash(digest, item)
        return
    digest.update(type(value).__name__.encode())
    digest.update(b"\0")
    digest.update(repr(value).encode())
    digest.update(b"\0")


def recursive_state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_recursive_hash(digest, value)
    return digest.hexdigest()


def parameter_group(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("sid_embedding."):
        return "sid_embedding"
    if name.startswith("code_heads."):
        return "code_heads"
    if name.startswith("control_head."):
        return "control_head"
    return "other"


def tensor_collection_summary(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    import torch

    digest = hashlib.sha256()
    squared_by_group: dict[str, float] = {"full": 0.0}
    count_by_group: dict[str, int] = {"full": 0}
    for name, value in items:
        tensor = value.detach().cpu().contiguous()
        group = parameter_group(name)
        squared = float((tensor.to(torch.float64) ** 2).sum().item())
        squared_by_group["full"] += squared
        squared_by_group[group] = squared_by_group.get(group, 0.0) + squared
        count_by_group["full"] += 1
        count_by_group[group] = count_by_group.get(group, 0) + 1
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": count_by_group,
        "squared_l2": squared_by_group,
        "l2": {key: math.sqrt(value) for key, value in squared_by_group.items()},
    }


def trainable_state_cpu(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def state_distance_squared(left: dict[str, Any], right: dict[str, Any]) -> float:
    import torch

    if set(left) != set(right):
        raise ValueError("trainable parameter names differ")
    squared = 0.0
    for name in sorted(left):
        squared += float(
            ((left[name].to(torch.float64) - right[name].to(torch.float64)) ** 2)
            .sum()
            .item()
        )
    return squared


def optimizer_param_group_snapshot(optimizer: Any) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]


def reset_optimizer_in_place(optimizer: Any, query_lr: float) -> None:
    optimizer.state.clear()
    for group in optimizer.param_groups:
        group["lr"] = float(query_lr)


def set_optimizer_lr(optimizer: Any, query_lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(query_lr)


def row_identifier(row: dict[str, Any]) -> str:
    index = str(row["index"])
    positive = row.get("positive_skill_id")
    return f"{index}|{positive}" if positive is not None else index


def row_ids_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_ids = [row_identifier(row) for row in rows]
    raw = "".join(f"{row_id}\n" for row_id in row_ids).encode()
    return {"row_ids": row_ids, "row_ids_sha256": hashlib.sha256(raw).hexdigest()}


class OptimizerBoundaryObserver:
    def __init__(
        self,
        *,
        optimizer_mode: str,
        query_lr: float,
        catalog_steps: int = CATALOG_STEPS,
        query_steps: int = QUERY_STEPS,
    ) -> None:
        if optimizer_mode not in OPTIMIZER_MODES:
            raise ValueError(f"unsupported optimizer mode: {optimizer_mode}")
        if query_lr <= 0.0:
            raise ValueError(f"query lr must be positive, got {query_lr}")
        self.optimizer_mode = optimizer_mode
        self.query_lr = float(query_lr)
        self.catalog_steps = catalog_steps
        self.query_steps = query_steps
        self.model = None
        self.optimizer = None
        self.batch_calls = 0
        self.forward_calls = 0
        self.optimizer_steps = 0
        self.initial: dict[str, Any] = {}
        self.batches: dict[str, Any] = {}
        self.losses: dict[str, float] = {}
        self.catalog: dict[str, Any] = {}
        self.transition: dict[str, Any] = {}
        self.query_step_1: dict[str, Any] = {}
        self.catalog_parameters: dict[str, Any] = {}
        self.checkpoint_saves: list[dict[str, Any]] = []

    def remember_model(self, model: Any) -> Any:
        self.model = model
        original_forward = model.forward

        def observed_forward(*args: Any, **kwargs: Any) -> Any:
            output = original_forward(*args, **kwargs)
            self.forward_calls += 1
            points = {
                1: "catalog_step_1",
                50: "catalog_step_50",
                self.catalog_steps: f"catalog_step_{self.catalog_steps}",
                self.catalog_steps + 1: "query_step_1",
                self.catalog_steps + self.query_steps: f"query_step_{self.query_steps}",
            }
            name = points.get(self.forward_calls)
            if name is not None:
                if getattr(output, "ndim", None) != 0:
                    raise ValueError(f"expected scalar loss at {name}")
                self.losses[name] = float(output.detach().cpu())
            return output

        model.forward = observed_forward
        return model

    def wrap_batch_builder(self, original_builder: Any) -> Any:
        def observed_builder(batch_rows: list[dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
            next_call = self.batch_calls + 1
            if next_call == self.catalog_steps + 1:
                self.capture_catalog_boundary()
            batch = original_builder(batch_rows, *args, **kwargs)
            if next_call in {1, self.catalog_steps + 1}:
                name = "catalog_step_1" if next_call == 1 else "query_step_1"
                self.batches[name] = {
                    **row_ids_payload(batch_rows),
                    "tensor_sha256": recursive_state_sha256(batch),
                }
            self.batch_calls = next_call
            return batch

        return observed_builder

    def wrap_optimizer(self, optimizer: Any) -> Any:
        import torch

        if self.model is None:
            raise RuntimeError("optimizer constructed before model capture")
        self.optimizer = optimizer
        self.initial = {
            "model_fingerprint": legacy_model_state_fingerprint(self.model),
            "model_state_sha256": _hash_tensor_state(self.model.state_dict()),
            "cpu_rng_sha256": tensor_bytes_sha256(torch.random.get_rng_state()),
            "cuda_rng_sha256": tensor_bytes_sha256(torch.cuda.get_rng_state()),
            "optimizer_param_groups": optimizer_param_group_snapshot(optimizer),
            "optimizer_state_entries": len(optimizer.state),
        }
        torch.cuda.reset_peak_memory_stats()
        original_step = optimizer.step

        def observed_step(*args: Any, **kwargs: Any) -> Any:
            if self.optimizer_steps == self.catalog_steps:
                if not self.catalog_parameters:
                    raise RuntimeError("query optimizer step started before catalog capture")
                gradients = [
                    (name, parameter.grad)
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.query_step_1["gradient"] = tensor_collection_summary(gradients)
                if self.optimizer_mode == "reset":
                    reset_optimizer_in_place(optimizer, self.query_lr)
                else:
                    set_optimizer_lr(optimizer, self.query_lr)
                self.transition = {
                    "mode": self.optimizer_mode,
                    "query_lr": self.query_lr,
                    "optimizer_state_entries_after_transition": len(optimizer.state),
                    "optimizer_state_sha256_after_transition": recursive_state_sha256(
                        optimizer.state_dict()
                    ),
                    "optimizer_param_groups_after_transition": optimizer_param_group_snapshot(
                        optimizer
                    ),
                }
            result = original_step(*args, **kwargs)
            self.optimizer_steps += 1
            if self.optimizer_steps == self.catalog_steps + 1:
                updates = []
                for name, parameter in self.model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    update = parameter.detach().cpu() - self.catalog_parameters[name]
                    updates.append((name, update))
                self.query_step_1["update"] = tensor_collection_summary(updates)
            return result

        optimizer.step = observed_step
        return optimizer

    def capture_catalog_boundary(self) -> None:
        import torch

        if self.model is None or self.optimizer is None:
            raise RuntimeError("catalog boundary reached before model/optimizer capture")
        if self.catalog_parameters:
            raise RuntimeError("catalog boundary captured more than once")
        if self.optimizer_steps != self.catalog_steps:
            raise RuntimeError(
                f"catalog boundary optimizer-step mismatch: {self.optimizer_steps}"
            )
        self.catalog_parameters = trainable_state_cpu(self.model)
        self.catalog = {
            "model_fingerprint": legacy_model_state_fingerprint(self.model),
            "model_state_sha256": _hash_tensor_state(self.model.state_dict()),
            "trainable_state_sha256": recursive_state_sha256(self.catalog_parameters),
            "optimizer_state_sha256": recursive_state_sha256(self.optimizer.state_dict()),
            "optimizer_state_entries": len(self.optimizer.state),
            "optimizer_param_groups": optimizer_param_group_snapshot(self.optimizer),
            "cpu_rng_sha256": tensor_bytes_sha256(torch.random.get_rng_state()),
            "cuda_rng_sha256": tensor_bytes_sha256(torch.cuda.get_rng_state()),
        }

    def suppress_checkpoint_save(self, *args: Any, **kwargs: Any) -> None:
        checkpoint_dir = Path(args[0])
        self.checkpoint_saves.append(
            {
                "name": checkpoint_dir.name,
                "path": str(checkpoint_dir),
                "optimizer_present": kwargs.get("optimizer") is not None,
            }
        )

    def validate(self) -> None:
        expected_total = self.catalog_steps + self.query_steps
        for name, count in {
            "batch_calls": self.batch_calls,
            "forward_calls": self.forward_calls,
            "optimizer_steps": self.optimizer_steps,
        }.items():
            if count != expected_total:
                raise ValueError(f"{name} mismatch: expected={expected_total}, actual={count}")
        expected_losses = {
            "catalog_step_1",
            "catalog_step_50",
            f"catalog_step_{self.catalog_steps}",
            "query_step_1",
            f"query_step_{self.query_steps}",
        }
        if set(self.losses) != expected_losses:
            raise ValueError(f"loss capture mismatch: {sorted(self.losses)}")
        if self.initial.get("model_fingerprint") != EXPECTED_INITIAL_MODEL_FINGERPRINT:
            raise ValueError(f"initial model fingerprint mismatch: {self.initial}")
        if self.initial.get("cpu_rng_sha256") != EXPECTED_CPU_RNG_AFTER_MODEL_SHA256:
            raise ValueError(f"initial CPU RNG mismatch: {self.initial}")
        if self.initial.get("cuda_rng_sha256") != EXPECTED_CUDA_RNG_BEFORE_FORWARD_SHA256:
            raise ValueError(f"initial CUDA RNG mismatch: {self.initial}")
        expected_batches = {
            "catalog_step_1": EXPECTED_CATALOG_FIRST_BATCH_SHA256,
            "query_step_1": EXPECTED_QUERY_FIRST_BATCH_SHA256,
        }
        for name, expected_hash in expected_batches.items():
            observed = self.batches.get(name, {}).get("row_ids_sha256")
            if observed != expected_hash:
                raise ValueError(
                    f"{name} row hash mismatch: expected={expected_hash}, observed={observed}"
                )
        if not self.catalog_parameters or not self.catalog or not self.transition:
            raise ValueError("catalog/optimizer transition capture is incomplete")
        if set(self.query_step_1) != {"gradient", "update"}:
            raise ValueError(f"query step-1 trace incomplete: {self.query_step_1.keys()}")
        transition_entries = self.transition["optimizer_state_entries_after_transition"]
        if self.optimizer_mode == "reset" and transition_entries != 0:
            raise ValueError(f"reset optimizer retained {transition_entries} state entries")
        if self.optimizer_mode == "carry" and transition_entries != self.catalog[
            "optimizer_state_entries"
        ]:
            raise ValueError("carry optimizer state-entry count changed at transition")
        final_lrs = {
            float(group["lr"])
            for group in self.transition["optimizer_param_groups_after_transition"]
        }
        if final_lrs != {self.query_lr}:
            raise ValueError(f"query lr transition mismatch: {final_lrs}")
        if [row["name"] for row in self.checkpoint_saves] != ["last", "final"]:
            raise ValueError(f"unexpected checkpoint save calls: {self.checkpoint_saves}")


def command_run(args: argparse.Namespace) -> None:
    if args.catalog_steps != CATALOG_STEPS or args.query_steps != QUERY_STEPS:
        raise ValueError(
            f"optimizer boundary probe requires catalog/query={CATALOG_STEPS}/{QUERY_STEPS}"
        )
    if args.cublas_workspace_config != "auto":
        observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if observed_workspace != args.cublas_workspace_config:
            raise ValueError(
                "CUBLAS_WORKSPACE_CONFIG must be set before Python starts: "
                f"expected={args.cublas_workspace_config}, observed={observed_workspace}"
            )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"each optimizer boundary arm requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite optimizer probe output: {args.output_dir}")
    if file_sha256(args.pipeline) != EXPECTED_PIPELINE_SHA256:
        raise ValueError(f"archived pipeline SHA-256 mismatch: {args.pipeline}")

    configured_precision = apply_precision_controls(torch, args)
    observer = OptimizerBoundaryObserver(
        optimizer_mode=args.optimizer_mode,
        query_lr=args.query_lr,
        catalog_steps=args.catalog_steps,
        query_steps=args.query_steps,
    )
    dropout_modules: list[Any] = []
    original_dropout = torch.nn.Dropout

    class ControlledDropout(original_dropout):
        def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
            super().__init__(p=args.dropout_rate, inplace=inplace)
            dropout_modules.append(self)

    torch.nn.Dropout = ControlledDropout
    original_adamw = torch.optim.AdamW

    def observed_adamw(*optimizer_args: Any, **optimizer_kwargs: Any) -> Any:
        return observer.wrap_optimizer(original_adamw(*optimizer_args, **optimizer_kwargs))

    torch.optim.AdamW = observed_adamw

    pipeline = load_module(args.pipeline, f"optimizer_boundary_{args.arm}")
    original_factory = pipeline.make_structured_t5_sid_model

    def observed_factory(*factory_args: Any, **factory_kwargs: Any) -> Any:
        return observer.remember_model(original_factory(*factory_args, **factory_kwargs))

    pipeline.make_structured_t5_sid_model = observed_factory
    pipeline.structured_sid_batch = observer.wrap_batch_builder(pipeline.structured_sid_batch)
    pipeline.save_structured_decoder_checkpoint = observer.suppress_checkpoint_save

    original_argv = sys.argv
    tee = TeeCapture(sys.stdout)
    pipeline_manifest: dict[str, Any]
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"skillgr-optimizer-boundary-{args.arm}.", dir="/tmp"
        ) as directory:
            pipeline_output = Path(directory) / "pipeline-output"
            sys.argv = [
                str(args.pipeline),
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
                str(pipeline_output),
                "--model",
                str(args.model),
                "--catalog-max-steps",
                str(args.catalog_steps),
                "--catalog-epochs",
                "3",
                "--max-steps",
                str(args.query_steps),
                "--epochs",
                "5",
                "--batch-size",
                str(args.batch_size),
                "--lr",
                "0.0003",
                "--weight-decay",
                "0",
                "--max-source-length",
                str(args.max_source_length),
                "--logging-steps",
                "50",
                "--eval-steps",
                "0",
                "--seed",
                str(args.seed),
                "--bf16",
                "--no-load-best-model-at-end",
            ]
            with contextlib.redirect_stdout(tee):
                pipeline.main()
            manifest_path = pipeline_output / "single_sid_decoder_train_manifest.json"
            pipeline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    finally:
        sys.argv = original_argv
        torch.nn.Dropout = original_dropout
        torch.optim.AdamW = original_adamw

    observer.validate()
    dropout_rates = sorted({float(module.p) for module in dropout_modules})
    if len(dropout_modules) != EXPECTED_DROPOUT_MODULES or dropout_rates != [args.dropout_rate]:
        raise ValueError(
            f"dropout override mismatch: count={len(dropout_modules)}, rates={dropout_rates}"
        )
    if args.path == "default":
        for name, expected in EXPECTED_DEFAULT_CATALOG_LOSSES.items():
            if observer.losses[name] != expected:
                raise ValueError(
                    f"default {name} loss mismatch: expected={expected}, observed={observer.losses[name]}"
                )
    expected_manifest = {
        "source_train_rows": 94912,
        "query_train_rows": 124175,
        "query_dev_rows": 48750,
        "catalog_rows": 14551,
        "catalog_steps": args.catalog_steps,
        "query_steps": args.query_steps,
        "batch_size": args.batch_size,
        "max_source_length": args.max_source_length,
        "seed": args.seed,
    }
    observed_manifest = {key: pipeline_manifest.get(key) for key in expected_manifest}
    if observed_manifest != expected_manifest:
        raise ValueError(
            f"pipeline manifest mismatch: expected={expected_manifest}, observed={observed_manifest}"
        )

    if observer.model is None or observer.optimizer is None:
        raise RuntimeError("pipeline did not expose final model/optimizer")
    query_parameters = trainable_state_cpu(observer.model)
    if set(query_parameters) != set(observer.catalog_parameters):
        raise ValueError("trainable parameter names changed across query probe")
    query_distance_from_catalog_squared = state_distance_squared(
        query_parameters, observer.catalog_parameters
    )
    weight_file = selected_weight_file(args.model)
    condition = {
        "arm": args.arm,
        "path": args.path,
        "optimizer_mode": args.optimizer_mode,
        "query_lr": args.query_lr,
        "catalog_steps": args.catalog_steps,
        "query_steps": args.query_steps,
        "batch_size": args.batch_size,
        "max_source_length": args.max_source_length,
        "seed": args.seed,
        "catalog_lr": 3e-4,
        "weight_decay": 0.0,
        "dropout_rate": args.dropout_rate,
        "outer_precision": "bf16",
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
        "artifact_format": "legacy_optimizer_boundary_arm_v1",
        "condition": condition,
        "condition_sha256": stable_json_sha256(condition),
        "runtime": runtime_snapshot(torch),
        "precision": precision_snapshot(torch),
        "configured_precision": configured_precision,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "file_sha256": {
            "probe": file_sha256(Path(__file__).resolve()),
            "pipeline": file_sha256(args.pipeline),
            "rows": file_sha256(args.rows),
            "dev_rows": file_sha256(args.dev_rows),
            "skills": file_sha256(args.skills_jsonl),
            "semantic_ids": file_sha256(args.semantic_ids),
            "model_weight": file_sha256(weight_file),
        },
        "initial": observer.initial,
        "batches": observer.batches,
        "losses": observer.losses,
        "dropout": {
            "requested_rate": args.dropout_rate,
            "module_count": len(dropout_modules),
            "module_rates": dropout_rates,
        },
        "catalog_boundary": {
            **observer.catalog,
            "trainable_state_artifact": "catalog_trainable_state.pt",
        },
        "optimizer_transition": observer.transition,
        "query_step_1": observer.query_step_1,
        "query_step_50": {
            "model_state_sha256": _hash_tensor_state(observer.model.state_dict()),
            "trainable_state_sha256": recursive_state_sha256(query_parameters),
            "optimizer_state_sha256": recursive_state_sha256(
                observer.optimizer.state_dict()
            ),
            "optimizer_state_entries": len(observer.optimizer.state),
            "trainable_state_artifact": "query_step_50_trainable_state.pt",
            "distance_from_catalog_squared": query_distance_from_catalog_squared,
        },
        "execution": {
            "batch_calls": observer.batch_calls,
            "forward_calls": observer.forward_calls,
            "optimizer_steps": observer.optimizer_steps,
            "suppressed_checkpoint_saves": observer.checkpoint_saves,
            "pipeline_manifest": pipeline_manifest,
        },
        "memory": {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }

    with tempfile.TemporaryDirectory(
        prefix=f"skillgr-optimizer-artifacts-{args.arm}.", dir="/tmp"
    ) as directory:
        temporary_root = Path(directory)
        catalog_state = temporary_root / "catalog_trainable_state.pt"
        query_state = temporary_root / "query_step_50_trainable_state.pt"
        torch.save(observer.catalog_parameters, catalog_state)
        torch.save(query_parameters, query_state)
        args.output_dir.mkdir(parents=True)
        shutil.copy2(catalog_state, args.output_dir / catalog_state.name)
        shutil.copy2(query_state, args.output_dir / query_state.name)
    (args.output_dir / "probe_stdout.log").write_text(tee.captured(), encoding="utf-8")
    temporary_json = args.output_dir / "optimizer_boundary_probe.json.tmp"
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, args.output_dir / "optimizer_boundary_probe.json")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "optimizer_mode": args.optimizer_mode,
                "query_lr": args.query_lr,
                "catalog_step_2730": observer.losses["catalog_step_2730"],
                "query_step_1": observer.losses["query_step_1"],
                "query_step_50": observer.losses["query_step_50"],
                "catalog_model_sha256": observer.catalog["model_state_sha256"]["full"],
                "query_model_sha256": payload["query_step_50"]["model_state_sha256"]["full"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--path", choices=("default", "workspace_16"), required=True)
    parser.add_argument("--optimizer-mode", choices=OPTIMIZER_MODES, required=True)
    parser.add_argument("--query-lr", type=float, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog-steps", type=int, default=CATALOG_STEPS)
    parser.add_argument("--query-steps", type=int, default=QUERY_STEPS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
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
    parser.set_defaults(precision="bf16", func=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.dropout_rate <= 1.0:
        raise ValueError(f"dropout-rate must be in [0, 1], got {args.dropout_rate}")
    args.func(args)


if __name__ == "__main__":
    main()
