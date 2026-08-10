#!/usr/bin/env python3
"""Trace one archived catalog update without modifying the archived probe."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

try:
    from legacy_baseline_parity_probe import _hash_tensor_state
    from legacy_runtime_matrix_probe import (
        apply_precision_controls,
        file_sha256,
        precision_snapshot,
        runtime_snapshot,
        selected_weight_file,
    )
except ModuleNotFoundError:
    from scripts.legacy_baseline_parity_probe import _hash_tensor_state
    from scripts.legacy_runtime_matrix_probe import (
        apply_precision_controls,
        file_sha256,
        precision_snapshot,
        runtime_snapshot,
        selected_weight_file,
    )


EXPECTED_PIPELINE_SHA256 = "7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76"
EXPECTED_INITIAL_MODEL_FINGERPRINT = "c219d07a2ac425cffd677cefc0792a81f47aa3ad9d0fa76a0bfa37cdf9f5ebde"
EXPECTED_FIRST_BATCH_SHA256 = "721a0d93aaef2f9d0104adae8501fcfabcefcd6bd9f23c1e4a1aae701710d443"
EXPECTED_DROPOUT_MODULES = 86
EXPECTED_DROPOUT0_DEFAULT_BF16_STEP1_LOSS = 4.554369926452637
OPTIONAL_BOOL_CHOICES = ("auto", "true", "false")


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _sample_indices(numel: int, count: int = 8) -> list[int]:
    if numel <= 0:
        return []
    if numel <= count:
        return list(range(numel))
    return sorted({round(index * (numel - 1) / (count - 1)) for index in range(count)})


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().cpu().contiguous()
    flat = value.reshape(-1)
    numeric = flat.to(torch.float64)
    finite = torch.isfinite(numeric)
    finite_values = numeric[finite]
    indices = _sample_indices(flat.numel())
    samples = numeric[indices].tolist() if indices else []
    if finite_values.numel():
        l2_norm = float(torch.linalg.vector_norm(finite_values).item())
        abs_max = float(finite_values.abs().max().item())
        abs_mean = float(finite_values.abs().mean().item())
    else:
        l2_norm = None
        abs_max = None
        abs_mean = None
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": int(value.numel()),
        "finite_fraction": float(finite.sum().item() / max(1, finite.numel())),
        "l2_norm": l2_norm,
        "abs_max": abs_max,
        "abs_mean": abs_mean,
        "sample_indices": indices,
        "sample_values": [float(item) for item in samples],
        "sha256": tensor_sha256(value),
    }


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


class TensorArtifactWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def capture(self, kind: str, name: str, tensor: Any, *, save: bool) -> dict[str, Any]:
        import torch

        record = {"kind": kind, "name": name, **tensor_summary(tensor)}
        if save:
            relative = Path("tensors") / f"{self.counter:05d}.pt"
            self.counter += 1
            target = self.root / relative.name
            torch.save(tensor.detach().cpu().contiguous(), target)
            record["artifact"] = str(relative)
        else:
            record["artifact"] = None
        return record


class TraceRecorder:
    def __init__(self, writer: TensorArtifactWriter, *, enabled: bool) -> None:
        self.writer = writer
        self.enabled = enabled
        self.model = None
        self.handles: list[Any] = []
        self.activation_records: list[dict[str, Any]] = []
        self.ce_records: list[dict[str, Any]] = []
        self.gradient_records: list[dict[str, Any]] = []
        self.update_records: list[dict[str, Any]] = []
        self.module_calls: dict[str, int] = {}
        self.event_counter = 0
        self.optimizer_steps = 0
        self._parameter_before_step: dict[str, Any] = {}

    def remember_model(self, model: Any) -> Any:
        self.model = model
        if self.enabled:
            self.attach_model(model)
        return model

    def _capture_tree(
        self,
        kind: str,
        name: str,
        value: Any,
        *,
        save_first_float: bool,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        saved = False
        for path, tensor in iter_tensors(value):
            should_save = bool(save_first_float and not saved and tensor.is_floating_point())
            records.append(
                self.writer.capture(kind, f"{name}:{path}", tensor, save=should_save)
            )
            saved = saved or should_save
        return records

    def register_module(self, name: str, module: Any) -> None:
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            call_index = self.module_calls.get(name, 0)
            self.module_calls[name] = call_index + 1
            call_name = f"{name}#{call_index}"
            event_order = self.event_counter
            self.event_counter += 1
            self.activation_records.append(
                {
                    "boundary": name,
                    "call_index": call_index,
                    "event_order": event_order,
                    "inputs": self._capture_tree(
                        "activation_input",
                        call_name,
                        {"args": args, "kwargs": kwargs},
                        save_first_float=False,
                    ),
                    "outputs": self._capture_tree(
                        "activation_output",
                        call_name,
                        output,
                        save_first_float=True,
                    ),
                }
            )

        self.handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def attach_model(self, model: Any) -> None:
        encoder_blocks = list(model.backbone.encoder.block)
        decoder_blocks = list(model.backbone.decoder.block)
        code_heads = list(model.code_heads)
        if len(encoder_blocks) != 12 or len(decoder_blocks) != 12 or len(code_heads) != 4:
            raise ValueError(
                "unexpected structured T5 topology: "
                f"encoder={len(encoder_blocks)}, decoder={len(decoder_blocks)}, "
                f"code_heads={len(code_heads)}"
            )
        self.register_module("shared_embedding", model.backbone.shared)
        for index, block in enumerate(encoder_blocks):
            self.register_module(f"encoder.block.{index}", block)
        self.register_module("sid_embedding", model.sid_embedding)
        for index, block in enumerate(decoder_blocks):
            self.register_module(f"decoder.block.{index}", block)
        for index, head in enumerate(code_heads):
            self.register_module(f"code_head.{index}", head)
        self.register_module("control_head", model.control_head)
        self.register_module("loss.total", model)

    def record_cross_entropy(self, logits: Any, labels: Any, output: Any) -> None:
        call_index = len(self.ce_records)
        name = f"cross_entropy.{call_index}"
        event_order = self.event_counter
        self.event_counter += 1
        self.ce_records.append(
            {
                "boundary": name,
                "call_index": call_index,
                "event_order": event_order,
                "inputs": self._capture_tree(
                    "ce_input",
                    name,
                    {"logits": logits, "labels": labels},
                    save_first_float=False,
                ),
                "outputs": self._capture_tree(
                    "ce_output",
                    name,
                    output,
                    save_first_float=True,
                ),
            }
        )

    def wrap_optimizer(self, optimizer: Any) -> Any:
        original_step = optimizer.step

        def traced_step(*args: Any, **kwargs: Any) -> Any:
            self.before_optimizer_step()
            result = original_step(*args, **kwargs)
            self.after_optimizer_step()
            return result

        optimizer.step = traced_step
        return optimizer

    def before_optimizer_step(self) -> None:
        if self.model is None:
            raise RuntimeError("optimizer step occurred before model capture")
        if self.optimizer_steps != 0:
            raise RuntimeError("layerwise trace supports exactly one optimizer step")
        self._parameter_before_step = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            self._parameter_before_step[name] = parameter.detach().clone()
            if parameter.grad is None:
                self.gradient_records.append({"name": name, "present": False})
            else:
                self.gradient_records.append(
                    {
                        "name": name,
                        "present": True,
                        **self.writer.capture("gradient", name, parameter.grad, save=True),
                    }
                )

    def after_optimizer_step(self) -> None:
        if self.model is None:
            raise RuntimeError("optimizer step occurred before model capture")
        for name, parameter in self.model.named_parameters():
            if name not in self._parameter_before_step:
                continue
            update = parameter.detach() - self._parameter_before_step[name]
            self.update_records.append(
                {"name": name, **self.writer.capture("update", name, update, save=True)}
            )
        self._parameter_before_step.clear()
        self.optimizer_steps += 1

    def validate(self) -> None:
        if not self.enabled:
            return
        expected_boundaries = {
            "shared_embedding",
            "sid_embedding",
            "control_head",
            "loss.total",
            *(f"encoder.block.{index}" for index in range(12)),
            *(f"decoder.block.{index}" for index in range(12)),
            *(f"code_head.{index}" for index in range(4)),
        }
        missing = sorted(expected_boundaries - set(self.module_calls))
        if missing:
            raise ValueError(f"trace missed expected boundaries: {missing}")
        if len(self.ce_records) != 5:
            raise ValueError(f"expected 5 CE calls, observed {len(self.ce_records)}")
        if self.optimizer_steps != 1:
            raise ValueError(f"expected 1 optimizer step, observed {self.optimizer_steps}")


class TeeCapture(io.TextIOBase):
    def __init__(self, target: Any) -> None:
        self.target = target
        self.buffer = io.StringIO()

    def write(self, text: str) -> int:
        self.target.write(text)
        self.buffer.write(text)
        return len(text)

    def flush(self) -> None:
        self.target.flush()

    def captured(self) -> str:
        return self.buffer.getvalue()


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_probe_events(stdout: str) -> dict[str, Any]:
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
    trains = [event for event in events if event.get("event") == "train"]
    if len(headers) != 1 or len(batches) != 1 or len(trains) != 1:
        raise ValueError(
            "unexpected archived probe events: "
            f"headers={len(headers)}, batches={len(batches)}, trains={len(trains)}"
        )
    return {"header": headers[0], "first_batch": batches[0], "train": trains[0]}


def command_run(args: argparse.Namespace) -> None:
    if args.steps != 1:
        raise ValueError(f"layerwise trace requires steps=1, got {args.steps}")
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
            f"each trace arm requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite trace output: {args.output_dir}")
    if not args.probe.is_file():
        raise FileNotFoundError(args.probe)
    pipeline_path = args.probe.parent / "skill_gr_pipeline.original_20260714.py"
    if file_sha256(pipeline_path) != EXPECTED_PIPELINE_SHA256:
        raise ValueError(f"archived pipeline SHA-256 mismatch: {pipeline_path}")

    configured_precision = apply_precision_controls(torch, args)
    trace_enabled = args.trace == "on"

    with tempfile.TemporaryDirectory(prefix=f"skillgr-layer-trace-{args.arm}.", dir="/tmp") as directory:
        temporary_root = Path(directory)
        writer = TensorArtifactWriter(temporary_root / "tensors")
        recorder = TraceRecorder(writer, enabled=trace_enabled)

        dropout_modules: list[Any] = []
        original_dropout = torch.nn.Dropout

        class ControlledDropout(original_dropout):
            def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
                super().__init__(p=args.dropout_rate, inplace=inplace)
                dropout_modules.append(self)

        torch.nn.Dropout = ControlledDropout

        original_autocast = torch.autocast
        if args.precision == "fp32":
            def controlled_autocast(*autocast_args: Any, **autocast_kwargs: Any) -> Any:
                device_type = (
                    autocast_args[0]
                    if autocast_args
                    else autocast_kwargs.get("device_type")
                )
                if device_type == "cuda":
                    autocast_kwargs["enabled"] = False
                return original_autocast(*autocast_args, **autocast_kwargs)

            torch.autocast = controlled_autocast

        original_cross_entropy = functional.cross_entropy
        if trace_enabled:
            def traced_cross_entropy(logits: Any, labels: Any, *ce_args: Any, **ce_kwargs: Any) -> Any:
                output = original_cross_entropy(logits, labels, *ce_args, **ce_kwargs)
                recorder.record_cross_entropy(logits, labels, output)
                return output

            functional.cross_entropy = traced_cross_entropy

        original_adamw = torch.optim.AdamW
        if trace_enabled:
            def traced_adamw(*optimizer_args: Any, **optimizer_kwargs: Any) -> Any:
                return recorder.wrap_optimizer(original_adamw(*optimizer_args, **optimizer_kwargs))

            torch.optim.AdamW = traced_adamw

        probe = load_module(args.probe, f"catalog_trace_probe_{args.arm}")
        original_load_pipeline = probe.load_pipeline

        def traced_load_pipeline(path: Path) -> Any:
            pipeline = original_load_pipeline(path)
            original_factory = pipeline.make_structured_t5_sid_model

            def traced_factory(*factory_args: Any, **factory_kwargs: Any) -> Any:
                return recorder.remember_model(original_factory(*factory_args, **factory_kwargs))

            pipeline.make_structured_t5_sid_model = traced_factory
            return pipeline

        probe.load_pipeline = traced_load_pipeline
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
            torch.autocast = original_autocast
            functional.cross_entropy = original_cross_entropy
            torch.optim.AdamW = original_adamw

        if recorder.model is None:
            raise RuntimeError("archived probe did not construct a model")
        recorder.validate()
        events = parse_probe_events(tee.captured())
        header = events["header"]
        first_batch = events["first_batch"]
        if header.get("source_sha256") != EXPECTED_PIPELINE_SHA256:
            raise ValueError(f"probe source mismatch: {header.get('source_sha256')}")
        if header.get("initial_model_fingerprint") != EXPECTED_INITIAL_MODEL_FINGERPRINT:
            raise ValueError(
                "initial model fingerprint mismatch: "
                f"{header.get('initial_model_fingerprint')}"
            )
        if first_batch.get("row_ids_sha256") != EXPECTED_FIRST_BATCH_SHA256:
            raise ValueError(f"first catalog batch mismatch: {first_batch.get('row_ids_sha256')}")
        observed_loss = float(events["train"]["train_loss"])
        if (
            args.path == "default_bf16"
            and args.precision == "bf16"
            and args.dropout_rate == 0.0
            and observed_loss != EXPECTED_DROPOUT0_DEFAULT_BF16_STEP1_LOSS
        ):
            raise ValueError(
                "dropout=0 default BF16 step-1 loss mismatch: "
                f"expected={EXPECTED_DROPOUT0_DEFAULT_BF16_STEP1_LOSS}, "
                f"observed={observed_loss}"
            )
        dropout_rates = sorted({float(module.p) for module in dropout_modules})
        if len(dropout_modules) != EXPECTED_DROPOUT_MODULES or dropout_rates != [args.dropout_rate]:
            raise ValueError(
                "dropout override mismatch: "
                f"count={len(dropout_modules)}, rates={dropout_rates}"
            )

        weight_file = selected_weight_file(args.model)
        condition = {
            "arm": args.arm,
            "trace": trace_enabled,
            "path": args.path,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "max_source_length": args.max_source_length,
            "seed": args.seed,
            "precision": args.precision,
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
            "artifact_format": "legacy_layerwise_trace_arm_v1",
            "condition": condition,
            "condition_sha256": stable_json_sha256(condition),
            "runtime": runtime_snapshot(torch),
            "precision": precision_snapshot(torch),
            "environment": {
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING"),
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
            "losses": {"step_1": observed_loss},
            "dropout": {
                "requested_rate": args.dropout_rate,
                "module_count": len(dropout_modules),
                "module_rates": dropout_rates,
            },
            "final_state_sha256": _hash_tensor_state(recorder.model.state_dict()),
            "trace": {
                "enabled": trace_enabled,
                "optimizer_steps": recorder.optimizer_steps,
                "module_calls": recorder.module_calls,
                "activations": recorder.activation_records,
                "cross_entropy": recorder.ce_records,
                "gradients": recorder.gradient_records,
                "updates": recorder.update_records,
            },
        }

        args.output_dir.mkdir(parents=True)
        if trace_enabled:
            shutil.copytree(temporary_root / "tensors", args.output_dir / "tensors")
        (args.output_dir / "probe_stdout.log").write_text(tee.captured(), encoding="utf-8")
        temporary_json = args.output_dir / "trace_probe.json.tmp"
        temporary_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_json, args.output_dir / "trace_probe.json")
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "trace": trace_enabled,
                    "loss": payload["losses"]["step_1"],
                    "final_state_sha256": payload["final_state_sha256"]["full"],
                    "artifacts": writer.counter,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--trace", choices=("on", "off"), required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
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
    parser.set_defaults(func=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.dropout_rate <= 1.0:
        raise ValueError(f"dropout-rate must be in [0, 1], got {args.dropout_rate}")
    args.func(args)


if __name__ == "__main__":
    main()
