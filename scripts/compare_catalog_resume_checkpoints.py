#!/usr/bin/env python3
"""Measure numerical drift across structured-decoder resume checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CHUNK_ELEMENTS = 1_000_000
DEFAULT_HASH_CHUNK_BYTES = 16 * 1024 * 1024


def parse_label_path(value: str) -> tuple[str, Path]:
    label, separator, path_text = value.partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(path_text).expanduser()


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key_token(key: Any) -> str:
    if isinstance(key, str):
        return f"str:{key}"
    if isinstance(key, int) and not isinstance(key, bool):
        return f"int:{key}"
    return f"{type(key).__name__}:{key!r}"


def object_skeleton(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {
            _key_token(key): object_skeleton(value[key])
            for key in sorted(value, key=_key_token)
        }
    if isinstance(value, (list, tuple)):
        return [object_skeleton(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stable_object_sha256(value: Any) -> str:
    encoded = json.dumps(
        object_skeleton(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def object_value_sha256(value: Any, *, prefix: str) -> str:
    digest = hashlib.sha256()
    digest.update(stable_object_sha256(value).encode("ascii"))
    digest.update(tensor_mapping_sha256(flatten_tensors(value, prefix=prefix)).encode("ascii"))
    return digest.hexdigest()


def flatten_tensors(value: Any, *, prefix: str = "root") -> dict[str, Any]:
    import torch

    tensors: dict[str, Any] = {}

    def visit(item: Any, path: str) -> None:
        if torch.is_tensor(item):
            tensors[path] = item.detach().cpu().contiguous()
            return
        if isinstance(item, dict):
            for key in sorted(item, key=_key_token):
                visit(item[key], f"{path}/{_key_token(key)}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")

    visit(value, prefix)
    return tensors


def tensor_mapping_sha256(
    tensors: dict[str, Any],
    *,
    chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
) -> str:
    import torch

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    for key in sorted(tensors):
        tensor = tensors[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        byte_view = tensor.reshape(-1).view(torch.uint8)
        for start in range(0, byte_view.numel(), chunk_bytes):
            digest.update(byte_view[start : start + chunk_bytes].numpy().tobytes())
    return digest.hexdigest()


def _tensor_chunks(tensor, chunk_elements: int) -> Iterable[Any]:
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), chunk_elements):
        yield flat[start : start + chunk_elements]


def compare_tensor_mappings(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
    top_k: int = 20,
) -> dict[str, Any]:
    import torch

    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    left_keys = set(left)
    right_keys = set(right)
    common_keys = sorted(left_keys & right_keys)
    missing_from_left = sorted(right_keys - left_keys)
    missing_from_right = sorted(left_keys - right_keys)

    total_numel = 0
    exact_numel = 0
    sum_sq_diff = 0.0
    sum_sq_left = 0.0
    sum_sq_right = 0.0
    sum_abs_diff = 0.0
    max_abs_diff = 0.0
    finite_numel = 0
    left_nonfinite_numel = 0
    right_nonfinite_numel = 0
    incompatible = []
    tensor_rows = []

    for key in common_keys:
        left_tensor = left[key]
        right_tensor = right[key]
        if tuple(left_tensor.shape) != tuple(right_tensor.shape) or left_tensor.dtype != right_tensor.dtype:
            incompatible.append(
                {
                    "key": key,
                    "left_shape": list(left_tensor.shape),
                    "right_shape": list(right_tensor.shape),
                    "left_dtype": str(left_tensor.dtype),
                    "right_dtype": str(right_tensor.dtype),
                }
            )
            continue

        tensor_numel = int(left_tensor.numel())
        tensor_exact = 0
        tensor_sum_sq_diff = 0.0
        tensor_sum_sq_left = 0.0
        tensor_sum_abs_diff = 0.0
        tensor_max_abs_diff = 0.0
        tensor_finite_numel = 0
        tensor_left_nonfinite = 0
        tensor_right_nonfinite = 0
        for left_chunk, right_chunk in zip(
            _tensor_chunks(left_tensor, chunk_elements),
            _tensor_chunks(right_tensor, chunk_elements),
        ):
            tensor_exact += int(torch.eq(left_chunk, right_chunk).sum().item())
            if left_chunk.is_complex():
                left_numeric = left_chunk.to(torch.complex128)
                right_numeric = right_chunk.to(torch.complex128)
            else:
                left_numeric = left_chunk.to(torch.float64)
                right_numeric = right_chunk.to(torch.float64)
            left_finite = torch.isfinite(left_numeric)
            right_finite = torch.isfinite(right_numeric)
            finite_pair = left_finite & right_finite
            tensor_left_nonfinite += int((~left_finite).sum().item())
            tensor_right_nonfinite += int((~right_finite).sum().item())
            tensor_finite_numel += int(finite_pair.sum().item())
            if not bool(finite_pair.any()):
                continue
            left_numeric = left_numeric[finite_pair]
            right_numeric = right_numeric[finite_pair]
            difference = left_numeric - right_numeric
            absolute = difference.abs()
            tensor_sum_sq_diff += float((absolute * absolute).sum().item())
            tensor_sum_sq_left += float((left_numeric.abs() ** 2).sum().item())
            sum_sq_right += float((right_numeric.abs() ** 2).sum().item())
            tensor_sum_abs_diff += float(absolute.sum().item())
            if absolute.numel():
                tensor_max_abs_diff = max(tensor_max_abs_diff, float(absolute.max().item()))

        total_numel += tensor_numel
        exact_numel += tensor_exact
        sum_sq_diff += tensor_sum_sq_diff
        sum_sq_left += tensor_sum_sq_left
        sum_abs_diff += tensor_sum_abs_diff
        max_abs_diff = max(max_abs_diff, tensor_max_abs_diff)
        finite_numel += tensor_finite_numel
        left_nonfinite_numel += tensor_left_nonfinite
        right_nonfinite_numel += tensor_right_nonfinite
        tensor_rows.append(
            {
                "key": key,
                "numel": tensor_numel,
                "exact_fraction": tensor_exact / max(1, tensor_numel),
                "finite_numel": tensor_finite_numel,
                "left_nonfinite_numel": tensor_left_nonfinite,
                "right_nonfinite_numel": tensor_right_nonfinite,
                "rms_diff": math.sqrt(tensor_sum_sq_diff / max(1, tensor_finite_numel)),
                "relative_l2": math.sqrt(tensor_sum_sq_diff)
                / max(math.sqrt(tensor_sum_sq_left), 1e-30),
                "max_abs_diff": tensor_max_abs_diff,
            }
        )

    tensor_rows.sort(key=lambda row: (row["relative_l2"], row["max_abs_diff"]), reverse=True)
    left_l2 = math.sqrt(sum_sq_left)
    right_l2 = math.sqrt(sum_sq_right)
    diff_l2 = math.sqrt(sum_sq_diff)
    return {
        "left_tensor_count": len(left),
        "right_tensor_count": len(right),
        "compared_tensor_count": len(common_keys) - len(incompatible),
        "missing_from_left": missing_from_left,
        "missing_from_right": missing_from_right,
        "incompatible_tensors": incompatible,
        "schema_exact": not missing_from_left and not missing_from_right and not incompatible,
        "total_numel": total_numel,
        "finite_numel": finite_numel,
        "left_nonfinite_numel": left_nonfinite_numel,
        "right_nonfinite_numel": right_nonfinite_numel,
        "exact_numel": exact_numel,
        "exact_fraction": exact_numel / max(1, total_numel),
        "mean_abs_diff": sum_abs_diff / max(1, finite_numel),
        "rms_diff": math.sqrt(sum_sq_diff / max(1, finite_numel)),
        "max_abs_diff": max_abs_diff,
        "difference_l2": diff_l2,
        "left_l2": left_l2,
        "right_l2": right_l2,
        "relative_l2_to_left": diff_l2 / max(left_l2, 1e-30),
        "symmetric_relative_l2": diff_l2 / max((left_l2 + right_l2) / 2.0, 1e-30),
        "top_drift_tensors": tensor_rows[:top_k],
    }


def checkpoint_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    model_tensors = flatten_tensors(payload["state_dict"], prefix="model")
    optimizer_tensors = flatten_tensors(payload.get("optimizer_state_dict", {}), prefix="optimizer")
    training_state = payload.get("training_state") or {}
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "model_state_sha256": tensor_mapping_sha256(model_tensors),
        "optimizer_tensor_sha256": tensor_mapping_sha256(optimizer_tensors),
        "optimizer_structure_sha256": stable_object_sha256(
            payload.get("optimizer_state_dict", {})
        ),
        "model_tensor_count": len(model_tensors),
        "optimizer_tensor_count": len(optimizer_tensors),
        "metadata": {
            "model": payload.get("model"),
            "seed": payload.get("seed"),
            "training_formulation": payload.get("training_formulation"),
            "semantic_ids_sha256": payload.get("semantic_ids_sha256"),
            "distributed_sampler": payload.get("distributed_sampler"),
            "optimizer_schedule": payload.get("optimizer_schedule"),
            "optimizer_param_groups": (
                payload.get("optimizer_state_dict", {}).get("param_groups")
                if isinstance(payload.get("optimizer_state_dict"), dict)
                else None
            ),
            "source_text_policy": payload.get("source_text_policy"),
            "method_setting": payload.get("method_setting"),
            "training_state": {
                key: training_state.get(key)
                for key in (
                    "format",
                    "stage",
                    "completed_steps",
                    "display_step",
                    "total_steps",
                    "next_epoch",
                    "next_batch_index",
                    "world_size",
                    "per_device_batch_size",
                )
            },
            "scheduler_state_sha256": object_value_sha256(
                training_state.get("scheduler_state_dict"),
                prefix="scheduler",
            ),
            "rng_state_sha256": object_value_sha256(
                training_state.get("rng_states"),
                prefix="rng",
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_label_path,
        required=True,
        metavar="LABEL=PATH",
        help="structured_decoder.pt checkpoint; repeat for each run",
    )
    parser.add_argument(
        "--input-file",
        action="append",
        type=parse_label_path,
        default=[],
        metavar="LABEL=PATH",
        help="input/config file to fingerprint for provenance",
    )
    parser.add_argument(
        "--metric",
        action="append",
        type=parse_label_path,
        default=[],
        metavar="LABEL=PATH",
        help="teacher-forced catalog metric JSON matching a checkpoint label",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-elements", type=int, default=DEFAULT_CHUNK_ELEMENTS)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main() -> None:
    import torch

    args = build_parser().parse_args()
    checkpoint_items = args.checkpoint
    labels = [label for label, _ in checkpoint_items]
    if len(checkpoint_items) < 2:
        raise SystemExit("at least two --checkpoint values are required")
    if len(set(labels)) != len(labels):
        raise SystemExit("checkpoint labels must be unique")
    if args.top_k < 0:
        raise SystemExit("top-k must be non-negative")

    checkpoint_paths = {
        label: path.resolve(strict=True)
        for label, path in checkpoint_items
    }
    summaries: dict[str, dict[str, Any]] = {}
    pairs = []
    for left_label, right_label in itertools.combinations(labels, 2):
        left_path = checkpoint_paths[left_label]
        right_path = checkpoint_paths[right_label]
        left_payload = torch.load(left_path, map_location="cpu", weights_only=False)
        right_payload = torch.load(right_path, map_location="cpu", weights_only=False)
        for label, path, payload in (
            (left_label, left_path, left_payload),
            (right_label, right_path, right_payload),
        ):
            if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
                raise SystemExit(f"invalid structured decoder checkpoint: {path}")
            if label not in summaries:
                summaries[label] = checkpoint_summary(path, payload)
        left_model_tensors = flatten_tensors(left_payload["state_dict"], prefix="model")
        right_model_tensors = flatten_tensors(right_payload["state_dict"], prefix="model")
        left_optimizer_tensors = flatten_tensors(
            left_payload.get("optimizer_state_dict", {}), prefix="optimizer"
        )
        right_optimizer_tensors = flatten_tensors(
            right_payload.get("optimizer_state_dict", {}), prefix="optimizer"
        )
        left_training_state = left_payload.get("training_state") or {}
        right_training_state = right_payload.get("training_state") or {}
        pairs.append(
            {
                "left": left_label,
                "right": right_label,
                "model": compare_tensor_mappings(
                    left_model_tensors,
                    right_model_tensors,
                    chunk_elements=args.chunk_elements,
                    top_k=args.top_k,
                ),
                "optimizer": compare_tensor_mappings(
                    left_optimizer_tensors,
                    right_optimizer_tensors,
                    chunk_elements=args.chunk_elements,
                    top_k=args.top_k,
                ),
                "optimizer_structure_equal": stable_object_sha256(
                    left_payload.get("optimizer_state_dict", {})
                )
                == stable_object_sha256(right_payload.get("optimizer_state_dict", {})),
                "scheduler_state_equal": object_value_sha256(
                    left_training_state.get("scheduler_state_dict"), prefix="scheduler"
                )
                == object_value_sha256(
                    right_training_state.get("scheduler_state_dict"), prefix="scheduler"
                ),
                "rng_state_equal": object_value_sha256(
                    left_training_state.get("rng_states"), prefix="rng"
                )
                == object_value_sha256(
                    right_training_state.get("rng_states"), prefix="rng"
                ),
            }
        )
        del (
            left_model_tensors,
            right_model_tensors,
            left_optimizer_tensors,
            right_optimizer_tensors,
            left_payload,
            right_payload,
        )
        gc.collect()

    inputs = {}
    for label, path in args.input_file:
        if label in inputs:
            raise SystemExit(f"duplicate input label: {label}")
        resolved = path.resolve(strict=True)
        inputs[label] = {
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "mtime_ns": resolved.stat().st_mtime_ns,
            "sha256": file_sha256(resolved),
        }

    metrics = {}
    for label, path in args.metric:
        if label not in checkpoint_paths:
            raise SystemExit(f"metric label has no matching checkpoint: {label}")
        if label in metrics:
            raise SystemExit(f"duplicate metric label: {label}")
        resolved = path.resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        metrics[label] = {
            "path": str(resolved),
            "file_sha256": file_sha256(resolved),
            "rows": int(payload["rows"]),
            "teacher_forced_ce": float(payload["teacher_forced_ce"]),
            "teacher_forced_sid_exact_accuracy": float(
                payload["teacher_forced_sid_exact_accuracy"]
            ),
            "teacher_forced_sequence_exact_accuracy": float(
                payload["teacher_forced_sequence_exact_accuracy"]
            ),
            "by_type": payload.get("by_type"),
            "conditions": payload.get("conditions"),
        }
    metric_pairs = []
    for left_label, right_label in itertools.combinations(labels, 2):
        if left_label not in metrics or right_label not in metrics:
            continue
        left_metric = metrics[left_label]
        right_metric = metrics[right_label]
        metric_pairs.append(
            {
                "left": left_label,
                "right": right_label,
                "teacher_forced_ce_delta_right_minus_left": (
                    right_metric["teacher_forced_ce"] - left_metric["teacher_forced_ce"]
                ),
                "teacher_forced_sid_exact_delta_right_minus_left": (
                    right_metric["teacher_forced_sid_exact_accuracy"]
                    - left_metric["teacher_forced_sid_exact_accuracy"]
                ),
            }
        )

    result = {
        "format": "structured_catalog_resume_comparison_v1",
        "checkpoints": summaries,
        "pairs": pairs,
        "input_files": inputs,
        "teacher_forced_metrics": metrics,
        "teacher_forced_metric_pairs": metric_pairs,
        "interpretation": {
            "acceptance_threshold": None,
            "historical_input_equivalence_proven": False,
            "note": (
                "input hashes describe files at diagnostic time; the original run did not "
                "record these hashes"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pairs": len(pairs)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
