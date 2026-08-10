#!/usr/bin/env python3
"""Validate trace/control equivalence and compare layerwise tensor divergence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_PATHS = ("default_bf16", "workspace_16_bf16", "cublaslt_bf16", "fp32")


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_format") != "legacy_layerwise_trace_arm_v1":
        raise ValueError(f"unexpected trace artifact at {path}")
    return payload


def normalized_condition(payload: dict[str, Any]) -> dict[str, Any]:
    condition = dict(payload["condition"])
    condition.pop("arm", None)
    condition.pop("trace", None)
    return condition


def compare_trace_control_pair(
    control: dict[str, Any],
    traced: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "condition": normalized_condition(control) == normalized_condition(traced),
        "file_sha256": control["file_sha256"] == traced["file_sha256"],
        "initial_model_fingerprint": (
            control["probe_header"]["initial_model_fingerprint"]
            == traced["probe_header"]["initial_model_fingerprint"]
        ),
        "cpu_rng_after_model": (
            control["probe_header"]["cpu_rng_state_sha256_after_model_init"]
            == traced["probe_header"]["cpu_rng_state_sha256_after_model_init"]
        ),
        "cuda_rng_before_forward": (
            control["probe_header"]["cuda_rng_state_sha256_before_first_forward"]
            == traced["probe_header"]["cuda_rng_state_sha256_before_first_forward"]
        ),
        "first_catalog_batch": (
            control["first_catalog_batch"] == traced["first_catalog_batch"]
        ),
        "step_1_loss": control["losses"]["step_1"] == traced["losses"]["step_1"],
        "final_state_sha256": (
            control["final_state_sha256"] == traced["final_state_sha256"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "control_loss": control["losses"]["step_1"],
        "trace_loss": traced["losses"]["step_1"],
        "control_final_state_sha256": control["final_state_sha256"],
        "trace_final_state_sha256": traced["final_state_sha256"],
    }


def _artifact_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    saved = [record for record in records if record.get("artifact")]
    if len(saved) != 1:
        raise ValueError(f"expected exactly one saved tensor, found {len(saved)}")
    return saved[0]


def activation_artifacts(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    sequence = sorted([
        *payload["trace"]["activations"],
        *payload["trace"]["cross_entropy"],
    ], key=lambda record: record["event_order"])
    for record in sequence:
        key = f"{record['boundary']}#{record['call_index']}"
        if key in records:
            raise ValueError(f"duplicate activation key: {key}")
        records[key] = {
            "order": int(record["event_order"]),
            **_artifact_record(record["outputs"]),
        }
    return records


def parameter_artifacts(
    payload: dict[str, Any],
    field: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for order, record in enumerate(payload["trace"][field]):
        if not record.get("present", True):
            continue
        name = record["name"]
        if name in records:
            raise ValueError(f"duplicate {field} tensor: {name}")
        if not record.get("artifact"):
            raise ValueError(f"missing {field} artifact: {name}")
        records[name] = {"order": order, **record}
    return records


def resolve_artifact(arm_dir: Path, relative: str) -> Path:
    root = arm_dir.resolve()
    path = (arm_dir / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact escapes arm directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def tensor_relative_l2(
    reference_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    import torch

    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"tensor shape mismatch: {reference_path}={tuple(reference.shape)}, "
            f"{candidate_path}={tuple(candidate.shape)}"
        )
    reference64 = reference.to(torch.float64)
    candidate64 = candidate.to(torch.float64)
    reference_norm = float(torch.linalg.vector_norm(reference64).item())
    candidate_norm = float(torch.linalg.vector_norm(candidate64).item())
    difference_norm = float(torch.linalg.vector_norm(candidate64 - reference64).item())
    if reference_norm == 0.0:
        relative_l2 = 0.0 if difference_norm == 0.0 else None
        denominator_zero = True
    else:
        relative_l2 = difference_norm / reference_norm
        denominator_zero = False
    return {
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "difference_norm": difference_norm,
        "relative_l2": relative_l2,
        "denominator_zero": denominator_zero,
        "exact": bool(torch.equal(reference, candidate)),
    }


def compare_artifact_group(
    reference_dir: Path,
    reference_records: dict[str, dict[str, Any]],
    candidate_dir: Path,
    candidate_records: dict[str, dict[str, Any]],
    *,
    include_growth: bool,
) -> dict[str, Any]:
    if set(reference_records) != set(candidate_records):
        missing = sorted(set(reference_records) - set(candidate_records))
        extra = sorted(set(candidate_records) - set(reference_records))
        raise ValueError(f"artifact key mismatch: missing={missing}, extra={extra}")
    ordered_keys = sorted(reference_records, key=lambda key: reference_records[key]["order"])
    comparisons: list[dict[str, Any]] = []
    previous_relative: float | None = None
    first_nonzero = None
    for key in ordered_keys:
        reference = reference_records[key]
        candidate = candidate_records[key]
        metrics = tensor_relative_l2(
            resolve_artifact(reference_dir, reference["artifact"]),
            resolve_artifact(candidate_dir, candidate["artifact"]),
        )
        relative = metrics["relative_l2"]
        growth_ratio = None
        if include_growth and relative is not None and previous_relative not in (None, 0.0):
            growth_ratio = relative / previous_relative
        row = {
            "name": key,
            "reference_dtype": reference["dtype"],
            "candidate_dtype": candidate["dtype"],
            **metrics,
            "growth_ratio": growth_ratio,
        }
        comparisons.append(row)
        if first_nonzero is None and not metrics["exact"]:
            first_nonzero = key
        if include_growth and relative is not None:
            previous_relative = relative
    finite_rows = [row for row in comparisons if row["relative_l2"] is not None]
    ranked = sorted(finite_rows, key=lambda row: row["relative_l2"], reverse=True)
    growth_ranked = sorted(
        [row for row in comparisons if row["growth_ratio"] is not None],
        key=lambda row: row["growth_ratio"],
        reverse=True,
    )
    return {
        "count": len(comparisons),
        "first_nonzero": first_nonzero,
        "max_relative_l2": ranked[0]["relative_l2"] if ranked else None,
        "top_relative_l2": ranked[:20],
        "top_growth_ratio": growth_ranked[:20],
        "records": comparisons,
    }


def compare_traced_arms(
    root: Path,
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_dir = root / "arms" / reference["condition"]["arm"]
    candidate_dir = root / "arms" / candidate["condition"]["arm"]
    return {
        "reference_path": reference["condition"]["path"],
        "candidate_path": candidate["condition"]["path"],
        "activation": compare_artifact_group(
            reference_dir,
            activation_artifacts(reference),
            candidate_dir,
            activation_artifacts(candidate),
            include_growth=True,
        ),
        "gradient": compare_artifact_group(
            reference_dir,
            parameter_artifacts(reference, "gradients"),
            candidate_dir,
            parameter_artifacts(candidate, "gradients"),
            include_growth=False,
        ),
        "update": compare_artifact_group(
            reference_dir,
            parameter_artifacts(reference, "updates"),
            candidate_dir,
            parameter_artifacts(candidate, "updates"),
            include_growth=False,
        ),
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Legacy A800 layerwise trace summary",
        "",
        "## Instrumentation equivalence",
        "",
        "| Path | Passed | Control loss | Trace loss |",
        "|---|---:|---:|---:|",
    ]
    for path in EXPECTED_PATHS:
        pair = summary["pair_equivalence"][path]
        lines.append(
            f"| `{path}` | `{pair['passed']}` | {pair['control_loss']:.12f} | "
            f"{pair['trace_loss']:.12f} |"
        )
    lines.extend(["", "## Traced-path divergence", ""])
    for name, comparison in summary["comparisons"].items():
        activation = comparison["activation"]
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- First nonzero activation boundary: `{activation['first_nonzero']}`",
                f"- Maximum activation relative L2: `{activation['max_relative_l2']}`",
                f"- Maximum gradient relative L2: `{comparison['gradient']['max_relative_l2']}`",
                f"- Maximum update relative L2: `{comparison['update']['max_relative_l2']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def command_analyze(args: argparse.Namespace) -> None:
    arm_files = sorted((args.root / "arms").glob("*/trace_probe.json"))
    payloads = [load_payload(path) for path in arm_files]
    if len(payloads) != 8:
        raise ValueError(f"expected 8 trace arms, found {len(payloads)}")
    by_path: dict[str, dict[bool, dict[str, Any]]] = {}
    for payload in payloads:
        path = payload["condition"]["path"]
        trace = bool(payload["condition"]["trace"])
        if path in by_path and trace in by_path[path]:
            raise ValueError(f"duplicate path/trace arm: {path}, trace={trace}")
        by_path.setdefault(path, {})[trace] = payload
    if set(by_path) != set(EXPECTED_PATHS):
        raise ValueError(f"unexpected paths: {sorted(by_path)}")
    for path in EXPECTED_PATHS:
        if set(by_path[path]) != {False, True}:
            raise ValueError(f"path {path} lacks trace/control pair")

    pair_equivalence = {
        path: compare_trace_control_pair(by_path[path][False], by_path[path][True])
        for path in EXPECTED_PATHS
    }
    reference = by_path["default_bf16"][True]
    comparisons = {
        path: compare_traced_arms(args.root, reference, by_path[path][True])
        for path in EXPECTED_PATHS
        if path != "default_bf16"
    }
    summary = {
        "artifact_format": "legacy_layerwise_trace_matrix_v1",
        "arms": len(payloads),
        "pair_equivalence_passed": all(
            result["passed"] for result in pair_equivalence.values()
        ),
        "pair_equivalence": pair_equivalence,
        "comparisons": comparisons,
    }
    temporary = args.root / "layerwise_comparison.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.root / "layerwise_comparison.json")
    (args.root / "layerwise_comparison.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "arms": len(payloads),
                "pair_equivalence_passed": summary["pair_equivalence_passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not summary["pair_equivalence_passed"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
