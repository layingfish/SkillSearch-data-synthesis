#!/usr/bin/env python3
"""Analyze the default/workspace16 local-FP32 50-step matrix."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any


REGIONS = ("none", "encoder", "decoder", "heads_ce")
PATHS = ("default", "workspace_16")


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_format") != "legacy_local_fp32_arm_v1":
        raise ValueError(f"unexpected local FP32 artifact: {path}")
    return payload


def normalized_condition(payload: dict[str, Any]) -> dict[str, Any]:
    condition = dict(payload["condition"])
    condition.pop("arm", None)
    condition.pop("path", None)
    controls = dict(condition["requested_controls"])
    controls.pop("cublas_workspace_config", None)
    condition["requested_controls"] = controls
    return condition


def resolve_state(root: Path, payload: dict[str, Any]) -> Path:
    arm = payload["condition"]["arm"]
    arm_root = (root / "arms" / arm).resolve()
    path = (
        arm_root / payload["trajectory"]["trainable_state_artifact"]
    ).resolve()
    if not path.is_relative_to(arm_root):
        raise ValueError(f"state artifact escapes arm directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def state_distance_squared(left_path: Path, right_path: Path) -> float:
    import torch

    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    if set(left) != set(right):
        raise ValueError("trainable state parameter names differ")
    squared = 0.0
    for name in sorted(left):
        if tuple(left[name].shape) != tuple(right[name].shape):
            raise ValueError(f"trainable state shape differs: {name}")
        delta = left[name].to(torch.float64) - right[name].to(torch.float64)
        squared += float((delta * delta).sum().item())
    del left, right
    gc.collect()
    return squared


def pair_audit(default: dict[str, Any], workspace: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "condition": normalized_condition(default) == normalized_condition(workspace),
        "file_sha256": default["file_sha256"] == workspace["file_sha256"],
        "initial_model_fingerprint": (
            default["probe_header"]["initial_model_fingerprint"]
            == workspace["probe_header"]["initial_model_fingerprint"]
        ),
        "cpu_rng_after_model": (
            default["probe_header"]["cpu_rng_state_sha256_after_model_init"]
            == workspace["probe_header"]["cpu_rng_state_sha256_after_model_init"]
        ),
        "cuda_rng_before_forward": (
            default["probe_header"]["cuda_rng_state_sha256_before_first_forward"]
            == workspace["probe_header"]["cuda_rng_state_sha256_before_first_forward"]
        ),
        "first_catalog_batch": (
            default["first_catalog_batch"] == workspace["first_catalog_batch"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def region_metrics(
    root: Path,
    default: dict[str, Any],
    workspace: dict[str, Any],
) -> dict[str, Any]:
    audit = pair_audit(default, workspace)
    if not audit["passed"]:
        raise ValueError(
            f"pair provenance mismatch for region={default['condition']['region']}: {audit}"
        )
    numerator_squared = state_distance_squared(
        resolve_state(root, default),
        resolve_state(root, workspace),
    )
    denominator_squared = float(
        default["trajectory"]["distance_from_initial_squared"]
    )
    if denominator_squared <= 0.0:
        raise ValueError(
            f"non-positive default trajectory denominator: {denominator_squared}"
        )
    return {
        "pair_audit": audit,
        "default_step_1": float(default["losses"]["step_1"]),
        "workspace_step_1": float(workspace["losses"]["step_1"]),
        "step_1_loss_gap": abs(
            float(default["losses"]["step_1"])
            - float(workspace["losses"]["step_1"])
        ),
        "default_step_50": float(default["losses"]["step_50"]),
        "workspace_step_50": float(workspace["losses"]["step_50"]),
        "step_50_loss_gap": abs(
            float(default["losses"]["step_50"])
            - float(workspace["losses"]["step_50"])
        ),
        "parameter_distance_squared": numerator_squared,
        "default_distance_from_initial_squared": denominator_squared,
        "D_param": math.sqrt(numerator_squared / denominator_squared),
        "default_max_allocated_bytes": int(default["memory"]["max_allocated_bytes"]),
        "workspace_max_allocated_bytes": int(workspace["memory"]["max_allocated_bytes"]),
        "default_final_state_sha256": default["trajectory"]["final_state_sha256"],
        "workspace_final_state_sha256": workspace["trajectory"]["final_state_sha256"],
        "default_dtype_audit": default["local_fp32"]["dtype_audit"],
        "workspace_dtype_audit": workspace["local_fp32"]["dtype_audit"],
    }


def reduction(control: float, candidate: float) -> float:
    if control <= 0.0:
        raise ValueError(f"control metric must be positive, got {control}")
    if candidate < 0.0:
        raise ValueError(f"candidate metric must be non-negative, got {candidate}")
    return 1.0 - candidate / control


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Legacy A800 local FP32 region summary",
        "",
        "| Region | D_param | D reduction | Step-50 loss gap | Gap reduction | Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for region in REGIONS:
        metrics = summary["regions"][region]
        lines.append(
            f"| `{region}` | {metrics['D_param']:.12g} | "
            f"{metrics['D_param_reduction']:.6f} | "
            f"{metrics['step_50_loss_gap']:.12g} | "
            f"{metrics['step_50_loss_gap_reduction']:.6f} | "
            f"`{metrics['passes_50pct_both']}` |"
        )
    lines.extend(
        [
            "",
            f"Selected regions: `{summary['passing_regions']}`",
            "",
        ]
    )
    return "\n".join(lines)


def command_analyze(args: argparse.Namespace) -> None:
    files = sorted((args.root / "arms").glob("*/local_fp32_probe.json"))
    payloads = [load_payload(path) for path in files]
    if len(payloads) != 8:
        raise ValueError(f"expected 8 local FP32 arms, found {len(payloads)}")
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for payload in payloads:
        region = payload["condition"]["region"]
        path = payload["condition"]["path"]
        if region in matrix and path in matrix[region]:
            raise ValueError(f"duplicate local FP32 arm: region={region}, path={path}")
        matrix.setdefault(region, {})[path] = payload
    if set(matrix) != set(REGIONS):
        raise ValueError(f"unexpected regions: {sorted(matrix)}")
    for region in REGIONS:
        if set(matrix[region]) != set(PATHS):
            raise ValueError(f"region {region} lacks default/workspace pair")

    regions = {
        region: region_metrics(
            args.root,
            matrix[region]["default"],
            matrix[region]["workspace_16"],
        )
        for region in REGIONS
    }
    control_d_param = regions["none"]["D_param"]
    control_loss_gap = regions["none"]["step_50_loss_gap"]
    for region, metrics in regions.items():
        d_reduction = reduction(control_d_param, metrics["D_param"])
        loss_reduction = reduction(control_loss_gap, metrics["step_50_loss_gap"])
        metrics["D_param_reduction"] = d_reduction
        metrics["step_50_loss_gap_reduction"] = loss_reduction
        metrics["passes_50pct_both"] = bool(
            region != "none" and d_reduction >= 0.5 and loss_reduction >= 0.5
        )
    passing = [
        region for region in REGIONS if regions[region]["passes_50pct_both"]
    ]
    summary = {
        "artifact_format": "legacy_local_fp32_matrix_v1",
        "arms": len(payloads),
        "regions": regions,
        "passing_regions": passing,
    }
    temporary = args.root / "matrix_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.root / "matrix_summary.json")
    (args.root / "matrix_summary.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    print(json.dumps({"arms": len(payloads), "passing_regions": passing}, sort_keys=True))


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
