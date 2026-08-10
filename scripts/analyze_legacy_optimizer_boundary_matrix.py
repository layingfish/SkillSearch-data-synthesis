#!/usr/bin/env python3
"""Analyze the A800 catalog-to-query optimizer carry/reset factorial."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from analyze_legacy_local_fp32_matrix import reduction
except ModuleNotFoundError:
    from scripts.analyze_legacy_local_fp32_matrix import reduction


PATHS = ("default", "workspace_16")
OPTIMIZER_MODES = ("carry", "reset")
QUERY_LRS = (0.0002, 0.0003)


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_format") != "legacy_optimizer_boundary_arm_v1":
        raise ValueError(f"unexpected optimizer-boundary artifact: {path}")
    return payload


def lr_key(value: float) -> str:
    return f"{float(value):.10g}"


def normalized_pair_condition(payload: dict[str, Any]) -> dict[str, Any]:
    condition = dict(payload["condition"])
    condition.pop("arm", None)
    condition.pop("path", None)
    controls = dict(condition["requested_controls"])
    controls.pop("cublas_workspace_config", None)
    condition["requested_controls"] = controls
    return condition


def stable_runtime_contract(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload["runtime"]
    keys = (
        "compute_capability",
        "cuda_runtime",
        "cudnn",
        "gpu",
        "gpu_total_memory",
        "huggingface_hub",
        "nvidia_cublas_cu12",
        "nvidia_cuda_nvrtc_cu12",
        "nvidia_cuda_runtime_cu12",
        "nvidia_cudnn_cu12",
        "platform",
        "python",
        "safetensors",
        "tokenizers",
        "torch",
        "torch_config",
        "torch_file_sha256",
        "transformers",
    )
    return {key: runtime.get(key) for key in keys}


def pair_audit(default: dict[str, Any], workspace: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "condition": normalized_pair_condition(default) == normalized_pair_condition(workspace),
        "file_sha256": default["file_sha256"] == workspace["file_sha256"],
        "initial": default["initial"] == workspace["initial"],
        "batches": default["batches"] == workspace["batches"],
        "dropout": default["dropout"] == workspace["dropout"],
        "runtime_contract": stable_runtime_contract(default) == stable_runtime_contract(workspace),
    }
    return {"passed": all(checks.values()), "checks": checks}


def catalog_invariant(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("catalog invariant requires payloads")
    fields = {
        "initial": [payload["initial"] for payload in payloads],
        "file_sha256": [payload["file_sha256"] for payload in payloads],
        "model_state_sha256": [
            payload["catalog_boundary"]["model_state_sha256"] for payload in payloads
        ],
        "trainable_state_sha256": [
            payload["catalog_boundary"]["trainable_state_sha256"] for payload in payloads
        ],
        "optimizer_state_sha256": [
            payload["catalog_boundary"]["optimizer_state_sha256"] for payload in payloads
        ],
        "optimizer_state_entries": [
            payload["catalog_boundary"]["optimizer_state_entries"] for payload in payloads
        ],
        "optimizer_param_groups": [
            payload["catalog_boundary"]["optimizer_param_groups"] for payload in payloads
        ],
        "cpu_rng_sha256": [
            payload["catalog_boundary"]["cpu_rng_sha256"] for payload in payloads
        ],
        "cuda_rng_sha256": [
            payload["catalog_boundary"]["cuda_rng_sha256"] for payload in payloads
        ],
        "catalog_losses": [
            {
                key: value
                for key, value in payload["losses"].items()
                if key.startswith("catalog_")
            }
            for payload in payloads
        ],
        "query_step_1_loss": [payload["losses"]["query_step_1"] for payload in payloads],
        "query_step_1_gradient": [
            payload["query_step_1"]["gradient"] for payload in payloads
        ],
        "batches": [payload["batches"] for payload in payloads],
    }
    checks = {name: all(value == values[0] for value in values) for name, values in fields.items()}
    return {"passed": all(checks.values()), "checks": checks}


def resolve_state(root: Path, payload: dict[str, Any], stage: str) -> Path:
    arm = payload["condition"]["arm"]
    arm_root = (root / "arms" / arm).resolve()
    if stage == "catalog":
        artifact = payload["catalog_boundary"]["trainable_state_artifact"]
    elif stage == "query":
        artifact = payload["query_step_50"]["trainable_state_artifact"]
    else:
        raise ValueError(f"unsupported state stage: {stage}")
    path = (arm_root / artifact).resolve()
    if not path.is_relative_to(arm_root):
        raise ValueError(f"state artifact escapes arm directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def query_update_distance(
    root: Path,
    default: dict[str, Any],
    workspace: dict[str, Any],
) -> dict[str, float]:
    import torch

    default_catalog = torch.load(
        resolve_state(root, default, "catalog"), map_location="cpu", weights_only=True
    )
    default_query = torch.load(
        resolve_state(root, default, "query"), map_location="cpu", weights_only=True
    )
    workspace_catalog = torch.load(
        resolve_state(root, workspace, "catalog"), map_location="cpu", weights_only=True
    )
    workspace_query = torch.load(
        resolve_state(root, workspace, "query"), map_location="cpu", weights_only=True
    )
    names = set(default_catalog)
    if not (
        names == set(default_query) == set(workspace_catalog) == set(workspace_query)
    ):
        raise ValueError("optimizer-boundary trainable parameter names differ")
    numerator_squared = 0.0
    denominator_squared = 0.0
    workspace_update_squared = 0.0
    for name in sorted(names):
        default_delta = default_query[name].to(torch.float64) - default_catalog[name].to(
            torch.float64
        )
        workspace_delta = workspace_query[name].to(
            torch.float64
        ) - workspace_catalog[name].to(torch.float64)
        numerator_squared += float(
            ((default_delta - workspace_delta) ** 2).sum().item()
        )
        denominator_squared += float((default_delta**2).sum().item())
        workspace_update_squared += float((workspace_delta**2).sum().item())
    del default_catalog, default_query, workspace_catalog, workspace_query
    gc.collect()
    if denominator_squared <= 0.0:
        raise ValueError(f"non-positive default query-update denominator: {denominator_squared}")
    return {
        "update_difference_squared": numerator_squared,
        "default_update_squared": denominator_squared,
        "workspace_update_squared": workspace_update_squared,
        "D_update": math.sqrt(numerator_squared / denominator_squared),
    }


def condition_metrics(
    root: Path,
    default: dict[str, Any],
    workspace: dict[str, Any],
) -> dict[str, Any]:
    audit = pair_audit(default, workspace)
    if not audit["passed"]:
        raise ValueError(f"optimizer-boundary pair audit failed: {audit}")
    distances = query_update_distance(root, default, workspace)
    return {
        **distances,
        "pair_audit": audit,
        "default_query_step_1_loss": default["losses"]["query_step_1"],
        "workspace_query_step_1_loss": workspace["losses"]["query_step_1"],
        "query_step_1_loss_gap": abs(
            default["losses"]["query_step_1"] - workspace["losses"]["query_step_1"]
        ),
        "default_query_step_50_loss": default["losses"]["query_step_50"],
        "workspace_query_step_50_loss": workspace["losses"]["query_step_50"],
        "query_step_50_loss_gap": abs(
            default["losses"]["query_step_50"] - workspace["losses"]["query_step_50"]
        ),
        "default_query_step_1_trace": default["query_step_1"],
        "workspace_query_step_1_trace": workspace["query_step_1"],
        "default_max_allocated_bytes": default["memory"]["max_allocated_bytes"],
        "workspace_max_allocated_bytes": workspace["memory"]["max_allocated_bytes"],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Legacy A800 optimizer-boundary summary",
        "",
        "| Query lr | Optimizer | D_update | D reduction vs carry | Step-50 loss gap | Gap reduction vs carry | Pass |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for query_lr in QUERY_LRS:
        for mode in OPTIMIZER_MODES:
            metrics = summary["conditions"][lr_key(query_lr)][mode]
            lines.append(
                f"| {query_lr:.1e} | `{mode}` | {metrics['D_update']:.12g} | "
                f"{metrics['D_update_reduction_vs_carry']:.6f} | "
                f"{metrics['query_step_50_loss_gap']:.12g} | "
                f"{metrics['loss_gap_reduction_vs_carry']:.6f} | "
                f"`{metrics['passes_50pct_both']}` |"
            )
    lines.extend(
        [
            "",
            f"Passing reset conditions: `{summary['passing_reset_conditions']}`",
            "",
        ]
    )
    return "\n".join(lines)


def command_analyze(args: argparse.Namespace) -> None:
    files = sorted((args.root / "arms").glob("*/optimizer_boundary_probe.json"))
    payloads = [load_payload(path) for path in files]
    if len(payloads) != 8:
        raise ValueError(f"expected 8 optimizer-boundary arms, found {len(payloads)}")

    matrix: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for payload in payloads:
        condition = payload["condition"]
        query_lr = lr_key(condition["query_lr"])
        mode = condition["optimizer_mode"]
        path = condition["path"]
        if path in matrix.get(query_lr, {}).get(mode, {}):
            raise ValueError(
                f"duplicate optimizer arm: lr={query_lr}, mode={mode}, path={path}"
            )
        matrix.setdefault(query_lr, {}).setdefault(mode, {})[path] = payload
    if set(matrix) != {lr_key(value) for value in QUERY_LRS}:
        raise ValueError(f"unexpected query lrs: {sorted(matrix)}")
    for query_lr in QUERY_LRS:
        key = lr_key(query_lr)
        if set(matrix[key]) != set(OPTIMIZER_MODES):
            raise ValueError(f"query lr {key} lacks optimizer modes")
        for mode in OPTIMIZER_MODES:
            if set(matrix[key][mode]) != set(PATHS):
                raise ValueError(f"query lr {key}, mode {mode} lacks backend pair")

    catalog_invariants = {
        path: catalog_invariant(
            [
                matrix[lr_key(query_lr)][mode][path]
                for query_lr in QUERY_LRS
                for mode in OPTIMIZER_MODES
            ]
        )
        for path in PATHS
    }
    if not all(audit["passed"] for audit in catalog_invariants.values()):
        raise ValueError(f"catalog-final invariant failed: {catalog_invariants}")

    conditions: dict[str, dict[str, dict[str, Any]]] = {}
    passing: list[dict[str, Any]] = []
    for query_lr in QUERY_LRS:
        key = lr_key(query_lr)
        conditions[key] = {}
        for mode in OPTIMIZER_MODES:
            conditions[key][mode] = condition_metrics(
                args.root,
                matrix[key][mode]["default"],
                matrix[key][mode]["workspace_16"],
            )
        carry = conditions[key]["carry"]
        for mode in OPTIMIZER_MODES:
            metrics = conditions[key][mode]
            d_reduction = reduction(carry["D_update"], metrics["D_update"])
            loss_reduction = reduction(
                carry["query_step_50_loss_gap"],
                metrics["query_step_50_loss_gap"],
            )
            metrics["D_update_reduction_vs_carry"] = d_reduction
            metrics["loss_gap_reduction_vs_carry"] = loss_reduction
            metrics["passes_50pct_both"] = bool(
                mode == "reset" and d_reduction >= 0.5 and loss_reduction >= 0.5
            )
            if metrics["passes_50pct_both"]:
                passing.append({"query_lr": query_lr, "optimizer_mode": mode})

    summary = {
        "artifact_format": "legacy_optimizer_boundary_matrix_v1",
        "arms": len(payloads),
        "catalog_invariants": catalog_invariants,
        "conditions": conditions,
        "passing_reset_conditions": passing,
    }
    temporary = args.root / "optimizer_boundary_summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.root / "optimizer_boundary_summary.json")
    (args.root / "optimizer_boundary_summary.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    print(json.dumps({"arms": len(payloads), "passing": passing}, sort_keys=True))


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
