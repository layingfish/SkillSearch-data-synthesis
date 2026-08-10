#!/usr/bin/env python3
"""Analyze Phase-3 local-FP32 decoder refinements against Phase-2 controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from analyze_legacy_local_fp32_matrix import (
        load_payload,
        reduction,
        region_metrics,
    )
except ModuleNotFoundError:
    from scripts.analyze_legacy_local_fp32_matrix import (
        load_payload,
        reduction,
        region_metrics,
    )


REGIONS = (
    "decoder_blocks_0_5",
    "decoder_blocks_6_11",
    "decoder_heads_ce",
    "decoder_blocks_0_5_heads_ce",
)
PATHS = ("default", "workspace_16")
RUNTIME_CONTRACT_KEYS = (
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


def normalized_cross_job_condition(payload: dict[str, Any]) -> dict[str, Any]:
    condition = dict(payload["condition"])
    for key in ("arm", "region", "precision_gate"):
        condition.pop(key, None)
    return condition


def normalized_file_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    hashes = dict(payload["file_sha256"])
    hashes.pop("probe", None)
    return hashes


def runtime_contract(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload["runtime"]
    return {key: runtime.get(key) for key in RUNTIME_CONTRACT_KEYS}


def control_compatibility(
    candidate: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "condition": (
            normalized_cross_job_condition(candidate)
            == normalized_cross_job_condition(control)
        ),
        "pipeline_data_model_sha256": (
            normalized_file_sha256(candidate) == normalized_file_sha256(control)
        ),
        "probe_header": candidate["probe_header"] == control["probe_header"],
        "first_catalog_batch": (
            candidate["first_catalog_batch"] == control["first_catalog_batch"]
        ),
        "precision": candidate["precision"] == control["precision"],
        "dropout": candidate["dropout"] == control["dropout"],
        "runtime_contract": runtime_contract(candidate) == runtime_contract(control),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_probe_sha256": candidate["file_sha256"]["probe"],
        "control_probe_sha256": control["file_sha256"]["probe"],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Legacy A800 local FP32 refinement summary",
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


def load_control_pair(root: Path) -> dict[str, dict[str, Any]]:
    controls = {
        path: load_payload(root / "arms" / f"{path}_none" / "local_fp32_probe.json")
        for path in PATHS
    }
    for path, payload in controls.items():
        condition = payload["condition"]
        if condition["path"] != path or condition["region"] != "none":
            raise ValueError(f"invalid Phase-2 control for path={path}: {condition}")
    return controls


def command_analyze(args: argparse.Namespace) -> None:
    files = sorted((args.root / "arms").glob("*/local_fp32_probe.json"))
    payloads = [load_payload(path) for path in files]
    if len(payloads) != 8:
        raise ValueError(f"expected 8 refinement arms, found {len(payloads)}")
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for payload in payloads:
        region = payload["condition"]["region"]
        path = payload["condition"]["path"]
        if region in matrix and path in matrix[region]:
            raise ValueError(f"duplicate refinement arm: region={region}, path={path}")
        matrix.setdefault(region, {})[path] = payload
    if set(matrix) != set(REGIONS):
        raise ValueError(f"unexpected refinement regions: {sorted(matrix)}")
    for region in REGIONS:
        if set(matrix[region]) != set(PATHS):
            raise ValueError(f"region {region} lacks default/workspace pair")

    controls = load_control_pair(args.control_root)
    control_metrics = region_metrics(
        args.control_root,
        controls["default"],
        controls["workspace_16"],
    )
    regions: dict[str, dict[str, Any]] = {}
    for region in REGIONS:
        metrics = region_metrics(
            args.root,
            matrix[region]["default"],
            matrix[region]["workspace_16"],
        )
        compatibility = {
            path: control_compatibility(matrix[region][path], controls[path])
            for path in PATHS
        }
        if not all(audit["passed"] for audit in compatibility.values()):
            raise ValueError(
                f"Phase-2 control compatibility failed for region={region}: {compatibility}"
            )
        d_reduction = reduction(control_metrics["D_param"], metrics["D_param"])
        loss_reduction = reduction(
            control_metrics["step_50_loss_gap"],
            metrics["step_50_loss_gap"],
        )
        metrics["control_compatibility"] = compatibility
        metrics["D_param_reduction"] = d_reduction
        metrics["step_50_loss_gap_reduction"] = loss_reduction
        metrics["passes_50pct_both"] = bool(
            d_reduction >= 0.5 and loss_reduction >= 0.5
        )
        regions[region] = metrics

    passing = [region for region in REGIONS if regions[region]["passes_50pct_both"]]
    summary = {
        "artifact_format": "legacy_local_fp32_refinement_v1",
        "arms": len(payloads),
        "control_root": str(args.control_root.resolve()),
        "control": {
            "D_param": control_metrics["D_param"],
            "step_50_loss_gap": control_metrics["step_50_loss_gap"],
            "pair_audit": control_metrics["pair_audit"],
        },
        "regions": regions,
        "passing_regions": passing,
    }
    temporary = args.root / "refinement_summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.root / "refinement_summary.json")
    (args.root / "refinement_summary.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    print(json.dumps({"arms": len(payloads), "passing_regions": passing}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
