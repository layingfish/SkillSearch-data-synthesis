#!/usr/bin/env python3
"""Audit and select the strict-base legacy CE-prefix ranking sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from analyze_legacy_full_query_optimizer_lr import (
        command_arm,
        read_json,
        recalls,
    )
except ModuleNotFoundError:
    from scripts.analyze_legacy_full_query_optimizer_lr import (
        command_arm,
        read_json,
        recalls,
    )


ARM_SPECS = (
    ("control_a", "anchor", "disabled", 0.0),
    ("control_b", "replica", "disabled", 0.0),
    ("cum_002", "selection", "cumulative", 0.02),
    ("cum_005", "selection", "cumulative", 0.05),
    ("cum_010", "selection", "cumulative", 0.10),
    ("cum_020", "selection", "cumulative", 0.20),
    ("c1_010", "selection", "c1_only", 0.10),
    ("seq_010", "selection", "sequence_only", 0.10),
)
CONTROL_LABELS = ("control_a", "control_b")
SELECTION_LABELS = tuple(label for label, role, _, _ in ARM_SPECS if role == "selection")
RTX_R1_ANCHORS = {
    "overall": 0.28368253968253965,
    "single": 10 / 24,
    "multi": 0.22110177404295053,
}
ABSOLUTE_THRESHOLDS = {
    "overall_r1": RTX_R1_ANCHORS["overall"] - 0.01,
    "overall_r10": 0.69,
    "overall_r50": 0.75,
    # A 24-query subset moves in 1/24 increments, so a one-point tolerance
    # still requires matching the RTX hit count exactly.
    "single_r1": RTX_R1_ANCHORS["single"],
    "multi_r1": RTX_R1_ANCHORS["multi"] - 0.01,
}
DEEP_RECALL_MARGIN = 0.01
EXPECTED_SUMMARY_SHA256 = "72b0f525d6c3d9feabc7770c054ee70d8478f1b280d1a100793da8d5b6c6dd6d"
EXPECTED_MANIFEST_SHA256 = "8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d"
HISTORICAL_CONTROL = {
    "best_step": 38200,
    "best_dev_loss": 0.7701343703584546,
    "final_model_full_sha256": "bbec7389f56b36a24db000111adf8ce99e3bf712609cfcd1562c64c9910172f8",
    "predictions_sha256": "3663a15ad9859abc76141d815f0da3c88bd9b3d5d9e914f09a3d8b0a3ed843f4",
    "recalls": {
        "overall": {
            "@1": 0.20923809523809525,
            "@10": 0.6955555555555555,
            "@50": 0.7533333333333333,
        },
        "single": {"@1": 0.25, "@10": 0.5, "@50": 0.5},
        "multi": {
            "@1": 0.19005602240896358,
            "@10": 0.7875816993464052,
            "@50": 0.8725490196078431,
        },
    },
}


def control_invariant(
    summaries: dict[str, dict[str, Any]], audits: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    left = summaries[CONTROL_LABELS[0]]
    right = summaries[CONTROL_LABELS[1]]
    checks = {
        "final_model_state": left["training"]["final_model_state_sha256"]
        == right["training"]["final_model_state_sha256"],
        "best_step": left["training"]["query_best_step"]
        == right["training"]["query_best_step"],
        "best_dev_loss": left["training"]["query_best_dev_loss"]
        == right["training"]["query_best_dev_loss"],
        "predictions": left["evaluation"]["predictions_sha256"]
        == right["evaluation"]["predictions_sha256"],
        "metrics": recalls(left) == recalls(right),
        "catalog_boundary": audits[CONTROL_LABELS[0]]["catalog_boundary"]
        == audits[CONTROL_LABELS[1]]["catalog_boundary"],
        "query_step_1_gradient": audits[CONTROL_LABELS[0]]["query_step_1"]
        == audits[CONTROL_LABELS[1]]["query_step_1"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def historical_control_invariant(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "best_step": int(summary["training"]["query_best_step"])
        == HISTORICAL_CONTROL["best_step"],
        "best_dev_loss": float(summary["training"]["query_best_dev_loss"])
        == HISTORICAL_CONTROL["best_dev_loss"],
        "final_model_full_sha256": summary["training"]["final_model_state_sha256"][
            "full"
        ]
        == HISTORICAL_CONTROL["final_model_full_sha256"],
        "predictions_sha256": summary["evaluation"]["predictions_sha256"]
        == HISTORICAL_CONTROL["predictions_sha256"],
        "recalls": recalls(summary) == HISTORICAL_CONTROL["recalls"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def selection_gate(
    summary: dict[str, Any], control_recalls: dict[str, dict[str, float]]
) -> dict[str, Any]:
    values = recalls(summary)
    checks = {
        "overall_r1": values["overall"]["@1"] >= ABSOLUTE_THRESHOLDS["overall_r1"],
        "overall_r10": values["overall"]["@10"]
        >= ABSOLUTE_THRESHOLDS["overall_r10"],
        "overall_r50": values["overall"]["@50"]
        >= ABSOLUTE_THRESHOLDS["overall_r50"],
        "single_r1": values["single"]["@1"] >= ABSOLUTE_THRESHOLDS["single_r1"],
        "multi_r1": values["multi"]["@1"] >= ABSOLUTE_THRESHOLDS["multi_r1"],
    }
    for subset in ("single", "multi"):
        for cutoff in ("@10", "@50"):
            checks[f"{subset}_r{cutoff[1:]}_control_noninferiority"] = (
                values[subset][cutoff]
                >= control_recalls[subset][cutoff] - DEEP_RECALL_MARGIN
            )
    return {"checks": checks, "passed": all(checks.values())}


def ranking_key(summary: dict[str, Any]) -> tuple[float, ...]:
    values = recalls(summary)
    return (
        values["overall"]["@1"],
        values["multi"]["@1"],
        values["single"]["@1"],
        values["overall"]["@10"],
        -float(summary["training"]["query_best_dev_loss"]),
    )


def choose_qualified_arm(
    summaries: dict[str, dict[str, Any]], gates: dict[str, dict[str, Any]]
) -> tuple[list[str], str | None, str]:
    passing = [label for label in SELECTION_LABELS if gates[label]["passed"]]
    if not passing:
        return [], None, "no_qualified_arm"
    selected = max(passing, key=lambda label: ranking_key(summaries[label]))
    return passing, selected, "qualified_only"


def catalog_invariant(audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "runtime",
        "precision",
        "configured_precision",
        "initial",
        "batches",
        "catalog_boundary",
        "dropout",
    )
    reference = audits[CONTROL_LABELS[0]]
    checks = {
        field: all(audit[field] == reference[field] for audit in audits.values())
        for field in fields
    }
    checks["catalog_losses"] = all(
        {
            key: value
            for key, value in audit["losses"].items()
            if key.startswith("catalog_step_")
        }
        == {
            key: value
            for key, value in reference["losses"].items()
            if key.startswith("catalog_step_")
        }
        for audit in audits.values()
    )
    return {"checks": checks, "passed": all(checks.values())}


def command_matrix(args: argparse.Namespace) -> None:
    summaries: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for label, role, scope, weight in ARM_SPECS:
        arm_root = args.root / "arms" / label
        if not (arm_root / "arm.done").is_file():
            raise FileNotFoundError(f"arm completion marker is missing: {label}")
        summary = read_json(arm_root / "result_summary.json")
        audit = read_json(arm_root / "full_query_arm.json")
        condition = audit["condition"]
        checks = {
            "label": condition["arm"] == label,
            "role": condition["role"] == role,
            "workspace_path": condition["path"] == "workspace_16",
            "optimizer_carry": condition["optimizer_mode"] == "carry",
            "query_lr": float(condition["query_lr"]) == 3e-4,
            "catalog_steps": int(condition["catalog_steps"]) == 2730,
            "query_steps": int(condition["query_steps"]) == 38805,
            "batch_size": int(condition["batch_size"]) == 16,
            "seed": int(condition["seed"]) == 20260626,
            "dropout": float(condition["dropout_rate"]) == 0.1,
            "workspace": condition["requested_controls"]["cublas_workspace_config"]
            == ":16:8",
            "scope": condition["prefix_ranking"]["scope"] == scope,
            "weight": float(condition["prefix_ranking"]["weight"]) == weight,
            "temperature": float(condition["prefix_ranking"]["temperature"]) == 1.0,
            "summary_sha256": condition["prefix_ranking"]["summary_sha256"]
            == EXPECTED_SUMMARY_SHA256,
            "manifest_sha256": condition["prefix_ranking"]["manifest_sha256"]
            == EXPECTED_MANIFEST_SHA256,
        }
        if not all(checks.values()):
            raise ValueError(f"arm condition mismatch for {label}: {checks}")
        execution = audit["prefix_ranking"]["execution"]
        expected_pairs = 0 if weight == 0.0 else 431525
        expected_comparisons = (
            0 if weight == 0.0 else expected_pairs * (3 if scope == "cumulative" else 1)
        )
        if (
            int(execution["eligible_pair_presentations"]) != expected_pairs
            or int(execution["active_comparisons"]) != expected_comparisons
        ):
            raise ValueError(f"prefix ranking count mismatch for {label}: {execution}")
        if weight > 0.0:
            parity = audit["prefix_ranking"]["trace"].get("score_parity", {})
            if not parity or not all(row.get("passed") for row in parity.values()):
                raise ValueError(f"score parity failed for {label}: {parity}")
        summaries[label] = summary
        audits[label] = audit

    catalog = catalog_invariant(audits)
    controls = control_invariant(summaries, audits)
    historical = {
        label: historical_control_invariant(summaries[label])
        for label in CONTROL_LABELS
    }
    if not catalog["passed"]:
        raise ValueError(f"catalog invariant failed: {catalog}")
    if not controls["passed"]:
        raise ValueError(f"control replica invariant failed: {controls}")
    if not all(row["passed"] for row in historical.values()):
        raise ValueError(f"historical control invariant failed: {historical}")

    control_recalls = recalls(summaries[CONTROL_LABELS[0]])
    gates = {
        label: selection_gate(summaries[label], control_recalls)
        for label in SELECTION_LABELS
    }
    passing, selected, selection_basis = choose_qualified_arm(summaries, gates)
    payload = {
        "artifact_format": "legacy_ce_prefix_ranking_matrix_v1",
        "protocol": {
            "source_rows": 94912,
            "expanded_query_rows": 124175,
            "catalog_steps": 2730,
            "query_steps": 38805,
            "batch_size": 16,
            "seed": 20260626,
            "cublas_workspace_config": ":16:8",
            "checkpoint": "dev_best",
            "evaluation": "Official core75=75 queries, beam=50, beam_order, CE log-prob",
        },
        "absolute_thresholds": ABSOLUTE_THRESHOLDS,
        "deep_recall_control_margin": DEEP_RECALL_MARGIN,
        "catalog_invariant": catalog,
        "control_replica_invariant": controls,
        "historical_control_invariants": historical,
        "selection": {
            "gates": gates,
            "passing_arms": passing,
            "selected_arm": selected,
            "selection_basis": selection_basis,
        },
        "arms": [summaries[label] for label, _, _, _ in ARM_SPECS],
    }
    output = args.root / "ce_prefix_ranking_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "passing_arms": passing,
                "selected_arm": selected,
                "selection_basis": selection_basis,
            },
            sort_keys=True,
        )
    )
    if selected is None:
        raise SystemExit(3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm = subparsers.add_parser("arm")
    arm.add_argument("--arm-root", type=Path, required=True)
    arm.add_argument("--archived-official", type=Path, required=True)
    arm.add_argument("--evaluator", type=Path, required=True)
    arm.set_defaults(func=command_arm)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--root", type=Path, required=True)
    matrix.set_defaults(func=command_matrix)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
