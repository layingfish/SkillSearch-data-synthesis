#!/usr/bin/env python3
"""Summarize and audit the full-query optimizer/lr sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ARM_SPECS = (
    ("workspace_16_carry_lr1e4", "selection", "workspace_16", "carry", 1e-4),
    ("workspace_16_reset_lr1e4", "selection", "workspace_16", "reset", 1e-4),
    ("workspace_16_carry_lr2e4", "selection", "workspace_16", "carry", 2e-4),
    ("workspace_16_reset_lr2e4", "selection", "workspace_16", "reset", 2e-4),
    ("workspace_16_carry_lr3e4", "selection", "workspace_16", "carry", 3e-4),
    ("workspace_16_reset_lr3e4", "selection", "workspace_16", "reset", 3e-4),
    ("workspace_16_carry_lr3e4_replica", "replica", "workspace_16", "carry", 3e-4),
    ("default_carry_lr3e4_anchor", "anchor", "default", "carry", 3e-4),
)

GATE_THRESHOLDS = {
    "overall_r1": 0.273683,
    "overall_r10": 0.69,
    "overall_r50": 0.75,
    "single_r1": 0.416667,
    "multi_r1": 0.211102,
}

HISTORICAL_REFERENCES = {
    "workspace_16_carry_lr3e4": {
        "best_step": 38200,
        "best_dev_loss": 0.7701343703584546,
        "recalls": {
            "overall": {"@1": 0.20923809523809525, "@10": 0.6955555555555555, "@50": 0.7533333333333333},
            "single": {"@1": 0.25, "@10": 0.5, "@50": 0.5},
            "multi": {"@1": 0.19005602240896358, "@10": 0.7875816993464052, "@50": 0.8725490196078431},
        },
    },
    "default_carry_lr3e4_anchor": {
        "best_step": 38800,
        "best_dev_loss": 0.8704000392564532,
        "recalls": {
            "overall": {"@1": 0.12723809523809523, "@10": 0.49542857142857133, "@50": 0.6588888888888889},
            "single": {"@1": 0.0, "@10": 0.20833333333333334, "@50": 0.4583333333333333},
            "multi": {"@1": 0.18711484593837535, "@10": 0.630532212885154, "@50": 0.7532679738562092},
        },
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recalls(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        subset: {
            cutoff: float(summary[subset][cutoff]["recall"])
            for cutoff in ("@1", "@10", "@50")
        }
        for subset in ("overall", "single", "multi")
    }


def selection_gate(summary: dict[str, Any]) -> dict[str, Any]:
    metric_values = recalls(summary)
    checks = {
        "overall_r1": metric_values["overall"]["@1"] >= GATE_THRESHOLDS["overall_r1"],
        "overall_r10": metric_values["overall"]["@10"] >= GATE_THRESHOLDS["overall_r10"],
        "overall_r50": metric_values["overall"]["@50"] >= GATE_THRESHOLDS["overall_r50"],
        "single_r1": metric_values["single"]["@1"] >= GATE_THRESHOLDS["single_r1"],
        "multi_r1": metric_values["multi"]["@1"] >= GATE_THRESHOLDS["multi_r1"],
    }
    return {
        "checks": checks,
        "deep_recall_passed": checks["overall_r10"] and checks["overall_r50"],
        "passed": all(checks.values()),
    }


def ranking_key(summary: dict[str, Any]) -> tuple[float, ...]:
    metric_values = recalls(summary)
    return (
        metric_values["overall"]["@1"],
        metric_values["single"]["@1"],
        metric_values["multi"]["@1"],
        metric_values["overall"]["@10"],
        -float(summary["training"]["query_best_dev_loss"]),
    )


def invariant_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime": payload["runtime"],
        "precision": payload["precision"],
        "configured_precision": payload["configured_precision"],
        "file_sha256": payload["file_sha256"],
        "initial": payload["initial"],
        "batches": payload["batches"],
        "catalog_boundary": payload["catalog_boundary"],
        "catalog_losses": {
            key: value
            for key, value in payload["losses"].items()
            if key.startswith("catalog_step_")
        },
        "query_step_1_loss": payload["losses"]["query_step_1"],
        "query_step_1_gradient": payload["query_step_1"]["gradient"],
        "dropout": payload["dropout"],
    }


def audit_workspace_catalog(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    projections = [invariant_projection(payload) for payload in payloads]
    reference = projections[0]
    checks = {
        key: all(projection[key] == reference[key] for projection in projections[1:])
        for key in reference
    }
    return {"arms": len(payloads), "checks": checks, "passed": all(checks.values())}


def historical_check(summary: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    observed_recalls = recalls(summary)
    checks = {
        "best_step": int(summary["training"]["query_best_step"]) == expected["best_step"],
        "best_dev_loss": float(summary["training"]["query_best_dev_loss"])
        == expected["best_dev_loss"],
        "recalls": observed_recalls == expected["recalls"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def command_arm(args: argparse.Namespace) -> None:
    arm_root = args.arm_root
    audit = read_json(arm_root / "full_query_arm.json")
    condition = audit["condition"]
    if condition["arm"] != arm_root.name:
        raise ValueError(
            f"arm label mismatch: directory={arm_root.name}, audit={condition['arm']}"
        )
    evaluation = arm_root / "eval" / "official_core75"
    predictions_path = evaluation / "predictions.jsonl"
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    archived = [
        json.loads(line)
        for line in args.archived_official.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(predictions) != 75 or len(archived) != 75:
        raise ValueError(
            f"Official core75 row mismatch: predictions={len(predictions)}, archived={len(archived)}"
        )
    metrics = {
        subset: read_json(evaluation / f"ranked_metrics_{metric_subset}.json")["overall"]
        for subset, metric_subset in (
            ("overall", "all"),
            ("single", "single"),
            ("multi", "multi"),
        )
    }
    exact_rows = sum(
        left.get("ranked_skill_ids") == right.get("ranked_skill_ids")
        for left, right in zip(archived, predictions)
    )
    history = audit["training"]["history"]
    summary = {
        "artifact_format": "legacy_full_query_optimizer_lr_result_v1",
        "arm": condition["arm"],
        "role": condition["role"],
        "condition": condition,
        "training": {
            "query_best_step": history["best_step"],
            "query_best_dev_loss": history["best_dev_loss"],
            "final_model_state_sha256": audit["checkpoints"]["final"][
                "model_state_sha256"
            ],
        },
        "evaluation": {
            "protocol": "Official core75, 75 queries, beam=50, beam_order, CE log-prob",
            "predictions_sha256": file_sha256(predictions_path),
            "evaluator_sha256": file_sha256(args.evaluator),
            "archived_top50_exact_rows": exact_rows,
        },
        **metrics,
    }
    output = arm_root / "result_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps({"arm": condition["arm"], "recalls": recalls(summary)}, sort_keys=True))


def command_matrix(args: argparse.Namespace) -> None:
    root = args.root
    expected = {label: (role, path, mode, lr) for label, role, path, mode, lr in ARM_SPECS}
    summaries: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for label, role, path, mode, lr in ARM_SPECS:
        arm_root = root / "arms" / label
        if not (arm_root / "arm.done").is_file():
            raise FileNotFoundError(f"arm completion marker is missing: {label}")
        summary = read_json(arm_root / "result_summary.json")
        audit = read_json(arm_root / "full_query_arm.json")
        observed = audit["condition"]
        condition_checks = {
            "label": observed["arm"] == label,
            "role": observed["role"] == role,
            "path": observed["path"] == path,
            "optimizer_mode": observed["optimizer_mode"] == mode,
            "query_lr": float(observed["query_lr"]) == lr,
            "catalog_steps": observed["catalog_steps"] == 2730,
            "query_steps": observed["query_steps"] == 38805,
            "eval_steps": observed["eval_steps"] == 200,
            "batch_size": observed["batch_size"] == 16,
            "seed": observed["seed"] == 20260626,
            "catalog_lr": float(observed["catalog_lr"]) == 3e-4,
            "dropout_rate": float(observed["dropout_rate"]) == 0.1,
        }
        if not all(condition_checks.values()):
            raise ValueError(f"arm condition mismatch for {label}: {condition_checks}")
        summaries[label] = summary
        audits[label] = audit

    workspace_labels = [label for label, _, path, _, _ in ARM_SPECS if path == "workspace_16"]
    catalog_audit = audit_workspace_catalog([audits[label] for label in workspace_labels])
    if not catalog_audit["passed"]:
        raise ValueError(f"workspace catalog invariant failed: {catalog_audit}")

    base_label = "workspace_16_carry_lr3e4"
    replica_label = "workspace_16_carry_lr3e4_replica"
    base = summaries[base_label]
    replica = summaries[replica_label]
    replica_checks = {
        "final_model_state_sha256": base["training"]["final_model_state_sha256"]
        == replica["training"]["final_model_state_sha256"],
        "best_step": base["training"]["query_best_step"]
        == replica["training"]["query_best_step"],
        "best_dev_loss": base["training"]["query_best_dev_loss"]
        == replica["training"]["query_best_dev_loss"],
        "predictions_sha256": base["evaluation"]["predictions_sha256"]
        == replica["evaluation"]["predictions_sha256"],
        "metrics": recalls(base) == recalls(replica),
    }
    replica_audit = {"checks": replica_checks, "passed": all(replica_checks.values())}
    if not replica_audit["passed"]:
        raise ValueError(f"workspace exact-replica invariant failed: {replica_audit}")

    historical = {
        label: historical_check(summaries[label], reference)
        for label, reference in HISTORICAL_REFERENCES.items()
    }
    if not all(row["passed"] for row in historical.values()):
        raise ValueError(f"historical anchor mismatch: {historical}")

    selection_labels = [
        label for label, role, _, _, _ in ARM_SPECS if role == "selection"
    ]
    gates = {label: selection_gate(summaries[label]) for label in selection_labels}
    passing = [label for label in selection_labels if gates[label]["passed"]]
    deep_recall = [
        label for label in selection_labels if gates[label]["deep_recall_passed"]
    ]
    candidate_pool = passing or deep_recall or selection_labels
    selected = max(candidate_pool, key=lambda label: ranking_key(summaries[label]))
    selection_basis = (
        "all_gates" if passing else "deep_recall_then_top1" if deep_recall else "top1_fallback"
    )

    summary_payload = {
        "artifact_format": "legacy_full_query_optimizer_lr_matrix_v1",
        "protocol": {
            "catalog_steps": 2730,
            "query_steps": 38805,
            "eval_steps": 200,
            "batch_size": 16,
            "seed": 20260626,
            "catalog_lr": 3e-4,
            "dropout_rate": 0.1,
            "checkpoint": "dev_best",
            "evaluation": "Official core75, 75 queries, beam=50, beam_order, CE log-prob",
        },
        "gate_thresholds": GATE_THRESHOLDS,
        "workspace_catalog_invariant": catalog_audit,
        "replica_invariant": replica_audit,
        "historical_anchor_invariant": historical,
        "selection": {
            "arms": selection_labels,
            "gates": gates,
            "passing_arms": passing,
            "deep_recall_arms": deep_recall,
            "selected_arm": selected,
            "selection_basis": selection_basis,
        },
        "arms": [summaries[label] for label in expected],
    }
    output = root / "full_query_sweep_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)

    lines = [
        "# Legacy A800 full-query optimizer/lr sweep",
        "",
        "Conditions: workspace_16 selection arms; catalog=2730 updates; query=38805 updates; "
        "batch=16; seed=20260626; catalog lr=3e-4; dropout=0.1; Official core75=75 queries; "
        "beam=50; beam-order.",
        "",
        "| arm | role | Overall R@1/10/50 | Single R@1 | Multi R@1 | best step | best dev loss | gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, role, _, _, _ in ARM_SPECS:
        arm = summaries[label]
        metric = recalls(arm)
        gate = gates[label]["passed"] if role == "selection" else "audit"
        lines.append(
            f"| `{label}` | {role} | "
            f"{100 * metric['overall']['@1']:.4f}/{100 * metric['overall']['@10']:.4f}/{100 * metric['overall']['@50']:.4f} | "
            f"{100 * metric['single']['@1']:.4f} | {100 * metric['multi']['@1']:.4f} | "
            f"{arm['training']['query_best_step']} | {arm['training']['query_best_dev_loss']:.9f} | {gate} |"
        )
    lines.extend(
        [
            "",
            f"Selected arm: `{selected}` ({selection_basis}).",
            f"Passing arms: {passing}.",
        ]
    )
    (root / "full_query_sweep_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_arm": selected,
                "selection_basis": selection_basis,
                "passing_arms": passing,
            },
            sort_keys=True,
        )
    )


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
