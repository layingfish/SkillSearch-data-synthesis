#!/usr/bin/env python3
"""Summarize and audit the full-query warmup/clipping ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from analyze_legacy_full_query_optimizer_lr import (
        HISTORICAL_REFERENCES,
        command_arm,
        historical_check,
        invariant_projection,
        ranking_key,
        read_json,
        recalls,
        selection_gate,
    )
except ModuleNotFoundError:
    from scripts.analyze_legacy_full_query_optimizer_lr import (
        HISTORICAL_REFERENCES,
        command_arm,
        historical_check,
        invariant_projection,
        ranking_key,
        read_json,
        recalls,
        selection_gate,
    )


ARM_SPECS = (
    ("default_control", "anchor", "default", "control"),
    ("workspace_16_control", "anchor", "workspace_16", "control"),
    ("default_warmup500", "pair", "default", "warmup"),
    ("workspace_16_warmup500", "selection", "workspace_16", "warmup"),
    ("default_clip1", "pair", "default", "clip"),
    ("workspace_16_clip1", "selection", "workspace_16", "clip"),
    ("workspace_16_warmup500_replica", "replica", "workspace_16", "warmup"),
    ("workspace_16_clip1_replica", "replica", "workspace_16", "clip"),
)

SELECTION_LABELS = ("workspace_16_warmup500", "workspace_16_clip1")
HISTORICAL_LABELS = {
    "default_control": "default_carry_lr3e4_anchor",
    "workspace_16_control": "workspace_16_carry_lr3e4",
}
HISTORICAL_ARTIFACTS = {
    "default_control": {
        "final_model_full_sha256": "ac722408becb624540f07c2aa8960ec146ffaa298aad9d6266f3834999d1a3e1",
        "predictions_sha256": "54b6c131c36703c43ca3210bed5aab4b5f0c5d6abed1221a1d19e4ee567c4daa",
    },
    "workspace_16_control": {
        "final_model_full_sha256": "bbec7389f56b36a24db000111adf8ce99e3bf712609cfcd1562c64c9910172f8",
        "predictions_sha256": "3663a15ad9859abc76141d815f0da3c88bd9b3d5d9e914f09a3d8b0a3ed843f4",
    },
}
REPLICA_PAIRS = {
    "warmup": ("workspace_16_warmup500", "workspace_16_warmup500_replica"),
    "clip": ("workspace_16_clip1", "workspace_16_clip1_replica"),
}


def expected_intervention(intervention: str) -> tuple[int, float | None]:
    if intervention == "control":
        return 0, None
    if intervention == "warmup":
        return 500, None
    if intervention == "clip":
        return 0, 1.0
    raise ValueError(f"unsupported intervention: {intervention}")


def expected_lr_trace(intervention: str) -> dict[str, float]:
    if intervention == "warmup":
        return {"1": 6e-7, "500": 3e-4, "501": 3e-4, "38805": 3e-4}
    return {"1": 3e-4, "38805": 3e-4}


def audit_catalog_group(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    projections = [invariant_projection(payload) for payload in payloads]
    reference = projections[0]
    checks = {
        key: all(projection[key] == reference[key] for projection in projections[1:])
        for key in reference
    }
    return {"arms": len(payloads), "checks": checks, "passed": all(checks.values())}


def audit_intervention(payload: dict[str, Any], intervention: str) -> dict[str, Any]:
    warmup_steps, max_grad_norm = expected_intervention(intervention)
    stability = payload["query_stability"]
    clipping = stability["gradient_clipping"]
    expected_clip_calls = 38805 if intervention == "clip" else 0
    expected_samples = {"1", "50", "500", "38805"} if intervention == "clip" else set()
    checks = {
        "condition_warmup": payload["condition"]["query_warmup_steps"] == warmup_steps,
        "condition_max_grad_norm": payload["condition"]["query_max_grad_norm"]
        == max_grad_norm,
        "lr_trace": stability["lr_trace"] == expected_lr_trace(intervention),
        "clip_max_grad_norm": clipping["max_grad_norm"] == max_grad_norm,
        "clip_norm_type": float(clipping["norm_type"]) == 2.0,
        "clip_foreach": clipping["foreach"] is True,
        "clip_calls": clipping["calls"] == expected_clip_calls,
        "clip_sample_steps": set(clipping["pre_clip_norm_samples"]) == expected_samples,
        "clip_counts": 0
        <= clipping["sampled_clipped_calls"]
        <= len(clipping["pre_clip_norm_samples"]),
    }
    return {
        "checks": checks,
        "sampled_clipped_calls": clipping["sampled_clipped_calls"],
        "passed": all(checks.values()),
    }


def audit_replica(
    summaries: dict[str, dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    base_label: str,
    replica_label: str,
) -> dict[str, Any]:
    base = summaries[base_label]
    replica = summaries[replica_label]
    checks = {
        "final_model_state_sha256": base["training"]["final_model_state_sha256"]
        == replica["training"]["final_model_state_sha256"],
        "best_step": base["training"]["query_best_step"]
        == replica["training"]["query_best_step"],
        "best_dev_loss": base["training"]["query_best_dev_loss"]
        == replica["training"]["query_best_dev_loss"],
        "predictions_sha256": base["evaluation"]["predictions_sha256"]
        == replica["evaluation"]["predictions_sha256"],
        "metrics": recalls(base) == recalls(replica),
        "query_stability": audits[base_label]["query_stability"]
        == audits[replica_label]["query_stability"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def pair_delta(
    summaries: dict[str, dict[str, Any]], default_label: str, workspace_label: str
) -> dict[str, Any]:
    default_metrics = recalls(summaries[default_label])
    workspace_metrics = recalls(summaries[workspace_label])
    return {
        subset: {
            cutoff: workspace_metrics[subset][cutoff] - default_metrics[subset][cutoff]
            for cutoff in ("@1", "@10", "@50")
        }
        for subset in ("overall", "single", "multi")
    }


def command_matrix(args: argparse.Namespace) -> None:
    root = args.root
    summaries: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    intervention_audits: dict[str, dict[str, Any]] = {}
    for label, role, path, intervention in ARM_SPECS:
        arm_root = root / "arms" / label
        if not (arm_root / "arm.done").is_file():
            raise FileNotFoundError(f"arm completion marker is missing: {label}")
        summary = read_json(arm_root / "result_summary.json")
        audit = read_json(arm_root / "full_query_arm.json")
        observed = audit["condition"]
        warmup_steps, max_grad_norm = expected_intervention(intervention)
        condition_checks = {
            "label": observed["arm"] == label,
            "role": observed["role"] == role,
            "path": observed["path"] == path,
            "optimizer_mode": observed["optimizer_mode"] == "carry",
            "query_lr": float(observed["query_lr"]) == 3e-4,
            "query_warmup_steps": observed["query_warmup_steps"] == warmup_steps,
            "query_max_grad_norm": observed["query_max_grad_norm"] == max_grad_norm,
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
        intervention_audit = audit_intervention(audit, intervention)
        if not intervention_audit["passed"]:
            raise ValueError(
                f"query intervention audit failed for {label}: {intervention_audit}"
            )
        summaries[label] = summary
        audits[label] = audit
        intervention_audits[label] = intervention_audit

    catalog_audits = {
        path: audit_catalog_group(
            [
                audits[label]
                for label, _, observed_path, _ in ARM_SPECS
                if observed_path == path
            ]
        )
        for path in ("default", "workspace_16")
    }
    if not all(row["passed"] for row in catalog_audits.values()):
        raise ValueError(f"catalog invariant failed: {catalog_audits}")

    replica_audits = {
        name: audit_replica(summaries, audits, base, replica)
        for name, (base, replica) in REPLICA_PAIRS.items()
    }
    if not all(row["passed"] for row in replica_audits.values()):
        raise ValueError(f"exact-replica invariant failed: {replica_audits}")

    historical = {
        label: historical_check(
            summaries[label], HISTORICAL_REFERENCES[reference_label]
        )
        for label, reference_label in HISTORICAL_LABELS.items()
    }
    for label, row in historical.items():
        expected_artifact = HISTORICAL_ARTIFACTS[label]
        row["checks"]["final_model_full_sha256"] = summaries[label]["training"][
            "final_model_state_sha256"
        ]["full"] == expected_artifact["final_model_full_sha256"]
        row["checks"]["predictions_sha256"] = summaries[label]["evaluation"][
            "predictions_sha256"
        ] == expected_artifact["predictions_sha256"]
        row["passed"] = all(row["checks"].values())
    if not all(row["passed"] for row in historical.values()):
        raise ValueError(f"historical anchor mismatch: {historical}")

    gates = {label: selection_gate(summaries[label]) for label in SELECTION_LABELS}
    passing = [label for label in SELECTION_LABELS if gates[label]["passed"]]
    selected = max(passing or SELECTION_LABELS, key=lambda label: ranking_key(summaries[label]))
    selection_basis = "all_gates" if passing else "top1_fallback"
    pair_deltas = {
        "control": pair_delta(summaries, "default_control", "workspace_16_control"),
        "warmup": pair_delta(
            summaries, "default_warmup500", "workspace_16_warmup500"
        ),
        "clip": pair_delta(summaries, "default_clip1", "workspace_16_clip1"),
    }

    payload = {
        "artifact_format": "legacy_query_stability_matrix_v1",
        "protocol": {
            "catalog_steps": 2730,
            "query_steps": 38805,
            "eval_steps": 200,
            "batch_size": 16,
            "seed": 20260626,
            "catalog_lr": 3e-4,
            "query_base_lr": 3e-4,
            "dropout_rate": 0.1,
            "checkpoint": "dev_best",
            "evaluation": "Official core75, 75 queries, beam=50, beam_order, CE log-prob",
        },
        "catalog_invariants": catalog_audits,
        "intervention_audits": intervention_audits,
        "replica_invariants": replica_audits,
        "historical_anchor_invariants": historical,
        "backend_pair_deltas": pair_deltas,
        "selection": {
            "arms": list(SELECTION_LABELS),
            "gates": gates,
            "passing_arms": passing,
            "selected_arm": selected,
            "selection_basis": selection_basis,
        },
        "arms": [summaries[label] for label, _, _, _ in ARM_SPECS],
    }
    output = root / "query_stability_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)

    lines = [
        "# Legacy A800 query stability ablation",
        "",
        "Conditions: catalog=2730 updates; query=38805 updates; batch=16; seed=20260626; "
        "catalog/query base lr=3e-4; dropout=0.1; Official core75=75 queries; beam=50; beam-order.",
        "",
        "| arm | role | Overall R@1/10/50 | Single R@1 | Multi R@1 | best step | best dev loss | gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, role, _, _ in ARM_SPECS:
        arm = summaries[label]
        metric = recalls(arm)
        gate = gates[label]["passed"] if label in gates else "audit"
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
    (root / "query_stability_summary.md").write_text(
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
