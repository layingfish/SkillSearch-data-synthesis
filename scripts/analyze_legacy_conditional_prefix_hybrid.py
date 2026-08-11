#!/usr/bin/env python3
"""Audit and select the strict-base gold-cardinality prefix-ranking sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from analyze_legacy_c1_hard_negative_coverage import file_sha256
    from analyze_legacy_ce_prefix_ranking import (
        ABSOLUTE_THRESHOLDS,
        DEEP_RECALL_MARGIN,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_SUMMARY_SHA256,
        catalog_invariant,
        control_invariant,
        historical_control_invariant,
        ranking_key,
        selection_gate,
    )
    from analyze_legacy_full_query_optimizer_lr import command_arm, read_json, recalls
except ModuleNotFoundError:
    from scripts.analyze_legacy_c1_hard_negative_coverage import file_sha256
    from scripts.analyze_legacy_ce_prefix_ranking import (
        ABSOLUTE_THRESHOLDS,
        DEEP_RECALL_MARGIN,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_SUMMARY_SHA256,
        catalog_invariant,
        control_invariant,
        historical_control_invariant,
        ranking_key,
        selection_gate,
    )
    from scripts.analyze_legacy_full_query_optimizer_lr import (
        command_arm,
        read_json,
        recalls,
    )


ARM_SPECS: dict[str, dict[str, Any]] = {
    "control_a": {
        "role": "anchor",
        "grouping": "fixed",
        "scope": "disabled",
        "weight": 0.0,
    },
    "control_b": {
        "role": "replica",
        "grouping": "fixed",
        "scope": "disabled",
        "weight": 0.0,
    },
    "seq_010": {
        "role": "anchor",
        "grouping": "fixed",
        "scope": "sequence_only",
        "weight": 0.10,
    },
    "cum_002": {
        "role": "anchor",
        "grouping": "fixed",
        "scope": "cumulative",
        "weight": 0.02,
    },
    "hybrid_s005_m001": {
        "role": "selection",
        "grouping": "gold_cardinality",
        "scope": "disabled",
        "weight": 0.0,
        "single_weight": 0.05,
        "multi_weight": 0.01,
    },
    "hybrid_s005_m002": {
        "role": "selection",
        "grouping": "gold_cardinality",
        "scope": "disabled",
        "weight": 0.0,
        "single_weight": 0.05,
        "multi_weight": 0.02,
    },
    "hybrid_s010_m001": {
        "role": "selection",
        "grouping": "gold_cardinality",
        "scope": "disabled",
        "weight": 0.0,
        "single_weight": 0.10,
        "multi_weight": 0.01,
    },
    "hybrid_s010_m002": {
        "role": "selection",
        "grouping": "gold_cardinality",
        "scope": "disabled",
        "weight": 0.0,
        "single_weight": 0.10,
        "multi_weight": 0.02,
    },
}
CONTROL_LABELS = ("control_a", "control_b")
FIXED_ANCHORS = ("seq_010", "cum_002")
SELECTION_LABELS = tuple(
    label for label, spec in ARM_SPECS.items() if spec["role"] == "selection"
)
PREVIOUS_SUMMARY_SHA256 = {
    "seq_010": "ac261dd47027a351b6558abec7c88bc2397c92f44530da1731bdfb3575f70f17",
    "cum_002": "e3a8182149c95edc47d6fedbed1c783a6ac3f5df40d1cb0249680f45727d1153",
}
EXPECTED_FROZEN_COUNTS = {
    "source_rows_by_group": {"multi": 17457, "single": 77455},
    "expanded_rows_by_group": {"multi": 46720, "single": 77455},
    "eligible_rows_by_group": {"multi": 32170, "single": 54135},
}
EXPECTED_PRESENTATIONS = {"multi": 160850, "single": 270675}
EXPECTED_CONDITIONAL_COMPARISONS = 753225
EXPECTED_GOLD_COUNT_MAPPING_SHA256 = (
    "016728eb7681486d41eb83a9de27fa32c56e70adce1a0b17d47d8631bff796aa"
)


def choose_qualified_arm(
    summaries: dict[str, dict[str, Any]], gates: dict[str, dict[str, Any]]
) -> tuple[list[str], str | None, str]:
    passing = [label for label in SELECTION_LABELS if gates[label]["passed"]]
    if not passing:
        return [], None, "no_qualified_arm"
    selected = max(passing, key=lambda label: ranking_key(summaries[label]))
    return passing, selected, "qualified_only"


def fixed_anchor_invariant(
    current: dict[str, Any], previous_path: Path, expected_sha256: str
) -> dict[str, Any]:
    previous_sha256 = file_sha256(previous_path)
    previous = read_json(previous_path)
    checks = {
        "previous_summary_sha256": previous_sha256 == expected_sha256,
        "final_model_state": current["training"]["final_model_state_sha256"]
        == previous["training"]["final_model_state_sha256"],
        "best_step": current["training"]["query_best_step"]
        == previous["training"]["query_best_step"],
        "best_dev_loss": current["training"]["query_best_dev_loss"]
        == previous["training"]["query_best_dev_loss"],
        "predictions": current["evaluation"]["predictions_sha256"]
        == previous["evaluation"]["predictions_sha256"],
        "recalls": recalls(current) == recalls(previous),
    }
    return {
        "previous_summary": str(previous_path),
        "previous_summary_sha256": previous_sha256,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parity_passed(trace: dict[str, Any], grouping: str) -> bool:
    parity = trace.get("score_parity", {})
    if grouping == "fixed":
        return set(parity) == {"positive", "negative"} and all(
            row.get("passed") for row in parity.values()
        )
    return set(parity) == {"single", "multi"} and all(
        set(group_rows) == {"positive", "negative"}
        and all(row.get("passed") for row in group_rows.values())
        for group_rows in parity.values()
    )


def command_matrix(args: argparse.Namespace) -> None:
    summaries: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    gold_count_mapping_hashes: dict[str, str] = {}
    for label, spec in ARM_SPECS.items():
        arm_root = args.root / "arms" / label
        if not (arm_root / "arm.done").is_file():
            raise FileNotFoundError(f"arm completion marker is missing: {label}")
        summary = read_json(arm_root / "result_summary.json")
        audit = read_json(arm_root / "full_query_arm.json")
        condition = audit["condition"]
        prefix = condition["prefix_ranking"]
        checks = {
            "label": condition["arm"] == label,
            "role": condition["role"] == spec["role"],
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
            "grouping": prefix["grouping"] == spec["grouping"],
            "scope": prefix["scope"] == spec["scope"],
            "weight": float(prefix["weight"]) == spec["weight"],
            "temperature": float(prefix["temperature"]) == 1.0,
            "summary_sha256": prefix["summary_sha256"] == EXPECTED_SUMMARY_SHA256,
            "manifest_sha256": prefix["manifest_sha256"] == EXPECTED_MANIFEST_SHA256,
        }
        if spec["grouping"] == "gold_cardinality":
            checks.update(
                {
                    "single_scope": prefix["single"]["scope"] == "sequence_only",
                    "single_weight": float(prefix["single"]["weight"])
                    == spec["single_weight"],
                    "multi_scope": prefix["multi"]["scope"] == "cumulative",
                    "multi_weight": float(prefix["multi"]["weight"])
                    == spec["multi_weight"],
                    "normalization": prefix["conditional_normalization"]
                    == "mean_over_active_depths_per_row_then_weight_then_mean_over_eligible_rows",
                }
            )
        if not all(checks.values()):
            raise ValueError(f"arm condition mismatch for {label}: {checks}")

        execution = audit["prefix_ranking"]["execution"]
        if spec["weight"] == 0.0 and spec["grouping"] == "fixed":
            if (
                int(execution["eligible_pair_presentations"]) != 0
                or int(execution["active_comparisons"]) != 0
            ):
                raise ValueError(f"disabled control executed auxiliary loss: {label}")
        else:
            expected_comparisons = (
                431525 * len((4,) if spec["scope"] == "sequence_only" else (2, 3, 4))
                if spec["grouping"] == "fixed"
                else EXPECTED_CONDITIONAL_COMPARISONS
            )
            execution_checks = {
                "eligible_presentations": int(execution["eligible_pair_presentations"])
                == 431525,
                "group_presentations": execution["eligible_pair_presentations_by_group"]
                == EXPECTED_PRESENTATIONS,
                "active_comparisons": int(execution["active_comparisons"])
                == expected_comparisons,
                "frozen_counts": execution["frozen_candidate_counts"]
                == EXPECTED_FROZEN_COUNTS,
                "target_layout": bool(execution["target_layout_verified"]),
                "gold_count_mapping": execution["gold_count_mapping_sha256"]
                == EXPECTED_GOLD_COUNT_MAPPING_SHA256,
                "score_parity": parity_passed(
                    audit["prefix_ranking"]["trace"], spec["grouping"]
                ),
            }
            if not all(execution_checks.values()):
                raise ValueError(
                    f"prefix ranking execution mismatch for {label}: {execution_checks}"
                )
            gold_count_mapping_hashes[label] = execution["gold_count_mapping_sha256"]
        summaries[label] = summary
        audits[label] = audit

    if len(set(gold_count_mapping_hashes.values())) != 1:
        raise ValueError(
            f"gold-count mappings differ across active arms: {gold_count_mapping_hashes}"
        )
    catalog = catalog_invariant(audits)
    controls = control_invariant(summaries, audits)
    historical_controls = {
        label: historical_control_invariant(summaries[label])
        for label in CONTROL_LABELS
    }
    fixed_anchors = {
        label: fixed_anchor_invariant(
            summaries[label],
            args.previous_root / "arms" / label / "result_summary.json",
            PREVIOUS_SUMMARY_SHA256[label],
        )
        for label in FIXED_ANCHORS
    }
    invariants = {
        "catalog": catalog,
        "control_replica": controls,
        "historical_controls": historical_controls,
        "fixed_anchors": fixed_anchors,
    }
    if not catalog["passed"]:
        raise ValueError(f"catalog invariant failed: {catalog}")
    if not controls["passed"]:
        raise ValueError(f"control replica invariant failed: {controls}")
    if not all(row["passed"] for row in historical_controls.values()):
        raise ValueError(f"historical control invariant failed: {historical_controls}")
    if not all(row["passed"] for row in fixed_anchors.values()):
        raise ValueError(f"fixed anchor invariant failed: {fixed_anchors}")

    control_recalls = recalls(summaries[CONTROL_LABELS[0]])
    gates = {
        label: selection_gate(summaries[label], control_recalls)
        for label in SELECTION_LABELS
    }
    combination_gates = {
        label: {
            "single_r1_at_least_seq_anchor": recalls(summaries[label])["single"]["@1"]
            >= recalls(summaries["seq_010"])["single"]["@1"],
            "multi_r1_at_least_cum_anchor": recalls(summaries[label])["multi"]["@1"]
            >= recalls(summaries["cum_002"])["multi"]["@1"],
        }
        for label in SELECTION_LABELS
    }
    for row in combination_gates.values():
        row["passed"] = all(row.values())
    passing, selected, selection_basis = choose_qualified_arm(summaries, gates)
    payload = {
        "artifact_format": "legacy_conditional_prefix_hybrid_matrix_v1",
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
            "selection_status": "exploratory_reuse_of_core75",
        },
        "absolute_thresholds": ABSOLUTE_THRESHOLDS,
        "deep_recall_control_margin": DEEP_RECALL_MARGIN,
        "expected_frozen_counts": EXPECTED_FROZEN_COUNTS,
        "expected_presentations": EXPECTED_PRESENTATIONS,
        "expected_conditional_comparisons": EXPECTED_CONDITIONAL_COMPARISONS,
        "gold_count_mapping_sha256": EXPECTED_GOLD_COUNT_MAPPING_SHA256,
        "invariants": invariants,
        "selection": {
            "gates": gates,
            "combination_diagnostic_gates": combination_gates,
            "passing_arms": passing,
            "selected_arm": selected,
            "selection_basis": selection_basis,
        },
        "arms": [summaries[label] for label in ARM_SPECS],
    }
    output = args.root / "conditional_prefix_hybrid_summary.json"
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
    matrix.add_argument("--previous-root", type=Path, required=True)
    matrix.set_defaults(func=command_matrix)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
