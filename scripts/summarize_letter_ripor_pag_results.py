#!/usr/bin/env python3
"""Collate the locked fixed-LETTER RIPOR/PAG experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


K_VALUES = (1, 10, 50)
MAIN_RUNS = ("B", "ripor", "pag", "ripor_pag")
DIAGNOSTIC_RUNS = ("pag", "pag_weight0", "ripor_pag", "ripor_pag_weight0")
RUN_PATHS = {
    "B": {
        "official": "baseline_eval/official_core75/predictions.jsonl",
        "private": "baseline_eval/private_test/predictions.jsonl",
    },
    "ripor": {
        "official": "ripor_only/eval/official_core75/predictions.jsonl",
        "private": "ripor_only/eval/private_test/predictions.jsonl",
    },
    "pag": {
        "official": "pag_only/eval/official_core75/planning/predictions.jsonl",
        "private": "pag_only/eval/private_test/planning/predictions.jsonl",
    },
    "ripor_pag": {
        "official": "ripor_pag/eval/official_core75/planning/predictions.jsonl",
        "private": "ripor_pag/eval/private_test/planning/predictions.jsonl",
    },
    "pag_weight0": {
        "official": "pag_only/eval/official_core75/weight0_diagnostic/predictions.jsonl",
        "private": "pag_only/eval/private_test/weight0_diagnostic/predictions.jsonl",
    },
    "ripor_pag_weight0": {
        "official": "ripor_pag/eval/official_core75/weight0_diagnostic/predictions.jsonl",
        "private": "ripor_pag/eval/private_test/weight0_diagnostic/predictions.jsonl",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metrics_at_k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": len(rows),
        "total_gold": sum(len(set(row["gold_skill_ids"])) for row in rows),
        "invalid_rate": sum(bool(row.get("invalid")) for row in rows) / max(1, len(rows)),
        "rows_with_duplicate_ranked_ids": sum(
            len(row.get("ranked_skill_ids") or []) != len(set(row.get("ranked_skill_ids") or []))
            for row in rows
        ),
    }
    for k in K_VALUES:
        total_hits = 0
        macro_recalls: list[float] = []
        hits = 0
        full_coverage = 0
        for row in rows:
            gold = set(str(value) for value in row["gold_skill_ids"])
            ranked = [str(value) for value in (row.get("ranked_skill_ids") or [])[:k]]
            hit_count = len(gold.intersection(ranked))
            total_hits += hit_count
            macro_recalls.append(hit_count / len(gold) if gold else 0.0)
            hits += int(hit_count > 0)
            full_coverage += int(bool(gold) and hit_count == len(gold))
        result[f"@{k}"] = {
            "macro_recall": sum(macro_recalls) / max(1, len(macro_recalls)),
            "micro_recall": total_hits / max(1, result["total_gold"]),
            "hit": hits / max(1, len(rows)),
            "full_coverage": full_coverage / max(1, len(rows)),
            "total_hits": total_hits,
        }
    candidate_recalls = [
        float(row["pag_candidate_recall"])
        for row in rows
        if row.get("pag_candidate_recall") is not None
    ]
    result["pag_candidate_recall_mean"] = (
        sum(candidate_recalls) / len(candidate_recalls) if candidate_recalls else None
    )
    return result


def grouped_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_count: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        count = len(set(row["gold_skill_ids"]))
        by_count.setdefault(str(count), []).append(row)
    return {
        "overall": metrics_at_k(rows),
        "single": metrics_at_k([row for row in rows if len(set(row["gold_skill_ids"])) == 1]),
        "multi": metrics_at_k([row for row in rows if len(set(row["gold_skill_ids"])) > 1]),
        "by_target_count": {
            count: metrics_at_k(group) for count, group in sorted(by_count.items(), key=lambda item: int(item[0]))
        },
    }


def row_identity(row: dict[str, Any]) -> tuple[Any, str, tuple[str, ...]]:
    return (
        row.get("index"),
        str(row.get("query") or ""),
        tuple(sorted(str(value) for value in row["gold_skill_ids"])),
    )


def validate_aligned_predictions(
    runs: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    for split in ("official", "private"):
        available = [(name, values[split]) for name, values in runs.items() if split in values]
        if not available:
            continue
        reference_name, reference_rows = available[0]
        reference = [row_identity(row) for row in reference_rows]
        for name, rows in available[1:]:
            actual = [row_identity(row) for row in rows]
            if actual != reference:
                raise ValueError(f"prediction rows are not aligned: {reference_name} vs {name} on {split}")
        expected_rows = 75 if split == "official" else 1000
        if len(reference_rows) != expected_rows:
            raise ValueError(f"unexpected {split} row count: {len(reference_rows)} != {expected_rows}")


def primary_values(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "single_r@1": metrics["single"]["@1"]["macro_recall"],
        "single_r@10": metrics["single"]["@10"]["macro_recall"],
        "single_r@50": metrics["single"]["@50"]["macro_recall"],
        "multi_macro_r@10": metrics["multi"]["@10"]["macro_recall"],
        "multi_macro_r@50": metrics["multi"]["@50"]["macro_recall"],
        "multi_micro_r@10": metrics["multi"]["@10"]["micro_recall"],
        "multi_micro_r@50": metrics["multi"]["@50"]["micro_recall"],
        "multi_hit@10": metrics["multi"]["@10"]["hit"],
        "multi_hit@50": metrics["multi"]["@50"]["hit"],
        "multi_fc@50": metrics["multi"]["@50"]["full_coverage"],
        "overall_r@10": metrics["overall"]["@10"]["macro_recall"],
        "overall_r@50": metrics["overall"]["@50"]["macro_recall"],
    }


def subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in left}


def comparison_rows(values: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    required = {"B", "ripor", "pag", "ripor_pag"}
    if not required.issubset(values):
        return {}
    result = {
        "ripor-B": subtract(values["ripor"], values["B"]),
        "pag-B": subtract(values["pag"], values["B"]),
        "ripor_pag-ripor": subtract(values["ripor_pag"], values["ripor"]),
        "ripor_pag-pag": subtract(values["ripor_pag"], values["pag"]),
    }
    result["interaction"] = {
        key: values["ripor_pag"][key] - values["ripor"][key] - values["pag"][key] + values["B"][key]
        for key in values["B"]
    }
    return result


def markdown_table(title: str, rows: dict[str, dict[str, float]]) -> list[str]:
    columns = (
        "single_r@1",
        "single_r@10",
        "multi_macro_r@10",
        "multi_macro_r@50",
        "multi_micro_r@50",
        "multi_fc@50",
        "overall_r@10",
    )
    lines = [f"## {title}", "", "| Setting | " + " | ".join(columns) + " |"]
    lines.append("|---|" + "---:|" * len(columns))
    for name, values in rows.items():
        lines.append("| " + name + " | " + " | ".join(f"{values[key]:.6f}" for key in columns) + " |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    missing: list[str] = []
    for run_name, split_paths in RUN_PATHS.items():
        for split, relative in split_paths.items():
            path = args.root / relative
            if path.is_file():
                runs.setdefault(run_name, {})[split] = load_jsonl(path)
            else:
                missing.append(f"{run_name}:{split}:{path}")
    if missing and not args.allow_incomplete:
        raise FileNotFoundError("missing required prediction artifacts:\n" + "\n".join(missing))
    validate_aligned_predictions(runs)

    metrics = {
        run_name: {split: grouped_metrics(rows) for split, rows in split_rows.items()}
        for run_name, split_rows in runs.items()
    }
    primary = {
        split: {
            run_name: primary_values(split_metrics[split])
            for run_name, split_metrics in metrics.items()
            if run_name in MAIN_RUNS and split in split_metrics
        }
        for split in ("official", "private")
    }
    diagnostics = {
        split: {
            run_name: primary_values(metrics[run_name][split])
            for run_name in DIAGNOSTIC_RUNS
            if run_name in metrics and split in metrics[run_name]
        }
        for split in ("official", "private")
    }
    comparisons = {split: comparison_rows(values) for split, values in primary.items()}
    payload = {
        "protocol": {
            "k_values": list(K_VALUES),
            "subsets": ["overall", "single", "multi", "by_target_count"],
            "official_is_primary": True,
            "missing_artifacts": missing,
        },
        "metrics": metrics,
        "primary": primary,
        "planning_weight0_diagnostics": diagnostics,
        "comparisons": comparisons,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = ["# Fixed-LETTER RIPOR / PAG Results", ""]
    for split in ("official", "private"):
        if primary[split]:
            markdown.extend(markdown_table(f"{split} main results", primary[split]))
        if comparisons[split]:
            markdown.extend(markdown_table(f"{split} deltas and interaction", comparisons[split]))
        if diagnostics[split]:
            markdown.extend(markdown_table(f"{split} PAG planning vs weight0", diagnostics[split]))
    (args.output_dir / "metrics_comparison.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"runs": sorted(runs), "missing": len(missing), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
