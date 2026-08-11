#!/usr/bin/env python3
"""Paired error-overlap analysis for strict-base CE-prefix ranking arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from analyze_legacy_core75_query_ranks import (
        analyze as analyze_query_ranks,
        rank_bucket,
        read_predictions,
    )
except ModuleNotFoundError:
    from scripts.analyze_legacy_core75_query_ranks import (
        analyze as analyze_query_ranks,
        rank_bucket,
        read_predictions,
    )


LABELS = ("control", "sequence", "cumulative")
CUTOFFS = (1, 10, 50)
SUBSETS = ("overall", "single", "multi")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subset_rows(rows: Sequence[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    if subset == "overall":
        return list(rows)
    if subset == "single":
        return [row for row in rows if row["target_skill_count"] == 1]
    if subset == "multi":
        return [row for row in rows if row["target_skill_count"] > 1]
    raise ValueError(f"unsupported subset: {subset}")


def _recall(row: Mapping[str, Any], label: str, cutoff: int) -> float:
    return float(row["systems"][label]["macro_recall_contribution"][f"@{cutoff}"])


def _macro_recall(rows: Sequence[dict[str, Any]], label: str, cutoff: int) -> float:
    if not rows:
        raise ValueError("cannot summarize an empty subset")
    return math.fsum(_recall(row, label, cutoff) for row in rows) / len(rows)


def _rank_map(raw_row: Mapping[str, Any]) -> dict[str, int | None]:
    ranked = raw_row["ranked_skill_ids"]
    rank_by_skill = {skill_id: rank for rank, skill_id in enumerate(ranked, start=1)}
    return {
        skill_id: rank_by_skill.get(skill_id) for skill_id in raw_row["gold_skill_ids"]
    }


def enrich_per_query(
    per_query: list[dict[str, Any]],
    systems: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    indexed = {
        label: {int(row["index"]): row for row in rows}
        for label, rows in systems.items()
    }
    for row in per_query:
        index = int(row["index"])
        for label in LABELS:
            row["systems"][label]["gold_rank_by_skill"] = _rank_map(
                indexed[label][index]
            )
        row["top1_hit_pattern"] = "".join(
            "1" if row["systems"][label]["top1_correct"] else "0" for label in LABELS
        )
        row["oracle_macro_recall_contribution"] = {
            f"@{cutoff}": max(_recall(row, label, cutoff) for label in LABELS)
            for cutoff in CUTOFFS
        }
        row["gold_cardinality_router"] = {
            "deployable": False,
            "reason": "uses gold target count",
            "selected_system": (
                "sequence" if row["target_skill_count"] == 1 else "cumulative"
            ),
        }
    return indexed


def _candidate_overlap(
    rows: Sequence[dict[str, Any]],
    indexed: Mapping[str, Mapping[int, Mapping[str, Any]]],
    left: str,
    right: str,
    cutoff: int,
) -> dict[str, float]:
    shared_fractions = []
    jaccards = []
    for row in rows:
        index = int(row["index"])
        left_set = set(indexed[left][index]["ranked_skill_ids"][:cutoff])
        right_set = set(indexed[right][index]["ranked_skill_ids"][:cutoff])
        intersection = len(left_set.intersection(right_set))
        shared_fractions.append(intersection / cutoff)
        jaccards.append(intersection / len(left_set.union(right_set)))
    return {
        "mean_shared_fraction": math.fsum(shared_fractions) / len(shared_fractions),
        "mean_jaccard": math.fsum(jaccards) / len(jaccards),
    }


def summarize_pair(
    rows: Sequence[dict[str, Any]],
    indexed: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    left: str,
    right: str,
) -> dict[str, Any]:
    hit_patterns: Counter[str] = Counter()
    same_top1 = 0
    cutoff_summaries: dict[str, Any] = {}
    bucket_transitions: Counter[str] = Counter()

    for row in rows:
        left_correct = bool(row["systems"][left]["top1_correct"])
        right_correct = bool(row["systems"][right]["top1_correct"])
        hit_patterns[f"{int(left_correct)}{int(right_correct)}"] += 1
        same_top1 += (
            row["systems"][left]["top1_skill_id"]
            == row["systems"][right]["top1_skill_id"]
        )
        for skill_id in row["gold_skill_ids"]:
            left_rank = row["systems"][left]["gold_rank_by_skill"][skill_id]
            right_rank = row["systems"][right]["gold_rank_by_skill"][skill_id]
            bucket_transitions[
                f"{rank_bucket(left_rank)}->{rank_bucket(right_rank)}"
            ] += 1

    for cutoff in CUTOFFS:
        key = f"@{cutoff}"
        gained_indices: list[int] = []
        lost_indices: list[int] = []
        tied_indices: list[int] = []
        entered_gold: list[dict[str, Any]] = []
        exited_gold: list[dict[str, Any]] = []
        entered_weight = 0.0
        exited_weight = 0.0
        for row in rows:
            index = int(row["index"])
            delta = _recall(row, right, cutoff) - _recall(row, left, cutoff)
            if delta > 0.0:
                gained_indices.append(index)
            elif delta < 0.0:
                lost_indices.append(index)
            else:
                tied_indices.append(index)
            weight = 1.0 / row["target_skill_count"]
            for skill_id in row["gold_skill_ids"]:
                left_rank = row["systems"][left]["gold_rank_by_skill"][skill_id]
                right_rank = row["systems"][right]["gold_rank_by_skill"][skill_id]
                left_in = left_rank is not None and left_rank <= cutoff
                right_in = right_rank is not None and right_rank <= cutoff
                if right_in and not left_in:
                    entered_weight += weight
                    entered_gold.append(
                        {
                            "index": index,
                            "skill_id": skill_id,
                            "left_rank": left_rank,
                            "right_rank": right_rank,
                        }
                    )
                elif left_in and not right_in:
                    exited_weight += weight
                    exited_gold.append(
                        {
                            "index": index,
                            "skill_id": skill_id,
                            "left_rank": left_rank,
                            "right_rank": right_rank,
                        }
                    )
        observed_delta = _macro_recall(rows, right, cutoff) - _macro_recall(
            rows, left, cutoff
        )
        transition_delta = (entered_weight - exited_weight) / len(rows)
        cutoff_summaries[key] = {
            "macro_recall_delta": observed_delta,
            "gained_query_indices": gained_indices,
            "lost_query_indices": lost_indices,
            "tied_query_indices": tied_indices,
            "entered_gold": entered_gold,
            "exited_gold": exited_gold,
            "entered_gold_weight": entered_weight,
            "exited_gold_weight": exited_weight,
            "transition_macro_recall_delta": transition_delta,
            "transition_delta_matches": math.isclose(
                observed_delta, transition_delta, rel_tol=0.0, abs_tol=1e-12
            ),
            "candidate_overlap": _candidate_overlap(rows, indexed, left, right, cutoff),
        }

    return {
        "left": left,
        "right": right,
        "rows": len(rows),
        "same_top1_skill_queries": same_top1,
        "top1_hit_patterns": {
            pattern: hit_patterns.get(pattern, 0)
            for pattern in ("00", "01", "10", "11")
        },
        "gold_rank_bucket_transitions": dict(sorted(bucket_transitions.items())),
        "cutoffs": cutoff_summaries,
    }


def summarize_subset(
    rows: Sequence[dict[str, Any]],
    indexed: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    pattern_indices: dict[str, list[int]] = {}
    for row in rows:
        pattern_indices.setdefault(row["top1_hit_pattern"], []).append(
            int(row["index"])
        )

    systems = {
        label: {
            "top1_hit_queries": sum(
                bool(row["systems"][label]["top1_correct"]) for row in rows
            ),
            "macro_recall": {
                f"@{cutoff}": _macro_recall(rows, label, cutoff) for cutoff in CUTOFFS
            },
        }
        for label in LABELS
    }
    oracle = {
        f"@{cutoff}": math.fsum(
            row["oracle_macro_recall_contribution"][f"@{cutoff}"] for row in rows
        )
        / len(rows)
        for cutoff in CUTOFFS
    }
    router = {
        f"@{cutoff}": math.fsum(
            _recall(
                row,
                row["gold_cardinality_router"]["selected_system"],
                cutoff,
            )
            for row in rows
        )
        / len(rows)
        for cutoff in CUTOFFS
    }
    return {
        "rows": len(rows),
        "total_gold": sum(row["target_skill_count"] for row in rows),
        "systems": systems,
        "top1_hit_pattern_order": list(LABELS),
        "top1_hit_pattern_indices": dict(sorted(pattern_indices.items())),
        "pairs": {
            "control_to_sequence": summarize_pair(
                rows, indexed, left="control", right="sequence"
            ),
            "control_to_cumulative": summarize_pair(
                rows, indexed, left="control", right="cumulative"
            ),
            "sequence_to_cumulative": summarize_pair(
                rows, indexed, left="sequence", right="cumulative"
            ),
        },
        "per_query_oracle": {
            "deployable": False,
            "reason": "chooses the highest-recall arm using gold labels per query",
            "macro_recall": oracle,
            "gain_over_best_constituent": {
                key: oracle[key]
                - max(system["macro_recall"][key] for system in systems.values())
                for key in oracle
            },
        },
        "gold_cardinality_router": {
            "deployable": False,
            "reason": "routes Single to sequence and Multi to cumulative using gold target count",
            "macro_recall": router,
        },
    }


def analyze_overlap(
    systems: Mapping[str, list[dict[str, Any]]],
    *,
    input_paths: Mapping[str, Path],
    metric_paths: Mapping[str, Path],
    expected_rows: int = 75,
    expected_depth: int = 50,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if tuple(systems) != LABELS:
        raise ValueError(f"systems must be ordered as {LABELS}, got {tuple(systems)}")
    base, per_query = analyze_query_ranks(
        systems,
        reference_label="control",
        expected_rows=expected_rows,
        expected_depth=expected_depth,
        cutoffs=CUTOFFS,
        input_paths=input_paths,
        metric_paths=metric_paths,
    )
    indexed = enrich_per_query(per_query, systems)
    subsets = {
        subset: summarize_subset(_subset_rows(per_query, subset), indexed)
        for subset in SUBSETS
    }

    checks: dict[str, bool] = {
        "base_forensics": bool(base["verification"]["passed"]),
        "overall_rows": subsets["overall"]["rows"] == expected_rows,
        "single_multi_partition": (
            subsets["single"]["rows"] + subsets["multi"]["rows"] == expected_rows
        ),
    }
    for subset, summary in subsets.items():
        checks[f"{subset}.top1_patterns"] = (
            sum(
                len(indices) for indices in summary["top1_hit_pattern_indices"].values()
            )
            == summary["rows"]
        )
        for pair_label, pair in summary["pairs"].items():
            checks[f"{subset}.{pair_label}.hit_patterns"] = (
                sum(pair["top1_hit_patterns"].values()) == summary["rows"]
            )
            checks[f"{subset}.{pair_label}.cutoff_partition"] = all(
                len(row["gained_query_indices"])
                + len(row["lost_query_indices"])
                + len(row["tied_query_indices"])
                == summary["rows"]
                for row in pair["cutoffs"].values()
            )
            checks[f"{subset}.{pair_label}.transition_delta"] = all(
                row["transition_delta_matches"] for row in pair["cutoffs"].values()
            )
        checks[f"{subset}.oracle_dominates"] = all(
            summary["per_query_oracle"]["macro_recall"][f"@{cutoff}"] + 1e-12
            >= system["macro_recall"][f"@{cutoff}"]
            for cutoff in CUTOFFS
            for system in summary["systems"].values()
        )
    if not all(checks.values()):
        raise ValueError(f"CE-prefix overlap verification failed: {checks}")

    payload = {
        "artifact_format": "legacy_ce_prefix_error_overlap_v1",
        "protocol": {
            "systems": list(LABELS),
            "query_count": expected_rows,
            "single_queries": subsets["single"]["rows"],
            "multi_queries": subsets["multi"]["rows"],
            "rank_depth": expected_depth,
            "cutoffs": list(CUTOFFS),
            "evaluation": "Official core75; beam=50; beam-order; macro recall",
        },
        "inputs": {
            label: {
                "predictions": str(input_paths[label]),
                "predictions_sha256": sha256_file(input_paths[label]),
                "result_summary": str(metric_paths[label]),
                "result_summary_sha256": sha256_file(metric_paths[label]),
            }
            for label in LABELS
        },
        "subsets": subsets,
        "base_forensics": {
            "systems": base["systems"],
            "comparisons_to_control": base["comparisons_to_reference"],
            "pairwise_rank_overlap": base["pairwise_rank_overlap"],
            "verification": base["verification"],
        },
        "verification": {"checks": checks, "passed": True},
    }
    return payload, per_query


def _pct(value: float) -> str:
    return f"{100.0 * value:.4f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    subsets = payload["subsets"]
    lines = [
        "# CE-prefix paired query error overlap",
        "",
        "Conditions: strict-base dev-best checkpoints; Official core75=75 queries; "
        "Single=24; Multi=51; beam=50; beam-order; macro recall.",
        "",
        "## Metrics",
        "",
        "| subset | system | R@1 | R@10 | R@50 | top-1 hit queries |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for subset in SUBSETS:
        for label in LABELS:
            system = subsets[subset]["systems"][label]
            recall = system["macro_recall"]
            lines.append(
                f"| `{subset}` | `{label}` | {_pct(recall['@1'])} | "
                f"{_pct(recall['@10'])} | {_pct(recall['@50'])} | "
                f"{system['top1_hit_queries']} |"
            )

    lines.extend(
        [
            "",
            "## Top-1 hit-pattern overlap",
            "",
            "Pattern order is `control/sequence/cumulative`; `1` means top-1 is a gold skill.",
            "",
            "| pattern | Overall indices | Single indices | Multi indices |",
            "|---|---|---|---|",
        ]
    )
    all_patterns = sorted(
        {
            pattern
            for subset in subsets.values()
            for pattern in subset["top1_hit_pattern_indices"]
        }
    )
    for pattern in all_patterns:
        values = [
            subsets[name]["top1_hit_pattern_indices"].get(pattern, [])
            for name in SUBSETS
        ]
        lines.append(f"| `{pattern}` | `{values[0]}` | `{values[1]}` | `{values[2]}` |")

    lines.extend(
        [
            "",
            "## Pairwise changes",
            "",
            "Gained/lost counts use per-query macro-recall contribution. Shared Top10/Top50 is candidate-set intersection divided by cutoff.",
            "",
            "| subset | pair | R@1 gained/lost | R@10 gained/lost | R@50 gained/lost | shared Top10/Top50 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for subset_name in SUBSETS:
        for pair_name, pair in subsets[subset_name]["pairs"].items():
            c1 = pair["cutoffs"]["@1"]
            c10 = pair["cutoffs"]["@10"]
            c50 = pair["cutoffs"]["@50"]
            lines.append(
                f"| `{subset_name}` | `{pair_name}` | "
                f"{len(c1['gained_query_indices'])}/{len(c1['lost_query_indices'])} | "
                f"{len(c10['gained_query_indices'])}/{len(c10['lost_query_indices'])} | "
                f"{len(c50['gained_query_indices'])}/{len(c50['lost_query_indices'])} | "
                f"{100 * c10['candidate_overlap']['mean_shared_fraction']:.2f}%/"
                f"{100 * c50['candidate_overlap']['mean_shared_fraction']:.2f}% |"
            )

    lines.extend(
        [
            "",
            "## Diagnostic upper bounds",
            "",
            "Both rows use gold labels and are not deployable inference policies.",
            "",
            "| subset | diagnostic | R@1 | R@10 | R@50 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for subset_name in SUBSETS:
        subset = subsets[subset_name]
        for key in ("per_query_oracle", "gold_cardinality_router"):
            recall = subset[key]["macro_recall"]
            lines.append(
                f"| `{subset_name}` | `{key}` | {_pct(recall['@1'])} | "
                f"{_pct(recall['@10'])} | {_pct(recall['@50'])} |"
            )

    lines.extend(
        [
            "",
            "Verification: input hashes, row/gold alignment, independent metric recomputation, "
            "Top-1 pattern conservation, cutoff gained/lost/tie conservation, weighted gold-transition "
            "conservation, and oracle dominance all passed.",
        ]
    )
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def command_analyze(args: argparse.Namespace) -> None:
    input_paths = {
        "control": args.control_predictions,
        "sequence": args.sequence_predictions,
        "cumulative": args.cumulative_predictions,
    }
    metric_paths = {
        "control": args.control_summary,
        "sequence": args.sequence_summary,
        "cumulative": args.cumulative_summary,
    }
    systems = {label: read_predictions(path) for label, path in input_paths.items()}
    payload, per_query = analyze_overlap(
        systems,
        input_paths=input_paths,
        metric_paths=metric_paths,
        expected_rows=args.expected_rows,
        expected_depth=args.expected_depth,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write(
        args.output_dir / "ce_prefix_error_overlap_summary.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(
        args.output_dir / "per_query.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_query),
    )
    atomic_write(
        args.output_dir / "ce_prefix_error_overlap.md",
        render_markdown(payload),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "verification_passed": payload["verification"]["passed"],
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--sequence-predictions", type=Path, required=True)
    parser.add_argument("--cumulative-predictions", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--sequence-summary", type=Path, required=True)
    parser.add_argument("--cumulative-summary", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=75)
    parser.add_argument("--expected-depth", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    command_analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
