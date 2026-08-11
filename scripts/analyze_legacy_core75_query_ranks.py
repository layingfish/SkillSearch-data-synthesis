#!/usr/bin/env python3
"""Paired query-rank forensics for aligned Official core75 predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_CUTOFFS = (1, 5, 10, 50)
RANK_BUCKETS = ("rank1", "rank2_5", "rank6_10", "rank11_50", "missing_top50")
ALIGNMENT_FIELDS = (
    "source_index",
    "query",
    "gold_skill_ids",
    "target_skill_count",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def parse_label_paths(items: Sequence[str], flag: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"{flag} must use LABEL=PATH, got: {item}")
        if label in parsed:
            raise ValueError(f"duplicate label for {flag}: {label}")
        parsed[label] = Path(raw_path)
    return parsed


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing_top50"
    if rank == 1:
        return "rank1"
    if rank <= 5:
        return "rank2_5"
    if rank <= 10:
        return "rank6_10"
    if rank <= 50:
        return "rank11_50"
    raise ValueError(f"rank exceeds Top-50 bucket contract: {rank}")


def rank_of(skill_id: str, ranked_skill_ids: Sequence[str]) -> int | None:
    try:
        return ranked_skill_ids.index(skill_id) + 1
    except ValueError:
        return None


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(materialized),
        "mean": math.fsum(materialized) / len(materialized),
        "median": statistics.median(materialized),
        "min": min(materialized),
        "max": max(materialized),
    }


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / denominator


def _score_map(
    row: Mapping[str, Any],
    ranked_skill_ids: Sequence[str],
    *,
    label: str,
    index: int,
) -> dict[str, float] | None:
    raw_scores = row.get("ranked_skill_scores")
    if raw_scores is None:
        return None
    if not isinstance(raw_scores, dict):
        raise ValueError(f"{label} index={index}: ranked_skill_scores must be object/null")
    if set(raw_scores) != set(ranked_skill_ids):
        missing = sorted(set(ranked_skill_ids) - set(raw_scores))
        extra = sorted(set(raw_scores) - set(ranked_skill_ids))
        raise ValueError(
            f"{label} index={index}: score keys differ from ranked Top-50; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    scores = {skill_id: float(raw_scores[skill_id]) for skill_id in ranked_skill_ids}
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError(f"{label} index={index}: non-finite ranked score")
    ordered = [scores[skill_id] for skill_id in ranked_skill_ids]
    if any(left + 1e-10 < right for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"{label} index={index}: ranked scores are not monotonic")
    return scores


def validate_and_align(
    systems: Mapping[str, list[dict[str, Any]]],
    *,
    reference_label: str,
    expected_rows: int,
    expected_depth: int,
) -> tuple[list[int], dict[str, dict[int, dict[str, Any]]], dict[str, str]]:
    if reference_label not in systems:
        raise ValueError(f"reference label is absent: {reference_label}")

    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    score_availability: dict[str, str] = {}
    for label, rows in systems.items():
        if len(rows) != expected_rows:
            raise ValueError(f"{label}: expected {expected_rows} rows, found {len(rows)}")
        by_index: dict[int, dict[str, Any]] = {}
        score_modes: set[str] = set()
        for row in rows:
            index = row.get("index")
            if not isinstance(index, int):
                raise ValueError(f"{label}: index must be int, got {index!r}")
            if index in by_index:
                raise ValueError(f"{label}: duplicate index={index}")
            ranked = row.get("ranked_skill_ids")
            gold = row.get("gold_skill_ids")
            if not isinstance(ranked, list) or not all(isinstance(x, str) for x in ranked):
                raise ValueError(f"{label} index={index}: invalid ranked_skill_ids")
            if len(ranked) != expected_depth or len(set(ranked)) != expected_depth:
                raise ValueError(
                    f"{label} index={index}: expected {expected_depth} unique ranked skills, "
                    f"found {len(ranked)} rows/{len(set(ranked))} unique"
                )
            if not isinstance(gold, list) or not gold or not all(isinstance(x, str) for x in gold):
                raise ValueError(f"{label} index={index}: invalid gold_skill_ids")
            if len(set(gold)) != len(gold):
                raise ValueError(f"{label} index={index}: duplicate gold skill")
            if row.get("target_skill_count") != len(gold):
                raise ValueError(f"{label} index={index}: target_skill_count differs from gold")
            predicted = row.get("predicted_skill_ids")
            if not isinstance(predicted, list) or not predicted or predicted[0] != ranked[0]:
                raise ValueError(f"{label} index={index}: top-1 prediction differs from ranked list")
            scores = _score_map(row, ranked, label=label, index=index)
            score_modes.add("full" if scores is not None else "none")
            by_index[index] = row
        if len(score_modes) != 1:
            raise ValueError(f"{label}: mixed ranked-score availability: {sorted(score_modes)}")
        indexed[label] = by_index
        score_availability[label] = next(iter(score_modes))

    reference_indices = sorted(indexed[reference_label])
    reference = indexed[reference_label]
    for label, by_index in indexed.items():
        if sorted(by_index) != reference_indices:
            raise ValueError(f"{label}: prediction indices differ from {reference_label}")
        for index in reference_indices:
            for field in ALIGNMENT_FIELDS:
                if by_index[index].get(field) != reference[index].get(field):
                    raise ValueError(
                        f"{label} index={index}: {field} differs from {reference_label}"
                    )
    return reference_indices, indexed, score_availability


def analyze_system_row(
    row: Mapping[str, Any],
    *,
    cutoffs: Sequence[int],
    label: str,
) -> dict[str, Any]:
    index = int(row["index"])
    ranked: list[str] = row["ranked_skill_ids"]
    gold: list[str] = row["gold_skill_ids"]
    gold_set = set(gold)
    gold_ranks = sorted(
        rank for skill_id in gold if (rank := rank_of(skill_id, ranked)) is not None
    )
    nearest_gold_rank = gold_ranks[0] if gold_ranks else None
    recalls = {
        f"@{cutoff}": len(gold_set.intersection(ranked[:cutoff])) / len(gold)
        for cutoff in cutoffs
    }
    result: dict[str, Any] = {
        "top1_skill_id": ranked[0],
        "top1_correct": ranked[0] in gold_set,
        "gold_ranks": gold_ranks,
        "nearest_gold_rank": nearest_gold_rank,
        "nearest_gold_rank_bucket": rank_bucket(nearest_gold_rank),
        "gold_found_top50": len(gold_ranks),
        "macro_recall_contribution": recalls,
        "score_margins": None,
    }

    scores = _score_map(row, ranked, label=label, index=index)
    if scores is not None:
        present_gold = [scores[skill_id] for skill_id in gold if skill_id in scores]
        nongold = [scores[skill_id] for skill_id in ranked if skill_id not in gold_set]
        best_gold = max(present_gold) if present_gold else None
        best_nongold = max(nongold) if nongold else None
        result["score_margins"] = {
            "score_semantics": "max completion CE log-prob; higher is better",
            "top1_score": scores[ranked[0]],
            "top2_score": scores[ranked[1]],
            "top1_minus_top2": scores[ranked[0]] - scores[ranked[1]],
            "best_gold_score": best_gold,
            "best_nongold_score": best_nongold,
            "best_gold_minus_best_nongold": (
                best_gold - best_nongold
                if best_gold is not None and best_nongold is not None
                else None
            ),
            "top1_minus_best_gold": (
                scores[ranked[0]] - best_gold if best_gold is not None else None
            ),
        }
    return result


def summarize_subset(
    rows: Sequence[dict[str, Any]],
    *,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    bucket_counts = Counter(
        row["system"]["nearest_gold_rank_bucket"] for row in rows
    )
    return {
        "rows": len(rows),
        "total_gold": sum(row["target_skill_count"] for row in rows),
        "top1_hit_queries": sum(row["system"]["top1_correct"] for row in rows),
        "macro_recall": {
            f"@{cutoff}": (
                math.fsum(
                    row["system"]["macro_recall_contribution"][f"@{cutoff}"]
                    for row in rows
                )
                / len(rows)
                if rows
                else None
            )
            for cutoff in cutoffs
        },
        "nearest_gold_rank_buckets": {
            bucket: bucket_counts.get(bucket, 0) for bucket in RANK_BUCKETS
        },
    }


def summarize_system(
    per_query: Sequence[dict[str, Any]],
    *,
    label: str,
    cutoffs: Sequence[int],
    score_availability: str,
) -> dict[str, Any]:
    projected = [
        {
            "target_skill_count": row["target_skill_count"],
            "system": row["systems"][label],
        }
        for row in per_query
    ]
    subsets = {
        "overall": projected,
        "single": [row for row in projected if row["target_skill_count"] == 1],
        "multi": [row for row in projected if row["target_skill_count"] > 1],
    }
    margins = [row["systems"][label]["score_margins"] for row in per_query]
    scored = [margin for margin in margins if margin is not None]
    wrong_with_gold = [
        row["systems"][label]["score_margins"]
        for row in per_query
        if not row["systems"][label]["top1_correct"]
        and row["systems"][label]["score_margins"] is not None
        and row["systems"][label]["score_margins"]["best_gold_score"] is not None
    ]
    return {
        "score_availability": score_availability,
        "subsets": {
            subset: summarize_subset(rows, cutoffs=cutoffs)
            for subset, rows in subsets.items()
        },
        "score_margin_summary": {
            "top1_minus_top2": numeric_summary(
                margin["top1_minus_top2"] for margin in scored
            ),
            "best_gold_minus_best_nongold_when_gold_in_top50": numeric_summary(
                margin["best_gold_minus_best_nongold"]
                for margin in scored
                if margin["best_gold_minus_best_nongold"] is not None
            ),
            "top1_minus_best_gold_when_top1_wrong_and_gold_in_top50": numeric_summary(
                margin["top1_minus_best_gold"] for margin in wrong_with_gold
            ),
        },
    }


def compare_query(
    reference_row: Mapping[str, Any],
    system_row: Mapping[str, Any],
    *,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    reference_correct = bool(reference_row["top1_correct"])
    system_correct = bool(system_row["top1_correct"])
    if reference_correct and system_correct:
        flip = "both_correct"
    elif reference_correct:
        flip = "reference_correct_system_wrong"
    elif system_correct:
        flip = "reference_wrong_system_correct"
    else:
        flip = "both_wrong"

    reference_ranked = reference_row["ranked_skill_ids"]
    system_ranked = system_row["ranked_skill_ids"]
    overlap: dict[str, Any] = {}
    for cutoff in cutoffs:
        reference_set = set(reference_ranked[:cutoff])
        system_set = set(system_ranked[:cutoff])
        intersection = len(reference_set.intersection(system_set))
        union = len(reference_set.union(system_set))
        overlap[f"@{cutoff}"] = {
            "intersection_count": intersection,
            "reference_coverage": intersection / cutoff,
            "jaccard": intersection / union,
        }
    reference_top1_rank = rank_of(reference_ranked[0], system_ranked)
    system_top1_rank = rank_of(system_ranked[0], reference_ranked)
    return {
        "top1_flip": flip,
        "same_top1": reference_ranked[0] == system_ranked[0],
        "reference_top1_rank_in_system": reference_top1_rank,
        "reference_top1_rank_bucket_in_system": rank_bucket(reference_top1_rank),
        "system_top1_rank_in_reference": system_top1_rank,
        "system_top1_rank_bucket_in_reference": rank_bucket(system_top1_rank),
        "candidate_overlap": overlap,
        "macro_recall_delta": {
            f"@{cutoff}": system_row["macro_recall_contribution"][f"@{cutoff}"]
            - reference_row["macro_recall_contribution"][f"@{cutoff}"]
            for cutoff in cutoffs
        },
    }


def summarize_comparison(
    per_query: Sequence[dict[str, Any]],
    *,
    label: str,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    comparisons = [row["comparisons_to_reference"][label] for row in per_query]
    flip_counts = Counter(row["top1_flip"] for row in comparisons)
    critical_rows = [
        row
        for row in per_query
        if row["comparisons_to_reference"][label]["top1_flip"]
        == "reference_correct_system_wrong"
    ]
    gained_rows = [
        row
        for row in per_query
        if row["comparisons_to_reference"][label]["top1_flip"]
        == "reference_wrong_system_correct"
    ]
    reference_top1_buckets = Counter(
        row["reference_top1_rank_bucket_in_system"] for row in comparisons
    )
    system_top1_buckets = Counter(
        row["system_top1_rank_bucket_in_reference"] for row in comparisons
    )
    overlap_summary: dict[str, Any] = {}
    for cutoff in cutoffs:
        key = f"@{cutoff}"
        overlap_summary[key] = {
            field: numeric_summary(row["candidate_overlap"][key][field] for row in comparisons)
            for field in ("intersection_count", "reference_coverage", "jaccard")
        }
    critical_margins = [
        row["systems"][label]["score_margins"]["top1_minus_best_gold"]
        for row in critical_rows
        if row["systems"][label]["score_margins"] is not None
        and row["systems"][label]["score_margins"]["best_gold_score"] is not None
    ]
    critical_gold_buckets = Counter(
        row["systems"][label]["nearest_gold_rank_bucket"] for row in critical_rows
    )
    critical_reference_top1_buckets = Counter(
        row["comparisons_to_reference"][label][
            "reference_top1_rank_bucket_in_system"
        ]
        for row in critical_rows
    )
    critical_cases = []
    for row in critical_rows:
        system = row["systems"][label]
        comparison = row["comparisons_to_reference"][label]
        margin = system["score_margins"]
        critical_cases.append(
            {
                "index": row["index"],
                "target_skill_count": row["target_skill_count"],
                "gold_skill_ids": row["gold_skill_ids"],
                "system_top1_skill_id": system["top1_skill_id"],
                "nearest_gold_rank": system["nearest_gold_rank"],
                "nearest_gold_rank_bucket": system["nearest_gold_rank_bucket"],
                "reference_top1_rank_in_system": comparison[
                    "reference_top1_rank_in_system"
                ],
                "reference_top1_rank_bucket_in_system": comparison[
                    "reference_top1_rank_bucket_in_system"
                ],
                "top1_minus_best_gold_margin": (
                    margin["top1_minus_best_gold"]
                    if margin is not None and margin["best_gold_score"] is not None
                    else None
                ),
            }
        )
    return {
        "rows": len(comparisons),
        "top1_same_queries": sum(row["same_top1"] for row in comparisons),
        "top1_flip_counts": {
            key: flip_counts.get(key, 0)
            for key in (
                "both_correct",
                "reference_correct_system_wrong",
                "reference_wrong_system_correct",
                "both_wrong",
            )
        },
        "reference_top1_rank_buckets_in_system": {
            bucket: reference_top1_buckets.get(bucket, 0) for bucket in RANK_BUCKETS
        },
        "system_top1_rank_buckets_in_reference": {
            bucket: system_top1_buckets.get(bucket, 0) for bucket in RANK_BUCKETS
        },
        "candidate_overlap": overlap_summary,
        "macro_recall_delta": {
            f"@{cutoff}": math.fsum(
                row["macro_recall_delta"][f"@{cutoff}"] for row in comparisons
            )
            / len(comparisons)
            for cutoff in cutoffs
        },
        "reference_correct_system_wrong_indices": [row["index"] for row in critical_rows],
        "reference_wrong_system_correct_indices": [row["index"] for row in gained_rows],
        "critical_system_gold_rank_buckets": {
            bucket: critical_gold_buckets.get(bucket, 0) for bucket in RANK_BUCKETS
        },
        "critical_reference_top1_rank_buckets_in_system": {
            bucket: critical_reference_top1_buckets.get(bucket, 0)
            for bucket in RANK_BUCKETS
        },
        "critical_top1_minus_best_gold_margin": numeric_summary(critical_margins),
        "critical_cases": critical_cases,
    }


def summarize_pairwise_rank_overlap(
    indexed: Mapping[str, Mapping[int, Mapping[str, Any]]],
    indices: Sequence[int],
    *,
    cutoffs: Sequence[int],
    score_availability: Mapping[str, str],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for left_label, right_label in combinations(indexed, 2):
        overlaps: dict[str, dict[str, list[float]]] = {
            f"@{cutoff}": {
                "intersection_count": [],
                "shared_fraction": [],
                "jaccard": [],
            }
            for cutoff in cutoffs
        }
        same_top1 = 0
        rank_correlations: list[float] = []
        score_correlations: list[float] = []
        for index in indices:
            left_row = indexed[left_label][index]
            right_row = indexed[right_label][index]
            left_ranked = left_row["ranked_skill_ids"]
            right_ranked = right_row["ranked_skill_ids"]
            same_top1 += left_ranked[0] == right_ranked[0]
            for cutoff in cutoffs:
                key = f"@{cutoff}"
                left_set = set(left_ranked[:cutoff])
                right_set = set(right_ranked[:cutoff])
                intersection = len(left_set.intersection(right_set))
                union = len(left_set.union(right_set))
                overlaps[key]["intersection_count"].append(float(intersection))
                overlaps[key]["shared_fraction"].append(intersection / cutoff)
                overlaps[key]["jaccard"].append(intersection / union)

            common = set(left_ranked).intersection(right_ranked)
            ordered_common = sorted(common)
            rank_correlation = pearson_correlation(
                [float(rank_of(skill_id, left_ranked)) for skill_id in ordered_common],
                [float(rank_of(skill_id, right_ranked)) for skill_id in ordered_common],
            )
            if rank_correlation is not None:
                rank_correlations.append(rank_correlation)
            if (
                score_availability[left_label] == "full"
                and score_availability[right_label] == "full"
            ):
                left_scores = left_row["ranked_skill_scores"]
                right_scores = right_row["ranked_skill_scores"]
                score_correlation = pearson_correlation(
                    [float(left_scores[skill_id]) for skill_id in ordered_common],
                    [float(right_scores[skill_id]) for skill_id in ordered_common],
                )
                if score_correlation is not None:
                    score_correlations.append(score_correlation)
        key = f"{left_label}__vs__{right_label}"
        summaries[key] = {
            "left_label": left_label,
            "right_label": right_label,
            "rows": len(indices),
            "same_top1_queries": same_top1,
            "candidate_overlap": {
                cutoff: {
                    field: numeric_summary(values) for field, values in fields.items()
                }
                for cutoff, fields in overlaps.items()
            },
            "common_top50_rank_spearman": numeric_summary(rank_correlations),
            "common_top50_score_pearson": numeric_summary(score_correlations),
        }
    return summaries


def summarize_critical_overlap(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    critical_sets = {
        label: set(summary["reference_correct_system_wrong_indices"])
        for label, summary in comparisons.items()
    }
    if not critical_sets:
        return {
            "comparison_systems": [],
            "lost_by_all_systems": [],
            "lost_by_at_least_two_systems": [],
            "loss_frequency_by_index": {},
        }
    frequency = Counter(index for indices in critical_sets.values() for index in indices)
    system_count = len(critical_sets)
    return {
        "comparison_systems": list(critical_sets),
        "lost_by_all_systems": sorted(
            index for index, count in frequency.items() if count == system_count
        ),
        "lost_by_at_least_two_systems": sorted(
            index for index, count in frequency.items() if count >= 2
        ),
        "loss_frequency_by_index": {
            str(index): frequency[index] for index in sorted(frequency)
        },
    }


def verify_existing_metrics(
    system_summaries: Mapping[str, dict[str, Any]],
    metric_paths: Mapping[str, Path],
    *,
    cutoffs: Sequence[int],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for label, path in metric_paths.items():
        if label not in system_summaries:
            raise ValueError(f"metrics label has no prediction system: {label}")
        payload = read_json(path)
        observed = system_summaries[label]["subsets"]["overall"]["macro_recall"]
        cutoff_checks: dict[str, Any] = {}
        for cutoff in cutoffs:
            key = f"@{cutoff}"
            if key not in payload.get("overall", {}):
                continue
            expected = float(payload["overall"][key]["macro_recall"])
            actual = float(observed[key])
            cutoff_checks[key] = {
                "expected": expected,
                "actual": actual,
                "absolute_error": abs(actual - expected),
                "passed": abs(actual - expected) <= tolerance,
            }
        if not cutoff_checks:
            raise ValueError(f"no requested cutoffs found in existing metrics: {path}")
        checks[label] = {
            "path": str(path),
            "cutoffs": cutoff_checks,
            "passed": all(row["passed"] for row in cutoff_checks.values()),
        }
    if not all(row["passed"] for row in checks.values()):
        raise ValueError(f"independent metric recomputation failed: {checks}")
    return checks


def analyze(
    systems: Mapping[str, list[dict[str, Any]]],
    *,
    reference_label: str,
    expected_rows: int,
    expected_depth: int,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    input_paths: Mapping[str, Path] | None = None,
    metric_paths: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not cutoffs or tuple(sorted(set(cutoffs))) != tuple(cutoffs):
        raise ValueError(f"cutoffs must be strictly increasing and unique: {cutoffs}")
    if max(cutoffs) > expected_depth:
        raise ValueError("largest cutoff exceeds expected rank depth")
    indices, indexed, score_availability = validate_and_align(
        systems,
        reference_label=reference_label,
        expected_rows=expected_rows,
        expected_depth=expected_depth,
    )

    per_query: list[dict[str, Any]] = []
    for index in indices:
        reference_raw = indexed[reference_label][index]
        system_rows = {
            label: analyze_system_row(row, cutoffs=cutoffs, label=label)
            for label, rows_by_index in indexed.items()
            for row in (rows_by_index[index],)
        }
        for label, system_row in system_rows.items():
            system_row["ranked_skill_ids"] = indexed[label][index]["ranked_skill_ids"]
        comparisons = {
            label: compare_query(
                system_rows[reference_label], system_row, cutoffs=cutoffs
            )
            for label, system_row in system_rows.items()
            if label != reference_label
        }
        for system_row in system_rows.values():
            del system_row["ranked_skill_ids"]
        per_query.append(
            {
                "index": index,
                "source_index": reference_raw.get("source_index"),
                "query": reference_raw["query"],
                "target_skill_count": reference_raw["target_skill_count"],
                "gold_skill_ids": reference_raw["gold_skill_ids"],
                "systems": system_rows,
                "comparisons_to_reference": comparisons,
            }
        )

    system_summaries = {
        label: summarize_system(
            per_query,
            label=label,
            cutoffs=cutoffs,
            score_availability=score_availability[label],
        )
        for label in systems
    }
    comparison_summaries = {
        label: summarize_comparison(per_query, label=label, cutoffs=cutoffs)
        for label in systems
        if label != reference_label
    }
    pairwise_rank_overlap = summarize_pairwise_rank_overlap(
        indexed,
        indices,
        cutoffs=cutoffs,
        score_availability=score_availability,
    )
    critical_overlap = summarize_critical_overlap(comparison_summaries)
    metric_checks = verify_existing_metrics(
        system_summaries, metric_paths or {}, cutoffs=cutoffs
    )

    conservation_checks: dict[str, bool] = {}
    for label, summary in system_summaries.items():
        for subset, subset_summary in summary["subsets"].items():
            conservation_checks[f"{label}.{subset}.rank_buckets"] = (
                sum(subset_summary["nearest_gold_rank_buckets"].values())
                == subset_summary["rows"]
            )
    for label, summary in comparison_summaries.items():
        conservation_checks[f"{label}.top1_flips"] = (
            sum(summary["top1_flip_counts"].values()) == expected_rows
        )
        conservation_checks[f"{label}.reference_top1_rank_buckets"] = (
            sum(summary["reference_top1_rank_buckets_in_system"].values())
            == expected_rows
        )
    if not all(conservation_checks.values()):
        raise ValueError(f"forensic conservation check failed: {conservation_checks}")

    payload = {
        "artifact_format": "legacy_core75_query_rank_forensics_v1",
        "protocol": {
            "reference_label": reference_label,
            "query_count": expected_rows,
            "rank_depth": expected_depth,
            "cutoffs": list(cutoffs),
            "evaluation": (
                "Official core75; beam=50; beam-order; macro recall; "
                "A800 ranked scores are max completion CE log-prob"
            ),
        },
        "inputs": {
            label: str(path) for label, path in (input_paths or {}).items()
        },
        "systems": system_summaries,
        "comparisons_to_reference": comparison_summaries,
        "pairwise_rank_overlap": pairwise_rank_overlap,
        "critical_overlap": critical_overlap,
        "verification": {
            "row_alignment": True,
            "ranked_list_depth_and_uniqueness": True,
            "score_coverage_and_monotonicity": True,
            "existing_metric_checks": metric_checks,
            "conservation_checks": conservation_checks,
            "passed": True,
        },
    }
    return payload, per_query


def _percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.4f}"


def _decimal(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    protocol = payload["protocol"]
    reference_label = protocol["reference_label"]
    cutoffs = protocol["cutoffs"]
    systems = payload["systems"]
    comparisons = payload["comparisons_to_reference"]
    pairwise = payload["pairwise_rank_overlap"]
    lines = [
        "# Legacy Official core75 query-rank forensics",
        "",
        (
            f"Conditions: Official core75={protocol['query_count']} queries; "
            f"beam={protocol['rank_depth']}; beam-order; macro recall; "
            "A800 scores=max completion CE log-prob (higher is better)."
        ),
        "",
        "## System summaries",
        "",
        "| system | R@1 | R@5 | R@10 | R@50 | top-1 hit queries | nearest gold rank buckets: 1 / 2-5 / 6-10 / 11-50 / missing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in systems.items():
        overall = summary["subsets"]["overall"]
        recalls = overall["macro_recall"]
        buckets = overall["nearest_gold_rank_buckets"]
        lines.append(
            f"| `{label}` | "
            f"{_percentage(recalls.get('@1'))} | {_percentage(recalls.get('@5'))} | "
            f"{_percentage(recalls.get('@10'))} | {_percentage(recalls.get('@50'))} | "
            f"{overall['top1_hit_queries']} | "
            f"{buckets['rank1']} / {buckets['rank2_5']} / {buckets['rank6_10']} / "
            f"{buckets['rank11_50']} / {buckets['missing_top50']} |"
        )

    lines.extend(
        [
            "",
            f"## Paired comparison against `{reference_label}`",
            "",
            "| system | delta R@1/R@10/R@50 (points) | same top-1 | RTX correct → system wrong | RTX wrong → system correct | mean Top10/Top50 reference coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, summary in comparisons.items():
        delta = summary["macro_recall_delta"]
        flips = summary["top1_flip_counts"]
        overlap = summary["candidate_overlap"]
        lines.append(
            f"| `{label}` | "
            f"{100 * delta['@1']:+.4f}/{100 * delta['@10']:+.4f}/{100 * delta['@50']:+.4f} | "
            f"{summary['top1_same_queries']} | "
            f"{flips['reference_correct_system_wrong']} | "
            f"{flips['reference_wrong_system_correct']} | "
            f"{100 * overlap['@10']['reference_coverage']['mean']:.2f}%/"
            f"{100 * overlap['@50']['reference_coverage']['mean']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Pairwise trajectory overlap",
            "",
            "| pair | same top-1 | mean shared Top10/Top50 | common Top50 rank Spearman mean | common Top50 score Pearson mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for summary in pairwise.values():
        overlap = summary["candidate_overlap"]
        rank_correlation = summary["common_top50_rank_spearman"]
        score_correlation = summary["common_top50_score_pearson"]
        lines.append(
            f"| `{summary['left_label']}` / `{summary['right_label']}` | "
            f"{summary['same_top1_queries']} | "
            f"{100 * overlap['@10']['shared_fraction']['mean']:.2f}%/"
            f"{100 * overlap['@50']['shared_fraction']['mean']:.2f}% | "
            f"{_decimal(rank_correlation['mean'])} | "
            f"{_decimal(score_correlation['mean'])} |"
        )

    scored_labels = [
        label for label, summary in systems.items() if summary["score_availability"] == "full"
    ]
    lines.extend(
        [
            "",
            "## A800 score margins",
            "",
            "`best_gold_minus_best_nongold` 为正表示 gold 排在所有 non-gold 之前。"
            "`top1_minus_best_gold` 只统计 top-1 错且 gold 在 Top50 的 query，越小越接近翻转。",
            "",
            "| system | best-gold minus best-nongold mean/median | wrong top-1 minus best-gold mean/median | wrong queries with gold in Top50 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in scored_labels:
        margin = systems[label]["score_margin_summary"]
        gold = margin["best_gold_minus_best_nongold_when_gold_in_top50"]
        wrong = margin["top1_minus_best_gold_when_top1_wrong_and_gold_in_top50"]
        lines.append(
            f"| `{label}` | {_decimal(gold['mean'])}/{_decimal(gold['median'])} | "
            f"{_decimal(wrong['mean'])}/{_decimal(wrong['median'])} | {wrong['count']} |"
        )

    lines.extend(["", "## Critical RTX-correct flips", ""])
    for label, summary in comparisons.items():
        indices = set(summary["reference_correct_system_wrong_indices"])
        buckets = summary["critical_system_gold_rank_buckets"]
        margin = summary["critical_top1_minus_best_gold_margin"]
        lines.append(
            f"- `{label}`: {len(indices)} RTX-correct queries lost; exact indices={sorted(indices)}; "
            f"system nearest-gold buckets (1/2-5/6-10/11-50/missing)="
            f"{buckets['rank1']}/{buckets['rank2_5']}/{buckets['rank6_10']}/"
            f"{buckets['rank11_50']}/{buckets['missing_top50']}; "
            f"critical margin count={margin['count']}, median={_decimal(margin['median'])}."
        )
    critical_overlap = payload["critical_overlap"]
    lines.append(
        f"- Lost by all A800 systems: {critical_overlap['lost_by_all_systems']}; "
        f"lost by at least two A800 systems: "
        f"{critical_overlap['lost_by_at_least_two_systems']}."
    )
    lines.extend(
        [
            "",
            "Verification: row alignment, Top50 uniqueness, A800 score coverage/monotonicity, "
            "independent metric recomputation, rank-bucket conservation, and flip conservation all passed.",
        ]
    )
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def command_analyze(args: argparse.Namespace) -> None:
    input_paths = parse_label_paths(args.system, "--system")
    metric_paths = parse_label_paths(args.metrics, "--metrics")
    systems = {label: read_predictions(path) for label, path in input_paths.items()}
    payload, per_query = analyze(
        systems,
        reference_label=args.reference_label,
        expected_rows=args.expected_rows,
        expected_depth=args.expected_depth,
        cutoffs=tuple(args.cutoff),
        input_paths=input_paths,
        metric_paths=metric_paths,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.output_dir / "query_rank_forensics.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        args.output_dir / "per_query.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_query),
    )
    atomic_write_text(
        args.output_dir / "query_rank_forensics.md", render_markdown(payload)
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "reference_label": args.reference_label,
                "systems": list(input_paths),
                "verification_passed": payload["verification"]["passed"],
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--metrics", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--expected-rows", type=int, default=75)
    parser.add_argument("--expected-depth", type=int, default=50)
    parser.add_argument("--cutoff", type=int, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cutoff is None:
        args.cutoff = list(DEFAULT_CUTOFFS)
    command_analyze(args)


if __name__ == "__main__":
    main()
