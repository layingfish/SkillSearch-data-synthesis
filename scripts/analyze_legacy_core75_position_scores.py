#!/usr/bin/env python3
"""Analyze per-position fixed-pool scores and constrained position weights."""

from __future__ import annotations

import argparse
import json
import math
import platform
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.analyze_legacy_core75_fixed_pool import (
    analyze as analyze_fixed_pool,
    atomic_write_text,
    file_sha256,
    load_system_shards,
    parse_shards,
    rank_of_any,
    recall_at,
)
from scripts.analyze_legacy_core75_query_ranks import (
    numeric_summary,
    pearson_correlation,
)


COMPONENT_LABELS = (
    "code_position_0",
    "code_position_1",
    "code_position_2",
    "code_position_3",
    "stop_control_1",
)
CUTOFFS = (1, 5, 10, 50)
GRID_RAW_VALUES = (0.5, 0.75, 1.0, 1.25, 1.5)
DEEP_GATES = {"@10": 0.69, "@50": 0.75}
R1_GATE = 0.28368253968253965 - 0.01


def normalized_weight_grid() -> list[tuple[float, ...]]:
    weights = {
        tuple(round(value / (sum(raw) / len(raw)), 8) for value in raw)
        for raw in product(GRID_RAW_VALUES, repeat=len(COMPONENT_LABELS))
    }
    return sorted(weights)


def validate_component_rows(
    label: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    atol: float = 1e-6,
) -> dict[str, Any]:
    maximum_error = 0.0
    candidate_count = 0
    for row in rows:
        index = row["index"]
        ranked = row["fixed_ranked_skill_ids"]
        totals = row["fixed_ranked_skill_scores"]
        components_by_skill = row.get("fixed_ranked_skill_score_components")
        if not isinstance(components_by_skill, dict):
            raise ValueError(f"{label} index={index}: score components are missing")
        if set(components_by_skill) != set(ranked):
            raise ValueError(f"{label} index={index}: component keys differ from ranking")
        reconstructed: dict[str, float] = {}
        for skill_id in ranked:
            components = components_by_skill[skill_id]
            if (
                not isinstance(components, list)
                or len(components) != len(COMPONENT_LABELS)
                or not all(math.isfinite(float(value)) for value in components)
            ):
                raise ValueError(
                    f"{label} index={index} candidate={skill_id}: invalid components"
                )
            reconstructed[skill_id] = sum(float(value) for value in components)
            error = abs(reconstructed[skill_id] - float(totals[skill_id]))
            maximum_error = max(maximum_error, error)
            candidate_count += 1
        if maximum_error > atol:
            raise ValueError(
                f"{label} index={index}: component sum error exceeds atol={atol}; "
                f"maximum={maximum_error}"
            )
        replay = sorted(
            ranked,
            key=lambda skill_id: reconstructed[skill_id],
            reverse=True,
        )
        if replay != list(ranked):
            raise ValueError(f"{label} index={index}: uniform component replay changed rank")
    return {
        "rows": len(rows),
        "candidates": candidate_count,
        "component_count": len(COMPONENT_LABELS),
        "maximum_sum_absolute_error": maximum_error,
        "atol": atol,
        "uniform_rank_exact": True,
    }


def baseline_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        f"@{cutoff}": math.fsum(
            recall_at(row["fixed_ranked_skill_ids"], set(row["gold_skill_ids"]), cutoff)
            for row in rows
        )
        / len(rows)
        for cutoff in CUTOFFS
    }


def grid_query_metrics(
    rows: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if weights.ndim != 2 or weights.shape[1] != len(COMPONENT_LABELS):
        raise ValueError("weights must have shape [count, 5]")
    query_recall = np.zeros(
        (len(rows), weights.shape[0], len(CUTOFFS)), dtype=np.float64
    )
    query_correct = np.zeros((len(rows), weights.shape[0]), dtype=np.bool_)
    for query_position, row in enumerate(rows):
        ranked = row["fixed_ranked_skill_ids"]
        components_by_skill = row["fixed_ranked_skill_score_components"]
        components = np.asarray(
            [components_by_skill[skill_id] for skill_id in ranked], dtype=np.float64
        )
        scores = np.zeros((components.shape[0], weights.shape[0]), dtype=np.float64)
        for component_position in range(len(COMPONENT_LABELS)):
            scores += (
                components[:, component_position, None]
                * weights[None, :, component_position]
            )
        order = np.argsort(-scores, axis=0, kind="stable")
        gold = set(row["gold_skill_ids"])
        gold_mask = np.asarray([skill_id in gold for skill_id in ranked], dtype=np.bool_)
        ranked_gold = gold_mask[order]
        query_correct[query_position] = ranked_gold[0]
        for cutoff_position, cutoff in enumerate(CUTOFFS):
            query_recall[query_position, :, cutoff_position] = (
                ranked_gold[:cutoff].sum(axis=0) / len(gold)
            )
    return query_recall, query_correct


def metric_dict(values: Sequence[float]) -> dict[str, float]:
    return {
        f"@{cutoff}": float(values[position])
        for position, cutoff in enumerate(CUTOFFS)
    }


def select_weight_index(
    metrics: np.ndarray,
    weights: np.ndarray,
    *,
    deep_gates: Mapping[str, float] = DEEP_GATES,
) -> tuple[int, bool]:
    cutoff_positions = {f"@{cutoff}": position for position, cutoff in enumerate(CUTOFFS)}
    eligible = np.ones(metrics.shape[0], dtype=np.bool_)
    for cutoff, threshold in deep_gates.items():
        eligible &= metrics[:, cutoff_positions[cutoff]] >= float(threshold) - 1e-12
    candidate_indices = np.flatnonzero(eligible)
    deep_gate_available = bool(candidate_indices.size)
    if not deep_gate_available:
        candidate_indices = np.arange(metrics.shape[0])
    distances = np.linalg.norm(weights - 1.0, axis=1)
    selected = min(
        (int(index) for index in candidate_indices),
        key=lambda index: (
            -float(metrics[index, 0]),
            -float(metrics[index, 2]),
            -float(metrics[index, 3]),
            float(distances[index]),
            tuple(float(value) for value in weights[index]),
        ),
    )
    return selected, deep_gate_available


def weighted_rank(
    row: Mapping[str, Any], weights: Sequence[float]
) -> tuple[list[str], dict[str, float]]:
    baseline = row["fixed_ranked_skill_ids"]
    components = row["fixed_ranked_skill_score_components"]
    scores = {
        skill_id: sum(
            float(weight) * float(component)
            for weight, component in zip(weights, components[skill_id])
        )
        for skill_id in baseline
    }
    ranked = sorted(baseline, key=lambda skill_id: scores[skill_id], reverse=True)
    return ranked, scores


def selected_weight_payload(
    rows: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
    query_recall: np.ndarray,
    query_correct: np.ndarray,
) -> tuple[dict[str, Any], int]:
    metrics = query_recall.mean(axis=0)
    selected, deep_gate_available = select_weight_index(metrics, weights)
    baseline_correct = np.asarray(
        [row["fixed_ranked_skill_ids"][0] in set(row["gold_skill_ids"]) for row in rows],
        dtype=np.bool_,
    )
    selected_correct = query_correct[:, selected]
    selected_metrics = metric_dict(metrics[selected])
    replay_contributions = {f"@{cutoff}": [] for cutoff in CUTOFFS}
    replay_correct: list[bool] = []
    query_rankings: list[list[str]] = []
    for row in rows:
        replay_ranked, _ = weighted_rank(row, weights[selected])
        query_rankings.append(replay_ranked)
        gold = set(row["gold_skill_ids"])
        replay_correct.append(replay_ranked[0] in gold)
        for cutoff in CUTOFFS:
            replay_contributions[f"@{cutoff}"].append(
                recall_at(replay_ranked, gold, cutoff)
            )
    replay_metrics = {
        f"@{cutoff}": math.fsum(replay_contributions[f"@{cutoff}"]) / len(rows)
        for cutoff in CUTOFFS
    }
    if replay_correct != selected_correct.tolist() or any(
        not math.isclose(
            replay_metrics[f"@{cutoff}"],
            selected_metrics[f"@{cutoff}"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for cutoff in CUTOFFS
    ):
        raise ValueError("selected weight NumPy metrics differ from Python rank replay")
    mention_categories = {
        key: {"mentions": 0, "macro_weight_sum": 0.0}
        for key in (
            "both_rank1",
            "baseline_rank1_weighted_not",
            "baseline_not_weighted_rank1",
            "neither_rank1",
        )
    }
    total_mentions = 0
    for row, replay_ranked in zip(rows, query_rankings):
        baseline_top1 = row["fixed_ranked_skill_ids"][0]
        weighted_top1 = replay_ranked[0]
        gold_ids = row["gold_skill_ids"]
        macro_weight = 1.0 / len(gold_ids)
        for skill_id in gold_ids:
            baseline_rank1 = skill_id == baseline_top1
            weighted_rank1 = skill_id == weighted_top1
            if baseline_rank1 and weighted_rank1:
                category = "both_rank1"
            elif baseline_rank1:
                category = "baseline_rank1_weighted_not"
            elif weighted_rank1:
                category = "baseline_not_weighted_rank1"
            else:
                category = "neither_rank1"
            mention_categories[category]["mentions"] += 1
            mention_categories[category]["macro_weight_sum"] += macro_weight
            total_mentions += 1
    weighted_gain = mention_categories["baseline_not_weighted_rank1"][
        "macro_weight_sum"
    ]
    baseline_loss = mention_categories["baseline_rank1_weighted_not"][
        "macro_weight_sum"
    ]
    reconstructed_delta = (weighted_gain - baseline_loss) / len(rows)
    observed_delta = selected_metrics["@1"] - baseline_metrics(rows)["@1"]
    mention_weight_total = math.fsum(
        category["macro_weight_sum"] for category in mention_categories.values()
    )
    if (
        total_mentions != sum(len(row["gold_skill_ids"]) for row in rows)
        or not math.isclose(
            mention_weight_total, float(len(rows)), rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            reconstructed_delta, observed_delta, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("selected weight gold-mention Top-1 conservation failed")
    payload = {
        "weights": [float(value) for value in weights[selected]],
        "metrics": selected_metrics,
        "deep_gate_available": deep_gate_available,
        "selection_status": (
            "deep_gate_candidate"
            if deep_gate_available
            else "best_effort_no_deep_gate_candidate"
        ),
        "eligible_candidate": deep_gate_available,
        "deep_gate_passed": (
            selected_metrics["@10"] >= DEEP_GATES["@10"] - 1e-12
            and selected_metrics["@50"] >= DEEP_GATES["@50"] - 1e-12
        ),
        "full_gate_passed": (
            selected_metrics["@1"] >= R1_GATE - 1e-12
            and selected_metrics["@10"] >= DEEP_GATES["@10"] - 1e-12
            and selected_metrics["@50"] >= DEEP_GATES["@50"] - 1e-12
        ),
        "query_top1_transitions": {
            "both_correct": int(np.sum(baseline_correct & selected_correct)),
            "baseline_correct_weighted_wrong": int(
                np.sum(baseline_correct & ~selected_correct)
            ),
            "baseline_wrong_weighted_correct": int(
                np.sum(~baseline_correct & selected_correct)
            ),
            "both_wrong": int(np.sum(~baseline_correct & ~selected_correct)),
        },
        "gold_mention_top1_transitions": {
            "total_mentions": total_mentions,
            "macro_weight_total": mention_weight_total,
            "categories": mention_categories,
            "macro_r1_delta_reconstructed": reconstructed_delta,
            "macro_r1_delta_observed": observed_delta,
            "passed": True,
        },
        "distance_from_uniform_l2": float(np.linalg.norm(weights[selected] - 1.0)),
        "python_rank_replay": True,
    }
    return payload, selected


def cross_validate_weights(
    rows: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
    query_recall: np.ndarray,
    *,
    folds: int = 5,
) -> dict[str, Any]:
    fold_rows = []
    for fold in range(folds):
        heldout_positions = [
            position
            for position, row in enumerate(rows)
            if int(row["row_position"]) % folds == fold
        ]
        train_positions = [
            position for position in range(len(rows)) if position not in heldout_positions
        ]
        train_metrics = query_recall[train_positions].mean(axis=0)
        selected, deep_gate_available = select_weight_index(train_metrics, weights)
        heldout_metrics = query_recall[heldout_positions, selected, :].mean(axis=0)
        fold_rows.append(
            {
                "fold": fold,
                "train_queries": len(train_positions),
                "heldout_queries": len(heldout_positions),
                "weights": [float(value) for value in weights[selected]],
                "train_metrics": metric_dict(train_metrics[selected]),
                "heldout_metrics": metric_dict(heldout_metrics),
                "train_deep_gate_available": deep_gate_available,
            }
        )
    aggregate = {
        f"@{cutoff}": math.fsum(
            fold["heldout_metrics"][f"@{cutoff}"] * fold["heldout_queries"]
            for fold in fold_rows
        )
        / len(rows)
        for cutoff in CUTOFFS
    }
    return {"folds": fold_rows, "aggregate_heldout_metrics": aggregate}


def descending_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(
        range(len(values)), key=lambda position: float(values[position]), reverse=True
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        value = float(values[order[start]])
        while end < len(order) and float(values[order[end]]) == value:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in order[start:end]:
            ranks[position] = average_rank
        start = end
    return ranks


def component_pair_agreement(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for component_position, component_label in enumerate(COMPONENT_LABELS):
        pearsons: list[float] = []
        spearmans: list[float] = []
        for reference, candidate in zip(reference_rows, candidate_rows):
            pool = reference["pool_skill_ids"]
            reference_values = [
                float(reference["fixed_ranked_skill_score_components"][skill_id][component_position])
                for skill_id in pool
            ]
            candidate_values = [
                float(candidate["fixed_ranked_skill_score_components"][skill_id][component_position])
                for skill_id in pool
            ]
            pearson = pearson_correlation(reference_values, candidate_values)
            spearman = pearson_correlation(
                descending_ranks(reference_values),
                descending_ranks(candidate_values),
            )
            if pearson is not None:
                pearsons.append(pearson)
            if spearman is not None:
                spearmans.append(spearman)
        summaries[component_label] = {
            "score_pearson": numeric_summary(pearsons),
            "rank_spearman": numeric_summary(spearmans),
        }
    return summaries


def component_delta_case(row: Mapping[str, Any]) -> dict[str, Any]:
    gold = set(row["gold_skill_ids"])
    ranked = row["fixed_ranked_skill_ids"]
    scores = row["fixed_ranked_skill_scores"]
    components = row["fixed_ranked_skill_score_components"]
    best_gold = max(gold, key=lambda skill_id: float(scores[skill_id]))
    best_nongold = next(skill_id for skill_id in ranked if skill_id not in gold)
    deltas = [
        float(gold_value) - float(nongold_value)
        for gold_value, nongold_value in zip(
            components[best_gold], components[best_nongold]
        )
    ]
    total_margin = float(scores[best_gold]) - float(scores[best_nongold])
    if not math.isclose(
        sum(deltas), total_margin, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"index={row['index']}: component deltas do not sum to margin")
    return {
        "index": row["index"],
        "target_skill_count": row["target_skill_count"],
        "best_gold_skill_id": best_gold,
        "best_nongold_skill_id": best_nongold,
        "total_margin": total_margin,
        "component_deltas": {
            label: deltas[position]
            for position, label in enumerate(COMPONENT_LABELS)
        },
    }


def critical_component_deltas(
    systems: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reference_label: str,
) -> dict[str, Any]:
    reference_rows = systems[reference_label]
    output: dict[str, Any] = {}
    for label, rows in systems.items():
        if label == reference_label:
            continue
        cases = [
            component_delta_case(row)
            for reference, row in zip(reference_rows, rows)
            if reference["fixed_ranked_skill_ids"][0]
            in set(reference["gold_skill_ids"])
            and row["fixed_ranked_skill_ids"][0] not in set(row["gold_skill_ids"])
        ]
        output[label] = {
            "queries": len(cases),
            "component_delta_summary": {
                component_label: numeric_summary(
                    case["component_deltas"][component_label] for case in cases
                )
                for component_label in COMPONENT_LABELS
            },
            "cases": cases,
        }
    return output


def analyze_position_scores(
    systems: Mapping[str, Sequence[dict[str, Any]]],
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reference_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if reference_label not in systems:
        raise ValueError(f"reference label is absent: {reference_label}")
    fixed_pool_payload, _ = analyze_fixed_pool(
        systems,
        manifests,
        reference_label=reference_label,
        cutoffs=CUTOFFS,
    )
    if not fixed_pool_payload["verification"]["passed"]:
        raise ValueError("fixed-pool protocol validation failed")
    for label, shard_manifests in manifests.items():
        for manifest in shard_manifests:
            if manifest["artifact_format"] != "legacy_core75_fixed_pool_score_shard_v2":
                raise ValueError(f"{label}: position analysis requires score shard v2")
            if tuple(manifest["scoring"].get("component_labels", ())) != COMPONENT_LABELS:
                raise ValueError(f"{label}: score component labels differ from protocol")

    validation = {
        label: validate_component_rows(label, rows) for label, rows in systems.items()
    }
    weight_rows = normalized_weight_grid()
    if len(weight_rows) != 3061:
        raise ValueError(f"normalized weight grid count changed: {len(weight_rows)}")
    weights = np.asarray(weight_rows, dtype=np.float64)
    uniform_indices = np.flatnonzero(np.all(weights == 1.0, axis=1))
    if uniform_indices.size != 1:
        raise ValueError("weight grid must contain exactly one uniform configuration")
    uniform_index = int(uniform_indices[0])

    systems_payload: dict[str, Any] = {}
    selected_indices: dict[str, int] = {}
    grid_output: list[dict[str, Any]] = []
    per_query_metrics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, rows in systems.items():
        query_recall, query_correct = grid_query_metrics(rows, weights)
        per_query_metrics[label] = (query_recall, query_correct)
        uniform_metrics = metric_dict(query_recall[:, uniform_index, :].mean(axis=0))
        baseline = baseline_metrics(rows)
        if any(
            not math.isclose(
                uniform_metrics[f"@{cutoff}"],
                baseline[f"@{cutoff}"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for cutoff in CUTOFFS
        ):
            raise ValueError(f"{label}: uniform weight metrics differ from baseline")
        selected, selected_index = selected_weight_payload(
            rows, weights, query_recall, query_correct
        )
        selected_indices[label] = selected_index
        cross_validation = cross_validate_weights(rows, weights, query_recall, folds=5)
        systems_payload[label] = {
            "baseline_uniform_metrics": baseline,
            "selected": selected,
            "cross_validation": cross_validation,
        }
        if label != reference_label:
            full_metrics = query_recall.mean(axis=0)
            distances = np.linalg.norm(weights - 1.0, axis=1)
            for position, weight_row in enumerate(weight_rows):
                metrics = metric_dict(full_metrics[position])
                grid_output.append(
                    {
                        "label": label,
                        "grid_position": position,
                        "weights": list(weight_row),
                        "distance_from_uniform_l2": float(distances[position]),
                        "metrics": metrics,
                        "deep_gate_passed": (
                            metrics["@10"] >= DEEP_GATES["@10"] - 1e-12
                            and metrics["@50"] >= DEEP_GATES["@50"] - 1e-12
                        ),
                        "full_gate_passed": (
                            metrics["@1"] >= R1_GATE - 1e-12
                            and metrics["@10"] >= DEEP_GATES["@10"] - 1e-12
                            and metrics["@50"] >= DEEP_GATES["@50"] - 1e-12
                        ),
                    }
                )

    component_agreement = {
        label: component_pair_agreement(systems[reference_label], rows)
        for label, rows in systems.items()
        if label != reference_label
    }
    critical = critical_component_deltas(systems, reference_label=reference_label)

    per_query_output: list[dict[str, Any]] = []
    for row_position in range(len(systems[reference_label])):
        reference = systems[reference_label][row_position]
        system_rows: dict[str, Any] = {}
        for label, rows in systems.items():
            row = rows[row_position]
            selected_weights = weights[selected_indices[label]]
            weighted, weighted_scores = weighted_rank(row, selected_weights)
            gold = set(row["gold_skill_ids"])
            system_rows[label] = {
                "baseline_top1_skill_id": row["fixed_ranked_skill_ids"][0],
                "weighted_top1_skill_id": weighted[0],
                "baseline_top1_correct": row["fixed_ranked_skill_ids"][0] in gold,
                "weighted_top1_correct": weighted[0] in gold,
                "baseline_nearest_gold_rank": rank_of_any(
                    gold, row["fixed_ranked_skill_ids"]
                ),
                "weighted_nearest_gold_rank": rank_of_any(gold, weighted),
                "weighted_best_gold_minus_best_nongold": max(
                    weighted_scores[skill_id] for skill_id in gold
                )
                - max(
                    score
                    for skill_id, score in weighted_scores.items()
                    if skill_id not in gold
                ),
            }
        per_query_output.append(
            {
                "row_position": row_position,
                "index": reference["index"],
                "source_index": reference["source_index"],
                "query": reference["query"],
                "gold_skill_ids": reference["gold_skill_ids"],
                "target_skill_count": reference["target_skill_count"],
                "systems": system_rows,
            }
        )

    fold_sizes = [
        {
            "fold": fold,
            "train": sum(int(row["row_position"]) % 5 != fold for row in systems[reference_label]),
            "heldout": sum(int(row["row_position"]) % 5 == fold for row in systems[reference_label]),
        }
        for fold in range(5)
    ]
    if any(row["train"] != 60 or row["heldout"] != 15 for row in fold_sizes):
        raise ValueError(f"unexpected five-fold split sizes: {fold_sizes}")

    payload = {
        "artifact_format": "legacy_core75_position_score_analysis_v1",
        "implementation": {
            "score_launch_commit": next(iter(manifests.values()))[0][
                "implementation"
            ].get("launch_commit"),
            "position_analyzer_sha256": file_sha256(Path(__file__).resolve()),
            "fixed_pool_analyzer_sha256": file_sha256(
                Path(analyze_fixed_pool.__code__.co_filename).resolve()
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "protocol": {
            "reference_label": reference_label,
            "query_count": len(systems[reference_label]),
            "component_labels": list(COMPONENT_LABELS),
            "grid_raw_values": list(GRID_RAW_VALUES),
            "normalized_grid_count": len(weight_rows),
            "normalization": "divide each five-weight tuple by its arithmetic mean, round to 8 decimals, deduplicate",
            "selection": "require R@10>=0.69 and R@50>=0.75, then maximize R@1/R@10/R@50 and minimize uniform L2 distance; if none qualify, report an ineligible best-effort weighting",
            "full_gate": {"@1": R1_GATE, "@10": 0.69, "@50": 0.75},
            "cross_validation": "five folds by row_position modulo 5",
        },
        "systems": systems_payload,
        "rtx_component_agreement": component_agreement,
        "rtx_correct_a800_wrong_component_deltas": critical,
        "verification": {
            "score_shard_v2": True,
            "fixed_pool_protocol": True,
            "component_labels": True,
            "component_sum_and_uniform_rank": validation,
            "uniform_metric_replay": True,
            "grid_count": len(weight_rows),
            "fold_sizes": fold_sizes,
            "passed": True,
        },
    }
    return payload, per_query_output, grid_output


def percentage(value: float) -> str:
    return f"{100.0 * value:.4f}"


def weight_text(weights: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.4f}" for value in weights) + "]"


def render_markdown(payload: Mapping[str, Any]) -> str:
    protocol = payload["protocol"]
    lines = [
        "# Legacy Official core75 position-score decomposition",
        "",
        (
            f"Conditions: Official core75={protocol['query_count']} queries; fixed pool=four "
            "beam-order Top50 union plus all gold; components=code positions 0/1/2/3 plus "
            "stop control=1; normalized weight grid=3,061."
        ),
        "",
        "## Constrained weight selection",
        "",
        "| system | uniform R@1/5/10/50 | reported R@1/5/10/50 | weights [c0,c1,c2,c3,stop] | role | top-1 gained/lost | fixed-pool gate | 5-fold heldout R@1/5/10/50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in payload["systems"].items():
        baseline = summary["baseline_uniform_metrics"]
        selected = summary["selected"]
        metrics = selected["metrics"]
        transitions = selected["query_top1_transitions"]
        heldout = summary["cross_validation"]["aggregate_heldout_metrics"]
        lines.append(
            f"| `{label}` | {percentage(baseline['@1'])}/{percentage(baseline['@5'])}/"
            f"{percentage(baseline['@10'])}/{percentage(baseline['@50'])} | "
            f"{percentage(metrics['@1'])}/{percentage(metrics['@5'])}/"
            f"{percentage(metrics['@10'])}/{percentage(metrics['@50'])} | "
            f"`{weight_text(selected['weights'])}` | "
            f"{selected['selection_status']} | "
            f"{transitions['baseline_wrong_weighted_correct']}/"
            f"{transitions['baseline_correct_weighted_wrong']} | "
            f"{selected['full_gate_passed']} | {percentage(heldout['@1'])}/"
            f"{percentage(heldout['@5'])}/{percentage(heldout['@10'])}/"
            f"{percentage(heldout['@50'])} |"
        )

    lines.extend(
        [
            "",
            "## RTX/A800 component rank agreement",
            "",
            "| A800 system | c0 | c1 | c2 | c3 | stop |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, components in payload["rtx_component_agreement"].items():
        means = [
            components[component]["rank_spearman"]["mean"]
            for component in COMPONENT_LABELS
        ]
        lines.append(
            f"| `{label}` | " + " | ".join(f"{float(value):.6f}" for value in means) + " |"
        )

    lines.extend(
        [
            "",
            "Verification: component coverage/sums, uniform exact ranking, uniform metrics, "
            "grid count, five-fold split sizes, and shard-v2 protocol all passed.",
        ]
    )
    return "\n".join(lines) + "\n"


def command_analyze(args: argparse.Namespace) -> None:
    shard_groups = parse_shards(args.shard)
    systems: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    for label, shard_dirs in shard_groups.items():
        systems[label], manifests[label] = load_system_shards(
            label,
            shard_dirs,
            expected_rows=args.expected_rows,
            expected_shards=args.expected_shards,
        )
    payload, per_query, grid_rows = analyze_position_scores(
        systems, manifests, reference_label=args.reference_label
    )
    atomic_write_text(
        args.output_root / "position_score_summary.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        args.output_root / "position_score_summary.md",
        render_markdown(payload),
    )
    atomic_write_text(
        args.output_root / "position_score_per_query.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_query),
    )
    atomic_write_text(
        args.output_root / "position_weight_grid.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in grid_rows),
    )
    (args.output_root / "position_scores.done").touch()
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "systems": list(systems),
                "grid_rows": len(grid_rows),
                "verification_passed": payload["verification"]["passed"],
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--expected-rows", type=int, default=75)
    parser.add_argument("--expected-shards", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    command_analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
