#!/usr/bin/env python3
"""Merge and analyze fixed-pool score shards for legacy Official core75."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.analyze_legacy_core75_query_ranks import (
    numeric_summary,
    pearson_correlation,
)
from scripts.score_legacy_core75_fixed_pool import file_sha256


FIXED_BUCKETS = ("rank1", "rank2_5", "rank6_10", "rank11_50", "rank51_plus")
RAW_BUCKETS = (*FIXED_BUCKETS[:-1], "missing_top50")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def parse_shards(items: Sequence[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for item in items:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"--shard must use LABEL=DIR, got: {item}")
        grouped[label].append(Path(raw_path))
    return dict(grouped)


def rank_of_any(gold: set[str], ranked: Sequence[str]) -> int | None:
    ranks = [position for position, skill_id in enumerate(ranked, start=1) if skill_id in gold]
    return min(ranks, default=None)


def rank_of_skill(skill_id: str, ranked: Sequence[str]) -> int | None:
    return next(
        (
            position
            for position, ranked_skill_id in enumerate(ranked, start=1)
            if ranked_skill_id == skill_id
        ),
        None,
    )


def rank_bucket(rank: int | None, *, fixed: bool) -> str:
    if rank is None:
        if fixed:
            raise ValueError("fixed pool must contain every gold skill")
        return "missing_top50"
    if rank == 1:
        return "rank1"
    if rank <= 5:
        return "rank2_5"
    if rank <= 10:
        return "rank6_10"
    if rank <= 50:
        return "rank11_50"
    return "rank51_plus"


def recall_at(ranked: Sequence[str], gold: set[str], cutoff: int) -> float:
    return len(gold.intersection(ranked[:cutoff])) / len(gold)


def load_system_shards(
    label: str,
    shard_dirs: Sequence[Path],
    *,
    expected_rows: int,
    expected_shards: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(shard_dirs) != expected_shards:
        raise ValueError(
            f"{label}: expected {expected_shards} shard dirs, found {len(shard_dirs)}"
        )
    manifests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    for shard_dir in shard_dirs:
        if not (shard_dir / "shard.done").is_file():
            raise FileNotFoundError(f"{label}: missing shard.done in {shard_dir}")
        manifest = read_json(shard_dir / "manifest.json")
        if manifest.get("artifact_format") not in {
            "legacy_core75_fixed_pool_score_shard_v1",
            "legacy_core75_fixed_pool_score_shard_v2",
        }:
            raise ValueError(f"{label}: unsupported shard artifact in {shard_dir}")
        if manifest.get("label") != label:
            raise ValueError(f"{label}: manifest label mismatch in {shard_dir}")
        if int(manifest.get("num_shards", -1)) != expected_shards:
            raise ValueError(f"{label}: manifest num_shards mismatch in {shard_dir}")
        if int(manifest.get("expected_total_rows", -1)) != expected_rows:
            raise ValueError(f"{label}: manifest expected_total_rows mismatch in {shard_dir}")
        shard_id = int(manifest["shard_id"])
        if shard_id in seen_shards:
            raise ValueError(f"{label}: duplicate shard_id={shard_id}")
        seen_shards.add(shard_id)
        score_path = shard_dir / "fixed_pool_scores.jsonl"
        if file_sha256(score_path) != manifest["fixed_pool_scores_sha256"]:
            raise ValueError(f"{label}: score file SHA-256 mismatch in {shard_dir}")
        shard_rows = read_jsonl(score_path)
        if len(shard_rows) != int(manifest["rows"]):
            raise ValueError(f"{label}: manifest row count mismatch in {shard_dir}")
        if any(int(row["shard_id"]) != shard_id for row in shard_rows):
            raise ValueError(f"{label}: row shard_id mismatch in {shard_dir}")
        manifests.append(manifest)
        rows.extend(shard_rows)
    if seen_shards != set(range(expected_shards)):
        raise ValueError(f"{label}: shard IDs are incomplete: {sorted(seen_shards)}")
    rows.sort(key=lambda row: int(row["row_position"]))
    if len(rows) != expected_rows:
        raise ValueError(f"{label}: expected {expected_rows} merged rows, found {len(rows)}")
    if [int(row["row_position"]) for row in rows] != list(range(expected_rows)):
        raise ValueError(f"{label}: row positions are missing or duplicated")
    return rows, sorted(manifests, key=lambda row: int(row["shard_id"]))


def validate_system_rows(label: str, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        index = row["index"]
        pool = row["pool_skill_ids"]
        ranked = row["fixed_ranked_skill_ids"]
        raw_ranked = row["raw_ranked_skill_ids"]
        scores = row["fixed_ranked_skill_scores"]
        gold_ids = row["gold_skill_ids"]
        gold = set(gold_ids)
        if row["label"] != label:
            raise ValueError(f"{label} index={index}: row label mismatch")
        if len(pool) != row["pool_size"] or len(set(pool)) != len(pool):
            raise ValueError(f"{label} index={index}: invalid pool")
        if set(ranked) != set(pool) or len(ranked) != len(pool):
            raise ValueError(f"{label} index={index}: fixed ranking differs from pool")
        if set(scores) != set(pool):
            raise ValueError(f"{label} index={index}: score keys differ from pool")
        if (
            not gold
            or len(gold) != len(gold_ids)
            or int(row["target_skill_count"]) != len(gold_ids)
            or not gold.issubset(pool)
        ):
            raise ValueError(f"{label} index={index}: pool does not cover gold")
        if len(raw_ranked) != len(set(raw_ranked)) or not set(raw_ranked).issubset(pool):
            raise ValueError(f"{label} index={index}: invalid raw ranking")
        ordered_scores = [float(scores[skill_id]) for skill_id in ranked]
        if not all(math.isfinite(score) for score in ordered_scores):
            raise ValueError(f"{label} index={index}: non-finite score")
        if any(left < right for left, right in zip(ordered_scores, ordered_scores[1:])):
            raise ValueError(f"{label} index={index}: scores are not monotonic")
        audit = row["exhaustive_audit"]
        if int(audit["capacity"]) != len(pool) or audit["frontier_sizes"][-1] != len(pool):
            raise ValueError(f"{label} index={index}: exhaustive capacity audit failed")
        replay = row["score_replay_validation"]
        if replay["available"] and (
            set(replay.get("rank_set_checks", {})) != {"@1", "@5", "@10", "@50"}
            or not all(replay["rank_set_checks"].values())
        ):
            raise ValueError(f"{label} index={index}: archived Top-K replay changed")


def validate_alignment(
    systems: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    reference_label = next(iter(systems))
    reference_rows = systems[reference_label]
    fields = (
        "row_position",
        "index",
        "source_index",
        "query",
        "gold_skill_ids",
        "target_skill_count",
        "pool_skill_ids",
        "pool_sha256",
        "pool_size",
    )
    for label, rows in systems.items():
        for reference, row in zip(reference_rows, rows):
            for field in fields:
                if row[field] != reference[field]:
                    raise ValueError(
                        f"{label} row_position={row['row_position']}: {field} differs from {reference_label}"
                    )


def analyze_system_row(row: Mapping[str, Any], cutoffs: Sequence[int]) -> dict[str, Any]:
    raw = row["raw_ranked_skill_ids"]
    fixed = row["fixed_ranked_skill_ids"]
    scores = row["fixed_ranked_skill_scores"]
    ordered_gold = row["gold_skill_ids"]
    gold = set(ordered_gold)
    raw_rank = rank_of_any(gold, raw)
    fixed_rank = rank_of_any(gold, fixed)
    best_gold_score = max(float(scores[skill_id]) for skill_id in gold)
    best_nongold_score = max(
        float(score) for skill_id, score in scores.items() if skill_id not in gold
    )
    raw_correct = raw[0] in gold
    fixed_correct = fixed[0] in gold
    if raw_correct and fixed_correct:
        transition = "both_correct"
    elif raw_correct:
        transition = "raw_correct_fixed_wrong"
    elif fixed_correct:
        transition = "raw_wrong_fixed_correct"
    else:
        transition = "both_wrong"
    gold_mentions = []
    for skill_id in ordered_gold:
        mention_raw_rank = rank_of_skill(skill_id, raw)
        mention_fixed_rank = rank_of_skill(skill_id, fixed)
        raw_rank1 = mention_raw_rank == 1
        fixed_rank1 = mention_fixed_rank == 1
        if raw_rank1 and fixed_rank1:
            mention_transition = "both_rank1"
        elif raw_rank1:
            mention_transition = "raw_rank1_fixed_not"
        elif fixed_rank1:
            mention_transition = "raw_not_fixed_rank1"
        else:
            mention_transition = "neither_rank1"
        raw_present = mention_raw_rank is not None
        fixed_top50 = mention_fixed_rank is not None and mention_fixed_rank <= 50
        if raw_present and fixed_top50:
            search_class = "raw_present_fixed_top50"
        elif raw_present:
            search_class = "raw_present_fixed_rank51_plus"
        elif fixed_top50:
            search_class = "raw_missing_fixed_top50"
        else:
            search_class = "raw_missing_fixed_rank51_plus"
        gold_mentions.append(
            {
                "skill_id": skill_id,
                "macro_weight": 1.0 / len(ordered_gold),
                "raw_rank": mention_raw_rank,
                "raw_rank_bucket": rank_bucket(mention_raw_rank, fixed=False),
                "fixed_rank": mention_fixed_rank,
                "fixed_rank_bucket": rank_bucket(mention_fixed_rank, fixed=True),
                "top1_transition": mention_transition,
                "search_class": search_class,
            }
        )
    return {
        "raw_top1_skill_id": raw[0],
        "fixed_top1_skill_id": fixed[0],
        "raw_top1_correct": raw_correct,
        "fixed_top1_correct": fixed_correct,
        "top1_transition": transition,
        "raw_nearest_gold_rank": raw_rank,
        "raw_nearest_gold_rank_bucket": rank_bucket(raw_rank, fixed=False),
        "fixed_nearest_gold_rank": fixed_rank,
        "fixed_nearest_gold_rank_bucket": rank_bucket(fixed_rank, fixed=True),
        "raw_macro_recall_contribution": {
            f"@{cutoff}": recall_at(raw, gold, cutoff) for cutoff in cutoffs
        },
        "fixed_macro_recall_contribution": {
            f"@{cutoff}": recall_at(fixed, gold, cutoff) for cutoff in cutoffs
        },
        "fixed_best_gold_score": best_gold_score,
        "fixed_best_nongold_score": best_nongold_score,
        "fixed_best_gold_minus_best_nongold": best_gold_score - best_nongold_score,
        "gold_mentions": gold_mentions,
    }


def summarize_system(
    per_query: Sequence[Mapping[str, Any]],
    *,
    label: str,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    rows = [row["systems"][label] for row in per_query]
    raw_buckets = Counter(row["raw_nearest_gold_rank_bucket"] for row in rows)
    fixed_buckets = Counter(row["fixed_nearest_gold_rank_bucket"] for row in rows)
    transitions = Counter(row["top1_transition"] for row in rows)
    raw_missing = [row for row in rows if row["raw_nearest_gold_rank"] is None]
    raw_missing_fixed_buckets = Counter(
        row["fixed_nearest_gold_rank_bucket"] for row in raw_missing
    )
    gold_mentions = [mention for row in rows for mention in row["gold_mentions"]]
    mention_raw_buckets = Counter(mention["raw_rank_bucket"] for mention in gold_mentions)
    mention_fixed_buckets = Counter(
        mention["fixed_rank_bucket"] for mention in gold_mentions
    )
    missing_mentions = [mention for mention in gold_mentions if mention["raw_rank"] is None]
    mention_raw_missing_fixed_buckets = Counter(
        mention["fixed_rank_bucket"] for mention in missing_mentions
    )
    mention_search_classes = Counter(
        mention["search_class"] for mention in gold_mentions
    )
    mention_transitions = Counter(
        mention["top1_transition"] for mention in gold_mentions
    )
    return {
        "rows": len(rows),
        "raw_macro_recall": {
            f"@{cutoff}": math.fsum(
                row["raw_macro_recall_contribution"][f"@{cutoff}"] for row in rows
            )
            / len(rows)
            for cutoff in cutoffs
        },
        "fixed_macro_recall": {
            f"@{cutoff}": math.fsum(
                row["fixed_macro_recall_contribution"][f"@{cutoff}"] for row in rows
            )
            / len(rows)
            for cutoff in cutoffs
        },
        "raw_top1_hit_queries": sum(row["raw_top1_correct"] for row in rows),
        "fixed_top1_hit_queries": sum(row["fixed_top1_correct"] for row in rows),
        "top1_transitions": {
            key: transitions.get(key, 0)
            for key in (
                "both_correct",
                "raw_correct_fixed_wrong",
                "raw_wrong_fixed_correct",
                "both_wrong",
            )
        },
        "raw_nearest_gold_rank_buckets": {
            bucket: raw_buckets.get(bucket, 0) for bucket in RAW_BUCKETS
        },
        "fixed_nearest_gold_rank_buckets": {
            bucket: fixed_buckets.get(bucket, 0) for bucket in FIXED_BUCKETS
        },
        "raw_missing_top50_fixed_rank_buckets": {
            bucket: raw_missing_fixed_buckets.get(bucket, 0) for bucket in FIXED_BUCKETS
        },
        "gold_mentions": {
            "total": len(gold_mentions),
            "raw_rank_buckets": {
                bucket: mention_raw_buckets.get(bucket, 0) for bucket in RAW_BUCKETS
            },
            "fixed_rank_buckets": {
                bucket: mention_fixed_buckets.get(bucket, 0) for bucket in FIXED_BUCKETS
            },
            "raw_missing_top50_fixed_rank_buckets": {
                bucket: mention_raw_missing_fixed_buckets.get(bucket, 0)
                for bucket in FIXED_BUCKETS
            },
            "search_classes": {
                key: mention_search_classes.get(key, 0)
                for key in (
                    "raw_present_fixed_top50",
                    "raw_present_fixed_rank51_plus",
                    "raw_missing_fixed_top50",
                    "raw_missing_fixed_rank51_plus",
                )
            },
            "top1_transitions": {
                key: mention_transitions.get(key, 0)
                for key in (
                    "both_rank1",
                    "raw_rank1_fixed_not",
                    "raw_not_fixed_rank1",
                    "neither_rank1",
                )
            },
            "weighted_raw_macro_recall": {
                f"@{cutoff}": math.fsum(
                    mention["macro_weight"]
                    for mention in gold_mentions
                    if mention["raw_rank"] is not None
                    and mention["raw_rank"] <= cutoff
                )
                / len(rows)
                for cutoff in cutoffs
            },
            "weighted_fixed_macro_recall": {
                f"@{cutoff}": math.fsum(
                    mention["macro_weight"]
                    for mention in gold_mentions
                    if mention["fixed_rank"] <= cutoff
                )
                / len(rows)
                for cutoff in cutoffs
            },
        },
        "fixed_best_gold_minus_best_nongold": numeric_summary(
            row["fixed_best_gold_minus_best_nongold"] for row in rows
        ),
    }


def summarize_pairwise(
    per_query: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for left, right in combinations(labels, 2):
        score_correlations: list[float] = []
        rank_correlations: list[float] = []
        same_top1 = 0
        for row in per_query:
            pool = row["pool_skill_ids"]
            left_raw = row["raw_system_rows"][left]
            right_raw = row["raw_system_rows"][right]
            left_scores = left_raw["fixed_ranked_skill_scores"]
            right_scores = right_raw["fixed_ranked_skill_scores"]
            score_correlation = pearson_correlation(
                [float(left_scores[skill_id]) for skill_id in pool],
                [float(right_scores[skill_id]) for skill_id in pool],
            )
            if score_correlation is not None:
                score_correlations.append(score_correlation)
            left_rank = {
                skill_id: rank
                for rank, skill_id in enumerate(
                    left_raw["fixed_ranked_skill_ids"], start=1
                )
            }
            right_rank = {
                skill_id: rank
                for rank, skill_id in enumerate(
                    right_raw["fixed_ranked_skill_ids"], start=1
                )
            }
            rank_correlation = pearson_correlation(
                [float(left_rank[skill_id]) for skill_id in pool],
                [float(right_rank[skill_id]) for skill_id in pool],
            )
            if rank_correlation is not None:
                rank_correlations.append(rank_correlation)
            same_top1 += (
                left_raw["fixed_ranked_skill_ids"][0]
                == right_raw["fixed_ranked_skill_ids"][0]
            )
        key = f"{left}__vs__{right}"
        summaries[key] = {
            "left_label": left,
            "right_label": right,
            "rows": len(per_query),
            "same_fixed_top1_queries": same_top1,
            "fixed_pool_score_pearson": numeric_summary(score_correlations),
            "fixed_pool_rank_spearman": numeric_summary(rank_correlations),
        }
    return summaries


def summarize_reference_critical(
    per_query: Sequence[Mapping[str, Any]],
    *,
    reference_label: str,
    labels: Sequence[str],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for label in labels:
        if label == reference_label:
            continue
        critical = [
            row
            for row in per_query
            if row["systems"][reference_label]["raw_top1_correct"]
            and not row["systems"][label]["raw_top1_correct"]
        ]
        fixed_buckets = Counter(
            row["systems"][label]["fixed_nearest_gold_rank_bucket"]
            for row in critical
        )
        summaries[label] = {
            "rows": len(critical),
            "indices": [row["index"] for row in critical],
            "fixed_top1_recovered": sum(
                row["systems"][label]["fixed_top1_correct"] for row in critical
            ),
            "fixed_nearest_gold_rank_buckets": {
                bucket: fixed_buckets.get(bucket, 0) for bucket in FIXED_BUCKETS
            },
            "cases": [
                {
                    "index": row["index"],
                    "target_skill_count": row["target_skill_count"],
                    "raw_nearest_gold_rank": row["systems"][label][
                        "raw_nearest_gold_rank"
                    ],
                    "fixed_nearest_gold_rank": row["systems"][label][
                        "fixed_nearest_gold_rank"
                    ],
                    "fixed_top1_correct": row["systems"][label][
                        "fixed_top1_correct"
                    ],
                    "fixed_best_gold_minus_best_nongold": row["systems"][label][
                        "fixed_best_gold_minus_best_nongold"
                    ],
                }
                for row in critical
            ],
        }
    return summaries


def analyze(
    systems: Mapping[str, Sequence[dict[str, Any]]],
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reference_label: str,
    cutoffs: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if reference_label not in systems:
        raise ValueError(f"reference label is absent: {reference_label}")
    labels = list(systems)
    for label, rows in systems.items():
        validate_system_rows(label, rows)
    validate_alignment(systems)

    global_pool_hashes = {
        manifest["candidate_pool"]["global_sha256"]
        for rows in manifests.values()
        for manifest in rows
    }
    if len(global_pool_hashes) != 1:
        raise ValueError(f"global candidate-pool hash mismatch: {global_pool_hashes}")
    scoring_protocols = {
        json.dumps(manifest["scoring"], sort_keys=True)
        for rows in manifests.values()
        for manifest in rows
    }
    if len(scoring_protocols) != 1:
        raise ValueError("scoring protocol differs across shards")

    def model_input_semantics(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
        model_inputs = manifest.get("model_inputs")
        if model_inputs is None:
            if manifest.get("artifact_format") == "legacy_core75_fixed_pool_score_shard_v2":
                raise ValueError("score shard v2 is missing model-input provenance")
            return None
        base_model = model_inputs["base_model"]
        tokenizer = model_inputs["tokenizer"]
        return {
            "base_model": {
                "config_json_sha256": base_model["config_json_sha256"],
                "model_safetensors_sha256": base_model[
                    "model_safetensors_sha256"
                ],
            },
            "tokenizer": {
                "tokenizer_json_sha256": tokenizer["tokenizer_json_sha256"],
                "tokenizer_semantic_config_sha256": tokenizer[
                    "tokenizer_semantic_config_sha256"
                ],
                "tokenized_query_inputs_sha256": tokenizer[
                    "tokenized_query_inputs_sha256"
                ],
            },
        }

    shared_projections = {
        json.dumps(
            {
                "semantic_ids_file_sha256": manifest["semantic_ids_file_sha256"],
                "semantic_ids_logical_sha256": manifest[
                    "semantic_ids_logical_sha256"
                ],
                "rows_sha256": manifest["rows_sha256"],
                "prediction_inputs": manifest["prediction_inputs"],
                "model_input_semantics": model_input_semantics(manifest),
                "implementation": manifest["implementation"],
                "runtime": manifest["runtime"],
            },
            sort_keys=True,
        )
        for rows in manifests.values()
        for manifest in rows
    }
    if len(shared_projections) != 1:
        raise ValueError("input, implementation, or runtime contract differs across shards")
    for label, rows in manifests.items():
        if len({row["checkpoint_file_sha256"] for row in rows}) != 1:
            raise ValueError(f"{label}: checkpoint hash differs across shards")
        if len(
            {
                json.dumps(row.get("model_inputs"), sort_keys=True)
                for row in rows
            }
        ) != 1:
            raise ValueError(f"{label}: model-input provenance differs across shards")
        validation = [row["score_replay_validation"] for row in rows]
        if label == reference_label:
            if any(row["available"] for row in validation):
                raise ValueError("scoreless reference unexpectedly reports replay validation")
            if any(int(row["queries"]) != 0 for row in validation):
                raise ValueError("scoreless reference unexpectedly validates replay queries")
        else:
            if not all(row["available"] for row in validation):
                raise ValueError(f"{label}: archived score replay validation is missing")
            if any(
                int(row["queries"]) != int(manifest["rows"])
                for row, manifest in zip(validation, rows)
            ):
                raise ValueError(f"{label}: replay validation does not cover every shard row")
            if any(
                float(row["maximum_absolute_error"]) > float(row["atol"])
                for row in validation
            ):
                raise ValueError(f"{label}: archived score replay validation failed")
            if any(
                set(row.get("rank_set_checks", {})) != {"@1", "@5", "@10", "@50"}
                or not all(row["rank_set_checks"].values())
                for row in validation
            ):
                raise ValueError(f"{label}: archived Top-K replay membership changed")

    per_query: list[dict[str, Any]] = []
    for row_position in range(len(next(iter(systems.values())))):
        raw_rows = {label: systems[label][row_position] for label in labels}
        reference = raw_rows[reference_label]
        per_query.append(
            {
                "row_position": row_position,
                "index": reference["index"],
                "source_index": reference["source_index"],
                "query": reference["query"],
                "gold_skill_ids": reference["gold_skill_ids"],
                "target_skill_count": reference["target_skill_count"],
                "pool_skill_ids": reference["pool_skill_ids"],
                "pool_size": reference["pool_size"],
                "systems": {
                    label: analyze_system_row(raw_rows[label], cutoffs)
                    for label in labels
                },
                "raw_system_rows": raw_rows,
            }
        )
    system_summaries = {
        label: summarize_system(per_query, label=label, cutoffs=cutoffs)
        for label in labels
    }
    pairwise = summarize_pairwise(per_query, labels)
    critical = summarize_reference_critical(
        per_query, reference_label=reference_label, labels=labels
    )

    total_gold_mentions = sum(len(row["gold_skill_ids"]) for row in per_query)
    mention_conservation: dict[str, Any] = {}
    for label, summary in system_summaries.items():
        mentions = summary["gold_mentions"]
        raw_recall_checks = {
            f"@{cutoff}": math.isclose(
                mentions["weighted_raw_macro_recall"][f"@{cutoff}"],
                summary["raw_macro_recall"][f"@{cutoff}"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for cutoff in cutoffs
        }
        fixed_recall_checks = {
            f"@{cutoff}": math.isclose(
                mentions["weighted_fixed_macro_recall"][f"@{cutoff}"],
                summary["fixed_macro_recall"][f"@{cutoff}"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for cutoff in cutoffs
        }
        checks = {
            "total_matches": mentions["total"] == total_gold_mentions,
            "raw_rank_partition": sum(mentions["raw_rank_buckets"].values())
            == total_gold_mentions,
            "fixed_rank_partition": sum(mentions["fixed_rank_buckets"].values())
            == total_gold_mentions,
            "raw_missing_fixed_rank_partition": sum(
                mentions["raw_missing_top50_fixed_rank_buckets"].values()
            )
            == mentions["raw_rank_buckets"]["missing_top50"],
            "search_class_partition": sum(mentions["search_classes"].values())
            == total_gold_mentions,
            "top1_transition_partition": sum(mentions["top1_transitions"].values())
            == total_gold_mentions,
            "raw_recall_reconstructed": raw_recall_checks,
            "fixed_recall_reconstructed": fixed_recall_checks,
        }
        passed = (
            all(value for key, value in checks.items() if not key.endswith("reconstructed"))
            and all(raw_recall_checks.values())
            and all(fixed_recall_checks.values())
        )
        checks["passed"] = passed
        mention_conservation[label] = checks
    if not all(checks["passed"] for checks in mention_conservation.values()):
        raise ValueError(f"gold-mention conservation failed: {mention_conservation}")

    output_per_query = []
    for row in per_query:
        output_per_query.append(
            {key: value for key, value in row.items() if key != "raw_system_rows"}
        )
    pool_sizes = [row["pool_size"] for row in output_per_query]
    payload = {
        "artifact_format": "legacy_core75_fixed_pool_analysis_v1",
        "protocol": {
            "reference_label": reference_label,
            "query_count": len(output_per_query),
            "gold_mention_count": total_gold_mentions,
            "cutoffs": list(cutoffs),
            "candidate_pool": "four beam-order Top50 union plus all gold",
            "score": "four code-token full-vocabulary log-probs plus stop control log-prob",
        },
        "candidate_pool": {
            "global_sha256": next(iter(global_pool_hashes)),
            "size": numeric_summary(pool_sizes),
            "gold_coverage": 1.0,
        },
        "systems": system_summaries,
        "pairwise_fixed_pool": pairwise,
        "reference_critical": critical,
        "verification": {
            "shard_merge": True,
            "row_and_pool_alignment": True,
            "input_implementation_runtime_alignment": True,
            "model_input_semantics_alignment": True,
            "exhaustive_pool_coverage": True,
            "score_replay": {
                label: {
                    "queries": sum(
                        int(row["score_replay_validation"]["queries"])
                        for row in rows
                    ),
                    "maximum_absolute_error": max(
                        (
                            float(row["score_replay_validation"][
                                "maximum_absolute_error"
                            ])
                            for row in rows
                            if row["score_replay_validation"]["available"]
                        ),
                        default=None,
                    ),
                    "rank_set_checks": (
                        {f"@{cutoff}": None for cutoff in (1, 5, 10, 50)}
                        if label == reference_label
                        else {
                            f"@{cutoff}": all(
                                row["score_replay_validation"]["rank_set_checks"][
                                    f"@{cutoff}"
                                ]
                                for row in rows
                            )
                            for cutoff in (1, 5, 10, 50)
                        }
                    ),
                }
                for label, rows in manifests.items()
            },
            "gold_mention_conservation": mention_conservation,
            "passed": True,
        },
    }
    return payload, output_per_query


def percentage(value: float) -> str:
    return f"{100 * value:.4f}"


def decimal(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    protocol = payload["protocol"]
    systems = payload["systems"]
    lines = [
        "# Legacy Official core75 fixed-pool scoring",
        "",
        (
            f"Conditions: Official core75={protocol['query_count']} queries; "
            "pool=four beam-order Top50 union plus all gold; CE score=four code-token "
            "full-vocabulary log-probs plus stop control log-prob; exhaustive per-query trie."
        ),
        "",
        "## Raw beam versus fixed pool",
        "",
        "| system | raw R@1/5/10/50 | fixed R@1/5/10/50 | raw→fixed top-1 gained/lost | raw missing Top50 → fixed rank 1/2-5/6-10/11-50/51+ |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, summary in systems.items():
        raw = summary["raw_macro_recall"]
        fixed = summary["fixed_macro_recall"]
        transitions = summary["top1_transitions"]
        missing = summary["raw_missing_top50_fixed_rank_buckets"]
        lines.append(
            f"| `{label}` | {percentage(raw['@1'])}/{percentage(raw['@5'])}/"
            f"{percentage(raw['@10'])}/{percentage(raw['@50'])} | "
            f"{percentage(fixed['@1'])}/{percentage(fixed['@5'])}/"
            f"{percentage(fixed['@10'])}/{percentage(fixed['@50'])} | "
            f"{transitions['raw_wrong_fixed_correct']}/{transitions['raw_correct_fixed_wrong']} | "
            f"{missing['rank1']}/{missing['rank2_5']}/{missing['rank6_10']}/"
            f"{missing['rank11_50']}/{missing['rank51_plus']} |"
        )

    lines.extend(
        [
            "",
            "## Gold-mention accounting",
            "",
            (
                f"Conditions: {protocol['gold_mention_count']} gold mentions across "
                f"Official core75={protocol['query_count']} queries; multi-GT mentions are "
                "weighted by 1/target_skill_count when reconstructing macro recall."
            ),
            "",
            "| system | mentions | raw rank 1/2-5/6-10/11-50/missing | fixed rank 1/2-5/6-10/11-50/51+ | raw-missing → fixed 1/2-5/6-10/11-50/51+ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, summary in systems.items():
        mentions = summary["gold_mentions"]
        raw = mentions["raw_rank_buckets"]
        fixed = mentions["fixed_rank_buckets"]
        missing = mentions["raw_missing_top50_fixed_rank_buckets"]
        lines.append(
            f"| `{label}` | {mentions['total']} | {raw['rank1']}/{raw['rank2_5']}/"
            f"{raw['rank6_10']}/{raw['rank11_50']}/{raw['missing_top50']} | "
            f"{fixed['rank1']}/{fixed['rank2_5']}/{fixed['rank6_10']}/"
            f"{fixed['rank11_50']}/{fixed['rank51_plus']} | {missing['rank1']}/"
            f"{missing['rank2_5']}/{missing['rank6_10']}/{missing['rank11_50']}/"
            f"{missing['rank51_plus']} |"
        )

    lines.extend(
        [
            "",
            "## Fixed-pool checkpoint agreement",
            "",
            "| pair | same fixed top-1 | score Pearson mean | rank Spearman mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for summary in payload["pairwise_fixed_pool"].values():
        lines.append(
            f"| `{summary['left_label']}` / `{summary['right_label']}` | "
            f"{summary['same_fixed_top1_queries']} | "
            f"{decimal(summary['fixed_pool_score_pearson']['mean'])} | "
            f"{decimal(summary['fixed_pool_rank_spearman']['mean'])} |"
        )

    lines.extend(["", "## RTX-correct/A800-raw-wrong cases", ""])
    for label, summary in payload["reference_critical"].items():
        buckets = summary["fixed_nearest_gold_rank_buckets"]
        lines.append(
            f"- `{label}`: raw lost={summary['rows']}; fixed top-1 recovered="
            f"{summary['fixed_top1_recovered']}; fixed nearest-gold buckets "
            f"(1/2-5/6-10/11-50/51+)={buckets['rank1']}/{buckets['rank2_5']}/"
            f"{buckets['rank6_10']}/{buckets['rank11_50']}/{buckets['rank51_plus']}; "
            f"indices={summary['indices']}."
        )
    pool = payload["candidate_pool"]["size"]
    lines.extend(
        [
            "",
            f"Pool size min/mean/max={int(pool['min'])}/{pool['mean']:.2f}/{int(pool['max'])} "
            f"(Official core75={protocol['query_count']} queries).",
            "",
            "Verification: shard merge, row/pool alignment, exhaustive coverage, gold coverage, "
            "and archived A800 score replay all passed.",
        ]
    )
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


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
    payload, per_query = analyze(
        systems,
        manifests,
        reference_label=args.reference_label,
        cutoffs=tuple(args.cutoff),
    )
    atomic_write_text(
        args.output_root / "fixed_pool_summary.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        args.output_root / "fixed_pool_per_query.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_query),
    )
    atomic_write_text(
        args.output_root / "fixed_pool_summary.md", render_markdown(payload)
    )
    (args.output_root / "fixed_pool.done").touch()
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "systems": list(systems),
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
    parser.add_argument("--cutoff", type=int, action="append", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cutoff is None:
        args.cutoff = [1, 5, 10, 50]
    command_analyze(args)


if __name__ == "__main__":
    main()
