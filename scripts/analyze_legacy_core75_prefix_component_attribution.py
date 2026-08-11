#!/usr/bin/env python3
"""Attribute legacy fixed-pool errors to semantic-prefix depth and score components."""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.analyze_legacy_core75_fixed_pool import (
    analyze as analyze_fixed_pool,
    atomic_write_text,
    load_system_shards,
    parse_shards,
)
from scripts.analyze_legacy_core75_query_ranks import numeric_summary
from scripts.legacy_core75_score_protocol import (
    COMPONENT_LABELS,
    CUTOFFS,
    DEEP_GATES,
    R1_GATE,
)
from scripts.score_legacy_core75_fixed_pool import file_sha256, pipeline


CODE_DEPTH = 4
MASK_COUNT = 1 << len(COMPONENT_LABELS)
ALL_RTX_MASK = MASK_COUNT - 1


def mask_components(mask: int) -> list[str]:
    if mask < 0 or mask >= MASK_COUNT:
        raise ValueError(f"component mask must be in [0, {MASK_COUNT})")
    return [
        label
        for position, label in enumerate(COMPONENT_LABELS)
        if mask & (1 << position)
    ]


def macro_metrics(
    rows: Sequence[Mapping[str, Any]],
    rankings: Sequence[Sequence[Any]],
    gold_rankings: Sequence[Sequence[Any]],
) -> dict[str, float]:
    if len(rows) != len(rankings) or len(rows) != len(gold_rankings):
        raise ValueError("metric rows, rankings, and gold rankings must align")
    return {
        f"@{cutoff}": math.fsum(
            sum(gold_item in set(ranked[:cutoff]) for gold_item in gold_items)
            / len(gold_items)
            for ranked, gold_items in zip(rankings, gold_rankings)
        )
        / len(rows)
        for cutoff in CUTOFFS
    }


def prefix_query_payload(
    row: Mapping[str, Any],
    semantic_ids: Mapping[str, Sequence[int]],
    depth: int,
    *,
    atol: float = 1e-6,
) -> dict[str, Any]:
    if depth < 1 or depth > CODE_DEPTH:
        raise ValueError(f"prefix depth must be in [1, {CODE_DEPTH}]")
    prefix_values: dict[tuple[int, ...], list[float]] = {}
    for skill_id in row["pool_skill_ids"]:
        codes = semantic_ids[skill_id]
        if len(codes) != CODE_DEPTH:
            raise ValueError(
                f"skill={skill_id}: expected semantic depth={CODE_DEPTH}, got={len(codes)}"
            )
        prefix = tuple(int(value) for value in codes[:depth])
        components = row["fixed_ranked_skill_score_components"][skill_id]
        value = math.fsum(float(component) for component in components[:depth])
        prefix_values.setdefault(prefix, []).append(value)
    maximum_spread = max(
        (max(values) - min(values) for values in prefix_values.values()),
        default=0.0,
    )
    if maximum_spread > atol:
        raise ValueError(
            f"index={row['index']} depth={depth}: same-prefix score spread "
            f"{maximum_spread} exceeds atol={atol}"
        )
    scores = {prefix: values[0] for prefix, values in prefix_values.items()}
    ranked = sorted(sorted(scores), key=lambda prefix: scores[prefix], reverse=True)
    gold_prefix_mentions = [
        tuple(int(value) for value in semantic_ids[skill_id][:depth])
        for skill_id in row["gold_skill_ids"]
    ]
    gold_prefixes = set(gold_prefix_mentions)
    nongold_scores = [
        score for prefix, score in scores.items() if prefix not in gold_prefixes
    ]
    if not nongold_scores:
        raise ValueError(f"index={row['index']} depth={depth}: no nongold prefix")
    recalls = {
        f"@{cutoff}": sum(
            prefix in set(ranked[:cutoff]) for prefix in gold_prefix_mentions
        )
        / len(gold_prefix_mentions)
        for cutoff in CUTOFFS
    }
    return {
        "ranked_prefixes": [list(prefix) for prefix in ranked],
        "gold_prefix_mentions": [list(prefix) for prefix in gold_prefix_mentions],
        "unique_prefixes": len(ranked),
        "unique_gold_prefixes": len(gold_prefixes),
        "maximum_same_prefix_score_spread": maximum_spread,
        "top1_correct": ranked[0] in gold_prefixes,
        "gold_minus_best_nongold_margin": max(scores[prefix] for prefix in gold_prefixes)
        - max(nongold_scores),
        "recall": recalls,
    }


def prefix_analysis(
    systems: Mapping[str, Sequence[Mapping[str, Any]]],
    semantic_ids: Mapping[str, Sequence[int]],
    *,
    reference_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_payloads: dict[str, list[dict[int, dict[str, Any]]]] = {}
    system_summary: dict[str, Any] = {}
    maximum_spread = 0.0
    for label, rows in systems.items():
        per_query = []
        for row in rows:
            depths = {
                depth: prefix_query_payload(row, semantic_ids, depth)
                for depth in range(1, CODE_DEPTH + 1)
            }
            maximum_spread = max(
                maximum_spread,
                max(
                    payload["maximum_same_prefix_score_spread"]
                    for payload in depths.values()
                ),
            )
            per_query.append(depths)
        query_payloads[label] = per_query
        depths_summary = {}
        for depth in range(1, CODE_DEPTH + 1):
            values = [query[depth] for query in per_query]
            depths_summary[f"code_depth_{depth}"] = {
                "macro_recall": {
                    f"@{cutoff}": math.fsum(
                        value["recall"][f"@{cutoff}"] for value in values
                    )
                    / len(values)
                    for cutoff in CUTOFFS
                },
                "top1_correct_queries": sum(value["top1_correct"] for value in values),
                "gold_minus_best_nongold_margin": numeric_summary(
                    value["gold_minus_best_nongold_margin"] for value in values
                ),
                "unique_prefixes": numeric_summary(
                    value["unique_prefixes"] for value in values
                ),
            }
        system_summary[label] = depths_summary

    reference_rows = systems[reference_label]
    critical_summary: dict[str, Any] = {}
    for label, rows in systems.items():
        if label == reference_label:
            continue
        cases = []
        for position, (reference, candidate) in enumerate(zip(reference_rows, rows)):
            reference_gold = set(reference["gold_skill_ids"])
            candidate_gold = set(candidate["gold_skill_ids"])
            if (
                reference["fixed_ranked_skill_ids"][0] not in reference_gold
                or candidate["fixed_ranked_skill_ids"][0] in candidate_gold
            ):
                continue
            earliest = None
            depths = {}
            for depth in range(1, CODE_DEPTH + 1):
                reference_prefix = query_payloads[reference_label][position][depth]
                candidate_prefix = query_payloads[label][position][depth]
                if (
                    earliest is None
                    and reference_prefix["top1_correct"]
                    and not candidate_prefix["top1_correct"]
                ):
                    earliest = depth
                depths[f"code_depth_{depth}"] = {
                    "reference_top1_correct": reference_prefix["top1_correct"],
                    "candidate_top1_correct": candidate_prefix["top1_correct"],
                    "reference_margin": reference_prefix[
                        "gold_minus_best_nongold_margin"
                    ],
                    "candidate_margin": candidate_prefix[
                        "gold_minus_best_nongold_margin"
                    ],
                }
            cases.append(
                {
                    "row_position": position,
                    "index": reference["index"],
                    "earliest_reference_correct_candidate_wrong_depth": earliest,
                    "depths": depths,
                }
            )
        earliest_counts = Counter(
            str(case["earliest_reference_correct_candidate_wrong_depth"])
            for case in cases
        )
        critical_summary[label] = {
            "queries": len(cases),
            "earliest_depth_counts": dict(sorted(earliest_counts.items())),
            "cases": cases,
        }

    per_query_output = []
    for position, reference in enumerate(reference_rows):
        per_query_output.append(
            {
                "row_position": position,
                "index": reference["index"],
                "source_index": reference["source_index"],
                "gold_skill_ids": reference["gold_skill_ids"],
                "systems": {
                    label: {
                        f"code_depth_{depth}": query_payloads[label][position][depth]
                        for depth in range(1, CODE_DEPTH + 1)
                    }
                    for label in systems
                },
            }
        )
    return {
        "systems": system_summary,
        "rtx_correct_a800_wrong": critical_summary,
        "maximum_same_prefix_score_spread": maximum_spread,
    }, per_query_output


def hybrid_rank(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mask: int,
) -> tuple[list[str], dict[str, float]]:
    pool = list(candidate["pool_skill_ids"])
    if pool != list(reference["pool_skill_ids"]) or pool != sorted(pool):
        raise ValueError(f"index={candidate['index']}: hybrid pool is not aligned and sorted")
    reference_components = reference["fixed_ranked_skill_score_components"]
    candidate_components = candidate["fixed_ranked_skill_score_components"]
    scores = {}
    for skill_id in pool:
        scores[skill_id] = math.fsum(
            float(reference_components[skill_id][position])
            if mask & (1 << position)
            else float(candidate_components[skill_id][position])
            for position in range(len(COMPONENT_LABELS))
        )
    ranked = sorted(pool, key=lambda skill_id: scores[skill_id], reverse=True)
    return ranked, scores


def shapley_values(values: Mapping[int, float]) -> dict[str, float]:
    if set(values) != set(range(MASK_COUNT)):
        raise ValueError(f"Shapley values require all {MASK_COUNT} component masks")
    factorial = math.factorial
    denominator = factorial(len(COMPONENT_LABELS))
    output = {}
    for position, label in enumerate(COMPONENT_LABELS):
        contribution = 0.0
        bit = 1 << position
        for mask in range(MASK_COUNT):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = (
                factorial(size)
                * factorial(len(COMPONENT_LABELS) - size - 1)
                / denominator
            )
            contribution += weight * (values[mask | bit] - values[mask])
        output[label] = contribution
    expected = values[ALL_RTX_MASK] - values[0]
    if not math.isclose(
        math.fsum(output.values()), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Shapley contributions do not reconstruct endpoint delta")
    return output


def component_swap_analysis(
    systems: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reference_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_rows = systems[reference_label]
    output: dict[str, Any] = {}
    grid_rows: list[dict[str, Any]] = []
    for label, rows in systems.items():
        if label == reference_label:
            continue
        critical_positions = [
            position
            for position, (reference, candidate) in enumerate(zip(reference_rows, rows))
            if reference["fixed_ranked_skill_ids"][0]
            in set(reference["gold_skill_ids"])
            and candidate["fixed_ranked_skill_ids"][0]
            not in set(candidate["gold_skill_ids"])
        ]
        baseline_correct = [
            row["fixed_ranked_skill_ids"][0] in set(row["gold_skill_ids"])
            for row in rows
        ]
        mask_payloads = []
        mask_metrics: dict[int, dict[str, float]] = {}
        for mask in range(MASK_COUNT):
            rankings = [
                hybrid_rank(reference, candidate, mask)[0]
                for reference, candidate in zip(reference_rows, rows)
            ]
            gold_rankings = [row["gold_skill_ids"] for row in rows]
            metrics = macro_metrics(rows, rankings, gold_rankings)
            hybrid_correct = [
                ranked[0] in set(row["gold_skill_ids"])
                for row, ranked in zip(rows, rankings)
            ]
            deep_gate = all(
                metrics[cutoff] >= threshold - 1e-12
                for cutoff, threshold in DEEP_GATES.items()
            )
            full_gate = deep_gate and metrics["@1"] >= R1_GATE - 1e-12
            payload = {
                "label": label,
                "mask": mask,
                "scope": "counterfactual_attribution_only",
                "deployable": False,
                "rtx_components": mask_components(mask),
                "rtx_component_count": mask.bit_count(),
                "metrics": metrics,
                "deep_gate_passed": deep_gate,
                "full_gate_passed": full_gate,
                "critical_top1_recovered": sum(
                    hybrid_correct[position] for position in critical_positions
                ),
                "critical_queries": len(critical_positions),
                "top1_gained": sum(
                    not before and after
                    for before, after in zip(baseline_correct, hybrid_correct)
                ),
                "top1_lost": sum(
                    before and not after
                    for before, after in zip(baseline_correct, hybrid_correct)
                ),
            }
            mask_payloads.append(payload)
            mask_metrics[mask] = metrics
            grid_rows.append(payload)
            if mask == 0 and rankings != [
                list(row["fixed_ranked_skill_ids"]) for row in rows
            ]:
                raise ValueError(f"{label}: all-A800 endpoint changed ranking")
            if mask == ALL_RTX_MASK and rankings != [
                list(row["fixed_ranked_skill_ids"]) for row in reference_rows
            ]:
                raise ValueError(f"{label}: all-RTX endpoint changed ranking")

        full_gate_rows = [row for row in mask_payloads if row["full_gate_passed"]]
        minimal_full_gate = (
            min(
                full_gate_rows,
                key=lambda row: (
                    row["rtx_component_count"],
                    -row["metrics"]["@1"],
                    -row["metrics"]["@10"],
                    -row["metrics"]["@50"],
                    row["mask"],
                ),
            )
            if full_gate_rows
            else None
        )
        deep_gate_rows = [row for row in mask_payloads if row["deep_gate_passed"]]
        best_deep_gate = (
            max(
                deep_gate_rows,
                key=lambda row: (
                    row["metrics"]["@1"],
                    row["metrics"]["@10"],
                    row["metrics"]["@50"],
                    -row["rtx_component_count"],
                    -row["mask"],
                ),
            )
            if deep_gate_rows
            else None
        )
        shapley = {
            f"@{cutoff}": shapley_values(
                {mask: metrics[f"@{cutoff}"] for mask, metrics in mask_metrics.items()}
            )
            for cutoff in CUTOFFS
        }
        output[label] = {
            "a800_endpoint": mask_payloads[0],
            "rtx_endpoint": mask_payloads[ALL_RTX_MASK],
            "single_component_swaps": {
                COMPONENT_LABELS[position]: mask_payloads[1 << position]
                for position in range(len(COMPONENT_LABELS))
            },
            "full_gate_masks": len(full_gate_rows),
            "minimal_full_gate": minimal_full_gate,
            "best_deep_gate": best_deep_gate,
            "shapley": shapley,
            "shapley_conservation": {
                f"@{cutoff}": {
                    "sum": math.fsum(shapley[f"@{cutoff}"].values()),
                    "endpoint_delta": mask_metrics[ALL_RTX_MASK][f"@{cutoff}"]
                    - mask_metrics[0][f"@{cutoff}"],
                    "passed": True,
                }
                for cutoff in CUTOFFS
            },
        }
    return output, grid_rows


def analyze_attribution(
    systems: Mapping[str, Sequence[dict[str, Any]]],
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    semantic_ids: Mapping[str, Sequence[int]],
    semantic_id_mode: str,
    *,
    semantic_ids_file_sha256: str,
    reference_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fixed_payload, _ = analyze_fixed_pool(
        systems, manifests, reference_label=reference_label, cutoffs=CUTOFFS
    )
    if not fixed_payload["verification"]["passed"]:
        raise ValueError("fixed-pool validation failed")
    logical_sha256 = pipeline.semantic_ids_sha256(semantic_ids, semantic_id_mode)
    for label, shard_manifests in manifests.items():
        for manifest in shard_manifests:
            if manifest["artifact_format"] != "legacy_core75_fixed_pool_score_shard_v2":
                raise ValueError(f"{label}: attribution requires score shard v2")
            if tuple(manifest["scoring"]["component_labels"]) != COMPONENT_LABELS:
                raise ValueError(f"{label}: component labels differ")
            if manifest["semantic_ids_file_sha256"] != semantic_ids_file_sha256:
                raise ValueError(f"{label}: semantic IDs file hash differs")
            if manifest["semantic_ids_logical_sha256"] != logical_sha256:
                raise ValueError(f"{label}: semantic IDs logical hash differs")
    pool_skills = {
        skill_id for rows in systems.values() for row in rows for skill_id in row["pool_skill_ids"]
    }
    missing = sorted(pool_skills - set(semantic_ids))
    if missing:
        raise ValueError(f"semantic IDs miss pool skills: {missing[:5]}")

    prefix, per_query = prefix_analysis(
        systems, semantic_ids, reference_label=reference_label
    )
    swaps, grid_rows = component_swap_analysis(
        systems, reference_label=reference_label
    )
    a800_system_count = len(systems) - 1
    expected_grid_rows = a800_system_count * MASK_COUNT
    if len(swaps) != a800_system_count or len(grid_rows) != expected_grid_rows:
        raise ValueError(
            "component-swap grid count mismatch: "
            f"systems={len(swaps)}/{a800_system_count}, "
            f"rows={len(grid_rows)}/{expected_grid_rows}"
        )
    payload = {
        "artifact_format": "legacy_core75_prefix_component_attribution_v1",
        "implementation": {
            "analyzer_sha256": file_sha256(Path(__file__).resolve()),
            "score_launch_commit": next(iter(manifests.values()))[0][
                "implementation"
            ].get("launch_commit"),
        },
        "runtime": {"python": platform.python_version()},
        "protocol": {
            "reference_label": reference_label,
            "query_count": len(systems[reference_label]),
            "gold_mention_count": sum(
                len(row["gold_skill_ids"]) for row in systems[reference_label]
            ),
            "component_labels": list(COMPONENT_LABELS),
            "component_swap_masks_per_a800": MASK_COUNT,
            "prefix_depths": list(range(1, CODE_DEPTH + 1)),
            "prefix_score_atol": 1e-6,
            "semantic_ids_file_sha256": semantic_ids_file_sha256,
            "semantic_ids_logical_sha256": logical_sha256,
            "hybrid_scope": "read-only diagnostic; not a deployable score",
        },
        "prefix": prefix,
        "component_swap": swaps,
        "verification": {
            "fixed_pool_protocol": True,
            "semantic_ids_provenance": True,
            "same_prefix_score_consistency": prefix[
                "maximum_same_prefix_score_spread"
            ]
            <= 1e-6,
            "component_swap": {
                "masks_per_a800": MASK_COUNT,
                "a800_systems": a800_system_count,
                "grid_rows": len(grid_rows),
            },
            "endpoint_ranking_exact": True,
            "shapley_conservation": True,
            "passed": True,
        },
    }
    return payload, per_query, grid_rows


def percentage(value: float) -> str:
    return f"{100.0 * value:.4f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    protocol = payload["protocol"]
    lines = [
        "# Legacy Official core75 prefix/component attribution",
        "",
        (
            f"Conditions: Official core75={protocol['query_count']} queries; "
            f"gold mentions={protocol['gold_mention_count']}; fixed pool; "
            f"semantic depth={len(protocol['prefix_depths'])}; "
            "RTX→A800 component masks per A800="
            f"{protocol['component_swap_masks_per_a800']}."
        ),
        "Hybrid scores are read-only attribution on the fixed pool; they are not deployable configurations.",
        "",
        "## Prefix macro R@1 by code depth",
        "",
        "| system | depth 1 | depth 2 | depth 3 | depth 4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, depths in payload["prefix"]["systems"].items():
        values = [
            percentage(depths[f"code_depth_{depth}"]["macro_recall"]["@1"])
            for depth in range(1, CODE_DEPTH + 1)
        ]
        lines.append(f"| `{label}` | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## RTX component swap into A800",
            "",
            "| A800 system | endpoint R@1 | best deep-gate R@1 | minimal full-gate RTX components | full-gate masks | R@1 Shapley [c0,c1,c2,c3,stop] |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for label, summary in payload["component_swap"].items():
        minimal = summary["minimal_full_gate"]
        best = summary["best_deep_gate"]
        shapley = summary["shapley"]["@1"]
        shapley_text = "[" + ", ".join(
            f"{100.0 * shapley[component]:.4f}" for component in COMPONENT_LABELS
        ) + "]"
        lines.append(
            f"| `{label}` | {percentage(summary['a800_endpoint']['metrics']['@1'])} | "
            f"{percentage(best['metrics']['@1']) if best else 'n/a'} | "
            f"`{minimal['rtx_components'] if minimal else None}` | "
            f"{summary['full_gate_masks']} | `{shapley_text}` |"
        )
    lines.extend(
        [
            "",
            "Verification: semantic-ID provenance, same-prefix score consistency, "
            "all-A800/all-RTX endpoint rankings, all component masks, and exact Shapley "
            "conservation passed.",
        ]
    )
    return "\n".join(lines) + "\n"


def command_analyze(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite attribution output: {args.output_dir}")
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
    semantic_ids, semantic_id_mode = pipeline.load_semantic_ids_with_mode(
        args.semantic_ids
    )
    payload, per_query, grid_rows = analyze_attribution(
        systems,
        manifests,
        semantic_ids,
        semantic_id_mode,
        semantic_ids_file_sha256=file_sha256(args.semantic_ids),
        reference_label=args.reference_label,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        args.output_dir / "prefix_component_attribution_summary.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        args.output_dir / "prefix_component_attribution_summary.md",
        render_markdown(payload),
    )
    atomic_write_text(
        args.output_dir / "prefix_component_attribution_per_query.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_query),
    )
    atomic_write_text(
        args.output_dir / "component_swap_grid.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in grid_rows),
    )
    (args.output_dir / "attribution.done").touch()
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "systems": list(systems),
                "component_swap_rows": len(grid_rows),
                "verification_passed": payload["verification"]["passed"],
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--expected-rows", type=int, default=75)
    parser.add_argument("--expected-shards", type=int, default=2)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    command_analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
