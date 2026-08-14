#!/usr/bin/env python3
"""Build a fixed-mass training schedule from diverse queries that cum_082 gets wrong."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from build_confuser_aware_queries import (
        MODE3_PREFIX,
        TASK5_PREFIX,
        file_sha256,
        load_jsonl,
        selected_single_skill,
        skill_id,
        task5_source_skill,
        target_ids_from_rows,
        write_json,
        write_jsonl,
    )
    from build_diverse_coverage_queries import COVERAGE_MODES, COVERAGE_PREFIX
    from build_fixed_mass_confuser_training import (
        _sort_mode3,
        _sort_task5,
        bind_candidates_to_source_indices,
        query_digest,
        scalar_text,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style test import
    from scripts.build_confuser_aware_queries import (
        MODE3_PREFIX,
        TASK5_PREFIX,
        file_sha256,
        load_jsonl,
        selected_single_skill,
        skill_id,
        task5_source_skill,
        target_ids_from_rows,
        write_json,
        write_jsonl,
    )
    from scripts.build_diverse_coverage_queries import COVERAGE_MODES, COVERAGE_PREFIX
    from scripts.build_fixed_mass_confuser_training import (
        _sort_mode3,
        _sort_task5,
        bind_candidates_to_source_indices,
        query_digest,
        scalar_text,
    )


SCHEDULE_VERSION = "fixed_mass_two_cycle_error_filtered_v1_20260814"
ERROR_SOURCE_KIND = "diverse_error"


def sort_error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode_order = {mode: index for index, mode in enumerate(COVERAGE_MODES)}

    def key(row: dict[str, Any]) -> tuple[int, int, str]:
        metadata = row.get("error_filtered_mining") or {}
        gold_rank = metadata.get("gold_rank")
        rank_score = 10_000 if gold_rank is None else int(gold_rank)
        mode = str((row.get("pseudo_query") or {}).get("mode") or "")
        return (-rank_score, mode_order.get(mode, len(mode_order)), scalar_text(row.get("index")))

    return sorted(rows, key=key)


def target_cycle_views(
    task5: list[dict[str, Any]],
    mode3: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> list[list[tuple[dict[str, Any], str]]]:
    """Place every error once and fill two five-view cycles with mode3/task5 rows."""
    if len(task5) != 5 or len(mode3) != 3 or len(errors) > len(COVERAGE_MODES):
        raise ValueError(
            f"target view pool must be task5=5, mode3=3, errors<=6; got "
            f"{len(task5)}/{len(mode3)}/{len(errors)}"
        )
    task5 = _sort_task5(task5)
    mode3 = _sort_mode3(mode3)
    errors = sort_error_rows(errors)
    base_needed = 10 - len(errors)
    base_unique = [*mode3, *task5]
    base_rows = [*base_unique, mode3[0], task5[0]][:base_needed]
    if len(base_rows) != base_needed:
        raise ValueError(f"insufficient fixed-mass base rows: needed={base_needed}, got={len(base_rows)}")

    cycles: list[list[tuple[dict[str, Any], str]]] = [[], []]
    cycle_sources: list[set[str]] = [set(), set()]

    def place(row: dict[str, Any], source_kind: str, preferred_cycle: int) -> None:
        source_index = scalar_text(row.get("index"))
        eligible = [
            cycle
            for cycle in (0, 1)
            if len(cycles[cycle]) < 5 and source_index not in cycle_sources[cycle]
        ]
        if not eligible:
            raise ValueError(f"cannot place fixed-mass source without within-cycle duplication: {source_index}")
        cycle = min(
            eligible,
            key=lambda candidate: (
                len(cycles[candidate]),
                0 if candidate == preferred_cycle else 1,
                candidate,
            ),
        )
        cycles[cycle].append((row, source_kind))
        cycle_sources[cycle].add(source_index)

    for index, row in enumerate(errors):
        place(row, ERROR_SOURCE_KIND, index % 2)
    for index, row in enumerate(base_rows):
        source_kind = "mode3" if scalar_text(row.get("index")).startswith(MODE3_PREFIX) else "task5"
        place(row, source_kind, index % 2)
    if [len(cycle) for cycle in cycles] != [5, 5]:
        raise ValueError(f"fixed-mass cycle sizes drifted: {[len(cycle) for cycle in cycles]}")
    return cycles


def build_fixed_mass_error_filtered_training(
    *,
    single_rows_path: Path,
    target_skills_path: Path,
    catalog_path: Path,
    legacy_candidates_path: Path,
    base_mining_rows_path: Path,
    base_candidates_path: Path,
    error_rows_path: Path,
    error_candidates_path: Path,
    error_manifest_path: Path,
    output_dir: Path,
    expected_catalog_skills: int,
    expected_targets: int,
    expected_negatives: int,
    expected_error_rows: int,
    expected_mining_checkpoint_sha256: str,
    cycles: int,
    pseudo_views_per_skill_per_cycle: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fixed-mass output: {output_dir}")
    if cycles != 2 or pseudo_views_per_skill_per_cycle != 5:
        raise ValueError("error-filtered schedule v1 requires two cycles and five pseudo views per cycle")

    catalog_ids = [skill_id(row) for row in load_jsonl(catalog_path)]
    if len(catalog_ids) != expected_catalog_skills or len(catalog_ids) != len(set(catalog_ids)):
        raise ValueError(
            f"catalog skill count/uniqueness mismatch: expected={expected_catalog_skills}, "
            f"observed={len(catalog_ids)}, unique={len(set(catalog_ids))}"
        )
    catalog_set = set(catalog_ids)
    targets = target_ids_from_rows(load_jsonl(target_skills_path))
    target_set = set(targets)
    if len(targets) != expected_targets or not target_set <= catalog_set:
        raise ValueError(
            f"target count/catalog closure mismatch: expected={expected_targets}, observed={len(targets)}"
        )

    error_manifest = json.loads(error_manifest_path.read_text(encoding="utf-8"))
    error_rows_sha256 = file_sha256(error_rows_path)
    error_candidates_sha256 = file_sha256(error_candidates_path)
    required_error_manifest = {
        "format": "diverse_coverage_error_filtered_v1",
        "target_skill_count": expected_targets,
        "views_per_target": len(COVERAGE_MODES),
        "error_rows": expected_error_rows,
        "mining_checkpoint_sha256": expected_mining_checkpoint_sha256,
        "negatives_per_error_row": expected_negatives,
    }
    drift = {
        key: (expected, error_manifest.get(key))
        for key, expected in required_error_manifest.items()
        if error_manifest.get(key) != expected
    }
    manifest_outputs = error_manifest.get("outputs") or {}
    for key, expected in (
        ("error_rows_sha256", error_rows_sha256),
        ("error_candidates_sha256", error_candidates_sha256),
    ):
        if manifest_outputs.get(key) != expected:
            drift[f"outputs.{key}"] = (expected, manifest_outputs.get(key))
    if drift:
        raise ValueError(f"error-filtered manifest drifted: {drift}")

    single_rows = load_jsonl(single_rows_path)
    base_mining_rows = load_jsonl(base_mining_rows_path)
    error_rows = load_jsonl(error_rows_path)
    if len(error_rows) != expected_error_rows:
        raise ValueError(
            f"error row count mismatch: expected={expected_error_rows}, observed={len(error_rows)}"
        )
    legacy_by_source = bind_candidates_to_source_indices(
        single_rows,
        load_jsonl(legacy_candidates_path),
        expected_negatives=expected_negatives,
    )
    base_by_source = bind_candidates_to_source_indices(
        base_mining_rows,
        load_jsonl(base_candidates_path),
        expected_negatives=expected_negatives,
        join_field="source_index",
    )
    error_by_source = bind_candidates_to_source_indices(
        error_rows,
        load_jsonl(error_candidates_path),
        expected_negatives=expected_negatives,
        join_field="source_index",
    )
    for source_name, candidate_map in (
        ("legacy", legacy_by_source),
        ("base", base_by_source),
        ("error", error_by_source),
    ):
        unknown = sorted(
            {
                negative
                for record in candidate_map.values()
                for negative in record["negative_skill_ids"]
                if negative not in catalog_set
            }
        )
        if unknown:
            raise ValueError(f"{source_name} candidates contain unknown skills: {unknown[:10]}")

    task5_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mode3_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_rows: list[dict[str, Any]] = []
    discarded_alias_task5 = 0
    for row in single_rows:
        positive = selected_single_skill(row)
        if positive not in catalog_set:
            raise ValueError(f"single row has unknown positive: index={row.get('index')} positive={positive}")
        index = scalar_text(row.get("index"))
        owner = task5_source_skill(index)
        if owner is not None:
            if owner == positive:
                task5_by_skill[positive].append(row)
            else:
                discarded_alias_task5 += 1
            continue
        if index.startswith(MODE3_PREFIX):
            mode3_by_skill[positive].append(row)
            continue
        original_rows.append(row)

    errors_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in error_rows:
        positive = selected_single_skill(row)
        index = scalar_text(row.get("index"))
        mode = str((row.get("pseudo_query") or {}).get("mode") or "")
        metadata = row.get("error_filtered_mining") or {}
        if positive not in target_set or not index.startswith(COVERAGE_PREFIX) or mode not in COVERAGE_MODES:
            raise ValueError(f"invalid diverse error row: index={index} positive={positive} mode={mode}")
        candidate = error_by_source.get(index)
        if candidate is None:
            raise ValueError(f"missing bound candidate for diverse error row: {index}")
        top1_confuser = str(metadata.get("top1_confuser_skill_id") or "")
        if not top1_confuser or candidate["negative_skill_ids"][0] != top1_confuser:
            raise ValueError(f"diverse error top-1 candidate mismatch: index={index}")
        errors_by_skill[positive].append(row)

    bad_task5 = {sid: len(task5_by_skill[sid]) for sid in catalog_ids if len(task5_by_skill[sid]) != 5}
    bad_mode3 = {sid: len(mode3_by_skill[sid]) for sid in targets if len(mode3_by_skill[sid]) != 3}
    bad_errors = {
        sid: len(rows)
        for sid, rows in errors_by_skill.items()
        if len(rows) > len(COVERAGE_MODES) or len({scalar_text(row.get("index")) for row in rows}) != len(rows)
    }
    if bad_task5 or bad_mode3 or bad_errors:
        raise ValueError(
            "pseudo pool closure failed: "
            f"task5={list(bad_task5.items())[:5]}, mode3={list(bad_mode3.items())[:5]}, "
            f"errors={list(bad_errors.items())[:5]}"
        )

    observed_error_histogram = Counter(len(errors_by_skill[sid]) for sid in targets)
    expected_error_histogram = {
        int(count): int(skills)
        for count, skills in (error_manifest.get("error_count_per_target_histogram") or {}).items()
    }
    if dict(sorted(observed_error_histogram.items())) != dict(sorted(expected_error_histogram.items())):
        raise ValueError(
            f"per-target error histogram drifted: observed={dict(observed_error_histogram)} "
            f"expected={expected_error_histogram}"
        )

    skill_cycles: dict[str, list[list[tuple[dict[str, Any], str]]]] = {}
    for sid in catalog_ids:
        if sid in target_set:
            skill_cycles[sid] = target_cycle_views(
                task5_by_skill[sid],
                mode3_by_skill[sid],
                errors_by_skill[sid],
            )
        else:
            canonical_task5 = _sort_task5(task5_by_skill[sid])
            skill_cycles[sid] = [
                [(row, "task5") for row in canonical_task5],
                [(row, "task5") for row in canonical_task5],
            ]

    output_rows: list[dict[str, Any]] = []
    output_candidates: list[dict[str, Any]] = []
    provenance_counts: Counter[str] = Counter()
    cycle_counts: Counter[int] = Counter()
    per_skill_pseudo_counts: Counter[str] = Counter()
    target_error_exposures: Counter[str] = Counter()

    def append_bound_row(row: dict[str, Any], cycle: int, source_kind: str) -> None:
        original_index = scalar_text(row.get("index"))
        positive = selected_single_skill(row)
        if positive is None:
            raise ValueError(f"fixed-mass row is not single-target: {original_index}")
        if source_kind == ERROR_SOURCE_KIND:
            record = error_by_source.get(original_index)
            candidate_provenance = "cum082_diverse_error_query_bound"
        elif positive in target_set and (
            original_index.startswith(TASK5_PREFIX) or original_index.startswith(MODE3_PREFIX)
        ):
            record = base_by_source.get(original_index)
            candidate_provenance = "cum082_base_query_bound"
        else:
            record = legacy_by_source.get(original_index)
            candidate_provenance = "step2400_query_bound"
        if record is None:
            raise ValueError(
                f"missing {candidate_provenance} candidate record: source={original_index}"
            )
        negatives = list(record["negative_skill_ids"])
        if len(negatives) != expected_negatives or len(negatives) != len(set(negatives)):
            raise ValueError(f"final negative contract failed for {original_index}")
        if positive in negatives:
            raise ValueError(f"final negatives include positive for {original_index}")
        output_index = len(output_rows)
        output_row = {
            **row,
            "index": output_index,
            "source_index": f"fixedmass-error:{cycle}:{original_index}",
            "fixed_mass": {
                "schedule_version": SCHEDULE_VERSION,
                "cycle": cycle,
                "source_kind": source_kind,
                "original_index": original_index,
            },
        }
        output_rows.append(output_row)
        output_candidates.append(
            {
                "row_index": output_index,
                "source_index": output_row["source_index"],
                "query_sha256": query_digest(output_row),
                "positive_skill_id": positive,
                "negative_skill_ids": negatives,
                "candidate_provenance": candidate_provenance,
            }
        )
        provenance_counts[candidate_provenance] += 1
        cycle_counts[cycle] += 1
        if source_kind != "original":
            per_skill_pseudo_counts[positive] += 1
        if source_kind == ERROR_SOURCE_KIND:
            target_error_exposures[positive] += 1

    for cycle in range(cycles):
        for row in original_rows:
            append_bound_row(row, cycle, "original")
        for sid in catalog_ids:
            for row, source_kind in skill_cycles[sid][cycle]:
                append_bound_row(row, cycle, source_kind)

    expected_pseudo_exposures = cycles * pseudo_views_per_skill_per_cycle
    bad_mass = {
        sid: per_skill_pseudo_counts[sid]
        for sid in catalog_ids
        if per_skill_pseudo_counts[sid] != expected_pseudo_exposures
    }
    bad_error_exposure = {
        sid: (target_error_exposures[sid], len(errors_by_skill[sid]))
        for sid in targets
        if target_error_exposures[sid] != len(errors_by_skill[sid])
    }
    if bad_mass or bad_error_exposure:
        raise ValueError(
            f"fixed mass exposure mismatch: skill_mass={list(bad_mass.items())[:5]}, "
            f"target_errors={list(bad_error_exposure.items())[:5]}"
        )
    expected_rows = cycles * (
        len(original_rows) + expected_catalog_skills * pseudo_views_per_skill_per_cycle
    )
    if len(output_rows) != expected_rows or len(output_candidates) != expected_rows:
        raise ValueError(
            f"fixed training row count mismatch: expected={expected_rows}, "
            f"rows={len(output_rows)}, candidates={len(output_candidates)}"
        )

    output_dir.mkdir(parents=True)
    rows_path = output_dir / "train_rows.two_cycle_fixed_mass.jsonl"
    candidates_path = output_dir / "candidates.two_cycle_fixed_mass.jsonl"
    write_jsonl(rows_path, output_rows)
    write_jsonl(candidates_path, output_candidates)
    manifest = {
        "format": "fixed_mass_error_filtered_training_v1",
        "schedule_version": SCHEDULE_VERSION,
        "single_rows": str(single_rows_path),
        "single_rows_sha256": file_sha256(single_rows_path),
        "target_skills": str(target_skills_path),
        "target_skills_sha256": file_sha256(target_skills_path),
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "legacy_candidates": str(legacy_candidates_path),
        "legacy_candidates_sha256": file_sha256(legacy_candidates_path),
        "base_mining_rows": str(base_mining_rows_path),
        "base_mining_rows_sha256": file_sha256(base_mining_rows_path),
        "base_candidates": str(base_candidates_path),
        "base_candidates_sha256": file_sha256(base_candidates_path),
        "error_rows": str(error_rows_path),
        "error_rows_sha256": error_rows_sha256,
        "error_candidates": str(error_candidates_path),
        "error_candidates_sha256": error_candidates_sha256,
        "error_manifest": str(error_manifest_path),
        "error_manifest_sha256": file_sha256(error_manifest_path),
        "mining_checkpoint_sha256": expected_mining_checkpoint_sha256,
        "catalog_skill_count": len(catalog_ids),
        "target_skill_count": len(targets),
        "targets_with_errors": sum(bool(errors_by_skill[sid]) for sid in targets),
        "cycles": cycles,
        "pseudo_views_per_skill_per_cycle": pseudo_views_per_skill_per_cycle,
        "pseudo_exposures_per_skill": expected_pseudo_exposures,
        "diverse_error_exposures": sum(target_error_exposures.values()),
        "error_count_per_target_histogram": {
            str(count): skills for count, skills in sorted(observed_error_histogram.items())
        },
        "original_single_rows_per_cycle": len(original_rows),
        "discarded_alias_task5_rows": discarded_alias_task5,
        "output_rows": len(output_rows),
        "output_candidates": len(output_candidates),
        "expected_negatives": expected_negatives,
        "cycle_row_counts": dict(sorted(cycle_counts.items())),
        "candidate_provenance_counts": dict(sorted(provenance_counts.items())),
        "rows_output": str(rows_path),
        "rows_output_sha256": file_sha256(rows_path),
        "candidates_output": str(candidates_path),
        "candidates_output_sha256": file_sha256(candidates_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-rows", type=Path, required=True)
    parser.add_argument("--target-skills", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--legacy-candidates", type=Path, required=True)
    parser.add_argument("--base-mining-rows", type=Path, required=True)
    parser.add_argument("--base-candidates", type=Path, required=True)
    parser.add_argument("--error-rows", type=Path, required=True)
    parser.add_argument("--error-candidates", type=Path, required=True)
    parser.add_argument("--error-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-catalog-skills", type=int, default=14551)
    parser.add_argument("--expected-targets", type=int, default=1722)
    parser.add_argument("--expected-negatives", type=int, default=10)
    parser.add_argument("--expected-error-rows", type=int, default=4789)
    parser.add_argument("--expected-mining-checkpoint-sha256", required=True)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--pseudo-views-per-skill-per-cycle", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = build_fixed_mass_error_filtered_training(
        single_rows_path=args.single_rows,
        target_skills_path=args.target_skills,
        catalog_path=args.skills_jsonl,
        legacy_candidates_path=args.legacy_candidates,
        base_mining_rows_path=args.base_mining_rows,
        base_candidates_path=args.base_candidates,
        error_rows_path=args.error_rows,
        error_candidates_path=args.error_candidates,
        error_manifest_path=args.error_manifest,
        output_dir=args.output_dir,
        expected_catalog_skills=args.expected_catalog_skills,
        expected_targets=args.expected_targets,
        expected_negatives=args.expected_negatives,
        expected_error_rows=args.expected_error_rows,
        expected_mining_checkpoint_sha256=args.expected_mining_checkpoint_sha256,
        cycles=args.cycles,
        pseudo_views_per_skill_per_cycle=args.pseudo_views_per_skill_per_cycle,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
