#!/usr/bin/env python3
"""Materialize fixed-mass query exposure cycles and query-bound GR negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from build_confuser_aware_queries import (
        CONTRASTIVE_PREFIX,
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
except ModuleNotFoundError:  # pragma: no cover - package-style test import
    from scripts.build_confuser_aware_queries import (
        CONTRASTIVE_PREFIX,
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


SCHEDULE_VERSION = "fixed_mass_two_cycle_confuser_v1_20260814"


def query_digest(row: dict[str, Any]) -> str:
    query = str(row.get("query") or row.get("query_text") or "")
    if not query:
        raise ValueError(f"row has no query: index={row.get('index')}")
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def bind_candidates_to_source_indices(
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    expected_negatives: int,
    join_field: str = "index",
) -> dict[str, dict[str, Any]]:
    if len(source_rows) != len(candidate_rows):
        raise ValueError(
            f"candidate/source row count mismatch: candidates={len(candidate_rows)}, rows={len(source_rows)}"
        )
    bound: dict[str, dict[str, Any]] = {}
    for candidate in candidate_rows:
        row_index = int(candidate["row_index"])
        if not 0 <= row_index < len(source_rows):
            raise ValueError(f"candidate row index is out of range: {row_index}")
        row = source_rows[row_index]
        source_index = str(row.get("source_index") or row.get("index") or "")
        payload_index = str(row.get("index") or "")
        positive = selected_single_skill(row)
        negatives = [str(value) for value in candidate.get("negative_skill_ids", []) or []]
        if str(candidate.get("positive_skill_id") or "") != positive:
            raise ValueError(f"candidate positive mismatch at row={row_index}")
        if len(negatives) != expected_negatives or len(negatives) != len(set(negatives)):
            raise ValueError(f"candidate negative count/uniqueness mismatch at row={row_index}")
        if positive in negatives:
            raise ValueError(f"candidate includes positive at row={row_index}")
        expected_digest = query_digest(row)
        observed_digest = str(candidate.get("query_sha256") or expected_digest)
        if observed_digest != expected_digest:
            raise ValueError(f"candidate query digest mismatch at row={row_index}")
        if join_field == "index":
            key = payload_index
        elif join_field == "source_index":
            key = source_index
        else:
            raise ValueError(f"unknown candidate join field: {join_field}")
        if not key:
            raise ValueError(f"empty candidate join key at row={row_index}, field={join_field}")
        if key in bound:
            raise ValueError(f"duplicate source payload index in candidate binding: {key}")
        bound[key] = {
            "positive_skill_id": positive,
            "negative_skill_ids": negatives,
            "query_sha256": expected_digest,
            "candidate_row_index": row_index,
        }
    return bound


def aggregate_target_negatives(
    current_by_source: dict[str, dict[str, Any]],
    target_ids: set[str],
    expected_negatives: int,
) -> dict[str, list[str]]:
    scores: dict[str, Counter[str]] = defaultdict(Counter)
    occurrences: dict[str, Counter[str]] = defaultdict(Counter)
    best_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for record in current_by_source.values():
        target = str(record["positive_skill_id"])
        if target not in target_ids:
            continue
        for rank, negative in enumerate(record["negative_skill_ids"], start=1):
            scores[target][negative] += 1.0 / rank
            occurrences[target][negative] += 1
            best_ranks[target][negative] = min(best_ranks[target].get(negative, rank), rank)
    result: dict[str, list[str]] = {}
    for target in sorted(target_ids):
        ranked = sorted(
            scores[target],
            key=lambda negative: (
                -scores[target][negative],
                -occurrences[target][negative],
                best_ranks[target][negative],
                negative,
            ),
        )
        if len(ranked) < expected_negatives:
            raise ValueError(
                f"target has insufficient aggregate negatives: target={target}, "
                f"expected={expected_negatives}, observed={len(ranked)}"
            )
        result[target] = ranked[:expected_negatives]
    return result


def _sort_task5(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(str(row["index"]).rpartition(":")[2]))


def _sort_mode3(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"outcome_first": 0, "workflow_embedded": 1, "constraint_failure": 2}
    return sorted(rows, key=lambda row: order[str((row.get("pseudo_query") or {}).get("mode") or "")])


def _sort_contrastive(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int((row.get("pseudo_query") or {}).get("confuser_rank") or 0))


def target_cycle_views(
    task5: list[dict[str, Any]],
    mode3: list[dict[str, Any]],
    contrastive: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Return two five-view cycles covering all new/mode3 and the top four task5 views."""
    if len(task5) != 5 or len(mode3) != 3 or len(contrastive) != 3:
        raise ValueError(
            f"target view pool must be 5/3/3, got {len(task5)}/{len(mode3)}/{len(contrastive)}"
        )
    task5 = _sort_task5(task5)
    mode3 = _sort_mode3(mode3)
    contrastive = _sort_contrastive(contrastive)
    return [
        [*contrastive, task5[0], mode3[0]],
        [*task5[1:4], *mode3[1:3]],
    ]


def build_fixed_mass_training(
    *,
    single_rows_path: Path,
    new_rows_path: Path,
    target_skills_path: Path,
    catalog_path: Path,
    legacy_candidates_path: Path,
    current_mining_rows_path: Path,
    current_candidates_path: Path,
    confuser_pairs_path: Path,
    output_dir: Path,
    expected_catalog_skills: int,
    expected_targets: int,
    expected_negatives: int,
    cycles: int,
    pseudo_views_per_skill_per_cycle: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fixed-mass output: {output_dir}")
    if cycles != 2 or pseudo_views_per_skill_per_cycle != 5:
        raise ValueError("schedule v1 requires exactly two cycles and five pseudo views per skill/cycle")
    catalog = load_jsonl(catalog_path)
    catalog_ids = [skill_id(row) for row in catalog]
    if len(catalog_ids) != expected_catalog_skills or len(catalog_ids) != len(set(catalog_ids)):
        raise ValueError(
            f"catalog skill count/uniqueness mismatch: expected={expected_catalog_skills}, "
            f"observed={len(catalog_ids)}, unique={len(set(catalog_ids))}"
        )
    catalog_set = set(catalog_ids)
    targets = target_ids_from_rows(load_jsonl(target_skills_path))
    if len(targets) != expected_targets or not set(targets) <= catalog_set:
        raise ValueError(
            f"target count/catalog closure mismatch: expected={expected_targets}, observed={len(targets)}"
        )
    target_set = set(targets)
    single_rows = load_jsonl(single_rows_path)
    new_rows = load_jsonl(new_rows_path)
    current_mining_rows = load_jsonl(current_mining_rows_path)

    legacy_by_source = bind_candidates_to_source_indices(
        single_rows,
        load_jsonl(legacy_candidates_path),
        expected_negatives=expected_negatives,
    )
    current_by_source = bind_candidates_to_source_indices(
        current_mining_rows,
        load_jsonl(current_candidates_path),
        expected_negatives=expected_negatives,
        join_field="source_index",
    )
    for source_name, candidate_map in (
        ("legacy", legacy_by_source),
        ("current", current_by_source),
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
    target_aggregate_negatives = aggregate_target_negatives(
        current_by_source,
        target_set,
        expected_negatives,
    )
    target_negative_universe: dict[str, set[str]] = defaultdict(set)
    for record in current_by_source.values():
        target_negative_universe[str(record["positive_skill_id"])].update(
            str(value) for value in record["negative_skill_ids"]
        )

    pairs_by_target: dict[str, dict[int, str]] = defaultdict(dict)
    for pair in load_jsonl(confuser_pairs_path):
        target = str(pair["target_skill_id"])
        confuser = str(pair["confuser_skill_id"])
        rank = int(pair["confuser_rank"])
        if target not in target_set or confuser not in catalog_set or target == confuser:
            raise ValueError(f"invalid confuser pair: target={target}, confuser={confuser}")
        if rank not in {1, 2, 3} or rank in pairs_by_target[target]:
            raise ValueError(
                f"invalid or duplicate confuser rank: target={target}, confuser={confuser}, rank={rank}"
            )
        pairs_by_target[target][rank] = confuser
    bad_pairs = {
        target: rank_to_confuser
        for target, rank_to_confuser in pairs_by_target.items()
        if set(rank_to_confuser) != {1, 2, 3}
        or len(rank_to_confuser.values()) != len(set(rank_to_confuser.values()))
    }
    if set(pairs_by_target) != target_set or bad_pairs:
        raise ValueError(
            f"confuser pair closure mismatch: targets={len(pairs_by_target)}, bad={list(bad_pairs)[:5]}"
        )
    for target, rank_to_confuser in pairs_by_target.items():
        outside_mined_pool = [
            confuser
            for confuser in rank_to_confuser.values()
            if confuser not in target_negative_universe[target]
        ]
        if outside_mined_pool:
            raise ValueError(
                f"confuser pairs fall outside current mined negatives for {target}: {outside_mined_pool}"
            )

    task5_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mode3_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_rows: list[dict[str, Any]] = []
    discarded_alias_task5 = 0
    for row in single_rows:
        positive = selected_single_skill(row)
        if positive not in catalog_set:
            raise ValueError(f"single row has unknown positive: index={row.get('index')} positive={positive}")
        index = str(row.get("index") or "")
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

    contrastive_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in new_rows:
        positive = selected_single_skill(row)
        index = str(row.get("index") or "")
        if positive not in target_set or not index.startswith(CONTRASTIVE_PREFIX):
            raise ValueError(f"invalid new contrastive row: index={index} positive={positive}")
        pseudo = row.get("pseudo_query") or {}
        confuser = str(pseudo.get("confuser_skill_id") or "")
        rank = int(pseudo.get("confuser_rank") or 0)
        expected_confuser = pairs_by_target[positive].get(rank)
        expected_index = f"{CONTRASTIVE_PREFIX}{positive}:{rank - 1}"
        if (
            pseudo.get("mode") != "confuser_contrastive"
            or int(pseudo.get("rank") or 0) != rank - 1
            or confuser != expected_confuser
            or index != expected_index
        ):
            raise ValueError(
                "new contrastive row does not match authoritative pair: "
                f"index={index}, target={positive}, confuser={confuser}, rank={rank}, "
                f"expected_confuser={expected_confuser}, expected_index={expected_index}"
            )
        contrastive_by_skill[positive].append(row)
    bad_task5 = {sid: len(task5_by_skill[sid]) for sid in catalog_ids if len(task5_by_skill[sid]) != 5}
    bad_mode3 = {sid: len(mode3_by_skill[sid]) for sid in targets if len(mode3_by_skill[sid]) != 3}
    bad_new = {sid: len(contrastive_by_skill[sid]) for sid in targets if len(contrastive_by_skill[sid]) != 3}
    if bad_task5 or bad_mode3 or bad_new:
        raise ValueError(
            "pseudo pool closure failed: "
            f"task5={list(bad_task5.items())[:5]}, mode3={list(bad_mode3.items())[:5]}, "
            f"contrastive={list(bad_new.items())[:5]}"
        )

    skill_cycles: dict[str, list[list[dict[str, Any]]]] = {}
    for sid in catalog_ids:
        if sid in target_set:
            skill_cycles[sid] = target_cycle_views(
                task5_by_skill[sid],
                mode3_by_skill[sid],
                contrastive_by_skill[sid],
            )
        else:
            canonical_task5 = _sort_task5(task5_by_skill[sid])
            skill_cycles[sid] = [canonical_task5, canonical_task5]

    output_rows: list[dict[str, Any]] = []
    output_candidates: list[dict[str, Any]] = []
    provenance_counts: Counter[str] = Counter()
    cycle_counts: Counter[int] = Counter()
    per_skill_pseudo_counts: Counter[str] = Counter()
    target_new_exposures: Counter[str] = Counter()

    def append_bound_row(row: dict[str, Any], cycle: int, source_kind: str) -> None:
        original_index = str(row.get("index") or "")
        positive = selected_single_skill(row)
        if positive is None:
            raise ValueError(f"fixed-mass row is not single-target: {original_index}")
        if source_kind == "contrastive":
            confuser = str((row.get("pseudo_query") or {}).get("confuser_skill_id") or "")
            aggregate = target_aggregate_negatives[positive]
            negatives = [confuser, *(value for value in aggregate if value != confuser)][:expected_negatives]
            candidate_provenance = "cum082_target_aggregate_with_paired_first"
        elif positive in target_set and (
            original_index.startswith(TASK5_PREFIX) or original_index.startswith(MODE3_PREFIX)
        ):
            record = current_by_source.get(original_index)
            if record is None:
                raise ValueError(f"missing current target candidate record: {original_index}")
            negatives = list(record["negative_skill_ids"])
            candidate_provenance = "cum082_query_bound"
        else:
            record = legacy_by_source.get(original_index)
            if record is None:
                raise ValueError(f"missing legacy candidate record: {original_index}")
            negatives = list(record["negative_skill_ids"])
            candidate_provenance = "step2400_query_bound"
        if len(negatives) != expected_negatives or len(negatives) != len(set(negatives)):
            raise ValueError(f"final negative contract failed for {original_index}")
        if positive in negatives:
            raise ValueError(f"final negatives include positive for {original_index}")
        output_index = len(output_rows)
        output_row = {
            **row,
            "index": output_index,
            "source_index": f"fixedmass:{cycle}:{original_index}",
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
        if source_kind == "contrastive":
            target_new_exposures[positive] += 1

    for cycle in range(cycles):
        for row in original_rows:
            append_bound_row(row, cycle, "original")
        for sid in catalog_ids:
            for row in skill_cycles[sid][cycle]:
                index = str(row.get("index") or "")
                source_kind = (
                    "contrastive"
                    if index.startswith(CONTRASTIVE_PREFIX)
                    else "mode3"
                    if index.startswith(MODE3_PREFIX)
                    else "task5"
                )
                append_bound_row(row, cycle, source_kind)

    expected_pseudo_exposures = cycles * pseudo_views_per_skill_per_cycle
    bad_mass = {
        sid: per_skill_pseudo_counts[sid]
        for sid in catalog_ids
        if per_skill_pseudo_counts[sid] != expected_pseudo_exposures
    }
    bad_new_exposure = {sid: target_new_exposures[sid] for sid in targets if target_new_exposures[sid] != 3}
    if bad_mass or bad_new_exposure:
        raise ValueError(
            f"fixed mass exposure mismatch: skill_mass={list(bad_mass.items())[:5]}, "
            f"target_new={list(bad_new_exposure.items())[:5]}"
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
        "format": "fixed_mass_confuser_training_v1",
        "schedule_version": SCHEDULE_VERSION,
        "single_rows": str(single_rows_path),
        "single_rows_sha256": file_sha256(single_rows_path),
        "new_rows": str(new_rows_path),
        "new_rows_sha256": file_sha256(new_rows_path),
        "target_skills": str(target_skills_path),
        "target_skills_sha256": file_sha256(target_skills_path),
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "legacy_candidates": str(legacy_candidates_path),
        "legacy_candidates_sha256": file_sha256(legacy_candidates_path),
        "current_mining_rows": str(current_mining_rows_path),
        "current_mining_rows_sha256": file_sha256(current_mining_rows_path),
        "current_candidates": str(current_candidates_path),
        "current_candidates_sha256": file_sha256(current_candidates_path),
        "confuser_pairs": str(confuser_pairs_path),
        "confuser_pairs_sha256": file_sha256(confuser_pairs_path),
        "catalog_skill_count": len(catalog_ids),
        "target_skill_count": len(targets),
        "cycles": cycles,
        "pseudo_views_per_skill_per_cycle": pseudo_views_per_skill_per_cycle,
        "pseudo_exposures_per_skill": expected_pseudo_exposures,
        "contrastive_exposures_per_target": 3,
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


def command_build(args: argparse.Namespace) -> None:
    manifest = build_fixed_mass_training(
        single_rows_path=args.single_rows,
        new_rows_path=args.new_rows,
        target_skills_path=args.target_skills,
        catalog_path=args.skills_jsonl,
        legacy_candidates_path=args.legacy_candidates,
        current_mining_rows_path=args.current_mining_rows,
        current_candidates_path=args.current_candidates,
        confuser_pairs_path=args.confuser_pairs,
        output_dir=args.output_dir,
        expected_catalog_skills=args.expected_catalog_skills,
        expected_targets=args.expected_targets,
        expected_negatives=args.expected_negatives,
        cycles=args.cycles,
        pseudo_views_per_skill_per_cycle=args.pseudo_views_per_skill_per_cycle,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-rows", type=Path, required=True)
    parser.add_argument("--new-rows", type=Path, required=True)
    parser.add_argument("--target-skills", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--legacy-candidates", type=Path, required=True)
    parser.add_argument("--current-mining-rows", type=Path, required=True)
    parser.add_argument("--current-candidates", type=Path, required=True)
    parser.add_argument("--confuser-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-catalog-skills", type=int, default=14551)
    parser.add_argument("--expected-targets", type=int, default=1722)
    parser.add_argument("--expected-negatives", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--pseudo-views-per-skill-per-cycle", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    command_build(args)


if __name__ == "__main__":
    main()
