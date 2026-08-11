#!/usr/bin/env python3
"""Audit c1-diverging hard-negative coverage in the frozen teacher panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from itertools import islice, zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


DEFAULT_TOP_KS = (1, 5, 10, 20, 50)
DEFAULT_DEPTH = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        os.chmod(path, 0o644)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield payload


def unique_strings(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def row_gold_skill_ids(row: Mapping[str, Any]) -> list[str]:
    values = row.get("selected_skill_ids")
    if values is None:
        values = row.get("gold_skill_ids")
    if values is None and isinstance(row.get("generated"), Mapping):
        values = row["generated"].get("selected_skill_ids")
    gold = unique_strings(values or [])
    if not gold:
        raise ValueError(f"index={row.get('index')}: row has no gold skills")
    return gold


def row_query(row: Mapping[str, Any]) -> str:
    value = row.get("query")
    if value is None and isinstance(row.get("generated"), Mapping):
        value = row["generated"].get("query")
    text = str(value or "")
    if not text:
        raise ValueError(f"index={row.get('index')}: row has an empty query")
    return text


def load_semantic_ids(path: Path, *, depth: int) -> dict[str, tuple[int, ...]]:
    semantic_ids: dict[str, tuple[int, ...]] = {}
    for row in load_jsonl(path):
        skill_id = str(row.get("skill_id") or "")
        values = row.get("semantic_id")
        if not skill_id or not isinstance(values, list):
            raise ValueError("semantic ID row is missing skill_id or semantic_id")
        codes = tuple(int(value) for value in values)
        if len(codes) != depth:
            raise ValueError(
                f"skill={skill_id}: semantic depth={len(codes)} does not match expected={depth}"
            )
        if skill_id in semantic_ids:
            raise ValueError(f"duplicate semantic ID skill={skill_id}")
        semantic_ids[skill_id] = codes
    if not semantic_ids:
        raise ValueError("semantic ID file is empty")
    return semantic_ids


def first_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    if len(left) != len(right):
        raise ValueError("semantic IDs must have the same depth")
    for position, (left_value, right_value) in enumerate(zip(left, right)):
        if int(left_value) != int(right_value):
            return position
    return None


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "p90": percentile(0.9),
        "max": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
    }


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def validate_teacher_row(
    row: Mapping[str, Any],
    semantic_ids: Mapping[str, Sequence[int]],
    *,
    candidate_limit: int,
) -> tuple[list[str], list[float]]:
    if row.get("invalid"):
        raise ValueError(f"index={row.get('index')}: teacher row is marked invalid")
    skill_ids = [str(value) for value in row.get("ranked_skill_ids") or []]
    raw_scores = row.get("ranked_scores") or []
    if len(skill_ids) != len(raw_scores) or not skill_ids:
        raise ValueError(
            f"index={row.get('index')}: teacher ids/scores are empty or misaligned"
        )
    if len(set(skill_ids)) != len(skill_ids):
        raise ValueError(f"index={row.get('index')}: teacher ranking has duplicate skills")
    if len(skill_ids) < candidate_limit:
        raise ValueError(
            f"index={row.get('index')}: teacher ranking has {len(skill_ids)} entries, "
            f"fewer than candidate_limit={candidate_limit}"
        )
    scores = [float(value) for value in raw_scores]
    if any(not math.isfinite(value) for value in scores):
        raise ValueError(f"index={row.get('index')}: teacher ranking has non-finite scores")
    # The frozen panel stores the sorted top-50 retrieval domain first, then
    # appends any missing GT skills with their independently computed scores.
    # Only the retrieval domain carries ranking semantics.
    ranked_domain_scores = scores[:candidate_limit]
    if any(
        left < right
        for left, right in zip(ranked_domain_scores, ranked_domain_scores[1:])
    ):
        raise ValueError(
            f"index={row.get('index')}: teacher candidate-domain scores are not non-increasing"
        )
    teacher_gold = row_gold_skill_ids(row)
    retrieval_domain = set(skill_ids[:candidate_limit])
    expected_suffix = [skill_id for skill_id in teacher_gold if skill_id not in retrieval_domain]
    observed_suffix = skill_ids[candidate_limit:]
    if len(observed_suffix) != len(expected_suffix) or set(observed_suffix) != set(
        expected_suffix
    ):
        raise ValueError(
            f"index={row.get('index')}: teacher suffix is not exactly the missing-GT set"
        )
    missing = [skill_id for skill_id in skill_ids if skill_id not in semantic_ids]
    if missing:
        raise ValueError(
            f"index={row.get('index')}: teacher candidates missing LETTER IDs: {missing[:5]}"
        )
    return skill_ids, scores


def _coverage_payload(
    *,
    covered_mentions: int,
    gold_mentions: int,
    rows_with_any: int,
    rows_with_all: int,
    rows: int,
) -> dict[str, float | int]:
    return {
        "covered_gold_mentions": covered_mentions,
        "gold_mentions": gold_mentions,
        "gold_mention_coverage": covered_mentions / gold_mentions,
        "rows_with_any_gold_covered": rows_with_any,
        "rows_with_all_golds_covered": rows_with_all,
        "rows": rows,
    }


def analyze(
    source_rows: Iterable[Mapping[str, Any]],
    teacher_rows: Iterable[Mapping[str, Any]],
    semantic_ids: Mapping[str, Sequence[int]],
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    candidate_limit: int = 50,
    manifest_writer: Callable[[Mapping[str, Any]], None] | None = None,
    collect_manifest: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_top_ks = tuple(sorted(set(int(value) for value in top_ks)))
    if not normalized_top_ks or normalized_top_ks[0] <= 0:
        raise ValueError("top-K values must be positive")
    if normalized_top_ks[-1] > candidate_limit:
        raise ValueError("top-K values cannot exceed candidate_limit")

    depth = len(next(iter(semantic_ids.values())))
    if depth < 2 or any(len(values) != depth for values in semantic_ids.values()):
        raise ValueError("semantic IDs must have one shared depth of at least two")
    skills_by_c0: dict[int, list[str]] = defaultdict(list)
    for skill_id, codes in semantic_ids.items():
        skills_by_c0[int(codes[0])].append(skill_id)

    seen_indices: set[Any] = set()
    manifest_rows: list[dict[str, Any]] = []
    gold_mentions = 0
    multi_gt_rows = 0
    collision_mentions = 0
    structural_c1_counts: list[int] = []
    structural_c1_zero_mentions = 0
    selected_c1_ranks: list[float] = []
    selected_c1_margins: list[float] = []
    nonpositive_selected_c1_margins = 0
    gold_mentions_by_row_type: Counter[str] = Counter()
    selected_c1_by_row_type: Counter[str] = Counter()
    divergence_counter: Counter[str] = Counter()
    covered_mentions_by_k: dict[int, Counter[int]] = {
        top_k: Counter() for top_k in normalized_top_ks
    }
    rows_with_any_by_k: dict[int, Counter[int]] = {
        top_k: Counter() for top_k in normalized_top_ks
    }
    rows_with_all_by_k: dict[int, Counter[int]] = {
        top_k: Counter() for top_k in normalized_top_ks
    }

    sentinel = object()
    row_count = 0
    for row_position, pair in enumerate(
        zip_longest(source_rows, teacher_rows, fillvalue=sentinel)
    ):
        source, teacher = pair
        if source is sentinel or teacher is sentinel:
            raise ValueError(
                f"source/teacher row count mismatch at row_position={row_position}"
            )
        row_count += 1
        source_index = source.get("index")
        teacher_index = teacher.get("index")
        if source_index is None or teacher_index is None:
            raise ValueError(f"row_position={row_position}: source/teacher index is missing")
        if source_index != teacher_index:
            raise ValueError(
                f"row_position={row_position}: source/teacher index mismatch "
                f"{source_index} != {teacher_index}"
            )
        if source_index in seen_indices:
            raise ValueError(f"duplicate source index={source_index}")
        seen_indices.add(source_index)
        source_query = row_query(source)
        teacher_query = row_query(teacher)
        if source_query != teacher_query:
            raise ValueError(f"index={source_index}: source/teacher query mismatch")
        source_gold = row_gold_skill_ids(source)
        teacher_gold = row_gold_skill_ids(teacher)
        if set(source_gold) != set(teacher_gold):
            raise ValueError(
                f"index={source_index}: source/teacher gold mismatch "
                f"{source_gold} != {teacher_gold}"
            )
        missing_gold = [skill_id for skill_id in source_gold if skill_id not in semantic_ids]
        if missing_gold:
            raise ValueError(
                f"index={source_index}: gold skills missing LETTER IDs: {missing_gold}"
            )
        ranked_ids, ranked_scores = validate_teacher_row(
            teacher,
            semantic_ids,
            candidate_limit=candidate_limit,
        )
        score_by_skill = dict(zip(ranked_ids, ranked_scores))
        unscored_gold = [skill_id for skill_id in source_gold if skill_id not in score_by_skill]
        if unscored_gold:
            raise ValueError(
                f"index={source_index}: gold skills missing frozen teacher scores: {unscored_gold}"
            )
        if len(source_gold) > 1:
            multi_gt_rows += 1
        row_type = "multi" if len(source_gold) > 1 else "single"
        gold_set = set(source_gold)
        ranked_candidates = [
            (rank, skill_id, score)
            for rank, (skill_id, score) in enumerate(
                zip(ranked_ids[:candidate_limit], ranked_scores[:candidate_limit]),
                start=1,
            )
            if skill_id not in gold_set
        ]

        gold_payloads: list[dict[str, Any]] = []
        positions_for_row_by_k: dict[int, list[set[int]]] = {
            top_k: [] for top_k in normalized_top_ks
        }
        for gold_skill_id in source_gold:
            gold_mentions += 1
            gold_mentions_by_row_type[row_type] += 1
            gold_codes = tuple(int(value) for value in semantic_ids[gold_skill_id])
            has_collision = False
            positions_by_top_k: dict[int, set[int]] = {
                top_k: set() for top_k in normalized_top_ks
            }
            selected_by_divergence: dict[int, dict[str, Any]] = {}
            for rank, negative_skill_id, negative_score in ranked_candidates:
                negative_codes = tuple(int(value) for value in semantic_ids[negative_skill_id])
                divergence = first_divergence(gold_codes, negative_codes)
                divergence_label = "collision" if divergence is None else f"c{divergence}"
                divergence_counter[divergence_label] += 1
                record = {
                    "skill_id": negative_skill_id,
                    "teacher_rank": rank,
                    "teacher_score": negative_score,
                    "first_divergence_position": divergence,
                    "semantic_id": list(negative_codes),
                }
                if divergence is None:
                    has_collision = True
                    continue
                selected_by_divergence.setdefault(divergence, record)
                for top_k in normalized_top_ks:
                    if rank <= top_k:
                        positions_by_top_k[top_k].add(divergence)

            structural_c1_count = sum(
                1
                for skill_id in skills_by_c0[gold_codes[0]]
                if skill_id not in gold_set
                and int(semantic_ids[skill_id][1]) != gold_codes[1]
            )
            structural_c1_counts.append(structural_c1_count)
            structural_c1_zero_mentions += int(structural_c1_count == 0)
            selected_c1 = selected_by_divergence.get(1)
            if selected_c1 is not None:
                selected_c1 = dict(selected_c1)
                selected_c1["gold_minus_negative_teacher_margin"] = (
                    score_by_skill[gold_skill_id] - float(selected_c1["teacher_score"])
                )
                selected_c1_ranks.append(float(selected_c1["teacher_rank"]))
                selected_c1_margins.append(
                    float(selected_c1["gold_minus_negative_teacher_margin"])
                )
                nonpositive_selected_c1_margins += int(
                    float(selected_c1["gold_minus_negative_teacher_margin"]) <= 0.0
                )
                selected_c1_by_row_type[row_type] += 1
            for top_k in normalized_top_ks:
                covered_mentions_by_k[top_k].update(positions_by_top_k[top_k])
                positions_for_row_by_k[top_k].append(positions_by_top_k[top_k])
            collision_mentions += int(has_collision)
            gold_payloads.append(
                {
                    "gold_skill_id": gold_skill_id,
                    "gold_semantic_id": list(gold_codes),
                    "gold_teacher_score": score_by_skill[gold_skill_id],
                    "structural_c1_negative_count": structural_c1_count,
                    "selected_c1_negative": selected_c1,
                    "covered_divergence_positions_by_top_k": {
                        str(top_k): sorted(positions_by_top_k[top_k])
                        for top_k in normalized_top_ks
                    },
                }
            )
        for top_k in normalized_top_ks:
            for position in range(depth):
                if any(
                    position in positions
                    for positions in positions_for_row_by_k[top_k]
                ):
                    rows_with_any_by_k[top_k][position] += 1
                if all(
                    position in positions
                    for positions in positions_for_row_by_k[top_k]
                ):
                    rows_with_all_by_k[top_k][position] += 1
        manifest_row = {
            "row_position": row_position,
            "index": source_index,
            "query_sha256": hashlib.sha256(source_query.encode("utf-8")).hexdigest(),
            "gold_skill_ids": source_gold,
            "teacher_candidate_limit": candidate_limit,
            "gold_candidates": gold_payloads,
        }
        if manifest_writer is not None:
            manifest_writer(manifest_row)
        if collect_manifest:
            manifest_rows.append(manifest_row)

    if row_count == 0:
        raise ValueError("source/teacher inputs are empty")

    coverage_by_top_k: dict[str, Any] = {}
    for top_k in normalized_top_ks:
        coverage_by_top_k[str(top_k)] = {
            f"first_divergence_c{position}": _coverage_payload(
                covered_mentions=covered_mentions_by_k[top_k][position],
                gold_mentions=gold_mentions,
                rows_with_any=rows_with_any_by_k[top_k][position],
                rows_with_all=rows_with_all_by_k[top_k][position],
                rows=row_count,
            )
            for position in range(depth)
        }

    selected_c1_mentions = len(selected_c1_ranks)
    summary = {
        "protocol": {
            "semantic_depth": depth,
            "candidate_limit": candidate_limit,
            "top_ks": list(normalized_top_ks),
            "candidate_domain": "frozen_teacher_rank_positions_then_exclude_all_sibling_gt",
            "target": "same_c0_first_divergence_at_c1",
        },
        "counts": {
            "rows": row_count,
            "multi_gt_rows": multi_gt_rows,
            "gold_mentions": gold_mentions,
            "semantic_id_skills": len(semantic_ids),
            "selected_c1_gold_mentions": selected_c1_mentions,
            "gold_mentions_with_teacher_top50_code_collision": collision_mentions,
        },
        "coverage_by_top_k": coverage_by_top_k,
        "selected_c1_top50": {
            "gold_mentions": gold_mentions,
            "covered_gold_mentions": selected_c1_mentions,
            "coverage": selected_c1_mentions / gold_mentions,
            "coverage_by_row_type": {
                row_type: {
                    "gold_mentions": gold_mentions_by_row_type[row_type],
                    "covered_gold_mentions": selected_c1_by_row_type[row_type],
                    "coverage": safe_ratio(
                        selected_c1_by_row_type[row_type],
                        gold_mentions_by_row_type[row_type],
                    ),
                }
                for row_type in ("single", "multi")
            },
            "teacher_rank": numeric_summary(selected_c1_ranks),
            "gold_minus_negative_teacher_margin": numeric_summary(selected_c1_margins),
            "nonpositive_gold_minus_negative_margin_mentions": (
                nonpositive_selected_c1_margins
            ),
            "nonpositive_gold_minus_negative_margin_rate": (
                nonpositive_selected_c1_margins / selected_c1_mentions
            ),
            "structural_candidate_count": numeric_summary(structural_c1_counts),
            "structural_zero_candidate_gold_mentions": structural_c1_zero_mentions,
            "structural_positive_candidate_gold_mentions": (
                gold_mentions - structural_c1_zero_mentions
            ),
            "coverage_conditional_on_structural_candidate": (
                safe_ratio(
                    selected_c1_mentions,
                    gold_mentions - structural_c1_zero_mentions,
                )
            ),
        },
        "teacher_candidate_first_divergence_counts": dict(sorted(divergence_counter.items())),
        "verification": {
            "passed": True,
            "source_teacher_alignment": True,
            "query_alignment": True,
            "gold_alignment": True,
            "teacher_ranking_valid": True,
            "semantic_id_coverage": True,
            "sibling_gt_exclusion": True,
            "manifest_rows": row_count,
            "manifest_gold_mentions": gold_mentions,
        },
    }
    return summary, manifest_rows


def analyze_teacher_prefix(
    source_rows: Iterable[Mapping[str, Any]],
    teacher_rows: Iterable[Mapping[str, Any]],
    semantic_ids: Mapping[str, Sequence[int]],
    *,
    expected_rows: int,
    expected_teacher_trailing_rows: int,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    candidate_limit: int = 50,
    manifest_writer: Callable[[Mapping[str, Any]], None] | None = None,
    collect_manifest: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Analyze an exact teacher prefix and audit the unused trailing row count."""
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if expected_teacher_trailing_rows < 0:
        raise ValueError("expected_teacher_trailing_rows must be non-negative")

    teacher_iterator = iter(teacher_rows)
    summary, manifest_rows = analyze(
        source_rows,
        islice(teacher_iterator, expected_rows),
        semantic_ids,
        top_ks=top_ks,
        candidate_limit=candidate_limit,
        manifest_writer=manifest_writer,
        collect_manifest=collect_manifest,
    )
    if summary["counts"]["rows"] != expected_rows:
        raise ValueError(
            f"source rows={summary['counts']['rows']} does not match expected={expected_rows}"
        )

    teacher_trailing_rows = sum(1 for _ in teacher_iterator)
    if teacher_trailing_rows != expected_teacher_trailing_rows:
        raise ValueError(
            f"teacher trailing rows={teacher_trailing_rows} does not match "
            f"expected={expected_teacher_trailing_rows}"
        )
    summary["protocol"]["teacher_scope"] = "exact_aligned_prefix"
    summary["verification"].update(
        {
            "teacher_prefix_rows": expected_rows,
            "teacher_trailing_rows": teacher_trailing_rows,
            "expected_teacher_trailing_rows": expected_teacher_trailing_rows,
        }
    )
    return summary, manifest_rows


def parse_top_ks(value: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(item) for item in value.split(",") if item.strip())))
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("top-K values must be a positive comma-separated list")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--teacher-panel", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=100_078)
    parser.add_argument("--expected-teacher-trailing-rows", type=int, default=0)
    parser.add_argument("--semantic-depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--top-ks", type=parse_top_ks, default=DEFAULT_TOP_KS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_paths = {
        "summary": args.output_dir / "c1_hard_negative_coverage_summary.json",
        "manifest": args.output_dir / "c1_hard_negative_candidates.jsonl",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    semantic_ids = load_semantic_ids(args.semantic_ids, depth=args.semantic_depth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_manifest_name = tempfile.mkstemp(
        prefix=".c1_hard_negative_candidates.",
        suffix=".jsonl.tmp",
        dir=args.output_dir,
    )
    temporary_manifest = Path(temporary_manifest_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as manifest_handle:
            summary, _ = analyze_teacher_prefix(
                iter_jsonl(args.rows),
                iter_jsonl(args.teacher_panel),
                semantic_ids,
                expected_rows=args.expected_rows,
                expected_teacher_trailing_rows=args.expected_teacher_trailing_rows,
                top_ks=args.top_ks,
                candidate_limit=args.candidate_limit,
                manifest_writer=lambda row: manifest_handle.write(
                    json.dumps(row, sort_keys=True) + "\n"
                ),
                collect_manifest=False,
            )
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
        manifest_sha256 = file_sha256(temporary_manifest)
        os.replace(temporary_manifest, output_paths["manifest"])
        os.chmod(output_paths["manifest"], 0o644)
    except BaseException:
        try:
            temporary_manifest.unlink()
        except FileNotFoundError:
            pass
        raise
    summary["inputs"] = {
        "rows": str(args.rows),
        "rows_sha256": file_sha256(args.rows),
        "teacher_panel": str(args.teacher_panel),
        "teacher_panel_sha256": file_sha256(args.teacher_panel),
        "semantic_ids": str(args.semantic_ids),
        "semantic_ids_sha256": file_sha256(args.semantic_ids),
    }
    summary["outputs"] = {
        "candidate_manifest": str(output_paths["manifest"]),
        "candidate_manifest_sha256": manifest_sha256,
    }
    atomic_write_text(
        output_paths["summary"],
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
