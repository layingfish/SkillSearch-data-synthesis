#!/usr/bin/env python3
"""Build a focused GR training subset from self-mined ranking boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from prepare_sequence_contrastive_hard_negatives import query_digest
from skill_gr_pipeline import file_sha256


def load_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            selected = [str(value) for value in row.get("selected_skill_ids", []) or []]
            if len(selected) != 1:
                raise ValueError(f"source row ordinal={len(rows)} must have one gold skill")
            query_digest(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"source rows are empty: {path}")
    return rows, digest.hexdigest()


def load_candidates(path: Path, expected_negatives: int) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_index = int(row["row_index"])
            if row_index in result:
                raise ValueError(f"duplicate candidate row_index={row_index}")
            negatives = [str(value) for value in row.get("negative_skill_ids", []) or []]
            if len(negatives) != expected_negatives or len(set(negatives)) != expected_negatives:
                raise ValueError(
                    f"candidate cardinality drifted at line={line_number}, row_index={row_index}"
                )
            positive = str(row.get("positive_skill_id") or "")
            if not positive or positive in negatives:
                raise ValueError(f"invalid candidate positive at row_index={row_index}")
            result[row_index] = row
    return result


def load_boundaries(
    paths: list[Path],
    source_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                prediction = json.loads(line)
                row_index = int(prediction["index"])
                if row_index in result:
                    raise ValueError(f"duplicate prediction row_index={row_index}")
                if row_index < 0 or row_index >= len(source_rows):
                    raise ValueError(f"prediction row_index is out of range: {row_index}")
                source = source_rows[row_index]
                positive = str(source["selected_skill_ids"][0])
                if [str(value) for value in prediction.get("gold_skill_ids", []) or []] != [positive]:
                    raise ValueError(f"prediction gold mismatch at row_index={row_index}")
                if query_digest(prediction) != query_digest(source):
                    raise ValueError(f"prediction query mismatch at row_index={row_index}")
                ranked = [str(value) for value in prediction.get("ranked_skill_ids", []) or []]
                if len(ranked) < 2:
                    raise ValueError(f"prediction has fewer than two ranked skills: row_index={row_index}")
                scores = prediction.get("ranked_skill_scores") or {}
                if not isinstance(scores, dict):
                    raise ValueError(f"ranked scores are not a mapping: row_index={row_index}")
                best_negative = next((skill_id for skill_id in ranked if skill_id != positive), None)
                if best_negative is None or best_negative not in scores:
                    raise ValueError(f"best-negative score is missing: row_index={row_index}")
                gold_score = scores.get(positive)
                margin = (
                    float(gold_score) - float(scores[best_negative])
                    if gold_score is not None
                    else None
                )
                top1_correct = ranked[0] == positive
                if top1_correct and (margin is None or not math.isfinite(margin) or margin < 0.0):
                    raise ValueError(
                        f"top1-correct row has an invalid boundary margin: row_index={row_index}"
                    )
                result[row_index] = {
                    "top1_correct": top1_correct,
                    "gold_margin": margin,
                    "best_negative_skill_id": best_negative,
                    "prediction_path": str(path),
                    "prediction_line": line_number,
                }
    return result


def build_focused_subset(
    *,
    rows_path: Path,
    candidates_path: Path,
    prediction_paths: list[Path],
    output_dir: Path,
    expected_negatives: int,
    correct_to_error_ratio: float,
) -> dict[str, Any]:
    if expected_negatives <= 0:
        raise ValueError(f"expected_negatives must be positive, got {expected_negatives}")
    if correct_to_error_ratio < 0.0:
        raise ValueError(
            f"correct_to_error_ratio must be non-negative, got {correct_to_error_ratio}"
        )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite focused subset: {output_dir}")
    rows, rows_sha256 = load_rows(rows_path)
    candidates = load_candidates(candidates_path, expected_negatives)
    boundaries = load_boundaries(prediction_paths, rows)
    expected_indices = set(range(len(rows)))
    if set(candidates) != expected_indices or set(boundaries) != expected_indices:
        raise ValueError(
            "focused source closure failed: "
            f"rows={len(rows)}, candidates={len(candidates)}, predictions={len(boundaries)}"
        )
    for row_index, source_row in enumerate(rows):
        candidate = candidates[row_index]
        positive = str(source_row["selected_skill_ids"][0])
        if str(candidate["positive_skill_id"]) != positive:
            raise ValueError(f"candidate positive mismatch at source row={row_index}")
        if str(candidate.get("query_sha256") or "") != query_digest(source_row):
            raise ValueError(f"candidate query mismatch at source row={row_index}")
    error_indices = sorted(
        row_index for row_index, boundary in boundaries.items() if not boundary["top1_correct"]
    )
    if not error_indices:
        raise ValueError("focused subset has no top1-error rows")
    correct_target = min(
        len(rows) - len(error_indices),
        int(round(len(error_indices) * correct_to_error_ratio)),
    )
    correct_ranked = sorted(
        (
            (float(boundary["gold_margin"]), row_index)
            for row_index, boundary in boundaries.items()
            if boundary["top1_correct"]
        ),
        key=lambda item: (item[0], item[1]),
    )
    boundary_correct_indices = sorted(row_index for _, row_index in correct_ranked[:correct_target])
    selected_indices = sorted([*error_indices, *boundary_correct_indices])
    selection_reason = {row_index: "top1_error" for row_index in error_indices}
    selection_reason.update(
        {row_index: "lowest_margin_top1_correct" for row_index in boundary_correct_indices}
    )

    output_dir.mkdir(parents=True)
    rows_output = output_dir / "rows.jsonl"
    candidates_output = output_dir / "candidates.jsonl"
    with rows_output.open("w", encoding="utf-8") as row_handle, candidates_output.open(
        "w", encoding="utf-8"
    ) as candidate_handle:
        for focused_index, source_index in enumerate(selected_indices):
            source_row = rows[source_index]
            row_handle.write(
                json.dumps(source_row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            source_candidate = candidates[source_index]
            boundary = boundaries[source_index]
            focused_candidate = {
                "row_index": focused_index,
                "source_row_index": source_index,
                "source_index": source_candidate.get("source_index"),
                "query_sha256": query_digest(source_row),
                "positive_skill_id": source_candidate["positive_skill_id"],
                "negative_skill_ids": source_candidate["negative_skill_ids"],
                "selection_reason": selection_reason[source_index],
                "mining_gold_margin": boundary["gold_margin"],
                "mining_best_negative_skill_id": boundary["best_negative_skill_id"],
            }
            if focused_candidate["positive_skill_id"] != source_row["selected_skill_ids"][0]:
                raise ValueError(f"focused positive mismatch at source row={source_index}")
            candidate_handle.write(
                json.dumps(focused_candidate, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    selected_correct_margins = [boundaries[index]["gold_margin"] for index in boundary_correct_indices]
    manifest = {
        "format": "gr_hard_focus_subset_v1",
        "source_rows": str(rows_path),
        "source_rows_sha256": rows_sha256,
        "source_candidates": str(candidates_path),
        "source_candidates_sha256": file_sha256(candidates_path),
        "prediction_paths": [str(path) for path in prediction_paths],
        "prediction_sha256": [file_sha256(path) for path in prediction_paths],
        "selection": {
            "source_rows": len(rows),
            "top1_error_rows": len(error_indices),
            "boundary_correct_rows": len(boundary_correct_indices),
            "focused_rows": len(selected_indices),
            "correct_to_error_ratio": correct_to_error_ratio,
            "boundary_correct_margin_max": max(selected_correct_margins, default=None),
            "all_top1_errors_included": True,
        },
        "expected_negatives": expected_negatives,
        "output_rows": str(rows_output),
        "output_rows_sha256": file_sha256(rows_output),
        "output_candidates": str(candidates_output),
        "output_candidates_sha256": file_sha256(candidates_output),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-negatives", type=int, default=10)
    parser.add_argument("--correct-to-error-ratio", type=float, default=1.0)
    args = parser.parse_args()
    manifest = build_focused_subset(
        rows_path=args.rows,
        candidates_path=args.candidates,
        prediction_paths=args.predictions,
        output_dir=args.output_dir,
        expected_negatives=args.expected_negatives,
        correct_to_error_ratio=args.correct_to_error_ratio,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
