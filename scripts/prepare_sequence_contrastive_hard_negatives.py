#!/usr/bin/env python3
"""Compact and close SkillsRouter hard negatives for sequence-level GR training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from skill_gr_pipeline import file_sha256, load_semantic_ids_with_mode


def query_digest(row: dict[str, Any]) -> str:
    query = str(row.get("query") or row.get("query_text") or "")
    if not query:
        raise ValueError(f"row has no query text: index={row.get('index')} source_index={row.get('source_index')}")
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def load_training_bindings(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    bindings: dict[str, dict[str, str]] = {}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        row_ordinal = 0
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            key = str(row_ordinal)
            if key in bindings:
                raise ValueError(f"duplicate training row ordinal={row_ordinal}")
            selected = [str(value) for value in row.get("selected_skill_ids", []) or []]
            if len(selected) != 1:
                raise ValueError(
                    f"training row ordinal={row_ordinal} must have exactly one selected skill"
                )
            bindings[key] = {
                "positive_skill_id": selected[0],
                "query_sha256": query_digest(row),
                "source_index": str(row.get("source_index")),
                "payload_index": str(row.get("index")),
            }
            row_ordinal += 1
    if not bindings:
        raise ValueError(f"training rows are empty: {path}")
    return bindings, digest.hexdigest()


def compact_candidates(
    source_path: Path,
    output_path: Path,
    bindings: dict[str, dict[str, str]],
    known_skill_ids: set[str],
    expected_negatives: int,
) -> tuple[dict[str, Any], str]:
    source_digest = hashlib.sha256()
    seen: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with source_path.open("rb") as source, temporary_path.open("w", encoding="utf-8") as output:
        for line_number, raw_line in enumerate(source, start=1):
            source_digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            row_index = row.get("row_index")
            key = str(row_index)
            binding = bindings.get(key)
            if binding is None:
                raise ValueError(
                    f"hard-negative line={line_number} has unknown row_index={row_index}"
                )
            if key in seen:
                raise ValueError(f"duplicate hard-negative row_index={row_index}")
            seen.add(key)
            source_index = row.get("source_index")
            if str(source_index) != binding["source_index"]:
                raise ValueError(
                    f"source_index mismatch for row_index={row_index}: "
                    f"{source_index} != {binding['source_index']}"
                )
            positive = str(row.get("positive_skill_id") or "")
            if positive != binding["positive_skill_id"]:
                raise ValueError(
                    f"positive mismatch for row_index={row_index}: {positive} != {binding['positive_skill_id']}"
                )
            if positive not in known_skill_ids:
                raise ValueError(f"unknown positive skill={positive} for row_index={row_index}")
            if query_digest(row) != binding["query_sha256"]:
                raise ValueError(f"query mismatch for row_index={row_index}")
            negatives = [str(value) for value in row.get("negative_skill_ids", []) or []]
            if len(negatives) != expected_negatives or len(set(negatives)) != expected_negatives:
                raise ValueError(
                    "hard-negative cardinality mismatch for "
                    f"row_index={row_index}: expected={expected_negatives}, got={len(negatives)}, "
                    f"unique={len(set(negatives))}"
                )
            invalid = [sid for sid in negatives if sid == positive or sid not in known_skill_ids]
            if invalid:
                raise ValueError(
                    f"invalid hard negatives for row_index={row_index}: first={invalid[:5]}"
                )
            compact = {
                "row_index": int(row_index),
                "source_index": source_index,
                "query_sha256": binding["query_sha256"],
                "positive_skill_id": positive,
                "negative_skill_ids": negatives,
            }
            output.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")
    missing = set(bindings) - seen
    if missing:
        raise ValueError(
            f"hard-negative source is missing {len(missing)} training rows; first={sorted(missing)[:5]}"
        )
    temporary_path.replace(output_path)
    return (
        {
            "rows": len(seen),
            "training_rows": len(bindings),
            "negatives_per_row": expected_negatives,
            "candidate_paths_per_row": expected_negatives + 1,
            "join_key": "zero_based_training_row_ordinal_to_candidates.row_index",
            "query_binding": "sha256",
            "positive_match": True,
            "unique_known_non_gold_negatives": True,
        },
        source_digest.hexdigest(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-negatives", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.expected_negatives <= 0:
        raise ValueError("expected negatives must be positive")
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    bindings, rows_sha256 = load_training_bindings(args.rows)
    output_path = args.output_dir / "candidates.jsonl"
    audit, source_sha256 = compact_candidates(
        args.hard_negatives,
        output_path,
        bindings,
        set(semantic_ids),
        args.expected_negatives,
    )
    manifest = {
        "format": "sequence_contrastive_candidates_v1",
        "rows": str(args.rows),
        "rows_sha256": rows_sha256,
        "hard_negatives": str(args.hard_negatives),
        "hard_negatives_sha256": source_sha256,
        "semantic_ids": str(args.semantic_ids),
        "semantic_ids_sha256": file_sha256(args.semantic_ids),
        "semantic_id_mode": semantic_id_mode,
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "audit": audit,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
