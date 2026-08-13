#!/usr/bin/env python3
"""Shard training rows and compact GR beam outputs into query-bound hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, TextIO

from prepare_sequence_contrastive_hard_negatives import load_training_bindings, query_digest
from skill_gr_pipeline import file_sha256, load_semantic_ids_with_mode


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def shard_training_rows(rows_path: Path, output_dir: Path, num_shards: int) -> dict[str, Any]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite shard directory: {output_dir}")
    output_dir.mkdir(parents=True)
    shard_paths = [output_dir / f"shard_{index:02d}.jsonl" for index in range(num_shards)]
    handles: list[TextIO] = [path.open("w", encoding="utf-8") for path in shard_paths]
    counts = [0] * num_shards
    source_digest = hashlib.sha256()
    row_count = 0
    try:
        with rows_path.open("rb") as source:
            for line_number, raw_line in enumerate(source, start=1):
                source_digest.update(raw_line)
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                selected = [str(value) for value in row.get("selected_skill_ids", []) or []]
                if len(selected) != 1:
                    raise ValueError(
                        f"training row ordinal={row_count} must have exactly one selected skill"
                    )
                query_digest(row)
                payload_index = row.get("index")
                row["_self_mining_payload_index"] = payload_index
                row["index"] = row_count
                shard_index = row_count % num_shards
                handles[shard_index].write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                counts[shard_index] += 1
                row_count += 1
    finally:
        for handle in handles:
            handle.close()
    if row_count == 0:
        raise ValueError(f"training rows are empty: {rows_path}")
    shards = [
        {"index": index, "path": str(path), "rows": counts[index], "sha256": file_sha256(path)}
        for index, path in enumerate(shard_paths)
    ]
    manifest = {
        "format": "gr_self_mining_shards_v1",
        "rows": str(rows_path),
        "rows_sha256": source_digest.hexdigest(),
        "row_count": row_count,
        "num_shards": num_shards,
        "assignment": "zero_based_training_row_ordinal_mod_num_shards",
        "shards": shards,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _load_eval_manifest(
    predictions_path: Path,
    *,
    expected_model_dir: Path,
    expected_beams: int,
) -> dict[str, Any]:
    manifest_path = predictions_path.parent / "eval_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing eval manifest beside predictions: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_model = Path(str(manifest.get("model_dir") or "")).resolve(strict=True)
    if observed_model != expected_model_dir.resolve(strict=True):
        raise ValueError(
            f"mining checkpoint mismatch: expected={expected_model_dir}, observed={observed_model}"
        )
    required = {
        "num_beams": expected_beams,
        "max_skills": 1,
        "rank_aggregation": "beam_order",
        "decoder_head": "ce",
        "score_transform": "logprob",
        "distributed_world_size": 1,
    }
    drift = {key: (value, manifest.get(key)) for key, value in required.items() if manifest.get(key) != value}
    if drift:
        raise ValueError(f"mining evaluation protocol drifted for {predictions_path}: {drift}")
    return manifest


def compact_self_mined_candidates(
    *,
    rows_path: Path,
    prediction_paths: list[Path],
    semantic_ids_path: Path,
    model_dir: Path,
    output_dir: Path,
    expected_negatives: int,
    expected_beams: int,
) -> dict[str, Any]:
    if expected_negatives <= 0:
        raise ValueError(f"expected_negatives must be positive, got {expected_negatives}")
    if expected_beams <= expected_negatives:
        raise ValueError(
            f"expected_beams must exceed expected_negatives: {expected_beams} <= {expected_negatives}"
        )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {output_dir}")
    output_dir.mkdir(parents=True)
    bindings, rows_sha256 = load_training_bindings(rows_path)
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(semantic_ids_path)
    known_skill_ids = set(semantic_ids)
    records: dict[int, dict[str, Any]] = {}
    prediction_manifests = []
    ranked_lengths: list[int] = []
    gold_ranks: list[int | None] = []
    for predictions_path in prediction_paths:
        prediction_manifests.append(
            {
                "predictions": str(predictions_path),
                "predictions_sha256": file_sha256(predictions_path),
                "eval_manifest": _load_eval_manifest(
                    predictions_path,
                    expected_model_dir=model_dir,
                    expected_beams=expected_beams,
                ),
            }
        )
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                prediction = json.loads(line)
                row_index = int(prediction["index"])
                if row_index in records:
                    raise ValueError(f"duplicate mined row_index={row_index}")
                binding = bindings.get(str(row_index))
                if binding is None:
                    raise ValueError(f"unknown mined row_index={row_index}")
                gold = [str(value) for value in prediction.get("gold_skill_ids", []) or []]
                positive = binding["positive_skill_id"]
                if gold != [positive]:
                    raise ValueError(
                        f"mined gold mismatch at row_index={row_index}: {gold} != {[positive]}"
                    )
                if query_digest(prediction) != binding["query_sha256"]:
                    raise ValueError(f"mined query mismatch at row_index={row_index}")
                ranked = [str(value) for value in prediction.get("ranked_skill_ids", []) or []]
                ranked_lengths.append(len(ranked))
                gold_ranks.append(ranked.index(positive) + 1 if positive in ranked else None)
                negatives: list[str] = []
                seen: set[str] = set()
                for skill_id in ranked:
                    if skill_id == positive or skill_id in seen:
                        continue
                    if skill_id not in known_skill_ids:
                        raise ValueError(
                            f"unknown mined skill={skill_id!r} at row_index={row_index}"
                        )
                    seen.add(skill_id)
                    negatives.append(skill_id)
                    if len(negatives) == expected_negatives:
                        break
                if len(negatives) != expected_negatives:
                    raise ValueError(
                        "insufficient unique GR near-misses at "
                        f"row_index={row_index}: expected={expected_negatives}, got={len(negatives)}, "
                        f"ranked={len(ranked)}, file={predictions_path}, line={line_number}"
                    )
                records[row_index] = {
                    "row_index": row_index,
                    "source_index": binding["source_index"],
                    "query_sha256": binding["query_sha256"],
                    "positive_skill_id": positive,
                    "negative_skill_ids": negatives,
                }
    expected_indices = set(range(len(bindings)))
    observed_indices = set(records)
    if observed_indices != expected_indices:
        missing = sorted(expected_indices - observed_indices)
        extra = sorted(observed_indices - expected_indices)
        raise ValueError(
            f"mined row closure failed: missing={len(missing)} first={missing[:5]}, "
            f"extra={len(extra)} first={extra[:5]}"
        )
    output_path = output_dir / "candidates.jsonl"
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for row_index in sorted(records):
            output.write(
                json.dumps(records[row_index], ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary_path.replace(output_path)
    present_gold_ranks = [rank for rank in gold_ranks if rank is not None]
    audit = {
        "rows": len(records),
        "training_rows": len(bindings),
        "negatives_per_row": expected_negatives,
        "candidate_paths_per_row": expected_negatives + 1,
        "known_unique_non_gold_negatives": True,
        "query_binding": "sha256",
        "join_key": "zero_based_training_row_ordinal_to_candidates.row_index",
        "min_ranked_skills": min(ranked_lengths),
        "max_ranked_skills": max(ranked_lengths),
        "gold_top1_rows": sum(rank == 1 for rank in gold_ranks),
        "gold_in_beam_rows": len(present_gold_ranks),
        "gold_missing_from_beam_rows": len(gold_ranks) - len(present_gold_ranks),
    }
    manifest = {
        "format": "sequence_contrastive_gr_self_mined_candidates_v1",
        "rows": str(rows_path),
        "rows_sha256": rows_sha256,
        "semantic_ids": str(semantic_ids_path),
        "semantic_ids_sha256": file_sha256(semantic_ids_path),
        "semantic_id_mode": semantic_id_mode,
        "mining_model_dir": str(model_dir.resolve(strict=True)),
        "mining_checkpoint_sha256": file_sha256(model_dir / "structured_decoder.pt"),
        "mining_protocol": {
            "num_beams": expected_beams,
            "max_skills": 1,
            "rank_aggregation": "beam_order",
            "decoder_head": "ce",
            "score_transform": "logprob",
        },
        "prediction_shards": prediction_manifests,
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "audit": audit,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard_parser = subparsers.add_parser("shard")
    shard_parser.add_argument("--rows", type=Path, required=True)
    shard_parser.add_argument("--output-dir", type=Path, required=True)
    shard_parser.add_argument("--num-shards", type=int, default=8)

    compact_parser = subparsers.add_parser("compact")
    compact_parser.add_argument("--rows", type=Path, required=True)
    compact_parser.add_argument("--predictions", type=Path, action="append", required=True)
    compact_parser.add_argument("--semantic-ids", type=Path, required=True)
    compact_parser.add_argument("--model-dir", type=Path, required=True)
    compact_parser.add_argument("--output-dir", type=Path, required=True)
    compact_parser.add_argument("--expected-negatives", type=int, default=10)
    compact_parser.add_argument("--expected-beams", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "shard":
        manifest = shard_training_rows(args.rows, args.output_dir, args.num_shards)
    elif args.command == "compact":
        manifest = compact_self_mined_candidates(
            rows_path=args.rows,
            prediction_paths=args.predictions,
            semantic_ids_path=args.semantic_ids,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            expected_negatives=args.expected_negatives,
            expected_beams=args.expected_beams,
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
