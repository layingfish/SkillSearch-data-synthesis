#!/usr/bin/env python3
"""Fail-closed provenance and result validation for legacy baseline parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_SEMANTIC_IDS = 14_551
EXPECTED_QUERY_TRAIN_ROWS = 129_341
EXPECTED_QUERY_DEV_ROWS = 2_501
EXPECTED_SOURCE_TRAIN_ROWS = 100_078
EXPECTED_OFFICIAL_ROWS = 75
EXPECTED_GOLD_MENTIONS = 200
EXPECTED_SINGLE_ROWS = 24
EXPECTED_MULTI_ROWS = 51
EXPECTED_SINGLE_GOLD = 24
EXPECTED_MULTI_GOLD = 176
EXPECTED_TRAIN_BATCH = 16
EXPECTED_EVAL_BATCH = 128
EXPECTED_CATALOG_EPOCHS = 3
EXPECTED_QUERY_EPOCHS = 5
EXPECTED_CATALOG_STEPS = 2_730
EXPECTED_QUERY_STEPS = 40_420
EXPECTED_SEED = 20_260_626
EXPECTED_BEAMS = 50
EXPECTED_MAX_SOURCE_LENGTH = 2_048
EXPECTED_SEMANTIC_CANONICAL_SHA256 = (
    "da2fff997b1979f7d51996c5ddbe4593a18d23bdfd546c61dd61fb783567349c"
)
EXPECTED_SOURCE_TEXT_POLICY = "skillrouter_query_1500_chars_with_instruction_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--new-run", type=Path, required=True)
    preflight.add_argument("--archived-run", type=Path, required=True)
    preflight.add_argument("--archived-eval-dir", type=Path, required=True)
    preflight.add_argument("--rows", type=Path, required=True)
    preflight.add_argument("--dev-rows", type=Path, required=True)
    preflight.add_argument("--official-rows", type=Path, required=True)
    preflight.add_argument("--skills-jsonl", type=Path, required=True)
    preflight.add_argument("--semantic-ids", type=Path, required=True)
    preflight.add_argument("--model", type=Path, required=True)
    preflight.add_argument("--new-checkpoint-sha256", required=True)
    preflight.add_argument("--new-train-manifest-sha256", required=True)
    preflight.add_argument("--archived-predictions-sha256", required=True)
    preflight.add_argument("--archived-eval-manifest-sha256", required=True)
    preflight.add_argument("--rows-sha256", required=True)
    preflight.add_argument("--dev-rows-sha256", required=True)
    preflight.add_argument("--official-rows-sha256", required=True)
    preflight.add_argument("--skills-sha256", required=True)
    preflight.add_argument("--semantic-ids-file-sha256", required=True)
    preflight.add_argument("--archived-tokenizer-json-sha256", required=True)
    preflight.add_argument("--new-tokenizer-json-sha256", required=True)
    preflight.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--new-run", type=Path, required=True)
    summarize.add_argument("--archived-eval-dir", type=Path, required=True)
    summarize.add_argument("--new-eval-dir", type=Path, required=True)
    summarize.add_argument("--official-rows", type=Path, required=True)
    summarize.add_argument("--semantic-ids", type=Path, required=True)
    summarize.add_argument("--provenance", type=Path, required=True)
    summarize.add_argument("--tokenizer-parity", type=Path, required=True)
    summarize.add_argument("--output-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in: {path}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r}, actual={actual!r}")


def require_hash(label: str, path: Path, expected: str) -> str:
    actual = file_sha256(path)
    require_equal(f"{label} SHA-256", actual, expected)
    return actual


def resolved(path: Path | str) -> Path:
    return Path(path).resolve(strict=True)


def require_same_path(label: str, actual: Path | str, expected: Path | str) -> None:
    require_equal(label, resolved(actual), resolved(expected))


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_index")),
        str(row.get("index")),
        str(row.get("query")),
    )


def validate_predictions(
    label: str,
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    require_equal(f"{label} prediction rows", len(rows), EXPECTED_OFFICIAL_ROWS)
    keys = [prediction_key(row) for row in rows]
    require_equal(f"{label} unique prediction keys", len(set(keys)), EXPECTED_OFFICIAL_ROWS)
    total_gold = 0
    for index, row in enumerate(rows):
        gold = row.get("gold_skill_ids")
        ranking = row.get("ranked_skill_ids")
        if not isinstance(gold, list) or not gold:
            raise ValueError(f"{label} row {index} has invalid gold_skill_ids")
        if not isinstance(ranking, list):
            raise ValueError(f"{label} row {index} has invalid ranked_skill_ids")
        require_equal(f"{label} row {index} ranking length", len(ranking), EXPECTED_BEAMS)
        require_equal(f"{label} row {index} unique ranking length", len(set(ranking)), EXPECTED_BEAMS)
        total_gold += len(gold)
    require_equal(f"{label} gold mentions", total_gold, EXPECTED_GOLD_MENTIONS)
    return dict(zip(keys, rows))


def validate_eval_manifest(
    label: str,
    manifest: dict[str, Any],
    model_dir: Path,
    official_rows: Path,
    semantic_ids: Path,
    expected_split: str | None,
) -> None:
    require_same_path(f"{label} model_dir", manifest["model_dir"], model_dir)
    require_same_path(f"{label} rows", manifest["rows"], official_rows)
    require_same_path(f"{label} semantic_ids", manifest["semantic_ids"], semantic_ids)
    expected = {
        "num_beams": EXPECTED_BEAMS,
        "max_skills": 1,
        "max_rank": EXPECTED_BEAMS,
        "rank_aggregation": "beam_order",
        "decoder_head": "ce",
        "score_transform": "logprob",
        "distributed_world_size": 8,
        "max_source_length": EXPECTED_MAX_SOURCE_LENGTH,
        "semantic_id_mode": "learned_codes",
        "semantic_ids_sha256": EXPECTED_SEMANTIC_CANONICAL_SHA256,
        "semantic_ids_verified": True,
        "source_text_policy": EXPECTED_SOURCE_TEXT_POLICY,
        "source_text_policy_verified": True,
        "method_setting": "custom",
        "k_values": [1, 10, 50],
    }
    for field, expected_value in expected.items():
        require_equal(f"{label} {field}", manifest.get(field), expected_value)
    if expected_split is not None:
        require_equal(f"{label} split", manifest.get("split"), expected_split)
    ranked_audit = manifest.get("ranked_length_audit") or {}
    require_equal(f"{label} min ranked skills", ranked_audit.get("min_ranked_skills"), EXPECTED_BEAMS)
    require_equal(f"{label} max ranked skills", ranked_audit.get("max_ranked_skills"), EXPECTED_BEAMS)
    require_equal(f"{label} rows below max rank", ranked_audit.get("rows_below_max_rank"), 0)
    source_audit = manifest.get("source_length_audit") or {}
    require_equal(f"{label} source rows", source_audit.get("rows"), EXPECTED_OFFICIAL_ROWS)
    gold_coverage = manifest.get("gold_coverage") or {}
    require_equal(f"{label} gold mentions", gold_coverage.get("gold_skill_mentions"), EXPECTED_GOLD_MENTIONS)
    require_equal(f"{label} missing gold mentions", gold_coverage.get("missing_gold_mentions"), 0)


def command_preflight(args: argparse.Namespace) -> None:
    import torch

    new_checkpoint_path = args.new_run / "last" / "structured_decoder.pt"
    new_manifest_path = args.new_run / "single_sid_decoder_train_manifest.json"
    archived_predictions_path = args.archived_eval_dir / "predictions.jsonl"
    archived_eval_manifest_path = args.archived_eval_dir / "eval_manifest.json"
    hashed_artifacts = {
        "new_last_checkpoint": require_hash(
            "new last checkpoint", new_checkpoint_path, args.new_checkpoint_sha256
        ),
        "new_train_manifest": require_hash(
            "new training manifest", new_manifest_path, args.new_train_manifest_sha256
        ),
        "archived_predictions": require_hash(
            "archived predictions", archived_predictions_path, args.archived_predictions_sha256
        ),
        "archived_eval_manifest": require_hash(
            "archived eval manifest", archived_eval_manifest_path, args.archived_eval_manifest_sha256
        ),
        "train_rows": require_hash("train rows", args.rows, args.rows_sha256),
        "dev_rows": require_hash("dev rows", args.dev_rows, args.dev_rows_sha256),
        "official_rows": require_hash(
            "Official core75 rows", args.official_rows, args.official_rows_sha256
        ),
        "skills": require_hash("skills", args.skills_jsonl, args.skills_sha256),
        "semantic_ids_file": require_hash(
            "semantic IDs file", args.semantic_ids, args.semantic_ids_file_sha256
        ),
        "archived_tokenizer_json": require_hash(
            "archived tokenizer.json",
            args.archived_run / "final" / "tokenizer.json",
            args.archived_tokenizer_json_sha256,
        ),
        "new_tokenizer_json": require_hash(
            "new tokenizer.json",
            args.new_run / "last" / "tokenizer.json",
            args.new_tokenizer_json_sha256,
        ),
    }

    train_manifest = load_json(new_manifest_path)
    expected_manifest = {
        "batch_size": EXPECTED_TRAIN_BATCH,
        "eval_batch_size": EXPECTED_EVAL_BATCH,
        "catalog_rows": EXPECTED_SEMANTIC_IDS,
        "catalog_steps": EXPECTED_CATALOG_STEPS,
        "query_train_rows": EXPECTED_QUERY_TRAIN_ROWS,
        "query_dev_rows": EXPECTED_QUERY_DEV_ROWS,
        "source_train_rows": EXPECTED_SOURCE_TRAIN_ROWS,
        "query_steps": EXPECTED_QUERY_STEPS,
        "seed": EXPECTED_SEED,
        "max_source_length": EXPECTED_MAX_SOURCE_LENGTH,
        "semantic_id_count": EXPECTED_SEMANTIC_IDS,
        "semantic_id_mode": "learned_codes",
        "semantic_ids_sha256": EXPECTED_SEMANTIC_CANONICAL_SHA256,
        "formulation": "catalog indexing followed by query->single SID finetuning",
        "load_best_model_at_end": True,
        "lr": 0.0003,
        "weight_decay": 0.0,
    }
    for field, expected_value in expected_manifest.items():
        require_equal(f"new training manifest {field}", train_manifest.get(field), expected_value)
    require_equal(
        "derived catalog epochs",
        train_manifest["catalog_steps"]
        // math.ceil(train_manifest["catalog_rows"] / train_manifest["batch_size"]),
        EXPECTED_CATALOG_EPOCHS,
    )
    require_equal(
        "derived query epochs",
        train_manifest["query_steps"]
        // math.ceil(train_manifest["query_train_rows"] / train_manifest["batch_size"]),
        EXPECTED_QUERY_EPOCHS,
    )
    require_same_path("new manifest last checkpoint", train_manifest["last_checkpoint"], args.new_run / "last")
    require_same_path("new manifest semantic IDs", train_manifest["semantic_ids"], args.semantic_ids)
    require_same_path("new manifest skills", train_manifest["skills_jsonl"], args.skills_jsonl)
    require_same_path("new manifest model", train_manifest["model"], args.model)

    checkpoint = torch.load(
        new_checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    checkpoint_expected = {
        "seed": EXPECTED_SEED,
        "last_query_step": EXPECTED_QUERY_STEPS,
        "training_formulation": "single_sid_catalog_indexing",
        "codebook_size": 256,
        "depth": 4,
        "dedup_size": 0,
        "semantic_id_mode": "learned_codes",
        "semantic_id_count": EXPECTED_SEMANTIC_IDS,
        "semantic_ids_sha256": EXPECTED_SEMANTIC_CANONICAL_SHA256,
        "max_source_length": EXPECTED_MAX_SOURCE_LENGTH,
        "source_text_policy": EXPECTED_SOURCE_TEXT_POLICY,
    }
    for field, expected_value in checkpoint_expected.items():
        require_equal(f"new checkpoint {field}", checkpoint.get(field), expected_value)
    require_same_path("new checkpoint model", checkpoint["model"], args.model)
    del checkpoint

    loss_history = load_jsonl(args.new_run / "loss_history.jsonl")
    query_train_events = [
        row for row in loss_history if row.get("stage") == "query" and row.get("event") == "train"
    ]
    query_eval_events = [
        row for row in loss_history if row.get("stage") == "query" and row.get("event") == "eval"
    ]
    require_equal("last query train step", query_train_events[-1].get("step"), EXPECTED_QUERY_STEPS)
    require_equal("last query train epoch", query_train_events[-1].get("epoch"), float(EXPECTED_QUERY_EPOCHS))
    require_equal("last query eval step", query_eval_events[-1].get("step"), EXPECTED_QUERY_STEPS)
    require_equal("last query eval epoch", query_eval_events[-1].get("epoch"), float(EXPECTED_QUERY_EPOCHS))

    archived_eval_manifest = load_json(archived_eval_manifest_path)
    validate_eval_manifest(
        "archived eval",
        archived_eval_manifest,
        args.archived_run / "final",
        args.official_rows,
        args.semantic_ids,
        expected_split=None,
    )
    archived_predictions = load_jsonl(archived_predictions_path)
    validate_predictions("archived", archived_predictions)

    result = {
        "artifact_format": "legacy_last_parity_provenance_v1",
        "new_run": str(resolved(args.new_run)),
        "new_checkpoint": str(resolved(args.new_run / "last")),
        "archived_run": str(resolved(args.archived_run)),
        "archived_checkpoint": str(resolved(args.archived_run / "final")),
        "archived_eval_dir": str(resolved(args.archived_eval_dir)),
        "official_rows": str(resolved(args.official_rows)),
        "semantic_ids": str(resolved(args.semantic_ids)),
        "hashed_artifacts": hashed_artifacts,
        "new_checkpoint_metadata": checkpoint_expected,
        "new_training_manifest": expected_manifest,
        "archived_eval_protocol": {
            field: archived_eval_manifest[field]
            for field in (
                "num_beams",
                "max_skills",
                "rank_aggregation",
                "decoder_head",
                "score_transform",
                "distributed_world_size",
                "max_source_length",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


def validate_subset_metrics(label: str, output_root: Path, eval_dir: Path) -> dict[str, Any]:
    expected = {
        "all": (EXPECTED_OFFICIAL_ROWS, EXPECTED_GOLD_MENTIONS),
        "single": (EXPECTED_SINGLE_ROWS, EXPECTED_SINGLE_GOLD),
        "multi": (EXPECTED_MULTI_ROWS, EXPECTED_MULTI_GOLD),
    }
    result = {}
    for subset, (expected_rows, expected_gold) in expected.items():
        path = (
            eval_dir / f"ranked_metrics_{subset}.json"
            if label == "new_last"
            else output_root / f"archived_ranked_metrics_{subset}.json"
        )
        overall = load_json(path)["overall"]
        require_equal(f"{label} {subset} metric rows", overall.get("rows"), expected_rows)
        require_equal(f"{label} {subset} metric gold", overall.get("total_gold"), expected_gold)
        result[subset] = overall
    return result


def command_summarize(args: argparse.Namespace) -> None:
    provenance = load_json(args.provenance)
    tokenizer_parity = load_json(args.tokenizer_parity)
    new_eval_manifest = load_json(args.new_eval_dir / "eval_manifest.json")
    validate_eval_manifest(
        "new eval",
        new_eval_manifest,
        args.new_run / "last",
        args.official_rows,
        args.semantic_ids,
        expected_split="official_core75",
    )

    new_predictions = load_jsonl(args.new_eval_dir / "predictions.jsonl")
    archived_predictions = load_jsonl(args.archived_eval_dir / "predictions.jsonl")
    new_by_key = validate_predictions("new last", new_predictions)
    archived_by_key = validate_predictions("archived", archived_predictions)
    require_equal("new/archive query key set", set(new_by_key), set(archived_by_key))
    for key in new_by_key:
        require_equal(
            f"gold skill IDs for query {key[1]}",
            new_by_key[key]["gold_skill_ids"],
            archived_by_key[key]["gold_skill_ids"],
        )
    exact_top50_rows = sum(
        new_by_key[key]["ranked_skill_ids"] == archived_by_key[key]["ranked_skill_ids"]
        for key in new_by_key
    )

    new_metrics = validate_subset_metrics("new_last", args.output_root, args.new_eval_dir)
    archived_metrics = validate_subset_metrics("archived", args.output_root, args.new_eval_dir)
    expected_tokenizer_rows = (
        EXPECTED_QUERY_TRAIN_ROWS
        + EXPECTED_QUERY_DEV_ROWS
        + EXPECTED_SEMANTIC_IDS
        + EXPECTED_OFFICIAL_ROWS
    )
    require_equal("tokenizer audit rows", tokenizer_parity.get("total_rows"), expected_tokenizer_rows)

    summary = {
        "artifact_format": "legacy_last_checkpoint_parity_v1",
        "new_checkpoint": str(resolved(args.new_run / "last")),
        "new_checkpoint_step": EXPECTED_QUERY_STEPS,
        "new_eval_protocol": {
            field: new_eval_manifest[field]
            for field in (
                "split",
                "num_beams",
                "max_skills",
                "rank_aggregation",
                "decoder_head",
                "score_transform",
                "distributed_world_size",
                "max_source_length",
            )
        },
        "archived_reference": {
            "predictions": str(resolved(args.archived_eval_dir / "predictions.jsonl")),
            "checkpoint": provenance["archived_checkpoint"],
            "protocol": provenance["archived_eval_protocol"],
        },
        "query_rows": len(new_predictions),
        "top50_exact_rows_vs_archived": exact_top50_rows,
        "new_last": new_metrics,
        "archived": archived_metrics,
        "tokenizer_parity": tokenizer_parity,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        command_preflight(args)
    elif args.command == "summarize":
        command_summarize(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
