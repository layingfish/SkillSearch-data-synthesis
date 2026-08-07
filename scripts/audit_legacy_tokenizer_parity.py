#!/usr/bin/env python3
"""Compare two saved tokenizers on every legacy training and evaluation source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--archived-tokenizer", type=Path, required=True)
    parser.add_argument("--new-tokenizer", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--official-rows", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--catalog-input-prefix", default="skill document: ")
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--official-batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_pipeline(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("legacy_skill_gr_pipeline_for_tokenizer_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def update_token_digest(digest: Any, token_ids: list[int]) -> None:
    digest.update(struct.pack("<I", len(token_ids)))
    if token_ids:
        digest.update(struct.pack(f"<{len(token_ids)}I", *token_ids))


def update_tensor_digest(digest: Any, tensor: Any) -> None:
    shape = [int(value) for value in tensor.shape]
    digest.update(struct.pack("<I", len(shape)))
    digest.update(struct.pack(f"<{len(shape)}I", *shape))
    values = [int(value) for value in tensor.reshape(-1).tolist()]
    digest.update(struct.pack("<Q", len(values)))
    if values:
        digest.update(struct.pack(f"<{len(values)}I", *values))


def first_difference(left: list[int], right: list[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def batched(values: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(values), batch_size):
        yield start, values[start : start + batch_size]


def compare_split(
    name: str,
    rows: list[dict[str, Any]],
    pipeline: ModuleType,
    archived_tokenizer: Any,
    new_tokenizer: Any,
    max_source_length: int,
    batch_size: int,
) -> dict[str, Any]:
    texts = [pipeline.decoder_source_text(row) for row in rows]
    archived_unpadded_digest = hashlib.sha256()
    new_unpadded_digest = hashlib.sha256()
    archived_padded_digest = hashlib.sha256()
    new_padded_digest = hashlib.sha256()
    archived_attention_digest = hashlib.sha256()
    new_attention_digest = hashlib.sha256()
    mismatch_rows = 0
    padded_input_mismatch_batches = 0
    attention_mismatch_batches = 0
    first_unpadded_mismatch = None
    first_padded_mismatch = None

    for start, text_batch in batched(texts, batch_size):
        archived_batch = archived_tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        new_batch = new_tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        archived_input_ids = archived_batch["input_ids"]
        new_input_ids = new_batch["input_ids"]
        archived_attention = archived_batch["attention_mask"]
        new_attention = new_batch["attention_mask"]
        update_tensor_digest(archived_padded_digest, archived_input_ids)
        update_tensor_digest(new_padded_digest, new_input_ids)
        update_tensor_digest(archived_attention_digest, archived_attention)
        update_tensor_digest(new_attention_digest, new_attention)
        input_ids_equal = (
            archived_input_ids.shape == new_input_ids.shape
            and bool((archived_input_ids == new_input_ids).all())
        )
        attention_equal = (
            archived_attention.shape == new_attention.shape
            and bool((archived_attention == new_attention).all())
        )
        if not input_ids_equal:
            padded_input_mismatch_batches += 1
        if not attention_equal:
            attention_mismatch_batches += 1
        if first_padded_mismatch is None and (not input_ids_equal or not attention_equal):
            first_padded_mismatch = {
                "batch_start_row": start,
                "batch_rows": len(text_batch),
                "archived_input_shape": list(archived_input_ids.shape),
                "new_input_shape": list(new_input_ids.shape),
                "archived_attention_shape": list(archived_attention.shape),
                "new_attention_shape": list(new_attention.shape),
                "input_ids_equal": input_ids_equal,
                "attention_mask_equal": attention_equal,
            }

        for offset in range(len(text_batch)):
            archived_ids = [
                int(value)
                for value in archived_input_ids[offset][archived_attention[offset].bool()].tolist()
            ]
            new_ids = [
                int(value)
                for value in new_input_ids[offset][new_attention[offset].bool()].tolist()
            ]
            update_token_digest(archived_unpadded_digest, archived_ids)
            update_token_digest(new_unpadded_digest, new_ids)
            if archived_ids == new_ids:
                continue
            mismatch_rows += 1
            if first_unpadded_mismatch is None:
                position = first_difference(archived_ids, new_ids)
                first_unpadded_mismatch = {
                    "row": start + offset,
                    "text_sha256": hashlib.sha256(text_batch[offset].encode("utf-8")).hexdigest(),
                    "token_position": position,
                    "archived_length": len(archived_ids),
                    "new_length": len(new_ids),
                    "archived_window": archived_ids[max(0, position - 3) : position + 4],
                    "new_window": new_ids[max(0, position - 3) : position + 4],
                }

    return {
        "split": name,
        "rows": len(rows),
        "batch_size": batch_size,
        "mismatch_rows": mismatch_rows,
        "padded_input_ids_mismatch_batches": padded_input_mismatch_batches,
        "attention_mask_mismatch_batches": attention_mismatch_batches,
        "archived_unpadded_input_ids_sha256": archived_unpadded_digest.hexdigest(),
        "new_unpadded_input_ids_sha256": new_unpadded_digest.hexdigest(),
        "archived_padded_input_ids_sha256": archived_padded_digest.hexdigest(),
        "new_padded_input_ids_sha256": new_padded_digest.hexdigest(),
        "archived_attention_mask_sha256": archived_attention_digest.hexdigest(),
        "new_attention_mask_sha256": new_attention_digest.hexdigest(),
        "first_unpadded_mismatch": first_unpadded_mismatch,
        "first_padded_mismatch": first_padded_mismatch,
    }


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    batch_sizes = (args.train_batch_size, args.eval_batch_size, args.official_batch_size)
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    if args.max_source_length <= 0:
        raise ValueError("max source length must be positive")

    pipeline = load_pipeline(args.pipeline)
    semantic_ids, _ = pipeline.load_semantic_ids_with_mode(args.semantic_ids)
    query_train_rows = pipeline.expand_single_sid_query_rows(
        pipeline.load_jsonl(args.rows), semantic_ids
    )
    query_dev_rows = pipeline.expand_single_sid_query_rows(
        pipeline.load_jsonl(args.dev_rows), semantic_ids
    )
    catalog_rows = pipeline.build_single_sid_catalog_rows(
        pipeline.load_jsonl(args.skills_jsonl),
        semantic_ids,
        args.catalog_input_prefix,
    )
    official_rows = pipeline.load_jsonl(args.official_rows)

    archived_tokenizer = AutoTokenizer.from_pretrained(
        args.archived_tokenizer, local_files_only=True
    )
    new_tokenizer = AutoTokenizer.from_pretrained(args.new_tokenizer, local_files_only=True)
    split_rows = (
        ("query_train", query_train_rows, args.train_batch_size),
        ("query_dev", query_dev_rows, args.eval_batch_size),
        ("catalog", catalog_rows, args.train_batch_size),
        ("official_core75", official_rows, args.official_batch_size),
    )
    splits = [
        compare_split(
            name,
            rows,
            pipeline,
            archived_tokenizer,
            new_tokenizer,
            args.max_source_length,
            batch_size,
        )
        for name, rows, batch_size in split_rows
    ]
    total_mismatch_rows = sum(split_result["mismatch_rows"] for split_result in splits)
    total_padded_input_mismatch_batches = sum(
        split_result["padded_input_ids_mismatch_batches"] for split_result in splits
    )
    total_attention_mismatch_batches = sum(
        split_result["attention_mask_mismatch_batches"] for split_result in splits
    )
    padding_side_exact = archived_tokenizer.padding_side == new_tokenizer.padding_side
    pad_token_id_exact = archived_tokenizer.pad_token_id == new_tokenizer.pad_token_id
    truncation_side_exact = archived_tokenizer.truncation_side == new_tokenizer.truncation_side
    result = {
        "artifact_format": "legacy_tokenizer_parity_v1",
        "archived_tokenizer": str(args.archived_tokenizer),
        "new_tokenizer": str(args.new_tokenizer),
        "max_source_length": args.max_source_length,
        "batch_sizes": {
            "query_train": args.train_batch_size,
            "query_dev": args.eval_batch_size,
            "catalog": args.train_batch_size,
            "official_core75": args.official_batch_size,
        },
        "archived_tokenizer_json_sha256": file_sha256(args.archived_tokenizer / "tokenizer.json"),
        "new_tokenizer_json_sha256": file_sha256(args.new_tokenizer / "tokenizer.json"),
        "archived_vocab_size": len(archived_tokenizer.get_vocab()),
        "new_vocab_size": len(new_tokenizer.get_vocab()),
        "vocab_exact": archived_tokenizer.get_vocab() == new_tokenizer.get_vocab(),
        "special_tokens_map_exact": (
            archived_tokenizer.special_tokens_map == new_tokenizer.special_tokens_map
        ),
        "archived_padding_side": archived_tokenizer.padding_side,
        "new_padding_side": new_tokenizer.padding_side,
        "padding_side_exact": padding_side_exact,
        "archived_pad_token_id": archived_tokenizer.pad_token_id,
        "new_pad_token_id": new_tokenizer.pad_token_id,
        "pad_token_id_exact": pad_token_id_exact,
        "archived_truncation_side": archived_tokenizer.truncation_side,
        "new_truncation_side": new_tokenizer.truncation_side,
        "truncation_side_exact": truncation_side_exact,
        "splits": splits,
        "total_rows": sum(split_result["rows"] for split_result in splits),
        "total_mismatch_rows": total_mismatch_rows,
        "total_padded_input_ids_mismatch_batches": total_padded_input_mismatch_batches,
        "total_attention_mask_mismatch_batches": total_attention_mismatch_batches,
        "input_ids_exact": (
            total_mismatch_rows == 0 and total_padded_input_mismatch_batches == 0
        ),
        "attention_masks_exact": total_attention_mismatch_batches == 0,
        "training_tensors_exact": (
            total_mismatch_rows == 0
            and total_padded_input_mismatch_batches == 0
            and total_attention_mismatch_batches == 0
            and padding_side_exact
            and pad_token_id_exact
            and truncation_side_exact
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
