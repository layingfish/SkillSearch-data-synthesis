#!/usr/bin/env python3
"""Evaluate catalog document-to-SID indexing with teacher forcing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from distributed_utils import barrier, destroy_distributed, initialize_distributed, shard_with_indices
from skill_gr_pipeline import (
    audit_source_token_lengths,
    build_single_sid_catalog_rows,
    infer_structured_sid_config,
    load_jsonl,
    load_semantic_ids_with_mode,
    load_structured_decoder_checkpoint,
    make_structured_t5_sid_model,
    resolve_eval_max_source_length,
    structured_sid_batch,
    validate_checkpoint_semantic_ids,
    validate_checkpoint_source_text_policy,
    write_json,
    write_jsonl,
)


def _load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_dir)
    except AttributeError as exc:
        config_path = model_dir / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        if not isinstance(config.get("extra_special_tokens"), list):
            raise
        if "object has no attribute 'keys'" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(model_dir, extra_special_tokens={})


def _resolve_architecture_model(checkpoint: dict, requested: Path | None) -> str:
    checkpoint_model = str(checkpoint["model"])
    if Path(checkpoint_model).is_dir():
        return checkpoint_model
    if requested is not None and requested.is_dir():
        return str(requested)
    env_model = os.environ.get("SKILLGR_MODEL", "").strip()
    if env_model and Path(env_model).is_dir():
        return env_model
    raise ValueError(
        f"checkpoint records unavailable architecture {checkpoint_model!r}; "
        "pass --architecture-model or set SKILLGR_MODEL"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--architecture-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialized-rows", type=Path)
    parser.add_argument("--catalog-input-prefix", default="skill document: ")
    parser.add_argument("--max-source-length", type=int)
    parser.add_argument("--allow-source-truncation", action="store_true")
    parser.add_argument("--allow-checkpoint-config-mismatch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"batch size must be positive, got {args.batch_size}")
    context = initialize_distributed(cpu=args.cpu)
    checkpoint = torch.load(args.model_dir / "structured_decoder.pt", map_location="cpu")
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    semantic_ids_verified = validate_checkpoint_semantic_ids(
        checkpoint, semantic_ids, semantic_id_mode
    )
    source_text_policy_verified = validate_checkpoint_source_text_policy(
        checkpoint, args.allow_checkpoint_config_mismatch
    )
    max_source_length, max_source_length_source = resolve_eval_max_source_length(
        args.max_source_length,
        checkpoint,
        args.allow_checkpoint_config_mismatch,
    )
    codebook_size = int(checkpoint["codebook_size"])
    depth = int(checkpoint["depth"])
    dedup_size = int(checkpoint.get("dedup_size", 0))
    sid_config = infer_structured_sid_config(
        semantic_ids,
        codebook_size=codebook_size,
        dedup_size=dedup_size,
        semantic_id_mode=semantic_id_mode,
    )
    if int(sid_config["depth"]) != depth:
        raise ValueError(
            f"semantic ID depth={sid_config['depth']} does not match checkpoint depth={depth}"
        )

    rows = build_single_sid_catalog_rows(
        load_jsonl(args.skills_jsonl), semantic_ids, args.catalog_input_prefix
    )
    tokenizer = _load_tokenizer(args.model_dir)
    if context.is_main:
        source_length_audit = audit_source_token_lengths(
            "catalog_eval",
            rows,
            tokenizer,
            max_source_length,
            args.allow_source_truncation,
        )
        if args.materialized_rows:
            write_jsonl(args.materialized_rows, rows)
    else:
        source_length_audit = None
    barrier(context)

    architecture_model = _resolve_architecture_model(checkpoint, args.architecture_model)
    model = make_structured_t5_sid_model(
        architecture_model,
        codebook_size,
        depth,
        dedup_size,
        semantic_id_mode,
        pag_set_vocab_size=int(checkpoint.get("pag_set_vocab_size", 0) or 0),
    )
    load_structured_decoder_checkpoint(
        model,
        args.model_dir,
        context.device,
        semantic_id_mode=semantic_id_mode,
    )
    model.to(context.device).eval()

    control_type = int(model.control_type)
    metric_width = control_type + 1
    type_loss_sum = torch.zeros(metric_width, dtype=torch.float64, device=context.device)
    type_correct = torch.zeros(metric_width, dtype=torch.float64, device=context.device)
    type_count = torch.zeros(metric_width, dtype=torch.float64, device=context.device)
    sid_exact = torch.zeros((), dtype=torch.float64, device=context.device)
    sequence_exact = torch.zeros((), dtype=torch.float64, device=context.device)
    row_count = torch.zeros((), dtype=torch.float64, device=context.device)

    local_rows = [row for _, row in shard_with_indices(rows, context)]
    with torch.no_grad():
        for start in range(0, len(local_rows), args.batch_size):
            batch_rows = local_rows[start : start + args.batch_size]
            batch = structured_sid_batch(
                batch_rows,
                tokenizer,
                semantic_ids,
                codebook_size,
                depth,
                dedup_size,
                max_source_length,
                context.device,
                semantic_id_mode,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=bool(args.bf16 and getattr(context.device, "type", "") == "cuda"),
            ):
                hidden = model._decoder_hidden(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["decoder_input_ids"],
                    batch["decoder_attention_mask"],
                )
            token_correct = torch.zeros_like(batch["labels"], dtype=torch.bool)
            for target_type in range(metric_width):
                mask = batch["target_types"] == target_type
                if target_type < depth:
                    head = model.code_heads[target_type]
                elif model.dedup_head is not None and target_type == depth:
                    head = model.dedup_head
                elif target_type == control_type:
                    head = model.control_head
                else:
                    raise ValueError(f"invalid catalog target type: {target_type}")
                logits = head(hidden[mask].to(dtype=head.weight.dtype))
                labels = batch["labels"][mask]
                float_logits = logits.float()
                type_loss_sum[target_type] += F.cross_entropy(
                    float_logits, labels, reduction="sum"
                ).double()
                correct = float_logits.argmax(dim=-1) == labels
                token_correct[mask] = correct
                type_correct[target_type] += correct.sum().double()
                type_count[target_type] += mask.sum().double()

            valid = batch["target_types"] >= 0
            sid_valid = valid & (batch["target_types"] != control_type)
            sid_exact += (token_correct | ~sid_valid).all(dim=1).sum().double()
            sequence_exact += (token_correct | ~valid).all(dim=1).sum().double()
            row_count += len(batch_rows)

    packed = torch.cat(
        [type_loss_sum, type_correct, type_count, sid_exact[None], sequence_exact[None], row_count[None]]
    )
    if context.enabled:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    width = metric_width
    type_loss_sum = packed[:width]
    type_correct = packed[width : 2 * width]
    type_count = packed[2 * width : 3 * width]
    sid_exact = packed[-3]
    sequence_exact = packed[-2]
    row_count = packed[-1]

    if context.is_main:
        type_names = [f"L{index}" for index in range(depth)]
        if model.dedup_head is not None:
            type_names.append("dedup")
        type_names.append("control")
        by_type = {}
        for target_type, name in enumerate(type_names):
            count = float(type_count[target_type].cpu())
            by_type[name] = {
                "count": int(count),
                "loss": float(type_loss_sum[target_type].cpu()) / max(1.0, count),
                "accuracy": float(type_correct[target_type].cpu()) / max(1.0, count),
            }
        total_loss = float(type_loss_sum.sum().cpu())
        total_count = float(type_count.sum().cpu())
        rows_evaluated = int(float(row_count.cpu()))
        metrics = {
            "rows": rows_evaluated,
            "teacher_forced_ce": total_loss / max(1.0, total_count),
            "teacher_forced_sid_exact_accuracy": float(sid_exact.cpu()) / max(1, rows_evaluated),
            "teacher_forced_sequence_exact_accuracy": float(sequence_exact.cpu()) / max(1, rows_evaluated),
            "by_type": by_type,
            "conditions": {
                "model_dir": str(args.model_dir),
                "architecture_model": architecture_model,
                "skills_jsonl": str(args.skills_jsonl),
                "semantic_ids": str(args.semantic_ids),
                "semantic_ids_verified": semantic_ids_verified,
                "source_text_policy_verified": source_text_policy_verified,
                "max_source_length": max_source_length,
                "max_source_length_source": max_source_length_source,
                "source_length_audit": source_length_audit,
                "per_rank_batch_size": args.batch_size,
                "world_size": context.world_size,
                "bf16": bool(args.bf16),
            },
        }
        write_json(args.output, metrics)
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
