#!/usr/bin/env python3
"""Audit PAG T5-SPLADE activations without modifying a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from distributed_utils import all_gather_object, barrier, destroy_distributed, initialize_distributed
from skill_gr_pipeline import format_skillrouter_skill, load_jsonl, make_pag_sparse_encoder, skill_id


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    context = initialize_distributed(cpu=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    vocab_size = int(AutoConfig.from_pretrained(args.model).vocab_size)
    model = make_pag_sparse_encoder(str(args.model), vocab_size).to(context.device).eval()
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(payload["state_dict"], strict=True)

    skills = load_jsonl(args.skills_jsonl)[: args.sample_size]
    local = [
        (index, skill)
        for index, skill in enumerate(skills)
        if index % context.world_size == context.rank
    ]
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(local), args.batch_size):
            chunk = local[start : start + args.batch_size]
            encoded = tokenizer(
                [str(skill.get("skillrouter_text") or format_skillrouter_skill(skill)) for _, skill in chunk],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(context.device)
            attention_mask = encoded["attention_mask"].to(context.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=input_ids,
                    return_dict=True,
                )
                raw_logits = outputs.logits
                lexical = torch.log1p(torch.relu(raw_logits))
                lexical = lexical * attention_mask.unsqueeze(-1).to(dtype=lexical.dtype)
                lexical = lexical.max(dim=1).values
            lexical_for_topk = lexical.clone()
            if special_ids:
                lexical_for_topk[:, sorted(special_ids)] = float("-inf")
            top_ids = torch.topk(lexical_for_topk, k=args.top_k, dim=-1).indices.cpu().tolist()
            for row_index, ((original_index, skill), token_ids) in enumerate(zip(chunk, top_ids)):
                valid_length = int(attention_mask[row_index].sum().item())
                row_raw = raw_logits[row_index, :valid_length]
                row_lexical = lexical[row_index]
                records.append(
                    {
                        "index": original_index,
                        "skill_id": skill_id(skill),
                        "raw_positive_coordinates": int((row_raw > 0).sum().item()),
                        "raw_max": float(row_raw.max().float().cpu()),
                        "raw_min": float(row_raw.min().float().cpu()),
                        "lexical_positive_coordinates": int((row_lexical > 0).sum().item()),
                        "lexical_max": float(row_lexical.max().float().cpu()),
                        "lexical_sum": float(row_lexical.sum().float().cpu()),
                        "top_ids": [int(value) for value in token_ids],
                    }
                )

    gathered = all_gather_object(records, context)
    rows = sorted((row for part in gathered for row in part), key=lambda row: row["index"])
    if context.is_main:
        audit = {
            "label": args.label,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "sample_rows": len(rows),
            "world_size": context.world_size,
            "top_k": args.top_k,
            "unique_unordered_top_sets": len({frozenset(row["top_ids"]) for row in rows}),
            "unique_top1_ids": len({row["top_ids"][0] for row in rows}),
            "raw_positive_coordinates": summarize(
                [float(row["raw_positive_coordinates"]) for row in rows]
            ),
            "raw_max": summarize([row["raw_max"] for row in rows]),
            "raw_min": summarize([row["raw_min"] for row in rows]),
            "lexical_positive_coordinates": summarize(
                [float(row["lexical_positive_coordinates"]) for row in rows]
            ),
            "lexical_max": summarize([row["lexical_max"] for row in rows]),
            "lexical_sum": summarize([row["lexical_sum"] for row in rows]),
        }
        serialized = json.dumps(audit, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized, flush=True)
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
