#!/usr/bin/env python3
"""Real-T5 invariants for the fixed-LETTER RIPOR score path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from distributed_utils import barrier, destroy_distributed, initialize_distributed
from skill_gr_pipeline import (
    decoder_source_text,
    file_sha256,
    infer_structured_sid_config,
    load_jsonl,
    load_semantic_ids_with_mode,
    load_structured_decoder_checkpoint,
    make_structured_t5_sid_model,
    module_state_sha256,
    prepared_selected_skill_ids,
    ripor_prefix_alphas,
    structured_sid_bos_id,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-decoder", type=Path, required=True)
    parser.add_argument("--architecture-model", required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-length", type=int, default=2048)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    context = initialize_distributed()
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    sid_config = infer_structured_sid_config(semantic_ids, semantic_id_mode=semantic_id_mode)
    model = make_structured_t5_sid_model(
        args.architecture_model,
        sid_config["codebook_size"],
        sid_config["depth"],
        sid_config["dedup_size"],
        semantic_id_mode,
    ).to(context.device)
    checkpoint = load_structured_decoder_checkpoint(
        model,
        args.base_decoder,
        context.device,
        semantic_id_mode=semantic_id_mode,
    )
    if not checkpoint.get("ripor_head_initialized_from_code_heads"):
        raise RuntimeError("legacy baseline did not initialize the independent RIPOR head from CE weights")
    for level, (ripor_head, ce_head) in enumerate(zip(model.ripor_heads, model.code_heads)):
        if ripor_head.bias is not None:
            raise RuntimeError(f"RIPOR head {level} unexpectedly has a bias")
        torch.testing.assert_close(ripor_head.weight, ce_head.weight, rtol=0.0, atol=0.0)

    rows = load_jsonl(args.rows)
    row = rows[0]
    positives = prepared_selected_skill_ids(row)
    positive_sid = next((sid for sid in positives if sid in semantic_ids), None)
    if positive_sid is None:
        raise RuntimeError("preflight row has no known positive skill")
    negative_sid = next(sid for sid in sorted(semantic_ids) if sid not in set(positives))
    candidate_ids = [positive_sid, negative_sid]
    candidate_codes = torch.tensor(
        [[semantic_ids[sid][: sid_config["depth"]] for sid in candidate_ids]],
        dtype=torch.long,
        device=context.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_decoder)
    encoded = tokenizer(
        [decoder_source_text(row)],
        padding=True,
        truncation=True,
        max_length=args.max_source_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(context.device)
    attention_mask = encoded["attention_mask"].to(context.device)
    model.eval()
    with torch.no_grad():
        encoder_hidden = model.encode_source(input_ids, attention_mask)
        teacher_forced = model.ripor_prefix_scores(
            encoder_hidden,
            attention_mask,
            candidate_codes,
        )
        bos_id = structured_sid_bos_id(
            sid_config["codebook_size"],
            sid_config["depth"],
            sid_config["dedup_size"],
            semantic_id_mode,
        )
        constrained_scores = torch.zeros_like(teacher_forced)
        for candidate_index in range(candidate_codes.shape[1]):
            decoder_ids = [bos_id]
            cumulative = torch.zeros((), device=context.device, dtype=teacher_forced.dtype)
            for level in range(sid_config["depth"]):
                decoder_tensor = torch.tensor([decoder_ids], dtype=torch.long, device=context.device)
                logits = model.logits_for_next_type_from_encoder(
                    encoder_hidden,
                    attention_mask,
                    decoder_tensor,
                    torch.ones_like(decoder_tensor),
                    level,
                    ripor_score=True,
                )
                code = int(candidate_codes[0, candidate_index, level])
                cumulative = cumulative + logits[0, code]
                constrained_scores[0, candidate_index, level] = cumulative
                decoder_ids.append(level * sid_config["codebook_size"] + code)
    torch.testing.assert_close(teacher_forced, constrained_scores, rtol=1e-5, atol=1e-5)
    alpha = ripor_prefix_alphas(sid_config["depth"])
    if alpha[-1] != 1.0 or any(left > right for left, right in zip(alpha, alpha[1:])):
        raise RuntimeError(f"invalid RIPOR alpha schedule: {alpha}")
    if context.is_main:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "passed",
            "base_checkpoint": str(args.base_decoder / "structured_decoder.pt"),
            "base_checkpoint_sha256": file_sha256(args.base_decoder / "structured_decoder.pt"),
            "semantic_ids_sha256": file_sha256(args.semantic_ids),
            "world_size": context.world_size,
            "candidate_skill_ids": candidate_ids,
            "ripor_head_bias": False,
            "ripor_head_initialization": "bitwise_copy_of_fixed_letter_ce_weights",
            "ripor_head_state_sha256": module_state_sha256(model.ripor_heads),
            "ce_head_state_sha256": module_state_sha256(model.code_heads),
            "teacher_forced_vs_constrained_max_error": float(
                (teacher_forced - constrained_scores).abs().max().cpu()
            ),
            "score": "sum_raw_conditional_logits_codes_only",
            "alpha": alpha,
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
