#!/usr/bin/env python3
"""DGR continuation for the fixed SkillGR data, identifiers, and base-Qwen teacher.

The implementation follows Li et al. (Findings ACL 2024): mine the current
student's Top-N results, rerank them with a stronger teacher, and continue
training with generation CE plus the distilled RankNet loss from Eq. (3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skill_gr_pipeline import (  # noqa: E402
    compute_ranked_subset_metrics,
    decoder_source_text,
    expand_single_sid_query_rows,
    global_sid_token_id,
    infer_structured_sid_config,
    load_jsonl,
    load_semantic_ids_with_mode,
    load_structured_decoder_checkpoint,
    make_structured_t5_sid_model,
    save_structured_decoder_checkpoint,
    semantic_ids_sha256,
    skill_id,
    structured_sid_batch,
    structured_sid_bos_id,
    structured_sid_control_type,
    validate_checkpoint_semantic_ids,
    write_json,
    write_jsonl,
)


DGR_PAPER_URL = "https://aclanthology.org/2024.findings-acl.662/"
DGR_CODE_URL = "https://github.com/liyongqi67/MINDER"
RIPOR_PAPER_URL = "https://arxiv.org/abs/2311.09134"
RIPOR_CODE_URL = "https://github.com/HansiZeng/RIPOR"
RIPOR_AUDITED_CODE_COMMIT = "03fd0cbfafdeeb9cdd866e0c44727fec39a96a52"


def structured_parameter_group(name: str) -> str:
    """Partition structured GR parameters for drift-controlled continuation."""

    if name.startswith("backbone.encoder.block.") or name.startswith(
        "backbone.encoder.final_layer_norm."
    ):
        return "encoder_blocks"
    if name.startswith("backbone."):
        return "decoder_and_shared"
    return "sid_heads"


def parse_trainable_parameter_groups(value: str) -> set[str]:
    allowed = {"encoder_blocks", "decoder_and_shared", "sid_heads"}
    requested = {part.strip() for part in value.split(",") if part.strip()}
    if requested == {"all"}:
        return allowed
    if "all" in requested or not requested or not requested <= allowed:
        raise ValueError(
            "trainable parameter groups must be all or a comma-separated subset of "
            "encoder_blocks,decoder_and_shared,sid_heads"
        )
    return requested


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dgr_row_fingerprint(row: dict[str, Any]) -> str:
    """Bind a mined candidate row to both student and teacher query inputs."""

    from skillrouter_embedding_train import query_text

    payload = {
        "index": row.get("index"),
        "source_index": row.get("source_index"),
        "decoder_source_text": decoder_source_text(row),
        "teacher_query_text": query_text(row, max_chars=1500),
        "selected_skill_ids": sorted({str(value) for value in row.get("selected_skill_ids", [])}),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dgr_rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(dgr_row_fingerprint(row).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_semantic_trie_children(
    semantic_ids: dict[str, list[int]],
) -> tuple[dict[tuple[int, ...], tuple[int, ...]], dict[tuple[int, ...], str], list[int]]:
    if not semantic_ids:
        raise ValueError("semantic_ids is empty")
    lengths = {len(values) for values in semantic_ids.values()}
    if len(lengths) != 1:
        raise ValueError(f"semantic IDs have inconsistent lengths: {sorted(lengths)}")
    depth = next(iter(lengths))
    children: dict[tuple[int, ...], set[int]] = {}
    sequence_to_skill: dict[tuple[int, ...], str] = {}
    prefix_counts: list[int] = []
    for sid, raw_values in semantic_ids.items():
        values = tuple(int(value) for value in raw_values)
        if values in sequence_to_skill:
            raise ValueError(f"duplicate semantic ID: {values}")
        sequence_to_skill[values] = sid
        for level in range(depth):
            children.setdefault(values[:level], set()).add(values[level])
    for prefix_length in range(1, depth + 1):
        prefix_counts.append(len({values[:prefix_length] for values in sequence_to_skill}))
    return (
        {prefix: tuple(sorted(values)) for prefix, values in children.items()},
        sequence_to_skill,
        prefix_counts,
    )


def _decoder_logits_in_chunks(
    model,
    encoder_hidden_states,
    encoder_attention_mask,
    query_indices,
    decoder_input_ids,
    target_type: int,
    chunk_size: int,
):
    outputs = []
    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    for start in range(0, decoder_input_ids.shape[0], chunk_size):
        end = min(start + chunk_size, decoder_input_ids.shape[0])
        selected_queries = query_indices[start:end]
        outputs.append(
            model.logits_for_next_type_from_encoder(
                encoder_hidden_states.index_select(0, selected_queries),
                encoder_attention_mask.index_select(0, selected_queries),
                decoder_input_ids[start:end],
                decoder_attention_mask[start:end],
                target_type,
            ).float()
        )
    return torch.cat(outputs, dim=0)


def build_trie_decode_tables(
    trie_children: dict[tuple[int, ...], tuple[int, ...]],
    *,
    codebook_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Materialize exact trie transitions once so beam decoding stays on-device."""

    if () not in trie_children:
        raise ValueError("semantic trie has no root")
    prefixes = set(trie_children)
    for prefix, children in trie_children.items():
        for value in children:
            value = int(value)
            if value < 0 or value >= codebook_size:
                raise ValueError(f"trie child is outside the codebook: {value}")
            prefixes.add((*prefix, value))
    ordered = sorted(prefixes, key=lambda prefix: (len(prefix), prefix))
    prefix_to_index = {prefix: index for index, prefix in enumerate(ordered)}
    transitions = torch.full(
        (len(ordered), codebook_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for prefix, children in trie_children.items():
        parent = prefix_to_index[prefix]
        child_values = torch.tensor(tuple(children), dtype=torch.long, device=device)
        child_nodes = torch.tensor(
            [prefix_to_index[(*prefix, int(value))] for value in children],
            dtype=torch.int32,
            device=device,
        )
        transitions[parent, child_values] = child_nodes
    return transitions, prefix_to_index[()]


@torch.no_grad()
def topk_single_sid_scores_from_encoder(
    model,
    encoder_hidden_states,
    encoder_attention_mask,
    trie_children: dict[tuple[int, ...], tuple[int, ...]],
    prefix_counts: list[int],
    *,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str,
    top_k: int,
    decoder_batch_size: int,
    include_stop: bool,
    score_mode: str = "softmax",
    trie_decode_tables: tuple[torch.Tensor, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Top-K trie beam search for one fixed-length SID per query."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if decoder_batch_size < 1:
        raise ValueError("decoder_batch_size must be positive")
    if score_mode not in {"softmax", "conditional_logit"}:
        raise ValueError(f"unsupported score_mode: {score_mode}")
    batch_size = encoder_hidden_states.shape[0]
    device = encoder_hidden_states.device
    bos_id = structured_sid_bos_id(codebook_size, depth, dedup_size, semantic_id_mode)
    prefixes = torch.empty((batch_size, 1, 0), dtype=torch.long, device=device)
    beam_scores = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    if trie_decode_tables is None:
        trie_transitions, root_node = build_trie_decode_tables(
            trie_children,
            codebook_size=codebook_size,
            device=device,
        )
    else:
        trie_transitions, root_node = trie_decode_tables
        if trie_transitions.device != device:
            raise ValueError("trie decode tables and encoder states must use the same device")
        if trie_transitions.shape[1] != codebook_size:
            raise ValueError("trie decode table codebook size mismatch")
    node_ids = torch.full(
        (batch_size, 1), int(root_node), dtype=torch.long, device=device
    )

    for level in range(depth):
        beam_width = prefixes.shape[1]
        flat_prefixes = prefixes.reshape(batch_size * beam_width, level)
        query_indices = torch.arange(batch_size, device=device).repeat_interleave(beam_width)
        decoder_columns = [
            torch.full((flat_prefixes.shape[0], 1), bos_id, dtype=torch.long, device=device)
        ]
        for previous_level in range(level):
            values = flat_prefixes[:, previous_level]
            decoder_columns.append(
                (
                    values
                    + global_sid_token_id(
                        previous_level,
                        0,
                        codebook_size,
                        depth,
                        dedup_size,
                        semantic_id_mode,
                    )
                ).unsqueeze(1)
            )
        decoder_input_ids = torch.cat(decoder_columns, dim=1)
        logits = _decoder_logits_in_chunks(
            model,
            encoder_hidden_states,
            encoder_attention_mask,
            query_indices,
            decoder_input_ids,
            level,
            decoder_batch_size,
        )
        step_scores = (
            F.log_softmax(logits, dim=-1)
            if score_mode == "softmax"
            else logits
        ).reshape(batch_size, beam_width, codebook_size)
        transitions = trie_transitions[node_ids]
        allowed_mask = transitions >= 0
        candidate_scores = (
            beam_scores.unsqueeze(-1) + step_scores.masked_fill(~allowed_mask, float("-inf"))
        )
        keep = min(top_k, prefix_counts[level])
        new_scores, flat_indices = torch.topk(
            candidate_scores.reshape(batch_size, -1), k=keep, dim=1
        )
        if not torch.isfinite(new_scores).all():
            raise ValueError(f"trie search produced invalid paths at level={level}")
        parent_indices = torch.div(flat_indices, codebook_size, rounding_mode="floor")
        next_values = flat_indices.remainder(codebook_size)
        gathered_prefixes = prefixes.gather(
            1, parent_indices.unsqueeze(-1).expand(-1, -1, level)
        )
        prefixes = torch.cat([gathered_prefixes, next_values.unsqueeze(-1)], dim=-1)
        selected_nodes = node_ids.gather(1, parent_indices)
        node_ids = trie_transitions[selected_nodes, next_values].long()
        if torch.any(node_ids < 0):
            raise ValueError(f"trie transition produced an invalid node at level={level}")
        beam_scores = new_scores

    if include_stop:
        beam_width = prefixes.shape[1]
        flat_prefixes = prefixes.reshape(batch_size * beam_width, depth)
        query_indices = torch.arange(batch_size, device=device).repeat_interleave(beam_width)
        decoder_columns = [
            torch.full((flat_prefixes.shape[0], 1), bos_id, dtype=torch.long, device=device)
        ]
        for level in range(depth):
            decoder_columns.append(
                flat_prefixes[:, level : level + 1]
                + global_sid_token_id(
                    level,
                    0,
                    codebook_size,
                    depth,
                    dedup_size,
                    semantic_id_mode,
                )
            )
        decoder_input_ids = torch.cat(decoder_columns, dim=1)
        control_type = structured_sid_control_type(depth, semantic_id_mode)
        stop_logits = _decoder_logits_in_chunks(
            model,
            encoder_hidden_states,
            encoder_attention_mask,
            query_indices,
            decoder_input_ids,
            control_type,
            decoder_batch_size,
        )
        stop_scores = F.log_softmax(stop_logits, dim=-1)[:, 1].reshape(batch_size, beam_width)
        beam_scores = beam_scores + stop_scores
        beam_scores, order = torch.sort(beam_scores, dim=1, descending=True)
        prefixes = prefixes.gather(1, order.unsqueeze(-1).expand_as(prefixes))

    return prefixes, beam_scores


def select_dgr_candidate_list(
    student_candidate_indices: list[int],
    teacher_ranks: list[int],
    gold_indices: set[int],
    positive_index: int,
    list_size: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """DGR Top&Random: one positive, one teacher-top negative, and random negatives."""

    if len(student_candidate_indices) != len(teacher_ranks):
        raise ValueError("candidate indices and teacher ranks must have equal length")
    if list_size < 3:
        raise ValueError("DGR list_size must be at least 3")
    if positive_index not in gold_indices:
        raise ValueError("the primary positive must be in the source row's gold set")
    unique_pool: list[tuple[int, int]] = []
    seen: set[int] = set()
    for candidate_index, teacher_rank in zip(student_candidate_indices, teacher_ranks):
        candidate_index = int(candidate_index)
        if candidate_index in gold_indices or candidate_index in seen:
            continue
        seen.add(candidate_index)
        unique_pool.append((candidate_index, int(teacher_rank)))
    required_negatives = list_size - 1
    if len(unique_pool) < required_negatives:
        raise ValueError(
            f"DGR candidate pool has {len(unique_pool)} negatives, needs {required_negatives}"
        )
    teacher_top = min(unique_pool, key=lambda item: item[1])
    random_pool = [item for item in unique_pool if item[0] != teacher_top[0]]
    sampled = rng.sample(random_pool, required_negatives - 1)
    selected = [teacher_top, *sampled]
    return [positive_index, *(item[0] for item in selected)], [-1, *(item[1] for item in selected)]


def candidate_retrieval_audit(
    rows: list[dict[str, Any]],
    candidate_indices: torch.Tensor,
    teacher_ranks: torch.Tensor,
    skill_to_index: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    """Audit student Top-N coverage and the teacher's Top-1 inside that fixed pool."""

    if candidate_indices.shape != teacher_ranks.shape or candidate_indices.shape[0] != len(rows):
        raise ValueError("candidate audit tensors do not align with rows")
    totals = {
        subset: {"rows": 0, "student_top1_hits": 0, "teacher_top1_hits": 0, "gold_recall_sum": 0.0}
        for subset in ("overall", "single", "multi")
    }
    for row_position, row in enumerate(rows):
        gold_indices = {
            skill_to_index[str(sid)] for sid in row.get("selected_skill_ids", [])
        }
        if not gold_indices:
            raise ValueError(f"candidate audit row has no gold skills: {row.get('index')}")
        student_row = [int(value) for value in candidate_indices[row_position].tolist()]
        rank_row = teacher_ranks[row_position]
        teacher_top_position = int(torch.argmin(rank_row))
        subsets = ["overall", "single" if len(gold_indices) == 1 else "multi"]
        for subset in subsets:
            totals[subset]["rows"] += 1
            totals[subset]["student_top1_hits"] += int(student_row[0] in gold_indices)
            totals[subset]["teacher_top1_hits"] += int(
                student_row[teacher_top_position] in gold_indices
            )
            totals[subset]["gold_recall_sum"] += len(set(student_row) & gold_indices) / len(
                gold_indices
            )
    result: dict[str, dict[str, float | int]] = {}
    for subset, values in totals.items():
        row_count = int(values["rows"])
        result[subset] = {
            **values,
            "student_top1_recall": float(values["student_top1_hits"] / row_count)
            if row_count
            else 0.0,
            "teacher_top1_within_student_pool_recall": float(
                values["teacher_top1_hits"] / row_count
            )
            if row_count
            else 0.0,
            "student_top_n_macro_recall": float(values["gold_recall_sum"] / row_count)
            if row_count
            else 0.0,
        }
    return result


def distilled_ranknet_loss(
    student_scores: torch.Tensor,
    teacher_ranks: torch.Tensor,
    *,
    base_margin: float,
    gap_margin: float,
) -> torch.Tensor:
    """Mean-normalized DGR Eq. (3), keeping column zero as the labeled positive."""

    if student_scores.ndim != 2 or teacher_ranks.shape != student_scores.shape:
        raise ValueError("student_scores and teacher_ranks must be equal rank-2 tensors")
    if student_scores.shape[1] < 2:
        raise ValueError("DGR needs at least two candidates")
    if base_margin < 0 or gap_margin < 0:
        raise ValueError("DGR margins must be non-negative")
    if not (teacher_ranks[:, 0] == -1).all():
        raise ValueError("teacher rank -1 must mark the labeled positive in column zero")
    negative_order = torch.argsort(teacher_ranks[:, 1:], dim=1, descending=False)
    ordered_negative_scores = student_scores[:, 1:].gather(1, negative_order)
    ordered_scores = torch.cat([student_scores[:, :1], ordered_negative_scores], dim=1)
    row_indices, column_indices = torch.triu_indices(
        ordered_scores.shape[1], ordered_scores.shape[1], offset=1, device=ordered_scores.device
    )
    margins = float(base_margin) + float(gap_margin) * (column_indices - row_indices - 1)
    pair_losses = F.relu(
        ordered_scores[:, column_indices] - ordered_scores[:, row_indices] + margins
    )
    return pair_losses.mean()


def select_prefix_candidate_list(
    student_candidate_indices: list[int],
    gold_indices: set[int],
    positive_index: int,
    list_size: int,
) -> list[int]:
    """Keep one labeled positive followed by the student's hardest non-gold paths."""

    if positive_index not in gold_indices:
        raise ValueError("positive_index must be one of the source row's labeled positives")
    if list_size < 2:
        raise ValueError("prefix ranking needs at least one positive and one negative")
    negatives: list[int] = []
    seen = set(gold_indices)
    for candidate_index in student_candidate_indices:
        candidate_index = int(candidate_index)
        if candidate_index in seen:
            continue
        negatives.append(candidate_index)
        seen.add(candidate_index)
        if len(negatives) == list_size - 1:
            break
    if len(negatives) != list_size - 1:
        raise ValueError(
            f"candidate pool has only {len(negatives)} non-gold paths; need {list_size - 1}"
        )
    return [int(positive_index), *negatives]


def generation_loss_from_encoder(model, encoder_hidden_states, batch: dict[str, torch.Tensor]):
    hidden = model._decoder_hidden_from_encoder(
        encoder_hidden_states,
        batch["attention_mask"],
        batch["decoder_input_ids"],
        batch["decoder_attention_mask"],
    )
    target_types = batch["target_types"]
    labels = batch["labels"]
    loss_sum = hidden.new_tensor(0.0, dtype=torch.float32)
    token_count = hidden.new_tensor(0.0, dtype=torch.float32)
    for level, head in enumerate(model.code_heads):
        mask = target_types == level
        if mask.any():
            loss_sum = loss_sum + F.cross_entropy(
                head(hidden[mask]).float(), labels[mask], reduction="sum"
            )
            token_count = token_count + mask.sum()
    if model.dedup_head is not None:
        dedup_mask = target_types == model.depth
        if dedup_mask.any():
            loss_sum = loss_sum + F.cross_entropy(
                model.dedup_head(hidden[dedup_mask]).float(), labels[dedup_mask], reduction="sum"
            )
            token_count = token_count + dedup_mask.sum()
    control_mask = target_types == model.control_type
    if control_mask.any():
        loss_sum = loss_sum + F.cross_entropy(
            model.control_head(hidden[control_mask]).float(), labels[control_mask], reduction="sum"
        )
        token_count = token_count + control_mask.sum()
    return loss_sum / token_count.clamp_min(1.0)


def candidate_sequence_log_scores(
    model,
    encoder_hidden_states,
    encoder_attention_mask,
    candidate_codes: torch.Tensor,
    *,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str,
) -> torch.Tensor:
    """Differentiable S(q,s): sum of full-head token log-probabilities."""

    if candidate_codes.ndim != 3 or candidate_codes.shape[2] != depth:
        raise ValueError(f"candidate_codes must have shape [batch, list, {depth}]")
    batch_size, list_size, _ = candidate_codes.shape
    device = candidate_codes.device
    flat_codes = candidate_codes.reshape(batch_size * list_size, depth)
    query_indices = torch.arange(batch_size, device=device).repeat_interleave(list_size)
    expanded_encoder = encoder_hidden_states.index_select(0, query_indices)
    expanded_attention = encoder_attention_mask.index_select(0, query_indices)
    bos_id = structured_sid_bos_id(codebook_size, depth, dedup_size, semantic_id_mode)
    sequence_scores = torch.zeros(flat_codes.shape[0], dtype=torch.float32, device=device)
    for level in range(depth):
        decoder_columns = [
            torch.full((flat_codes.shape[0], 1), bos_id, dtype=torch.long, device=device)
        ]
        for previous_level in range(level):
            decoder_columns.append(
                flat_codes[:, previous_level : previous_level + 1]
                + global_sid_token_id(
                    previous_level,
                    0,
                    codebook_size,
                    depth,
                    dedup_size,
                    semantic_id_mode,
                )
            )
        decoder_input_ids = torch.cat(decoder_columns, dim=1)
        hidden = model._decoder_hidden_from_encoder(
            expanded_encoder,
            expanded_attention,
            decoder_input_ids,
            torch.ones_like(decoder_input_ids),
        )
        logits = model.logits_for_hidden_type(hidden, level).float()
        target_values = flat_codes[:, level]
        sequence_scores = sequence_scores + F.log_softmax(logits, dim=-1).gather(
            1, target_values.unsqueeze(1)
        ).squeeze(1)
    return sequence_scores.reshape(batch_size, list_size)


def candidate_prefix_logit_scores(
    model,
    encoder_hidden_states,
    encoder_attention_mask,
    candidate_codes: torch.Tensor,
    *,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str,
) -> torch.Tensor:
    """Differentiable cumulative conditional-logit score at every SID prefix."""

    if candidate_codes.ndim != 3 or candidate_codes.shape[2] != depth:
        raise ValueError(f"candidate_codes must have shape [batch, list, {depth}]")
    batch_size, list_size, _ = candidate_codes.shape
    device = candidate_codes.device
    flat_codes = candidate_codes.reshape(batch_size * list_size, depth)
    query_indices = torch.arange(batch_size, device=device).repeat_interleave(list_size)
    expanded_encoder = encoder_hidden_states.index_select(0, query_indices)
    expanded_attention = encoder_attention_mask.index_select(0, query_indices)
    bos_id = structured_sid_bos_id(codebook_size, depth, dedup_size, semantic_id_mode)
    token_scores = []
    for level in range(depth):
        decoder_columns = [
            torch.full((flat_codes.shape[0], 1), bos_id, dtype=torch.long, device=device)
        ]
        for previous_level in range(level):
            decoder_columns.append(
                flat_codes[:, previous_level : previous_level + 1]
                + global_sid_token_id(
                    previous_level,
                    0,
                    codebook_size,
                    depth,
                    dedup_size,
                    semantic_id_mode,
                )
            )
        decoder_input_ids = torch.cat(decoder_columns, dim=1)
        hidden = model._decoder_hidden_from_encoder(
            expanded_encoder,
            expanded_attention,
            decoder_input_ids,
            torch.ones_like(decoder_input_ids),
        )
        logits = model.logits_for_hidden_type(hidden, level).float()
        token_scores.append(
            logits.gather(1, flat_codes[:, level : level + 1]).squeeze(1)
        )
    cumulative = torch.cumsum(torch.stack(token_scores, dim=1), dim=1)
    return cumulative.reshape(batch_size, list_size, depth)


def candidate_prefix_logprob_scores(
    model,
    encoder_hidden_states,
    encoder_attention_mask,
    candidate_codes: torch.Tensor,
    *,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str,
    include_final_stop: bool = True,
) -> torch.Tensor:
    """Cumulative full-head log-probability, with the inference STOP score at full depth."""

    if candidate_codes.ndim != 3 or candidate_codes.shape[2] != depth:
        raise ValueError(f"candidate_codes must have shape [batch, list, {depth}]")
    batch_size, list_size, _ = candidate_codes.shape
    device = candidate_codes.device
    flat_codes = candidate_codes.reshape(batch_size * list_size, depth)
    query_indices = torch.arange(batch_size, device=device).repeat_interleave(list_size)
    expanded_encoder = encoder_hidden_states.index_select(0, query_indices)
    expanded_attention = encoder_attention_mask.index_select(0, query_indices)
    bos_id = structured_sid_bos_id(codebook_size, depth, dedup_size, semantic_id_mode)
    token_scores = []
    for level in range(depth):
        decoder_columns = [
            torch.full((flat_codes.shape[0], 1), bos_id, dtype=torch.long, device=device)
        ]
        for previous_level in range(level):
            decoder_columns.append(
                flat_codes[:, previous_level : previous_level + 1]
                + global_sid_token_id(
                    previous_level,
                    0,
                    codebook_size,
                    depth,
                    dedup_size,
                    semantic_id_mode,
                )
            )
        decoder_input_ids = torch.cat(decoder_columns, dim=1)
        hidden = model._decoder_hidden_from_encoder(
            expanded_encoder,
            expanded_attention,
            decoder_input_ids,
            torch.ones_like(decoder_input_ids),
        )
        logits = model.logits_for_hidden_type(hidden, level).float()
        token_scores.append(
            F.log_softmax(logits, dim=-1)
            .gather(1, flat_codes[:, level : level + 1])
            .squeeze(1)
        )
    cumulative = torch.cumsum(torch.stack(token_scores, dim=1), dim=1)
    if include_final_stop:
        full_decoder_columns = [
            torch.full((flat_codes.shape[0], 1), bos_id, dtype=torch.long, device=device)
        ]
        for level in range(depth):
            full_decoder_columns.append(
                flat_codes[:, level : level + 1]
                + global_sid_token_id(
                    level,
                    0,
                    codebook_size,
                    depth,
                    dedup_size,
                    semantic_id_mode,
                )
            )
        full_decoder_input_ids = torch.cat(full_decoder_columns, dim=1)
        hidden = model._decoder_hidden_from_encoder(
            expanded_encoder,
            expanded_attention,
            full_decoder_input_ids,
            torch.ones_like(full_decoder_input_ids),
        )
        control_type = structured_sid_control_type(depth, semantic_id_mode)
        stop_logits = model.logits_for_hidden_type(hidden, control_type).float()
        stop_scores = F.log_softmax(stop_logits, dim=-1)[:, 1]
        stop_addition = torch.zeros_like(cumulative)
        stop_addition[:, -1] = stop_scores
        cumulative = cumulative + stop_addition
    return cumulative.reshape(batch_size, list_size, depth)


def loss_gradient_alignment(
    generation_component: torch.Tensor,
    rank_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> dict[str, float]:
    """Measure unmodified component gradients on the current training batch."""

    generation_gradients = torch.autograd.grad(
        generation_component,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    rank_gradients = torch.autograd.grad(
        rank_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    generation_squared_norm = 0.0
    rank_squared_norm = 0.0
    dot_product = 0.0
    for generation_gradient, rank_gradient in zip(generation_gradients, rank_gradients):
        if generation_gradient is not None:
            generation_float = generation_gradient.detach().float()
            generation_squared_norm += float(generation_float.square().sum().cpu())
        else:
            generation_float = None
        if rank_gradient is not None:
            rank_float = rank_gradient.detach().float()
            rank_squared_norm += float(rank_float.square().sum().cpu())
        else:
            rank_float = None
        if generation_float is not None and rank_float is not None:
            dot_product += float((generation_float * rank_float).sum().cpu())
    generation_norm = math.sqrt(generation_squared_norm)
    rank_norm = math.sqrt(rank_squared_norm)
    cosine = dot_product / (generation_norm * rank_norm) if generation_norm and rank_norm else 0.0
    return {
        "generation_component_grad_norm": generation_norm,
        "unweighted_rank_grad_norm": rank_norm,
        "gradient_dot_product": dot_product,
        "gradient_cosine": cosine,
    }


def prefix_margin_mse_loss(
    prefix_scores: torch.Tensor,
    *,
    active_prefixes: int,
    target_margin: float | torch.Tensor,
    margin_beta: float = 0.0,
    candidate_codes: torch.Tensor | None = None,
    active_prefix_indices: list[int] | None = None,
    example_mask: torch.Tensor | None = None,
    positive_target_margins_only: bool = False,
) -> torch.Tensor:
    """RIPOR-style gold-vs-negative MarginMSE over all active prefixes."""

    if prefix_scores.ndim != 3 or prefix_scores.shape[1] < 2:
        raise ValueError("prefix_scores must have shape [batch, candidates>=2, depth]")
    if active_prefixes < 1 or active_prefixes > prefix_scores.shape[2]:
        raise ValueError(
            f"active_prefixes must be in [1, {prefix_scores.shape[2]}], got {active_prefixes}"
        )
    if margin_beta < 0 or margin_beta >= 1:
        raise ValueError("margin_beta must be in [0, 1) for prefixes starting at length one")
    if active_prefix_indices is None:
        active_prefix_indices = list(range(active_prefixes))
    if (
        len(active_prefix_indices) != active_prefixes
        or len(set(active_prefix_indices)) != active_prefixes
        or min(active_prefix_indices) < 0
        or max(active_prefix_indices) >= prefix_scores.shape[2]
    ):
        raise ValueError("active_prefix_indices must be unique valid zero-based prefix positions")
    selected = torch.tensor(active_prefix_indices, dtype=torch.long, device=prefix_scores.device)
    positive_scores = prefix_scores[:, :1, :].index_select(2, selected)
    negative_scores = prefix_scores[:, 1:, :].index_select(2, selected)
    predicted_margins = positive_scores - negative_scores
    if isinstance(target_margin, torch.Tensor):
        if target_margin.shape == prefix_scores.shape[:2]:
            target_margin_tensor = target_margin[:, 1:]
        else:
            expected_shape = predicted_margins.shape[:2]
            if target_margin.shape != expected_shape:
                raise ValueError(f"tensor target_margin must have shape {expected_shape}")
            target_margin_tensor = target_margin
        target_margin_tensor = target_margin_tensor.to(
            device=prefix_scores.device,
            dtype=predicted_margins.dtype,
        )
    else:
        if target_margin < 0:
            raise ValueError("fixed target_margin must be non-negative")
        target_margin_tensor = predicted_margins.new_full(
            predicted_margins.shape[:2],
            float(target_margin),
        )
    full_depth = prefix_scores.shape[2]
    prefix_positions = torch.arange(
        1,
        full_depth + 1,
        dtype=predicted_margins.dtype,
        device=predicted_margins.device,
    ).index_select(0, selected)
    alpha = (1.0 - float(margin_beta) / prefix_positions) / (
        1.0 - float(margin_beta) / full_depth
    )
    targets = target_margin_tensor.unsqueeze(-1) * alpha.view(1, 1, active_prefixes)
    squared_errors = (predicted_margins - targets).square()
    active_pairs = torch.ones_like(squared_errors, dtype=torch.bool)
    if positive_target_margins_only:
        active_pairs &= target_margin_tensor.unsqueeze(-1) > 0
    if candidate_codes is not None:
        if candidate_codes.shape != prefix_scores.shape:
            raise ValueError(
                "candidate_codes must have the same [batch, candidates, depth] shape as prefix_scores"
            )
        token_differs = candidate_codes[:, :1, :] != candidate_codes[:, 1:, :]
        prefix_has_diverged = token_differs.to(torch.int32).cumsum(dim=2) > 0
        active_pairs &= prefix_has_diverged.index_select(2, selected)
    if example_mask is not None:
        if example_mask.ndim != 1 or example_mask.shape[0] != prefix_scores.shape[0]:
            raise ValueError("example_mask must have shape [batch]")
        active_pairs &= example_mask.to(device=prefix_scores.device, dtype=torch.bool).view(-1, 1, 1)
    if not active_pairs.any():
        return predicted_margins.sum() * 0.0
    return squared_errors[active_pairs].mean()


def teacher_margins_for_candidates(
    query_embeddings: torch.Tensor,
    skill_embeddings: torch.Tensor,
    source_positions: torch.Tensor,
    selected_indices: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return teacher positive-minus-negative score margins for selected candidate lists."""

    if query_embeddings.ndim != 2 or skill_embeddings.ndim != 2:
        raise ValueError("teacher query and skill embeddings must be rank-2 tensors")
    if query_embeddings.shape[1] != skill_embeddings.shape[1]:
        raise ValueError("teacher query and skill embedding dimensions differ")
    if source_positions.ndim != 1 or selected_indices.ndim != 2:
        raise ValueError("source_positions must be rank 1 and selected_indices rank 2")
    if source_positions.shape[0] != selected_indices.shape[0] or selected_indices.shape[1] < 2:
        raise ValueError("each source position needs a positive and at least one negative")
    if scale <= 0.0:
        raise ValueError("teacher margin scale must be positive")
    selected_queries = query_embeddings.index_select(0, source_positions.to(query_embeddings.device))
    selected_skills = skill_embeddings[selected_indices.to(skill_embeddings.device)]
    scores = torch.einsum("bd,bnd->bn", selected_queries.float(), selected_skills.float())
    return float(scale) * (scores[:, :1] - scores[:, 1:])


def teacher_best_gold_indices_for_rows(
    rows: list[dict[str, Any]],
    query_embeddings: torch.Tensor,
    skill_embeddings: torch.Tensor,
    skill_to_index: dict[str, int],
    *,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Select one teacher-best labeled skill per source query for top-1 distillation."""

    if query_embeddings.ndim != 2 or skill_embeddings.ndim != 2:
        raise ValueError("teacher query and skill embeddings must be rank-2 tensors")
    if len(rows) != query_embeddings.shape[0]:
        raise ValueError("teacher query embeddings do not align with source rows")
    if query_embeddings.shape[1] != skill_embeddings.shape[1]:
        raise ValueError("teacher query and skill embedding dimensions differ")
    if batch_size < 1:
        raise ValueError("teacher-best-gold batch size must be positive")

    flat_source_positions = []
    flat_gold_indices = []
    for source_position, row in enumerate(rows):
        gold_ids = [str(value) for value in row.get("selected_skill_ids", [])]
        if not gold_ids:
            raise ValueError(f"row position={source_position} has no selected_skill_ids")
        try:
            gold_indices = [skill_to_index[sid] for sid in gold_ids]
        except KeyError as error:
            raise ValueError(f"unknown gold skill ID at row position={source_position}: {error}") from error
        flat_source_positions.extend([source_position] * len(gold_indices))
        flat_gold_indices.extend(gold_indices)

    source_positions = torch.tensor(flat_source_positions, dtype=torch.long)
    gold_indices = torch.tensor(flat_gold_indices, dtype=torch.long)
    score_chunks = []
    for start in range(0, len(flat_gold_indices), batch_size):
        end = min(start + batch_size, len(flat_gold_indices))
        queries = query_embeddings.index_select(0, source_positions[start:end])
        golds = skill_embeddings.index_select(0, gold_indices[start:end])
        score_chunks.append((queries.float() * golds.float()).sum(dim=1))
    scores = torch.cat(score_chunks)
    best_scores = torch.full((len(rows),), -torch.inf, dtype=scores.dtype)
    best_scores.scatter_reduce_(
        0,
        source_positions,
        scores,
        reduce="amax",
        include_self=True,
    )
    flat_positions = torch.arange(len(flat_gold_indices), dtype=torch.long)
    sentinel = len(flat_gold_indices)
    winning_positions = torch.where(
        scores == best_scores.index_select(0, source_positions),
        flat_positions,
        torch.full_like(flat_positions, sentinel),
    )
    best_flat_positions = torch.full((len(rows),), sentinel, dtype=torch.long)
    best_flat_positions.scatter_reduce_(
        0,
        source_positions,
        winning_positions,
        reduce="amin",
        include_self=True,
    )
    if (best_flat_positions == sentinel).any():
        raise ValueError("failed to select a teacher-best gold for every source row")
    return gold_indices.index_select(0, best_flat_positions)


def progressive_prefix_indices(
    *,
    depth: int,
    epoch: int,
    full_sequence_warmup: bool,
) -> list[int]:
    """Return zero-based prefix positions for a short-SID progressive stage."""

    if depth < 1 or epoch < 1:
        raise ValueError("depth and epoch must be positive")
    if not full_sequence_warmup:
        return list(range(min(depth, epoch)))
    if epoch == 1:
        return [depth - 1]
    return sorted({*range(min(depth - 1, epoch - 1)), depth - 1})


def _load_candidate_shards(
    candidates_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    manifest_path = candidates_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"candidate mining is not complete: {manifest.get('status')}")
    if int(manifest.get("rows", -1)) != len(rows):
        raise ValueError("candidate artifact row count does not match training rows")
    all_candidates = []
    all_teacher_ranks = []
    fingerprints: list[str] = []
    for shard_name in manifest.get("shards", []):
        shard = torch.load(candidates_dir / "shards" / shard_name, map_location="cpu")
        all_candidates.append(shard["candidate_indices"].to(torch.int64))
        teacher_ranks = shard.get("teacher_ranks")
        if teacher_ranks is None:
            teacher_ranks = torch.full_like(shard["candidate_indices"], -1)
        all_teacher_ranks.append(teacher_ranks.to(torch.int64))
        fingerprints.extend(str(value) for value in shard["row_fingerprints"])
    if len(fingerprints) != len(rows):
        raise ValueError("candidate shard fingerprints are incomplete")
    for row_position, (row, fingerprint) in enumerate(zip(rows, fingerprints)):
        expected = dgr_row_fingerprint(row)
        if fingerprint != expected:
            raise ValueError(f"candidate row fingerprint mismatch at position={row_position}")
    return torch.cat(all_candidates, dim=0), torch.cat(all_teacher_ranks, dim=0), manifest


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _load_teacher_query_embedding_shards(
    embeddings_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    manifest_path = embeddings_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"teacher query encoding is not complete: {manifest.get('status')}")
    if int(manifest.get("rows", -1)) != len(rows):
        raise ValueError("teacher query embedding row count does not match training rows")
    if manifest.get("rows_fingerprint_sha256") != dgr_rows_fingerprint(rows):
        raise ValueError("teacher query embedding row fingerprint does not match training rows")

    chunks = []
    expected_start = 0
    expected_shard_sha256 = manifest.get("shard_sha256", {})
    for shard_name in manifest.get("shards", []):
        shard_path = embeddings_dir / "shards" / shard_name
        if shard_name in expected_shard_sha256:
            actual_sha256 = file_sha256(shard_path)
            if actual_sha256 != expected_shard_sha256[shard_name]:
                raise ValueError(f"teacher query embedding shard checksum mismatch: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        start = int(shard.get("start", -1))
        end = int(shard.get("end", -1))
        embeddings = shard.get("embeddings")
        if start != expected_start or end <= start or not isinstance(embeddings, torch.Tensor):
            raise ValueError(f"invalid teacher query embedding shard: {shard_path}")
        if tuple(embeddings.shape) != (end - start, int(manifest["embedding_dim"])):
            raise ValueError(f"teacher query embedding shard shape mismatch: {shard_path}")
        chunks.append(embeddings.float())
        expected_start = end
    if expected_start != len(rows):
        raise ValueError("teacher query embedding shards are incomplete")
    return torch.cat(chunks, dim=0), manifest


def command_encode_teacher_queries(args: argparse.Namespace) -> None:
    from skillrouter_embedding_train import last_token_pool, load_qwen_encoder, query_text

    rows_all = load_jsonl(args.rows)
    rows = rows_all[: args.max_rows] if args.max_rows is not None else rows_all
    if not rows:
        raise ValueError("no teacher queries to encode")
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("teacher query batch and shard sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    config = {
        "rows_path": str(args.rows),
        "rows": len(rows),
        "full_source_rows": len(rows_all),
        "rows_fingerprint_sha256": dgr_rows_fingerprint(rows),
        "teacher_model": args.teacher_model,
        "teacher_query_text_policy": "skillrouter_query_text_max_chars_1500",
        "teacher_max_length": args.teacher_max_length,
        "teacher_bf16": bool(args.teacher_bf16),
        "embedding_normalization": "l2",
        "embedding_dtype": "float32",
        "shard_size": args.shard_size,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in config.items():
            if previous.get(key) != value:
                raise ValueError(
                    f"teacher query manifest mismatch for {key}: previous={previous.get(key)!r}, current={value!r}"
                )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer, model = load_qwen_encoder(args.teacher_model, device, bf16=bool(args.teacher_bf16))
    model.eval()
    shard_count = math.ceil(len(rows) / args.shard_size)
    completed_shards: list[str] = []
    started = time.time()
    embedding_dim = None
    for shard_index in range(shard_count):
        shard_name = f"part-{shard_index:06d}.pt"
        shard_path = shard_dir / shard_name
        start = shard_index * args.shard_size
        end = min(start + args.shard_size, len(rows))
        shard_rows = rows[start:end]
        if shard_path.exists():
            existing = torch.load(shard_path, map_location="cpu", weights_only=False)
            existing_embeddings = existing.get("embeddings")
            if (
                int(existing.get("start", -1)) != start
                or int(existing.get("end", -1)) != end
                or not isinstance(existing_embeddings, torch.Tensor)
                or existing_embeddings.shape[0] != len(shard_rows)
            ):
                raise ValueError(f"existing teacher query shard failed validation: {shard_path}")
            embedding_dim = int(existing_embeddings.shape[1])
            completed_shards.append(shard_name)
            continue

        chunks = []
        texts = [query_text(row, max_chars=1500) for row in shard_rows]
        for batch_start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[batch_start : batch_start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.teacher_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                chunks.append(F.normalize(pooled.float(), p=2, dim=1).cpu())
        embeddings = torch.cat(chunks, dim=0)
        embedding_dim = int(embeddings.shape[1])
        _atomic_torch_save({"start": start, "end": end, "embeddings": embeddings}, shard_path)
        completed_shards.append(shard_name)
        write_json(
            manifest_path,
            {
                **config,
                "status": "running",
                "embedding_dim": embedding_dim,
                "shards": completed_shards,
                "completed_rows": end,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(f"encoded teacher queries {end}/{len(rows)} elapsed={time.time() - started:.1f}s", flush=True)

    embeddings, _ = _load_teacher_query_embedding_shards(
        args.output_dir,
        rows,
    ) if manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete" else (
        torch.cat(
            [
                torch.load(shard_dir / name, map_location="cpu", weights_only=False)["embeddings"].float()
                for name in completed_shards
            ],
            dim=0,
        ),
        None,
    )
    norms = embeddings.norm(p=2, dim=1)
    write_json(
        manifest_path,
        {
            **config,
            "status": "complete",
            "embedding_dim": int(embeddings.shape[1]),
            "shards": completed_shards,
            "completed_rows": len(rows),
            "elapsed_seconds": time.time() - started,
            "norm_min": float(norms.min()),
            "norm_mean": float(norms.mean()),
            "norm_max": float(norms.max()),
            "shard_sha256": {
                name: file_sha256(shard_dir / name) for name in completed_shards
            },
        },
    )
    print(f"saved teacher query embeddings to {args.output_dir}", flush=True)


def command_encode_teacher_skills(args: argparse.Namespace) -> None:
    """Encode the canonical catalog with the same bi-encoder used for teacher queries."""

    import numpy as np

    from skillrouter_embedding_train import encode_texts_eval, load_qwen_encoder, skill_text

    skills = load_jsonl(args.skills_jsonl)
    if not skills:
        raise ValueError("teacher skill catalog is empty")
    skill_ids = [str(skill_id(row)) for row in skills]
    if any(not value for value in skill_ids):
        raise ValueError("every teacher skill row must have a skill_id")
    if len(set(skill_ids)) != len(skill_ids):
        raise ValueError("teacher skill catalog contains duplicate skill IDs")
    if args.batch_size < 1:
        raise ValueError("teacher skill batch size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer, model = load_qwen_encoder(
        args.teacher_model, device, bf16=bool(args.teacher_bf16)
    )
    embeddings = encode_texts_eval(
        [skill_text(row, args.desc_max, args.body_max) for row in skills],
        tokenizer,
        model,
        device,
        args.teacher_max_length,
        args.batch_size,
        "teacher skills",
    ).float()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(skills):
        raise ValueError(
            f"unexpected teacher skill embedding shape: {tuple(embeddings.shape)}"
        )
    norms = embeddings.norm(dim=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings.numpy())
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "status": "complete",
        "skills_jsonl": str(args.skills_jsonl),
        "skills_jsonl_sha256": file_sha256(args.skills_jsonl),
        "skill_ids_sha256": hashlib.sha256(
            "\n".join(skill_ids).encode("utf-8")
        ).hexdigest(),
        "rows": len(skills),
        "teacher_model": args.teacher_model,
        "teacher_skill_text_policy": (
            f"skillrouter_name_description_body_desc{args.desc_max}_body{args.body_max}"
        ),
        "teacher_max_length": args.teacher_max_length,
        "teacher_bf16": bool(args.teacher_bf16),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": "float32",
        "embedding_normalization": "l2",
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


def _teacher_margin_summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot summarize empty teacher margins")
    absolute = values.abs()
    positive = values[values > 0]
    quantile_points = torch.tensor(
        [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
        dtype=torch.float32,
    )

    def quantiles(tensor: torch.Tensor) -> dict[str, float]:
        measured = torch.quantile(tensor, quantile_points)
        return {
            name: float(value)
            for name, value in zip(
                ["q00", "q01", "q10", "q25", "q50", "q75", "q90", "q99", "q100"],
                measured,
            )
        }

    positive_median = float(positive.median()) if positive.numel() else None
    absolute_median = float(absolute.median())
    absolute_mean = float(absolute.mean())
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "positive_fraction": float((values > 0).float().mean()),
        "zero_fraction": float((values == 0).float().mean()),
        "signed_quantiles": quantiles(values),
        "absolute_quantiles": quantiles(absolute),
        "suggested_scale_positive_median_to_one": (
            1.0 / positive_median if positive_median is not None and positive_median > 0 else None
        ),
        "suggested_scale_absolute_median_to_one": (
            1.0 / absolute_median if absolute_median > 0 else None
        ),
        "suggested_scale_absolute_mean_to_one": (
            1.0 / absolute_mean if absolute_mean > 0 else None
        ),
    }


def command_audit_teacher_margins(args: argparse.Namespace) -> None:
    import numpy as np

    if args.batch_size < 1:
        raise ValueError("teacher margin audit batch size must be positive")
    rows = load_jsonl(args.rows)
    skills = load_jsonl(args.skills_jsonl)
    skill_ids = [skill_id(row) for row in skills]
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    if len(skill_to_index) != len(skills):
        raise ValueError("skill catalog contains duplicate IDs")
    candidate_indices, _, candidate_manifest = _load_candidate_shards(args.candidates_dir, rows)
    query_embeddings, query_manifest = _load_teacher_query_embedding_shards(
        args.teacher_query_embeddings_dir,
        rows,
    )
    skill_embeddings_sha256 = file_sha256(args.teacher_skill_embeddings)
    if (
        args.expected_teacher_skill_embeddings_sha256
        and skill_embeddings_sha256 != args.expected_teacher_skill_embeddings_sha256
    ):
        raise ValueError(
            "teacher skill embedding checksum changed: "
            f"expected={args.expected_teacher_skill_embeddings_sha256}, "
            f"actual={skill_embeddings_sha256}"
        )
    skill_embeddings_np = np.load(args.teacher_skill_embeddings, mmap_mode="r")
    skill_embeddings = torch.from_numpy(
        np.asarray(skill_embeddings_np, dtype=np.float32).copy()
    )
    if tuple(skill_embeddings.shape) != (len(skills), query_embeddings.shape[1]):
        raise ValueError("teacher skill embeddings do not align with catalog and query embeddings")

    source_positions = []
    positive_indices = []
    negative_indices = []
    is_multi_target = []
    is_multi_source = []
    for source_position, row in enumerate(rows):
        gold_ids = [str(value) for value in row.get("selected_skill_ids", [])]
        if not gold_ids:
            raise ValueError(f"row position={source_position} has no selected_skill_ids")
        try:
            gold_indices = {skill_to_index[sid] for sid in gold_ids}
        except KeyError as error:
            raise ValueError(f"unknown gold skill ID at row position={source_position}: {error}") from error
        negative_index = next(
            (
                int(candidate_index)
                for candidate_index in candidate_indices[source_position].tolist()
                if int(candidate_index) not in gold_indices
            ),
            None,
        )
        if negative_index is None:
            raise ValueError(f"no non-gold student candidate at row position={source_position}")
        is_multi_source.append(len(gold_ids) > 1)
        for gold_id in gold_ids:
            source_positions.append(source_position)
            positive_indices.append(skill_to_index[gold_id])
            negative_indices.append(negative_index)
            is_multi_target.append(len(gold_ids) > 1)

    all_margins = []
    source_positions_tensor = torch.tensor(source_positions, dtype=torch.long)
    positive_indices_tensor = torch.tensor(positive_indices, dtype=torch.long)
    negative_indices_tensor = torch.tensor(negative_indices, dtype=torch.long)
    for start in range(0, len(source_positions), args.batch_size):
        end = min(start + args.batch_size, len(source_positions))
        queries = query_embeddings.index_select(0, source_positions_tensor[start:end])
        positive_scores = (
            queries * skill_embeddings.index_select(0, positive_indices_tensor[start:end])
        ).sum(dim=1)
        negative_scores = (
            queries * skill_embeddings.index_select(0, negative_indices_tensor[start:end])
        ).sum(dim=1)
        all_margins.append((positive_scores - negative_scores).float())
    margins = torch.cat(all_margins)
    multi_mask = torch.tensor(is_multi_target, dtype=torch.bool)
    best_gold_margins = torch.full((len(rows),), -torch.inf, dtype=torch.float32)
    best_gold_margins.scatter_reduce_(
        0,
        source_positions_tensor,
        margins,
        reduce="amax",
        include_self=True,
    )
    source_multi_mask = torch.tensor(is_multi_source, dtype=torch.bool)
    summary = {
        "method": "bi_encoder_cosine_gold_minus_student_top_non_gold",
        "teacher_model": query_manifest.get("teacher_model"),
        "rows": len(rows),
        "expanded_target_rows": len(source_positions),
        "candidate_student_top_n": int(candidate_manifest["student_top_n"]),
        "teacher_query_manifest": query_manifest,
        "teacher_skill_embeddings": str(args.teacher_skill_embeddings),
        "teacher_skill_embeddings_sha256": skill_embeddings_sha256,
        "overall": _teacher_margin_summary(margins),
        "single": _teacher_margin_summary(margins[~multi_mask]),
        "multi": _teacher_margin_summary(margins[multi_mask]),
        "teacher_best_gold_by_query": {
            "overall": _teacher_margin_summary(best_gold_margins),
            "single": _teacher_margin_summary(best_gold_margins[~source_multi_mask]),
            "multi": _teacher_margin_summary(best_gold_margins[source_multi_mask]),
        },
    }
    write_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def command_mine(args: argparse.Namespace) -> None:
    import numpy as np
    from transformers import AutoTokenizer

    from skillrouter_embedding_train import last_token_pool, load_qwen_encoder, query_text

    rows_all = load_jsonl(args.rows)
    rows = rows_all[: args.max_rows] if args.max_rows is not None else rows_all
    if not rows:
        raise ValueError("no rows to mine")
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    sid_config = infer_structured_sid_config(
        semantic_ids, semantic_id_mode=semantic_id_mode
    )
    codebook_size = sid_config["codebook_size"]
    depth = sid_config["depth"]
    dedup_size = sid_config["dedup_size"]
    if depth != 4 or semantic_id_mode != "learned_codes":
        raise ValueError(
            f"this frozen experiment expects four learned codes, got depth={depth}, mode={semantic_id_mode}"
        )
    skills = load_jsonl(args.skills_jsonl)
    skill_ids = [skill_id(row) for row in skills]
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("skills_jsonl contains duplicate skill IDs")
    if set(skill_ids) != set(semantic_ids):
        raise ValueError("skills_jsonl and semantic_ids contain different skill IDs")
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    trie_children, sequence_to_skill, prefix_counts = build_semantic_trie_children(semantic_ids)
    sequence_to_index = {sequence: skill_to_index[sid] for sequence, sid in sequence_to_skill.items()}

    use_teacher_scores = not args.skip_teacher_scores
    teacher_embeddings_sha256 = None
    teacher_skill_embeddings = None
    if use_teacher_scores:
        if args.teacher_skill_embeddings is None:
            raise ValueError("--teacher-skill-embeddings is required unless --skip-teacher-scores is set")
        teacher_skill_embeddings_np = np.load(args.teacher_skill_embeddings, mmap_mode="r")
        if teacher_skill_embeddings_np.shape != (len(skills), 1024):
            raise ValueError(
                "teacher skill embeddings must align to canonical_skills.jsonl with shape "
                f"({len(skills)}, 1024), got {teacher_skill_embeddings_np.shape}"
            )
        teacher_embeddings_sha256 = file_sha256(args.teacher_skill_embeddings)
        if (
            args.expected_teacher_skill_embeddings_sha256
            and teacher_embeddings_sha256 != args.expected_teacher_skill_embeddings_sha256
        ):
            raise ValueError(
                "teacher skill embedding checksum changed: "
                f"expected={args.expected_teacher_skill_embeddings_sha256}, "
                f"actual={teacher_embeddings_sha256}"
            )
        teacher_skill_embeddings = torch.from_numpy(
            np.asarray(teacher_skill_embeddings_np, dtype=np.float32).copy()
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    init_payload = torch.load(
        args.checkpoint_dir / "structured_decoder.pt", map_location="cpu"
    )
    model = make_structured_t5_sid_model(
        args.model,
        codebook_size,
        depth,
        dedup_size,
        semantic_id_mode,
        tie_code_input_output=bool(init_payload.get("tie_code_input_output", False)),
    ).to(device)
    model.sid_codebook_initialized = bool(init_payload.get("sid_codebook_initialized", False))
    model.sid_codebook_init_scale = init_payload.get("sid_codebook_init_scale")
    checkpoint = load_structured_decoder_checkpoint(
        model, args.checkpoint_dir, device, semantic_id_mode=semantic_id_mode
    )
    validate_checkpoint_semantic_ids(checkpoint, semantic_ids, semantic_id_mode)
    model.eval()
    teacher_tokenizer = teacher_model = None
    if use_teacher_scores:
        teacher_tokenizer, teacher_model = load_qwen_encoder(
            args.teacher_model, device, bf16=bool(args.teacher_bf16)
        )
        teacher_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "paper": DGR_PAPER_URL,
        "official_code": DGR_CODE_URL,
        "rows_path": str(args.rows),
        "semantic_ids_path": str(args.semantic_ids),
        "skills_jsonl_path": str(args.skills_jsonl),
        "checkpoint_dir": str(args.checkpoint_dir),
        "teacher_model": args.teacher_model if use_teacher_scores else None,
        "teacher_skill_embeddings": str(args.teacher_skill_embeddings) if use_teacher_scores else None,
        "teacher_skill_embeddings_sha256": teacher_embeddings_sha256,
        "semantic_ids_sha256": semantic_ids_sha256(semantic_ids, semantic_id_mode),
        "student_top_n": args.student_top_n,
        "max_source_length": args.max_source_length,
        "query_max_chars": 1500,
        "rows": len(rows),
        "full_source_rows": len(rows_all),
        "shard_size": args.shard_size,
        "codebook_size": codebook_size,
        "depth": depth,
        "semantic_id_mode": semantic_id_mode,
        "student_score": (
            "sum_raw_conditional_logits_over_four_semantic_codes"
            if args.student_score_mode == "conditional_logit"
            else "sum_full_head_log_softmax_over_four_semantic_codes"
        ),
        "teacher_score": (
            "base_qwen_normalized_query_dot_frozen_document_embedding"
            if use_teacher_scores
            else "disabled"
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in config.items():
            if previous.get(key) != value:
                raise ValueError(
                    f"candidate manifest mismatch for {key}: previous={previous.get(key)!r}, current={value!r}"
                )

    shard_count = math.ceil(len(rows) / args.shard_size)
    completed_shards: list[str] = []
    started = time.time()
    for shard_index in range(shard_count):
        shard_name = f"part-{shard_index:06d}.pt"
        shard_path = shard_dir / shard_name
        start = shard_index * args.shard_size
        end = min(start + args.shard_size, len(rows))
        shard_rows = rows[start:end]
        expected_fingerprints = [dgr_row_fingerprint(row) for row in shard_rows]
        if shard_path.exists():
            existing = torch.load(shard_path, map_location="cpu")
            if (
                int(existing.get("start", -1)) != start
                or existing.get("row_fingerprints") != expected_fingerprints
                or tuple(existing["candidate_indices"].shape)
                != (len(shard_rows), args.student_top_n)
            ):
                raise ValueError(f"existing candidate shard failed validation: {shard_path}")
            completed_shards.append(shard_name)
            continue

        candidate_chunks = []
        teacher_rank_chunks = []
        for batch_start in range(0, len(shard_rows), args.query_batch_size):
            batch_rows = shard_rows[batch_start : batch_start + args.query_batch_size]
            encoded = tokenizer(
                [decoder_source_text(row) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=args.max_source_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.no_grad():
                encoder_hidden_states = model.encode_source(input_ids, attention_mask)
                candidate_codes, _ = topk_single_sid_scores_from_encoder(
                    model,
                    encoder_hidden_states,
                    attention_mask,
                    trie_children,
                    prefix_counts,
                    codebook_size=codebook_size,
                    depth=depth,
                    dedup_size=dedup_size,
                    semantic_id_mode=semantic_id_mode,
                    top_k=args.student_top_n,
                    decoder_batch_size=args.decoder_batch_size,
                    include_stop=False,
                    score_mode=args.student_score_mode,
                )
            candidate_indices = torch.tensor(
                [
                    [sequence_to_index[tuple(int(value) for value in sequence)] for sequence in row_codes]
                    for row_codes in candidate_codes.detach().cpu().tolist()
                ],
                dtype=torch.int64,
            )
            candidate_chunks.append(candidate_indices.to(torch.int32))
            if use_teacher_scores:
                query_embedding_chunks = []
                teacher_query_texts = [query_text(row, max_chars=1500) for row in batch_rows]
                for teacher_start in range(0, len(teacher_query_texts), args.teacher_batch_size):
                    teacher_encoded = teacher_tokenizer(
                        teacher_query_texts[teacher_start : teacher_start + args.teacher_batch_size],
                        padding=True,
                        truncation=True,
                        max_length=args.teacher_max_length,
                        return_tensors="pt",
                    )
                    teacher_encoded = {key: value.to(device) for key, value in teacher_encoded.items()}
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**teacher_encoded)
                        teacher_pooled = last_token_pool(
                            teacher_outputs.last_hidden_state, teacher_encoded["attention_mask"]
                        )
                        query_embedding_chunks.append(
                            F.normalize(teacher_pooled.float(), p=2, dim=1).detach().cpu()
                        )
                query_embeddings = torch.cat(query_embedding_chunks, dim=0)
                selected_skill_embeddings = teacher_skill_embeddings[candidate_indices]
                teacher_scores = torch.einsum(
                    "bd,bnd->bn", query_embeddings.float(), selected_skill_embeddings.float()
                )
                teacher_order = torch.argsort(teacher_scores, dim=1, descending=True)
                teacher_ranks = torch.empty_like(teacher_order)
                teacher_ranks.scatter_(
                    1,
                    teacher_order,
                    torch.arange(args.student_top_n).unsqueeze(0).expand_as(teacher_order),
                )
                teacher_rank_chunks.append(teacher_ranks.to(torch.int16))

        shard_payload = {
            "start": start,
            "end": end,
            "candidate_indices": torch.cat(candidate_chunks, dim=0),
            "row_fingerprints": expected_fingerprints,
        }
        if use_teacher_scores:
            shard_payload["teacher_ranks"] = torch.cat(teacher_rank_chunks, dim=0)
        _atomic_torch_save(shard_payload, shard_path)
        completed_shards.append(shard_name)
        write_json(
            manifest_path,
            {
                **config,
                "status": "running",
                "shards": completed_shards,
                "completed_rows": end,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(
            f"mined DGR candidates {end}/{len(rows)} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    audit_totals = {
        subset: {"rows": 0, "student_top1_hits": 0, "teacher_top1_hits": 0, "gold_recall_sum": 0.0}
        for subset in ("overall", "single", "multi")
    }
    for shard_name in completed_shards:
        shard = torch.load(shard_dir / shard_name, map_location="cpu")
        shard_start = int(shard["start"])
        shard_end = int(shard["end"])
        shard_audit = candidate_retrieval_audit(
            rows[shard_start:shard_end],
            shard["candidate_indices"],
            shard.get(
                "teacher_ranks",
                torch.full_like(shard["candidate_indices"], -1),
            ),
            skill_to_index,
        )
        for subset, values in shard_audit.items():
            for key in ("rows", "student_top1_hits", "teacher_top1_hits", "gold_recall_sum"):
                audit_totals[subset][key] += values[key]
    candidate_audit = {}
    for subset, values in audit_totals.items():
        row_count = int(values["rows"])
        candidate_audit[subset] = {
            **values,
            "student_top1_recall": float(values["student_top1_hits"] / row_count)
            if row_count
            else 0.0,
            "teacher_top1_within_student_pool_recall": float(
                values["teacher_top1_hits"] / row_count
            )
            if row_count
            else 0.0,
            "student_top_n_macro_recall": float(values["gold_recall_sum"] / row_count)
            if row_count
            else 0.0,
        }
        if not use_teacher_scores:
            candidate_audit[subset]["teacher_top1_hits"] = None
            candidate_audit[subset]["teacher_top1_within_student_pool_recall"] = None
    write_json(
        manifest_path,
        {
            **config,
            "status": "complete",
            "shards": completed_shards,
            "completed_rows": len(rows),
            "elapsed_seconds": time.time() - started,
            "candidate_retrieval_audit": candidate_audit,
        },
    )
    print(f"saved complete DGR candidate artifact to {args.output_dir}", flush=True)


@torch.no_grad()
def evaluate_retrieval(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    semantic_ids: dict[str, list[int]],
    semantic_id_mode: str,
    *,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    max_source_length: int,
    device,
    batch_size: int,
    decoder_batch_size: int,
    k_values: list[int],
    score_mode: str = "softmax",
    include_stop: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trie_children, sequence_to_skill, prefix_counts = build_semantic_trie_children(semantic_ids)
    predictions = []
    was_training = model.training
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        encoded = tokenizer(
            [decoder_source_text(row) for row in batch_rows],
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        encoder_hidden_states = model.encode_source(input_ids, attention_mask)
        codes, scores = topk_single_sid_scores_from_encoder(
            model,
            encoder_hidden_states,
            attention_mask,
            trie_children,
            prefix_counts,
            codebook_size=codebook_size,
            depth=depth,
            dedup_size=dedup_size,
            semantic_id_mode=semantic_id_mode,
            top_k=max(k_values),
            decoder_batch_size=decoder_batch_size,
            include_stop=include_stop,
            score_mode=score_mode,
        )
        for row, row_codes, row_scores in zip(
            batch_rows, codes.detach().cpu().tolist(), scores.detach().cpu().tolist()
        ):
            ranked = [sequence_to_skill[tuple(int(value) for value in values)] for values in row_codes]
            predictions.append(
                {
                    "index": row.get("index"),
                    "source_index": row.get("source_index"),
                    "query": row.get("query"),
                    "gold_skill_ids": [str(value) for value in row.get("selected_skill_ids", [])],
                    "ranked_skill_ids": ranked,
                    "ranked_skill_scores": [float(value) for value in row_scores],
                }
            )
        print(f"evaluated retrieval {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    if was_training:
        model.train()
    return predictions, compute_ranked_subset_metrics(rows, predictions, k_values)


def command_eval_retrieval(args: argparse.Namespace) -> None:
    """Evaluate a frozen single-SID checkpoint with batched trie beam search."""

    from transformers import AutoTokenizer

    rows = load_jsonl(args.rows)
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(
            f"evaluation row count changed: expected={args.expected_rows}, actual={len(rows)}"
        )
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    sid_config = infer_structured_sid_config(
        semantic_ids, semantic_id_mode=semantic_id_mode
    )
    checkpoint_path = args.model_dir / "structured_decoder.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_checkpoint_semantic_ids(checkpoint, semantic_ids, semantic_id_mode)
    for key in ("codebook_size", "depth", "dedup_size"):
        expected = int(sid_config[key])
        actual = int(checkpoint.get(key, 0))
        if actual != expected:
            raise ValueError(
                f"checkpoint {key} does not match semantic IDs: {actual} != {expected}"
            )
    include_stop = args.include_stop
    if include_stop is None:
        include_stop = args.score_mode == "softmax"
    if args.score_mode == "conditional_logit" and include_stop:
        raise ValueError("conditional_logit evaluates only the fixed-length SID code logits")
    k_values = sorted({int(value) for value in args.k_values.split(",")})
    if not k_values or min(k_values) < 1:
        raise ValueError("k_values must contain positive integers")
    if (args.output_dir / "predictions.jsonl").exists():
        raise FileExistsError(
            f"refusing to overwrite retrieval evaluation: {args.output_dir}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = make_structured_t5_sid_model(
        str(checkpoint["model"]),
        sid_config["codebook_size"],
        sid_config["depth"],
        sid_config["dedup_size"],
        semantic_id_mode,
        tie_code_input_output=bool(checkpoint.get("tie_code_input_output", False)),
    ).to(device)
    load_structured_decoder_checkpoint(
        model, args.model_dir, device, semantic_id_mode=semantic_id_mode
    )
    predictions, metrics = evaluate_retrieval(
        model,
        tokenizer,
        rows,
        semantic_ids,
        semantic_id_mode,
        codebook_size=sid_config["codebook_size"],
        depth=sid_config["depth"],
        dedup_size=sid_config["dedup_size"],
        max_source_length=args.max_source_length,
        device=device,
        batch_size=args.batch_size,
        decoder_batch_size=args.decoder_batch_size,
        k_values=k_values,
        score_mode=args.score_mode,
        include_stop=bool(include_stop),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "ranked_subset_metrics.json", metrics)
    write_json(
        args.output_dir / "eval_manifest.json",
        {
            "rows": str(args.rows),
            "row_count": len(rows),
            "model_dir": str(args.model_dir),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "semantic_ids": str(args.semantic_ids),
            "semantic_ids_sha256": semantic_ids_sha256(semantic_ids, semantic_id_mode),
            "score_mode": args.score_mode,
            "include_stop": bool(include_stop),
            "batch_size": args.batch_size,
            "decoder_batch_size": args.decoder_batch_size,
            "max_source_length": args.max_source_length,
            "k_values": k_values,
            "device": str(device),
        },
    )
    compact = {
        subset: {
            f"R@{k}": float(subset_metrics["overall"][f"@{k}"]["macro_recall"])
            for k in k_values
        }
        for subset, subset_metrics in metrics.items()
    }
    print(json.dumps({"retrieval_metrics": compact}, sort_keys=True), flush=True)


def command_train(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    is_prefix_objective = args.rank_objective == "ripor_prefix_margin_mse"
    retrieval_score_mode = (
        "softmax"
        if not is_prefix_objective or args.prefix_eval_standard_softmax
        else "conditional_logit"
    )
    retrieval_include_stop = not is_prefix_objective or args.prefix_eval_standard_softmax
    source_rows_all = load_jsonl(args.rows)
    source_rows = (
        source_rows_all[: args.max_source_rows]
        if args.max_source_rows is not None
        else source_rows_all
    )
    if len(source_rows) != args.expected_source_rows:
        raise ValueError(
            f"source row count changed: expected={args.expected_source_rows}, actual={len(source_rows)}"
        )
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    sid_config = infer_structured_sid_config(
        semantic_ids, semantic_id_mode=semantic_id_mode
    )
    codebook_size = sid_config["codebook_size"]
    depth = sid_config["depth"]
    dedup_size = sid_config["dedup_size"]
    if depth != 4 or semantic_id_mode != "learned_codes":
        raise ValueError(
            f"this frozen experiment expects four learned codes, got depth={depth}, mode={semantic_id_mode}"
        )
    skills = load_jsonl(args.skills_jsonl)
    skill_ids = [skill_id(row) for row in skills]
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    if len(skill_to_index) != len(skills) or set(skill_to_index) != set(semantic_ids):
        raise ValueError("skill catalog and semantic IDs do not align one-to-one")
    candidate_indices, teacher_ranks, candidate_manifest = _load_candidate_shards(
        args.candidates_dir, source_rows
    )
    if not is_prefix_objective and candidate_manifest.get("teacher_score") == "disabled":
        raise ValueError("dgr_distilled_ranknet requires candidate artifacts with teacher scores")
    if args.rank_weight < 0.0 or args.generation_weight < 0.0:
        raise ValueError("generation and rank weights must be non-negative")
    if args.rank_weight == 0.0 and args.generation_weight == 0.0:
        raise ValueError("at least one training loss weight must be positive")
    if args.rank_gradient_target_ratio is not None:
        if not is_prefix_objective:
            raise ValueError("rank gradient calibration currently requires the RIPOR prefix objective")
        if args.rank_gradient_target_ratio <= 0.0 or args.generation_weight <= 0.0:
            raise ValueError("rank gradient target ratio and generation weight must be positive")
    if not is_prefix_objective and args.prefix_score_mode != "conditional_logit":
        raise ValueError("prefix score mode applies only to the RIPOR prefix objective")
    if args.teacher_margin_scale <= 0.0:
        raise ValueError("teacher margin scale must be positive")

    teacher_query_embeddings = None
    teacher_skill_embeddings = None
    teacher_query_manifest = None
    teacher_skill_embeddings_sha256 = None
    teacher_best_gold_indices = None
    if is_prefix_objective and args.prefix_target_mode == "teacher_margin":
        import numpy as np

        if args.teacher_query_embeddings_dir is None or args.teacher_skill_embeddings is None:
            raise ValueError(
                "teacher_margin prefix targets require --teacher-query-embeddings-dir and --teacher-skill-embeddings"
            )
        teacher_query_embeddings, teacher_query_manifest = _load_teacher_query_embedding_shards(
            args.teacher_query_embeddings_dir,
            source_rows,
        )
        teacher_skill_embeddings_sha256 = file_sha256(args.teacher_skill_embeddings)
        if (
            args.expected_teacher_skill_embeddings_sha256
            and teacher_skill_embeddings_sha256 != args.expected_teacher_skill_embeddings_sha256
        ):
            raise ValueError(
                "teacher skill embedding checksum changed: "
                f"expected={args.expected_teacher_skill_embeddings_sha256}, "
                f"actual={teacher_skill_embeddings_sha256}"
            )
        teacher_skill_embeddings_np = np.load(args.teacher_skill_embeddings, mmap_mode="r")
        if teacher_skill_embeddings_np.shape[0] != len(skills):
            raise ValueError("teacher skill embeddings do not align with skills_jsonl")
        teacher_skill_embeddings = torch.from_numpy(
            np.asarray(teacher_skill_embeddings_np, dtype=np.float32).copy()
        )
        if teacher_query_embeddings.shape[1] != teacher_skill_embeddings.shape[1]:
            raise ValueError("teacher query and skill embedding dimensions differ")
        if args.prefix_positive_policy == "teacher_best_gold":
            teacher_best_gold_indices = teacher_best_gold_indices_for_rows(
                source_rows,
                teacher_query_embeddings,
                teacher_skill_embeddings,
                skill_to_index,
            )
    elif args.prefix_target_mode == "teacher_margin":
        raise ValueError("teacher_margin prefix targets require ripor_prefix_margin_mse")
    if args.prefix_positive_policy == "teacher_best_gold" and (
        not is_prefix_objective or args.prefix_target_mode != "teacher_margin"
    ):
        raise ValueError("teacher_best_gold requires RIPOR with teacher_margin targets")
    if args.teacher_positive_margins_only and (
        not is_prefix_objective or args.prefix_target_mode != "teacher_margin"
    ):
        raise ValueError("teacher-positive-margins-only requires RIPOR teacher_margin targets")

    source_rows_with_positions = []
    for source_position, row in enumerate(source_rows):
        copied = dict(row)
        copied["_dgr_source_position"] = source_position
        source_rows_with_positions.append(copied)
    train_rows = expand_single_sid_query_rows(source_rows_with_positions, semantic_ids)
    if len(train_rows) != args.expected_expanded_rows:
        raise ValueError(
            f"expanded row count changed: expected={args.expected_expanded_rows}, actual={len(train_rows)}"
        )
    dev_rows_all = load_jsonl(args.dev_rows)
    dev_rows = (
        dev_rows_all[: args.max_dev_rows]
        if args.max_dev_rows is not None
        else dev_rows_all
    )
    official_rows = load_jsonl(args.official_rows) if args.official_rows else []
    candidate_code_table = torch.tensor(
        [semantic_ids[sid] for sid in skill_ids], dtype=torch.long
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    init_payload = torch.load(
        args.init_model_dir / "structured_decoder.pt", map_location="cpu"
    )
    model = make_structured_t5_sid_model(
        args.model,
        codebook_size,
        depth,
        dedup_size,
        semantic_id_mode,
        tie_code_input_output=bool(init_payload.get("tie_code_input_output", False)),
    ).to(device)
    model.sid_codebook_initialized = bool(init_payload.get("sid_codebook_initialized", False))
    model.sid_codebook_init_scale = init_payload.get("sid_codebook_init_scale")
    init_checkpoint = load_structured_decoder_checkpoint(
        model, args.init_model_dir, device, semantic_id_mode=semantic_id_mode
    )
    validate_checkpoint_semantic_ids(init_checkpoint, semantic_ids, semantic_id_mode)

    trainable_groups = parse_trainable_parameter_groups(args.trainable_parameter_groups)
    trainable_parameters = []
    trainable_parameter_names = []
    frozen_parameter_count = 0
    for name, parameter in model.named_parameters():
        trainable = structured_parameter_group(name) in trainable_groups
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_parameters.append(parameter)
            trainable_parameter_names.append(name)
        else:
            frozen_parameter_count += parameter.numel()
    if not trainable_parameters:
        raise ValueError("trainable parameter group selection froze the entire model")

    # DGR is a separate continuation stage: model weights continue, Adam moments do not.
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8
    )
    updates_per_epoch = math.ceil(len(train_rows) / args.batch_size / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )
    k_values = sorted({int(value) for value in args.eval_k_values.split(",")})
    if not k_values or min(k_values) < 1:
        raise ValueError("eval_k_values must contain positive integers")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_history_path = args.output_dir / "loss_history.jsonl"
    if loss_history_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing DGR training history: {loss_history_path}"
        )
    training_method = (
        "ripor_prefix_margin_mse_continuation"
        if is_prefix_objective
        else "dgr_distilled_ranknet_continuation"
    )
    checkpoint_metadata = {
        "max_source_length": args.max_source_length,
        "source_text_policy": init_checkpoint.get("source_text_policy"),
        "semantic_id_count": len(semantic_ids),
        "semantic_ids_sha256": semantic_ids_sha256(semantic_ids, semantic_id_mode),
        "decoder_score_mode": retrieval_score_mode,
        "decoder_include_stop": retrieval_include_stop,
        "dgr_paper": DGR_PAPER_URL,
        "dgr_official_code": DGR_CODE_URL,
        "dgr_student_top_n": int(candidate_manifest["student_top_n"]),
        "dgr_list_size": args.list_size,
        "dgr_sampling": (
            "one_labeled_positive_plus_student_highest_ranked_non_gold"
            if is_prefix_objective
            else "one_labeled_positive_plus_teacher_top_negative_plus_random_student_negatives"
        ),
        "dgr_base_margin": args.base_margin,
        "dgr_gap_margin": args.gap_margin,
        "dgr_generation_weight": args.generation_weight,
        "dgr_rank_weight": args.rank_weight,
        "dgr_rank_gradient_target_ratio": args.rank_gradient_target_ratio,
        "dgr_effective_rank_weight": (
            None if args.rank_gradient_target_ratio is not None else args.rank_weight
        ),
        "dgr_effective_rank_weight_by_epoch": {},
        "dgr_student_score": (
            (
                "sum_raw_conditional_logits_over_four_semantic_codes"
                if args.prefix_score_mode == "conditional_logit"
                else "sum_full_head_log_softmax_plus_final_stop_logprob"
            )
            if is_prefix_objective
            else "sum_full_head_log_softmax_over_four_semantic_codes"
        ),
        "rank_objective": args.rank_objective,
        "ripor_paper": RIPOR_PAPER_URL if is_prefix_objective else None,
        "ripor_official_code": RIPOR_CODE_URL if is_prefix_objective else None,
        "ripor_audited_code_commit": RIPOR_AUDITED_CODE_COMMIT if is_prefix_objective else None,
        "ripor_negative_sampling": (
            "student_highest_ranked_non_gold_paths" if is_prefix_objective else None
        ),
        "ripor_prefix_score": (
            (
                "cumulative_sum_of_raw_conditional_logits"
                if args.prefix_score_mode == "conditional_logit"
                else "cumulative_full_head_log_softmax_with_final_stop_logprob"
            )
            if is_prefix_objective
            else None
        ),
        "ripor_prefix_score_mode": args.prefix_score_mode if is_prefix_objective else None,
        "ripor_target_mode": args.prefix_target_mode if is_prefix_objective else None,
        "ripor_positive_policy": args.prefix_positive_policy if is_prefix_objective else None,
        "ripor_target_margin": (
            args.prefix_target_margin
            if is_prefix_objective and args.prefix_target_mode == "fixed_margin"
            else None
        ),
        "ripor_teacher_margin_scale": (
            args.teacher_margin_scale
            if is_prefix_objective and args.prefix_target_mode == "teacher_margin"
            else None
        ),
        "ripor_teacher_positive_margins_only": bool(
            is_prefix_objective and args.teacher_positive_margins_only
        ),
        "ripor_teacher_query_embeddings_dir": (
            str(args.teacher_query_embeddings_dir)
            if is_prefix_objective and args.prefix_target_mode == "teacher_margin"
            else None
        ),
        "ripor_teacher_query_manifest": teacher_query_manifest,
        "ripor_teacher_skill_embeddings": (
            str(args.teacher_skill_embeddings)
            if is_prefix_objective and args.prefix_target_mode == "teacher_margin"
            else None
        ),
        "ripor_teacher_skill_embeddings_sha256": teacher_skill_embeddings_sha256,
        "ripor_prefix_margin_beta": args.prefix_margin_beta if is_prefix_objective else None,
        "ripor_mask_identical_prefix_pairs": bool(is_prefix_objective),
        "ripor_progressive_prefixes": bool(args.progressive_prefixes and is_prefix_objective),
        "ripor_full_sequence_warmup": bool(args.full_sequence_warmup and is_prefix_objective),
        "ripor_short_sid_adaptation": (
            "teacher relevance margin decomposed across active prefixes with expanded CE anchor"
            if is_prefix_objective and args.prefix_target_mode == "teacher_margin"
            else (
                "equal binary-relevance target margin at each active prefix; no teacher ranking"
                if is_prefix_objective
                else None
            )
        ),
        "fresh_optimizer": True,
        "trainable_parameter_groups": sorted(trainable_groups),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
        "frozen_parameter_count": frozen_parameter_count,
        "trainable_parameter_names": trainable_parameter_names,
        "checkpoint_selection": "private_dev_overall_macro_recall_at_1",
        "checkpoint_selection_uses_test_set": False,
        "checkpoint_selection_includes_init": bool(args.include_init_in_checkpoint_selection),
    }
    write_json(
        args.output_dir / "run_config.json",
        {
            **checkpoint_metadata,
            "rows": str(args.rows),
            "dev_rows": str(args.dev_rows),
            "official_rows": str(args.official_rows) if args.official_rows else None,
            "semantic_ids": str(args.semantic_ids),
            "skills_jsonl": str(args.skills_jsonl),
            "candidates_dir": str(args.candidates_dir),
            "init_model_dir": str(args.init_model_dir),
            "source_rows": len(source_rows),
            "full_source_rows": len(source_rows_all),
            "expanded_rows": len(train_rows),
            "dev_source_rows": len(dev_rows),
            "full_dev_source_rows": len(dev_rows_all),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "total_updates": total_updates,
            "warmup_steps": warmup_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "rank_gradient_target_ratio": args.rank_gradient_target_ratio,
            "bf16": bool(args.bf16),
            "seed": args.seed,
        },
    )

    best_dev_r1: float | None = None
    best_dir = args.output_dir / "best"
    global_update = 0
    effective_rank_weight = float(args.rank_weight)
    gradient_calibration_events = []
    gradient_calibration_epoch = None
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    if args.include_init_in_checkpoint_selection:
        predictions, initial_dev_metrics = evaluate_retrieval(
            model,
            tokenizer,
            dev_rows,
            semantic_ids,
            semantic_id_mode,
            codebook_size=codebook_size,
            depth=depth,
            dedup_size=dedup_size,
            max_source_length=args.max_source_length,
            device=device,
            batch_size=args.eval_batch_size,
            decoder_batch_size=args.decoder_batch_size,
            k_values=k_values,
            score_mode=retrieval_score_mode,
            include_stop=retrieval_include_stop,
        )
        initial_dev_dir = args.output_dir / "dev_eval" / "epoch_000"
        initial_dev_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(initial_dev_dir / "predictions.jsonl", predictions)
        write_json(initial_dev_dir / "ranked_subset_metrics.json", initial_dev_metrics)
        best_dev_r1 = float(
            initial_dev_metrics["overall"]["overall"]["@1"]["macro_recall"]
        )
        initial_eval_event = {
            "event": "dev_retrieval",
            "epoch": 0,
            "update": 0,
            "dev_overall_r1": best_dev_r1,
            "dev_metrics": {
                subset: {
                    f"R@{k}": float(metrics["overall"][f"@{k}"]["macro_recall"])
                    for k in k_values
                }
                for subset, metrics in initial_dev_metrics.items()
            },
        }
        with loss_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(initial_eval_event, sort_keys=True) + "\n")
        print(json.dumps(initial_eval_event, sort_keys=True), flush=True)
        save_structured_decoder_checkpoint(
            best_dir,
            model,
            tokenizer,
            args.model,
            codebook_size,
            depth,
            dedup_size,
            args.seed,
            training_method,
            {
                **checkpoint_metadata,
                "best_epoch": 0,
                "global_update": 0,
                "best_dev_overall_r1": best_dev_r1,
                "scheduler_state_dict": scheduler.state_dict(),
            },
            optimizer=optimizer,
            semantic_id_mode=semantic_id_mode,
        )
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_rows)
        model.train()
        active_prefix_indices = (
            progressive_prefix_indices(
                depth=depth,
                epoch=epoch,
                full_sequence_warmup=args.full_sequence_warmup,
            )
            if args.progressive_prefixes
            else list(range(depth))
        )
        active_prefixes = len(active_prefix_indices)
        for batch_index, start in enumerate(range(0, len(train_rows), args.batch_size), start=1):
            batch_rows = train_rows[start : start + args.batch_size]
            batch = structured_sid_batch(
                batch_rows,
                tokenizer,
                semantic_ids,
                codebook_size,
                depth,
                dedup_size,
                args.max_source_length,
                device,
                semantic_id_mode,
            )
            selected_indices = []
            selected_teacher_ranks = []
            for row in batch_rows:
                source_position = int(row["_dgr_source_position"])
                gold_ids = {str(value) for value in row["source_selected_skill_ids"]}
                gold_indices = {skill_to_index[sid] for sid in gold_ids}
                positive_index = skill_to_index[str(row["positive_skill_id"])]
                if is_prefix_objective:
                    row_candidates = select_prefix_candidate_list(
                        candidate_indices[source_position].tolist(),
                        gold_indices,
                        positive_index,
                        args.list_size,
                    )
                    row_teacher_ranks = []
                else:
                    row_candidates, row_teacher_ranks = select_dgr_candidate_list(
                        candidate_indices[source_position].tolist(),
                        teacher_ranks[source_position].tolist(),
                        gold_indices,
                        positive_index,
                        args.list_size,
                        rng,
                    )
                selected_indices.append(row_candidates)
                selected_teacher_ranks.append(row_teacher_ranks)
            selected_indices_tensor = torch.tensor(
                selected_indices, dtype=torch.long, device=device
            )
            candidate_codes = candidate_code_table[selected_indices_tensor.cpu()].to(device)
            prefix_target_margin: float | torch.Tensor = args.prefix_target_margin
            teacher_margin_stats = None
            rank_example_mask = None
            if is_prefix_objective and args.prefix_target_mode == "teacher_margin":
                source_positions = torch.tensor(
                    [int(row["_dgr_source_position"]) for row in batch_rows],
                    dtype=torch.long,
                )
                prefix_target_margin = teacher_margins_for_candidates(
                    teacher_query_embeddings,
                    teacher_skill_embeddings,
                    source_positions,
                    selected_indices_tensor.detach().cpu(),
                    scale=args.teacher_margin_scale,
                ).to(device)
                if args.prefix_positive_policy == "teacher_best_gold":
                    rank_example_mask = (
                        selected_indices_tensor[:, 0].detach().cpu()
                        == teacher_best_gold_indices.index_select(0, source_positions)
                    ).to(device)
                margins_for_stats = (
                    prefix_target_margin[rank_example_mask]
                    if rank_example_mask is not None and rank_example_mask.any()
                    else prefix_target_margin
                )
                teacher_margin_stats = {
                    "teacher_margin_mean": float(margins_for_stats.detach().float().mean().cpu()),
                    "teacher_margin_min": float(margins_for_stats.detach().float().min().cpu()),
                    "teacher_margin_max": float(margins_for_stats.detach().float().max().cpu()),
                    "teacher_margin_positive_fraction": float(
                        (margins_for_stats.detach() > 0).float().mean().cpu()
                    ),
                    "rank_example_active_fraction": (
                        float(rank_example_mask.float().mean().detach().cpu())
                        if rank_example_mask is not None
                        else 1.0
                    ),
                }
            selected_teacher_ranks_tensor = (
                None
                if is_prefix_objective
                else torch.tensor(selected_teacher_ranks, dtype=torch.long, device=device)
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=bool(args.bf16 and device.type == "cuda"),
            ):
                encoder_hidden_states = model.encode_source(batch["input_ids"], batch["attention_mask"])
                generation_loss = generation_loss_from_encoder(model, encoder_hidden_states, batch)
                if is_prefix_objective:
                    if args.prefix_score_mode == "conditional_logit":
                        prefix_scores = candidate_prefix_logit_scores(
                            model,
                            encoder_hidden_states,
                            batch["attention_mask"],
                            candidate_codes,
                            codebook_size=codebook_size,
                            depth=depth,
                            dedup_size=dedup_size,
                            semantic_id_mode=semantic_id_mode,
                        )
                    else:
                        prefix_scores = candidate_prefix_logprob_scores(
                            model,
                            encoder_hidden_states,
                            batch["attention_mask"],
                            candidate_codes,
                            codebook_size=codebook_size,
                            depth=depth,
                            dedup_size=dedup_size,
                            semantic_id_mode=semantic_id_mode,
                            include_final_stop=(depth - 1 in active_prefix_indices),
                        )
                    rank_loss = prefix_margin_mse_loss(
                        prefix_scores,
                        active_prefixes=active_prefixes,
                        target_margin=prefix_target_margin,
                        margin_beta=args.prefix_margin_beta,
                        candidate_codes=candidate_codes,
                        active_prefix_indices=active_prefix_indices,
                        example_mask=rank_example_mask,
                        positive_target_margins_only=args.teacher_positive_margins_only,
                    )
                else:
                    student_scores = candidate_sequence_log_scores(
                        model,
                        encoder_hidden_states,
                        batch["attention_mask"],
                        candidate_codes,
                        codebook_size=codebook_size,
                        depth=depth,
                        dedup_size=dedup_size,
                        semantic_id_mode=semantic_id_mode,
                    )
                    rank_loss = distilled_ranknet_loss(
                        student_scores,
                        selected_teacher_ranks_tensor,
                        base_margin=args.base_margin,
                        gap_margin=args.gap_margin,
                    )
            if (
                args.rank_gradient_target_ratio is not None
                and gradient_calibration_epoch != epoch
            ):
                gradient_calibration = loss_gradient_alignment(
                    args.generation_weight * generation_loss,
                    rank_loss,
                    trainable_parameters,
                )
                rank_gradient_norm = gradient_calibration["unweighted_rank_grad_norm"]
                generation_gradient_norm = gradient_calibration["generation_component_grad_norm"]
                if rank_gradient_norm <= 0.0 or generation_gradient_norm <= 0.0:
                    raise ValueError(
                        "cannot calibrate rank weight from zero component gradient: "
                        f"{gradient_calibration}"
                    )
                effective_rank_weight = (
                    args.rank_gradient_target_ratio
                    * generation_gradient_norm
                    / rank_gradient_norm
                )
                gradient_calibration = {
                    "event": "gradient_calibration",
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "generation_weight": args.generation_weight,
                    "requested_rank_weight": args.rank_weight,
                    "rank_gradient_target_ratio": args.rank_gradient_target_ratio,
                    "effective_rank_weight": effective_rank_weight,
                    **gradient_calibration,
                }
                gradient_calibration_epoch = epoch
                gradient_calibration_events.append(gradient_calibration)
                checkpoint_metadata["dgr_effective_rank_weight"] = effective_rank_weight
                checkpoint_metadata["dgr_effective_rank_weight_by_epoch"][str(epoch)] = (
                    effective_rank_weight
                )
                write_json(
                    args.output_dir / f"gradient_calibration_epoch_{epoch:03d}.json",
                    gradient_calibration,
                )
                write_json(
                    args.output_dir / "gradient_calibration.json",
                    {"events": gradient_calibration_events},
                )
                with loss_history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(gradient_calibration, sort_keys=True) + "\n")
                print(json.dumps(gradient_calibration, sort_keys=True), flush=True)
            loss = (
                args.generation_weight * generation_loss
                + effective_rank_weight * rank_loss
            )
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            should_update = (
                batch_index % args.gradient_accumulation_steps == 0
                or start + args.batch_size >= len(train_rows)
            )
            if should_update:
                pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if (
                    global_update == 1
                    or global_update % args.logging_steps == 0
                    or global_update == total_updates
                ):
                    event = {
                        "event": "train",
                        "epoch": epoch,
                        "update": global_update,
                        "loss": float(loss.detach().cpu()),
                        "generation_loss": float(generation_loss.detach().cpu()),
                        "rank_loss": float(rank_loss.detach().cpu()),
                        "generation_weight": args.generation_weight,
                        "rank_weight": effective_rank_weight,
                        "pre_clip_grad_norm": float(pre_clip_grad_norm.detach().float().cpu()),
                        "grad_was_clipped": bool(
                            float(pre_clip_grad_norm.detach().float().cpu()) > args.max_grad_norm
                        ),
                        **(teacher_margin_stats or {}),
                        "active_prefixes": active_prefixes,
                        "active_prefix_levels": (
                            [index + 1 for index in active_prefix_indices]
                            if is_prefix_objective
                            else None
                        ),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "elapsed_seconds": time.time() - started,
                    }
                    with loss_history_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)

        predictions, dev_metrics = evaluate_retrieval(
            model,
            tokenizer,
            dev_rows,
            semantic_ids,
            semantic_id_mode,
            codebook_size=codebook_size,
            depth=depth,
            dedup_size=dedup_size,
            max_source_length=args.max_source_length,
            device=device,
            batch_size=args.eval_batch_size,
            decoder_batch_size=args.decoder_batch_size,
            k_values=k_values,
            score_mode=retrieval_score_mode,
            include_stop=retrieval_include_stop,
        )
        dev_dir = args.output_dir / "dev_eval" / f"epoch_{epoch:03d}"
        dev_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(dev_dir / "predictions.jsonl", predictions)
        write_json(dev_dir / "ranked_subset_metrics.json", dev_metrics)
        dev_r1 = float(dev_metrics["overall"]["overall"]["@1"]["macro_recall"])
        eval_event = {
            "event": "dev_retrieval",
            "epoch": epoch,
            "update": global_update,
            "dev_overall_r1": dev_r1,
            "dev_metrics": {
                subset: {
                    f"R@{k}": float(metrics["overall"][f"@{k}"]["macro_recall"])
                    for k in k_values
                }
                for subset, metrics in dev_metrics.items()
            },
        }
        with loss_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(eval_event, sort_keys=True) + "\n")
        print(json.dumps(eval_event, sort_keys=True), flush=True)
        epoch_dir = args.output_dir / "checkpoints" / f"epoch_{epoch:03d}"
        save_structured_decoder_checkpoint(
            epoch_dir,
            model,
            tokenizer,
            args.model,
            codebook_size,
            depth,
            dedup_size,
            args.seed,
            training_method,
            {
                **checkpoint_metadata,
                "epoch": epoch,
                "global_update": global_update,
                "dev_overall_r1": dev_r1,
                "scheduler_state_dict": scheduler.state_dict(),
            },
            optimizer=optimizer,
            semantic_id_mode=semantic_id_mode,
        )
        if best_dev_r1 is None or dev_r1 > best_dev_r1:
            best_dev_r1 = dev_r1
            save_structured_decoder_checkpoint(
                best_dir,
                model,
                tokenizer,
                args.model,
                codebook_size,
                depth,
                dedup_size,
                args.seed,
                training_method,
                {
                    **checkpoint_metadata,
                    "best_epoch": epoch,
                    "global_update": global_update,
                    "best_dev_overall_r1": best_dev_r1,
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                optimizer=optimizer,
                semantic_id_mode=semantic_id_mode,
            )

    load_structured_decoder_checkpoint(
        model, best_dir, device, semantic_id_mode=semantic_id_mode
    )
    final_dir = args.output_dir / "final"
    save_structured_decoder_checkpoint(
        final_dir,
        model,
        tokenizer,
        args.model,
        codebook_size,
        depth,
        dedup_size,
        args.seed,
        training_method,
        {
            **checkpoint_metadata,
            "selected_checkpoint": "best_private_dev_retrieval_r1",
            "best_dev_overall_r1": best_dev_r1,
        },
        semantic_id_mode=semantic_id_mode,
    )

    official_metrics = None
    if official_rows:
        predictions, official_metrics = evaluate_retrieval(
            model,
            tokenizer,
            official_rows,
            semantic_ids,
            semantic_id_mode,
            codebook_size=codebook_size,
            depth=depth,
            dedup_size=dedup_size,
            max_source_length=args.max_source_length,
            device=device,
            batch_size=args.eval_batch_size,
            decoder_batch_size=args.decoder_batch_size,
            k_values=k_values,
            score_mode=retrieval_score_mode,
            include_stop=retrieval_include_stop,
        )
        official_dir = args.output_dir / "official_final"
        official_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(official_dir / "predictions.jsonl", predictions)
        write_json(official_dir / "ranked_subset_metrics.json", official_metrics)
        print(json.dumps({"official_final": official_metrics}, sort_keys=True), flush=True)

    write_json(
        args.output_dir / "train_manifest.json",
        {
            **checkpoint_metadata,
            "source_rows": len(source_rows),
            "expanded_rows": len(train_rows),
            "dev_source_rows": len(dev_rows),
            "official_source_rows": len(official_rows),
            "epochs": args.epochs,
            "global_updates": global_update,
            "best_dev_overall_r1": best_dev_r1,
            "best_checkpoint": str(best_dir),
            "final_checkpoint": str(final_dir),
            "official_final_metrics": official_metrics,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"saved {training_method} to {final_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_teacher = subparsers.add_parser("encode-teacher-queries")
    encode_teacher.add_argument("--rows", type=Path, required=True)
    encode_teacher.add_argument("--output-dir", type=Path, required=True)
    encode_teacher.add_argument("--teacher-model", default="Qwen/Qwen3-Embedding-0.6B")
    encode_teacher.add_argument("--batch-size", type=int, default=32)
    encode_teacher.add_argument("--teacher-max-length", type=int, default=2048)
    encode_teacher.add_argument("--shard-size", type=int, default=4096)
    encode_teacher.add_argument("--max-rows", type=int)
    encode_teacher.add_argument("--teacher-bf16", action=argparse.BooleanOptionalAction, default=True)
    encode_teacher.add_argument("--cpu", action="store_true")
    encode_teacher.set_defaults(func=command_encode_teacher_queries)

    encode_teacher_skills = subparsers.add_parser("encode-teacher-skills")
    encode_teacher_skills.add_argument("--skills-jsonl", type=Path, required=True)
    encode_teacher_skills.add_argument("--output", type=Path, required=True)
    encode_teacher_skills.add_argument("--manifest", type=Path)
    encode_teacher_skills.add_argument(
        "--teacher-model", default="Qwen/Qwen3-Embedding-0.6B"
    )
    encode_teacher_skills.add_argument("--batch-size", type=int, default=8)
    encode_teacher_skills.add_argument("--teacher-max-length", type=int, default=2048)
    encode_teacher_skills.add_argument("--desc-max", type=int, default=300)
    encode_teacher_skills.add_argument("--body-max", type=int, default=2500)
    encode_teacher_skills.add_argument(
        "--teacher-bf16", action=argparse.BooleanOptionalAction, default=True
    )
    encode_teacher_skills.add_argument("--cpu", action="store_true")
    encode_teacher_skills.set_defaults(func=command_encode_teacher_skills)

    audit_teacher = subparsers.add_parser("audit-teacher-margins")
    audit_teacher.add_argument("--rows", type=Path, required=True)
    audit_teacher.add_argument("--skills-jsonl", type=Path, required=True)
    audit_teacher.add_argument("--candidates-dir", type=Path, required=True)
    audit_teacher.add_argument("--teacher-query-embeddings-dir", type=Path, required=True)
    audit_teacher.add_argument("--teacher-skill-embeddings", type=Path, required=True)
    audit_teacher.add_argument("--expected-teacher-skill-embeddings-sha256")
    audit_teacher.add_argument("--output", type=Path, required=True)
    audit_teacher.add_argument("--batch-size", type=int, default=8192)
    audit_teacher.set_defaults(func=command_audit_teacher_margins)

    mine = subparsers.add_parser("mine")
    mine.add_argument("--rows", type=Path, required=True)
    mine.add_argument("--semantic-ids", type=Path, required=True)
    mine.add_argument("--skills-jsonl", type=Path, required=True)
    mine.add_argument("--checkpoint-dir", type=Path, required=True)
    mine.add_argument("--teacher-skill-embeddings", type=Path)
    mine.add_argument("--expected-teacher-skill-embeddings-sha256")
    mine.add_argument("--skip-teacher-scores", action="store_true")
    mine.add_argument("--output-dir", type=Path, required=True)
    mine.add_argument("--model", default="t5-base")
    mine.add_argument("--teacher-model", default="Qwen/Qwen3-Embedding-0.6B")
    mine.add_argument("--student-top-n", type=int, default=200)
    mine.add_argument(
        "--student-score-mode",
        choices=["softmax", "conditional_logit"],
        default="softmax",
    )
    mine.add_argument("--query-batch-size", type=int, default=8)
    mine.add_argument("--decoder-batch-size", type=int, default=256)
    mine.add_argument("--teacher-batch-size", type=int, default=8)
    mine.add_argument("--max-source-length", type=int, default=2048)
    mine.add_argument("--teacher-max-length", type=int, default=2048)
    mine.add_argument("--shard-size", type=int, default=1024)
    mine.add_argument("--max-rows", type=int)
    mine.add_argument("--teacher-bf16", action=argparse.BooleanOptionalAction, default=True)
    mine.add_argument("--cpu", action="store_true")
    mine.set_defaults(func=command_mine)

    evaluate = subparsers.add_parser("eval-retrieval")
    evaluate.add_argument("--rows", type=Path, required=True)
    evaluate.add_argument("--semantic-ids", type=Path, required=True)
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--expected-rows", type=int)
    evaluate.add_argument(
        "--score-mode",
        choices=["softmax", "conditional_logit"],
        default="softmax",
    )
    evaluate.add_argument(
        "--include-stop",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--decoder-batch-size", type=int, default=512)
    evaluate.add_argument("--k-values", default="1,5,10,50")
    evaluate.add_argument("--max-source-length", type=int, default=2048)
    evaluate.add_argument("--cpu", action="store_true")
    evaluate.set_defaults(func=command_eval_retrieval)

    train = subparsers.add_parser("train")
    train.add_argument("--rows", type=Path, required=True)
    train.add_argument("--dev-rows", type=Path, required=True)
    train.add_argument("--official-rows", type=Path)
    train.add_argument("--semantic-ids", type=Path, required=True)
    train.add_argument("--skills-jsonl", type=Path, required=True)
    train.add_argument("--candidates-dir", type=Path, required=True)
    train.add_argument("--init-model-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--model", default="t5-base")
    train.add_argument("--expected-source-rows", type=int, default=94912)
    train.add_argument("--expected-expanded-rows", type=int, default=124175)
    train.add_argument("--max-source-rows", type=int)
    train.add_argument("--max-dev-rows", type=int)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--eval-batch-size", type=int, default=8)
    train.add_argument("--decoder-batch-size", type=int, default=256)
    train.add_argument("--eval-k-values", default="1,5,10,50")
    train.add_argument("--list-size", type=int, default=6)
    train.add_argument(
        "--rank-objective",
        choices=["dgr_distilled_ranknet", "ripor_prefix_margin_mse"],
        default="dgr_distilled_ranknet",
    )
    train.add_argument("--base-margin", type=float, default=3.0)
    train.add_argument("--gap-margin", type=float, default=1.0)
    train.add_argument(
        "--prefix-target-mode",
        choices=["fixed_margin", "teacher_margin"],
        default="fixed_margin",
    )
    train.add_argument(
        "--prefix-positive-policy",
        choices=["expanded_gold", "teacher_best_gold"],
        default="expanded_gold",
    )
    train.add_argument("--prefix-target-margin", type=float, default=1.0)
    train.add_argument("--teacher-query-embeddings-dir", type=Path)
    train.add_argument("--teacher-skill-embeddings", type=Path)
    train.add_argument("--expected-teacher-skill-embeddings-sha256")
    train.add_argument("--teacher-margin-scale", type=float, default=1.0)
    train.add_argument("--teacher-positive-margins-only", action="store_true")
    train.add_argument(
        "--prefix-score-mode",
        choices=["conditional_logit", "softmax_stop"],
        default="conditional_logit",
    )
    train.add_argument("--prefix-margin-beta", type=float, default=0.0)
    train.add_argument(
        "--progressive-prefixes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train.add_argument("--full-sequence-warmup", action="store_true")
    train.add_argument("--prefix-eval-standard-softmax", action="store_true")
    train.add_argument("--include-init-in-checkpoint-selection", action="store_true")
    train.add_argument("--generation-weight", type=float, default=5.0)
    train.add_argument("--rank-weight", type=float, default=1.0)
    train.add_argument(
        "--rank-gradient-target-ratio",
        type=float,
        help=(
            "calibrate rank weight on the first training batch so its gradient norm is this "
            "fraction of the weighted generation-loss gradient norm"
        ),
    )
    train.add_argument(
        "--trainable-parameter-groups",
        default="all",
        help=(
            "comma-separated groups from encoder_blocks,decoder_and_shared,sid_heads; "
            "use all for the complete model"
        ),
    )
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--warmup-ratio", type=float, default=0.1)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--max-source-length", type=int, default=2048)
    train.add_argument("--logging-steps", type=int, default=100)
    train.add_argument("--seed", type=int, default=20260626)
    train.add_argument("--bf16", action="store_true")
    train.add_argument("--cpu", action="store_true")
    train.set_defaults(func=command_train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
