#!/usr/bin/env python3
"""Skill generative-retrieval training pipeline over the 2000-row dataset.

The skill flattening and query instruction follow SkillRouter:
skill text is "name | description | body"; the default embedding model is the
base Qwen3-Embedding-0.6B rather than SkillRouter's fine-tuned embedding model.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from distributed_utils import (
    DistributedContext,
    all_gather_object,
    all_gather_tensor_with_grad,
    barrier,
    destroy_distributed,
    distributed_epoch_batches,
    global_normalized_loss,
    initialize_distributed,
    reduce_sum_count,
    shard_with_indices,
    unwrap_model,
    wrap_ddp,
)


SKILLROUTER_QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)

SKILL_SEP_TOKEN = "<SKILL_SEP>"
NATIVE_SKILL_SEP_TEXT = "|"
TRIE_TERMINAL_KEY = "$"
SKILLROUTER_QUERY_MAX_CHARS = 1500
DEFAULT_GR_MAX_SOURCE_LENGTH = 2048
DEFAULT_GR_MAX_SKILLS = 4
DEFAULT_GR_NUM_BEAMS = 50
GR_SOURCE_TEXT_POLICY = "skillrouter_query_1500_chars_with_instruction_v1"
SEMANTIC_ID_MODE_DEDUP = "rq_codes_with_dedup"
SEMANTIC_ID_MODE_LEARNED = "learned_codes"
SEMANTIC_ID_MODES = (SEMANTIC_ID_MODE_DEDUP, SEMANTIC_ID_MODE_LEARNED)
DEFAULT_RIPOR_ALPHA_BETA = 0.5
DEFAULT_PAG_SET_TOP_N = 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_payload_sha256(payload: Any) -> str:
    """Hash JSON-compatible provenance metadata independently of formatting."""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 for a checkpoint/artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_loss_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def init_wandb_run(args: argparse.Namespace, config: dict[str, Any]):
    project = getattr(args, "wandb_project", None)
    if not project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required when --wandb-project is set; install wandb or omit the flag") from exc
    tags_text = str(getattr(args, "wandb_tags", "") or "")
    tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
    run = wandb.init(
        project=project,
        entity=getattr(args, "wandb_entity", None) or None,
        name=getattr(args, "wandb_run_name", None) or None,
        group=getattr(args, "wandb_group", None) or None,
        tags=tags or None,
        config=config,
        dir=str(args.output_dir),
        mode=getattr(args, "wandb_mode", "online"),
    )
    run.define_metric("query/step")
    run.define_metric("query/*", step_metric="query/step")
    run.define_metric("catalog/step")
    run.define_metric("catalog/*", step_metric="catalog/step")
    return run


def log_training_event(wandb_run, row: dict[str, Any]) -> None:
    if wandb_run is None:
        return
    stage = str(row.get("stage") or "train")
    payload: dict[str, Any] = {
        f"{stage}/step": row.get("step"),
        f"{stage}/epoch": row.get("epoch"),
    }
    if "train_loss" in row:
        payload[f"{stage}/train_loss"] = row["train_loss"]
    if "dev_loss" in row:
        payload[f"{stage}/dev_loss"] = row["dev_loss"]
    if "lr" in row:
        payload["lr"] = row["lr"]
    wandb_run.log(payload)


def save_structured_decoder_checkpoint(
    checkpoint_dir: Path,
    model,
    tokenizer,
    base_model: str,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    seed: int,
    training_formulation: str,
    extra: dict[str, Any] | None = None,
    optimizer=None,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> None:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model": base_model,
        "codebook_size": codebook_size,
        "depth": depth,
        "dedup_size": dedup_size,
        "semantic_id_mode": normalize_semantic_id_mode(semantic_id_mode),
        "pag_set_vocab_size": int(getattr(model, "pag_set_vocab_size", 0) or 0),
        "seed": seed,
        "training_formulation": training_formulation,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_dir / "structured_decoder.pt")
    tokenizer.save_pretrained(str(checkpoint_dir))


def load_structured_decoder_checkpoint(
    model,
    checkpoint_dir: Path,
    device,
    optimizer=None,
    lr: float | None = None,
    semantic_id_mode: str | None = None,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_dir / "structured_decoder.pt", map_location=device)
    if semantic_id_mode is not None:
        expected_mode = normalize_semantic_id_mode(semantic_id_mode)
        checkpoint_mode = normalize_semantic_id_mode(checkpoint.get("semantic_id_mode"))
        if checkpoint_mode != expected_mode:
            raise ValueError(
                "semantic id mode does not match the decoder checkpoint: "
                f"checkpoint={checkpoint_mode}, requested={expected_mode}"
            )
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
    optional_missing_prefixes = ("ripor_doc_embeddings.", "pag_set_head.")
    unexpected_required = [key for key in unexpected_keys if not key.startswith(optional_missing_prefixes)]
    missing_required = [key for key in missing_keys if not key.startswith(optional_missing_prefixes)]
    if unexpected_required or missing_required:
        raise ValueError(
            "structured decoder checkpoint parameter mismatch: "
            f"missing={missing_required[:5]}, unexpected={unexpected_required[:5]}"
        )
    optimizer_state_loaded = False
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_state_loaded = True
        except (RuntimeError, ValueError) as exc:
            # Older checkpoints predate the RIPOR/PAG parameter groups.  The
            # model weights remain a valid initialization, but their optimizer
            # state cannot be mapped onto the expanded parameter set.
            if not any(key.startswith(optional_missing_prefixes) for key in missing_keys):
                raise
            logging.warning("Skipping incompatible optimizer state from %s: %s", checkpoint_dir, exc)
        if lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr
    checkpoint["optimizer_state_loaded"] = optimizer_state_loaded
    return checkpoint


def skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "")


def format_skillrouter_query(query: str, max_chars: int | None = SKILLROUTER_QUERY_MAX_CHARS) -> str:
    raw = str(query or "")
    if max_chars is not None and max_chars > 0:
        raw = raw[:max_chars]
    return f"{SKILLROUTER_QUERY_INSTRUCTION}{raw}"


def format_skillrouter_skill(skill: dict[str, Any], desc_max: int = 300, body_max: int = 2500) -> str:
    name = str(skill.get("skill_name") or skill.get("name") or skill_id(skill))
    desc = str(skill.get("description") or skill.get("task_description") or "")[:desc_max]
    body = str(skill.get("body") or "")[:body_max]
    return f"{name} | {desc} | {body}"


def normalize_semantic_id_mode(value: str | None) -> str:
    mode = str(value or SEMANTIC_ID_MODE_DEDUP)
    if mode not in SEMANTIC_ID_MODES:
        raise ValueError(f"unknown semantic id mode: {mode}; expected one of {SEMANTIC_ID_MODES}")
    return mode


def normalized_skill_document_text(skill: dict[str, Any]) -> str:
    text = str(skill.get("skillrouter_text") or format_skillrouter_skill(skill))
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_exact_skill_documents(
    skills: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Collapse normalized-identical retrieval documents into deterministic alias groups."""
    by_id: dict[str, dict[str, Any]] = {}
    by_document: dict[str, list[str]] = {}
    document_hashes: dict[str, str] = {}
    for row_index, skill in enumerate(skills):
        sid = skill_id(skill)
        if not sid:
            raise ValueError(f"skill row {row_index} has no skill_id")
        if sid in by_id:
            raise ValueError(f"duplicate skill_id in skill catalog: {sid}")
        document = normalized_skill_document_text(skill)
        if not document:
            raise ValueError(f"skill document is empty after normalization: {sid}")
        by_id[sid] = skill
        by_document.setdefault(document, []).append(sid)
        document_hashes[sid] = hashlib.sha256(document.encode("utf-8")).hexdigest()

    alias_to_canonical: dict[str, str] = {}
    canonical_rows: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    duplicate_groups = 0
    aliased_skills = 0
    max_alias_count = 0
    for document, member_ids_unsorted in by_document.items():
        member_ids = sorted(member_ids_unsorted)
        canonical_id = member_ids[0]
        max_alias_count = max(max_alias_count, len(member_ids))
        if len(member_ids) > 1:
            duplicate_groups += 1
            aliased_skills += len(member_ids) - 1
        for sid in member_ids:
            alias_to_canonical[sid] = canonical_id
            alias_rows.append(
                {
                    "skill_id": sid,
                    "canonical_skill_id": canonical_id,
                    "is_canonical": sid == canonical_id,
                    "document_sha256": document_hashes[sid],
                }
            )
        canonical_row = dict(by_id[canonical_id])
        canonical_row.update(
            {
                "canonical_skill_id": canonical_id,
                "alias_skill_ids": member_ids,
                "alias_count": len(member_ids),
                "canonical_document_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
            }
        )
        canonical_rows.append(canonical_row)

    canonical_rows.sort(key=skill_id)
    alias_rows.sort(key=lambda row: str(row["skill_id"]))
    manifest = {
        "source_skills": len(skills),
        "canonical_skills": len(canonical_rows),
        "duplicate_document_groups": duplicate_groups,
        "aliased_skills": aliased_skills,
        "max_alias_count": max_alias_count,
        "canonical_selection": "lexicographically smallest skill_id",
        "document_equivalence": "whitespace-normalized skillrouter_text",
    }
    return canonical_rows, alias_rows, alias_to_canonical, manifest


def _remap_alias_id_list(
    values: Any,
    alias_to_canonical: dict[str, str],
    *,
    field_name: str,
    allow_unknown: bool,
) -> tuple[list[str], list[str]]:
    mapped: list[str] = []
    unknown: list[str] = []
    for raw_sid in values or []:
        sid = str(raw_sid)
        canonical_id = alias_to_canonical.get(sid)
        if canonical_id is None:
            unknown.append(sid)
            if allow_unknown:
                canonical_id = sid
            else:
                continue
        mapped.append(canonical_id)
    if unknown and not allow_unknown:
        raise KeyError(f"{field_name} contains unknown skill ids; first={unknown[:5]}")
    return _unique_preserving_order(mapped), unknown


def remap_row_skill_aliases(
    row: dict[str, Any],
    alias_to_canonical: dict[str, str],
    *,
    allow_unknown: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    remapped = dict(row)
    changed_pairs: dict[str, str] = {}
    unknown_ids: list[str] = []
    removed_duplicate_mentions = 0

    for field_name in ("selected_skill_ids", "source_selected_skill_ids", "gold_skill_ids"):
        if field_name not in row:
            continue
        original = [str(sid) for sid in (row.get(field_name) or [])]
        mapped, unknown = _remap_alias_id_list(
            original,
            alias_to_canonical,
            field_name=field_name,
            allow_unknown=allow_unknown,
        )
        remapped[field_name] = mapped
        removed_duplicate_mentions += len(original) - len(mapped)
        unknown_ids.extend(unknown)
        for sid in original:
            canonical_id = alias_to_canonical.get(sid, sid)
            if canonical_id != sid:
                changed_pairs[sid] = canonical_id

    for field_name in ("positive_skill_id", "skill_id"):
        if field_name not in row or row.get(field_name) in (None, ""):
            continue
        sid = str(row[field_name])
        canonical_id = alias_to_canonical.get(sid)
        if canonical_id is None:
            unknown_ids.append(sid)
            if not allow_unknown:
                raise KeyError(f"{field_name} contains unknown skill id: {sid}")
            canonical_id = sid
        remapped[field_name] = canonical_id
        if canonical_id != sid:
            changed_pairs[sid] = canonical_id

    generated = row.get("generated")
    if isinstance(generated, dict):
        generated_out = dict(generated)
        if "selected_skill_ids" in generated:
            original = [str(sid) for sid in (generated.get("selected_skill_ids") or [])]
            mapped, unknown = _remap_alias_id_list(
                original,
                alias_to_canonical,
                field_name="generated.selected_skill_ids",
                allow_unknown=allow_unknown,
            )
            generated_out["selected_skill_ids"] = mapped
            removed_duplicate_mentions += len(original) - len(mapped)
            unknown_ids.extend(unknown)
            for sid in original:
                canonical_id = alias_to_canonical.get(sid, sid)
                if canonical_id != sid:
                    changed_pairs[sid] = canonical_id
            if "target_skill_count" in generated_out:
                generated_out["target_skill_count"] = len(mapped)
        remapped["generated"] = generated_out

    if "selected_skill_ids" in remapped and "target_skill_count" in remapped:
        old_count = remapped.get("target_skill_count")
        new_count = len(remapped["selected_skill_ids"])
        if old_count != new_count:
            remapped.setdefault("pre_alias_target_skill_count", old_count)
            remapped["target_skill_count"] = new_count
    if changed_pairs:
        remapped["canonicalized_skill_aliases"] = dict(sorted(changed_pairs.items()))
    audit = {
        "changed": bool(changed_pairs or removed_duplicate_mentions),
        "changed_aliases": len(changed_pairs),
        "removed_duplicate_mentions": removed_duplicate_mentions,
        "unknown_skill_ids": sorted(set(unknown_ids)),
    }
    return remapped, audit


def selected_skill_ids(row: dict[str, Any]) -> list[str]:
    generated = row.get("generated") or {}
    return [str(sid) for sid in (generated.get("selected_skill_ids") or []) if sid]


def prepared_selected_skill_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("selected_skill_ids")
    if ids is None:
        return selected_skill_ids(row)
    return [str(sid) for sid in ids if sid]


def collect_selected_ids(rows: list[dict[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for row in rows:
        ids.update(selected_skill_ids(row))
    return sorted(ids)


def load_skill_lookup_for_ids(skills_path: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    with skills_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = skill_id(row)
            if sid in ids:
                lookup[sid] = row
                if len(lookup) == len(ids):
                    break
    missing = sorted(ids - set(lookup))
    if missing:
        raise SystemExit(f"missing {len(missing)} selected skills in skill lookup; first={missing[:5]}")
    return lookup


def semantic_token(level: int, code: int, width: int) -> str:
    return f"<RQ{level}_{code:0{width}d}>"


def dedup_token(value: int, width: int) -> str:
    return f"<DUP_{value:0{width}d}>"


def deduplicate_code_tuples(code_tuples: dict[str, list[int]]) -> dict[str, list[int]]:
    """Append the previous-hit count for each code tuple, matching TIGER-style dedup columns."""
    seen: Counter[tuple[int, ...]] = Counter()
    out: dict[str, list[int]] = {}
    for sid in sorted(code_tuples):
        key = tuple(int(x) for x in code_tuples[sid])
        out[sid] = [*key, seen[key]]
        seen[key] += 1
    return out


def collect_query_skill_interactions(
    rows: list[dict[str, Any]],
    skill_to_index: dict[str, int],
) -> tuple[list[tuple[int, int]], Counter[str]]:
    interactions: list[tuple[int, int]] = []
    missing: Counter[str] = Counter()
    for query_index, row in enumerate(rows):
        seen_skill_indices: set[int] = set()
        for sid in prepared_selected_skill_ids(row):
            skill_index = skill_to_index.get(sid)
            if skill_index is None:
                missing[sid] += 1
                continue
            if skill_index not in seen_skill_indices:
                interactions.append((query_index, skill_index))
                seen_skill_indices.add(skill_index)
    return interactions, missing


def structured_sid_length(depth: int, semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP) -> int:
    mode = normalize_semantic_id_mode(semantic_id_mode)
    return int(depth) + (1 if mode == SEMANTIC_ID_MODE_DEDUP else 0)


def structured_sid_control_type(depth: int, semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP) -> int:
    return structured_sid_length(depth, semantic_id_mode)


def global_sid_token_id(
    level: int,
    value: int,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> int:
    mode = normalize_semantic_id_mode(semantic_id_mode)
    if mode == SEMANTIC_ID_MODE_LEARNED and dedup_size != 0:
        raise ValueError(f"learned_codes requires dedup_size=0, got {dedup_size}")
    if level < depth:
        return level * codebook_size + int(value)
    if mode == SEMANTIC_ID_MODE_DEDUP and level == depth:
        return depth * codebook_size + int(value)
    control_type = structured_sid_control_type(depth, mode)
    if level == control_type:
        return depth * codebook_size + dedup_size + int(value)
    if level == control_type + 1:
        return depth * codebook_size + dedup_size + 2
    raise ValueError(f"invalid structured SID level: {level}")


def structured_sid_vocab_size(codebook_size: int, depth: int, dedup_size: int) -> int:
    return depth * codebook_size + dedup_size + 3


def structured_sid_bos_id(
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> int:
    control_type = structured_sid_control_type(depth, semantic_id_mode)
    return global_sid_token_id(
        control_type + 1,
        0,
        codebook_size,
        depth,
        dedup_size,
        semantic_id_mode,
    )


def encode_structured_sid_target(
    skill_ids: list[str],
    semantic_ids: dict[str, list[int]],
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> dict[str, Any]:
    mode = normalize_semantic_id_mode(semantic_id_mode)
    target_token_ids: list[int] = []
    types: list[int] = []
    labels: list[int] = []
    sorted_skill_ids = sorted(skill_ids)
    for skill_index, sid in enumerate(sorted_skill_ids):
        if sid not in semantic_ids:
            raise KeyError(f"missing semantic id for selected skill: {sid}")
        semantic_id = [int(v) for v in semantic_ids[sid]]
        expected_length = structured_sid_length(depth, mode)
        if len(semantic_id) != expected_length:
            raise ValueError(
                f"semantic id length {len(semantic_id)} does not match schema length={expected_length}: {sid}"
            )
        code_values = semantic_id[:-1] if mode == SEMANTIC_ID_MODE_DEDUP else semantic_id
        for level, code in enumerate(code_values):
            target_token_ids.append(
                global_sid_token_id(level, code, codebook_size, depth, dedup_size, mode)
            )
            types.append(level)
            labels.append(code)
        if mode == SEMANTIC_ID_MODE_DEDUP:
            dedup_value = semantic_id[-1]
            target_token_ids.append(
                global_sid_token_id(depth, dedup_value, codebook_size, depth, dedup_size, mode)
            )
            types.append(depth)
            labels.append(dedup_value)
        control = 0 if skill_index < len(sorted_skill_ids) - 1 else 1
        control_type = structured_sid_control_type(depth, mode)
        target_token_ids.append(
            global_sid_token_id(control_type, control, codebook_size, depth, dedup_size, mode)
        )
        types.append(control_type)
        labels.append(control)
    bos_id = structured_sid_bos_id(codebook_size, depth, dedup_size, mode)
    return {
        "skill_ids": sorted_skill_ids,
        "target_token_ids": target_token_ids,
        "decoder_input_ids": [bos_id, *target_token_ids[:-1]],
        "types": types,
        "labels": labels,
        "bos_id": bos_id,
    }


def semantic_id_tokens(
    semantic_id: list[int],
    code_width: int,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> list[str]:
    mode = normalize_semantic_id_mode(semantic_id_mode)
    minimum_length = 2 if mode == SEMANTIC_ID_MODE_DEDUP else 1
    if len(semantic_id) < minimum_length:
        raise ValueError(f"semantic id is too short for mode={mode}: {semantic_id}")
    code_values = semantic_id[:-1] if mode == SEMANTIC_ID_MODE_DEDUP else semantic_id
    tokens = [semantic_token(level, int(code), code_width) for level, code in enumerate(code_values)]
    if mode == SEMANTIC_ID_MODE_DEDUP:
        tokens.append(dedup_token(int(semantic_id[-1]), code_width))
    return tokens


def native_semantic_id_text(
    semantic_id: list[int],
    code_width: int,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> str:
    """Represent a semantic id using only the base tokenizer vocabulary."""
    mode = normalize_semantic_id_mode(semantic_id_mode)
    minimum_length = 2 if mode == SEMANTIC_ID_MODE_DEDUP else 1
    if len(semantic_id) < minimum_length:
        raise ValueError(f"semantic id is too short for mode={mode}: {semantic_id}")
    parts: list[str] = []
    code_values = semantic_id[:-1] if mode == SEMANTIC_ID_MODE_DEDUP else semantic_id
    for level, code in enumerate(code_values):
        parts.extend(["code", str(level), f"{int(code):0{code_width}d}"])
    if mode == SEMANTIC_ID_MODE_DEDUP:
        parts.extend(["duplicate", f"{int(semantic_id[-1]):0{code_width}d}"])
    return " ".join(parts)


def build_decoder_target(
    skill_ids: list[str],
    semantic_ids: dict[str, list[int]],
    code_width: int = 4,
    target_mode: str = "special",
    separator_text: str | None = None,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> str:
    parts: list[str] = []
    for sid in sorted(skill_ids):
        if sid not in semantic_ids:
            raise KeyError(f"missing semantic id for selected skill: {sid}")
        if target_mode == "special":
            parts.append(" ".join(semantic_id_tokens(semantic_ids[sid], code_width, semantic_id_mode)))
        elif target_mode == "native":
            parts.append(native_semantic_id_text(semantic_ids[sid], code_width, semantic_id_mode))
        else:
            raise ValueError(f"unknown target mode: {target_mode}")
    sep = separator_text or (SKILL_SEP_TOKEN if target_mode == "special" else NATIVE_SKILL_SEP_TEXT)
    return f" {sep} ".join(parts)


def stratified_split_rows(
    rows: list[dict[str, Any]],
    dev_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_field: str = "target_skill_count",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(stratify_field))
        by_key.setdefault(key, []).append(row)

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}
    for key in sorted(by_key):
        bucket = list(by_key[key])
        rng.shuffle(bucket)
        n = len(bucket)
        test_n = int(round(n * test_ratio))
        dev_n = int(round(n * dev_ratio))
        if test_ratio > 0 and test_n == 0 and n > 0:
            test_n = 1
        if dev_ratio > 0 and dev_n == 0 and n - test_n > 0:
            dev_n = 1
        if test_n + dev_n >= n:
            raise ValueError(f"split ratios leave no train rows for stratum {key}: n={n}")
        test_part = bucket[:test_n]
        dev_part = bucket[test_n : test_n + dev_n]
        train_part = bucket[test_n + dev_n :]
        test_rows.extend(test_part)
        dev_rows.extend(dev_part)
        train_rows.extend(train_part)
        split_counts[key] = {"train": len(train_part), "dev": len(dev_part), "test": len(test_part), "total": n}

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    rng.shuffle(test_rows)
    manifest = {
        "rows": len(rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "dev_ratio": dev_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "stratify_field": stratify_field,
        "split_counts": split_counts,
    }
    return train_rows, dev_rows, test_rows, manifest


def build_prefix_trie(sequences: list[list[int]]) -> dict[str, Any]:
    trie: dict[str, Any] = {}
    for index, sequence in enumerate(sequences):
        node = trie
        for token_id in sequence:
            node = node.setdefault(str(int(token_id)), {})
        node.setdefault(TRIE_TERMINAL_KEY, []).append(index)
    return trie


def count_trie_nodes(node: dict[str, Any]) -> int:
    total = 1
    for key, child in node.items():
        if key != TRIE_TERMINAL_KEY:
            total += count_trie_nodes(child)
    return total


def _strip_decoder_start_tokens(generated_ids: list[int], decoder_start_token_id: int | None = None) -> list[int]:
    ids = [int(token_id) for token_id in generated_ids]
    if decoder_start_token_id is not None and ids and ids[0] == int(decoder_start_token_id):
        ids = ids[1:]
    return ids


def parse_skill_token_sequences(
    generated_ids: list[int],
    trie: dict[str, Any],
    sep_id: int,
    eos_id: int,
    decoder_start_token_id: int | None = None,
    finalize_open_terminal: bool = False,
) -> tuple[list[tuple[int, ...]], bool]:
    """Parse generated token ids into complete single-skill semantic-id token sequences."""
    ids = _strip_decoder_start_tokens(generated_ids, decoder_start_token_id)
    node = trie
    current: list[int] = []
    completed: list[tuple[int, ...]] = []
    invalid = False
    for token_id in ids:
        if token_id == eos_id:
            if current:
                if TRIE_TERMINAL_KEY in node:
                    completed.append(tuple(current))
                else:
                    invalid = True
            return completed, invalid
        if token_id == sep_id:
            if current and TRIE_TERMINAL_KEY in node:
                completed.append(tuple(current))
                current = []
                node = trie
                continue
            invalid = True
            break
        key = str(token_id)
        if key not in node:
            invalid = True
            break
        current.append(token_id)
        node = node[key]
    if finalize_open_terminal and current and TRIE_TERMINAL_KEY in node:
        completed.append(tuple(current))
    return completed, invalid


def _trie_has_unseen_terminal(node: dict[str, Any], prefix: list[int], completed: set[tuple[int, ...]]) -> bool:
    if TRIE_TERMINAL_KEY in node and tuple(prefix) not in completed:
        return True
    return any(
        _trie_has_unseen_terminal(child, [*prefix, int(key)], completed)
        for key, child in node.items()
        if key != TRIE_TERMINAL_KEY
    )


def allowed_next_tokens_for_skill_trie(
    generated_ids: list[int],
    trie: dict[str, Any],
    sep_id: int,
    eos_id: int,
    max_skills: int,
    decoder_start_token_id: int | None = None,
) -> list[int]:
    """Allowed next tokens for one-skill semantic-id trie reused across a multi-skill sequence."""
    ids = _strip_decoder_start_tokens(generated_ids, decoder_start_token_id)
    node = trie
    current: list[int] = []
    completed: list[tuple[int, ...]] = []
    for token_id in ids:
        if token_id == eos_id:
            return [int(eos_id)]
        if token_id == sep_id:
            if current and TRIE_TERMINAL_KEY in node:
                completed.append(tuple(current))
                current = []
                node = trie
                continue
            return [int(eos_id)]
        key = str(token_id)
        if key not in node:
            return [int(eos_id)]
        current.append(token_id)
        node = node[key]

    if current and TRIE_TERMINAL_KEY in node:
        allowed = [int(eos_id)]
        if len(completed) + 1 < max_skills:
            allowed.append(int(sep_id))
        return allowed
    if not current and len(completed) >= max_skills:
        return [int(eos_id)]

    completed_set = set(completed)
    allowed = sorted(
        int(key)
        for key, child in node.items()
        if key != TRIE_TERMINAL_KEY and _trie_has_unseen_terminal(child, [*current, int(key)], completed_set)
    )
    return allowed or [int(eos_id)]


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def compute_retrieval_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError(f"rows/predictions length mismatch: {len(rows)} != {len(predictions)}")

    grouped: dict[str, list[dict[str, float]]] = {"overall": []}
    for row, pred in zip(rows, predictions):
        gold_ids = set(str(sid) for sid in row.get("selected_skill_ids", []))
        pred_ids = _unique_preserving_order([str(sid) for sid in pred.get("predicted_skill_ids", [])])
        pred_set = set(pred_ids)
        hit_count = len(gold_ids & pred_set)
        gold_count = len(gold_ids)
        pred_count = len(pred_set)
        item = {
            "hit_count": float(hit_count),
            "gold_count": float(gold_count),
            "pred_count": float(pred_count),
            "recall": float(hit_count / gold_count) if gold_count else 0.0,
            "precision": float(hit_count / pred_count) if pred_count else 0.0,
            "hit_rate": 1.0 if hit_count > 0 else 0.0,
            "exact_match": 1.0 if gold_ids == pred_set else 0.0,
            "count_match": 1.0 if gold_count == pred_count else 0.0,
            "invalid": 1.0 if pred.get("invalid") else 0.0,
            "duplicate_predictions": float(max(0, len(pred.get("predicted_skill_ids", [])) - pred_count)),
        }
        grouped["overall"].append(item)
        grouped.setdefault(str(row.get("target_skill_count", gold_count)), []).append(item)

    def summarize(items: list[dict[str, float]]) -> dict[str, Any]:
        rows_n = len(items)
        total_hits = int(sum(item["hit_count"] for item in items))
        total_gold = int(sum(item["gold_count"] for item in items))
        total_pred = int(sum(item["pred_count"] for item in items))
        return {
            "rows": rows_n,
            "total_hits": total_hits,
            "total_gold": total_gold,
            "total_pred": total_pred,
            "micro_recall": float(total_hits / total_gold) if total_gold else 0.0,
            "micro_precision": float(total_hits / total_pred) if total_pred else 0.0,
            "macro_recall": float(sum(item["recall"] for item in items) / rows_n) if rows_n else 0.0,
            "macro_precision": float(sum(item["precision"] for item in items) / rows_n) if rows_n else 0.0,
            "hit_rate": float(sum(item["hit_rate"] for item in items) / rows_n) if rows_n else 0.0,
            "exact_match_rate": float(sum(item["exact_match"] for item in items) / rows_n) if rows_n else 0.0,
            "count_accuracy": float(sum(item["count_match"] for item in items) / rows_n) if rows_n else 0.0,
            "invalid_rate": float(sum(item["invalid"] for item in items) / rows_n) if rows_n else 0.0,
            "duplicate_predictions": int(sum(item["duplicate_predictions"] for item in items)),
        }

    return {
        "overall": summarize(grouped["overall"]),
        "by_target_skill_count": {
            key: summarize(grouped[key])
            for key in sorted(grouped)
            if key != "overall"
        },
    }


def compute_at_k_metrics(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    k_values: list[int],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError(f"rows/predictions length mismatch: {len(rows)} != {len(predictions)}")
    if not k_values:
        raise ValueError("k_values must not be empty")
    k_values = sorted({int(k) for k in k_values})
    if any(k <= 0 for k in k_values):
        raise ValueError(f"k_values must be positive: {k_values}")

    grouped: dict[str, list[dict[str, Any]]] = {"overall": []}
    for row, pred in zip(rows, predictions):
        gold_ids = set(str(sid) for sid in row.get("selected_skill_ids", []))
        ranked_ids = _unique_preserving_order(
            [str(sid) for sid in pred.get("ranked_skill_ids") or pred.get("predicted_skill_ids", [])]
        )
        item = {
            "gold_ids": gold_ids,
            "ranked_ids": ranked_ids,
            "invalid": bool(pred.get("invalid")),
        }
        grouped["overall"].append(item)
        grouped.setdefault(str(row.get("target_skill_count", len(gold_ids))), []).append(item)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        rows_n = len(items)
        total_gold = sum(len(item["gold_ids"]) for item in items)
        out: dict[str, Any] = {"rows": rows_n, "total_gold": total_gold}
        for k in k_values:
            total_hits = 0
            hit_rows = 0
            full_coverage_rows = 0
            recall_sum = 0.0
            for item in items:
                gold_ids = item["gold_ids"]
                topk = set(item["ranked_ids"][:k])
                hit_count = len(gold_ids & topk)
                total_hits += hit_count
                hit_rows += int(hit_count > 0)
                full_coverage_rows += int(bool(gold_ids) and gold_ids.issubset(topk))
                recall_sum += hit_count / len(gold_ids) if gold_ids else 0.0
            macro_recall = float(recall_sum / rows_n) if rows_n else 0.0
            micro_recall = float(total_hits / total_gold) if total_gold else 0.0
            out[f"@{k}"] = {
                "total_hits": int(total_hits),
                "hit": float(hit_rows / rows_n) if rows_n else 0.0,
                "recall": macro_recall,
                "macro_recall": macro_recall,
                "micro_recall": micro_recall,
                "fc": float(full_coverage_rows / rows_n) if rows_n else 0.0,
            }
        return out

    return {
        "k_values": k_values,
        "recall_aggregation": "macro",
        "overall": summarize(grouped["overall"]),
        "by_target_skill_count": {
            key: summarize(grouped[key])
            for key in sorted(grouped)
            if key != "overall"
        },
    }


def parse_k_values(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def command_rescore_ranked(args: argparse.Namespace) -> None:
    predictions = load_jsonl(args.predictions)
    rows: list[dict[str, Any]] = []
    filtered_predictions: list[dict[str, Any]] = []
    for prediction in predictions:
        gold_ids = prediction.get("gold_skill_ids")
        if gold_ids is None:
            gold_ids = prediction.get("selected_skill_ids")
        if gold_ids is None:
            raise ValueError(f"prediction is missing gold_skill_ids: {prediction.get('index')}")
        gold_ids = [str(sid) for sid in gold_ids]
        if args.subset == "single" and len(gold_ids) != 1:
            continue
        if args.subset == "multi" and len(gold_ids) <= 1:
            continue
        rows.append(
            {
                "selected_skill_ids": gold_ids,
                "target_skill_count": len(gold_ids),
            }
        )
        filtered_predictions.append(prediction)

    metrics = compute_at_k_metrics(rows, filtered_predictions, parse_k_values(args.k_values))
    output = args.output or args.predictions.parent / "ranked_metrics.json"
    write_json(output, metrics)
    print(json.dumps({"output": str(output), "subset": args.subset, **metrics["overall"]}, sort_keys=True))


def row_query_text(row: dict[str, Any]) -> str:
    return str(row.get("query") or row.get("decoder_input") or row.get("query_text") or "")


def decoder_source_text(row: dict[str, Any]) -> str:
    if row.get("decoder_input"):
        return str(row["decoder_input"])
    if row.get("query"):
        raw_query = str(row["query"])
        if raw_query.startswith(SKILLROUTER_QUERY_INSTRUCTION):
            query = raw_query[len(SKILLROUTER_QUERY_INSTRUCTION) :]
            return format_skillrouter_query(query)
        return format_skillrouter_query(raw_query)
    return str(row.get("query_text") or "")


def _nearest_rank_percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return int(ordered[index])


def summarize_source_token_lengths(
    rows: list[dict[str, Any]],
    tokenizer,
    max_source_length: int,
    batch_size: int = 256,
) -> dict[str, Any]:
    if max_source_length <= 0:
        raise ValueError(f"max_source_length must be positive, got {max_source_length}")
    lengths: list[int] = []
    for start in range(0, len(rows), batch_size):
        texts = [decoder_source_text(row) for row in rows[start : start + batch_size]]
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    over_limit = sum(length > max_source_length for length in lengths)
    return {
        "rows": len(lengths),
        "max_source_length": int(max_source_length),
        "min_tokens": min(lengths) if lengths else None,
        "mean_tokens": round(sum(lengths) / len(lengths), 3) if lengths else None,
        "p50_tokens": _nearest_rank_percentile(lengths, 0.50),
        "p95_tokens": _nearest_rank_percentile(lengths, 0.95),
        "p99_tokens": _nearest_rank_percentile(lengths, 0.99),
        "max_tokens": max(lengths) if lengths else None,
        "over_limit_rows": over_limit,
        "over_limit_fraction": round(over_limit / len(lengths), 8) if lengths else 0.0,
    }


def audit_source_token_lengths(
    label: str,
    rows: list[dict[str, Any]],
    tokenizer,
    max_source_length: int,
    allow_source_truncation: bool,
) -> dict[str, Any]:
    summary = summarize_source_token_lengths(rows, tokenizer, max_source_length)
    print(json.dumps({"source_length_audit": label, **summary}, sort_keys=True), flush=True)
    if summary["over_limit_rows"] and not allow_source_truncation:
        raise ValueError(
            f"{label}: {summary['over_limit_rows']}/{summary['rows']} inputs exceed "
            f"max_source_length={max_source_length} (max={summary['max_tokens']}); "
            "increase --max-source-length or explicitly pass --allow-source-truncation"
        )
    return summary


def load_candidate_skill_texts(skills_jsonl: Path) -> list[dict[str, str]]:
    rows = []
    for row in load_jsonl(skills_jsonl):
        sid = skill_id(row)
        text = str(row.get("skillrouter_text") or format_skillrouter_skill(row))
        rows.append({"skill_id": sid, "text": text})
    if not rows:
        raise ValueError(f"no candidate skills loaded from {skills_jsonl}")
    return rows


def load_pag_set_token_ids(path: Path | None) -> tuple[dict[str, list[int]], int]:
    if path is None:
        return {}, 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "skill_token_ids" in payload:
        values = payload["skill_token_ids"]
        vocab_size = int(payload.get("vocab_size", 0) or 0)
        expected_m = int(payload.get("max_tokens", 0) or 0)
    else:
        raise ValueError(f"PAG set id manifest must contain skill_token_ids: {path}")
    if not isinstance(values, dict):
        raise ValueError(f"PAG set id manifest must be a mapping: {path}")
    result: dict[str, list[int]] = {}
    for sid, token_ids in values.items():
        normalized = sorted({int(token_id) for token_id in token_ids})
        if not normalized:
            raise ValueError(f"PAG set id manifest contains an empty set: {sid}")
        if expected_m > 0 and len(normalized) != expected_m:
            raise ValueError(
                f"PAG set id manifest must contain exactly m={expected_m} unique tokens; "
                f"skill={sid} has {len(normalized)}"
            )
        result[str(sid)] = normalized
        if normalized:
            vocab_size = max(vocab_size, max(normalized) + 1)
    if vocab_size <= 0:
        raise ValueError(f"PAG set id manifest has no vocabulary size: {path}")
    return result, vocab_size


def expand_single_sid_query_rows(
    rows: list[dict[str, Any]],
    semantic_ids: dict[str, list[int]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        gold_ids = sorted(_unique_preserving_order([str(sid) for sid in row.get("selected_skill_ids", [])]))
        for sid in gold_ids:
            if sid not in semantic_ids:
                raise KeyError(f"missing semantic id for selected skill: {sid}")
            expanded_row = dict(row)
            expanded_row["selected_skill_ids"] = [sid]
            expanded_row["positive_skill_id"] = sid
            expanded_row["source_selected_skill_ids"] = gold_ids
            expanded_row["source_target_skill_count"] = row.get("target_skill_count", len(gold_ids))
            expanded.append(expanded_row)
    return expanded


def split_and_expand_single_sid_query_rows(
    rows: list[dict[str, Any]],
    semantic_ids: dict[str, list[int]],
    dev_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(source_rows)
    dev_size = max(1, math.ceil(len(source_rows) * dev_ratio)) if dev_ratio > 0 else 0
    dev_source_rows = source_rows[:dev_size]
    train_source_rows = source_rows[dev_size:]
    return (
        expand_single_sid_query_rows(train_source_rows, semantic_ids),
        expand_single_sid_query_rows(dev_source_rows, semantic_ids),
    )


def build_single_sid_catalog_rows(
    skills: list[dict[str, Any]],
    semantic_ids: dict[str, list[int]],
    input_prefix: str = "skill document: ",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for skill in sorted(skills, key=lambda item: skill_id(item)):
        sid = skill_id(skill)
        if sid not in semantic_ids:
            continue
        text = str(skill.get("skillrouter_text") or format_skillrouter_skill(skill))
        rows.append(
            {
                "index": f"catalog:{sid}",
                "source": "catalog",
                "target_skill_count": 1,
                "decoder_input": f"{input_prefix}{text}",
                "query": text,
                "selected_skill_ids": [sid],
                "positive_skill_id": sid,
            }
        )
    if not rows:
        raise ValueError("no catalog rows matched semantic ids")
    return rows


def retrieval_row_keys(row: dict[str, Any]) -> list[str]:
    """Stable keys shared by training rows and retrieval prediction files."""
    keys: list[str] = []
    for field in ("source_index", "index", "qid", "query_id"):
        value = row.get(field)
        if value is not None and str(value) != "":
            keys.append(f"{field}:{value}")
    query = str(row.get("query") or row.get("query_text") or row.get("decoder_input") or "")
    if query:
        keys.append(f"query_sha256:{hashlib.sha256(query.encode('utf-8')).hexdigest()}")
    if not keys:
        raise ValueError("row needs source_index, index, or non-empty query to join retrieval candidates")
    return keys


def _unique_known_skill_ids(values: list[Any], semantic_ids: dict[str, list[int]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        sid = str(value)
        if sid in semantic_ids and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def ripor_positive_skill_ids(row: dict[str, Any], semantic_ids: dict[str, list[int]]) -> list[str]:
    """Return every true skill for a source query, including single-SID siblings."""
    source_ids = row.get("source_selected_skill_ids")
    values = source_ids if source_ids is not None else row.get("selected_skill_ids", [])
    positives = _unique_known_skill_ids(list(values or []), semantic_ids)
    if not positives:
        raise ValueError(f"RIPOR row has no known positive skill ids: {row.get('index')}")
    return positives


def ripor_prefix_alphas(depth: int, beta: float = DEFAULT_RIPOR_ALPHA_BETA) -> list[float]:
    """Return the normalized concave prefix weights from RIPOR Eq. (1).

    ``alpha_L`` is exactly one, so the full DocID margin remains calibrated to
    the teacher margin while early prefixes receive a larger relative target.
    """
    if depth <= 0:
        raise ValueError(f"RIPOR depth must be positive, got {depth}")
    if not 0.0 < beta < 1.0:
        raise ValueError(f"RIPOR alpha beta must be in (0, 1), got {beta}")
    normalizer = 1.0 - beta**depth
    return [(1.0 - beta**length) / normalizer for length in range(1, depth + 1)]


def build_pag_set_token_ids(
    skills: list[dict[str, Any]],
    tokenizer,
    *,
    max_tokens: int = 16,
) -> dict[str, list[int]]:
    """Build lexical set identifiers used by the PAG set branch.

    The official PAG branch represents each document by a sparse lexical set.
    This project has no external SPLADE checkpoint in the decoder pipeline, so
    the manifest is intentionally explicit and reproducible: it keeps the
    first ``max_tokens`` unique non-special tokenizer ids from the frozen skill
    document text.  A learned PAG sparse head is trained against this manifest;
    no dense/BM25 predictions enter the decoding score.
    """
    if max_tokens <= 0:
        raise ValueError(f"PAG set max_tokens must be positive, got {max_tokens}")
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    result: dict[str, list[int]] = {}
    for skill in skills:
        sid = skill_id(skill)
        if not sid:
            raise ValueError("PAG set manifest contains a skill without skill_id")
        text = str(skill.get("skillrouter_text") or format_skillrouter_skill(skill))
        encoded = tokenizer(text, add_special_tokens=True, padding=False, truncation=False)
        values: list[int] = []
        seen: set[int] = set()
        for raw_value in encoded["input_ids"]:
            value = int(raw_value)
            if value in special_ids or value in seen:
                continue
            seen.add(value)
            values.append(value)
            if len(values) >= max_tokens:
                break
        if not values:
            raise ValueError(f"PAG set manifest has no lexical ids for skill: {sid}")
        result[sid] = values
    return result


def load_ranked_skill_candidates_with_scores(
    path: Path | None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """Load ranked candidates and their teacher relevance scores.

    Dense predictions usually provide ``ranked_scores`` while structured GR
    marginal decoding uses ``ranked_skill_scores``.  Candidate IDs and teacher
    scores are deliberately kept separate: an ordered list without numeric
    scores is a candidate pool only, never an observed teacher target.  Weak
    reciprocal-rank/anchor targets are created later only when the caller
    explicitly enables ``allow_weak_teacher``.
    """
    if path is None:
        return {}, {}
    candidate_map: dict[str, list[str]] = {}
    teacher_score_map: dict[str, dict[str, float]] = {}
    for row in load_jsonl(path):
        values: list[Any] = []
        for field in (
            "ranked_skill_ids",
            "predicted_skill_ids",
            "negative_skill_ids",
            "hard_negative_skill_ids",
            "ripor_negative_skill_ids",
            "docids",
            "candidate_skill_ids",
            "positive_skill_ids",
        ):
            values.extend(list(row.get(field) or []))
        if not values:
            continue
        ordered = _unique_preserving_order([str(value) for value in values])
        score_payload = (
            row.get("ranked_scores")
            or row.get("ranked_skill_scores")
            or row.get("teacher_scores")
            or row.get("skill_scores")
            or row.get("scores")
            or []
        )
        score_by_skill = (
            {str(sid): value for sid, value in score_payload.items()}
            if isinstance(score_payload, dict)
            else None
        )
        raw_scores = list(score_payload) if score_by_skill is None else []
        ranked_ids = [
            str(value)
            for value in (
                row.get("ranked_skill_ids")
                or row.get("docids")
                or row.get("candidate_skill_ids")
                or row.get("positive_skill_ids")
                or []
            )
        ]
        row_scores: dict[str, float] = {}
        for rank, sid in enumerate(ranked_ids):
            if score_by_skill is not None:
                if sid not in score_by_skill:
                    continue
                value = score_by_skill[sid]
            elif rank < len(raw_scores):
                value = raw_scores[rank]
            else:
                # The ranked ID is still a valid candidate, but it does not
                # have an observed teacher score.
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                row_scores[sid] = score
        # Some preprocessing stages store scores by skill id without a
        # corresponding ranked list.  Preserve those values so strict RIPOR/
        # PAG MarginMSE training can consume the same manifest.
        if isinstance(score_payload, dict):
            for sid, value in score_payload.items():
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(score):
                    row_scores[str(sid)] = score
        for key in retrieval_row_keys(row):
            candidate_map.setdefault(key, ordered)
            teacher_score_map.setdefault(key, {}).update(row_scores)
    return candidate_map, teacher_score_map


def build_teacher_rank_examples(
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    candidate_map: dict[str, list[str]],
    teacher_score_map: dict[str, dict[str, float]],
    *,
    max_negatives: int,
    allow_weak_teacher: bool,
) -> list[dict[str, Any]]:
    """Create query/positive/negative examples for the official MarginMSE stages.

    RIPOR and PAG both train a frozen-teacher ranking objective before their
    identifier stages.  The project rows provide positives while the optional
    ranked-candidate file supplies hard negatives and calibrated scores.  A
    missing score is fatal in the reproduction path; the binary/anchor fallback
    is available only through an explicit weak-teacher ablation flag.
    """
    known_skills = {
        skill_id(skill): str(skill.get("skillrouter_text") or format_skillrouter_skill(skill))
        for skill in skills
        if skill_id(skill)
    }
    if not known_skills:
        raise ValueError("no skill documents are available for ranking-stage training")
    if max_negatives <= 0:
        raise ValueError(f"max_negatives must be positive, got {max_negatives}")
    examples: list[dict[str, Any]] = []
    for row in rows:
        query = decoder_source_text(row)
        source_positive_ids = row.get("source_selected_skill_ids")
        positive_values = source_positive_ids if source_positive_ids is not None else row.get("selected_skill_ids", [])
        positives = [
            sid for sid in _unique_preserving_order([str(value) for value in positive_values or []])
            if sid in known_skills
        ]
        if not query or not positives:
            continue
        positive_set = set(positives)
        candidates = ranked_candidates_for_row(row, candidate_map)
        negatives = [sid for sid in candidates if sid in known_skills and sid not in positive_set]
        if len(negatives) < max_negatives:
            # Keep the deterministic random completion only for the explicitly
            # weak-teacher mode.  Strict reproduction must use mined negatives.
            if not allow_weak_teacher:
                raise ValueError(
                    "strict ranking-stage reproduction requires mined negatives for every row; "
                    f"missing row={row.get('index')}"
                )
            stable_seed = int.from_bytes(
                hashlib.sha256(str(retrieval_row_keys(row)[0]).encode("utf-8")).digest()[:8],
                byteorder="big",
            )
            pool = [sid for sid in sorted(known_skills) if sid not in positive_set and sid not in negatives]
            random.Random(stable_seed).shuffle(pool)
            negatives.extend(pool[: max_negatives - len(negatives)])
        if not negatives:
            raise ValueError(f"no negative skill is available for row={row.get('index')}")
        row_scores: dict[str, float] = {}
        for key in retrieval_row_keys(row):
            row_scores.update(teacher_score_map.get(key, {}))
        known_scores: list[float] = []
        for value in row_scores.values():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                known_scores.append(numeric)
        for positive_sid in positives:
            positive_score = row_scores.get(positive_sid)
            selected_negative = None
            negative_score = None
            for negative_sid in negatives[:max_negatives]:
                if negative_sid in row_scores:
                    selected_negative = negative_sid
                    negative_score = row_scores[negative_sid]
                    break
            if positive_score is None or negative_score is None:
                if not allow_weak_teacher:
                    missing = positive_sid if positive_score is None else selected_negative
                    raise ValueError(
                        "strict ranking-stage reproduction requires teacher scores for positive and negative "
                        f"skills; missing skill={missing}, row={row.get('index')}"
                    )
                positive_score = max(known_scores, default=0.0) + 1.0
                selected_negative = selected_negative or negatives[0]
                negative_score = min(known_scores, default=0.0) - 1.0
            examples.append(
                {
                    "query": query,
                    "positive_skill_id": positive_sid,
                    "positive_text": known_skills[positive_sid],
                    "negative_skill_id": selected_negative,
                    "negative_text": known_skills[selected_negative],
                    "teacher_positive_score": float(positive_score),
                    "teacher_negative_score": float(negative_score),
                }
            )
    if not examples:
        raise ValueError("no query/positive/negative examples were built for ranking-stage training")
    return examples


def build_pag_set_candidate_batch(
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    pag_set_token_ids: dict[str, list[int]],
    candidate_map: dict[str, list[str]],
    teacher_score_map: dict[str, dict[str, float]],
    *,
    max_negatives: int,
    allow_weak_teacher: bool,
    device,
) -> dict[str, Any]:
    """Build the lexical set candidates consumed by PAG MarginMSE."""
    import torch

    examples_by_row: list[list[dict[str, Any]]] = []
    for row in rows:
        row_examples = build_teacher_rank_examples(
            [row],
            skills,
            candidate_map,
            teacher_score_map,
            max_negatives=max_negatives,
            allow_weak_teacher=allow_weak_teacher,
        )
        if not row_examples:
            raise ValueError(f"no PAG set MarginMSE example for row={row.get('index')}")
        examples_by_row.append(row_examples)
    if not examples_by_row:
        raise ValueError("no PAG set MarginMSE examples were built for the batch")
    candidate_ids_by_row: list[list[str]] = []
    candidate_scores_by_row: list[dict[str, float]] = []
    positive_ids_by_row: list[set[str]] = []
    for row_examples in examples_by_row:
        candidate_ids: list[str] = []
        candidate_scores: dict[str, float] = {}
        positive_ids: set[str] = set()
        for example in row_examples:
            positive_sid = str(example["positive_skill_id"])
            negative_sid = str(example["negative_skill_id"])
            positive_ids.add(positive_sid)
            for sid in (positive_sid, negative_sid):
                if sid not in candidate_ids:
                    candidate_ids.append(sid)
            candidate_scores[positive_sid] = float(example["teacher_positive_score"])
            candidate_scores.setdefault(negative_sid, float(example["teacher_negative_score"]))
        candidate_ids_by_row.append(candidate_ids)
        candidate_scores_by_row.append(candidate_scores)
        positive_ids_by_row.append(positive_ids)
    max_candidates = max(len(values) for values in candidate_ids_by_row)
    max_tokens = max(
        len(pag_set_token_ids.get(sid, []))
        for candidate_ids in candidate_ids_by_row
        for sid in candidate_ids
    )
    if max_tokens <= 0:
        raise ValueError("PAG set manifest has no token ids for a sampled candidate")
    token_ids = torch.zeros((len(candidate_ids_by_row), max_candidates, max_tokens), dtype=torch.long, device=device)
    token_mask = torch.zeros((len(candidate_ids_by_row), max_candidates, max_tokens), dtype=torch.bool, device=device)
    valid = torch.zeros((len(candidate_ids_by_row), max_candidates), dtype=torch.bool, device=device)
    positive = torch.zeros((len(candidate_ids_by_row), max_candidates), dtype=torch.bool, device=device)
    teacher_scores = torch.zeros((len(candidate_ids_by_row), max_candidates), dtype=torch.float32, device=device)
    for index, candidate_ids in enumerate(candidate_ids_by_row):
        for candidate_index, sid in enumerate(candidate_ids):
            values = [int(value) for value in pag_set_token_ids.get(sid, [])]
            if not values:
                raise ValueError(f"PAG set manifest is missing token ids for skill={sid}")
            token_ids[index, candidate_index, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
            token_mask[index, candidate_index, : len(values)] = True
            valid[index, candidate_index] = True
            positive[index, candidate_index] = sid in positive_ids_by_row[index]
            teacher_scores[index, candidate_index] = candidate_scores_by_row[index][sid]
    return {
        "pag_set_candidate_token_ids": token_ids,
        "pag_set_candidate_token_mask": token_mask,
        "pag_set_candidate_valid": valid,
        "pag_set_candidate_positive": positive,
        "pag_set_candidate_teacher_scores": teacher_scores,
    }


def load_ranked_skill_candidates(path: Path | None) -> dict[str, list[str]]:
    """Backward-compatible ID-only view of the RIPOR candidate pool."""
    candidate_map, _ = load_ranked_skill_candidates_with_scores(path)
    return candidate_map


def ranked_candidates_for_row(row: dict[str, Any], candidate_map: dict[str, list[str]]) -> list[str]:
    values: list[str] = []
    for key in retrieval_row_keys(row):
        values.extend(candidate_map.get(key, []))
    for field in ("ripor_negative_skill_ids", "hard_negative_skill_ids", "negative_skill_ids"):
        values.extend(str(value) for value in row.get(field, []) or [])
    return _unique_preserving_order(values)


def build_ripor_candidate_batch(
    rows: list[dict[str, Any]],
    semantic_ids: dict[str, list[int]],
    *,
    depth: int,
    max_negatives: int,
    hard_negative_map: dict[str, list[str]],
    teacher_score_map: dict[str, dict[str, float]] | None = None,
    seed: int,
    device,
    require_teacher_scores: bool = False,
) -> dict[str, Any]:
    """Build teacher-forced SID paths for RIPOR relevance-prefix ranking.

    Each expanded single-SID row retains all original query positives.  Those
    siblings are positives here, never accidental in-batch negatives.  Every
    candidate also receives a frozen teacher score; absent scores use explicit
    reciprocal-rank/positive-negative anchor fallbacks.
    """
    import torch

    if max_negatives < 0:
        raise ValueError(f"ripor max negatives must be non-negative, got {max_negatives}")
    known_skill_ids = sorted(semantic_ids)
    positives_by_row = [ripor_positive_skill_ids(row, semantic_ids) for row in rows]
    in_batch_positives = _unique_preserving_order(
        [sid for positives in positives_by_row for sid in positives]
    )
    candidate_lists: list[list[str]] = []
    for row, positives in zip(rows, positives_by_row):
        positive_set = set(positives)
        negatives: list[str] = []
        for sid in [*ranked_candidates_for_row(row, hard_negative_map), *in_batch_positives]:
            if sid in semantic_ids and sid not in positive_set and sid not in negatives:
                negatives.append(sid)
                if len(negatives) >= max_negatives:
                    break
        if len(negatives) < max_negatives:
            stable_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{retrieval_row_keys(row)[0]}".encode("utf-8")).digest()[:8],
                byteorder="big",
            )
            rng = random.Random(stable_seed)
            random_pool = [sid for sid in known_skill_ids if sid not in positive_set and sid not in negatives]
            rng.shuffle(random_pool)
            negatives.extend(random_pool[: max_negatives - len(negatives)])
        candidate_lists.append([*positives, *negatives[:max_negatives]])

    max_candidates = max(len(values) for values in candidate_lists)
    candidate_codes = torch.zeros((len(rows), max_candidates, depth), dtype=torch.long, device=device)
    candidate_valid = torch.zeros((len(rows), max_candidates), dtype=torch.bool, device=device)
    candidate_positive = torch.zeros((len(rows), max_candidates), dtype=torch.bool, device=device)
    candidate_teacher_scores = torch.zeros((len(rows), max_candidates), dtype=torch.float32, device=device)
    candidate_teacher_observed = torch.zeros((len(rows), max_candidates), dtype=torch.bool, device=device)
    for row_index, (positives, candidates) in enumerate(zip(positives_by_row, candidate_lists)):
        positive_set = set(positives)
        row_teacher_scores: dict[str, float] = {}
        if teacher_score_map:
            for key in retrieval_row_keys(rows[row_index]):
                row_teacher_scores.update(teacher_score_map.get(key, {}))
        known_scores = [score for sid, score in row_teacher_scores.items() if sid in candidates and math.isfinite(score)]
        positive_fallback = max(known_scores, default=0.0) + 1.0
        negative_fallback = min(known_scores, default=0.0) - 1.0
        for candidate_index, sid in enumerate(candidates):
            candidate_codes[row_index, candidate_index] = torch.tensor(
                semantic_ids[sid][:depth], dtype=torch.long, device=device
            )
            candidate_valid[row_index, candidate_index] = True
            candidate_positive[row_index, candidate_index] = sid in positive_set
            if sid in row_teacher_scores and math.isfinite(row_teacher_scores[sid]):
                teacher_score = row_teacher_scores[sid]
                candidate_teacher_observed[row_index, candidate_index] = True
            else:
                if require_teacher_scores:
                    raise ValueError(
                        "RIPOR reproduction requires a frozen teacher score for every sampled "
                        f"candidate; missing skill={sid}, row={rows[row_index].get('index')}"
                    )
                teacher_score = positive_fallback if sid in positive_set else negative_fallback
            candidate_teacher_scores[row_index, candidate_index] = float(teacher_score)
    return {
        "ripor_candidate_codes": candidate_codes,
        "ripor_candidate_valid": candidate_valid,
        "ripor_candidate_positive": candidate_positive,
        "ripor_candidate_teacher_scores": candidate_teacher_scores,
        "ripor_candidate_teacher_observed": candidate_teacher_observed,
    }


def load_pag_planner_scores(path: Path | None, score_mode: str) -> dict[str, dict[str, float]]:
    """Load a frozen dense/BM25 planner without ever accessing evaluation gold."""
    if path is None:
        return {}
    if score_mode not in {"reciprocal_rank", "raw"}:
        raise ValueError(f"unknown PAG planner score mode: {score_mode}")
    planner: dict[str, dict[str, float]] = {}
    for row in load_jsonl(path):
        ranked = [str(sid) for sid in row.get("ranked_skill_ids", [])]
        if not ranked:
            continue
        raw_scores = list(row.get("ranked_scores") or [])
        scores: dict[str, float] = {}
        for rank, sid in enumerate(ranked):
            if score_mode == "raw" and rank < len(raw_scores):
                value = float(raw_scores[rank])
                if not math.isfinite(value):
                    continue
            else:
                value = 1.0 / float(rank + 1)
            scores[sid] = max(scores.get(sid, float("-inf")), value)
        if score_mode == "raw" and scores:
            # Missing planner entries are interpreted as zero.  Shift each
            # query's observed raw scores so a negative minimum cannot make an
            # unobserved child (zero) look artificially attractive.
            minimum = min(scores.values())
            scores = {sid: score - minimum for sid, score in scores.items()}
        for key in retrieval_row_keys(row):
            planner.setdefault(key, scores)
    return planner


def build_semantic_prefix_descendants(
    semantic_ids: dict[str, list[int]],
    depth: int,
) -> dict[tuple[int, ...], list[str]]:
    descendants: dict[tuple[int, ...], list[str]] = {}
    for sid, semantic_id in semantic_ids.items():
        for length in range(1, depth + 1):
            descendants.setdefault(tuple(semantic_id[:length]), []).append(sid)
    return descendants


def pag_prefix_planner_score(
    prefix: list[int],
    planner_scores: dict[str, float],
    prefix_descendants: dict[tuple[int, ...], list[str]],
    aggregation: str,
    excluded_skill_ids: set[str] | None = None,
) -> float:
    excluded = excluded_skill_ids or set()
    values = [
        planner_scores[sid]
        for sid in prefix_descendants.get(tuple(prefix), [])
        if sid in planner_scores and sid not in excluded
    ]
    if not values:
        return 0.0
    if aggregation == "max":
        return float(max(values))
    if aggregation == "logsumexp":
        high = max(values)
        return float(high + math.log(sum(math.exp(value - high) for value in values)))
    raise ValueError(f"unknown PAG planner aggregation: {aggregation}")


_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")


def bm25_tokenize(text: str) -> list[str]:
    return _BM25_TOKEN_RE.findall(str(text).lower())


def command_build_pag_set_ids(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    skills = load_jsonl(args.skills_jsonl)
    model_name = str(args.model)
    if args.sparse_checkpoint is not None:
        checkpoint = torch.load(args.sparse_checkpoint / "pag_sparse_encoder.pt", map_location="cpu")
        model_name = str(checkpoint["model"])
        vocab_size = int(checkpoint["vocab_size"])
        tokenizer = AutoTokenizer.from_pretrained(args.sparse_checkpoint)
        model = make_pag_sparse_encoder(model_name, vocab_size)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
        token_ids: dict[str, list[int]] = {}
        with torch.no_grad():
            for start in range(0, len(skills), args.batch_size):
                chunk = skills[start : start + args.batch_size]
                encoded = tokenizer(
                    [str(skill.get("skillrouter_text") or format_skillrouter_skill(skill)) for skill in chunk],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                logits = model(encoded["input_ids"], encoded["attention_mask"])
                available = logits.shape[-1] - len(special_ids)
                if args.max_tokens <= 0 or args.max_tokens > available:
                    raise ValueError(
                        f"PAG top-m={args.max_tokens} exceeds available non-special vocabulary {available}"
                    )
                masked_logits = logits.clone()
                if special_ids:
                    masked_logits[:, sorted(special_ids)] = float("-inf")
                top_ids = torch.topk(masked_logits, k=args.max_tokens, dim=-1).indices.tolist()
                for skill, values in zip(chunk, top_ids):
                    if len(set(values)) != args.max_tokens or any(int(value) in special_ids for value in values):
                        raise ValueError(f"PAG set for {skill_id(skill)} is not exactly top-m unique non-special ids")
                    token_ids[skill_id(skill)] = [int(value) for value in values]
        source_format = "pag_sparse_encoder_top_tokens"
    else:
        if not args.lexical_fallback:
            raise ValueError(
                "build-pag-set-ids requires --sparse-checkpoint for the PAG set branch; "
                "pass --lexical-fallback only for a clearly labeled ablation"
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        token_ids = build_pag_set_token_ids(skills, tokenizer, max_tokens=args.max_tokens)
        vocab_size = int(AutoConfig.from_pretrained(args.model).vocab_size)
        if vocab_size <= 0:
            vocab_size = max(max(values) for values in token_ids.values()) + 1
        source_format = "explicit_lexical_fallback_ablation"
    payload = {
        "format": "pag_lexical_set_ids_v2",
        "model": model_name,
        "vocab_size": vocab_size,
        "max_tokens": args.max_tokens,
        "skill_token_ids": token_ids,
        "source_skills": str(args.skills_jsonl),
        "source_format": source_format,
        "sparse_checkpoint": str(args.sparse_checkpoint) if args.sparse_checkpoint else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "skills": len(token_ids), "vocab_size": vocab_size}, sort_keys=True))


def command_canonicalize_skill_aliases(args: argparse.Namespace) -> None:
    skills = load_jsonl(args.skills_jsonl)
    canonical_skills, alias_rows, alias_to_canonical, catalog_manifest = canonicalize_exact_skill_documents(skills)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "canonical_skills.jsonl", canonical_skills)
    write_jsonl(args.output_dir / "skill_aliases.jsonl", alias_rows)
    write_json(args.output_dir / "skill_alias_map.json", alias_to_canonical)

    embedding_manifest = None
    if args.embeddings is not None:
        import torch

        try:
            embedding_payload = torch.load(args.embeddings, map_location="cpu", weights_only=False)
        except TypeError:
            embedding_payload = torch.load(args.embeddings, map_location="cpu")
        if not isinstance(embedding_payload, dict):
            raise ValueError("--embeddings must contain a dict with skill_ids and embeddings")
        embedding_skill_ids = [str(sid) for sid in embedding_payload.get("skill_ids", [])]
        embeddings_value = embedding_payload.get("embeddings")
        if embeddings_value is None:
            raise ValueError("--embeddings payload must contain a 2D embeddings tensor")
        embeddings = torch.as_tensor(embeddings_value)
        if len(embeddings.shape) != 2:
            raise ValueError("--embeddings payload must contain a 2D embeddings tensor")
        if len(embedding_skill_ids) != int(embeddings.shape[0]):
            raise ValueError(
                f"embedding skill_ids/rows mismatch: {len(embedding_skill_ids)} != {embeddings.shape[0]}"
            )
        if len(set(embedding_skill_ids)) != len(embedding_skill_ids):
            raise ValueError("--embeddings contains duplicate skill_ids")
        embedding_index = {sid: index for index, sid in enumerate(embedding_skill_ids)}
        canonical_ids = [skill_id(row) for row in canonical_skills]
        missing_embedding_ids = [sid for sid in canonical_ids if sid not in embedding_index]
        if missing_embedding_ids:
            raise ValueError(
                f"--embeddings is missing {len(missing_embedding_ids)} canonical skills; "
                f"first={missing_embedding_ids[:5]}"
            )
        selected_indices = torch.tensor(
            [embedding_index[sid] for sid in canonical_ids],
            dtype=torch.long,
        )
        canonical_embeddings = embeddings.index_select(0, selected_indices)
        embeddings_output = args.output_dir / "canonical_skill_embeddings.pt"
        torch.save(
            {
                "skill_ids": canonical_ids,
                "embeddings": canonical_embeddings,
                "source_embeddings": str(args.embeddings),
                "alias_manifest": str(args.output_dir / "alias_manifest.json"),
            },
            embeddings_output,
        )
        embedding_manifest = {
            "source": str(args.embeddings),
            "output": str(embeddings_output),
            "source_rows": int(embeddings.shape[0]),
            "canonical_rows": int(canonical_embeddings.shape[0]),
            "embedding_dim": int(canonical_embeddings.shape[1]),
        }

    row_manifests: list[dict[str, Any]] = []
    for row_file_index, rows_path in enumerate(args.rows or []):
        source_rows = load_jsonl(rows_path)
        output_rows: list[dict[str, Any]] = []
        changed_rows = 0
        changed_aliases = 0
        removed_duplicate_mentions = 0
        unknown_ids: set[str] = set()
        for row in source_rows:
            remapped, audit = remap_row_skill_aliases(
                row,
                alias_to_canonical,
                allow_unknown=args.allow_unknown_row_skills,
            )
            output_rows.append(remapped)
            changed_rows += int(audit["changed"])
            changed_aliases += int(audit["changed_aliases"])
            removed_duplicate_mentions += int(audit["removed_duplicate_mentions"])
            unknown_ids.update(audit["unknown_skill_ids"])
        output_name = f"rows_{row_file_index:03d}.{rows_path.name}"
        output_path = args.output_dir / output_name
        write_jsonl(output_path, output_rows)
        row_manifests.append(
            {
                "source": str(rows_path),
                "output": str(output_path),
                "rows": len(output_rows),
                "changed_rows": changed_rows,
                "changed_aliases": changed_aliases,
                "removed_duplicate_mentions": removed_duplicate_mentions,
                "unknown_skill_ids": sorted(unknown_ids),
            }
        )

    manifest = {
        "command": "canonicalize-skill-aliases",
        "skills_jsonl": str(args.skills_jsonl),
        **catalog_manifest,
        "allow_unknown_row_skills": bool(args.allow_unknown_row_skills),
        "embeddings": embedding_manifest,
        "row_files": row_manifests,
    }
    write_json(args.output_dir / "alias_manifest.json", manifest)
    print(
        f"canonicalized skills={manifest['source_skills']} -> {manifest['canonical_skills']} "
        f"aliases={manifest['aliased_skills']} row_files={len(row_manifests)} output={args.output_dir}"
    )


def command_prepare(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.dataset)
    ids = set(collect_selected_ids(rows))
    lookup = load_skill_lookup_for_ids(args.skills, ids)

    skill_rows = []
    for sid in sorted(ids):
        skill = lookup[sid]
        skill_rows.append(
            {
                "skill_id": sid,
                "name": skill.get("skill_name") or skill.get("name") or sid,
                "description": skill.get("description") or skill.get("task_description") or "",
                "body": skill.get("body") or "",
                "skillrouter_text": format_skillrouter_skill(skill, args.desc_max, args.body_max),
            }
        )

    train_rows = []
    for index, row in enumerate(rows):
        ids_for_row = selected_skill_ids(row)
        query = str((row.get("generated") or {}).get("query") or "")
        train_rows.append(
            {
                "index": index,
                "source_index": row.get("index"),
                "target_skill_count": int(row.get("target_skill_count") or len(ids_for_row)),
                "query": query,
                "query_text": format_skillrouter_query(query, args.query_max_chars),
                "selected_skill_ids": ids_for_row,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "selected_skills.jsonl", skill_rows)
    write_jsonl(args.output_dir / "decoder_rows.base.jsonl", train_rows)
    write_json(
        args.output_dir / "prepare_manifest.json",
        {
            "dataset": str(args.dataset),
            "skills": str(args.skills),
            "rows": len(rows),
            "unique_selected_skills": len(ids),
            "skill_format": "name | description[:desc_max] | body[:body_max]",
            "query_instruction": SKILLROUTER_QUERY_INSTRUCTION,
            "desc_max": args.desc_max,
            "body_max": args.body_max,
            "query_max_chars": args.query_max_chars,
            "query_truncation_enabled": bool(args.query_max_chars and args.query_max_chars > 0),
        },
    )
    print(f"prepared rows={len(train_rows)} unique_skills={len(skill_rows)} output={args.output_dir}")


def make_pag_sparse_encoder(base_model: str, vocab_size: int):
    import torch
    import torch.nn as nn
    from transformers import T5ForConditionalGeneration

    class PAGSparseEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = T5ForConditionalGeneration.from_pretrained(base_model)
            if int(self.backbone.config.vocab_size) != int(vocab_size):
                raise ValueError(
                    "PAG SPLADE vocabulary must equal the T5 LM head vocabulary: "
                    f"model={self.backbone.config.vocab_size}, requested={vocab_size}"
                )

        def forward(self, input_ids, attention_mask):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # PAG's lexical branch scores every query-token decoder
                # position and max-pools the transformed LM coordinates.
                decoder_input_ids=input_ids,
                return_dict=True,
            )
            # This is the official SPLADE/T5 representation used by PAG:
            # log(1 + ReLU(T5 logits)), followed by top-m lexical coordinates.
            lexical = torch.log1p(torch.relu(outputs.logits))
            lexical = lexical * attention_mask.unsqueeze(-1).to(dtype=lexical.dtype)
            return lexical.max(dim=1).values

    return PAGSparseEncoder()


def command_train_pag_sparse_encoder(args: argparse.Namespace) -> None:
    """Train PAG's official SPLADE sparse encoder with MarginMSE + FLOPS."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoTokenizer

    context = initialize_distributed(cpu=args.cpu)
    torch.manual_seed(args.seed + context.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + context.rank)
    rows = load_jsonl(args.rows) if args.rows is not None else []
    skills = load_jsonl(args.skills_jsonl)
    candidate_map, teacher_score_map = load_ranked_skill_candidates_with_scores(args.hard_negatives)
    examples = build_teacher_rank_examples(
        rows,
        skills,
        candidate_map,
        teacher_score_map,
        max_negatives=args.max_negatives,
        allow_weak_teacher=args.allow_weak_teacher,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab_size = int(AutoConfig.from_pretrained(args.model).vocab_size)
    if vocab_size <= 0:
        raise ValueError("PAG sparse encoder requires a tokenizer vocabulary")
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    model_core = make_pag_sparse_encoder(args.model, vocab_size).to(context.device)
    model = wrap_ddp(model_core, context)
    optimizer = torch.optim.AdamW(model_core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(examples) / (args.batch_size * context.world_size)))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    step = 0
    epoch = 0
    model.train()
    while step < total_steps:
        for batch_rows in distributed_epoch_batches(examples, args.batch_size, context, seed=args.seed, epoch=epoch):
            if step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=bool(args.bf16 and getattr(context.device, "type", "") == "cuda"),
            ):
                def encode_batch(texts: list[str]):
                    encoded = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=args.max_length,
                        return_tensors="pt",
                    )
                    return model(
                        input_ids=encoded["input_ids"].to(context.device),
                        attention_mask=encoded["attention_mask"].to(context.device),
                    )

                query_rep = encode_batch(
                    [row["query"] for row in batch_rows]
                )
                pos_rep = encode_batch(
                    [row["positive_text"] for row in batch_rows]
                )
                neg_rep = encode_batch(
                    [row["negative_text"] for row in batch_rows]
                )
                student_margin = (query_rep * pos_rep).sum(dim=-1) - (query_rep * neg_rep).sum(dim=-1)
                teacher_margin = torch.tensor(
                    [row["teacher_positive_score"] - row["teacher_negative_score"] for row in batch_rows],
                    dtype=student_margin.dtype,
                    device=context.device,
                )
                rank_loss = F.mse_loss(student_margin, teacher_margin, reduction="sum")
                global_query_rep = all_gather_tensor_with_grad(query_rep, context)
                global_pos_rep = all_gather_tensor_with_grad(pos_rep, context)
                global_neg_rep = all_gather_tensor_with_grad(neg_rep, context)
                query_reg = torch.mean(torch.abs(global_query_rep), dim=0).square().sum()
                doc_reg = (
                    torch.mean(torch.abs(global_pos_rep), dim=0).square().sum()
                    + torch.mean(torch.abs(global_neg_rep), dim=0).square().sum()
                ) / 2.0
                rank_loss = global_normalized_loss(
                    rank_loss,
                    torch.tensor(float(len(batch_rows)), device=context.device),
                    context,
                )
                # FLOPS is computed after an autograd-preserving all-gather, so
                # it is the same global-batch objective under 1 or 8 GPUs.
                loss = rank_loss + float(args.query_reg_weight) * query_reg + float(args.doc_reg_weight) * doc_reg
            loss.backward()
            optimizer.step()
            step += 1
            if context.is_main and (step % args.logging_steps == 0 or step == total_steps):
                print(f"pag_sparse_step={step} loss={float(loss.detach().cpu()):.6f}", flush=True)
        epoch += 1
    barrier(context)
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model_core.state_dict(),
                "model": args.model,
                "vocab_size": vocab_size,
                "max_length": args.max_length,
                "top_k": args.top_k,
                "objective": "T5SPLADE MarginMSE + FLOPS (PAG stage 1)",
                "query_reg_weight": args.query_reg_weight,
                "doc_reg_weight": args.doc_reg_weight,
                "examples": len(examples),
            },
            args.output_dir / "pag_sparse_encoder.pt",
        )
        tokenizer.save_pretrained(str(args.output_dir))
    barrier(context)
    local_rows = [
        (index, skill_id(skill), str(skill.get("skillrouter_text") or format_skillrouter_skill(skill)))
        for index, skill in enumerate(skills)
        if skill_id(skill) and index % context.world_size == context.rank
    ]
    local_sets: list[tuple[int, str, list[int]]] = []
    model_core.eval()
    with torch.no_grad():
        for start in range(0, len(local_rows), args.batch_size):
            chunk = local_rows[start : start + args.batch_size]
            encoded = tokenizer(
                [item[2] for item in chunk],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            reps = model_core(
                encoded["input_ids"].to(context.device),
                encoded["attention_mask"].to(context.device),
            )
            if args.top_k <= 0 or args.top_k > reps.shape[-1] - len(special_ids):
                raise ValueError(
                    f"PAG top-m={args.top_k} exceeds available non-special vocabulary "
                    f"{reps.shape[-1] - len(special_ids)}"
                )
            masked_reps = reps.clone()
            if special_ids:
                masked_reps[:, sorted(special_ids)] = float("-inf")
            top_ids = torch.topk(masked_reps, k=args.top_k, dim=-1).indices.cpu().tolist()
            local_sets.extend(
                (item[0], item[1], [int(value) for value in values])
                for item, values in zip(chunk, top_ids)
            )
    gathered = all_gather_object(local_sets, context)
    if context.is_main:
        merged = sorted((item for part in gathered for item in part), key=lambda item: item[0])
        write_json(
            args.output_dir / "pag_set_ids.json",
            {
                "format": "pag_splade_lexical_set_ids_v2",
                "model": args.model,
                "vocab_size": vocab_size,
                "max_tokens": args.top_k,
                "skill_token_ids": {item[1]: item[2] for item in merged},
                "source_checkpoint": str(args.output_dir / "pag_sparse_encoder.pt"),
                "objective": "T5SPLADE top-m lexical coordinates",
            },
        )
    barrier(context)
    destroy_distributed(context)


def command_train_ripor_relevance_encoder(args: argparse.Namespace) -> None:
    """Train RIPOR's relevance-based T5 dense encoder with MarginMSE."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, T5Model

    context = initialize_distributed(cpu=args.cpu)
    torch.manual_seed(args.seed + context.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + context.rank)
    rows = load_jsonl(args.rows)
    skills = load_jsonl(args.skills_jsonl)
    candidate_map, teacher_score_map = load_ranked_skill_candidates_with_scores(args.hard_negatives)
    examples = build_teacher_rank_examples(
        rows,
        skills,
        candidate_map,
        teacher_score_map,
        max_negatives=args.max_negatives,
        allow_weak_teacher=args.allow_weak_teacher,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class RelevanceEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = T5Model.from_pretrained(args.model)
            self.decoder_length = int(args.decoder_length)

        def forward(self, input_ids, attention_mask):
            start_id = self.backbone.config.decoder_start_token_id
            if start_id is None:
                start_id = self.backbone.config.pad_token_id
            decoder_input_ids = torch.full(
                (input_ids.shape[0], self.decoder_length),
                int(self.backbone.config.pad_token_id),
                dtype=torch.long,
                device=input_ids.device,
            )
            decoder_input_ids[:, 0] = int(start_id)
            return self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True,
            ).last_hidden_state[:, 0, :]

    model_core = RelevanceEncoder().to(context.device)
    optimizer = torch.optim.AdamW(model_core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = wrap_ddp(model_core, context)
    steps_per_epoch = max(1, math.ceil(len(examples) / (args.batch_size * context.world_size)))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    step = 0
    epoch = 0
    model.train()

    def encode_batch(texts: list[str], max_length: int):
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return model(
            input_ids=encoded["input_ids"].to(context.device),
            attention_mask=encoded["attention_mask"].to(context.device),
        )

    while step < total_steps:
        for batch_rows in distributed_epoch_batches(examples, args.batch_size, context, seed=args.seed, epoch=epoch):
            if step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=bool(args.bf16 and getattr(context.device, "type", "") == "cuda"),
            ):
                query_rep = encode_batch([row["query"] for row in batch_rows], args.max_query_length)
                pos_rep = encode_batch([row["positive_text"] for row in batch_rows], args.max_skill_length)
                neg_rep = encode_batch([row["negative_text"] for row in batch_rows], args.max_skill_length)
                student_margin = (query_rep * pos_rep).sum(dim=-1) - (query_rep * neg_rep).sum(dim=-1)
                teacher_margin = torch.tensor(
                    [row["teacher_positive_score"] - row["teacher_negative_score"] for row in batch_rows],
                    dtype=student_margin.dtype,
                    device=context.device,
                )
                loss = global_normalized_loss(
                    F.mse_loss(student_margin, teacher_margin, reduction="sum"),
                    torch.tensor(float(len(batch_rows)), device=context.device),
                    context,
                )
            loss.backward()
            optimizer.step()
            step += 1
            if context.is_main and (step % args.logging_steps == 0 or step == total_steps):
                print(f"relevance_step={step} loss={float(loss.detach().cpu()):.6f}", flush=True)
        epoch += 1

    barrier(context)
    output_dir = args.output_dir
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        m0_checkpoint_path = output_dir / "relevance_encoder.pt"
        m0_config = model_core.backbone.config.to_dict()
        m0_config_sha256 = stable_payload_sha256(m0_config)
        m0_state_keys = sorted(str(key) for key in model_core.state_dict())
        torch.save(
            {
                "artifact_format": "ripor_m0_relevance_v1",
                "state_dict": model_core.state_dict(),
                "state_keys": m0_state_keys,
                "model": args.model,
                "model_config": m0_config,
                "model_config_sha256": m0_config_sha256,
                "max_query_length": args.max_query_length,
                "max_skill_length": args.max_skill_length,
                "decoder_length": args.decoder_length,
                "pairs": len(examples),
                "world_size": context.world_size,
                "objective": "RIPOR T5 dense encoder MarginMSE",
            },
            m0_checkpoint_path,
        )
        tokenizer.save_pretrained(str(output_dir))
    barrier(context)
    indexed_skills = [
        (index, skill_id(skill), str(skill.get("skillrouter_text") or format_skillrouter_skill(skill)))
        for index, skill in enumerate(skills)
        if skill_id(skill)
    ]
    local_skill_rows = [item for item in indexed_skills if item[0] % context.world_size == context.rank]
    local_vectors: list[tuple[int, str, list[float]]] = []
    model_core.eval()
    with torch.no_grad():
        for start in range(0, len(local_skill_rows), args.batch_size):
            chunk = local_skill_rows[start : start + args.batch_size]
            encoded = tokenizer(
                [item[2] for item in chunk],
                padding=True,
                truncation=True,
                max_length=args.max_skill_length,
                return_tensors="pt",
            )
            vectors = model_core(
                encoded["input_ids"].to(context.device),
                encoded["attention_mask"].to(context.device),
            )
            local_vectors.extend(
                (item[0], item[1], vector.cpu().float().tolist())
                for item, vector in zip(chunk, vectors)
            )
    gathered = all_gather_object(local_vectors, context)
    if context.is_main:
        merged = sorted((item for part in gathered for item in part), key=lambda item: item[0])
        m0_checkpoint_path = output_dir / "relevance_encoder.pt"
        m0_checkpoint_sha256 = file_sha256(m0_checkpoint_path)
        torch.save(
            {
                "skill_ids": [item[1] for item in merged],
                "embeddings": torch.tensor([item[2] for item in merged], dtype=torch.float32),
                "source": str(output_dir / "relevance_encoder.pt"),
                "objective": "RIPOR relevance-based DocID initialization",
                "artifact_format": "ripor_m0_skill_embeddings_v1",
                "source_relevance_checkpoint": str(m0_checkpoint_path),
                "source_relevance_checkpoint_sha256": m0_checkpoint_sha256,
                "source_relevance_model": args.model,
                "source_relevance_model_config_sha256": m0_config_sha256,
                "source_relevance_decoder_length": int(args.decoder_length),
                "source_relevance_state_keys": m0_state_keys,
            },
            output_dir / "skill_embeddings.pt",
        )
        write_json(
            output_dir / "relevance_encoder_manifest.json",
            {
                "rows": str(args.rows),
                "hard_negatives": str(args.hard_negatives) if args.hard_negatives else None,
                "skills_jsonl": str(args.skills_jsonl),
                "pairs": len(examples),
                "skills": len(merged),
                "objective": "RIPOR T5 dense MarginMSE before RQ/LETTER quantization",
                "global_batch_size": args.batch_size * context.world_size,
                "world_size": context.world_size,
                "strict_teacher_scores": not args.allow_weak_teacher,
                "relevance_checkpoint_sha256": m0_checkpoint_sha256,
                "model_config_sha256": m0_config_sha256,
                "decoder_length": int(args.decoder_length),
            },
        )
    barrier(context)
    destroy_distributed(context)


def command_encode(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        seq_lens = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), seq_lens]

    context = initialize_distributed(cpu=args.cpu)
    skills = load_jsonl(args.skills_jsonl)
    texts = [str(row["skillrouter_text"]) for row in skills]
    ids = [str(row["skill_id"]) for row in skills]
    local_indices = list(range(context.rank, len(texts), context.world_size))
    device = context.device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embs = []
    for start in range(0, len(local_indices), args.batch_size):
        batch_indices = local_indices[start : start + args.batch_size]
        batch = [texts[index] for index in batch_indices]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            embs = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().float())
        if context.is_main:
            print(f"encoded {min(start + args.batch_size, len(local_indices))}/{len(local_indices)} on rank0", flush=True)

    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier(context)
    shard_dir = args.output_dir / ".encode_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    embedding_dim = int(all_embs[0].shape[1]) if all_embs else int(getattr(model.config, "hidden_size", 0))
    local_embeddings = torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, embedding_dim), dtype=torch.float32)
    torch.save(
        {"indices": local_indices, "embeddings": local_embeddings},
        shard_dir / f"rank-{context.rank:04d}.pt",
    )
    barrier(context)
    if context.is_main:
        merged = torch.empty((len(texts), embedding_dim), dtype=torch.float32)
        for rank in range(context.world_size):
            shard = torch.load(shard_dir / f"rank-{rank:04d}.pt", map_location="cpu")
            merged[torch.tensor(shard["indices"], dtype=torch.long)] = shard["embeddings"]
        torch.save({"skill_ids": ids, "embeddings": merged}, args.output_dir / "skill_embeddings.pt")
        write_json(
            args.output_dir / "embedding_manifest.json",
            {
                "model": args.model,
                "skills": len(ids),
                "max_length": args.max_length,
                "per_device_batch_size": args.batch_size,
                "ddp_world_size": context.world_size,
                "embedding_dim": embedding_dim,
            },
        )
        print(f"saved {args.output_dir / 'skill_embeddings.pt'}")
    barrier(context)
    destroy_distributed(context)


def command_train_rqvae(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from vector_quantize_pytorch import ResidualVQ

    class SkillRQVAE(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            codebook_size: int,
            depth: int,
            identity_latent: bool,
        ) -> None:
            super().__init__()
            self.identity_latent = bool(identity_latent)
            if self.identity_latent:
                if input_dim != latent_dim:
                    raise ValueError(
                        "RIPOR identity RQ requires latent_dim to equal the M0 embedding width: "
                        f"input_dim={input_dim}, latent_dim={latent_dim}"
                    )
                self.encoder = nn.Identity()
                self.decoder = nn.Identity()
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, input_dim),
                )
            self.quantizer = ResidualVQ(
                dim=latent_dim,
                num_quantizers=depth,
                codebook_size=codebook_size,
                kmeans_init=True,
            )

        def forward(self, x):
            z = self.encoder(x)
            quantized, indices, commit_loss = self.quantizer(z)
            recon = self.decoder(quantized)
            return recon, indices, commit_loss

        @torch.no_grad()
        def encode_codes(self, x):
            z = self.encoder(x)
            _, indices, _ = self.quantizer(z)
            return indices

    payload = torch.load(args.embeddings, map_location="cpu")
    if not isinstance(payload, dict) or "embeddings" not in payload or "skill_ids" not in payload:
        raise ValueError("--embeddings must be a skill_embeddings.pt artifact from train-ripor-relevance-encoder")
    embeddings = payload["embeddings"].float()
    skill_ids = list(payload["skill_ids"])
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    latent_dim = int(args.latent_dim or embeddings.shape[1])
    identity_latent = bool(args.ripor_identity)
    m0_provenance_keys = (
        "source_relevance_checkpoint",
        "source_relevance_checkpoint_sha256",
        "source_relevance_model",
        "source_relevance_model_config_sha256",
        "source_relevance_decoder_length",
        "source_relevance_state_keys",
    )
    if identity_latent:
        if payload.get("artifact_format") != "ripor_m0_skill_embeddings_v1":
            raise ValueError(
                "strict RIPOR identity RQ requires skill_embeddings.pt emitted by "
                "train-ripor-relevance-encoder"
            )
        missing_provenance = [key for key in m0_provenance_keys if payload.get(key) in (None, "", [])]
        if missing_provenance:
            raise ValueError(f"M0 skill embeddings are missing provenance fields: {missing_provenance}")
        source_checkpoint = Path(str(payload["source_relevance_checkpoint"]))
        if source_checkpoint.exists():
            actual_source_hash = file_sha256(source_checkpoint)
            if actual_source_hash != str(payload["source_relevance_checkpoint_sha256"]):
                raise ValueError(
                    "M0 skill embeddings do not match their source relevance checkpoint: "
                    f"expected={payload['source_relevance_checkpoint_sha256']}, got={actual_source_hash}"
                )
    if identity_latent and latent_dim != int(embeddings.shape[1]):
        raise ValueError(
            "--ripor-identity requires --latent-dim to equal the M0 embedding width; "
            f"embedding_dim={embeddings.shape[1]}, latent_dim={latent_dim}. "
            "Use --no-ripor-identity only for a legacy learned bottleneck ablation."
        )
    model = SkillRQVAE(
        input_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=latent_dim,
        codebook_size=args.codebook_size,
        depth=args.depth,
        identity_latent=identity_latent,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(embeddings), batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _, commit_loss = model(batch)
            loss = F.mse_loss(recon, batch) + args.commit_weight * commit_loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = sum(losses) / max(1, len(losses))
        print(f"epoch={epoch} loss={epoch_loss:.6f}", flush=True)

    model.eval()
    codes = model.encode_codes(embeddings.to(device)).detach().cpu()
    while codes.ndim > 2:
        codes = codes.squeeze(1)
    raw_codes = {sid: [int(x) for x in codes[i].tolist()] for i, sid in enumerate(skill_ids)}
    semantic_ids = deduplicate_code_tuples(raw_codes)
    code_width = max(3, len(str(max(args.codebook_size - 1, max(v[-1] for v in semantic_ids.values())))))

    rows = [
        {
            "skill_id": sid,
            "rq_codes": raw_codes[sid],
            "semantic_id": semantic_ids[sid],
            "semantic_id_mode": SEMANTIC_ID_MODE_DEDUP,
            "tokens": semantic_id_tokens(semantic_ids[sid], code_width, SEMANTIC_ID_MODE_DEDUP),
        }
        for sid in sorted(skill_ids)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    codebook_embeddings = model.quantizer.codebooks.detach().cpu().float()
    if codebook_embeddings.ndim != 3:
        raise ValueError(
            "ResidualVQ codebooks must have shape [depth, codebook_size, dim], "
            f"got {tuple(codebook_embeddings.shape)}"
        )
    torch.save(
        {
            "artifact_format": "ripor_m0_rq_codebook_v1",
            "codebook_embeddings": codebook_embeddings,
            "codebook_size": int(args.codebook_size),
            "depth": int(args.depth),
            "embedding_dim": int(codebook_embeddings.shape[-1]),
            "source_embeddings": str(args.embeddings),
            "objective": "RIPOR relevance-based RQ centroid initialization",
            "identity_latent": identity_latent,
            **{key: payload.get(key) for key in m0_provenance_keys},
        },
        args.output_dir / "codebook_embeddings.pt",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": embeddings.shape[1],
            "hidden_dim": args.hidden_dim,
            "latent_dim": latent_dim,
            "codebook_size": args.codebook_size,
            "depth": args.depth,
            "semantic_id_mode": SEMANTIC_ID_MODE_DEDUP,
            "identity_latent": identity_latent,
            **{key: payload.get(key) for key in m0_provenance_keys},
        },
        args.output_dir / "rqvae.pt",
    )
    write_jsonl(args.output_dir / "semantic_ids.jsonl", rows)
    write_json(
        args.output_dir / "rqvae_manifest.json",
        {
            "skills": len(skill_ids),
            "codebook_size": args.codebook_size,
            "depth": args.depth,
            "hidden_dim": args.hidden_dim,
            "latent_dim": latent_dim,
            "identity_latent": identity_latent,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "commit_weight": args.commit_weight,
            "seed": args.seed,
            "final_loss": epoch_loss if args.epochs > 0 else None,
            "code_width": code_width,
            "codebook_embeddings": str(args.output_dir / "codebook_embeddings.pt"),
            **{key: payload.get(key) for key in m0_provenance_keys},
            "dedup_column": True,
            "collision_groups": sum(count > 1 for count in Counter(tuple(v) for v in raw_codes.values()).values()),
        },
    )
    print(f"saved semantic ids to {args.output_dir / 'semantic_ids.jsonl'}")


def load_semantic_ids_with_mode(path: Path) -> tuple[dict[str, list[int]], str]:
    semantic_ids: dict[str, list[int]] = {}
    modes: set[str] = set()
    for row_index, row in enumerate(load_jsonl(path)):
        sid = str(row.get("skill_id") or "")
        if not sid:
            raise ValueError(f"semantic id row {row_index} has no skill_id: {path}")
        if sid in semantic_ids:
            raise ValueError(f"duplicate skill_id in semantic ids: {sid}")
        values = [int(x) for x in row.get("semantic_id", [])]
        semantic_ids[sid] = values
        modes.add(normalize_semantic_id_mode(row.get("semantic_id_mode")))
    if len(modes) != 1:
        raise ValueError(f"semantic id rows must use one schema, got {sorted(modes)}: {path}")
    mode = next(iter(modes), SEMANTIC_ID_MODE_DEDUP)
    validate_semantic_ids(semantic_ids, mode)
    return semantic_ids, mode


def load_semantic_ids(path: Path) -> dict[str, list[int]]:
    semantic_ids, _ = load_semantic_ids_with_mode(path)
    return semantic_ids


def load_semantic_id_mode(path: Path) -> str:
    _, mode = load_semantic_ids_with_mode(path)
    return mode


def validate_semantic_ids(
    semantic_ids: dict[str, list[int]],
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> None:
    mode = normalize_semantic_id_mode(semantic_id_mode)
    if not semantic_ids:
        raise ValueError("semantic ids must not be empty")
    lengths = {len(values) for values in semantic_ids.values()}
    if len(lengths) != 1:
        raise ValueError(f"semantic ids must have a single fixed length, got {sorted(lengths)}")
    semantic_len = next(iter(lengths))
    minimum_length = 2 if mode == SEMANTIC_ID_MODE_DEDUP else 1
    if semantic_len < minimum_length:
        raise ValueError(f"semantic ids are too short for mode={mode}: length={semantic_len}")
    negative = [sid for sid, values in semantic_ids.items() if any(value < 0 for value in values)]
    if negative:
        raise ValueError(f"semantic ids contain negative values, first skill: {negative[0]}")
    sequence_counts = Counter(tuple(values) for values in semantic_ids.values())
    duplicate_sequences = [values for values, count in sequence_counts.items() if count > 1]
    if duplicate_sequences:
        raise ValueError(f"semantic ids must be unique, duplicate sequence: {duplicate_sequences[0]}")


def semantic_ids_sha256(
    semantic_ids: dict[str, list[int]],
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> str:
    validate_semantic_ids(semantic_ids, semantic_id_mode)
    canonical = json.dumps(
        sorted((sid, [int(value) for value in values]) for sid, values in semantic_ids.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_checkpoint_semantic_ids(
    checkpoint: dict[str, Any],
    semantic_ids: dict[str, list[int]],
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> bool:
    mode = normalize_semantic_id_mode(semantic_id_mode)
    checkpoint_mode = normalize_semantic_id_mode(checkpoint.get("semantic_id_mode"))
    if checkpoint_mode != mode:
        raise ValueError(
            "semantic id mode does not match the decoder checkpoint: "
            f"checkpoint={checkpoint_mode}, semantic_ids={mode}"
        )
    expected_hash = checkpoint.get("semantic_ids_sha256")
    expected_count = checkpoint.get("semantic_id_count")
    if expected_hash is None:
        return False
    actual_hash = semantic_ids_sha256(semantic_ids, mode)
    if str(expected_hash) != actual_hash:
        raise ValueError(
            "semantic ids do not match the decoder checkpoint: "
            f"expected sha256={expected_hash}, got {actual_hash}"
        )
    if expected_count is not None and int(expected_count) != len(semantic_ids):
        raise ValueError(
            "semantic id count does not match the decoder checkpoint: "
            f"expected {expected_count}, got {len(semantic_ids)}"
        )
    return True


def decoder_checkpoint_metadata(
    args: argparse.Namespace,
    semantic_ids: dict[str, list[int]],
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> dict[str, Any]:
    mode = normalize_semantic_id_mode(semantic_id_mode)
    return {
        "max_source_length": int(args.max_source_length),
        "source_text_policy": GR_SOURCE_TEXT_POLICY,
        "semantic_id_count": len(semantic_ids),
        "semantic_ids_sha256": semantic_ids_sha256(semantic_ids, mode),
        "semantic_id_mode": mode,
    }


def validate_checkpoint_source_length(
    checkpoint: dict[str, Any],
    max_source_length: int,
    allow_mismatch: bool,
) -> bool:
    expected = checkpoint.get("max_source_length")
    if expected is None:
        return False
    if int(expected) != int(max_source_length) and not allow_mismatch:
        raise ValueError(
            "max_source_length does not match the decoder checkpoint: "
            f"checkpoint={expected}, requested={max_source_length}; "
            "use the checkpoint setting or explicitly pass --allow-checkpoint-config-mismatch"
        )
    return int(expected) == int(max_source_length)


def validate_checkpoint_source_text_policy(checkpoint: dict[str, Any], allow_mismatch: bool) -> bool:
    expected = checkpoint.get("source_text_policy")
    if expected is None:
        return False
    if str(expected) != GR_SOURCE_TEXT_POLICY and not allow_mismatch:
        raise ValueError(
            "source_text_policy does not match the decoder checkpoint: "
            f"checkpoint={expected}, current={GR_SOURCE_TEXT_POLICY}; "
            "use matching preprocessing or explicitly pass --allow-checkpoint-config-mismatch"
        )
    return str(expected) == GR_SOURCE_TEXT_POLICY


def resolve_eval_max_source_length(
    requested: int | None,
    checkpoint: dict[str, Any],
    allow_mismatch: bool,
) -> tuple[int, str]:
    checkpoint_value = checkpoint.get("max_source_length")
    if requested is None:
        if checkpoint_value is not None:
            return int(checkpoint_value), "checkpoint"
        raise ValueError(
            "legacy decoder checkpoint does not record max_source_length; "
            "pass --max-source-length explicitly"
        )
    validate_checkpoint_source_length(checkpoint, requested, allow_mismatch)
    return int(requested), "cli"


def infer_structured_sid_config(
    semantic_ids: dict[str, list[int]],
    codebook_size: int | None = None,
    dedup_size: int | None = None,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> dict[str, Any]:
    mode = normalize_semantic_id_mode(semantic_id_mode)
    validate_semantic_ids(semantic_ids, mode)
    semantic_len = len(next(iter(semantic_ids.values())))
    depth = semantic_len - 1 if mode == SEMANTIC_ID_MODE_DEDUP else semantic_len
    code_values = [values[:-1] if mode == SEMANTIC_ID_MODE_DEDUP else values for values in semantic_ids.values()]
    inferred_codebook_size = max(max(values) for values in code_values) + 1
    inferred_dedup_size = (
        max(values[-1] for values in semantic_ids.values()) + 1
        if mode == SEMANTIC_ID_MODE_DEDUP
        else 0
    )
    if codebook_size is not None and codebook_size < inferred_codebook_size:
        raise ValueError(
            f"codebook_size={codebook_size} cannot represent observed code value "
            f"{inferred_codebook_size - 1}"
        )
    if mode == SEMANTIC_ID_MODE_LEARNED and dedup_size not in (None, 0):
        raise ValueError(f"learned_codes has no dedup head; expected dedup_size=0, got {dedup_size}")
    if dedup_size is not None and dedup_size < inferred_dedup_size:
        raise ValueError(
            f"dedup_size={dedup_size} cannot represent observed dedup value "
            f"{inferred_dedup_size - 1}"
        )
    return {
        "depth": depth,
        "codebook_size": int(codebook_size or inferred_codebook_size),
        "dedup_size": int(dedup_size if dedup_size is not None else inferred_dedup_size),
        "semantic_id_mode": mode,
    }


def command_build_decoder_data(args: argparse.Namespace) -> None:
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    value_max = max(max(values) for values in semantic_ids.values())
    code_width = max(3, len(str(value_max)))
    rows = load_jsonl(args.rows)
    sep = args.separator_text or (SKILL_SEP_TOKEN if args.target_mode == "special" else NATIVE_SKILL_SEP_TEXT)
    out_rows = []
    for row in rows:
        target = build_decoder_target(
            row["selected_skill_ids"],
            semantic_ids,
            code_width=code_width,
            target_mode=args.target_mode,
            separator_text=sep,
            semantic_id_mode=semantic_id_mode,
        )
        out_rows.append(
            {
                **row,
                "decoder_input": f"{args.input_prefix}{row['query']}",
                "decoder_target": target,
            }
        )

    code_tokens = (
        sorted(
            {
                token
                for sem_id in semantic_ids.values()
                for token in semantic_id_tokens(sem_id, code_width, semantic_id_mode)
            }
        )
        if args.target_mode == "special"
        else []
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "decoder_rows.jsonl", out_rows)
    write_json(
        args.output_dir / "decoder_special_tokens.json",
        [SKILL_SEP_TOKEN, *code_tokens] if args.target_mode == "special" else [],
    )
    write_json(
        args.output_dir / "decoder_data_manifest.json",
        {
            "rows": len(out_rows),
            "semantic_ids": str(args.semantic_ids),
            "input_prefix": args.input_prefix,
            "code_width": code_width,
            "target_mode": args.target_mode,
            "semantic_id_mode": semantic_id_mode,
            "separator_text": sep,
            "ordering": "selected skill ids sorted lexicographically before serialization",
        },
    )
    print(f"saved decoder data rows={len(out_rows)} output={args.output_dir}")


def command_split_decoder_data(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    train_rows, dev_rows, test_rows, manifest = stratified_split_rows(
        rows=rows,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_field=args.stratify_field,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "dev.jsonl", dev_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    write_json(args.output_dir / "split_manifest.json", {**manifest, "source_rows": str(args.rows)})
    print(
        f"saved split train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)} output={args.output_dir}"
    )


def command_build_decoding_trie(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    semantic_rows = load_jsonl(args.semantic_ids)
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    special_tokens = json.loads(args.special_tokens.read_text(encoding="utf-8")) if args.special_tokens else []
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    sep = args.separator_text or (SKILL_SEP_TOKEN if args.target_mode == "special" else NATIVE_SKILL_SEP_TEXT)
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    if len(sep_ids) != 1:
        raise SystemExit(f"separator must be one tokenizer token for current decoder state machine: {sep!r} -> {sep_ids}")
    value_max = max(max(values) for values in semantic_ids.values())
    code_width = max(3, len(str(value_max)))

    token_rows = []
    token_sequences = []
    for row in semantic_rows:
        native_text = None
        if args.target_mode == "special":
            tokens = [
                str(token)
                for token in row.get(
                    "tokens",
                    semantic_id_tokens([int(v) for v in row["semantic_id"]], code_width, semantic_id_mode),
                )
            ]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            for token, token_id in zip(tokens, token_ids):
                encoded = tokenizer.encode(token, add_special_tokens=False)
                if token_id == tokenizer.unk_token_id or len(encoded) != 1 or encoded[0] != token_id:
                    raise SystemExit(f"special token is not single-token after tokenizer setup: {token} -> {encoded}")
        elif args.target_mode == "native":
            native_text = native_semantic_id_text(
                [int(v) for v in row["semantic_id"]],
                code_width,
                semantic_id_mode,
            )
            token_ids = tokenizer.encode(native_text, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
        else:
            raise ValueError(f"unknown target mode: {args.target_mode}")
        token_sequences.append([int(token_id) for token_id in token_ids])
        token_rows.append(
            {
                "skill_id": row["skill_id"],
                "tokens": tokens,
                "token_ids": [int(token_id) for token_id in token_ids],
                "native_text": native_text,
            }
        )

    trie = build_prefix_trie(token_sequences)
    sep_id = int(sep_ids[0])
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise SystemExit("tokenizer has no eos_token_id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "semantic_id_token_ids.jsonl", token_rows)
    write_json(args.output_dir / "semantic_id_trie.json", trie)
    write_json(
        args.output_dir / "decoding_trie_manifest.json",
        {
            "semantic_ids": str(args.semantic_ids),
            "special_tokens": str(args.special_tokens) if args.special_tokens else None,
            "model": args.model,
            "target_mode": args.target_mode,
            "semantic_id_mode": semantic_id_mode,
            "code_width": code_width,
            "skills": len(semantic_rows),
            "sequences": len(token_sequences),
            "unique_sequences": len({tuple(seq) for seq in token_sequences}),
            "sequence_length": sorted({len(seq) for seq in token_sequences}),
            "trie_nodes": count_trie_nodes(trie),
            "root_branching": len([key for key in trie if key != TRIE_TERMINAL_KEY]),
            "separator_text": sep,
            "separator_token_id": int(sep_id),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": int(eos_id),
            "max_skills": args.max_skills,
            "decode_policy": (
                "Use semantic_id_trie for one skill id. After a terminal node, allow EOS or "
                "SKILL_SEP if emitted skill count is below max_skills; after SKILL_SEP return "
                "to trie root. Track emitted semantic ids dynamically to block duplicates."
            ),
        },
    )
    print(f"saved decoding trie skills={len(semantic_rows)} output={args.output_dir}")


def command_train_decoder(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    class JsonlSeq2SeqDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]], tokenizer, max_source_length: int, max_target_length: int) -> None:
            self.rows = rows
            self.tokenizer = tokenizer
            self.max_source_length = max_source_length
            self.max_target_length = max_target_length

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = self.rows[idx]
            model_inputs = self.tokenizer(
                row["decoder_input"],
                max_length=self.max_source_length,
                truncation=True,
            )
            labels = self.tokenizer(
                text_target=row["decoder_target"],
                max_length=self.max_target_length,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    rows = load_jsonl(args.decoder_rows)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.dev_rows is not None:
        train_rows = rows
        dev_rows = load_jsonl(args.dev_rows)
    else:
        dev_size = max(1, math.ceil(len(rows) * args.dev_ratio)) if args.dev_ratio > 0 else 0
        dev_rows = rows[:dev_size]
        train_rows = rows[dev_size:]
    special_tokens = json.loads(args.special_tokens.read_text(encoding="utf-8")) if args.special_tokens else []
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    train_ds = JsonlSeq2SeqDataset(train_rows, tokenizer, args.max_source_length, args.max_target_length)
    dev_ds = JsonlSeq2SeqDataset(dev_rows, tokenizer, args.max_source_length, args.max_target_length) if dev_rows else None
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    load_best_model = bool(
        dev_ds is not None
        and args.load_best_model_at_end
        and args.eval_steps > 0
        and args.save_steps > 0
    )
    if load_best_model and args.save_steps % args.eval_steps != 0:
        raise ValueError("--save-steps must be a multiple of --eval-steps when loading the best decoder model")

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if dev_ds is not None and args.eval_steps > 0 else "no",
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss" if load_best_model else None,
        greater_is_better=False if load_best_model else None,
        predict_with_generate=False,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
        seed=args.seed,
    )
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_ds,
        "eval_dataset": dev_ds,
        "data_collator": collator,
    }
    trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(str(args.output_dir / "final"))
        tokenizer.save_pretrained(str(args.output_dir / "final"))
        write_json(
            args.output_dir / "decoder_train_manifest.json",
            {
                "model": args.model,
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "max_source_length": args.max_source_length,
                "max_target_length": args.max_target_length,
                "seed": args.seed,
                "special_tokens": len(special_tokens),
                "save_total_limit": args.save_total_limit,
                "load_best_model_at_end": load_best_model,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "ddp_world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "per_device_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * int(os.environ.get("WORLD_SIZE", "1")),
            },
        )


def command_eval_decoder(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = load_jsonl(args.rows)
    trie = json.loads(args.trie.read_text(encoding="utf-8"))
    token_rows = load_jsonl(args.semantic_id_token_ids)
    sequence_to_skill_id = {
        tuple(int(token_id) for token_id in row["token_ids"]): str(row["skill_id"])
        for row in token_rows
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    model.eval()

    sep = args.separator_text or SKILL_SEP_TOKEN
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    if len(sep_ids) != 1:
        raise SystemExit(f"separator must be one tokenizer token for current decoder state machine: {sep!r} -> {sep_ids}")
    sep_id = int(sep_ids[0])
    eos_id = int(tokenizer.eos_token_id)
    decoder_start_token_id = model.config.decoder_start_token_id
    if eos_id is None:
        raise SystemExit("tokenizer has no eos_token_id")

    def prefix_allowed_tokens_fn(_batch_id: int, input_ids) -> list[int]:
        return allowed_next_tokens_for_skill_trie(
            [int(token_id) for token_id in input_ids.tolist()],
            trie=trie,
            sep_id=sep_id,
            eos_id=eos_id,
            max_skills=args.max_skills,
            decoder_start_token_id=decoder_start_token_id,
        )

    predictions: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        encoded = tokenizer(
            [str(row["decoder_input"]) for row in batch],
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=1,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                early_stopping=True,
            )
        for row, output_ids in zip(batch, outputs.detach().cpu().tolist()):
            sequences, invalid = parse_skill_token_sequences(
                output_ids,
                trie=trie,
                sep_id=sep_id,
                eos_id=eos_id,
                decoder_start_token_id=decoder_start_token_id,
                finalize_open_terminal=True,
            )
            predicted_ids = [
                sequence_to_skill_id[sequence]
                for sequence in sequences
                if sequence in sequence_to_skill_id
            ]
            unknown_sequences = [list(sequence) for sequence in sequences if sequence not in sequence_to_skill_id]
            gold_ids = [str(sid) for sid in row.get("selected_skill_ids", [])]
            pred_set = set(predicted_ids)
            gold_set = set(gold_ids)
            hit_count = len(pred_set & gold_set)
            predictions.append(
                {
                    "index": row.get("index"),
                    "source_index": row.get("source_index"),
                    "target_skill_count": row.get("target_skill_count"),
                    "query": row.get("query"),
                    "gold_skill_ids": gold_ids,
                    "predicted_skill_ids": predicted_ids,
                    "predicted_token_sequences": [list(sequence) for sequence in sequences],
                    "generated_token_ids": output_ids,
                    "generated_text": tokenizer.decode(output_ids, skip_special_tokens=False),
                    "hit_count": hit_count,
                    "recall": float(hit_count / len(gold_set)) if gold_set else 0.0,
                    "hit_rate": 1.0 if hit_count > 0 else 0.0,
                    "exact_match": pred_set == gold_set,
                    "count_match": len(pred_set) == len(gold_set),
                    "invalid": bool(invalid or unknown_sequences),
                    "unknown_token_sequences": unknown_sequences,
                }
            )
        print(f"decoded {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    metrics = compute_retrieval_metrics(
        rows,
        [{"predicted_skill_ids": pred["predicted_skill_ids"], "invalid": pred["invalid"]} for pred in predictions],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "eval_metrics.json", metrics)
    write_json(
        args.output_dir / "eval_manifest.json",
        {
            "rows": str(args.rows),
            "model_dir": str(args.model_dir),
            "trie": str(args.trie),
            "semantic_id_token_ids": str(args.semantic_id_token_ids),
            "batch_size": args.batch_size,
            "max_source_length": args.max_source_length,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "max_skills": args.max_skills,
            "separator_text": sep,
            "separator_token_id": sep_id,
            "split": args.split_name,
            "metrics": "top-1 constrained generated set; hit_count=|pred∩gold|; recall=hit_count/|gold|",
        },
    )
    print(json.dumps(metrics["overall"], ensure_ascii=False, sort_keys=True))


def command_eval_bm25(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    skills = load_candidate_skill_texts(args.skills_jsonl)
    k_values = parse_k_values(args.k_values)
    max_rank = max(k_values)

    tokenized_docs = [bm25_tokenize(skill["text"]) for skill in skills]
    doc_freq: Counter[str] = Counter()
    doc_term_freqs: list[Counter[str]] = []
    doc_lengths: list[int] = []
    for tokens in tokenized_docs:
        counts = Counter(tokens)
        doc_term_freqs.append(counts)
        doc_lengths.append(len(tokens))
        doc_freq.update(counts.keys())
    num_docs = len(skills)
    avgdl = sum(doc_lengths) / max(1, num_docs)
    idf = {
        term: math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        for term, df in doc_freq.items()
    }

    predictions: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        query_terms = bm25_tokenize(row_query_text(row))
        query_counts = Counter(query_terms)
        scores: list[tuple[float, str]] = []
        for doc_index, skill in enumerate(skills):
            score = 0.0
            dl = doc_lengths[doc_index]
            tf = doc_term_freqs[doc_index]
            for term in query_counts:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + args.k1 * (1.0 - args.b + args.b * dl / max(avgdl, 1e-9))
                score += idf.get(term, 0.0) * (freq * (args.k1 + 1.0)) / denom
            scores.append((score, skill["skill_id"]))
        ranked = [sid for _, sid in sorted(scores, key=lambda item: (-item[0], item[1]))[:max_rank]]
        gold_ids = [str(sid) for sid in row.get("selected_skill_ids", [])]
        predictions.append(
            {
                "index": row.get("index"),
                "source_index": row.get("source_index"),
                "target_skill_count": row.get("target_skill_count"),
                "query": row.get("query"),
                "gold_skill_ids": gold_ids,
                "ranked_skill_ids": ranked,
                "invalid": False,
            }
        )
        if (row_index + 1) % args.log_every == 0 or row_index + 1 == len(rows):
            print(f"ranked {row_index + 1}/{len(rows)}", flush=True)

    metrics = compute_at_k_metrics(rows, predictions, k_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "ranked_metrics.json", metrics)
    write_json(
        args.output_dir / "eval_manifest.json",
        {
            "baseline": "bm25",
            "rows": str(args.rows),
            "skills_jsonl": str(args.skills_jsonl),
            "candidate_skills": len(skills),
            "k_values": k_values,
            "max_rank": max_rank,
            "k1": args.k1,
            "b": args.b,
            "metrics": "Hit@K, Recall@K, FC@K over ranked skill list",
        },
    )
    print(json.dumps(metrics["overall"], ensure_ascii=False, sort_keys=True))


def mean_pool_hidden(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def command_eval_t5_dense(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, T5EncoderModel

    rows = load_jsonl(args.rows)
    skills = load_candidate_skill_texts(args.skills_jsonl)
    k_values = parse_k_values(args.k_values)
    max_rank = max(k_values)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = T5EncoderModel.from_pretrained(args.model).to(device)
    model.eval()

    def encode_texts(texts: list[str], max_length: int) -> torch.Tensor:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), args.batch_size):
                encoded = tokenizer(
                    texts[start : start + args.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                pooled = mean_pool_hidden(hidden, encoded["attention_mask"])
                chunks.append(F.normalize(pooled.detach().cpu().float(), p=2, dim=1))
        return torch.cat(chunks, dim=0)

    skill_embeddings = encode_texts([skill["text"] for skill in skills], args.max_skill_length)
    query_embeddings = encode_texts([row_query_text(row) for row in rows], args.max_query_length)

    predictions: list[dict[str, Any]] = []
    scores = query_embeddings @ skill_embeddings.T
    top_scores, top_indices = torch.topk(scores, k=min(max_rank, len(skills)), dim=1)
    for row_index, row in enumerate(rows):
        ranked = [skills[int(index)]["skill_id"] for index in top_indices[row_index].tolist()]
        predictions.append(
            {
                "index": row.get("index"),
                "source_index": row.get("source_index"),
                "target_skill_count": row.get("target_skill_count"),
                "query": row.get("query"),
                "gold_skill_ids": [str(sid) for sid in row.get("selected_skill_ids", [])],
                "ranked_skill_ids": ranked,
                "ranked_scores": [float(score) for score in top_scores[row_index].tolist()],
                "invalid": False,
            }
        )
    metrics = compute_at_k_metrics(rows, predictions, k_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "ranked_metrics.json", metrics)
    write_json(
        args.output_dir / "eval_manifest.json",
        {
            "baseline": "t5_dense",
            "rows": str(args.rows),
            "skills_jsonl": str(args.skills_jsonl),
            "candidate_skills": len(skills),
            "model": args.model,
            "k_values": k_values,
            "max_rank": max_rank,
            "max_query_length": args.max_query_length,
            "max_skill_length": args.max_skill_length,
            "metrics": "Hit@K, Recall@K, FC@K over ranked skill list",
        },
    )
    print(json.dumps(metrics["overall"], ensure_ascii=False, sort_keys=True))


def _load_ripor_codebook_embeddings(
    path: Path | str,
    *,
    depth: int,
    codebook_size: int,
    d_model: int,
    expected_relevance_checkpoint: Path | str | None = None,
):
    """Load the position-specific RQ centroids used by RIPOR's M0 model."""
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        # RIPOR uses residual-quantizer centroids directly as the
        # position-specific T5 hidden-space vectors E_i.  A learned
        # bottleneck encoder would put them in an unrelated coordinate
        # system even when the width happens to match.
        if payload.get("artifact_format") != "ripor_m0_rq_codebook_v1":
            raise ValueError(
                "strict RIPOR codebook initialization requires the provenance-bearing "
                "codebook_embeddings.pt artifact from train-rqvae"
            )
        if payload.get("identity_latent") is not True:
            raise ValueError(
                "strict RIPOR codebook initialization requires RQ centroids "
                "trained directly in the M0/T5 embedding space "
                "(identity_latent=true)"
            )
        provenance_keys = (
            "source_relevance_checkpoint_sha256",
            "source_relevance_model_config_sha256",
            "source_relevance_decoder_length",
            "source_relevance_state_keys",
        )
        missing_provenance = [key for key in provenance_keys if payload.get(key) in (None, "", [])]
        if missing_provenance:
            raise ValueError(f"RIPOR codebook is missing M0 provenance fields: {missing_provenance}")
        if expected_relevance_checkpoint is None:
            raise ValueError(
                "strict RIPOR codebook initialization requires the matching "
                "--ripor-relevance-checkpoint"
            )
        expected_path = Path(expected_relevance_checkpoint)
        actual_hash = file_sha256(expected_path)
        if actual_hash != str(payload["source_relevance_checkpoint_sha256"]):
            raise ValueError(
                "RIPOR codebook/M0 checkpoint hash mismatch: "
                f"codebook={payload['source_relevance_checkpoint_sha256']}, checkpoint={actual_hash}"
            )
        values = payload.get("codebook_embeddings")
        if values is None:
            values = payload.get("codebooks")
    else:
        raise ValueError(
            "strict RIPOR codebook initialization requires a provenance-bearing "
            "codebook_embeddings.pt dictionary, not a raw tensor"
        )
    if values is None:
        raise ValueError(f"RIPOR codebook file has no codebook_embeddings/codebooks tensor: {path}")
    if isinstance(values, (list, tuple)):
        values = torch.stack([torch.as_tensor(value) for value in values])
    values = torch.as_tensor(values).float()
    expected = (int(depth), int(codebook_size), int(d_model))
    if tuple(values.shape) != expected:
        raise ValueError(
            "RIPOR codebook centroids must match [depth, codebook_size, d_model]: "
            f"expected={expected}, got={tuple(values.shape)} from {path}"
        )
    return values


def make_structured_t5_sid_model(
    base_model: str,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
    pag_set_vocab_size: int | None = None,
    ripor_codebook_embeddings: Path | str | None = None,
    ripor_relevance_checkpoint: Path | str | None = None,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForSeq2SeqLM

    class StructuredT5SIDModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModelForSeq2SeqLM.from_pretrained(base_model)
            d_model = int(self.backbone.config.d_model)
            self.codebook_size = int(codebook_size)
            self.depth = int(depth)
            self.dedup_size = int(dedup_size)
            self.semantic_id_mode = normalize_semantic_id_mode(semantic_id_mode)
            self.pag_set_vocab_size = int(pag_set_vocab_size or 0)
            if self.semantic_id_mode == SEMANTIC_ID_MODE_LEARNED and self.dedup_size != 0:
                raise ValueError(f"learned_codes requires dedup_size=0, got {self.dedup_size}")
            self.control_type = structured_sid_control_type(self.depth, self.semantic_id_mode)
            self.distributed_context: DistributedContext | None = None
            self.sid_embedding = nn.Embedding(
                structured_sid_vocab_size(self.codebook_size, self.depth, self.dedup_size),
                d_model,
            )
            # RIPOR's position-specific output tables are dot products with RQ
            # centroids; they do not carry a per-token bias term.
            self.code_heads = nn.ModuleList(
                nn.Linear(d_model, self.codebook_size, bias=False) for _ in range(self.depth)
            )
            self.dedup_head = (
                nn.Linear(d_model, self.dedup_size)
                if self.semantic_id_mode == SEMANTIC_ID_MODE_DEDUP
                else None
            )
            self.control_head = nn.Linear(d_model, 2)
            self.ripor_codebook_init = str(ripor_codebook_embeddings) if ripor_codebook_embeddings else None
            if ripor_codebook_embeddings:
                codebooks = _load_ripor_codebook_embeddings(
                    ripor_codebook_embeddings,
                    depth=self.depth,
                    codebook_size=self.codebook_size,
                    d_model=d_model,
                    expected_relevance_checkpoint=ripor_relevance_checkpoint,
                )
                with torch.no_grad():
                    for level in range(self.depth):
                        start = level * self.codebook_size
                        stop = start + self.codebook_size
                        self.sid_embedding.weight[start:stop].copy_(codebooks[level])
                        self.code_heads[level].weight.copy_(codebooks[level])
                        if self.code_heads[level].bias is not None:
                            self.code_heads[level].bias.zero_()
            # PAG's lexical branch uses the T5 LM head over a one-token
            # decoder, exactly as T5-SPLADE.  It is not a separately learned
            # pooled classifier; ``pag_set_vocab_size`` records the manifest
            # vocabulary and is checked against the backbone at runtime.
            self.pag_set_head = None
            if self.pag_set_vocab_size and self.pag_set_vocab_size != int(self.backbone.config.vocab_size):
                raise ValueError(
                    "PAG set vocabulary must match the T5 LM head vocabulary: "
                    f"manifest={self.pag_set_vocab_size}, model={self.backbone.config.vocab_size}"
                )

        def configure_distributed(self, context: DistributedContext | None) -> None:
            self.distributed_context = context

        def encode_source(self, input_ids, attention_mask):
            encoder_outputs = self.backbone.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return encoder_outputs.last_hidden_state

        def _decoder_hidden_from_encoder(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_prefix_input_ids=None,
            decoder_prefix_attention_mask=None,
        ):
            decoder_embeds = self.sid_embedding(decoder_input_ids)
            if decoder_prefix_input_ids is not None:
                prefix_embeds = self.backbone.shared(decoder_prefix_input_ids)
                decoder_embeds = torch.cat([prefix_embeds, decoder_embeds], dim=1)
                if decoder_prefix_attention_mask is None:
                    decoder_prefix_attention_mask = torch.ones(
                        decoder_prefix_input_ids.shape[:2],
                        dtype=decoder_attention_mask.dtype,
                        device=decoder_attention_mask.device,
                    )
                decoder_attention_mask = torch.cat(
                    [decoder_prefix_attention_mask, decoder_attention_mask], dim=1
                )
            decoder_outputs = self.backbone.decoder(
                inputs_embeds=decoder_embeds,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
            )
            hidden = decoder_outputs.last_hidden_state
            if decoder_prefix_input_ids is not None:
                hidden = hidden[:, decoder_prefix_input_ids.shape[1] :, :]
            return hidden

        def _decoder_hidden(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_prefix_input_ids=None,
            decoder_prefix_attention_mask=None,
        ):
            encoder_hidden_states = self.encode_source(input_ids, attention_mask)
            return self._decoder_hidden_from_encoder(
                encoder_hidden_states,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                decoder_prefix_input_ids=decoder_prefix_input_ids,
                decoder_prefix_attention_mask=decoder_prefix_attention_mask,
            )

        def _normalize_loss(self, loss_sum, count, global_loss_normalization: bool):
            if global_loss_normalization and self.distributed_context is not None:
                return global_normalized_loss(loss_sum, count, self.distributed_context)
            return loss_sum / count.clamp_min(1.0)

        def ripor_prefix_scores(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            candidate_codes,
            decoder_prefix_input_ids=None,
            decoder_prefix_attention_mask=None,
        ):
            """Return RIPOR's cumulative conditional *logit* score.

            The official RIPOR decoder replaces the normal vocabulary softmax by
            position-specific codebook output tables and ranks a candidate with
            ``sum_i h_i @ E_i[c_i]``.  The rank objective therefore uses the raw
            conditional logits, not ``log_softmax`` probabilities.  Decoding
            below consumes the same raw score so the teacher-forced objective and
            constrained beam have identical semantics.
            """
            batch_size, candidate_count, rank_depth = candidate_codes.shape
            if rank_depth != self.depth:
                raise ValueError(
                    f"RIPOR candidate depth={rank_depth} does not match model code depth={self.depth}"
                )
            bos_id = structured_sid_bos_id(
                self.codebook_size,
                self.depth,
                self.dedup_size,
                self.semantic_id_mode,
            )
            decoder_input_ids = candidate_codes.new_full(
                (batch_size, candidate_count, rank_depth), bos_id
            )
            for level in range(1, rank_depth):
                decoder_input_ids[:, :, level] = (
                    (level - 1) * self.codebook_size + candidate_codes[:, :, level - 1]
                )
            flat_decoder_ids = decoder_input_ids.reshape(batch_size * candidate_count, rank_depth)
            flat_decoder_mask = torch.ones_like(flat_decoder_ids)
            flat_encoder = encoder_hidden_states.repeat_interleave(candidate_count, dim=0)
            flat_encoder_mask = encoder_attention_mask.repeat_interleave(candidate_count, dim=0)
            flat_prefix_ids = None
            flat_prefix_mask = None
            if decoder_prefix_input_ids is not None:
                flat_prefix_ids = decoder_prefix_input_ids.repeat_interleave(candidate_count, dim=0)
                if decoder_prefix_attention_mask is not None:
                    flat_prefix_mask = decoder_prefix_attention_mask.repeat_interleave(candidate_count, dim=0)
            hidden = self._decoder_hidden_from_encoder(
                flat_encoder,
                flat_encoder_mask,
                flat_decoder_ids,
                flat_decoder_mask,
                decoder_prefix_input_ids=flat_prefix_ids,
                decoder_prefix_attention_mask=flat_prefix_mask,
            )
            cumulative = hidden.new_zeros(batch_size * candidate_count)
            prefixes = []
            flat_codes = candidate_codes.reshape(batch_size * candidate_count, rank_depth)
            for level, head in enumerate(self.code_heads):
                logits = head(hidden[:, level, :])
                selected = logits.gather(1, flat_codes[:, level : level + 1]).squeeze(1)
                cumulative = cumulative + selected
                prefixes.append(cumulative)
            return torch.stack(prefixes, dim=-1).reshape(batch_size, candidate_count, rank_depth)

        # Kept as a compatibility alias for older callers; the reproduction
        # path now uses the official conditional-logit score above.
        def ripor_prefix_log_scores(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            candidate_codes,
            decoder_prefix_input_ids=None,
            decoder_prefix_attention_mask=None,
        ):
            return self.ripor_prefix_scores(
                encoder_hidden_states,
                encoder_attention_mask,
                candidate_codes,
                decoder_prefix_input_ids=decoder_prefix_input_ids,
                decoder_prefix_attention_mask=decoder_prefix_attention_mask,
            )

        def ripor_prefix_rank_loss(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            candidate_codes,
            candidate_valid,
            candidate_positive,
            margin: float,
            temperature: float,
            candidate_teacher_scores=None,
            candidate_teacher_observed=None,
            alpha_beta: float = DEFAULT_RIPOR_ALPHA_BETA,
            prefix_max_length: int | None = None,
            decoder_prefix_input_ids=None,
            decoder_prefix_attention_mask=None,
        ):
            """Official RIPOR prefix-oriented MarginMSE objective.

            The ``margin`` and ``temperature`` parameters remain accepted for
            checkpoint/CLI compatibility but are intentionally not used: RIPOR
            regresses the raw teacher margin with MSE, weighted by the
            normalized concave prefix coefficient ``alpha_i``.
            """
            if candidate_teacher_scores is None:
                candidate_teacher_scores = candidate_positive.to(dtype=torch.float32)
            if candidate_teacher_scores.shape != candidate_valid.shape:
                raise ValueError(
                    "RIPOR teacher scores must have shape [batch, candidates], "
                    f"got {tuple(candidate_teacher_scores.shape)} vs {tuple(candidate_valid.shape)}"
                )
            if candidate_teacher_observed is not None and candidate_teacher_observed.shape != candidate_valid.shape:
                raise ValueError("RIPOR teacher observation mask must match candidate validity shape")
            full_rank_depth = int(candidate_codes.shape[-1])
            prefix_scores = self.ripor_prefix_log_scores(
                encoder_hidden_states,
                encoder_attention_mask,
                candidate_codes,
                decoder_prefix_input_ids=decoder_prefix_input_ids,
                decoder_prefix_attention_mask=decoder_prefix_attention_mask,
            )
            if prefix_max_length is not None:
                prefix_max_length = max(1, min(int(prefix_max_length), prefix_scores.shape[-1]))
                prefix_scores = prefix_scores[..., :prefix_max_length]
            candidate_teacher_scores = candidate_teacher_scores.to(
                device=prefix_scores.device,
                dtype=prefix_scores.dtype,
            )
            alpha = torch.tensor(
                ripor_prefix_alphas(full_rank_depth, alpha_beta)[: prefix_scores.shape[-1]],
                device=prefix_scores.device,
                dtype=prefix_scores.dtype,
            )
            loss_sum = prefix_scores.new_tensor(0.0)
            pair_count = prefix_scores.new_tensor(0.0)
            for row_index in range(prefix_scores.shape[0]):
                valid = candidate_valid[row_index]
                positive = valid & candidate_positive[row_index]
                negative = valid & ~candidate_positive[row_index]
                if not bool(positive.any()) or not bool(negative.any()):
                    continue
                positive_scores = prefix_scores[row_index, positive]
                negative_scores = prefix_scores[row_index, negative]
                positive_codes = candidate_codes[row_index, positive]
                negative_codes = candidate_codes[row_index, negative]
                diverged = (positive_codes[:, None, :] != negative_codes[None, :, :]).cumsum(dim=-1) > 0
                teacher_positive_scores = candidate_teacher_scores[row_index, positive]
                teacher_negative_scores = candidate_teacher_scores[row_index, negative]
                teacher_margin = teacher_positive_scores[:, None] - teacher_negative_scores[None, :]
                student_margin = positive_scores[:, None, :] - negative_scores[None, :, :]
                pair_losses = (
                    student_margin
                    - teacher_margin[:, :, None] * alpha.view(1, 1, -1)
                ).square()
                loss_sum = loss_sum + (pair_losses * diverged).sum()
                pair_count = pair_count + diverged.sum()
            return loss_sum, pair_count, prefix_scores

        def forward(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            target_types,
            labels,
            ripor_candidate_codes=None,
            ripor_candidate_valid=None,
            ripor_candidate_positive=None,
            ripor_candidate_teacher_scores=None,
            ripor_candidate_teacher_observed=None,
            ripor_loss_weight: float = 0.0,
            ripor_margin: float = 1.0,
            ripor_temperature: float = 1.0,
            ripor_alpha_beta: float = DEFAULT_RIPOR_ALPHA_BETA,
            ripor_prefix_max_length: int | None = None,
            pag_query_prefix: bool = False,
            sequential_ce_weight: float = 1.0,
            pag_set_targets=None,
            pag_set_candidate_token_ids=None,
            pag_set_candidate_token_mask=None,
            pag_set_candidate_valid=None,
            pag_set_candidate_positive=None,
            pag_set_candidate_teacher_scores=None,
            pag_set_loss_weight: float = 0.0,
            pag_set_sparsity_weight: float = 0.0,
            global_loss_normalization: bool = True,
            return_components: bool = False,
        ):
            encoder_hidden_states = self.encode_source(input_ids, attention_mask)
            hidden = self._decoder_hidden_from_encoder(
                encoder_hidden_states,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                decoder_prefix_input_ids=input_ids if pag_query_prefix else None,
                decoder_prefix_attention_mask=attention_mask if pag_query_prefix else None,
            )
            loss_sum = hidden.new_tensor(0.0)
            token_count = hidden.new_tensor(0.0)
            for level, head in enumerate(self.code_heads):
                mask = target_types == level
                if mask.any():
                    loss_sum = loss_sum + F.cross_entropy(head(hidden[mask]), labels[mask], reduction="sum")
                    token_count = token_count + mask.sum()
            if self.dedup_head is not None:
                dedup_mask = target_types == self.depth
                if dedup_mask.any():
                    loss_sum = loss_sum + F.cross_entropy(
                        self.dedup_head(hidden[dedup_mask]),
                        labels[dedup_mask],
                        reduction="sum",
                    )
                    token_count = token_count + dedup_mask.sum()
            control_mask = target_types == self.control_type
            if control_mask.any():
                loss_sum = loss_sum + F.cross_entropy(self.control_head(hidden[control_mask]), labels[control_mask], reduction="sum")
                token_count = token_count + control_mask.sum()
            ce_loss = self._normalize_loss(loss_sum, token_count, global_loss_normalization)
            ripor_loss_sum = hidden.new_tensor(0.0)
            ripor_pair_count = hidden.new_tensor(0.0)
            ripor_loss = hidden.new_tensor(0.0)
            if float(ripor_loss_weight) != 0.0:
                if (
                    ripor_candidate_codes is None
                    or ripor_candidate_valid is None
                    or ripor_candidate_positive is None
                ):
                    raise ValueError("RIPOR loss requires candidate codes, validity, and positive masks")
                if ripor_candidate_teacher_scores is None:
                    # Backward-compatible explicit binary weak-teacher fallback
                    # for callers that construct candidates manually.  The
                    # training path always supplies frozen ranked scores.
                    ripor_candidate_teacher_scores = ripor_candidate_positive.to(dtype=encoder_hidden_states.dtype)
                ripor_loss_sum, ripor_pair_count, _ = self.ripor_prefix_rank_loss(
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    candidate_codes=ripor_candidate_codes,
                    candidate_valid=ripor_candidate_valid,
                    candidate_positive=ripor_candidate_positive,
                    candidate_teacher_scores=ripor_candidate_teacher_scores,
                    margin=ripor_margin,
                    temperature=ripor_temperature,
                    alpha_beta=ripor_alpha_beta,
                    prefix_max_length=ripor_prefix_max_length,
                    decoder_prefix_input_ids=input_ids if pag_query_prefix else None,
                    decoder_prefix_attention_mask=attention_mask if pag_query_prefix else None,
                )
                ripor_loss = self._normalize_loss(
                    ripor_loss_sum,
                    ripor_pair_count,
                    global_loss_normalization,
                )
            loss = float(sequential_ce_weight) * ce_loss + float(ripor_loss_weight) * ripor_loss
            pag_set_loss_sum = hidden.new_tensor(0.0)
            pag_set_count = hidden.new_tensor(0.0)
            pag_set_loss = hidden.new_tensor(0.0)
            if float(pag_set_loss_weight) != 0.0:
                if pag_set_candidate_token_ids is not None:
                    if (
                        pag_set_candidate_token_mask is None
                        or pag_set_candidate_valid is None
                        or pag_set_candidate_positive is None
                        or pag_set_candidate_teacher_scores is None
                    ):
                        raise ValueError("PAG MarginMSE requires candidate token masks, validity, labels, and teacher scores")
                    set_logits = self.pag_set_logits_from_encoder(
                        encoder_hidden_states,
                        attention_mask,
                        source_input_ids=input_ids,
                    )
                    token_ids = pag_set_candidate_token_ids.clamp_min(0).clamp_max(set_logits.shape[-1] - 1)
                    token_scores = torch.gather(
                        set_logits[:, None, :].expand(-1, token_ids.shape[1], -1),
                        2,
                        token_ids,
                    )
                    candidate_scores = (
                        token_scores * pag_set_candidate_token_mask.to(dtype=token_scores.dtype)
                    ).sum(dim=-1)
                    teacher_scores = pag_set_candidate_teacher_scores.to(
                        device=candidate_scores.device,
                        dtype=candidate_scores.dtype,
                    )
                    pag_set_loss_sum = hidden.new_tensor(0.0)
                    pag_set_count = hidden.new_tensor(0.0)
                    for row_index in range(candidate_scores.shape[0]):
                        valid = pag_set_candidate_valid[row_index]
                        positive = valid & pag_set_candidate_positive[row_index]
                        negative = valid & ~pag_set_candidate_positive[row_index]
                        if not bool(positive.any()) or not bool(negative.any()):
                            continue
                        student_margin = candidate_scores[row_index, positive][:, None] - candidate_scores[row_index, negative][None, :]
                        teacher_margin = teacher_scores[row_index, positive][:, None] - teacher_scores[row_index, negative][None, :]
                        pag_set_loss_sum = pag_set_loss_sum + (student_margin - teacher_margin).square().sum()
                        pag_set_count = pag_set_count + student_margin.numel()
                    pag_set_loss = self._normalize_loss(
                        pag_set_loss_sum,
                        pag_set_count,
                        global_loss_normalization,
                    )
                    if float(pag_set_sparsity_weight) != 0.0:
                        global_set_logits = all_gather_tensor_with_grad(
                            set_logits,
                            self.distributed_context or DistributedContext(0, 0, 1, set_logits.device),
                        )
                        pag_set_loss = pag_set_loss + float(pag_set_sparsity_weight) * torch.mean(
                            torch.abs(global_set_logits), dim=0
                        ).square().sum()
                elif pag_set_targets is not None:
                    # Kept only for an explicitly constructed ablation.  The
                    # reproduction path supplies teacher-ranked set candidates.
                    set_logits = self.pag_set_logits_from_encoder(
                        encoder_hidden_states,
                        attention_mask,
                        source_input_ids=input_ids,
                    )
                    if set_logits.shape != pag_set_targets.shape:
                        raise ValueError(
                            "PAG set targets must match the T5 lexical logits: "
                            f"{tuple(pag_set_targets.shape)} != {tuple(set_logits.shape)}"
                        )
                    pag_set_loss_sum = F.binary_cross_entropy(
                        torch.sigmoid(set_logits),
                        pag_set_targets.to(dtype=set_logits.dtype),
                        reduction="sum",
                    )
                    pag_set_count = hidden.new_tensor(float(pag_set_targets.numel()))
                    pag_set_loss = self._normalize_loss(
                        pag_set_loss_sum,
                        pag_set_count,
                        global_loss_normalization,
                    )
                else:
                    raise ValueError("PAG set loss requires teacher-ranked candidate token ids")
                loss = loss + float(pag_set_loss_weight) * pag_set_loss
            if return_components:
                return {
                    "loss": loss,
                    "ce_loss": ce_loss,
                    "ripor_loss": ripor_loss,
                    "local_ce_loss_sum": loss_sum,
                    "local_ce_token_count": token_count,
                    "local_ripor_loss_sum": ripor_loss_sum,
                    "local_ripor_pair_count": ripor_pair_count,
                    "pag_set_loss": pag_set_loss,
                    "local_pag_set_loss_sum": pag_set_loss_sum,
                    "local_pag_set_count": pag_set_count,
                }
            return loss

        @torch.no_grad()
        def logits_for_next_type(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            target_type: int,
            decoder_prefix_input_ids=None,
            decoder_prefix_attention_mask=None,
        ):
            hidden = self._decoder_hidden(
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                decoder_prefix_input_ids=decoder_prefix_input_ids,
                decoder_prefix_attention_mask=decoder_prefix_attention_mask,
            )
            return self.logits_for_hidden_type(hidden, target_type)

        @torch.no_grad()
        def logits_for_next_type_from_encoder(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            target_type: int,
            decoder_prefix_input_ids=None,
            decoder_prefix_attention_mask=None,
        ):
            hidden = self._decoder_hidden_from_encoder(
                encoder_hidden_states,
                encoder_attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                decoder_prefix_input_ids=decoder_prefix_input_ids,
                decoder_prefix_attention_mask=decoder_prefix_attention_mask,
            )
            return self.logits_for_hidden_type(hidden, target_type)

        def logits_for_hidden_type(self, hidden, target_type: int):
            last_hidden = hidden[:, -1, :]
            if target_type < self.depth:
                return self.code_heads[target_type](last_hidden)
            if self.dedup_head is not None and target_type == self.depth:
                return self.dedup_head(last_hidden)
            if target_type == self.control_type:
                return self.control_head(last_hidden)
            raise ValueError(f"invalid target type: {target_type}")

        def pag_set_logits_from_encoder(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            source_input_ids=None,
        ):
            if source_input_ids is None:
                decoder_start = self.backbone.config.decoder_start_token_id
                if decoder_start is None:
                    decoder_start = self.backbone.config.pad_token_id
                decoder_input_ids = encoder_hidden_states.new_full(
                    (encoder_hidden_states.shape[0], 1),
                    int(decoder_start),
                    dtype=torch.long,
                )
                decoder_attention_mask = torch.ones_like(decoder_input_ids)
            else:
                decoder_input_ids = source_input_ids
                decoder_attention_mask = encoder_attention_mask
            decoder_outputs = self.backbone.decoder(
                input_ids=decoder_input_ids,
                # The official T5-SPLADE collator supplies the query token
                # sequence as decoder_input_ids and leaves decoder self-attention
                # unmasked; the encoder attention mask is applied during the
                # lexical max-pool below.
                attention_mask=None if source_input_ids is not None else decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            decoder_hidden = decoder_outputs.last_hidden_state
            if getattr(self.backbone.config, "tie_word_embeddings", False):
                decoder_hidden = decoder_hidden * (float(self.backbone.config.d_model) ** -0.5)
            lexical_logits = self.backbone.lm_head(decoder_hidden)
            lexical_rep = torch.log1p(torch.relu(lexical_logits))
            if lexical_rep.shape[1] > 1:
                lexical_rep = lexical_rep * encoder_attention_mask.unsqueeze(-1).to(dtype=lexical_rep.dtype)
                lexical_rep = lexical_rep.max(dim=1).values
            else:
                lexical_rep = lexical_rep[:, 0, :]
            # Match ``T5Splade.encode``: lexical coordinates are non-negative
            # log(1+ReLU) activations, which are then summed over a set ID.
            return lexical_rep

    return StructuredT5SIDModel()


def structured_sid_batch(
    rows: list[dict[str, Any]],
    tokenizer,
    semantic_ids: dict[str, list[int]],
    codebook_size: int,
    depth: int,
    dedup_size: int,
    max_source_length: int,
    device,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
    pag_set_token_ids: dict[str, list[int]] | None = None,
    pag_set_vocab_size: int = 0,
) -> dict[str, Any]:
    import torch

    encoded_inputs = tokenizer(
        [decoder_source_text(row) for row in rows],
        padding=True,
        truncation=True,
        max_length=max_source_length,
        return_tensors="pt",
    )
    targets = [
        encode_structured_sid_target(
            [str(sid) for sid in row["selected_skill_ids"]],
            semantic_ids,
            codebook_size=codebook_size,
            depth=depth,
            dedup_size=dedup_size,
            semantic_id_mode=semantic_id_mode,
        )
        for row in rows
    ]
    max_len = max(len(target["labels"]) for target in targets)
    decoder_input_ids = []
    decoder_attention_mask = []
    target_types = []
    labels = []
    for target in targets:
        length = len(target["labels"])
        pad = max_len - length
        decoder_input_ids.append([*target["decoder_input_ids"], *([target["bos_id"]] * pad)])
        decoder_attention_mask.append([*(1 for _ in range(length)), *([0] * pad)])
        target_types.append([*target["types"], *([-100] * pad)])
        labels.append([*target["labels"], *([-100] * pad)])

    result = {
        "input_ids": encoded_inputs["input_ids"].to(device),
        "attention_mask": encoded_inputs["attention_mask"].to(device),
        "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long, device=device),
        "decoder_attention_mask": torch.tensor(decoder_attention_mask, dtype=torch.long, device=device),
        "target_types": torch.tensor(target_types, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }
    if pag_set_token_ids:
        if pag_set_vocab_size <= 0:
            raise ValueError("PAG set targets require a positive vocabulary size")
        target = torch.zeros((len(rows), pag_set_vocab_size), dtype=torch.float32, device=device)
        for row_index, row in enumerate(rows):
            selected = [
                str(sid)
                for sid in (
                    row.get("source_selected_skill_ids")
                    if row.get("source_selected_skill_ids") is not None
                    else row.get("selected_skill_ids", [])
                )
            ]
            token_ids = {
                int(token_id)
                for sid in selected
                for token_id in pag_set_token_ids.get(sid, [])
            }
            if not token_ids:
                raise ValueError(f"PAG set target is empty for row={row.get('index')}")
            target[row_index, torch.tensor(sorted(token_ids), dtype=torch.long, device=device)] = 1.0
        result["pag_set_targets"] = target
    return result


def _structured_global_train_metrics(components: dict[str, Any], context: DistributedContext) -> dict[str, float]:
    ce_sum, ce_count = reduce_sum_count(
        components["local_ce_loss_sum"], components["local_ce_token_count"], context
    )
    ripor_sum, ripor_count = reduce_sum_count(
        components["local_ripor_loss_sum"], components["local_ripor_pair_count"], context
    )
    pag_sum, pag_count = reduce_sum_count(
        components["local_pag_set_loss_sum"], components["local_pag_set_count"], context
    )
    return {
        "ce_loss": ce_sum / max(1, ce_count),
        "ripor_loss": ripor_sum / max(1, ripor_count),
        "ripor_pairs": float(ripor_count),
        "pag_set_loss": pag_sum / max(1, pag_count),
    }


def _distributed_structured_eval_loss(
    model,
    eval_rows: list[dict[str, Any]],
    tokenizer,
    semantic_ids: dict[str, list[int]],
    codebook_size: int,
    depth: int,
    dedup_size: int,
    max_source_length: int,
    batch_size: int,
    device,
    semantic_id_mode: str,
    context: DistributedContext,
    pag_query_prefix: bool = False,
) -> float:
    """Evaluate CE as a true global token mean without padded examples."""
    import torch

    core_model = unwrap_model(model)
    local_rows = [row for _, row in shard_with_indices(eval_rows, context)]
    core_model.eval()
    local_sum = torch.zeros((), device=device)
    local_count = torch.zeros((), device=device)
    with torch.no_grad():
        for start in range(0, len(local_rows), batch_size):
            batch = structured_sid_batch(
                local_rows[start : start + batch_size],
                tokenizer,
                semantic_ids,
                codebook_size,
                depth,
                dedup_size,
                max_source_length,
                device,
                semantic_id_mode,
            )
            components = core_model(
                **batch,
                pag_query_prefix=pag_query_prefix,
                global_loss_normalization=False,
                return_components=True,
            )
            local_sum = local_sum + components["local_ce_loss_sum"].detach()
            local_count = local_count + components["local_ce_token_count"].detach()
    total_sum, total_count = reduce_sum_count(local_sum, local_count, context)
    model.train()
    return total_sum / max(1, total_count)


def _run_distributed_structured_training(
    *,
    args: argparse.Namespace,
    model,
    optimizer,
    tokenizer,
    semantic_ids: dict[str, list[int]],
    semantic_id_mode: str,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    catalog_rows: list[dict[str, Any]],
    query_train_rows: list[dict[str, Any]],
    query_dev_rows: list[dict[str, Any]],
    context: DistributedContext,
    checkpoint_metadata: dict[str, Any],
    training_formulation: str,
    wandb_run,
    ripor_enabled: bool,
    ripor_hard_negative_map: dict[str, list[str]],
    ripor_teacher_score_map: dict[str, dict[str, float]],
    pag_set_token_ids: dict[str, list[int]] | None = None,
    pag_set_vocab_size: int = 0,
    pag_set_skills: list[dict[str, Any]] | None = None,
    pag_set_candidate_map: dict[str, list[str]] | None = None,
    pag_set_teacher_score_map: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Shared DDP-safe structured training loop for single and multi SID modes."""
    import torch

    device = context.device
    eval_batch_size = args.eval_batch_size or args.batch_size
    loss_history_path = args.output_dir / "loss_history.jsonl"

    def save_checkpoint(directory: Path, extra: dict[str, Any]) -> None:
        if context.is_main:
            save_structured_decoder_checkpoint(
                directory,
                unwrap_model(model),
                tokenizer,
                args.model,
                codebook_size,
                depth,
                dedup_size,
                args.seed,
                training_formulation,
                extra,
                optimizer=optimizer,
                semantic_id_mode=semantic_id_mode,
            )
        barrier(context)

    def train_stage(
        stage_name: str,
        stage_rows: list[dict[str, Any]],
        total_steps: int,
        eval_rows: list[dict[str, Any]] | None,
        *,
        use_ripor: bool,
        step_offset: int,
        initial_best_loss: float | None,
    ) -> dict[str, Any]:
        if total_steps < 0:
            raise ValueError(f"{stage_name} total steps must be non-negative, got {total_steps}")
        if total_steps == 0:
            return {
                "steps": 0,
                "last_train_loss": None,
                "last_ce_loss": None,
                "last_ripor_loss": None,
                "best_dev_loss": initial_best_loss,
                "best_checkpoint": str(args.init_model_dir) if initial_best_loss is not None and args.init_model_dir else None,
            }
        best_loss = initial_best_loss
        best_checkpoint = args.init_model_dir if best_loss is not None else None
        step = 0
        epoch_index = 0
        last_metrics: dict[str, float] | None = None
        steps_per_epoch = max(1, math.ceil(len(stage_rows) / (args.batch_size * context.world_size)))
        model.train()
        while step < total_steps:
            for batch_rows in distributed_epoch_batches(
                stage_rows,
                args.batch_size,
                context,
                seed=args.seed + (10_000 if stage_name == "catalog" else 0),
                epoch=epoch_index,
            ):
                if step >= total_steps:
                    break
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
                    pag_set_token_ids=(
                        pag_set_token_ids
                        if pag_set_token_ids and stage_name != "catalog" and not pag_set_candidate_map
                        else None
                    ),
                    pag_set_vocab_size=pag_set_vocab_size,
                )
                batch["pag_query_prefix"] = bool(pag_set_token_ids and stage_name != "catalog")
                batch["sequential_ce_weight"] = float(getattr(args, "sequential_ce_weight", 1.0))
                if use_ripor:
                    batch.update(
                        build_ripor_candidate_batch(
                            batch_rows,
                            semantic_ids,
                            depth=depth,
                            max_negatives=args.ripor_max_negatives,
                            hard_negative_map=ripor_hard_negative_map,
                            teacher_score_map=ripor_teacher_score_map,
                            seed=args.seed + step,
                            device=device,
                            require_teacher_scores=not args.ripor_allow_weak_teacher,
                        )
                    )
                    batch.update(
                        {
                            "ripor_loss_weight": args.ripor_loss_weight,
                            "ripor_margin": args.ripor_margin,
                            "ripor_temperature": args.ripor_temperature,
                            "ripor_alpha_beta": args.ripor_alpha_beta,
                            "ripor_prefix_max_length": (
                                max(
                                    1,
                                    min(
                                        depth,
                                        math.ceil((step + 1) * depth / max(1, total_steps)),
                                    ),
                                )
                                if args.ripor_progressive_prefix
                                else depth
                            ),
                        }
                    )
                if pag_set_token_ids and stage_name != "catalog":
                    batch["pag_set_loss_weight"] = args.pag_set_loss_weight
                    batch["pag_set_sparsity_weight"] = args.pag_set_sparsity_weight
                    if pag_set_candidate_map is not None:
                        if not pag_set_skills or pag_set_teacher_score_map is None:
                            raise ValueError("PAG set MarginMSE requires skill documents and teacher scores")
                        batch.update(
                            build_pag_set_candidate_batch(
                                batch_rows,
                                pag_set_skills,
                                pag_set_token_ids,
                                pag_set_candidate_map,
                                pag_set_teacher_score_map,
                                max_negatives=args.pag_set_max_negatives,
                                allow_weak_teacher=args.pag_set_allow_weak_teacher,
                                device=device,
                            )
                        )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=bool(args.bf16 and getattr(device, "type", "") == "cuda"),
                ):
                    components = model(**batch, return_components=True)
                    loss = components["loss"]
                loss.backward()
                optimizer.step()
                step += 1
                display_step = step + step_offset
                should_log = step % args.logging_steps == 0 or step == total_steps
                if should_log:
                    last_metrics = _structured_global_train_metrics(components, context)
                    pag_set_weight = float(getattr(args, "pag_set_loss_weight", 0.0))
                    ripor_weight = float(getattr(args, "ripor_loss_weight", 0.0))
                    ce_weight = float(getattr(args, "sequential_ce_weight", 1.0))
                    train_loss = ce_weight * last_metrics["ce_loss"] + (
                        ripor_weight * last_metrics["ripor_loss"] if use_ripor else 0.0
                    ) + pag_set_weight * last_metrics["pag_set_loss"]
                    if context.is_main:
                        print(f"{stage_name}_step={display_step} train_loss={train_loss:.6f}", flush=True)
                        event = {
                            "stage": stage_name,
                            "event": "train",
                            "step": display_step,
                            "epoch": display_step / steps_per_epoch,
                            "train_loss": train_loss,
                            "ce_loss": last_metrics["ce_loss"],
                            "sequential_ce_weight": ce_weight,
                            "ripor_loss": last_metrics["ripor_loss"] if use_ripor else None,
                            "ripor_pairs": int(last_metrics["ripor_pairs"]) if use_ripor else 0,
                            "pag_set_loss": last_metrics["pag_set_loss"] if pag_set_token_ids else None,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        append_loss_history(loss_history_path, event)
                        log_training_event(wandb_run, event)
                if eval_rows and args.eval_steps > 0 and (step % args.eval_steps == 0 or step == total_steps):
                    dev_loss = _distributed_structured_eval_loss(
                        model,
                        eval_rows,
                        tokenizer,
                        semantic_ids,
                        codebook_size,
                        depth,
                        dedup_size,
                        args.max_source_length,
                        eval_batch_size,
                        device,
                        semantic_id_mode,
                        context,
                        pag_query_prefix=bool(pag_set_token_ids and stage_name != "catalog"),
                    )
                    if context.is_main:
                        print(f"{stage_name}_step={display_step} dev_loss={dev_loss:.6f}", flush=True)
                        event = {
                            "stage": stage_name,
                            "event": "eval",
                            "step": display_step,
                            "epoch": display_step / steps_per_epoch,
                            "dev_loss": dev_loss,
                        }
                        append_loss_history(loss_history_path, event)
                        log_training_event(wandb_run, event)
                    if best_loss is None or dev_loss < best_loss:
                        best_loss = dev_loss
                        best_checkpoint = args.output_dir / "best"
                        save_checkpoint(
                            best_checkpoint,
                            {
                                **checkpoint_metadata,
                                "best_dev_loss": best_loss,
                                "best_step": display_step,
                            },
                        )
            epoch_index += 1
        if last_metrics is None:
            # This branch is only reachable when logging_steps is invalid; keep
            # an explicit error instead of silently writing rank-local metrics.
            raise RuntimeError("training stage completed without a globally reduced loss event")
        return {
            "steps": step,
            "last_train_loss": float(getattr(args, "sequential_ce_weight", 1.0)) * last_metrics["ce_loss"] + (
                float(getattr(args, "ripor_loss_weight", 0.0)) * last_metrics["ripor_loss"] if use_ripor else 0.0
            ) + float(getattr(args, "pag_set_loss_weight", 0.0)) * last_metrics["pag_set_loss"],
            "last_ce_loss": last_metrics["ce_loss"],
            "last_ripor_loss": last_metrics["ripor_loss"] if use_ripor else None,
            "best_dev_loss": best_loss,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        }

    catalog_total_steps = (
        args.catalog_max_steps
        if args.catalog_max_steps > 0
        else int(math.ceil(len(catalog_rows) / (args.batch_size * context.world_size)) * args.catalog_epochs)
    )
    catalog_result = train_stage(
        "catalog",
        catalog_rows,
        catalog_total_steps,
        None,
        use_ripor=False,
        step_offset=0,
        initial_best_loss=None,
    ) if catalog_total_steps > 0 else {
        "steps": 0,
        "last_train_loss": None,
        "last_ce_loss": None,
        "last_ripor_loss": None,
        "best_dev_loss": None,
        "best_checkpoint": None,
    }
    query_total_steps = args.max_steps if args.max_steps > 0 else int(
        math.ceil(len(query_train_rows) / (args.batch_size * context.world_size)) * args.epochs
    )
    query_result = train_stage(
        "query",
        query_train_rows,
        query_total_steps,
        query_dev_rows,
        use_ripor=ripor_enabled,
        step_offset=args.step_offset,
        initial_best_loss=args.initial_best_dev_loss,
    )
    last_dir = args.output_dir / "last"
    save_checkpoint(
        last_dir,
        {
            **checkpoint_metadata,
            "last_query_step": query_result["steps"] + args.step_offset,
            "last_train_loss": query_result["last_train_loss"],
            "last_ce_loss": query_result["last_ce_loss"],
            "last_ripor_loss": query_result["last_ripor_loss"],
        },
    )
    if query_result["best_checkpoint"] is not None and args.load_best_model_at_end:
        load_structured_decoder_checkpoint(
            unwrap_model(model),
            Path(str(query_result["best_checkpoint"])),
            device,
            optimizer=optimizer,
            lr=args.lr,
            semantic_id_mode=semantic_id_mode,
        )
    barrier(context)
    final_dir = args.output_dir / "final"
    save_checkpoint(
        final_dir,
        {
            **checkpoint_metadata,
            "selected_checkpoint": "best" if query_result["best_checkpoint"] and args.load_best_model_at_end else "last",
            "ddp_world_size": context.world_size,
            "per_device_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * context.world_size,
        },
    )
    return {
        "catalog": catalog_result,
        "query": query_result,
        "last_dir": last_dir,
        "final_dir": final_dir,
        "loss_history_path": loss_history_path,
        "catalog_total_steps": catalog_total_steps,
        "query_total_steps": query_total_steps,
    }


def _initialize_structured_training_model(
    args: argparse.Namespace,
    *,
    semantic_ids: dict[str, list[int]],
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str,
    context: DistributedContext,
):
    import torch

    core_model = make_structured_t5_sid_model(
        args.model,
        codebook_size,
        depth,
        dedup_size,
        semantic_id_mode,
        pag_set_vocab_size=int(getattr(args, "pag_set_vocab_size", 0) or 0),
        ripor_codebook_embeddings=getattr(args, "ripor_codebook_embeddings", None),
        ripor_relevance_checkpoint=getattr(args, "ripor_relevance_checkpoint", None),
    ).to(context.device)
    optimizer = torch.optim.AdamW(core_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loaded_optimizer_state = False
    init_semantic_ids_verified = False
    init_source_length_verified = False
    init_source_text_policy_verified = False
    merged_initialization = False
    relevance_initialization = False
    sparse_initialization = False
    if getattr(args, "ripor_relevance_checkpoint", None):
        if args.init_model_dir or getattr(args, "pag_set_init_model_dir", None) or getattr(args, "pag_seq_init_model_dir", None):
            raise ValueError(
                "RIPOR relevance M0 initialization is only applied to a fresh decoder; "
                "do not combine --ripor-relevance-checkpoint with a decoder checkpoint"
            )
        try:
            relevance_checkpoint = torch.load(
                args.ripor_relevance_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            relevance_checkpoint = torch.load(args.ripor_relevance_checkpoint, map_location="cpu")
        relevance_state = relevance_checkpoint.get("state_dict") if isinstance(relevance_checkpoint, dict) else None
        if not isinstance(relevance_state, dict):
            raise ValueError("--ripor-relevance-checkpoint must contain a state_dict")
        if not isinstance(relevance_checkpoint, dict) or relevance_checkpoint.get("artifact_format") != "ripor_m0_relevance_v1":
            raise ValueError(
                "--ripor-relevance-checkpoint must be the provenance-bearing M0 artifact "
                "from train-ripor-relevance-encoder"
            )
        m0_config_hash = relevance_checkpoint.get("model_config_sha256")
        if not m0_config_hash:
            raise ValueError("M0 relevance checkpoint is missing model_config_sha256 provenance")
        actual_config_hash = stable_payload_sha256(core_model.backbone.config.to_dict())
        if str(m0_config_hash) != actual_config_hash:
            raise ValueError(
                "M0/GR model configuration mismatch: "
                f"m0={m0_config_hash}, gr={actual_config_hash}"
            )
        expected_state_keys = relevance_checkpoint.get("state_keys")
        if not isinstance(expected_state_keys, list) or not expected_state_keys:
            raise ValueError("M0 relevance checkpoint is missing its complete state_keys schema")
        normalized_expected_keys = {
            str(key)[len("backbone.") :] if str(key).startswith("backbone.") else str(key)
            for key in expected_state_keys
        }
        actual_m0_state_keys = {
            str(key)[len("backbone.") :] if str(key).startswith("backbone.") else str(key)
            for key in relevance_state
        }
        canonical_m0_keys = {
            str(key)
            for key in core_model.backbone.state_dict()
            if not str(key).startswith("lm_head.")
        }
        if normalized_expected_keys != canonical_m0_keys:
            raise ValueError(
                "M0 relevance checkpoint state schema does not match the current T5 architecture: "
                f"missing={sorted(canonical_m0_keys - normalized_expected_keys)[:5]}, "
                f"extra={sorted(normalized_expected_keys - canonical_m0_keys)[:5]}"
            )
        if actual_m0_state_keys != canonical_m0_keys:
            raise ValueError(
                "M0 relevance checkpoint state_dict is not complete or contains unexpected keys: "
                f"missing={sorted(canonical_m0_keys - actual_m0_state_keys)[:5]}, "
                f"extra={sorted(actual_m0_state_keys - canonical_m0_keys)[:5]}"
            )
        if not relevance_checkpoint.get("decoder_length"):
            raise ValueError("M0 relevance checkpoint is missing decoder_length provenance")
        if args.ripor_codebook_embeddings:
            try:
                codebook_payload = torch.load(
                    args.ripor_codebook_embeddings,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                codebook_payload = torch.load(args.ripor_codebook_embeddings, map_location="cpu")
            if not isinstance(codebook_payload, dict):
                raise ValueError("RIPOR codebook artifact must be a provenance-bearing dictionary")
            if str(codebook_payload.get("source_relevance_model")) != str(relevance_checkpoint.get("model")):
                raise ValueError("RIPOR codebook/M0 model provenance mismatch")
            if str(codebook_payload.get("source_relevance_model_config_sha256")) != str(m0_config_hash):
                raise ValueError("RIPOR codebook was built from a different M0 model configuration")
            if int(codebook_payload.get("source_relevance_decoder_length", -1)) != int(relevance_checkpoint["decoder_length"]):
                raise ValueError("RIPOR codebook/M0 decoder_length provenance mismatch")
            codebook_state_keys = {
                str(key)[len("backbone.") :] if str(key).startswith("backbone.") else str(key)
                for key in (codebook_payload.get("source_relevance_state_keys") or [])
            }
            if codebook_state_keys != normalized_expected_keys:
                raise ValueError("RIPOR codebook/M0 state schema mismatch")
        backbone_state = core_model.backbone.state_dict()
        transferable = {}
        for key, value in relevance_state.items():
            normalized_key = key[len("backbone.") :] if key.startswith("backbone.") else key
            if normalized_key in backbone_state and tuple(value.shape) == tuple(backbone_state[normalized_key].shape):
                transferable[normalized_key] = value
        missing_m0_keys = sorted(normalized_expected_keys - set(transferable))
        if missing_m0_keys:
            raise ValueError(
                "RIPOR relevance M0 checkpoint is not fully compatible with GR; "
                f"missing_or_mismatched={missing_m0_keys[:5]}"
            )
        missing_keys, _ = core_model.backbone.load_state_dict(transferable, strict=False)
        relevance_initialization = True
    if getattr(args, "pag_sparse_checkpoint", None):
        if not getattr(args, "pag_set_stage_only", False):
            raise ValueError("--pag-sparse-checkpoint is only valid for the explicit PAG Msp→Mset stage")
        if args.init_model_dir or getattr(args, "pag_set_init_model_dir", None) or getattr(args, "pag_seq_init_model_dir", None):
            raise ValueError("PAG Msp→Mset initialization cannot be combined with decoder checkpoints")
        try:
            sparse_checkpoint = torch.load(
                args.pag_sparse_checkpoint / "pag_sparse_encoder.pt",
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            sparse_checkpoint = torch.load(
                args.pag_sparse_checkpoint / "pag_sparse_encoder.pt",
                map_location="cpu",
            )
        sparse_state = sparse_checkpoint.get("state_dict") if isinstance(sparse_checkpoint, dict) else None
        if not isinstance(sparse_state, dict):
            raise ValueError("--pag-sparse-checkpoint must contain pag_sparse_encoder.pt/state_dict")
        if int(sparse_checkpoint.get("vocab_size", 0) or 0) != int(getattr(args, "pag_set_vocab_size", 0) or 0):
            raise ValueError("PAG Msp vocabulary does not match the lexical set manifest vocabulary")
        backbone_state = core_model.backbone.state_dict()
        transferable = {
            key[len("backbone.") :]: value
            for key, value in sparse_state.items()
            if key.startswith("backbone.")
            and key[len("backbone.") :] in backbone_state
            and tuple(value.shape) == tuple(backbone_state[key[len("backbone.") :]].shape)
        }
        if not transferable:
            raise ValueError("PAG Msp checkpoint has no shape-compatible GR backbone parameters")
        core_model.backbone.load_state_dict(transferable, strict=False)
        sparse_initialization = True
    if getattr(args, "pag_set_init_model_dir", None) or getattr(args, "pag_seq_init_model_dir", None):
        if not (args.pag_set_init_model_dir and args.pag_seq_init_model_dir) or args.init_model_dir:
            raise ValueError(
                "PAG joint reproduction requires both --pag-set-init-model-dir and "
                "--pag-seq-init-model-dir, without --init-model-dir"
            )
        set_checkpoint = torch.load(
            args.pag_set_init_model_dir / "structured_decoder.pt",
            map_location=context.device,
        )
        seq_checkpoint = torch.load(
            args.pag_seq_init_model_dir / "structured_decoder.pt",
            map_location=context.device,
        )
        set_role = str(set_checkpoint.get("stage_role") or set_checkpoint.get("pag", {}).get("stage_role") or "")
        seq_role = str(seq_checkpoint.get("stage_role") or seq_checkpoint.get("pag", {}).get("stage_role") or "")
        if set_role != "pag_mset" or seq_role != "pag_mseq":
            raise ValueError(
                "PAG joint initialization requires provenance-checked Mset/Mseq checkpoints: "
                f"set_role={set_role!r}, seq_role={seq_role!r}"
            )
        if str(set_checkpoint.get("training_formulation")) != "pag_mset_t5_splade_margin_mse":
            raise ValueError("PAG Mset checkpoint does not carry the Msp→Mset training formulation")
        if str(seq_checkpoint.get("training_formulation")) != "ripor_mseq_prefix_margin_mse":
            raise ValueError("PAG Mseq checkpoint does not carry the RIPOR sequential training formulation")
        set_pag_meta = set_checkpoint.get("pag", {})
        if not set_pag_meta.get("pag_sparse_initialization"):
            raise ValueError("PAG Mset checkpoint lacks verified Msp initialization provenance")
        if not set_pag_meta.get("strict_teacher_scores"):
            raise ValueError("PAG Mset checkpoint was not trained with frozen numeric teacher scores")
        if float(set_pag_meta.get("set_loss_weight", 0.0)) <= 0.0:
            raise ValueError("PAG Mset checkpoint has no positive lexical set loss")
        if float(set_pag_meta.get("sequential_ce_weight", 1.0)) != 0.0:
            raise ValueError("PAG Mset checkpoint must disable the sequential CE ablation")
        seq_ripor = seq_checkpoint.get("ripor", {})
        if not (seq_ripor.get("initialized_from_relevance_m0") and seq_ripor.get("initialized_from_rq_centroids")):
            raise ValueError("PAG joint Mseq checkpoint lacks verified RIPOR M0/RQ centroid provenance")
        if not seq_ripor.get("enabled") or float(seq_ripor.get("loss_weight", 0.0)) <= 0.0:
            raise ValueError("PAG joint Mseq checkpoint has no positive RIPOR prefix objective")
        if not seq_ripor.get("strict_teacher_scores"):
            raise ValueError("PAG joint Mseq checkpoint was not trained with frozen numeric teacher scores")
        seq_pag_meta = seq_checkpoint.get("pag", {})
        if float(seq_pag_meta.get("sequential_ce_weight", 1.0)) != 0.0:
            raise ValueError("PAG joint Mseq checkpoint must disable the sequential CE ablation")
        for label, checkpoint in (("set", set_checkpoint), ("seq", seq_checkpoint)):
            expected_config = {
                "codebook_size": codebook_size,
                "depth": depth,
                "dedup_size": dedup_size,
            }
            mismatches = {
                key: (checkpoint.get(key), value)
                for key, value in expected_config.items()
                if checkpoint.get(key) is not None and int(checkpoint[key]) != int(value)
            }
            checkpoint_pag_vocab = int(checkpoint.get("pag_set_vocab_size", 0) or 0)
            expected_pag_vocab = int(getattr(args, "pag_set_vocab_size", 0) or 0)
            if checkpoint_pag_vocab not in (0, expected_pag_vocab):
                mismatches["pag_set_vocab_size"] = (checkpoint_pag_vocab, expected_pag_vocab)
            if mismatches:
                raise ValueError(f"PAG {label} checkpoint configuration mismatch: {mismatches}")
            validate_checkpoint_semantic_ids(checkpoint, semantic_ids, semantic_id_mode=semantic_id_mode)
            validate_checkpoint_source_length(
                checkpoint,
                args.max_source_length,
                args.allow_checkpoint_config_mismatch,
            )
            validate_checkpoint_source_text_policy(checkpoint, args.allow_checkpoint_config_mismatch)
        set_state = set_checkpoint["state_dict"]
        seq_state = seq_checkpoint["state_dict"]
        merged_state: dict[str, Any] = {}
        sequential_only_prefixes = ("sid_embedding.", "code_heads.", "dedup_head.", "control_head.")
        for key, value in core_model.state_dict().items():
            if key not in set_state or key not in seq_state:
                continue
            left = set_state[key].to(device=context.device)
            right = seq_state[key].to(device=context.device)
            if key.startswith(sequential_only_prefixes):
                # Mset trains the lexical T5 backbone; Mseq owns the
                # position-specific semantic-ID input/output tables.  Averaging
                # Mset's unused/random semantic heads would destroy the RIPOR
                # initialization, so keep the sequential branch from Mseq.
                merged_state[key] = right
            elif torch.is_floating_point(left):
                merged_state[key] = (left + right) / 2.0
            else:
                merged_state[key] = left
        missing_keys, unexpected_keys = core_model.load_state_dict(merged_state, strict=False)
        optional_prefixes = ("ripor_doc_embeddings.", "pag_set_head.")
        missing_required = [key for key in missing_keys if not key.startswith(optional_prefixes)]
        if missing_required or unexpected_keys:
            raise ValueError(
                "PAG set/sequence checkpoint merge mismatch: "
                f"missing={missing_required[:5]}, unexpected={unexpected_keys[:5]}"
            )
        merged_initialization = True
        init_semantic_ids_verified = True
        init_source_length_verified = True
        init_source_text_policy_verified = True
    elif args.init_model_dir:
        init_checkpoint = load_structured_decoder_checkpoint(
            core_model,
            args.init_model_dir,
            context.device,
            optimizer=optimizer,
            lr=args.lr,
            semantic_id_mode=semantic_id_mode,
        )
        loaded_optimizer_state = "optimizer_state_dict" in init_checkpoint
        init_semantic_ids_verified = validate_checkpoint_semantic_ids(
            init_checkpoint,
            semantic_ids,
            semantic_id_mode=semantic_id_mode,
        )
        init_source_length_verified = validate_checkpoint_source_length(
            init_checkpoint,
            args.max_source_length,
            args.allow_checkpoint_config_mismatch,
        )
        init_source_text_policy_verified = validate_checkpoint_source_text_policy(
            init_checkpoint,
            args.allow_checkpoint_config_mismatch,
        )
        if (
            float(getattr(args, "ripor_loss_weight", 0.0)) > 0
            and not getattr(args, "ripor_allow_weak_teacher", False)
        ):
            ripor_meta = init_checkpoint.get("ripor", {})
            if not (ripor_meta.get("initialized_from_relevance_m0") and ripor_meta.get("initialized_from_rq_centroids")):
                raise ValueError(
                    "strict RIPOR continuation requires an init checkpoint carrying verified M0/RQ centroid provenance"
                )
    core_model.configure_distributed(context)
    return core_model, optimizer, {
        "loaded_optimizer_state": loaded_optimizer_state,
        "init_semantic_ids_verified": init_semantic_ids_verified,
        "init_source_length_verified": init_source_length_verified,
        "init_source_text_policy_verified": init_source_text_policy_verified,
        "merged_pag_initialization": merged_initialization,
        "ripor_relevance_initialization": relevance_initialization,
        "ripor_codebook_initialization": bool(getattr(args, "ripor_codebook_embeddings", None)),
        "pag_sparse_initialization": sparse_initialization,
    }


def command_train_single_sid_decoder(args: argparse.Namespace) -> None:
    """Train the single-SID GR with the RIPOR prefix-ranking objective."""
    import torch
    from transformers import AutoTokenizer

    context = initialize_distributed(cpu=args.cpu)
    if args.ripor_loss_weight > 0 and args.ripor_hard_negatives is None and not args.ripor_allow_weak_teacher:
        raise ValueError(
            "strict RIPOR reproduction requires --ripor-hard-negatives with frozen teacher scores; "
            "pass --ripor-allow-weak-teacher only for an explicitly labeled ablation"
        )
    if args.ripor_loss_weight > 0 and not args.ripor_allow_weak_teacher and not args.init_model_dir:
        if not args.ripor_relevance_checkpoint or not args.ripor_codebook_embeddings:
            raise ValueError(
                "strict RIPOR reproduction from a fresh decoder requires both "
                "--ripor-relevance-checkpoint (M0) and --ripor-codebook-embeddings (RQ centroids)"
            )
    torch.manual_seed(args.seed + context.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + context.rank)
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    eval_batch_size = args.eval_batch_size or args.batch_size
    sid_config = infer_structured_sid_config(
        semantic_ids,
        codebook_size=args.codebook_size,
        dedup_size=args.dedup_size,
        semantic_id_mode=semantic_id_mode,
    )
    codebook_size = sid_config["codebook_size"]
    depth = sid_config["depth"]
    dedup_size = sid_config["dedup_size"]
    query_rows_source = load_jsonl(args.rows)
    if args.dev_rows:
        query_train_rows = expand_single_sid_query_rows(query_rows_source, semantic_ids)
        query_dev_rows = expand_single_sid_query_rows(load_jsonl(args.dev_rows), semantic_ids)
    else:
        query_train_rows, query_dev_rows = split_and_expand_single_sid_query_rows(
            query_rows_source,
            semantic_ids,
            args.dev_ratio,
            args.seed,
        )
    catalog_rows = build_single_sid_catalog_rows(
        load_jsonl(args.skills_jsonl), semantic_ids, args.catalog_input_prefix
    )
    random.Random(args.seed).shuffle(catalog_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.pag_set_vocab_size = 0
    if context.is_main:
        query_train_length_audit = audit_source_token_lengths(
            "query_train", query_train_rows, tokenizer, args.max_source_length, args.allow_source_truncation
        )
        query_dev_length_audit = audit_source_token_lengths(
            "query_dev", query_dev_rows, tokenizer, args.max_source_length, args.allow_source_truncation
        )
        catalog_length_audit = audit_source_token_lengths(
            "catalog", catalog_rows, tokenizer, args.max_source_length, args.allow_source_truncation
        )
    else:
        query_train_length_audit = query_dev_length_audit = catalog_length_audit = None
    barrier(context)
    core_model, optimizer, init_info = _initialize_structured_training_model(
        args,
        semantic_ids=semantic_ids,
        codebook_size=codebook_size,
        depth=depth,
        dedup_size=dedup_size,
        semantic_id_mode=semantic_id_mode,
        context=context,
    )
    model = wrap_ddp(core_model, context)
    checkpoint_metadata = decoder_checkpoint_metadata(args, semantic_ids, semantic_id_mode)
    checkpoint_metadata["ripor"] = {
        "enabled": bool(args.ripor_loss_weight > 0),
        "loss_weight": float(args.ripor_loss_weight),
        "max_negatives": int(args.ripor_max_negatives),
        "margin": float(args.ripor_margin),
        "temperature": float(args.ripor_temperature),
        "hard_negative_predictions": str(args.ripor_hard_negatives) if args.ripor_hard_negatives else None,
        "ranked_levels": int(depth),
        "dedup_suffix_ranked": False,
        "teacher_objective": "official_prefix_margin_mse_with_concave_alpha",
        "alpha_beta": float(args.ripor_alpha_beta),
        "multi_objective_prefix_supervision": True,
        "progressive_schedule": "linear_prefix_expansion_with_multi_objective_retention"
        if args.ripor_progressive_prefix
        else "disabled",
        "teacher_score_fallback": "explicit_weak_ablation_only",
        "strict_teacher_scores": not args.ripor_allow_weak_teacher,
        "relevance_m0_checkpoint": str(args.ripor_relevance_checkpoint) if args.ripor_relevance_checkpoint else None,
        "rq_codebook_embeddings": str(args.ripor_codebook_embeddings) if args.ripor_codebook_embeddings else None,
        "initialized_from_relevance_m0": bool(init_info["ripor_relevance_initialization"]),
        "initialized_from_rq_centroids": bool(init_info["ripor_codebook_initialization"]),
    }
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier(context)
    loss_history_path = args.output_dir / "loss_history.jsonl"
    if context.is_main and loss_history_path.exists():
        loss_history_path.unlink()
    barrier(context)
    wandb_run = init_wandb_run(
        args,
        {
            "command": "train-single-sid-decoder",
            "model": args.model,
            "source_train_rows": len(query_rows_source),
            "query_train_rows": len(query_train_rows),
            "query_dev_rows": len(query_dev_rows),
            "catalog_rows": len(catalog_rows),
            "codebook_size": codebook_size,
            "depth": depth,
            "dedup_size": dedup_size,
            "semantic_id_mode": semantic_id_mode,
            "per_device_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * context.world_size,
            "ddp_world_size": context.world_size,
            "max_source_length": args.max_source_length,
            "source_text_policy": checkpoint_metadata["source_text_policy"],
            "semantic_ids_sha256": checkpoint_metadata["semantic_ids_sha256"],
            "allow_source_truncation": args.allow_source_truncation,
            "ripor": checkpoint_metadata["ripor"],
            "query_train_length_audit": query_train_length_audit,
            "query_dev_length_audit": query_dev_length_audit,
            "catalog_length_audit": catalog_length_audit,
            "max_steps": args.max_steps,
            "epochs": args.epochs,
            "catalog_epochs": args.catalog_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "init_model_dir": str(args.init_model_dir) if args.init_model_dir else None,
            **init_info,
        },
    ) if context.is_main else None
    ripor_hard_negative_map, ripor_teacher_score_map = load_ranked_skill_candidates_with_scores(
        args.ripor_hard_negatives
    )
    result = _run_distributed_structured_training(
        args=args,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        semantic_ids=semantic_ids,
        semantic_id_mode=semantic_id_mode,
        codebook_size=codebook_size,
        depth=depth,
        dedup_size=dedup_size,
        catalog_rows=catalog_rows,
        query_train_rows=query_train_rows,
        query_dev_rows=query_dev_rows,
        context=context,
        checkpoint_metadata=checkpoint_metadata,
        training_formulation="single_sid_catalog_indexing_with_official_ripor_prefix_margin_mse",
        wandb_run=wandb_run,
        ripor_enabled=args.ripor_loss_weight > 0,
        ripor_hard_negative_map=ripor_hard_negative_map,
        ripor_teacher_score_map=ripor_teacher_score_map,
    )
    if context.is_main:
        write_json(
            args.output_dir / "single_sid_decoder_train_manifest.json",
            {
                "model": args.model,
                "semantic_ids": str(args.semantic_ids),
                "skills_jsonl": str(args.skills_jsonl),
                "source_train_rows": len(query_rows_source),
                "query_train_rows": len(query_train_rows),
                "query_dev_rows": len(query_dev_rows),
                "catalog_rows": len(catalog_rows),
                "catalog_steps": result["catalog_total_steps"],
                "query_steps": result["query_total_steps"],
                "catalog_result": result["catalog"],
                "query_result": result["query"],
                "best_checkpoint": result["query"]["best_checkpoint"],
                "last_checkpoint": str(result["last_dir"]),
                "final_checkpoint": str(result["final_dir"]),
                "loss_history": str(result["loss_history_path"]),
                "codebook_size": codebook_size,
                "depth": depth,
                "dedup_size": dedup_size,
                "semantic_id_mode": semantic_id_mode,
                "per_device_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * context.world_size,
                "ddp_world_size": context.world_size,
                "eval_batch_size": eval_batch_size,
                "max_source_length": args.max_source_length,
                "source_text_policy": checkpoint_metadata["source_text_policy"],
                "semantic_ids_sha256": checkpoint_metadata["semantic_ids_sha256"],
                "allow_source_truncation": args.allow_source_truncation,
                "query_train_length_audit": query_train_length_audit,
                "query_dev_length_audit": query_dev_length_audit,
                "catalog_length_audit": catalog_length_audit,
                "ripor": checkpoint_metadata["ripor"],
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "init_model_dir": str(args.init_model_dir) if args.init_model_dir else None,
                **init_info,
                "load_best_model_at_end": bool(result["query"]["best_checkpoint"] and args.load_best_model_at_end),
            },
        )
        print(f"saved single-SID decoder to {result['final_dir']}")
    if wandb_run is not None:
        wandb_run.finish()
    barrier(context)
    destroy_distributed(context)


def command_train_structured_decoder(args: argparse.Namespace) -> None:
    """Train multi-SID set generation with DDP-safe global loss accounting."""
    import torch
    from transformers import AutoTokenizer

    context = initialize_distributed(cpu=args.cpu)
    torch.manual_seed(args.seed + context.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + context.rank)
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    eval_batch_size = args.eval_batch_size or args.batch_size
    sid_config = infer_structured_sid_config(
        semantic_ids,
        codebook_size=args.codebook_size,
        dedup_size=args.dedup_size,
        semantic_id_mode=semantic_id_mode,
    )
    codebook_size = sid_config["codebook_size"]
    depth = sid_config["depth"]
    dedup_size = sid_config["dedup_size"]
    rows = load_jsonl(args.rows)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.dev_rows:
        train_rows = rows
        dev_rows = load_jsonl(args.dev_rows)
    else:
        dev_size = max(1, math.ceil(len(rows) * args.dev_ratio)) if args.dev_ratio > 0 else 0
        dev_rows = rows[:dev_size]
        train_rows = rows[dev_size:]
    catalog_enabled = bool(args.skills_jsonl and args.catalog_max_steps != 0 and args.catalog_epochs != 0)
    catalog_rows = (
        build_single_sid_catalog_rows(load_jsonl(args.skills_jsonl), semantic_ids, args.catalog_input_prefix)
        if catalog_enabled
        else []
    )
    pag_set_skills = load_jsonl(args.skills_jsonl) if args.skills_jsonl else []
    random.Random(args.seed).shuffle(catalog_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pag_set_token_ids, pag_set_vocab_size = load_pag_set_token_ids(args.pag_set_ids)
    pag_set_candidate_map, pag_set_teacher_score_map = load_ranked_skill_candidates_with_scores(
        args.pag_set_hard_negatives
    )
    ripor_candidate_map, ripor_teacher_score_map = load_ranked_skill_candidates_with_scores(
        args.ripor_hard_negatives
    )
    if args.ripor_loss_weight > 0 and args.ripor_hard_negatives is None and not args.ripor_allow_weak_teacher:
        raise ValueError(
            "strict RIPOR sequential reproduction requires --ripor-hard-negatives with frozen teacher scores; "
            "pass --ripor-allow-weak-teacher only for an explicitly labeled ablation"
        )
    if (
        args.ripor_loss_weight > 0
        and not args.ripor_allow_weak_teacher
        and not args.init_model_dir
        and not args.pag_set_init_model_dir
        and not args.pag_seq_init_model_dir
    ):
        if not args.ripor_relevance_checkpoint or not args.ripor_codebook_embeddings:
            raise ValueError(
                "strict RIPOR reproduction from a fresh decoder requires both "
                "--ripor-relevance-checkpoint (M0) and --ripor-codebook-embeddings (RQ centroids)"
            )
    if pag_set_token_ids:
        if args.pag_stage_role == "pag_mseq":
            raise ValueError("PAG Mseq is the standalone sequential stage and must omit --pag-set-ids")
        missing_pag_ids = sorted(set(semantic_ids) - set(pag_set_token_ids))
        if missing_pag_ids:
            raise ValueError(
                f"PAG set manifest is missing {len(missing_pag_ids)} semantic skills; "
                f"first={missing_pag_ids[:5]}"
            )
        args.pag_set_vocab_size = pag_set_vocab_size
        if args.pag_set_loss_weight <= 0:
            raise ValueError("--pag-set-ids requires a positive --pag-set-loss-weight")
        if not pag_set_candidate_map and not args.pag_set_allow_weak_teacher:
            raise ValueError(
                "strict PAG reproduction requires --pag-set-hard-negatives with frozen teacher scores; "
                "pass --pag-set-allow-weak-teacher only for an explicit ablation"
            )
        if args.pag_set_stage_only:
            if not args.pag_sparse_checkpoint:
                raise ValueError("PAG Msp→Mset stage requires --pag-sparse-checkpoint")
            if args.pag_set_init_model_dir or args.pag_seq_init_model_dir or args.init_model_dir:
                raise ValueError("PAG Msp→Mset stage starts from Msp and cannot take decoder checkpoints")
            if args.ripor_loss_weight != 0.0:
                raise ValueError("PAG Mset stage has no sequential RIPOR objective; set --ripor-loss-weight 0")
            if args.sequential_ce_weight != 0.0:
                raise ValueError("PAG Mset stage is Lset only; set --sequential-ce-weight 0")
        elif not args.pag_set_allow_weak_teacher:
            if args.ripor_loss_weight <= 0:
                raise ValueError(
                    "strict PAG joint reproduction requires the RIPOR prefix objective; "
                    "set --ripor-loss-weight > 0"
                )
            if not (args.pag_set_init_model_dir and args.pag_seq_init_model_dir):
                raise ValueError(
                    "strict PAG joint reproduction requires Mset and Mseq checkpoints; "
                    "pass --pag-set-init-model-dir and --pag-seq-init-model-dir"
                )
            if args.pag_stage_role != "pag_joint":
                raise ValueError("strict PAG joint reproduction requires --pag-stage-role pag_joint")
            if args.sequential_ce_weight != 0.0:
                raise ValueError("strict PAG objective is Lset + Lseq; set --sequential-ce-weight 0")
            if args.pag_sparse_checkpoint:
                raise ValueError("strict PAG joint stage consumes Mset/Mseq, not Msp; remove --pag-sparse-checkpoint")
    else:
        args.pag_set_vocab_size = 0
        if args.pag_set_stage_only or args.pag_sparse_checkpoint:
            raise ValueError("PAG Msp/Mset options require --pag-set-ids")
        if args.pag_stage_role == "pag_mseq" and args.pag_set_allow_weak_teacher:
            raise ValueError(
                "--pag-set-allow-weak-teacher is unrelated to the standalone Mseq stage; "
                "do not use it to bypass RIPOR provenance checks"
            )
        if args.pag_stage_role == "pag_mseq" and args.ripor_allow_weak_teacher:
            raise ValueError(
                "a weak RIPOR teacher cannot be labeled pag_mseq; use a baseline/ablation role instead"
            )
        if args.pag_stage_role == "pag_mseq":
            # Mseq is the standalone RIPOR sequential stage.  It must be
            # produced from the M0/RQ initialization with the prefix
            # MarginMSE objective; otherwise a CE-only or arbitrary decoder
            # checkpoint could be mislabeled as the provenance input to the
            # final PAG joint stage.
            if args.init_model_dir or args.pag_set_init_model_dir or args.pag_seq_init_model_dir:
                raise ValueError("strict PAG Mseq stage starts from M0/RQ and cannot take decoder checkpoints")
            if args.ripor_loss_weight <= 0:
                raise ValueError("strict PAG Mseq reproduction requires --ripor-loss-weight > 0")
            if args.sequential_ce_weight != 0.0:
                raise ValueError("strict PAG Mseq reproduction uses RIPOR prefix MarginMSE; set --sequential-ce-weight 0")
            if not args.ripor_relevance_checkpoint or not args.ripor_codebook_embeddings:
                raise ValueError(
                    "strict PAG Mseq reproduction requires both --ripor-relevance-checkpoint (M0) "
                    "and --ripor-codebook-embeddings (RQ centroids)"
                )
        elif args.pag_stage_role != "baseline":
            raise ValueError(
                f"strict structured training without PAG set IDs does not support stage role {args.pag_stage_role!r}"
            )
    if context.is_main:
        query_train_length_audit = audit_source_token_lengths(
            "query_train", train_rows, tokenizer, args.max_source_length, args.allow_source_truncation
        )
        query_dev_length_audit = audit_source_token_lengths(
            "query_dev", dev_rows, tokenizer, args.max_source_length, args.allow_source_truncation
        )
        catalog_length_audit = audit_source_token_lengths(
            "catalog", catalog_rows, tokenizer, args.max_source_length, args.allow_source_truncation
        )
    else:
        query_train_length_audit = query_dev_length_audit = catalog_length_audit = None
    barrier(context)
    core_model, optimizer, init_info = _initialize_structured_training_model(
        args,
        semantic_ids=semantic_ids,
        codebook_size=codebook_size,
        depth=depth,
        dedup_size=dedup_size,
        semantic_id_mode=semantic_id_mode,
        context=context,
    )
    model = wrap_ddp(core_model, context)
    checkpoint_metadata = decoder_checkpoint_metadata(args, semantic_ids, semantic_id_mode)
    checkpoint_metadata["pag"] = {
        "enabled": bool(pag_set_token_ids),
        "set_branch": "T5-SPLADE lexical logits over top-m set IDs",
        "joint_objective": (
            "lexical_set_margin_mse"
            if args.pag_set_stage_only
            else "ripor_prefix_margin_mse + lexical_set_margin_mse"
        ),
        "sequential_objective": (
            "RIPOR prefix MarginMSE" if args.ripor_loss_weight > 0 else "CE baseline"
        ),
        "sequential_loss_weight": float(args.ripor_loss_weight),
        "set_ids": str(args.pag_set_ids) if args.pag_set_ids else None,
        "set_vocab_size": int(pag_set_vocab_size),
        "set_loss_weight": float(args.pag_set_loss_weight),
        "set_sparsity_weight": float(args.pag_set_sparsity_weight),
        "hard_negative_predictions": str(args.pag_set_hard_negatives) if args.pag_set_hard_negatives else None,
        "strict_teacher_scores": not args.pag_set_allow_weak_teacher,
        "external_prediction_scores": False,
        "merged_set_sequential_initialization": bool(init_info["merged_pag_initialization"]),
        "stage_role": "pag_mset" if args.pag_set_stage_only else args.pag_stage_role,
        "sequential_ce_weight": float(args.sequential_ce_weight),
        "pag_sparse_initialization": bool(init_info["pag_sparse_initialization"]),
        "merge_policy": "average_t5_backbone_keep_mseq_semantic_heads",
    }
    checkpoint_metadata["stage_role"] = "pag_mset" if args.pag_set_stage_only else args.pag_stage_role
    checkpoint_metadata["ripor"] = {
        "enabled": bool(args.ripor_loss_weight > 0),
        "loss_weight": float(args.ripor_loss_weight),
        "teacher_objective": "official_prefix_margin_mse_raw_conditional_logits",
        "strict_teacher_scores": not args.ripor_allow_weak_teacher,
        "relevance_m0_checkpoint": str(args.ripor_relevance_checkpoint) if args.ripor_relevance_checkpoint else None,
        "rq_codebook_embeddings": str(args.ripor_codebook_embeddings) if args.ripor_codebook_embeddings else None,
        "initialized_from_relevance_m0": bool(init_info["ripor_relevance_initialization"]),
        "initialized_from_rq_centroids": bool(init_info["ripor_codebook_initialization"]),
    }
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier(context)
    loss_history_path = args.output_dir / "loss_history.jsonl"
    if context.is_main and loss_history_path.exists():
        loss_history_path.unlink()
    barrier(context)
    wandb_run = init_wandb_run(
        args,
        {
            "command": "train-structured-decoder",
            "model": args.model,
            "source_rows": len(rows),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "catalog_rows": len(catalog_rows),
            "pag": checkpoint_metadata["pag"],
            "catalog_enabled": catalog_enabled,
            "codebook_size": codebook_size,
            "depth": depth,
            "dedup_size": dedup_size,
            "semantic_id_mode": semantic_id_mode,
            "per_device_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * context.world_size,
            "ddp_world_size": context.world_size,
            "eval_batch_size": eval_batch_size,
            "max_source_length": args.max_source_length,
            "source_text_policy": checkpoint_metadata["source_text_policy"],
            "semantic_ids_sha256": checkpoint_metadata["semantic_ids_sha256"],
            "allow_source_truncation": args.allow_source_truncation,
            "query_train_length_audit": query_train_length_audit,
            "query_dev_length_audit": query_dev_length_audit,
            "catalog_length_audit": catalog_length_audit,
            "max_steps": args.max_steps,
            "epochs": args.epochs,
            "catalog_epochs": args.catalog_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "init_model_dir": str(args.init_model_dir) if args.init_model_dir else None,
            **init_info,
        },
    ) if context.is_main else None
    result = _run_distributed_structured_training(
        args=args,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        semantic_ids=semantic_ids,
        semantic_id_mode=semantic_id_mode,
        codebook_size=codebook_size,
        depth=depth,
        dedup_size=dedup_size,
        catalog_rows=catalog_rows,
        query_train_rows=train_rows,
        query_dev_rows=dev_rows,
        context=context,
        checkpoint_metadata=checkpoint_metadata,
        training_formulation=(
            "pag_mset_t5_splade_margin_mse" if args.pag_set_stage_only
            else (
                "ripor_mseq_prefix_margin_mse"
                if args.pag_stage_role == "pag_mseq"
                else "multi_sid_pag_lexical_set_ripor_joint"
            )
        ),
        wandb_run=wandb_run,
        ripor_enabled=args.ripor_loss_weight > 0,
        ripor_hard_negative_map=ripor_candidate_map,
        ripor_teacher_score_map=ripor_teacher_score_map,
        pag_set_token_ids=pag_set_token_ids,
        pag_set_vocab_size=pag_set_vocab_size,
        pag_set_skills=pag_set_skills,
        pag_set_candidate_map=pag_set_candidate_map if pag_set_token_ids else None,
        pag_set_teacher_score_map=pag_set_teacher_score_map if pag_set_token_ids else None,
    )
    if context.is_main:
        write_json(
            args.output_dir / "structured_decoder_train_manifest.json",
            {
                "model": args.model,
                "semantic_ids": str(args.semantic_ids),
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "skills_jsonl": str(args.skills_jsonl) if args.skills_jsonl else None,
                "catalog_rows": len(catalog_rows),
                "catalog_steps": result["catalog_total_steps"],
                "query_steps": result["query_total_steps"],
                "catalog_result": result["catalog"],
                "query_result": result["query"],
                "best_checkpoint": result["query"]["best_checkpoint"],
                "last_checkpoint": str(result["last_dir"]),
                "final_checkpoint": str(result["final_dir"]),
                "loss_history": str(result["loss_history_path"]),
                "codebook_size": codebook_size,
                "depth": depth,
                "dedup_size": dedup_size,
                "semantic_id_mode": semantic_id_mode,
                "per_device_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * context.world_size,
                "ddp_world_size": context.world_size,
                "eval_batch_size": eval_batch_size,
                "max_source_length": args.max_source_length,
                "allow_source_truncation": args.allow_source_truncation,
                "query_train_length_audit": query_train_length_audit,
                "query_dev_length_audit": query_dev_length_audit,
                "catalog_length_audit": catalog_length_audit,
                "semantic_ids_sha256": checkpoint_metadata["semantic_ids_sha256"],
                "semantic_id_count": checkpoint_metadata["semantic_id_count"],
                "source_text_policy": checkpoint_metadata["source_text_policy"],
                "pag": checkpoint_metadata["pag"],
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                **init_info,
                "load_best_model_at_end": bool(result["query"]["best_checkpoint"] and args.load_best_model_at_end),
            },
        )
        print(f"saved structured decoder to {result['final_dir']}")
    if wandb_run is not None:
        wandb_run.finish()
    barrier(context)
    destroy_distributed(context)


def _semantic_id_trie_from_ids(
    semantic_ids: dict[str, list[int]],
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> tuple[dict[str, Any], dict[tuple[int, ...], str]]:
    validate_semantic_ids(semantic_ids, semantic_id_mode)
    rows = sorted((sid, [int(v) for v in values]) for sid, values in semantic_ids.items())
    return build_prefix_trie([values for _, values in rows]), {tuple(values): sid for sid, values in rows}


def _trie_node_for_prefix(trie: dict[str, Any], prefix: list[int]) -> dict[str, Any] | None:
    node = trie
    for value in prefix:
        key = str(int(value))
        if key not in node:
            return None
        node = node[key]
    return node


def structured_allowed_values(
    current_prefix: list[int],
    completed: list[tuple[int, ...]],
    target_type: int,
    trie: dict[str, Any],
    depth: int,
    max_skills: int,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
) -> list[int]:
    completed_set = set(completed)
    semantic_length = structured_sid_length(depth, semantic_id_mode)
    if target_type < semantic_length:
        node = _trie_node_for_prefix(trie, current_prefix)
        if node is None:
            return []
        return sorted(
            int(key)
            for key, child in node.items()
            if key != TRIE_TERMINAL_KEY and _trie_has_unseen_terminal(child, [*current_prefix, int(key)], completed_set)
        )
    if target_type == semantic_length:
        node = _trie_node_for_prefix(trie, current_prefix)
        if node is None or TRIE_TERMINAL_KEY not in node or tuple(current_prefix) in completed_set:
            return []
        allowed = [1]
        if len(completed) + 1 < max_skills:
            allowed.append(0)
        return allowed
    raise ValueError(f"invalid target type: {target_type}")


def advance_structured_beam(
    beam: dict[str, Any],
    target_type: int,
    value: int,
    log_prob: float,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
    planner_increment: float = 0.0,
    planner_weight: float = 0.0,
    planner_absolute_score: float | None = None,
) -> dict[str, Any]:
    new_prefix = list(beam["current_prefix"])
    new_completed = list(beam["completed"])
    new_completed_records = list(beam.get("completed_records", []))
    new_stopped = False
    semantic_length = structured_sid_length(depth, semantic_id_mode)
    if target_type < semantic_length:
        new_prefix.append(int(value))
        next_global_id = global_sid_token_id(
            target_type,
            int(value),
            codebook_size,
            depth,
            dedup_size,
            semantic_id_mode,
        )
    elif target_type == semantic_length:
        semantic_tuple = tuple(new_prefix)
        new_completed.append(semantic_tuple)
        new_prefix = []
        new_stopped = int(value) == 1
        next_global_id = global_sid_token_id(
            semantic_length,
            int(value),
            codebook_size,
            depth,
            dedup_size,
            semantic_id_mode,
        )
    else:
        raise ValueError(f"invalid target type: {target_type}")
    model_score = float(beam.get("model_score", beam["score"])) + float(log_prob)
    if planner_absolute_score is None:
        planner_score = float(beam.get("planner_score", 0.0)) + float(planner_increment)
    else:
        planner_score = float(planner_absolute_score)
    combined_score = model_score + float(planner_weight) * planner_score
    if target_type == semantic_length:
        completion_decoder_length = len(beam["decoder_input_ids"]) + 1
        new_completed_records.append(
            {
                "semantic_id": tuple(new_completed[-1]),
                "model_score": model_score,
                "planner_score": planner_score,
                "score": combined_score,
                "decoder_length": completion_decoder_length,
            }
        )
    return {
        "decoder_input_ids": [*beam["decoder_input_ids"], next_global_id],
        "current_prefix": new_prefix,
        "completed": new_completed,
        "completed_records": new_completed_records,
        "model_score": model_score,
        "planner_score": planner_score,
        "score": combined_score,
        "stopped": new_stopped,
    }


def ranked_skill_ids_from_structured_beams(
    beams: list[dict[str, Any]],
    sequence_to_skill_id: dict[tuple[int, ...], str],
    max_rank: int,
) -> list[str]:
    # A beam's final score includes tokens generated after an earlier SID was
    # completed.  Rank multi-SID outputs by each completion snapshot globally,
    # not by the eventual score of the containing beam.
    completion_candidates: list[tuple[float, int, int, str]] = []
    for beam_index, beam in enumerate(beams):
        records = beam.get("completed_records")
        if records is None:
            records = [{"semantic_id": tuple(values), "score": beam["score"]} for values in beam["completed"]]
        for record_index, record in enumerate(records):
            sid = sequence_to_skill_id.get(tuple(record["semantic_id"]))
            if sid is not None:
                completion_candidates.append(
                    (float(record.get("score", beam["score"])), beam_index, record_index, sid)
                )
    completion_candidates.sort(key=lambda item: item[0], reverse=True)
    ranked: list[str] = []
    seen: set[str] = set()
    for _, _, _, sid in completion_candidates:
        if sid in seen:
            continue
        ranked.append(sid)
        seen.add(sid)
        if len(ranked) >= max_rank:
            return ranked
    return ranked


def _logsumexp_pair(left: float | None, right: float) -> float:
    if left is None:
        return float(right)
    high = max(float(left), float(right))
    low = min(float(left), float(right))
    return high + math.log1p(math.exp(low - high))


def ranked_skill_ids_from_marginal_structured_beams(
    beams: list[dict[str, Any]],
    sequence_to_skill_id: dict[tuple[int, ...], str],
    max_rank: int,
    *,
    length_normalize: bool = False,
) -> tuple[list[str], dict[str, float]]:
    skill_scores: dict[str, float] = {}
    for beam in beams:
        records = beam.get("completed_records")
        if records is None:
            records = [{"semantic_id": tuple(values), "score": beam["score"]} for values in beam["completed"]]
        for record in records:
            score = float(record.get("score", beam["score"]))
            if length_normalize:
                completion_length = int(
                    record.get("decoder_length", len(beam.get("decoder_input_ids", [])))
                )
                score = score / max(1, completion_length - 1)
            values = tuple(record["semantic_id"])
            sid = sequence_to_skill_id.get(values)
            if sid is None:
                continue
            skill_scores[sid] = _logsumexp_pair(skill_scores.get(sid), score)
    ranked = [
        sid
        for sid, _ in sorted(skill_scores.items(), key=lambda item: item[1], reverse=True)[:max_rank]
    ]
    return ranked, {sid: skill_scores[sid] for sid in ranked}


def summarize_eval_gold_coverage(
    rows: list[dict[str, Any]],
    semantic_ids: dict[str, list[int]],
    max_skills: int,
) -> dict[str, Any]:
    known = set(semantic_ids)
    missing_mentions: list[str] = []
    rows_with_missing = 0
    target_counts: list[int] = []
    for row in rows:
        gold_ids = [str(sid) for sid in row.get("selected_skill_ids", [])]
        target_counts.append(len(set(gold_ids)))
        missing = [sid for sid in gold_ids if sid not in known]
        if missing:
            rows_with_missing += 1
            missing_mentions.extend(missing)
    return {
        "gold_skill_mentions": sum(target_counts),
        "unique_missing_gold_skill_ids": sorted(set(missing_mentions)),
        "missing_gold_mentions": len(missing_mentions),
        "rows_with_missing_gold": rows_with_missing,
        "max_gold_skills": max(target_counts, default=0),
        "rows_over_max_skills": sum(count > max_skills for count in target_counts),
    }


def validate_rank_capacity(num_beams: int, max_skills: int, max_rank: int) -> None:
    if num_beams <= 0:
        raise ValueError(f"num_beams must be positive, got {num_beams}")
    if max_skills <= 0:
        raise ValueError(f"max_skills must be positive, got {max_skills}")
    capacity = num_beams * max_skills
    if capacity < max_rank:
        raise ValueError(
            f"decoding can return at most {capacity} ranked skills with "
            f"num_beams={num_beams} and max_skills={max_skills}, but max K is {max_rank}"
        )


def command_eval_structured_decoder(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    context = initialize_distributed(cpu=args.cpu)
    checkpoint = torch.load(args.model_dir / "structured_decoder.pt", map_location="cpu")
    semantic_ids, semantic_id_mode = load_semantic_ids_with_mode(args.semantic_ids)
    semantic_ids_verified = validate_checkpoint_semantic_ids(
        checkpoint,
        semantic_ids,
        semantic_id_mode,
    )
    source_text_policy_verified = validate_checkpoint_source_text_policy(
        checkpoint,
        args.allow_checkpoint_config_mismatch,
    )
    args.max_source_length, max_source_length_source = resolve_eval_max_source_length(
        args.max_source_length,
        checkpoint,
        args.allow_checkpoint_config_mismatch,
    )
    codebook_size = int(checkpoint["codebook_size"])
    depth = int(checkpoint["depth"])
    dedup_size = int(checkpoint.get("dedup_size", 0))
    semantic_id_config = infer_structured_sid_config(
        semantic_ids,
        codebook_size=codebook_size,
        dedup_size=dedup_size,
        semantic_id_mode=semantic_id_mode,
    )
    if semantic_id_config["depth"] != depth:
        raise ValueError(
            f"semantic id depth {semantic_id_config['depth']} does not match checkpoint depth {depth}"
        )
    k_values = parse_k_values(args.k_values)
    max_rank = max(k_values)
    validate_rank_capacity(args.num_beams, args.max_skills, max_rank)
    trie, sequence_to_skill_id = _semantic_id_trie_from_ids(semantic_ids, semantic_id_mode)
    pag_set_weight = float(getattr(args, "pag_set_weight", 0.0))
    pag_set_token_ids, pag_set_vocab_size = load_pag_set_token_ids(getattr(args, "pag_set_ids", None))
    checkpoint_pag_vocab_size = int(checkpoint.get("pag_set_vocab_size", 0) or 0)
    if pag_set_weight != 0.0:
        if not pag_set_token_ids:
            raise ValueError("--pag-set-weight requires --pag-set-ids")
        if args.pag_set_top_n <= 0:
            raise ValueError(
                "PAG reproduction requires a fixed top-N lexical candidate domain; "
                "pass --pag-set-top-n (use --pag-set-weight 0 only for the same-domain baseline)"
            )
        missing_pag_ids = sorted(set(semantic_ids) - set(pag_set_token_ids))
        if missing_pag_ids:
            raise ValueError(
                f"PAG set manifest is missing {len(missing_pag_ids)} semantic skills; "
                f"first={missing_pag_ids[:5]}"
            )
        if checkpoint_pag_vocab_size <= 0 or checkpoint_pag_vocab_size != pag_set_vocab_size:
            raise ValueError(
                "PAG set manifest/checkpoint vocabulary mismatch: "
                f"checkpoint={checkpoint_pag_vocab_size}, manifest={pag_set_vocab_size}"
            )
    if args.pag_set_top_n > 0:
        if not pag_set_token_ids or checkpoint_pag_vocab_size <= 0 or checkpoint_pag_vocab_size != pag_set_vocab_size:
            raise ValueError(
                "--pag-set-top-n requires a checkpoint/manifest with the trained PAG set branch "
                "so the candidate domain is identical at weight=0 and weight>0"
            )
    # The external dense/BM25 planner is retained only as a separately named
    # legacy ablation.  It is never used by the PAG reproduction path.
    legacy_pag_planner_weight = float(getattr(args, "pag_planner_weight", 0.0))
    if legacy_pag_planner_weight != 0.0:
        raise ValueError(
            "external --pag-planner-predictions is not PAG; use --pag-set-ids and "
            "--pag-set-weight for the trained set branch"
        )
    prefix_descendants = build_semantic_prefix_descendants(semantic_ids, depth)
    device = context.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = make_structured_t5_sid_model(
        str(checkpoint["model"]),
        codebook_size,
        depth,
        dedup_size,
        semantic_id_mode,
        pag_set_vocab_size=checkpoint_pag_vocab_size,
    )
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
    optional_missing_prefixes = ("ripor_doc_embeddings.", "pag_set_head.")
    missing_required = [key for key in missing_keys if not key.startswith(optional_missing_prefixes)]
    unexpected_required = [key for key in unexpected_keys if not key.startswith(optional_missing_prefixes)]
    if missing_required or unexpected_required:
        raise ValueError(
            "structured decoder checkpoint parameter mismatch: "
            f"missing={missing_required[:5]}, unexpected={unexpected_required[:5]}"
        )
    model.to(device).eval()

    rows = load_jsonl(args.rows)
    local_indexed_rows = shard_with_indices(rows, context)
    if context.is_main:
        source_length_audit = audit_source_token_lengths(
            "eval",
            rows,
            tokenizer,
            args.max_source_length,
            args.allow_source_truncation,
        )
        gold_coverage = summarize_eval_gold_coverage(rows, semantic_ids, args.max_skills)
        print(json.dumps({"eval_gold_coverage": gold_coverage}, sort_keys=True), flush=True)
    else:
        source_length_audit = None
        gold_coverage = None
    predictions: list[dict[str, Any]] = []
    bos_id = structured_sid_bos_id(codebook_size, depth, dedup_size, semantic_id_mode)
    semantic_length = structured_sid_length(depth, semantic_id_mode)
    steps_per_skill = semantic_length + 1
    max_steps = args.max_skills * steps_per_skill
    for row_index, row in local_indexed_rows:
        row_planner_scores: dict[str, float] = {}
        encoded = tokenizer(
            [decoder_source_text(row)],
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            encoder_hidden_states = model.encode_source(input_ids, attention_mask)
            if pag_set_weight != 0.0 or args.pag_set_top_n > 0:
                set_logits = model.pag_set_logits_from_encoder(
                    encoder_hidden_states,
                    attention_mask,
                    source_input_ids=input_ids,
                )[0]
                for sid, token_ids in pag_set_token_ids.items():
                    valid_token_ids = [token_id for token_id in token_ids if token_id < set_logits.numel()]
                    if valid_token_ids:
                        row_planner_scores[sid] = float(set_logits[valid_token_ids].sum().cpu())
        decode_trie = trie
        planner_candidate_ids = set(semantic_ids)
        row_prefix_descendants = prefix_descendants
        if pag_set_token_ids and args.pag_set_top_n > 0:
            planner_candidate_ids = {
                sid
                for sid, _ in sorted(
                    row_planner_scores.items(), key=lambda item: item[1], reverse=True
                )[: args.pag_set_top_n]
                if sid in semantic_ids
            }
            if planner_candidate_ids:
                filtered_semantic_ids = {sid: semantic_ids[sid] for sid in planner_candidate_ids}
                decode_trie, _ = _semantic_id_trie_from_ids(
                    filtered_semantic_ids,
                    semantic_id_mode,
                )
                row_prefix_descendants = build_semantic_prefix_descendants(
                    filtered_semantic_ids,
                    depth,
                )
        beams = [
            {
                "decoder_input_ids": [bos_id],
                "current_prefix": [],
                "completed": [],
                "completed_records": [],
                "score": 0.0,
                "model_score": 0.0,
                "planner_score": 0.0,
                "stopped": False,
            }
        ]
        for step in range(max_steps):
            candidates = []
            target_type = step % steps_per_skill
            active_items = []
            for beam in beams:
                if beam["stopped"]:
                    candidates.append(beam)
                    continue
                allowed = structured_allowed_values(
                    beam["current_prefix"],
                    beam["completed"],
                    target_type,
                    decode_trie,
                    depth,
                    args.max_skills,
                    semantic_id_mode,
                )
                if not allowed:
                    candidates.append({**beam, "stopped": True})
                    continue
                active_items.append((beam, allowed))
            if active_items:
                max_decoder_length = max(len(beam["decoder_input_ids"]) for beam, _ in active_items)
                decoder_input_rows = []
                decoder_mask_rows = []
                for beam, _ in active_items:
                    decoder_ids = list(beam["decoder_input_ids"])
                    pad_length = max_decoder_length - len(decoder_ids)
                    decoder_input_rows.append(decoder_ids + [bos_id] * pad_length)
                    decoder_mask_rows.append([1] * len(decoder_ids) + [0] * pad_length)
                batch_size = len(active_items)
                decoder_input_ids = torch.tensor(decoder_input_rows, dtype=torch.long, device=device)
                decoder_attention_mask = torch.tensor(decoder_mask_rows, dtype=torch.long, device=device)
                logits_batch = model.logits_for_next_type_from_encoder(
                    encoder_hidden_states.repeat(batch_size, 1, 1),
                    attention_mask.repeat(batch_size, 1),
                    decoder_input_ids,
                    decoder_attention_mask,
                    target_type,
                    decoder_prefix_input_ids=input_ids if pag_set_token_ids else None,
                    decoder_prefix_attention_mask=attention_mask if pag_set_token_ids else None,
                )
                # RIPOR's constrained decoder ranks with the raw conditional
                # code logits, matching ``ripor_prefix_scores`` and the paper's
                # position-specific codebook dot products.
                raw_scores_batch = logits_batch
                for (beam, allowed), raw_scores in zip(active_items, raw_scores_batch):
                    allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
                    allowed_scores = raw_scores.index_select(0, allowed_tensor)
                    ranked_with_planner = []
                    for raw_score, value in zip(allowed_scores.detach().cpu().tolist(), allowed):
                        planner_absolute_score = float(beam.get("planner_score", 0.0))
                        if pag_set_weight != 0.0 and target_type < semantic_length:
                            completed_skill_ids = {
                                sequence_to_skill_id[tuple(values)]
                                for values in beam["completed"]
                                if tuple(values) in sequence_to_skill_id
                            }
                            planner_absolute_score = pag_prefix_planner_score(
                                [*beam["current_prefix"], int(value)],
                                row_planner_scores,
                                row_prefix_descendants,
                                args.pag_set_aggregation,
                                excluded_skill_ids=completed_skill_ids,
                            )
                        combined_score = float(raw_score) + pag_set_weight * planner_absolute_score
                        ranked_with_planner.append(
                            (combined_score, float(raw_score), int(value), planner_absolute_score)
                        )
                    ranked = sorted(ranked_with_planner, reverse=True)[: args.num_beams]
                    for _, log_prob, value, planner_absolute_score in ranked:
                        candidates.append(
                            advance_structured_beam(
                                beam,
                                target_type,
                                value,
                                log_prob,
                                codebook_size,
                                depth,
                                dedup_size,
                                semantic_id_mode,
                                planner_absolute_score=planner_absolute_score,
                                planner_weight=pag_set_weight,
                            )
                        )
            beams = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)[: args.num_beams]
            if all(beam["stopped"] for beam in beams):
                break
        best = max(beams, key=lambda item: float(item["score"]))
        ranked_skill_scores = None
        if args.rank_aggregation == "beam_order":
            ranked_skill_ids = ranked_skill_ids_from_structured_beams(beams, sequence_to_skill_id, max_rank)
        elif args.rank_aggregation == "marginal":
            ranked_skill_ids, ranked_skill_scores = ranked_skill_ids_from_marginal_structured_beams(
                beams,
                sequence_to_skill_id,
                max_rank,
                length_normalize=False,
            )
        elif args.rank_aggregation == "marginal_length_norm":
            ranked_skill_ids, ranked_skill_scores = ranked_skill_ids_from_marginal_structured_beams(
                beams,
                sequence_to_skill_id,
                max_rank,
                length_normalize=True,
            )
        else:
            raise ValueError(f"unknown rank aggregation: {args.rank_aggregation}")
        predicted_ids = [sequence_to_skill_id[tuple(values)] for values in best["completed"] if tuple(values) in sequence_to_skill_id]
        unknown_sequences = [list(values) for values in best["completed"] if tuple(values) not in sequence_to_skill_id]
        gold_ids = [str(sid) for sid in row.get("selected_skill_ids", [])]
        pred_set = set(predicted_ids)
        gold_set = set(gold_ids)
        hit_count = len(pred_set & gold_set)
        candidate_recall = (
            float(len(gold_set & planner_candidate_ids) / len(gold_set))
            if pag_set_token_ids and args.pag_set_top_n > 0 and gold_set
            else None
        )
        predictions.append(
            {
                "index": row.get("index"),
                "source_index": row.get("source_index"),
                "target_skill_count": row.get("target_skill_count"),
                "query": row.get("query"),
                "gold_skill_ids": gold_ids,
                "predicted_skill_ids": predicted_ids,
                "ranked_skill_ids": ranked_skill_ids,
                "ranked_skill_scores": ranked_skill_scores,
                "predicted_semantic_ids": [list(values) for values in best["completed"]],
                "decoder_input_sid_ids": best["decoder_input_ids"],
                "_row_position": row_index,
                "score": best["score"],
                "model_score": best.get("model_score", best["score"]),
                "planner_score": best.get("planner_score", 0.0),
                "hit_count": hit_count,
                "recall": float(hit_count / len(gold_set)) if gold_set else 0.0,
                "hit_rate": 1.0 if hit_count > 0 else 0.0,
                "exact_match": pred_set == gold_set,
                "count_match": len(pred_set) == len(gold_set),
                "pag_candidate_recall": candidate_recall,
                "invalid": bool(unknown_sequences),
                "unknown_semantic_ids": unknown_sequences,
            }
        )
        if context.is_main and (len(predictions) % args.log_every == 0 or len(predictions) == len(rows)):
            print(f"decoded {len(predictions)}/{len(rows)}", flush=True)

    gathered_predictions = all_gather_object(predictions, context)
    if not context.is_main:
        barrier(context)
        destroy_distributed(context)
        return
    predictions = [prediction for part in gathered_predictions for prediction in part]
    predictions.sort(key=lambda item: int(item.get("_row_position", 0)))
    metrics = compute_retrieval_metrics(
        rows,
        [{"predicted_skill_ids": pred["predicted_skill_ids"], "invalid": pred["invalid"]} for pred in predictions],
    )
    ranked_metrics = compute_at_k_metrics(rows, predictions, k_values)
    pag_candidate_recalls = [
        float(prediction["pag_candidate_recall"])
        for prediction in predictions
        if prediction.get("pag_candidate_recall") is not None
    ]
    ranked_lengths = [len(prediction["ranked_skill_ids"]) for prediction in predictions]
    ranked_length_audit = {
        "min_ranked_skills": min(ranked_lengths, default=0),
        "max_ranked_skills": max(ranked_lengths, default=0),
        "rows_below_max_rank": sum(length < max_rank for length in ranked_lengths),
    }
    for prediction in predictions:
        prediction.pop("_row_position", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "eval_metrics.json", metrics)
    write_json(args.output_dir / "ranked_metrics.json", ranked_metrics)
    write_json(
        args.output_dir / "eval_manifest.json",
        {
            "rows": str(args.rows),
            "model_dir": str(args.model_dir),
            "semantic_ids": str(args.semantic_ids),
            "batch_size": 1,
            "max_source_length": args.max_source_length,
            "max_source_length_source": max_source_length_source,
            "allow_source_truncation": args.allow_source_truncation,
            "source_length_audit": source_length_audit,
            "semantic_ids_verified": semantic_ids_verified,
            "semantic_ids_sha256": semantic_ids_sha256(semantic_ids, semantic_id_mode),
            "source_text_policy": GR_SOURCE_TEXT_POLICY,
            "source_text_policy_verified": source_text_policy_verified,
            "gold_coverage": gold_coverage,
            "ranked_length_audit": ranked_length_audit,
            "num_beams": args.num_beams,
            "max_skills": args.max_skills,
            "rank_aggregation": args.rank_aggregation,
            "pag_set_ids": str(args.pag_set_ids) if args.pag_set_ids else None,
            "pag_set_weight": pag_set_weight,
            "pag_set_top_n": args.pag_set_top_n,
            "pag_set_aggregation": args.pag_set_aggregation,
            "pag_candidate_recall": "reported below when top_n > 0",
            "pag_candidate_recall_mean": (
                sum(pag_candidate_recalls) / len(pag_candidate_recalls)
                if pag_candidate_recalls else None
            ),
            "k_values": k_values,
            "max_rank": max_rank,
            "codebook_size": codebook_size,
            "depth": depth,
            "dedup_size": dedup_size,
            "semantic_id_mode": semantic_id_mode,
            "distributed_world_size": context.world_size,
            "split": args.split_name,
            "metrics": "top-1 constrained structured SID generated set; hit_count=|pred∩gold|; recall=hit_count/|gold|",
        },
    )
    print(json.dumps({"top1_set": metrics["overall"], "ranked": ranked_metrics["overall"]}, ensure_ascii=False, sort_keys=True))
    barrier(context)
    destroy_distributed(context)


def command_eval_single_sid_decoder(args: argparse.Namespace) -> None:
    args.max_skills = 1
    if not hasattr(args, "rank_aggregation"):
        args.rank_aggregation = "beam_order"
    command_eval_structured_decoder(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_wandb_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--wandb-project")
        subparser.add_argument("--wandb-entity")
        subparser.add_argument("--wandb-run-name")
        subparser.add_argument("--wandb-group")
        subparser.add_argument("--wandb-tags", default="")
        subparser.add_argument("--wandb-mode", default="online")

    p = sub.add_parser("canonicalize-skill-aliases")
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--rows", type=Path, action="append", default=[])
    p.add_argument("--embeddings", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-unknown-row-skills", action="store_true")
    p.set_defaults(func=command_canonicalize_skill_aliases)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--skills", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--desc-max", type=int, default=300)
    p.add_argument("--body-max", type=int, default=2500)
    p.add_argument(
        "--query-max-chars",
        type=int,
        default=SKILLROUTER_QUERY_MAX_CHARS,
        help="character cap for the shared SkillRouter query view; 0 disables it",
    )
    p.set_defaults(func=command_prepare)

    p = sub.add_parser("build-pag-set-ids")
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="t5-base")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--sparse-checkpoint", type=Path)
    p.add_argument("--lexical-fallback", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.set_defaults(func=command_build_pag_set_ids)

    p = sub.add_parser("encode-skills")
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_encode)

    p = sub.add_parser("train-ripor-relevance-encoder")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument(
        "--hard-negatives",
        type=Path,
        required=True,
        help="frozen teacher-ranked candidate JSONL with scores; required for strict RIPOR reproduction",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="t5-base")
    p.add_argument("--max-query-length", type=int, default=256)
    p.add_argument("--max-skill-length", type=int, default=512)
    p.add_argument("--decoder-length", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-negatives", type=int, default=1)
    p.add_argument("--allow-weak-teacher", action="store_true")
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_train_ripor_relevance_encoder)

    p = sub.add_parser("train-pag-sparse-encoder")
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument(
        "--hard-negatives",
        type=Path,
        required=True,
        help="frozen teacher-ranked candidate JSONL with scores; required for strict PAG SPLADE training",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="t5-base")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-negatives", type=int, default=1)
    p.add_argument("--allow-weak-teacher", action="store_true")
    p.add_argument("--query-reg-weight", type=float, default=0.01)
    p.add_argument("--doc-reg-weight", type=float, default=0.008)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_train_pag_sparse_encoder)

    p = sub.add_parser("train-rqvae")
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="RQ latent width; default keeps the relevance encoder width so RIPOR centroids can initialize GR",
    )
    p.add_argument(
        "--ripor-identity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "quantize M0/T5 relevance embeddings directly so exported centroids are valid RIPOR E_i; "
            "use --no-ripor-identity only for a legacy learned-bottleneck ablation"
        ),
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--commit-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_train_rqvae)

    p = sub.add_parser("build-decoder-data")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--semantic-ids", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--input-prefix", default="skill routing query: ")
    p.add_argument("--target-mode", choices=["native", "special"], default="special")
    p.add_argument("--separator-text")
    p.set_defaults(func=command_build_decoder_data)

    p = sub.add_parser("split-decoder-data")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dev-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--stratify-field", default="target_skill_count")
    p.set_defaults(func=command_split_decoder_data)

    p = sub.add_parser("build-decoding-trie")
    p.add_argument("--semantic-ids", type=Path, required=True)
    p.add_argument("--special-tokens", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="t5-base")
    p.add_argument("--target-mode", choices=["native", "special"], default="special")
    p.add_argument("--separator-text")
    p.add_argument("--max-skills", type=int, default=DEFAULT_GR_MAX_SKILLS)
    p.set_defaults(func=command_build_decoding_trie)

    p = sub.add_parser("train-decoder")
    p.add_argument("--decoder-rows", type=Path, required=True)
    p.add_argument("--dev-rows", type=Path)
    p.add_argument("--special-tokens", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="t5-base")
    p.add_argument("--max-source-length", type=int, default=512)
    p.add_argument("--max-target-length", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260626)
    p.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.set_defaults(func=command_train_decoder)

    p = sub.add_parser("train-structured-decoder")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--dev-rows", type=Path)
    p.add_argument("--semantic-ids", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="t5-base")
    p.add_argument("--codebook-size", type=int)
    p.add_argument("--dedup-size", type=int)
    p.add_argument("--catalog-input-prefix", default="skill document: ")
    p.add_argument("--catalog-epochs", type=int, default=0)
    p.add_argument("--catalog-max-steps", type=int, default=-1)
    p.add_argument(
        "--pag-set-ids",
        type=Path,
        help="lexical set-ID manifest produced by build-pag-set-ids; enables the joint PAG set branch",
    )
    p.add_argument(
        "--pag-set-hard-negatives",
        type=Path,
        help="frozen teacher-ranked candidate JSONL for the PAG lexical MarginMSE branch",
    )
    p.add_argument("--pag-set-max-negatives", type=int, default=1)
    p.add_argument("--pag-set-allow-weak-teacher", action="store_true")
    p.add_argument("--pag-set-loss-weight", type=float, default=0.0)
    p.add_argument("--pag-set-sparsity-weight", type=float, default=0.0)
    p.add_argument("--max-source-length", type=int, default=DEFAULT_GR_MAX_SOURCE_LENGTH)
    p.add_argument("--allow-source-truncation", action="store_true")
    p.add_argument("--allow-checkpoint-config-mismatch", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260626)
    p.add_argument("--init-model-dir", type=Path)
    p.add_argument(
        "--ripor-relevance-checkpoint",
        type=Path,
        help="RIPOR M0 relevance encoder checkpoint (fresh decoder initialization only)",
    )
    p.add_argument(
        "--ripor-codebook-embeddings",
        type=Path,
        help="RQ centroid tensor from train-rqvae/codebook_embeddings.pt",
    )
    p.add_argument("--pag-set-init-model-dir", type=Path)
    p.add_argument("--pag-seq-init-model-dir", type=Path)
    p.add_argument(
        "--pag-sparse-checkpoint",
        type=Path,
        help="PAG Msp pag_sparse_encoder directory for the explicit Msp→Mset stage",
    )
    p.add_argument(
        "--pag-set-stage-only",
        action="store_true",
        help="train Mset from Msp using lexical MarginMSE only; emits a role-checked pag_mset checkpoint",
    )
    p.add_argument(
        "--pag-stage-role",
        choices=["baseline", "pag_mseq", "pag_joint"],
        default="baseline",
        help="provenance role written into structured checkpoints",
    )
    p.add_argument(
        "--sequential-ce-weight",
        type=float,
        default=1.0,
        help="explicit CE ablation weight; strict PAG uses 0 and trains Lset+Lseq",
    )
    p.add_argument("--step-offset", type=int, default=0)
    p.add_argument("--initial-best-dev-loss", type=float)
    p.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--ripor-loss-weight", type=float, default=0.0)
    p.add_argument("--ripor-max-negatives", type=int, default=16)
    p.add_argument("--ripor-margin", type=float, default=1.0)
    p.add_argument("--ripor-temperature", type=float, default=1.0)
    p.add_argument("--ripor-alpha-beta", type=float, default=DEFAULT_RIPOR_ALPHA_BETA)
    p.add_argument(
        "--ripor-progressive-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="expand RIPOR prefix objectives from short to full paths while retaining earlier prefixes",
    )
    p.add_argument("--ripor-hard-negatives", type=Path)
    p.add_argument("--ripor-allow-weak-teacher", action="store_true")
    add_wandb_args(p)
    p.set_defaults(func=command_train_structured_decoder)

    p = sub.add_parser("train-single-sid-decoder")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--dev-rows", type=Path)
    p.add_argument("--semantic-ids", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="t5-base")
    p.add_argument("--codebook-size", type=int)
    p.add_argument("--dedup-size", type=int)
    p.add_argument("--catalog-input-prefix", default="skill document: ")
    p.add_argument("--catalog-epochs", type=int, default=3)
    p.add_argument("--catalog-max-steps", type=int, default=-1)
    p.add_argument("--max-source-length", type=int, default=DEFAULT_GR_MAX_SOURCE_LENGTH)
    p.add_argument("--allow-source-truncation", action="store_true")
    p.add_argument("--allow-checkpoint-config-mismatch", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260626)
    p.add_argument("--init-model-dir", type=Path)
    p.add_argument(
        "--ripor-relevance-checkpoint",
        type=Path,
        help="RIPOR M0 relevance encoder checkpoint (fresh decoder initialization only)",
    )
    p.add_argument(
        "--ripor-codebook-embeddings",
        type=Path,
        help="RQ centroid tensor from train-rqvae/codebook_embeddings.pt",
    )
    p.add_argument("--step-offset", type=int, default=0)
    p.add_argument("--initial-best-dev-loss", type=float)
    p.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--ripor-loss-weight",
        type=float,
        default=0.0,
        help="RIPOR prefix-ranking weight; zero preserves the original CE-only objective",
    )
    p.add_argument("--ripor-max-negatives", type=int, default=16)
    p.add_argument("--ripor-margin", type=float, default=1.0)
    p.add_argument("--ripor-temperature", type=float, default=1.0)
    p.add_argument("--ripor-alpha-beta", type=float, default=DEFAULT_RIPOR_ALPHA_BETA)
    p.add_argument(
        "--ripor-progressive-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="expand RIPOR prefix objectives from short to full paths while retaining earlier prefixes",
    )
    p.add_argument(
        "--ripor-allow-weak-teacher",
        action="store_true",
        help="allow reciprocal-rank/anchor fallback scores; never use for a strict RIPOR comparison",
    )
    p.add_argument(
        "--ripor-hard-negatives",
        type=Path,
        help="ranked GR/dense prediction JSONL used as a refreshable hard-negative pool",
    )
    add_wandb_args(p)
    p.set_defaults(func=command_train_single_sid_decoder)

    p = sub.add_parser("eval-structured-decoder")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--semantic-ids", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-name", default="dev")
    p.add_argument("--max-source-length", type=int)
    p.add_argument("--allow-source-truncation", action="store_true")
    p.add_argument("--allow-checkpoint-config-mismatch", action="store_true")
    p.add_argument("--num-beams", type=int, default=DEFAULT_GR_NUM_BEAMS)
    p.add_argument("--max-skills", type=int, default=DEFAULT_GR_MAX_SKILLS)
    p.add_argument("--k-values", default="1,10,50")
    p.add_argument(
        "--rank-aggregation",
        choices=["beam_order", "marginal", "marginal_length_norm"],
        default="beam_order",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--pag-set-ids", type=Path)
    p.add_argument("--pag-set-weight", type=float, default=0.0)
    p.add_argument("--pag-set-top-n", type=int, default=DEFAULT_PAG_SET_TOP_N)
    p.add_argument("--pag-set-aggregation", choices=["max", "logsumexp"], default="max")
    p.add_argument(
        "--pag-planner-predictions",
        type=Path,
        help="deprecated external planner ablation; rejected by the PAG reproduction path",
    )
    p.add_argument("--pag-planner-weight", type=float, default=0.0)
    p.add_argument("--pag-planner-aggregation", choices=["max", "logsumexp"], default="max")
    p.add_argument("--pag-planner-score-mode", choices=["reciprocal_rank", "raw"], default="reciprocal_rank")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_eval_structured_decoder)

    p = sub.add_parser("eval-single-sid-decoder")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--semantic-ids", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-name", default="dev")
    p.add_argument("--max-source-length", type=int)
    p.add_argument("--allow-source-truncation", action="store_true")
    p.add_argument("--allow-checkpoint-config-mismatch", action="store_true")
    p.add_argument("--num-beams", type=int, default=DEFAULT_GR_NUM_BEAMS)
    p.add_argument("--max-skills", type=int, default=1)
    p.add_argument("--k-values", default="1,10,50")
    p.add_argument(
        "--rank-aggregation",
        choices=["beam_order", "marginal", "marginal_length_norm"],
        default="beam_order",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--pag-set-ids", type=Path)
    p.add_argument("--pag-set-weight", type=float, default=0.0)
    p.add_argument("--pag-set-top-n", type=int, default=DEFAULT_PAG_SET_TOP_N)
    p.add_argument("--pag-set-aggregation", choices=["max", "logsumexp"], default="max")
    p.add_argument("--pag-planner-predictions", type=Path)
    p.add_argument("--pag-planner-weight", type=float, default=0.0)
    p.add_argument("--pag-planner-aggregation", choices=["max", "logsumexp"], default="max")
    p.add_argument("--pag-planner-score-mode", choices=["reciprocal_rank", "raw"], default="reciprocal_rank")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_eval_single_sid_decoder)

    p = sub.add_parser("eval-bm25")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--k-values", default="1,10,50")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--log-every", type=int, default=25)
    p.set_defaults(func=command_eval_bm25)

    p = sub.add_parser("eval-t5-dense")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="t5-base")
    p.add_argument("--k-values", default="1,10,50")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-query-length", type=int, default=256)
    p.add_argument("--max-skill-length", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_eval_t5_dense)

    p = sub.add_parser("eval-decoder")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--trie", type=Path, required=True)
    p.add_argument("--semantic-id-token-ids", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-name", default="dev")
    p.add_argument("--max-source-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-beams", type=int, default=10)
    p.add_argument("--max-skills", type=int, default=DEFAULT_GR_MAX_SKILLS)
    p.add_argument("--separator-text", default=SKILL_SEP_TOKEN)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_eval_decoder)

    p = sub.add_parser("rescore-ranked")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--subset", choices=["all", "single", "multi"], default="all")
    p.add_argument("--k-values", default="1,10,50")
    p.set_defaults(func=command_rescore_ranked)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
