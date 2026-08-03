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
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr
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
    for field in ("source_index", "index"):
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


def load_ranked_skill_candidates_with_scores(
    path: Path | None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """Load ranked candidates and their teacher relevance scores.

    Dense predictions usually provide ``ranked_scores`` while structured GR
    marginal decoding uses ``ranked_skill_scores``.  When a prediction only
    contains an ordered list, reciprocal rank is retained as an explicit weak
    teacher score instead of silently discarding the ranking signal.
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
        ):
            values.extend(list(row.get(field) or []))
        if not values:
            continue
        ordered = _unique_preserving_order([str(value) for value in values])
        score_payload = row.get("ranked_scores") or row.get("ranked_skill_scores") or []
        score_by_skill = (
            {str(sid): value for sid, value in score_payload.items()}
            if isinstance(score_payload, dict)
            else None
        )
        raw_scores = list(score_payload) if score_by_skill is None else []
        ranked_ids = [str(value) for value in row.get("ranked_skill_ids", [])]
        row_scores: dict[str, float] = {}
        for rank, sid in enumerate(ranked_ids):
            if score_by_skill is not None:
                value = score_by_skill.get(sid, 1.0 / float(rank + 1))
            elif rank < len(raw_scores):
                value = raw_scores[rank]
            else:
                value = 1.0 / float(rank + 1)
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = 1.0 / float(rank + 1)
            row_scores[sid] = score if math.isfinite(score) else 1.0 / float(rank + 1)
        for key in retrieval_row_keys(row):
            candidate_map.setdefault(key, ordered)
            teacher_score_map.setdefault(key, {}).update(row_scores)
    return candidate_map, teacher_score_map


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
            else:
                teacher_score = positive_fallback if sid in positive_set else negative_fallback
            candidate_teacher_scores[row_index, candidate_index] = float(teacher_score)
    return {
        "ripor_candidate_codes": candidate_codes,
        "ripor_candidate_valid": candidate_valid,
        "ripor_candidate_positive": candidate_positive,
        "ripor_candidate_teacher_scores": candidate_teacher_scores,
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
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, codebook_size: int, depth: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.quantizer = ResidualVQ(
                dim=latent_dim,
                num_quantizers=depth,
                codebook_size=codebook_size,
                kmeans_init=True,
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
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
    embeddings = payload["embeddings"].float()
    skill_ids = list(payload["skill_ids"])
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SkillRQVAE(
        input_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        depth=args.depth,
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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": embeddings.shape[1],
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "codebook_size": args.codebook_size,
            "depth": args.depth,
            "semantic_id_mode": SEMANTIC_ID_MODE_DEDUP,
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
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "commit_weight": args.commit_weight,
            "seed": args.seed,
            "final_loss": epoch_loss if args.epochs > 0 else None,
            "code_width": code_width,
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


def make_structured_t5_sid_model(
    base_model: str,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str = SEMANTIC_ID_MODE_DEDUP,
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
            if self.semantic_id_mode == SEMANTIC_ID_MODE_LEARNED and self.dedup_size != 0:
                raise ValueError(f"learned_codes requires dedup_size=0, got {self.dedup_size}")
            self.control_type = structured_sid_control_type(self.depth, self.semantic_id_mode)
            self.distributed_context: DistributedContext | None = None
            self.sid_embedding = nn.Embedding(
                structured_sid_vocab_size(self.codebook_size, self.depth, self.dedup_size),
                d_model,
            )
            self.code_heads = nn.ModuleList(nn.Linear(d_model, self.codebook_size) for _ in range(self.depth))
            self.dedup_head = (
                nn.Linear(d_model, self.dedup_size)
                if self.semantic_id_mode == SEMANTIC_ID_MODE_DEDUP
                else None
            )
            self.control_head = nn.Linear(d_model, 2)

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
        ):
            decoder_embeds = self.sid_embedding(decoder_input_ids)
            decoder_outputs = self.backbone.decoder(
                inputs_embeds=decoder_embeds,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
            )
            return decoder_outputs.last_hidden_state

        def _decoder_hidden(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
            encoder_hidden_states = self.encode_source(input_ids, attention_mask)
            return self._decoder_hidden_from_encoder(
                encoder_hidden_states,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
            )

        def _normalize_loss(self, loss_sum, count, global_loss_normalization: bool):
            if global_loss_normalization and self.distributed_context is not None:
                return global_normalized_loss(loss_sum, count, self.distributed_context)
            return loss_sum / count.clamp_min(1.0)

        def ripor_prefix_log_scores(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            candidate_codes,
        ):
            """Teacher-force semantic code paths and return cumulative log scores.

            The dedup suffix is deliberately excluded: it makes identifiers
            unique but carries no semantic relevance signal.
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
            hidden = self._decoder_hidden_from_encoder(
                flat_encoder,
                flat_encoder_mask,
                flat_decoder_ids,
                flat_decoder_mask,
            )
            cumulative = hidden.new_zeros(batch_size * candidate_count)
            prefixes = []
            flat_codes = candidate_codes.reshape(batch_size * candidate_count, rank_depth)
            for level, head in enumerate(self.code_heads):
                log_probs = F.log_softmax(head(hidden[:, level, :]), dim=-1)
                cumulative = cumulative + log_probs.gather(1, flat_codes[:, level : level + 1]).squeeze(1)
                prefixes.append(cumulative)
            return torch.stack(prefixes, dim=-1).reshape(batch_size, candidate_count, rank_depth)

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
        ):
            """Align student prefix margins to frozen teacher relevance margins."""
            if temperature <= 0:
                raise ValueError(f"RIPOR temperature must be positive, got {temperature}")
            if candidate_teacher_scores is None:
                candidate_teacher_scores = candidate_positive.to(dtype=torch.float32)
            if candidate_teacher_scores.shape != candidate_valid.shape:
                raise ValueError(
                    "RIPOR teacher scores must have shape [batch, candidates], "
                    f"got {tuple(candidate_teacher_scores.shape)} vs {tuple(candidate_valid.shape)}"
                )
            prefix_scores = self.ripor_prefix_log_scores(
                encoder_hidden_states,
                encoder_attention_mask,
                candidate_codes,
            )
            candidate_teacher_scores = candidate_teacher_scores.to(
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
                teacher_margin_raw = teacher_positive_scores[:, None] - teacher_negative_scores[None, :]
                # Preserve scored teacher differences.  The CLI margin is only
                # a fallback target for ties/non-positive weak-teacher pairs;
                # it is not applied to every pair as a fixed binary objective.
                teacher_margin = teacher_margin_raw.clamp_min(0.0)
                if float(margin) > 0:
                    teacher_margin = torch.where(
                        teacher_margin_raw <= 0,
                        teacher_margin.new_full(teacher_margin.shape, float(margin)),
                        teacher_margin,
                    )
                student_margin = positive_scores[:, None, :] - negative_scores[None, :, :]
                pair_losses = F.smooth_l1_loss(
                    student_margin / float(temperature),
                    teacher_margin[:, :, None].expand_as(student_margin) / float(temperature),
                    reduction="none",
                ) * float(temperature)
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
            ripor_loss_weight: float = 0.0,
            ripor_margin: float = 1.0,
            ripor_temperature: float = 1.0,
            global_loss_normalization: bool = True,
            return_components: bool = False,
        ):
            encoder_hidden_states = self.encode_source(input_ids, attention_mask)
            hidden = self._decoder_hidden_from_encoder(
                encoder_hidden_states,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
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
                )
                ripor_loss = self._normalize_loss(
                    ripor_loss_sum,
                    ripor_pair_count,
                    global_loss_normalization,
                )
            loss = ce_loss + float(ripor_loss_weight) * ripor_loss
            if return_components:
                return {
                    "loss": loss,
                    "ce_loss": ce_loss,
                    "ripor_loss": ripor_loss,
                    "local_ce_loss_sum": loss_sum,
                    "local_ce_token_count": token_count,
                    "local_ripor_loss_sum": ripor_loss_sum,
                    "local_ripor_pair_count": ripor_pair_count,
                }
            return loss

        @torch.no_grad()
        def logits_for_next_type(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, target_type: int):
            hidden = self._decoder_hidden(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
            return self.logits_for_hidden_type(hidden, target_type)

        @torch.no_grad()
        def logits_for_next_type_from_encoder(
            self,
            encoder_hidden_states,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            target_type: int,
        ):
            hidden = self._decoder_hidden_from_encoder(
                encoder_hidden_states,
                encoder_attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
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

    return {
        "input_ids": encoded_inputs["input_ids"].to(device),
        "attention_mask": encoded_inputs["attention_mask"].to(device),
        "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long, device=device),
        "decoder_attention_mask": torch.tensor(decoder_attention_mask, dtype=torch.long, device=device),
        "target_types": torch.tensor(target_types, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _structured_global_train_metrics(components: dict[str, Any], context: DistributedContext) -> dict[str, float]:
    ce_sum, ce_count = reduce_sum_count(
        components["local_ce_loss_sum"], components["local_ce_token_count"], context
    )
    ripor_sum, ripor_count = reduce_sum_count(
        components["local_ripor_loss_sum"], components["local_ripor_pair_count"], context
    )
    return {
        "ce_loss": ce_sum / max(1, ce_count),
        "ripor_loss": ripor_sum / max(1, ripor_count),
        "ripor_pairs": float(ripor_count),
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
                )
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
                        )
                    )
                    batch.update(
                        {
                            "ripor_loss_weight": args.ripor_loss_weight,
                            "ripor_margin": args.ripor_margin,
                            "ripor_temperature": args.ripor_temperature,
                        }
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
                    train_loss = last_metrics["ce_loss"] + (
                        args.ripor_loss_weight * last_metrics["ripor_loss"] if use_ripor else 0.0
                    )
                    if context.is_main:
                        print(f"{stage_name}_step={display_step} train_loss={train_loss:.6f}", flush=True)
                        event = {
                            "stage": stage_name,
                            "event": "train",
                            "step": display_step,
                            "epoch": display_step / steps_per_epoch,
                            "train_loss": train_loss,
                            "ce_loss": last_metrics["ce_loss"],
                            "ripor_loss": last_metrics["ripor_loss"] if use_ripor else None,
                            "ripor_pairs": int(last_metrics["ripor_pairs"]) if use_ripor else 0,
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
            "last_train_loss": last_metrics["ce_loss"] + (
                args.ripor_loss_weight * last_metrics["ripor_loss"] if use_ripor else 0.0
            ),
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
    ).to(context.device)
    optimizer = torch.optim.AdamW(core_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loaded_optimizer_state = False
    init_semantic_ids_verified = False
    init_source_length_verified = False
    init_source_text_policy_verified = False
    if args.init_model_dir:
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
    core_model.configure_distributed(context)
    return core_model, optimizer, {
        "loaded_optimizer_state": loaded_optimizer_state,
        "init_semantic_ids_verified": init_semantic_ids_verified,
        "init_source_length_verified": init_source_length_verified,
        "init_source_text_policy_verified": init_source_text_policy_verified,
    }


def command_train_single_sid_decoder(args: argparse.Namespace) -> None:
    """Train the single-SID GR with RIPOR prefix ranking and optional DDP."""
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
        "teacher_objective": "smooth_l1_prefix_margin_alignment",
        "teacher_score_fallback": "reciprocal_rank_or_positive_negative_anchor",
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
        training_formulation="single_sid_catalog_indexing_with_ripor_prefix_ranking",
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
    random.Random(args.seed).shuffle(catalog_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
        "training": "planner_only_at_decode",
        "multi_sid_prefix_ranking": "disabled_due_to_lexicographic_set_serialization",
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
        training_formulation="multi_sid_set_generation",
        wandb_run=wandb_run,
        ripor_enabled=False,
        ripor_hard_negative_map={},
        ripor_teacher_score_map={},
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
) -> dict[str, Any]:
    new_prefix = list(beam["current_prefix"])
    new_completed = list(beam["completed"])
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
    planner_score = float(beam.get("planner_score", 0.0)) + float(planner_increment)
    return {
        "decoder_input_ids": [*beam["decoder_input_ids"], next_global_id],
        "current_prefix": new_prefix,
        "completed": new_completed,
        "model_score": model_score,
        "planner_score": planner_score,
        "score": model_score + float(planner_weight) * planner_score,
        "stopped": new_stopped,
    }


def ranked_skill_ids_from_structured_beams(
    beams: list[dict[str, Any]],
    sequence_to_skill_id: dict[tuple[int, ...], str],
    max_rank: int,
) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for beam in sorted(beams, key=lambda item: float(item["score"]), reverse=True):
        for values in beam["completed"]:
            sid = sequence_to_skill_id.get(tuple(values))
            if sid is not None and sid not in seen:
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
        score = float(beam["score"])
        if length_normalize:
            score = score / max(1, len(beam["decoder_input_ids"]) - 1)
        for values in set(tuple(values) for values in beam["completed"]):
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
    pag_planner_weight = float(getattr(args, "pag_planner_weight", 0.0))
    if pag_planner_weight != 0.0 and args.pag_planner_predictions is None:
        raise ValueError("--pag-planner-predictions is required when --pag-planner-weight is non-zero")
    pag_planner = (
        load_pag_planner_scores(args.pag_planner_predictions, args.pag_planner_score_mode)
        if pag_planner_weight != 0.0
        else {}
    )
    prefix_descendants = (
        build_semantic_prefix_descendants(semantic_ids, depth)
        if pag_planner_weight != 0.0
        else {}
    )
    device = context.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = make_structured_t5_sid_model(
        str(checkpoint["model"]),
        codebook_size,
        depth,
        dedup_size,
        semantic_id_mode,
    )
    model.load_state_dict(checkpoint["state_dict"])
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
        if pag_planner_weight != 0.0:
            for key in retrieval_row_keys(row):
                for sid, score in pag_planner.get(key, {}).items():
                    row_planner_scores[sid] = max(row_planner_scores.get(sid, float("-inf")), score)
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
        beams = [
            {
                "decoder_input_ids": [bos_id],
                "current_prefix": [],
                "completed": [],
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
                    trie,
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
                )
                log_probs_batch = F.log_softmax(logits_batch, dim=-1)
                for (beam, allowed), log_probs in zip(active_items, log_probs_batch):
                    allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
                    allowed_scores = log_probs.index_select(0, allowed_tensor)
                    ranked_with_planner = []
                    for raw_score, value in zip(allowed_scores.detach().cpu().tolist(), allowed):
                        planner_increment = 0.0
                        if pag_planner_weight != 0.0 and target_type < semantic_length:
                            completed_skill_ids = {
                                sequence_to_skill_id[tuple(values)]
                                for values in beam["completed"]
                                if tuple(values) in sequence_to_skill_id
                            }
                            planner_increment = pag_prefix_planner_score(
                                [*beam["current_prefix"], int(value)],
                                row_planner_scores,
                                prefix_descendants,
                                args.pag_planner_aggregation,
                                excluded_skill_ids=completed_skill_ids,
                            )
                        combined_score = float(raw_score) + pag_planner_weight * planner_increment
                        ranked_with_planner.append(
                            (combined_score, float(raw_score), int(value), planner_increment)
                        )
                    ranked = sorted(ranked_with_planner, reverse=True)[: args.num_beams]
                    for _, log_prob, value, planner_increment in ranked:
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
                                planner_increment=planner_increment,
                                planner_weight=pag_planner_weight,
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
            "pag_planner_predictions": str(args.pag_planner_predictions) if args.pag_planner_predictions else None,
            "pag_planner_weight": pag_planner_weight,
            "pag_planner_aggregation": args.pag_planner_aggregation,
            "pag_planner_score_mode": args.pag_planner_score_mode,
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

    p = sub.add_parser("encode-skills")
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_encode)

    p = sub.add_parser("train-rqvae")
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=128)
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
    p.add_argument("--step-offset", type=int, default=0)
    p.add_argument("--initial-best-dev-loss", type=float)
    p.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu", action="store_true")
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
    p.add_argument(
        "--pag-planner-predictions",
        type=Path,
        help="frozen dense/BM25 ranked predictions used by the PAG-style prefix planner",
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
