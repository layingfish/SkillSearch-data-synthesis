#!/usr/bin/env python3
"""Exhaustively score an aligned per-query skill pool with one legacy checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import skill_gr_pipeline as pipeline  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_semantic_config_sha256(path: Path) -> str:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"tokenizer config must be an object: {path}")
    config.pop("is_local", None)
    return stable_json_sha256(config)


def tokenized_query_inputs_sha256(
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_source_length: int,
) -> str:
    projection = []
    for row_position, row in enumerate(rows):
        encoded = tokenizer(
            [pipeline.decoder_source_text(row)],
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )
        projection.append(
            {
                "row_position": row_position,
                "index": int(row["index"]),
                "input_ids": [int(value) for value in encoded["input_ids"][0]],
                "attention_mask": [
                    int(value) for value in encoded["attention_mask"][0]
                ],
            }
        )
    return stable_json_sha256(projection)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def parse_label_paths(items: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"--predictions must use LABEL=PATH, got: {item}")
        if label in parsed:
            raise ValueError(f"duplicate prediction label: {label}")
        parsed[label] = Path(raw_path)
    return parsed


def aligned_prediction_panel(
    prediction_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_rows: int,
    expected_depth: int,
) -> tuple[list[int], dict[str, dict[int, Mapping[str, Any]]]]:
    if not prediction_rows:
        raise ValueError("at least one prediction input is required")
    indexed: dict[str, dict[int, Mapping[str, Any]]] = {}
    for label, rows in prediction_rows.items():
        if len(rows) != expected_rows:
            raise ValueError(f"{label}: expected {expected_rows} rows, found {len(rows)}")
        by_index: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            index = row.get("index")
            if not isinstance(index, int) or index in by_index:
                raise ValueError(f"{label}: invalid or duplicate index={index!r}")
            ranked = row.get("ranked_skill_ids")
            gold = row.get("gold_skill_ids")
            if (
                not isinstance(ranked, list)
                or len(ranked) != expected_depth
                or len(set(ranked)) != expected_depth
                or not all(isinstance(skill_id, str) for skill_id in ranked)
            ):
                raise ValueError(f"{label} index={index}: invalid Top-{expected_depth}")
            if (
                not isinstance(gold, list)
                or not gold
                or len(set(gold)) != len(gold)
                or not all(isinstance(skill_id, str) for skill_id in gold)
            ):
                raise ValueError(f"{label} index={index}: invalid gold_skill_ids")
            if row.get("target_skill_count") != len(gold):
                raise ValueError(f"{label} index={index}: target_skill_count mismatch")
            by_index[index] = row
        indexed[label] = by_index

    reference_label = next(iter(indexed))
    reference = indexed[reference_label]
    indices = sorted(reference)
    for label, by_index in indexed.items():
        if sorted(by_index) != indices:
            raise ValueError(f"{label}: indices differ from {reference_label}")
        for index in indices:
            for field in (
                "source_index",
                "query",
                "gold_skill_ids",
                "target_skill_count",
            ):
                if by_index[index].get(field) != reference[index].get(field):
                    raise ValueError(
                        f"{label} index={index}: {field} differs from {reference_label}"
                    )
    return indices, indexed


def build_candidate_pools(
    indexed: Mapping[str, Mapping[int, Mapping[str, Any]]],
    indices: Sequence[int],
    *,
    known_skill_ids: set[str],
) -> dict[int, list[str]]:
    pools: dict[int, list[str]] = {}
    for index in indices:
        candidates: set[str] = set()
        gold: set[str] | None = None
        for rows in indexed.values():
            row = rows[index]
            candidates.update(row["ranked_skill_ids"])
            row_gold = set(row["gold_skill_ids"])
            if gold is None:
                gold = row_gold
            elif row_gold != gold:
                raise ValueError(f"index={index}: gold mismatch while building pool")
        candidates.update(gold or set())
        unknown = sorted(candidates - known_skill_ids)
        if unknown:
            raise ValueError(f"index={index}: candidate pool has unknown skills: {unknown[:5]}")
        if not gold or not gold.issubset(candidates):
            raise ValueError(f"index={index}: candidate pool does not cover gold")
        pools[index] = sorted(candidates)
    return pools


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_dir)
    except AttributeError as exc:
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        tokenizer_config = (
            json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            if tokenizer_config_path.is_file()
            else {}
        )
        if not isinstance(tokenizer_config.get("extra_special_tokens"), list):
            raise
        if "object has no attribute 'keys'" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(model_dir, extra_special_tokens={})


def load_model(
    checkpoint_dir: Path,
    base_model: Path,
    semantic_ids: dict[str, list[int]],
    semantic_id_mode: str,
    device,
):
    import torch

    checkpoint_path = checkpoint_dir / "structured_decoder.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    pipeline.validate_checkpoint_semantic_ids(
        checkpoint,
        semantic_ids,
        semantic_id_mode,
    )
    if not pipeline.validate_checkpoint_source_text_policy(checkpoint, False):
        raise ValueError("checkpoint does not record the required source-text policy")
    codebook_size = int(checkpoint["codebook_size"])
    depth = int(checkpoint["depth"])
    dedup_size = int(checkpoint.get("dedup_size", 0))
    sid_config = pipeline.infer_structured_sid_config(
        semantic_ids,
        codebook_size=codebook_size,
        dedup_size=dedup_size,
        semantic_id_mode=semantic_id_mode,
    )
    if sid_config["depth"] != depth:
        raise ValueError(
            f"semantic depth={sid_config['depth']} differs from checkpoint depth={depth}"
        )
    model = pipeline.make_structured_t5_sid_model(
        str(base_model),
        codebook_size,
        depth,
        dedup_size,
        semantic_id_mode,
    )
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    optional_prefixes = (*pipeline.RIPOR_OPTIONAL_STATE_PREFIXES, "pag_set_head.")
    missing_required = [
        key for key in missing_keys if not key.startswith(optional_prefixes)
    ]
    unexpected_required = [
        key for key in unexpected_keys if not key.startswith(optional_prefixes)
    ]
    if missing_required or unexpected_required:
        raise ValueError(
            "checkpoint parameter mismatch: "
            f"missing={missing_required[:5]}, unexpected={unexpected_required[:5]}"
        )
    if any(key.startswith("ripor_heads.") for key in missing_keys):
        model.initialize_ripor_heads_from_code_heads()
    model.to(device).eval()
    return model, checkpoint, sid_config


def exhaustive_score_candidate_pool(
    model,
    encoder_hidden_states,
    encoder_attention_mask,
    candidate_semantic_ids: Mapping[str, list[int]],
    *,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str,
) -> tuple[
    list[str],
    dict[str, float],
    dict[str, list[float]],
    dict[str, Any],
]:
    import torch
    import torch.nn.functional as F

    trie, sequence_to_skill_id = pipeline._semantic_id_trie_from_ids(
        dict(candidate_semantic_ids), semantic_id_mode
    )
    capacity = len(candidate_semantic_ids)
    if capacity <= 0:
        raise ValueError("candidate pool must not be empty")
    bos_id = pipeline.structured_sid_bos_id(
        codebook_size, depth, dedup_size, semantic_id_mode
    )
    semantic_length = pipeline.structured_sid_length(depth, semantic_id_mode)
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
            "score_components": [],
        }
    ]
    frontier_sizes: list[int] = []
    for target_type in range(semantic_length + 1):
        candidates = []
        active_items = []
        for beam in beams:
            if beam["stopped"]:
                candidates.append(beam)
                continue
            allowed = pipeline.structured_allowed_values(
                beam["current_prefix"],
                beam["completed"],
                target_type,
                trie,
                depth,
                1,
                semantic_id_mode,
            )
            if not allowed:
                raise ValueError(
                    f"exhaustive pool reached a dead prefix at target_type={target_type}"
                )
            active_items.append((beam, allowed))
        max_decoder_length = max(
            len(beam["decoder_input_ids"]) for beam, _ in active_items
        )
        decoder_input_rows = []
        decoder_mask_rows = []
        for beam, _ in active_items:
            decoder_ids = list(beam["decoder_input_ids"])
            pad_length = max_decoder_length - len(decoder_ids)
            decoder_input_rows.append(decoder_ids + [bos_id] * pad_length)
            decoder_mask_rows.append([1] * len(decoder_ids) + [0] * pad_length)
        decoder_input_ids = torch.tensor(
            decoder_input_rows,
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        decoder_attention_mask = torch.tensor(
            decoder_mask_rows,
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        batch_size = len(active_items)
        logits_batch = model.logits_for_next_type_from_encoder(
            encoder_hidden_states.repeat(batch_size, 1, 1),
            encoder_attention_mask.repeat(batch_size, 1),
            decoder_input_ids,
            decoder_attention_mask,
            target_type,
            ripor_score=False,
        )
        score_batch = F.log_softmax(logits_batch, dim=-1)
        for (beam, allowed), step_scores in zip(active_items, score_batch):
            ranked_allowed = sorted(
                (
                    (float(step_scores[value].detach().cpu()), int(value))
                    for value in allowed
                ),
                reverse=True,
            )
            for log_prob, value in ranked_allowed:
                advanced = pipeline.advance_structured_beam(
                    beam,
                    target_type,
                    value,
                    log_prob,
                    codebook_size,
                    depth,
                    dedup_size,
                    semantic_id_mode,
                )
                advanced["score_components"] = [
                    *beam.get("score_components", []),
                    log_prob,
                ]
                candidates.append(advanced)
        if len(candidates) > capacity:
            raise ValueError(
                f"pool frontier={len(candidates)} exceeds exhaustive capacity={capacity}"
            )
        beams = sorted(
            candidates, key=lambda item: float(item["score"]), reverse=True
        )
        frontier_sizes.append(len(beams))

    if len(beams) != capacity or not all(beam["stopped"] for beam in beams):
        raise ValueError(
            f"exhaustive completion mismatch: beams={len(beams)}, capacity={capacity}"
        )
    ranked = pipeline.ranked_skill_ids_from_structured_beams(
        beams, sequence_to_skill_id, capacity
    )
    scores = pipeline.ranked_skill_scores_from_structured_beams(
        beams, sequence_to_skill_id, ranked
    )
    if len(ranked) != capacity or set(ranked) != set(candidate_semantic_ids):
        raise ValueError("exhaustive scorer did not return the exact candidate pool")
    if set(scores) != set(ranked):
        raise ValueError("exhaustive scorer is missing ranked scores")
    score_components: dict[str, list[float]] = {}
    for beam in beams:
        if len(beam["completed"]) != 1:
            raise ValueError("single-skill exhaustive beam must contain one completion")
        skill_id = sequence_to_skill_id.get(tuple(beam["completed"][0]))
        if skill_id is None or skill_id in score_components:
            raise ValueError("single-skill exhaustive beam has an invalid completion")
        components = [float(value) for value in beam.get("score_components", [])]
        if len(components) != semantic_length + 1:
            raise ValueError(
                f"candidate={skill_id}: expected {semantic_length + 1} score components, "
                f"found {len(components)}"
            )
        if not math.isclose(
            math.fsum(components),
            float(scores[skill_id]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"candidate={skill_id}: component sum differs from total score")
        score_components[skill_id] = components
    if set(score_components) != set(ranked):
        raise ValueError("exhaustive scorer is missing score components")
    ordered_scores = [scores[skill_id] for skill_id in ranked]
    if any(left < right for left, right in zip(ordered_scores, ordered_scores[1:])):
        raise ValueError("exhaustive scores are not monotonic")
    return (
        ranked,
        scores,
        score_components,
        {"frontier_sizes": frontier_sizes, "capacity": capacity},
    )


def score_single_candidate_stepwise(
    model,
    encoder_hidden_states,
    encoder_attention_mask,
    codes: Sequence[int],
    *,
    codebook_size: int,
    depth: int,
    dedup_size: int,
    semantic_id_mode: str,
) -> float:
    """Reference replay used by the fake-model invariant test."""
    import torch
    import torch.nn.functional as F

    if len(codes) != pipeline.structured_sid_length(depth, semantic_id_mode):
        raise ValueError("candidate code length differs from structured SID length")
    decoder_ids = [
        pipeline.structured_sid_bos_id(
            codebook_size, depth, dedup_size, semantic_id_mode
        )
    ]
    total = 0.0
    labels = [*codes, 1]
    for target_type, value in enumerate(labels):
        decoder_input_ids = torch.tensor(
            [decoder_ids], dtype=torch.long, device=encoder_hidden_states.device
        )
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        logits = model.logits_for_next_type_from_encoder(
            encoder_hidden_states,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            target_type,
            ripor_score=False,
        )
        total += float(F.log_softmax(logits, dim=-1)[0, value])
        decoder_ids.append(
            pipeline.global_sid_token_id(
                target_type,
                int(value),
                codebook_size,
                depth,
                dedup_size,
                semantic_id_mode,
            )
        )
    return total


def validate_archived_scores(
    fixed_scores: Mapping[str, float],
    prediction: Mapping[str, Any],
    *,
    atol: float,
) -> dict[str, Any]:
    archived = prediction.get("ranked_skill_scores")
    if archived is None:
        return {"available": False, "count": 0, "max_absolute_error": None}
    if not isinstance(archived, dict):
        raise ValueError("ranked_skill_scores must be object/null")
    missing = set(prediction["ranked_skill_ids"]) - set(archived)
    if missing:
        raise ValueError(f"archived prediction is missing scores: {sorted(missing)[:5]}")
    errors = {
        skill_id: abs(float(fixed_scores[skill_id]) - float(archived[skill_id]))
        for skill_id in prediction["ranked_skill_ids"]
    }
    max_skill = max(errors, key=errors.get)
    maximum = errors[max_skill]
    if maximum > atol:
        raise ValueError(
            f"fixed-pool score replay exceeds atol={atol}: "
            f"skill={max_skill}, max_absolute_error={maximum}"
        )
    raw_ranked = list(prediction["ranked_skill_ids"])
    replay_ranked = sorted(
        raw_ranked,
        key=lambda skill_id: float(fixed_scores[skill_id]),
        reverse=True,
    )
    rank_set_checks = {
        f"@{cutoff}": set(raw_ranked[:cutoff]) == set(replay_ranked[:cutoff])
        for cutoff in (1, 5, 10, 50)
    }
    if not all(rank_set_checks.values()):
        raise ValueError(
            "fixed-pool score replay changes archived Top-K membership: "
            f"{rank_set_checks}"
        )
    return {
        "available": True,
        "count": len(errors),
        "max_absolute_error": maximum,
        "max_error_skill_id": max_skill,
        "atol": atol,
        "rank_set_checks": rank_set_checks,
    }


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def command_score(args: argparse.Namespace) -> None:
    import torch
    from importlib import metadata

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite shard output: {args.output_dir}")
    checkpoint_path = args.checkpoint_dir / "structured_decoder.pt"
    observed_checkpoint_sha = file_sha256(checkpoint_path)
    if observed_checkpoint_sha != args.checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected={args.checkpoint_sha256}, observed={observed_checkpoint_sha}"
        )
    prediction_paths = parse_label_paths(args.predictions)
    prediction_rows = {
        label: read_jsonl(path) for label, path in prediction_paths.items()
    }
    indices, indexed = aligned_prediction_panel(
        prediction_rows,
        expected_rows=args.expected_rows,
        expected_depth=args.prediction_depth,
    )
    if args.label not in indexed:
        raise ValueError(f"checkpoint label is absent from prediction panel: {args.label}")
    semantic_ids, semantic_id_mode = pipeline.load_semantic_ids_with_mode(
        args.semantic_ids
    )
    pools = build_candidate_pools(
        indexed, indices, known_skill_ids=set(semantic_ids)
    )
    rows = pipeline.load_jsonl(args.rows)
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} eval rows, found {len(rows)}")
    for row_position, row in enumerate(rows):
        index = row.get("index")
        if index not in pools:
            raise ValueError(f"eval row position={row_position} has unknown index={index}")
        prediction = indexed[next(iter(indexed))][index]
        if pipeline.row_query_text(row) != prediction["query"]:
            raise ValueError(f"eval row index={index}: query differs from prediction panel")
        if pipeline.prepared_selected_skill_ids(row) != prediction["gold_skill_ids"]:
            raise ValueError(f"eval row index={index}: gold differs from prediction panel")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tokenizer = load_tokenizer(args.checkpoint_dir)
    tokenizer_json_path = args.checkpoint_dir / "tokenizer.json"
    tokenizer_config_path = args.checkpoint_dir / "tokenizer_config.json"
    base_config_path = args.base_model / "config.json"
    base_weights_path = args.base_model / "model.safetensors"
    model_inputs = {
        "base_model": {
            "path": str(args.base_model),
            "config_json_sha256": file_sha256(base_config_path),
            "model_safetensors_sha256": file_sha256(base_weights_path),
        },
        "tokenizer": {
            "path": str(args.checkpoint_dir),
            "tokenizer_json_sha256": file_sha256(tokenizer_json_path),
            "tokenizer_config_json_sha256": file_sha256(tokenizer_config_path),
            "tokenizer_semantic_config_sha256": tokenizer_semantic_config_sha256(
                tokenizer_config_path
            ),
            "tokenized_query_inputs_sha256": tokenized_query_inputs_sha256(
                tokenizer,
                rows,
                max_source_length=args.max_source_length,
            ),
        },
    }
    model, checkpoint, sid_config = load_model(
        args.checkpoint_dir,
        args.base_model,
        semantic_ids,
        semantic_id_mode,
        device,
    )
    if int(checkpoint.get("max_source_length", 0)) != args.max_source_length:
        raise ValueError(
            "checkpoint max_source_length differs from fixed-pool protocol: "
            f"{checkpoint.get('max_source_length')} != {args.max_source_length}"
        )

    output_rows: list[dict[str, Any]] = []
    validation_errors: list[float] = []
    validation_rank_checks: list[Mapping[str, bool]] = []
    with torch.no_grad():
        for row_position, row in enumerate(rows):
            if row_position % args.num_shards != args.shard_id:
                continue
            index = int(row["index"])
            pool = pools[index]
            encoded = tokenizer(
                [pipeline.decoder_source_text(row)],
                padding=True,
                truncation=True,
                max_length=args.max_source_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            encoder_hidden_states = model.encode_source(input_ids, attention_mask)
            candidate_semantic_ids = {
                skill_id: semantic_ids[skill_id] for skill_id in pool
            }
            ranked, scores, score_components, exhaustive_audit = (
                exhaustive_score_candidate_pool(
                    model,
                    encoder_hidden_states,
                    attention_mask,
                    candidate_semantic_ids,
                    codebook_size=int(sid_config["codebook_size"]),
                    depth=int(sid_config["depth"]),
                    dedup_size=int(sid_config["dedup_size"]),
                    semantic_id_mode=semantic_id_mode,
                )
            )
            score_validation = validate_archived_scores(
                scores,
                indexed[args.label][index],
                atol=args.score_atol,
            )
            if score_validation["available"]:
                validation_errors.append(score_validation["max_absolute_error"])
                validation_rank_checks.append(score_validation["rank_set_checks"])
            output_rows.append(
                {
                    "label": args.label,
                    "shard_id": args.shard_id,
                    "row_position": row_position,
                    "index": index,
                    "source_index": row.get("source_index"),
                    "query": pipeline.row_query_text(row),
                    "gold_skill_ids": pipeline.prepared_selected_skill_ids(row),
                    "target_skill_count": pipeline.prepared_target_skill_count(row),
                    "pool_skill_ids": pool,
                    "pool_sha256": stable_json_sha256(pool),
                    "pool_size": len(pool),
                    "fixed_ranked_skill_ids": ranked,
                    "fixed_ranked_skill_scores": scores,
                    "fixed_ranked_skill_score_components": score_components,
                    "raw_ranked_skill_ids": indexed[args.label][index][
                        "ranked_skill_ids"
                    ],
                    "score_replay_validation": score_validation,
                    "exhaustive_audit": exhaustive_audit,
                }
            )
            print(
                json.dumps(
                    {
                        "label": args.label,
                        "shard_id": args.shard_id,
                        "row_position": row_position,
                        "pool_size": len(pool),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    expected_shard_rows = sum(
        row_position % args.num_shards == args.shard_id
        for row_position in range(args.expected_rows)
    )
    if len(output_rows) != expected_shard_rows:
        raise ValueError(
            f"shard row count mismatch: expected={expected_shard_rows}, got={len(output_rows)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_output = args.output_dir / "fixed_pool_scores.jsonl"
    atomic_write_text(
        predictions_output,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
    )
    pool_projection = [
        {"index": index, "pool_skill_ids": pools[index]} for index in indices
    ]
    manifest = {
        "artifact_format": "legacy_core75_fixed_pool_score_shard_v2",
        "label": args.label,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "rows": len(output_rows),
        "expected_total_rows": args.expected_rows,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_file_sha256": observed_checkpoint_sha,
        "checkpoint_method_setting": checkpoint.get("method_setting"),
        "model_inputs": model_inputs,
        "semantic_ids": str(args.semantic_ids),
        "semantic_ids_file_sha256": file_sha256(args.semantic_ids),
        "semantic_ids_logical_sha256": pipeline.semantic_ids_sha256(
            semantic_ids, semantic_id_mode
        ),
        "rows_path": str(args.rows),
        "rows_sha256": file_sha256(args.rows),
        "prediction_inputs": {
            label: {"path": str(path), "sha256": file_sha256(path)}
            for label, path in prediction_paths.items()
        },
        "candidate_pool": {
            "definition": "sorted union of four beam-order Top50 lists plus all gold",
            "global_sha256": stable_json_sha256(pool_projection),
            "minimum_size": min(map(len, pools.values())),
            "maximum_size": max(map(len, pools.values())),
            "mean_size": math.fsum(map(len, pools.values())) / len(pools),
            "gold_coverage": 1.0,
        },
        "scoring": {
            "decoder_head": "ce",
            "score_transform": "full_vocabulary_logprob",
            "components": "four learned-code tokens plus stop control=1",
            "component_labels": [
                *[
                    f"code_position_{position}"
                    for position in range(int(sid_config["depth"]))
                ],
                "stop_control_1",
            ],
            "trie_child_renormalization": False,
            "max_skills": 1,
            "exhaustive_capacity": "per-query pool size",
            "max_source_length": args.max_source_length,
        },
        "score_replay_validation": {
            "available": bool(validation_errors),
            "queries": len(validation_errors),
            "maximum_absolute_error": max(validation_errors, default=None),
            "atol": args.score_atol,
            "rank_set_checks": {
                f"@{cutoff}": (
                    all(checks[f"@{cutoff}"] for checks in validation_rank_checks)
                    if validation_rank_checks
                    else None
                )
                for cutoff in (1, 5, 10, 50)
            },
        },
        "implementation": {
            "launch_commit": os.environ.get("SKILLGR_LAUNCH_COMMIT"),
            "scorer_sha256": file_sha256(Path(__file__).resolve()),
            "pipeline_sha256": file_sha256(Path(pipeline.__file__).resolve()),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "transformers": metadata.version("transformers"),
            "tokenizers": metadata.version("tokenizers"),
            "gpu_name": torch.cuda.get_device_name(device),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    manifest["fixed_pool_scores_sha256"] = file_sha256(predictions_output)
    atomic_write_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    (args.output_dir / "shard.done").touch()
    print(json.dumps({"completed": manifest}, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--expected-rows", type=int, default=75)
    parser.add_argument("--prediction-depth", type=int, default=50)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--score-atol", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    command_score(build_parser().parse_args())


if __name__ == "__main__":
    main()
