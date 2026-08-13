#!/usr/bin/env python3
"""Score every frozen LETTER SID for each Official core75 query."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import score_legacy_core75_fixed_pool as fixed  # noqa: E402
import skill_gr_pipeline as pipeline  # noqa: E402


def atomic_write_json(path: Path, payload: Any) -> None:
    fixed.atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def load_beam_predictions(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_depth: int,
) -> dict[int, dict[str, Any]]:
    predictions = fixed.read_jsonl(path)
    if len(predictions) != len(rows):
        raise ValueError(
            f"beam prediction row count mismatch: expected={len(rows)}, got={len(predictions)}"
        )
    indexed: dict[int, dict[str, Any]] = {}
    for row_position, (row, prediction) in enumerate(zip(rows, predictions)):
        index = prediction.get("index")
        if not isinstance(index, int) or index in indexed:
            raise ValueError(f"invalid or duplicate beam prediction index={index!r}")
        if index != row.get("index"):
            raise ValueError(
                f"beam prediction order mismatch at row_position={row_position}: "
                f"prediction={index}, row={row.get('index')}"
            )
        ranked = prediction.get("ranked_skill_ids")
        if (
            not isinstance(ranked, list)
            or len(ranked) != expected_depth
            or len(set(ranked)) != expected_depth
            or not all(isinstance(skill_id, str) for skill_id in ranked)
        ):
            raise ValueError(
                f"beam prediction index={index}: expected unique Top-{expected_depth}"
            )
        gold = pipeline.prepared_selected_skill_ids(dict(row))
        if prediction.get("gold_skill_ids") != gold:
            raise ValueError(f"beam prediction index={index}: gold IDs differ from eval rows")
        if prediction.get("query") != pipeline.row_query_text(dict(row)):
            raise ValueError(f"beam prediction index={index}: query differs from eval rows")
        if prediction.get("target_skill_count") != len(gold):
            raise ValueError(f"beam prediction index={index}: target count mismatch")
        indexed[index] = prediction
    return indexed


def ranked_skill_ids(
    scores: Mapping[str, float],
) -> list[str]:
    return sorted(
        scores,
        key=lambda skill_id: (float(scores[skill_id]), skill_id),
        reverse=True,
    )


def prefix_rank_diagnostics(
    semantic_ids: Mapping[str, Sequence[int]],
    score_components: Mapping[str, Sequence[float]],
    gold_skill_ids: Sequence[str],
    *,
    beam_width: int,
    consistency_atol: float = 1e-7,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    if not semantic_ids:
        raise ValueError("semantic ID catalog must not be empty")
    semantic_length = len(next(iter(semantic_ids.values())))
    prefix_ranks_by_gold = {skill_id: [] for skill_id in gold_skill_ids}
    unique_prefix_counts: list[int] = []
    maximum_shared_prefix_spread = 0.0

    for prefix_length in range(1, semantic_length + 1):
        prefix_values: dict[tuple[int, ...], list[float]] = {}
        for skill_id, codes in semantic_ids.items():
            components = score_components[skill_id]
            if len(components) != semantic_length + 1:
                raise ValueError(
                    f"candidate={skill_id}: score component length differs from SID length"
                )
            prefix = tuple(int(value) for value in codes[:prefix_length])
            prefix_score = math.fsum(float(value) for value in components[:prefix_length])
            prefix_values.setdefault(prefix, []).append(prefix_score)

        representative_scores: dict[tuple[int, ...], float] = {}
        for prefix, values in prefix_values.items():
            spread = max(values) - min(values)
            maximum_shared_prefix_spread = max(maximum_shared_prefix_spread, spread)
            if spread > consistency_atol:
                raise ValueError(
                    f"shared prefix={prefix} has score spread={spread} above "
                    f"atol={consistency_atol}"
                )
            representative_scores[prefix] = math.fsum(values) / len(values)
        ordered_prefixes = sorted(
            representative_scores,
            key=lambda prefix: (representative_scores[prefix], prefix),
            reverse=True,
        )
        rank_by_prefix = {
            prefix: rank for rank, prefix in enumerate(ordered_prefixes, start=1)
        }
        unique_prefix_counts.append(len(ordered_prefixes))
        for skill_id in gold_skill_ids:
            gold_prefix = tuple(
                int(value) for value in semantic_ids[skill_id][:prefix_length]
            )
            prefix_ranks_by_gold[skill_id].append(rank_by_prefix[gold_prefix])

    first_over_width = {
        skill_id: next(
            (
                position
                for position, rank in enumerate(ranks, start=1)
                if rank > beam_width
            ),
            None,
        )
        for skill_id, ranks in prefix_ranks_by_gold.items()
    }
    return prefix_ranks_by_gold, {
        "unique_prefix_counts": unique_prefix_counts,
        "maximum_shared_prefix_score_spread": maximum_shared_prefix_spread,
        "shared_prefix_consistency_atol": consistency_atol,
        "first_prefix_position_over_beam_by_gold": first_over_width,
    }


def build_row_diagnostic(
    row_position: int,
    row: Mapping[str, Any],
    beam_prediction: Mapping[str, Any],
    semantic_ids: Mapping[str, Sequence[int]],
    ranked: Sequence[str],
    scores: Mapping[str, float],
    score_components: Mapping[str, Sequence[float]],
    exhaustive_audit: Mapping[str, Any],
    score_replay_validation: Mapping[str, Any],
    *,
    beam_width: int,
    store_top_k: int,
) -> dict[str, Any]:
    gold_skill_ids = pipeline.prepared_selected_skill_ids(dict(row))
    gold_set = set(gold_skill_ids)
    rank_by_skill = {
        skill_id: rank for rank, skill_id in enumerate(ranked, start=1)
    }
    beam_ranked = list(beam_prediction["ranked_skill_ids"])
    beam_rank_by_skill = {
        skill_id: rank for rank, skill_id in enumerate(beam_ranked, start=1)
    }
    prefix_ranks, prefix_audit = prefix_rank_diagnostics(
        semantic_ids,
        score_components,
        gold_skill_ids,
        beam_width=beam_width,
    )
    first_over_width = prefix_audit["first_prefix_position_over_beam_by_gold"]
    gold_diagnostics = []
    for skill_id in gold_skill_ids:
        full_rank = rank_by_skill[skill_id]
        beam_rank = beam_rank_by_skill.get(skill_id)
        if full_rank <= beam_width and beam_rank is not None:
            topk_transition = "both_topk"
        elif full_rank <= beam_width:
            topk_transition = "beam_pruned"
        elif beam_rank is not None:
            topk_transition = "beam_promoted"
        else:
            topk_transition = "both_below_topk"
        gold_diagnostics.append(
            {
                "skill_id": skill_id,
                "semantic_id": list(semantic_ids[skill_id]),
                "full_catalog_rank": full_rank,
                "beam_rank": beam_rank,
                "full_catalog_score": float(scores[skill_id]),
                "score_components": [
                    float(value) for value in score_components[skill_id]
                ],
                "prefix_ranks": list(prefix_ranks[skill_id]),
                "first_prefix_position_over_beam": first_over_width[skill_id],
                "topk_transition": topk_transition,
            }
        )

    full_top1_hit = ranked[0] in gold_set
    beam_top1_hit = beam_ranked[0] in gold_set
    if beam_top1_hit and full_top1_hit:
        top1_diagnosis = "both_hit"
    elif beam_top1_hit:
        top1_diagnosis = "beam_hit_full_miss"
    elif full_top1_hit:
        top1_diagnosis = "beam_pruning"
    else:
        top1_diagnosis = "generator_scoring"
    stored = list(ranked[:store_top_k])
    return {
        "row_position": row_position,
        "index": int(row["index"]),
        "source_index": row.get("source_index"),
        "query": pipeline.row_query_text(dict(row)),
        "gold_skill_ids": gold_skill_ids,
        "target_skill_count": len(gold_skill_ids),
        "catalog_size": len(ranked),
        "catalog_ranking_sha256": fixed.stable_json_sha256(list(ranked)),
        "catalog_ranked_scores_sha256": fixed.stable_json_sha256(
            [[skill_id, float(scores[skill_id])] for skill_id in ranked]
        ),
        "full_catalog_top_skill_ids": stored,
        "full_catalog_top_skill_scores": {
            skill_id: float(scores[skill_id]) for skill_id in stored
        },
        "beam_ranked_skill_ids": beam_ranked,
        "beam_ranked_skill_scores": beam_prediction.get("ranked_skill_scores"),
        "full_catalog_top1_hit": full_top1_hit,
        "beam_top1_hit": beam_top1_hit,
        "top1_diagnosis": top1_diagnosis,
        "gold_diagnostics": gold_diagnostics,
        "prefix_audit": prefix_audit,
        "score_replay_validation": dict(score_replay_validation),
        "exhaustive_audit": dict(exhaustive_audit),
    }


def command_score(args: argparse.Namespace) -> None:
    import torch

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if args.store_top_k < args.beam_width:
        raise ValueError("store-top-k must be at least beam-width")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite shard output: {args.output_dir}")

    checkpoint_path = args.checkpoint_dir / "structured_decoder.pt"
    checkpoint_sha256 = fixed.file_sha256(checkpoint_path)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected={args.checkpoint_sha256}, observed={checkpoint_sha256}"
        )
    rows = pipeline.load_jsonl(args.rows)
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} eval rows, found {len(rows)}")
    semantic_ids, semantic_id_mode = pipeline.load_semantic_ids_with_mode(
        args.semantic_ids
    )
    if len(semantic_ids) != args.expected_catalog_size:
        raise ValueError(
            f"expected catalog size={args.expected_catalog_size}, got={len(semantic_ids)}"
        )
    beam_predictions = load_beam_predictions(
        args.beam_predictions,
        rows,
        expected_depth=args.beam_width,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tokenizer = fixed.load_tokenizer(args.checkpoint_dir)
    model, checkpoint, sid_config = fixed.load_model(
        args.checkpoint_dir,
        args.base_model,
        semantic_ids,
        semantic_id_mode,
        device,
    )
    if int(checkpoint.get("max_source_length", 0)) != args.max_source_length:
        raise ValueError(
            "checkpoint max_source_length differs from full-catalog protocol: "
            f"{checkpoint.get('max_source_length')} != {args.max_source_length}"
        )

    output_rows: list[dict[str, Any]] = []
    replay_errors: list[float] = []
    replay_rank_checks: list[Mapping[str, bool]] = []
    with torch.no_grad():
        for row_position, row in enumerate(rows):
            if row_position % args.num_shards != args.shard_id:
                continue
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
            ranked, scores, components, exhaustive_audit = (
                fixed.exhaustive_score_candidate_pool(
                    model,
                    encoder_hidden_states,
                    attention_mask,
                    semantic_ids,
                    codebook_size=int(sid_config["codebook_size"]),
                    depth=int(sid_config["depth"]),
                    dedup_size=int(sid_config["dedup_size"]),
                    semantic_id_mode=semantic_id_mode,
                    forward_batch_size=args.forward_batch_size,
                    replay_prefix_size=args.beam_width,
                )
            )
            index = int(row["index"])
            score_validation = fixed.validate_archived_scores(
                scores,
                beam_predictions[index],
                atol=args.score_atol,
            )
            replay_errors.append(float(score_validation["max_absolute_error"]))
            replay_rank_checks.append(score_validation["rank_set_checks"])
            output_rows.append(
                build_row_diagnostic(
                    row_position,
                    row,
                    beam_predictions[index],
                    semantic_ids,
                    ranked,
                    scores,
                    components,
                    exhaustive_audit,
                    score_validation,
                    beam_width=args.beam_width,
                    store_top_k=args.store_top_k,
                )
            )
            print(
                json.dumps(
                    {
                        "shard_id": args.shard_id,
                        "row_position": row_position,
                        "catalog_size": len(ranked),
                        "top1_diagnosis": output_rows[-1]["top1_diagnosis"],
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
    score_path = args.output_dir / "full_catalog_scores.jsonl"
    fixed.atomic_write_text(
        score_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
    )
    tokenizer_path = args.checkpoint_dir / "tokenizer.json"
    tokenizer_config_path = args.checkpoint_dir / "tokenizer_config.json"
    manifest = {
        "artifact_format": "legacy_core75_full_catalog_score_shard_v1",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "rows": len(output_rows),
        "expected_total_rows": args.expected_rows,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_file_sha256": checkpoint_sha256,
        "checkpoint_method_setting": checkpoint.get("method_setting"),
        "base_model": {
            "path": str(args.base_model),
            "config_json_sha256": fixed.file_sha256(args.base_model / "config.json"),
            "model_safetensors_sha256": fixed.file_sha256(
                args.base_model / "model.safetensors"
            ),
        },
        "tokenizer": {
            "path": str(args.checkpoint_dir),
            "tokenizer_json_sha256": fixed.file_sha256(tokenizer_path),
            "tokenizer_config_json_sha256": fixed.file_sha256(tokenizer_config_path),
            "tokenizer_semantic_config_sha256": fixed.tokenizer_semantic_config_sha256(
                tokenizer_config_path
            ),
            "tokenized_query_inputs_sha256": fixed.tokenized_query_inputs_sha256(
                tokenizer,
                rows,
                max_source_length=args.max_source_length,
            ),
        },
        "rows_path": str(args.rows),
        "rows_sha256": fixed.file_sha256(args.rows),
        "semantic_ids": str(args.semantic_ids),
        "semantic_ids_file_sha256": fixed.file_sha256(args.semantic_ids),
        "semantic_ids_logical_sha256": pipeline.semantic_ids_sha256(
            semantic_ids, semantic_id_mode
        ),
        "beam_predictions": {
            "path": str(args.beam_predictions),
            "sha256": fixed.file_sha256(args.beam_predictions),
            "width": args.beam_width,
            "rank_aggregation": "beam_order",
        },
        "catalog": {
            "size": len(semantic_ids),
            "definition": "all frozen LETTER semantic IDs",
        },
        "scoring": {
            "decoder_head": "ce",
            "score_transform": "full_vocabulary_logprob",
            "components": "all learned-code tokens plus stop control=1",
            "trie_child_renormalization": False,
            "max_skills": 1,
            "forward_batch_size": args.forward_batch_size,
            "beam_replay_prefix_size": args.beam_width,
            "max_source_length": args.max_source_length,
            "stored_top_k": args.store_top_k,
        },
        "score_replay_validation": {
            "queries": len(replay_errors),
            "maximum_absolute_error": max(replay_errors, default=None),
            "atol": args.score_atol,
            "rank_set_checks": {
                f"@{cutoff}": all(
                    checks[f"@{cutoff}"] for checks in replay_rank_checks
                )
                for cutoff in (1, 5, 10, 50)
            },
        },
        "implementation": {
            "launch_commit": os.environ.get("SKILLGR_LAUNCH_COMMIT"),
            "scorer_sha256": fixed.file_sha256(Path(__file__).resolve()),
            "fixed_pool_scorer_sha256": fixed.file_sha256(Path(fixed.__file__).resolve()),
            "pipeline_sha256": fixed.file_sha256(Path(pipeline.__file__).resolve()),
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
    manifest["full_catalog_scores_sha256"] = fixed.file_sha256(score_path)
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "shard.done").touch()
    print(json.dumps({"completed": manifest}, sort_keys=True), flush=True)


def recall_metrics(
    rows: Sequence[Mapping[str, Any]],
    ranking_field: str,
    cutoffs: Sequence[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        recalls = []
        for row in rows:
            gold = set(row["gold_skill_ids"])
            ranked = row[ranking_field][:cutoff]
            recalls.append(len(gold & set(ranked)) / len(gold))
        metrics[f"@{cutoff}"] = math.fsum(recalls) / len(recalls) if recalls else 0.0
    return metrics


def group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {
        "overall": list(rows),
        "single": [row for row in rows if row["target_skill_count"] == 1],
        "multi": [row for row in rows if row["target_skill_count"] > 1],
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    beam_width: int,
) -> dict[str, Any]:
    cutoffs = (1, 10, 50)
    groups = group_rows(rows)
    metrics: dict[str, Any] = {}
    top1_diagnosis: dict[str, Any] = {}
    gold_transitions: dict[str, Any] = {}
    prefix_failures: dict[str, Any] = {}
    for label, selected in groups.items():
        metrics[label] = {
            "rows": len(selected),
            "full_catalog_macro_recall": recall_metrics(
                selected, "full_catalog_top_skill_ids", cutoffs
            ),
            "beam_macro_recall": recall_metrics(
                selected, "beam_ranked_skill_ids", cutoffs
            ),
        }
        top1_diagnosis[label] = dict(
            sorted(Counter(row["top1_diagnosis"] for row in selected).items())
        )
        transitions = Counter(
            gold["topk_transition"]
            for row in selected
            for gold in row["gold_diagnostics"]
        )
        gold_transitions[label] = {
            "gold_mentions": sum(transitions.values()),
            "at_beam_width": dict(sorted(transitions.items())),
        }
        failures = Counter()
        for row in selected:
            for gold in row["gold_diagnostics"]:
                if gold["beam_rank"] is not None:
                    continue
                position = gold["first_prefix_position_over_beam"]
                failures["none" if position is None else f"c{position - 1}"] += 1
        prefix_failures[label] = dict(sorted(failures.items()))

    replay_errors = [
        float(row["score_replay_validation"]["max_absolute_error"])
        for row in rows
    ]
    replay_checks = [
        row["score_replay_validation"]["rank_set_checks"] for row in rows
    ]
    catalog_sizes = {int(row["catalog_size"]) for row in rows}
    row_positions = [int(row["row_position"]) for row in rows]
    verification = {
        "row_positions_exact": row_positions == list(range(len(rows))),
        "single_catalog_size": len(catalog_sizes) == 1,
        "score_replay_max_absolute_error": max(replay_errors, default=None),
        "score_replay_rank_sets": {
            f"@{cutoff}": all(checks[f"@{cutoff}"] for checks in replay_checks)
            for cutoff in (1, 5, 10, 50)
        },
        "prefix_score_consistency": max(
            float(row["prefix_audit"]["maximum_shared_prefix_score_spread"])
            for row in rows
        ),
    }
    verification["passed"] = (
        verification["row_positions_exact"]
        and verification["single_catalog_size"]
        and all(verification["score_replay_rank_sets"].values())
        and verification["prefix_score_consistency"] <= 1e-7
    )
    return {
        "artifact_format": "legacy_core75_full_catalog_summary_v1",
        "protocol": {
            "queries": len(rows),
            "catalog_size": next(iter(catalog_sizes)) if len(catalog_sizes) == 1 else None,
            "beam_width": beam_width,
            "rank_aggregation": "beam_order",
            "decoder_head": "ce",
            "score_transform": "full_vocabulary_logprob",
        },
        "metrics": metrics,
        "top1_diagnosis": top1_diagnosis,
        "gold_topk_transitions": gold_transitions,
        "first_missing_gold_prefix_over_beam": prefix_failures,
        "verification": verification,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    protocol = summary["protocol"]
    lines = [
        "# Full-catalog generator scoring result",
        "",
        (
            f"Conditions: Official core75={protocol['queries']} queries; "
            f"catalog={protocol['catalog_size']} frozen LETTER SIDs; "
            f"beam={protocol['beam_width']}; beam-order; CE full-vocabulary log-prob."
        ),
        "",
        "| group | rows | exhaustive R@1/R@10/R@50 | beam R@1/R@10/R@50 |",
        "|---|---:|---:|---:|",
    ]
    for label in ("overall", "single", "multi"):
        group = summary["metrics"][label]
        full = group["full_catalog_macro_recall"]
        beam = group["beam_macro_recall"]
        lines.append(
            f"| {label} | {group['rows']} | "
            f"{100 * full['@1']:.4f}/{100 * full['@10']:.4f}/{100 * full['@50']:.4f}% | "
            f"{100 * beam['@1']:.4f}/{100 * beam['@10']:.4f}/{100 * beam['@50']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "Top-1 diagnosis counts:",
            "",
            "```json",
            json.dumps(summary["top1_diagnosis"], indent=2, sort_keys=True),
            "```",
            "",
            "Gold Top-beam transitions:",
            "",
            "```json",
            json.dumps(summary["gold_topk_transitions"], indent=2, sort_keys=True),
            "```",
            "",
            "Verification:",
            "",
            "```json",
            json.dumps(summary["verification"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def command_analyze(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for shard_id in range(args.expected_shards):
        shard_dir = args.input_root / "shards" / f"shard_{shard_id}"
        if not (shard_dir / "shard.done").is_file():
            raise ValueError(f"missing shard completion marker: {shard_dir}")
        manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("artifact_format") != "legacy_core75_full_catalog_score_shard_v1":
            raise ValueError(f"invalid shard manifest format: {shard_dir}")
        if manifest.get("shard_id") != shard_id:
            raise ValueError(f"shard ID mismatch: {shard_dir}")
        if manifest.get("num_shards") != args.expected_shards:
            raise ValueError(f"shard count mismatch: {shard_dir}")
        score_path = shard_dir / "full_catalog_scores.jsonl"
        if fixed.file_sha256(score_path) != manifest.get("full_catalog_scores_sha256"):
            raise ValueError(f"shard score SHA-256 mismatch: {shard_dir}")
        rows.extend(fixed.read_jsonl(score_path))
        manifests.append(manifest)

    rows.sort(key=lambda row: int(row["row_position"]))
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} merged rows, got={len(rows)}")
    if len({row["row_position"] for row in rows}) != len(rows):
        raise ValueError("merged shards contain duplicate row positions")
    common_fields = (
        "checkpoint_file_sha256",
        "base_model",
        "tokenizer",
        "rows_sha256",
        "semantic_ids_file_sha256",
        "semantic_ids_logical_sha256",
        "beam_predictions",
        "catalog",
        "scoring",
        "implementation",
    )
    reference = manifests[0]
    for manifest in manifests[1:]:
        for field in common_fields:
            if manifest.get(field) != reference.get(field):
                raise ValueError(f"shard manifests differ at field={field}")

    summary = summarize_rows(rows, beam_width=args.beam_width)
    summary["inputs"] = {
        "checkpoint_file_sha256": reference["checkpoint_file_sha256"],
        "rows_sha256": reference["rows_sha256"],
        "semantic_ids_file_sha256": reference["semantic_ids_file_sha256"],
        "semantic_ids_logical_sha256": reference["semantic_ids_logical_sha256"],
        "beam_predictions": reference["beam_predictions"],
        "shard_manifest_sha256": [
            fixed.file_sha256(
                args.input_root / "shards" / f"shard_{shard_id}" / "manifest.json"
            )
            for shard_id in range(args.expected_shards)
        ],
    }
    if not summary["verification"]["passed"]:
        raise ValueError(f"full-catalog verification failed: {summary['verification']}")
    atomic_write_json(args.input_root / "full_catalog_summary.json", summary)
    fixed.atomic_write_text(
        args.input_root / "full_catalog_per_query.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    fixed.atomic_write_text(
        args.input_root / "full_catalog_report.md",
        render_markdown(summary),
    )
    (args.input_root / "analysis.done").touch()
    print(json.dumps({"completed": summary}, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--checkpoint-dir", type=Path, required=True)
    score.add_argument("--checkpoint-sha256", required=True)
    score.add_argument("--base-model", type=Path, required=True)
    score.add_argument("--rows", type=Path, required=True)
    score.add_argument("--semantic-ids", type=Path, required=True)
    score.add_argument("--beam-predictions", type=Path, required=True)
    score.add_argument("--shard-id", type=int, required=True)
    score.add_argument("--num-shards", type=int, default=8)
    score.add_argument("--expected-rows", type=int, default=75)
    score.add_argument("--expected-catalog-size", type=int, default=14551)
    score.add_argument("--beam-width", type=int, default=50)
    score.add_argument("--max-source-length", type=int, default=2048)
    score.add_argument("--forward-batch-size", type=int, default=256)
    score.add_argument("--store-top-k", type=int, default=100)
    score.add_argument("--score-atol", type=float, default=5e-5)
    score.add_argument("--output-dir", type=Path, required=True)
    score.set_defaults(func=command_score)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--input-root", type=Path, required=True)
    analyze.add_argument("--expected-shards", type=int, default=8)
    analyze.add_argument("--expected-rows", type=int, default=75)
    analyze.add_argument("--beam-width", type=int, default=50)
    analyze.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
