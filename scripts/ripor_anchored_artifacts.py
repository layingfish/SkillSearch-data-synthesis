#!/usr/bin/env python3
"""Seal RIPOR self-negatives and build the preregistered anchored evaluation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

from skill_gr_pipeline import (
    compute_at_k_metrics,
    file_sha256,
    load_jsonl,
    load_ranked_skill_candidates_with_scores,
    prepared_selected_skill_ids,
    retrieval_row_keys,
    row_query_text,
    write_json,
    write_jsonl,
)


def _assert_aligned(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], label: str) -> None:
    if len(rows) != len(predictions):
        raise ValueError(f"{label} row count mismatch: rows={len(rows)}, predictions={len(predictions)}")
    for position, (row, prediction) in enumerate(zip(rows, predictions)):
        if str(prediction.get("query") or "") != row_query_text(row):
            raise ValueError(f"{label} query mismatch at position={position}")
        if prediction.get("index") != row.get("index"):
            raise ValueError(
                f"{label} index mismatch at position={position}: "
                f"prediction={prediction.get('index')!r}, row={row.get('index')!r}"
            )


def _resolved(path: Path) -> str:
    return str(path.resolve(strict=True))


def _assert_manifest_path(manifest: dict[str, Any], key: str, expected: Path) -> None:
    recorded = manifest.get(key)
    if not recorded or _resolved(Path(str(recorded))) != _resolved(expected):
        raise ValueError(
            f"self-negative manifest {key} mismatch: recorded={recorded!r}, expected={str(expected)!r}"
        )


def _assert_manifest_sha(manifest: dict[str, Any], key: str, path: Path) -> None:
    actual = file_sha256(path)
    if str(manifest.get(key) or "") != actual:
        raise ValueError(
            f"self-negative manifest {key} mismatch: recorded={manifest.get(key)!r}, actual={actual}"
        )


def command_seal_self_negatives(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    predictions = load_jsonl(args.miner_predictions)
    _assert_aligned(rows, predictions, "M3 miner")
    eval_manifest = json.loads(args.miner_eval_manifest.read_text(encoding="utf-8"))
    if eval_manifest.get("decoder_head") != "ripor" or eval_manifest.get("score_transform") != "raw":
        raise ValueError("self-negatives must come from the frozen RIPOR raw-score decoder")
    if eval_manifest.get("docid_control_score") != "excluded":
        raise ValueError("self-negative miner included a structural control score")
    if _resolved(Path(str(eval_manifest.get("model_dir") or ""))) != _resolved(args.miner_checkpoint):
        raise ValueError(
            "self-negative miner manifest was not produced by the declared M3 checkpoint: "
            f"manifest={eval_manifest.get('model_dir')!r}, checkpoint={str(args.miner_checkpoint)!r}"
        )
    identities = [(str(row.get("index")), row_query_text(row)) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("M3 mining source rows contain duplicate (index, query) identities")
    _, teacher_score_map = load_ranked_skill_candidates_with_scores(args.teacher_panel)

    panel: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    skipped_no_scored_negative = 0
    excluded_sibling_mentions = 0
    total_self_negatives = 0
    for row, prediction in zip(rows, predictions):
        positives = [str(value) for value in prepared_selected_skill_ids(row)]
        source_gold = row.get("source_selected_skill_ids")
        if source_gold is None:
            source_gold = row.get("selected_skill_ids", [])
        exclusion = {str(value) for value in source_gold or []}
        exclusion.update(positives)
        row_scores: dict[str, float] = {}
        for key in retrieval_row_keys(row):
            row_scores.update(teacher_score_map.get(key, {}))
        missing_positive_scores = [sid for sid in positives if sid not in row_scores]
        if missing_positive_scores:
            raise ValueError(
                "frozen teacher panel is missing a positive score for M4: "
                f"row={row.get('index')}, skills={missing_positive_scores[:5]}"
            )
        negatives: list[str] = []
        for sid_value in prediction.get("ranked_skill_ids", []) or []:
            sid = str(sid_value)
            if sid in exclusion:
                excluded_sibling_mentions += 1
                continue
            score = row_scores.get(sid)
            if score is None or not math.isfinite(float(score)):
                continue
            if sid not in negatives:
                negatives.append(sid)
            if len(negatives) >= args.max_negatives:
                break
        if not negatives:
            skipped_no_scored_negative += 1
            continue
        ordered_ids = [*negatives, *[sid for sid in positives if sid not in negatives]]
        panel.append(
            {
                "index": row.get("index"),
                "query": row_query_text(row),
                "gold_skill_ids": positives,
                "ranked_skill_ids": ordered_ids,
                "ranked_scores": [float(row_scores[sid]) for sid in ordered_ids],
                "candidate_source": "frozen_m3_ripor_raw_intersect_frozen_teacher",
            }
        )
        eligible_rows.append(row)
        total_self_negatives += len(negatives)

    if not panel:
        raise ValueError("M3 mining produced no teacher-scored self-negative example")
    args.output_panel.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_panel, panel)
    write_jsonl(args.output_rows, eligible_rows)
    manifest = {
        "artifact_format": "fixed_letter_ripor_self_negatives_v1",
        "rows": str(args.rows),
        "rows_sha256": file_sha256(args.rows),
        "source_query_count": len(rows),
        "unique_mined_query_count": len(predictions),
        "eligible_query_count": len(eligible_rows),
        "skipped_no_scored_negative": skipped_no_scored_negative,
        "total_self_negatives": total_self_negatives,
        "excluded_sibling_mentions": excluded_sibling_mentions,
        "miner_predictions": str(args.miner_predictions),
        "miner_predictions_sha256": file_sha256(args.miner_predictions),
        "miner_eval_manifest": str(args.miner_eval_manifest),
        "miner_eval_manifest_sha256": file_sha256(args.miner_eval_manifest),
        "miner_checkpoint": str(args.miner_checkpoint),
        "miner_checkpoint_sha256": file_sha256(args.miner_checkpoint / "structured_decoder.pt"),
        "teacher_panel": str(args.teacher_panel),
        "teacher_panel_sha256": file_sha256(args.teacher_panel),
        "max_negatives_per_query": args.max_negatives,
        "output_panel": str(args.output_panel),
        "output_panel_sha256": file_sha256(args.output_panel),
        "output_rows": str(args.output_rows),
        "output_rows_sha256": file_sha256(args.output_rows),
        "merge_order": "original_source_row_position",
        "missing_teacher_score_policy": "skip_negative_and_audit; missing_positive_is_fatal",
        "sibling_gt_policy": "exclude_all_before_pair_construction",
    }
    write_json(args.output_manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def command_verify_self_negatives(args: argparse.Namespace) -> None:
    """Fail closed unless M4 inputs are the exact artifacts sealed from M3."""
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("artifact_format") != "fixed_letter_ripor_self_negatives_v1":
        raise ValueError("unexpected self-negative artifact format")
    path_fields = {
        "rows": args.rows,
        "miner_predictions": args.miner_predictions,
        "miner_eval_manifest": args.miner_eval_manifest,
        "miner_checkpoint": args.miner_checkpoint,
        "teacher_panel": args.teacher_panel,
        "output_panel": args.output_panel,
        "output_rows": args.output_rows,
    }
    for key, expected in path_fields.items():
        _assert_manifest_path(manifest, key, expected)
    hash_fields = {
        "rows_sha256": args.rows,
        "miner_predictions_sha256": args.miner_predictions,
        "miner_eval_manifest_sha256": args.miner_eval_manifest,
        "miner_checkpoint_sha256": args.miner_checkpoint / "structured_decoder.pt",
        "teacher_panel_sha256": args.teacher_panel,
        "output_panel_sha256": args.output_panel,
        "output_rows_sha256": args.output_rows,
    }
    for key, expected in hash_fields.items():
        _assert_manifest_sha(manifest, key, expected)

    eval_manifest = json.loads(args.miner_eval_manifest.read_text(encoding="utf-8"))
    if _resolved(Path(str(eval_manifest.get("model_dir") or ""))) != _resolved(args.miner_checkpoint):
        raise ValueError("verified miner eval manifest is not bound to the declared M3 checkpoint")
    if eval_manifest.get("decoder_head") != "ripor" or eval_manifest.get("score_transform") != "raw":
        raise ValueError("verified miner eval manifest does not use the raw RIPOR head")
    if eval_manifest.get("docid_control_score") != "excluded":
        raise ValueError("verified miner eval manifest included a structural control score")

    rows = load_jsonl(args.rows)
    predictions = load_jsonl(args.miner_predictions)
    panel = load_jsonl(args.output_panel)
    eligible_rows = load_jsonl(args.output_rows)
    _assert_aligned(rows, predictions, "verified M3 miner")
    _assert_aligned(eligible_rows, panel, "verified M4 panel")
    identities = [(str(row.get("index")), row_query_text(row)) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("verified M3 source rows contain duplicate identities")
    expected_counts = {
        "source_query_count": len(rows),
        "unique_mined_query_count": len(set(identities)),
        "eligible_query_count": len(eligible_rows),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_counts.items()
        if int(manifest.get(key, -1)) != value
    }
    if mismatches:
        raise ValueError(f"self-negative manifest count mismatch: {mismatches}")
    print(json.dumps({"verified": True, **expected_counts}, sort_keys=True))


def _score_map(prediction: dict[str, Any]) -> dict[str, float]:
    raw = prediction.get("ranked_skill_scores") or {}
    if not isinstance(raw, dict):
        raise ValueError("anchored evaluation requires per-skill ranked_skill_scores")
    result = {str(key): float(value) for key, value in raw.items() if math.isfinite(float(value))}
    missing = [sid for sid in prediction.get("ranked_skill_ids", []) or [] if str(sid) not in result]
    if missing:
        raise ValueError(f"prediction is missing candidate scores: {missing[:5]}")
    return result


def _normalized(scores: dict[str, float]) -> tuple[dict[str, float], float]:
    values = list(scores.values())
    if not values:
        raise ValueError("cannot normalize an empty candidate score map")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    scale = max(std, 1e-6)
    normalized = {sid: (score - mean) / scale for sid, score in scores.items()}
    return normalized, min(normalized.values()) - 1.0


def _candidate_recall(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> float:
    recalls: list[float] = []
    for row, prediction in zip(rows, predictions):
        gold = set(prepared_selected_skill_ids(row))
        candidates = set(str(value) for value in prediction.get("ranked_skill_ids", []) or [])
        recalls.append(len(gold & candidates) / len(gold) if gold else 0.0)
    return statistics.fmean(recalls) if recalls else 0.0


def command_anchor(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    baseline = load_jsonl(args.baseline_predictions)
    _assert_aligned(rows, baseline, "baseline")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.anchor_weight == 0.0:
        # The gate=0 path is intentionally an exact file copy.  It does not
        # inspect or union RIPOR candidates and therefore proves a real bypass.
        shutil.copy2(args.baseline_predictions, args.output_dir / "predictions.jsonl")
        manifest = {
            "artifact_format": "fixed_letter_ripor_anchor_v1",
            "gate_zero_bypass": True,
            "anchor_weight": 0.0,
            "baseline_predictions_sha256": file_sha256(args.baseline_predictions),
            "output_predictions_sha256": file_sha256(args.output_dir / "predictions.jsonl"),
        }
        if manifest["baseline_predictions_sha256"] != manifest["output_predictions_sha256"]:
            raise RuntimeError("gate=0 did not reproduce the baseline predictions bitwise")
        write_json(args.output_dir / "anchor_manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return

    ripor = load_jsonl(args.ripor_predictions)
    _assert_aligned(rows, ripor, "RIPOR")
    output: list[dict[str, Any]] = []
    baseline_only = ripor_only = both = 0
    union_recalls: list[float] = []
    for row, base_prediction, ripor_prediction in zip(rows, baseline, ripor):
        base_scores, base_floor = _normalized(_score_map(base_prediction))
        ripor_scores, ripor_floor = _normalized(_score_map(ripor_prediction))
        candidates = set(base_scores) | set(ripor_scores)
        ranked_records = []
        gold = set(prepared_selected_skill_ids(row))
        for sid in candidates:
            in_base = sid in base_scores
            in_ripor = sid in ripor_scores
            source = "both" if in_base and in_ripor else ("baseline" if in_base else "ripor")
            baseline_only += int(source == "baseline")
            ripor_only += int(source == "ripor")
            both += int(source == "both")
            score = base_scores.get(sid, base_floor) + args.anchor_weight * ripor_scores.get(sid, ripor_floor)
            ranked_records.append((float(score), sid, source))
        ranked_records.sort(key=lambda item: (-item[0], item[1]))
        ranked_records = ranked_records[: args.max_rank]
        ranked_ids = [sid for _, sid, _ in ranked_records]
        union_recalls.append(len(gold & candidates) / len(gold) if gold else 0.0)
        output.append(
            {
                "index": row.get("index"),
                "query": row_query_text(row),
                "gold_skill_ids": list(prepared_selected_skill_ids(row)),
                "predicted_skill_ids": ranked_ids[:1],
                "ranked_skill_ids": ranked_ids,
                "ranked_skill_scores": {sid: score for score, sid, _ in ranked_records},
                "candidate_sources": {sid: source for _, sid, source in ranked_records},
                "invalid": False,
            }
        )
    write_jsonl(args.output_dir / "predictions.jsonl", output)
    metrics = compute_at_k_metrics(rows, output, [int(value) for value in args.k_values.split(",")])
    write_json(args.output_dir / "ranked_metrics.json", metrics)
    manifest = {
        "artifact_format": "fixed_letter_ripor_anchor_v1",
        "gate_zero_bypass": False,
        "anchor_weight": args.anchor_weight,
        "formula": "zscore(B_logprob)+lambda*zscore(R_raw); missing_branch=min_zscore-1",
        "candidate_domain": "union(B_logprob_topN,R_raw_topN)",
        "candidate_tie_break": "descending anchored score, then ascending skill_id",
        "max_rank": args.max_rank,
        "baseline_predictions": str(args.baseline_predictions),
        "baseline_predictions_sha256": file_sha256(args.baseline_predictions),
        "ripor_predictions": str(args.ripor_predictions),
        "ripor_predictions_sha256": file_sha256(args.ripor_predictions),
        "baseline_candidate_macro_recall": _candidate_recall(rows, baseline),
        "ripor_candidate_macro_recall": _candidate_recall(rows, ripor),
        "union_candidate_macro_recall": statistics.fmean(union_recalls) if union_recalls else 0.0,
        "candidate_source_mentions": {"baseline": baseline_only, "ripor": ripor_only, "both": both},
    }
    write_json(args.output_dir / "anchor_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    seal = sub.add_parser("seal-self-negatives")
    seal.add_argument("--rows", type=Path, required=True)
    seal.add_argument("--miner-predictions", type=Path, required=True)
    seal.add_argument("--miner-eval-manifest", type=Path, required=True)
    seal.add_argument("--miner-checkpoint", type=Path, required=True)
    seal.add_argument("--teacher-panel", type=Path, required=True)
    seal.add_argument("--output-panel", type=Path, required=True)
    seal.add_argument("--output-rows", type=Path, required=True)
    seal.add_argument("--output-manifest", type=Path, required=True)
    seal.add_argument("--max-negatives", type=int, default=16)
    seal.set_defaults(func=command_seal_self_negatives)

    verify = sub.add_parser("verify-self-negatives")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--rows", type=Path, required=True)
    verify.add_argument("--miner-predictions", type=Path, required=True)
    verify.add_argument("--miner-eval-manifest", type=Path, required=True)
    verify.add_argument("--miner-checkpoint", type=Path, required=True)
    verify.add_argument("--teacher-panel", type=Path, required=True)
    verify.add_argument("--output-panel", type=Path, required=True)
    verify.add_argument("--output-rows", type=Path, required=True)
    verify.set_defaults(func=command_verify_self_negatives)

    anchor = sub.add_parser("anchor")
    anchor.add_argument("--rows", type=Path, required=True)
    anchor.add_argument("--baseline-predictions", type=Path, required=True)
    anchor.add_argument("--ripor-predictions", type=Path, required=True)
    anchor.add_argument("--output-dir", type=Path, required=True)
    anchor.add_argument("--anchor-weight", type=float, required=True)
    anchor.add_argument("--max-rank", type=int, default=50)
    anchor.add_argument("--k-values", default="1,10,50")
    anchor.set_defaults(func=command_anchor)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
