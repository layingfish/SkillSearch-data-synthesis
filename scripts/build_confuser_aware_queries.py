#!/usr/bin/env python3
"""Build confuser-aware pseudo queries from training-side GR near misses."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from build_longtail_pseudo_queries import (
        aggregate_usage,
        apply_task_spec_override,
        call_query_generator,
        canonicalize_query,
        configure_profile_transport,
        exact_leak_reasons,
        load_profile,
        normalize_task_spec_layout,
        parse_single_generated_query_with_mode,
        public_profile,
        response_choice_contents,
        sanitize_skill_references,
        selected_query_stats,
        skill_id,
        skill_name,
        validate_clean_generation_profile,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style test import
    from scripts.build_longtail_pseudo_queries import (
        aggregate_usage,
        apply_task_spec_override,
        call_query_generator,
        canonicalize_query,
        configure_profile_transport,
        exact_leak_reasons,
        load_profile,
        normalize_task_spec_layout,
        parse_single_generated_query_with_mode,
        public_profile,
        response_choice_contents,
        sanitize_skill_references,
        selected_query_stats,
        skill_id,
        skill_name,
        validate_clean_generation_profile,
    )


TASK5_PREFIX = "pseudo:catalog_task:"
MODE3_PREFIX = "pseudo:zero_multi_mode:"
CONTRASTIVE_PREFIX = "pseudo:confuser_contrastive:"
CONTRASTIVE_SOURCE = "training_gr_confuser_contrastive_pseudo_query"
PROMPT_VERSION = "gr_confuser_boundary_v1_20260814"
MATERIALIZATION_VERSION = "confuser_task_layout_v1_20260814"
SELECTION_BASIS = "zero_multi_training_gr_nearmiss"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_ids_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    ids = [str(row.get("skill_id") or row.get("positive_skill_id") or "") for row in rows]
    if any(not sid for sid in ids):
        raise ValueError("target skill rows must contain skill_id")
    if len(ids) != len(set(ids)):
        raise ValueError("target skill ids must be unique")
    return ids


def selected_single_skill(row: dict[str, Any]) -> str | None:
    selected = [str(value) for value in row.get("selected_skill_ids", []) or []]
    if len(selected) != 1:
        return None
    positive = str(row.get("positive_skill_id") or selected[0])
    if positive != selected[0]:
        raise ValueError(
            f"positive/selected mismatch for row index={row.get('index')}: {positive} != {selected[0]}"
        )
    return positive


def task5_source_skill(row_index: str) -> str | None:
    if not row_index.startswith(TASK5_PREFIX):
        return None
    suffix = row_index[len(TASK5_PREFIX) :]
    owner, separator, rank = suffix.rpartition(":")
    if not separator or not owner or not rank.isdigit():
        raise ValueError(f"invalid task5 pseudo index: {row_index}")
    return owner


def prepare_mining_rows(
    *,
    augmented_rows_path: Path,
    target_skills_path: Path,
    output_dir: Path,
    expected_targets: int,
    task5_per_skill: int,
    mode3_per_skill: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite mining preparation: {output_dir}")
    target_ids = target_ids_from_rows(load_jsonl(target_skills_path))
    if len(target_ids) != expected_targets:
        raise ValueError(f"target count mismatch: expected={expected_targets}, observed={len(target_ids)}")
    target_set = set(target_ids)
    task5: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mode3: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alias_task5_rows = 0
    for row in load_jsonl(augmented_rows_path):
        positive = selected_single_skill(row)
        if positive not in target_set:
            continue
        row_index = str(row.get("index") or "")
        owner = task5_source_skill(row_index)
        if owner is not None:
            if owner == positive:
                task5[positive].append(row)
            else:
                alias_task5_rows += 1
            continue
        if row_index.startswith(MODE3_PREFIX):
            mode3[positive].append(row)

    selected_rows: list[dict[str, Any]] = []
    per_skill_counts: dict[str, dict[str, int]] = {}
    for sid in sorted(target_ids):
        task_rows = sorted(task5.get(sid, []), key=lambda row: str(row.get("index") or ""))
        mode_rows = sorted(mode3.get(sid, []), key=lambda row: str(row.get("index") or ""))
        if len(task_rows) != task5_per_skill:
            raise ValueError(
                f"canonical task5 count mismatch for {sid}: expected={task5_per_skill}, observed={len(task_rows)}"
            )
        if len(mode_rows) != mode3_per_skill:
            raise ValueError(
                f"mode3 count mismatch for {sid}: expected={mode3_per_skill}, observed={len(mode_rows)}"
            )
        mode_names = [str((row.get("pseudo_query") or {}).get("mode") or "") for row in mode_rows]
        if len(set(mode_names)) != mode3_per_skill:
            raise ValueError(f"mode3 modes are not unique for {sid}: {mode_names}")
        for source_kind, row in [
            *(("task5", item) for item in task_rows),
            *(("mode3", item) for item in mode_rows),
        ]:
            source_index = str(row.get("index") or "")
            selected_rows.append(
                {
                    **row,
                    "index": len(selected_rows),
                    "source_index": source_index,
                    "confuser_mining": {
                        "source_kind": source_kind,
                        "selection_basis": SELECTION_BASIS,
                    },
                }
            )
        per_skill_counts[sid] = {"task5": len(task_rows), "mode3": len(mode_rows)}

    expected_rows = expected_targets * (task5_per_skill + mode3_per_skill)
    if len(selected_rows) != expected_rows:
        raise ValueError(f"mining row count mismatch: expected={expected_rows}, observed={len(selected_rows)}")
    output_dir.mkdir(parents=True)
    output_path = output_dir / "mining_rows.jsonl"
    write_jsonl(output_path, selected_rows)
    manifest = {
        "format": "confuser_aware_mining_rows_v1",
        "selection_basis": SELECTION_BASIS,
        "augmented_rows": str(augmented_rows_path),
        "augmented_rows_sha256": file_sha256(augmented_rows_path),
        "target_skills": str(target_skills_path),
        "target_skills_sha256": file_sha256(target_skills_path),
        "target_skill_count": len(target_ids),
        "task5_per_skill": task5_per_skill,
        "mode3_per_skill": mode3_per_skill,
        "expected_rows": expected_rows,
        "selected_rows": len(selected_rows),
        "discarded_alias_task5_rows": alias_task5_rows,
        "all_skills_exact": all(
            counts == {"task5": task5_per_skill, "mode3": mode3_per_skill}
            for counts in per_skill_counts.values()
        ),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def aggregate_confusers(
    *,
    mining_rows_path: Path,
    candidates_path: Path,
    candidate_manifest_path: Path,
    catalog_path: Path,
    target_skills_path: Path,
    output_dir: Path,
    confusers_per_skill: int,
    expected_views_per_skill: int,
    prediction_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite confuser aggregation: {output_dir}")
    targets = target_ids_from_rows(load_jsonl(target_skills_path))
    target_set = set(targets)
    catalog_ids = {skill_id(row) for row in load_jsonl(catalog_path)}
    if not target_set <= catalog_ids:
        raise ValueError(f"target skills missing from catalog: {sorted(target_set - catalog_ids)[:10]}")
    mining_rows = load_jsonl(mining_rows_path)
    by_source: dict[str, dict[str, Any]] = {}
    target_view_counts: Counter[str] = Counter()
    for row in mining_rows:
        source_index = str(row.get("source_index") or "")
        positive = selected_single_skill(row)
        if not source_index or positive not in target_set:
            raise ValueError(f"invalid mining row: index={row.get('index')} source={source_index}")
        if source_index in by_source:
            raise ValueError(f"duplicate mining source index: {source_index}")
        by_source[source_index] = row
        target_view_counts[positive] += 1
    bad_views = {sid: target_view_counts[sid] for sid in targets if target_view_counts[sid] != expected_views_per_skill}
    if bad_views:
        raise ValueError(f"target mining view count mismatch: {list(bad_views.items())[:10]}")

    reciprocal_scores: dict[str, Counter[str]] = defaultdict(Counter)
    occurrences: dict[str, Counter[str]] = defaultdict(Counter)
    best_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    candidates_by_source: dict[str, list[str]] = {}
    seen_sources: set[str] = set()
    for candidate in load_jsonl(candidates_path):
        source_index = str(candidate.get("source_index") or "")
        source_row = by_source.get(source_index)
        if source_row is None:
            raise ValueError(f"candidate references unknown mining source: {source_index}")
        if source_index in seen_sources:
            raise ValueError(f"duplicate candidate source index: {source_index}")
        seen_sources.add(source_index)
        positive = selected_single_skill(source_row)
        if str(candidate.get("positive_skill_id") or "") != positive:
            raise ValueError(f"candidate positive mismatch for source={source_index}")
        negatives = [str(value) for value in candidate.get("negative_skill_ids", []) or []]
        if len(negatives) != len(set(negatives)) or positive in negatives:
            raise ValueError(f"invalid negative list for source={source_index}")
        unknown = [sid for sid in negatives if sid not in catalog_ids]
        if unknown:
            raise ValueError(f"unknown negative skills for source={source_index}: {unknown[:5]}")
        candidates_by_source[source_index] = negatives
        for rank, negative in enumerate(negatives, start=1):
            reciprocal_scores[positive][negative] += 1.0 / rank
            occurrences[positive][negative] += 1
            best_ranks[positive][negative] = min(best_ranks[positive].get(negative, rank), rank)
    if seen_sources != set(by_source):
        missing = sorted(set(by_source) - seen_sources)
        raise ValueError(f"candidate closure is missing {len(missing)} mining rows: {missing[:5]}")

    top1_error_counts: dict[str, Counter[str]] = defaultdict(Counter)
    gold_top1_views: Counter[str] = Counter()
    prediction_records = 0
    if prediction_paths:
        reciprocal_scores = defaultdict(Counter)
        occurrences = defaultdict(Counter)
        best_ranks = defaultdict(dict)
        mining_by_index = {int(row["index"]): row for row in mining_rows}
        seen_prediction_indices: set[int] = set()
        for prediction_path in prediction_paths:
            for prediction in load_jsonl(prediction_path):
                row_index = int(prediction["index"])
                if row_index in seen_prediction_indices:
                    raise ValueError(f"duplicate prediction row index: {row_index}")
                source_row = mining_by_index.get(row_index)
                if source_row is None:
                    raise ValueError(f"prediction references unknown mining row index: {row_index}")
                seen_prediction_indices.add(row_index)
                prediction_records += 1
                source_index = str(source_row["source_index"])
                positive = selected_single_skill(source_row)
                gold = [str(value) for value in prediction.get("gold_skill_ids", []) or []]
                if gold != [positive]:
                    raise ValueError(f"prediction gold mismatch at row index={row_index}: {gold} != {[positive]}")
                ranked = [str(value) for value in prediction.get("ranked_skill_ids", []) or []]
                negatives = candidates_by_source[source_index]
                observed_negatives = [sid for sid in ranked if sid != positive][: len(negatives)]
                if observed_negatives != negatives:
                    raise ValueError(
                        f"prediction/candidate ordering mismatch at row index={row_index}: "
                        f"prediction={observed_negatives[:3]}, candidates={negatives[:3]}"
                    )
                if ranked and ranked[0] == positive:
                    gold_top1_views[positive] += 1
                elif ranked:
                    top1_error_counts[positive][ranked[0]] += 1
                absolute_ranks = {sid: rank for rank, sid in enumerate(ranked, start=1)}
                for negative in negatives:
                    rank = absolute_ranks[negative]
                    reciprocal_scores[positive][negative] += 1.0 / rank
                    occurrences[positive][negative] += 1
                    best_ranks[positive][negative] = min(
                        best_ranks[positive].get(negative, rank), rank
                    )
        if seen_prediction_indices != set(mining_by_index):
            missing = sorted(set(mining_by_index) - seen_prediction_indices)
            raise ValueError(
                f"prediction closure is missing {len(missing)} mining rows: {missing[:5]}"
            )

    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if Path(str(candidate_manifest.get("output") or "")).resolve() != candidates_path.resolve():
        raise ValueError("candidate manifest output path does not match candidates")
    pairs: list[dict[str, Any]] = []
    for target in sorted(targets):
        ranked = sorted(
            reciprocal_scores[target],
            key=lambda negative: (
                -(top1_error_counts[target][negative] > 0),
                -top1_error_counts[target][negative],
                -reciprocal_scores[target][negative],
                -occurrences[target][negative],
                best_ranks[target][negative],
                negative,
            ),
        )
        if len(ranked) < confusers_per_skill:
            raise ValueError(
                f"insufficient aggregate confusers for {target}: expected={confusers_per_skill}, observed={len(ranked)}"
            )
        for pair_rank, confuser in enumerate(ranked[:confusers_per_skill], start=1):
            pairs.append(
                {
                    "pair_index": len(pairs),
                    "target_skill_id": target,
                    "confuser_skill_id": confuser,
                    "confuser_rank": pair_rank,
                    "reciprocal_rank_sum": reciprocal_scores[target][confuser],
                    "supporting_view_count": occurrences[target][confuser],
                    "best_view_rank": best_ranks[target][confuser],
                    "top1_error_view_count": top1_error_counts[target][confuser],
                    "target_gold_top1_view_count": gold_top1_views[target],
                    "target_view_count": target_view_counts[target],
                    "selection_basis": SELECTION_BASIS,
                }
            )
    expected_pairs = len(targets) * confusers_per_skill
    if len(pairs) != expected_pairs:
        raise ValueError(f"pair count mismatch: expected={expected_pairs}, observed={len(pairs)}")
    output_dir.mkdir(parents=True)
    output_path = output_dir / "confuser_pairs.jsonl"
    write_jsonl(output_path, pairs)
    manifest = {
        "format": "confuser_aware_skill_pairs_v1",
        "selection_basis": SELECTION_BASIS,
        "mining_rows": str(mining_rows_path),
        "mining_rows_sha256": file_sha256(mining_rows_path),
        "candidates": str(candidates_path),
        "candidates_sha256": file_sha256(candidates_path),
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "mining_checkpoint_sha256": candidate_manifest.get("mining_checkpoint_sha256"),
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "target_skill_count": len(targets),
        "views_per_target": expected_views_per_skill,
        "confusers_per_target": confusers_per_skill,
        "pair_count": len(pairs),
        "ranking": (
            "top1_error_first_then_absolute_reciprocal_rank_v1"
            if prediction_paths
            else "nongold_reciprocal_rank_v1"
        ),
        "predictions": [str(path) for path in prediction_paths or []],
        "prediction_sha256": [file_sha256(path) for path in prediction_paths or []],
        "prediction_records": prediction_records,
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def skill_excerpt(skill: dict[str, Any], body_chars: int) -> str:
    description = str(skill.get("description") or skill.get("task_description") or "")[:300]
    body = str(skill.get("body") or "")[: min(max(body_chars, 0), 2500)]
    return f"Name: {skill_name(skill)}\nDescription: {description}\nBody:\n{body}"


def build_contrastive_messages(
    target: dict[str, Any],
    confuser: dict[str, Any],
    *,
    target_body_chars: int,
    confuser_body_chars: int,
) -> list[dict[str, str]]:
    prompt = (
        "Generate one synthetic retrieval query for training a single-skill generative retriever. "
        "The positive document is the required capability. The confuser document is a common wrong retrieval.\n\n"
        "Positive private document:\n"
        f"{skill_excerpt(target, target_body_chars)}\n\n"
        "Confuser private document:\n"
        f"{skill_excerpt(confuser, confuser_body_chars)}\n\n"
        "Write a realistic executable task that is fully supported by the positive document and cannot be completed correctly "
        "by relying only on the confuser. Include at least one operational requirement, artifact constraint, or verification step "
        "that exposes this capability boundary. The user must ask for the positive capability directly; do not describe a "
        "comparison, rejection, retrieval decision, or taxonomy exercise.\n\n"
        "Requirements:\n"
        "- The task must have exactly one specialized retrieval target and be solvable using the positive document alone.\n"
        "- Open with one or two sentences giving a concrete situation and available input, artifact, or current state.\n"
        "- Then write `You need to:` followed by 2-5 numbered requirements in execution order.\n"
        "- End with an explicit deliverable and a supported acceptance or verification criterion.\n"
        "- Use neutral placeholders for specialized values absent from the positive document; never fabricate them.\n"
        "- Do not mention either skill name, either skill id, retrieval, a confuser, or the existence of a skill document.\n"
        "- Do not request capabilities, tools, APIs, formats, or evaluations supported only by the confuser.\n"
        "- Aim for 800-1,500 characters, but do not pad the task with unsupported work.\n\n"
        "Return JSON only with exactly this schema:\n"
        '{"query": "..."}'
    )
    return [{"role": "user", "content": prompt}]


def contrastive_work_key(pair: dict[str, Any]) -> str:
    return f"{pair['target_skill_id']}\t{pair['confuser_skill_id']}"


def build_generation_record(
    *,
    pair: dict[str, Any],
    target: dict[str, Any],
    confuser: dict[str, Any],
    response: dict[str, Any],
    max_query_chars: int,
) -> dict[str, Any]:
    contents = response_choice_contents(response)
    raw_content = contents[0][1] if contents else ""
    parsed_query, parse_mode = parse_single_generated_query_with_mode(raw_content)
    task_query, task_spec_override = apply_task_spec_override(parsed_query, target)
    normalized = normalize_task_spec_layout(task_query)
    sanitized, target_sanitizations = sanitize_skill_references(normalized, target)
    sanitized, confuser_sanitizations = sanitize_skill_references(sanitized, confuser)
    query = canonicalize_query(sanitized, max_query_chars)
    reject_reasons = [] if query else ["empty"]
    if query:
        reject_reasons.extend(f"target:{reason}" for reason in exact_leak_reasons(query, target, []))
        reject_reasons.extend(f"confuser:{reason}" for reason in exact_leak_reasons(query, confuser, []))
    return {
        "schema_version": 1,
        "work_key": contrastive_work_key(pair),
        "target_skill_id": pair["target_skill_id"],
        "target_skill_name": skill_name(target),
        "confuser_skill_id": pair["confuser_skill_id"],
        "confuser_skill_name": skill_name(confuser),
        "confuser_rank": int(pair["confuser_rank"]),
        "status": "complete" if not reject_reasons else "invalid_generation",
        "prompt_version": PROMPT_VERSION,
        "response_model": str(response.get("model") or ""),
        "usage": response.get("usage") or {},
        "query": query,
        "parse_mode": parse_mode,
        "task_spec_override": task_spec_override,
        "sanitized_reference_types": sorted(
            set([*target_sanitizations, *confuser_sanitizations])
        ),
        "reject_reasons": reject_reasons,
    }


def build_error_record(pair: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "work_key": contrastive_work_key(pair),
        "target_skill_id": pair["target_skill_id"],
        "confuser_skill_id": pair["confuser_skill_id"],
        "confuser_rank": int(pair["confuser_rank"]),
        "status": "api_error",
        "prompt_version": PROMPT_VERSION,
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
    }


def latest_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("work_key") or "")
        if key:
            latest[key] = record
    return latest


def load_forced_retry_keys(path: Path | None, pairs: list[dict[str, Any]]) -> set[str]:
    if path is None:
        return set()
    keys = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(keys) != len(set(keys)):
        raise ValueError(f"forced retry work keys are not unique: {path}")
    known = {contrastive_work_key(pair) for pair in pairs}
    unknown = sorted(set(keys) - known)
    if unknown:
        raise ValueError(f"forced retry work keys are outside the pair closure: {unknown[:5]}")
    return set(keys)


def materialize_generation(
    *,
    output_dir: Path,
    pairs: list[dict[str, Any]],
    profile: dict[str, Any],
    run_config: dict[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    records_path = output_dir / "contrastive_generations.jsonl"
    records = load_jsonl(records_path) if records_path.exists() else []
    latest = latest_records(records)
    attempts = Counter(str(record.get("work_key") or "") for record in records)
    selected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for pair in pairs:
        key = contrastive_work_key(pair)
        record = latest.get(key)
        if record is None or record.get("status") != "complete":
            failures.append(
                {
                    "work_key": key,
                    "target_skill_id": pair["target_skill_id"],
                    "confuser_skill_id": pair["confuser_skill_id"],
                    "status": str((record or {}).get("status") or "missing"),
                    "attempts": attempts[key],
                    "retryable": attempts[key] < max_attempts,
                }
            )
            continue
        target = str(pair["target_skill_id"])
        confuser = str(pair["confuser_skill_id"])
        rank = int(pair["confuser_rank"])
        query = str(record["query"])
        selected.append(
            {
                "index": f"{CONTRASTIVE_PREFIX}{target}:{rank - 1}",
                "source": CONTRASTIVE_SOURCE,
                "query": query,
                "query_text": (
                    "Instruct: Given a task description, retrieve the most relevant skill document that would help an agent "
                    f"complete the task\nQuery:{query}"
                ),
                "target_skill_count": 1,
                "selected_skill_ids": [target],
                "positive_skill_id": target,
                "pseudo_query": {
                    "mode": "confuser_contrastive",
                    "rank": rank - 1,
                    "clean_catalog": True,
                    "selection_basis": SELECTION_BASIS,
                    "prompt_version": PROMPT_VERSION,
                    "materialization_version": MATERIALIZATION_VERSION,
                    "generator_model": str(record.get("response_model") or profile.get("model") or ""),
                    "parse_mode": str(record.get("parse_mode") or ""),
                    "confuser_skill_id": confuser,
                    "confuser_rank": rank,
                    "reference_sanitized": bool(record.get("sanitized_reference_types")),
                    "sanitized_reference_types": record.get("sanitized_reference_types") or [],
                },
            }
        )
    indices = [str(row["index"]) for row in selected]
    if len(indices) != len(set(indices)):
        raise ValueError("materialized contrastive pseudo indices are not unique")
    selected_path = output_dir / "selected_confuser_contrastive.jsonl"
    failures_path = output_dir / "failures.jsonl"
    write_jsonl(selected_path, selected)
    write_jsonl(failures_path, failures)
    expected = len(pairs)
    manifest = {
        "format": "confuser_aware_generation_v1",
        "complete": not failures and len(selected) == expected,
        "selection_basis": SELECTION_BASIS,
        "prompt_version": PROMPT_VERSION,
        "materialization_version": MATERIALIZATION_VERSION,
        "run_config": run_config,
        "profile": public_profile(profile),
        "generation_checkpoint": str(records_path),
        "selected_rows_path": str(selected_path),
        "failures_path": str(failures_path),
        "expected_rows": expected,
        "selected_rows": len(selected),
        "failed_rows": len(failures),
        "all_attempt_usage": aggregate_usage(records),
        "latest_record_usage": aggregate_usage(list(latest.values())),
        "selected_query_stats": selected_query_stats(selected),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def command_prepare_mining(args: argparse.Namespace) -> None:
    manifest = prepare_mining_rows(
        augmented_rows_path=args.augmented_rows,
        target_skills_path=args.target_skills,
        output_dir=args.output_dir,
        expected_targets=args.expected_targets,
        task5_per_skill=args.task5_per_skill,
        mode3_per_skill=args.mode3_per_skill,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def command_aggregate(args: argparse.Namespace) -> None:
    manifest = aggregate_confusers(
        mining_rows_path=args.mining_rows,
        candidates_path=args.candidates,
        candidate_manifest_path=args.candidate_manifest,
        catalog_path=args.skills_jsonl,
        target_skills_path=args.target_skills,
        output_dir=args.output_dir,
        confusers_per_skill=args.confusers_per_skill,
        expected_views_per_skill=args.expected_views_per_skill,
        prediction_paths=args.predictions,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def command_preview(args: argparse.Namespace) -> None:
    pairs = load_jsonl(args.pairs)
    catalog = {skill_id(row): row for row in load_jsonl(args.skills_jsonl)}
    previews = []
    for pair in pairs[: args.limit]:
        target = catalog[str(pair["target_skill_id"])]
        confuser = catalog[str(pair["confuser_skill_id"])]
        previews.append(
            {
                "work_key": contrastive_work_key(pair),
                "messages": build_contrastive_messages(
                    target,
                    confuser,
                    target_body_chars=args.target_body_chars,
                    confuser_body_chars=args.confuser_body_chars,
                ),
            }
        )
    write_jsonl(args.output, previews)
    print(f"saved prompt previews={len(previews)} output={args.output}")


def command_generate(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.max_attempts < 1:
        raise SystemExit("--workers and --max-attempts must be positive")
    pairs = load_jsonl(args.pairs)
    if args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]
    catalog = {skill_id(row): row for row in load_jsonl(args.skills_jsonl)}
    required = {
        str(pair["target_skill_id"])
        for pair in pairs
    } | {str(pair["confuser_skill_id"]) for pair in pairs}
    missing = sorted(required - set(catalog))
    if missing:
        raise SystemExit(f"pair skills missing from catalog: {missing[:10]}")
    profile = load_profile(args.profile)
    validate_clean_generation_profile(profile)
    configure_profile_transport(profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": 1,
        "selection_basis": SELECTION_BASIS,
        "prompt_version": PROMPT_VERSION,
        "pairs": str(args.pairs),
        "pairs_sha256": file_sha256(args.pairs),
        "skills_jsonl": str(args.skills_jsonl),
        "skills_sha256": file_sha256(args.skills_jsonl),
        "pair_count": len(pairs),
        "target_body_chars": args.target_body_chars,
        "confuser_body_chars": args.confuser_body_chars,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_query_chars": args.max_query_chars,
        "model": profile["model"],
    }
    run_config_path = args.output_dir / "run_config.json"
    if run_config_path.exists():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise SystemExit("generation run config differs from the existing resumable output")
    else:
        write_json(run_config_path, run_config)

    records_path = args.output_dir / "contrastive_generations.jsonl"
    records = load_jsonl(records_path) if records_path.exists() else []
    latest = latest_records(records)
    attempts = Counter(str(record.get("work_key") or "") for record in records)
    forced_retry_keys = load_forced_retry_keys(args.force_retry_work_keys, pairs)
    exhausted_forced = sorted(key for key in forced_retry_keys if attempts[key] >= args.max_attempts)
    if exhausted_forced:
        raise SystemExit(
            "forced retry keys have exhausted --max-attempts; increase the limit: "
            f"{exhausted_forced[:5]}"
        )
    forced_retry_attempt_floor = {key: attempts[key] + 1 for key in forced_retry_keys}

    def needs_generation(pair: dict[str, Any]) -> bool:
        key = contrastive_work_key(pair)
        status_incomplete = str((latest.get(key) or {}).get("status") or "") != "complete"
        forced_attempt_pending = attempts[key] < forced_retry_attempt_floor.get(key, 0)
        return (status_incomplete or forced_attempt_pending) and attempts[key] < args.max_attempts

    pending = [
        pair
        for pair in pairs
        if needs_generation(pair)
    ]
    print(
        f"confuser generation pairs={len(pairs)} pending={len(pending)} "
        f"complete={len(pairs) - len(pending)} forced_retry={len(forced_retry_keys)} "
        f"workers={args.workers}",
        flush=True,
    )
    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed = 0

    def run_pair(pair: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        target = catalog[str(pair["target_skill_id"])]
        confuser = catalog[str(pair["confuser_skill_id"])]
        try:
            messages = build_contrastive_messages(
                target,
                confuser,
                target_body_chars=args.target_body_chars,
                confuser_body_chars=args.confuser_body_chars,
            )
            response = call_query_generator(
                profile,
                messages,
                args.temperature,
                args.top_p,
                args.timeout,
                args.max_tokens,
                n=1,
            )
            record = build_generation_record(
                pair=pair,
                target=target,
                confuser=confuser,
                response=response,
                max_query_chars=args.max_query_chars,
            )
        except Exception as exc:  # Resumable transport and parse failure checkpoint.
            record = build_error_record(pair, exc)
        append_jsonl(records_path, record, write_lock)
        with progress_lock:
            completed += 1
            if completed == len(pending) or completed % args.progress_every == 0:
                print(
                    f"progress={completed}/{len(pending)} work_key={record['work_key']} status={record['status']}",
                    flush=True,
                )
        return record

    for attempt_round in range(args.max_attempts):
        if not pending:
            break
        if args.workers == 1:
            for pair in pending:
                run_pair(pair)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(run_pair, pending))
        records = load_jsonl(records_path)
        latest = latest_records(records)
        attempts = Counter(str(record.get("work_key") or "") for record in records)
        pending = [
            pair
            for pair in pairs
            if needs_generation(pair)
        ]
        if pending:
            print(f"retry_round={attempt_round + 1} pending={len(pending)}", flush=True)

    manifest = materialize_generation(
        output_dir=args.output_dir,
        pairs=pairs,
        profile=profile,
        run_config=run_config,
        max_attempts=args.max_attempts,
    )
    print(
        f"selected_rows={manifest['selected_rows']}/{manifest['expected_rows']} "
        f"failed_rows={manifest['failed_rows']}",
        flush=True,
    )
    if args.require_complete and not manifest["complete"]:
        raise SystemExit(2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-mining")
    prepare.add_argument("--augmented-rows", type=Path, required=True)
    prepare.add_argument("--target-skills", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--expected-targets", type=int, default=1722)
    prepare.add_argument("--task5-per-skill", type=int, default=5)
    prepare.add_argument("--mode3-per-skill", type=int, default=3)
    prepare.set_defaults(func=command_prepare_mining)

    aggregate = subparsers.add_parser("aggregate-confusers")
    aggregate.add_argument("--mining-rows", type=Path, required=True)
    aggregate.add_argument("--candidates", type=Path, required=True)
    aggregate.add_argument("--candidate-manifest", type=Path, required=True)
    aggregate.add_argument("--skills-jsonl", type=Path, required=True)
    aggregate.add_argument("--target-skills", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--confusers-per-skill", type=int, default=3)
    aggregate.add_argument("--expected-views-per-skill", type=int, default=8)
    aggregate.add_argument("--predictions", type=Path, action="append")
    aggregate.set_defaults(func=command_aggregate)

    preview = subparsers.add_parser("preview")
    preview.add_argument("--pairs", type=Path, required=True)
    preview.add_argument("--skills-jsonl", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--limit", type=int, default=3)
    preview.add_argument("--target-body-chars", type=int, default=2200)
    preview.add_argument("--confuser-body-chars", type=int, default=1200)
    preview.set_defaults(func=command_preview)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--pairs", type=Path, required=True)
    generate.add_argument("--skills-jsonl", type=Path, required=True)
    generate.add_argument("--profile", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--limit-pairs", type=int, default=-1)
    generate.add_argument("--target-body-chars", type=int, default=2200)
    generate.add_argument("--confuser-body-chars", type=int, default=1200)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--max-tokens", type=int, default=512)
    generate.add_argument("--max-query-chars", type=int, default=1500)
    generate.add_argument("--timeout", type=int, default=180)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--progress-every", type=int, default=25)
    generate.add_argument("--max-attempts", type=int, default=3)
    generate.add_argument(
        "--force-retry-work-keys",
        type=Path,
        help="newline-delimited target<TAB>confuser work keys to regenerate once in this invocation",
    )
    generate.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    generate.set_defaults(func=command_generate)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
