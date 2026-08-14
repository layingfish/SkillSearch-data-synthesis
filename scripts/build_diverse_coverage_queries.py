#!/usr/bin/env python3
"""Generate training-only scenario-diverse queries for zero-multi SkillGR targets."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from build_longtail_pseudo_queries import (
        aggregate_usage,
        call_query_generator,
        canonicalize_query,
        configure_profile_transport,
        exact_leak_reasons,
        jaccard,
        load_profile,
        normalize_task_spec_layout,
        public_profile,
        sanitize_skill_references,
        selected_query_stats,
        skill_id,
        skill_name,
        token_set,
        validate_clean_generation_profile,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style test import
    from scripts.build_longtail_pseudo_queries import (
        aggregate_usage,
        call_query_generator,
        canonicalize_query,
        configure_profile_transport,
        exact_leak_reasons,
        jaccard,
        load_profile,
        normalize_task_spec_layout,
        public_profile,
        sanitize_skill_references,
        selected_query_stats,
        skill_id,
        skill_name,
        token_set,
        validate_clean_generation_profile,
    )


TASK5_PREFIX = "pseudo:catalog_task:"
MODE3_PREFIX = "pseudo:zero_multi_mode:"
COVERAGE_PREFIX = "pseudo:diverse_coverage:"
COVERAGE_SOURCE = "training_gr_diverse_coverage_pseudo_query"
PROMPT_VERSION = "gr_diverse_coverage_v1_20260814"
MATERIALIZATION_VERSION = "diverse_coverage_layout_v3_20260814"
SELECTION_BASIS = "zero_multi_training_query_coverage"
QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant skill document "
    "that would help an agent complete the task\nQuery:"
)
COVERAGE_MODES = (
    "implicit_outcome",
    "artifact_pipeline",
    "workflow_stage",
    "failure_recovery",
    "verification_first",
    "terse_natural",
)
MODE_REQUIREMENTS = {
    "implicit_outcome": (
        "Describe the desired result while keeping the canonical capability terminology implicit. "
        "The operational clues must still make the private document uniquely useful."
    ),
    "artifact_pipeline": (
        "Center the task on a concrete supported input-to-output artifact flow. Include filenames, schemas, or paths only when "
        "the document supports those artifact types; otherwise use neutral placeholders."
    ),
    "workflow_stage": (
        "Place the capability at one decisive stage of a broader practical workflow. Other stages may only cover ordinary loading, "
        "saving, and verification, so the query still has exactly one specialized retrieval target."
    ),
    "failure_recovery": (
        "Begin from a failed attempt, edge case, or hard operational constraint that requires the capability. Do not invent a "
        "tool-specific error, API, or factual claim absent from the private document."
    ),
    "verification_first": (
        "Foreground a supported acceptance test or observable success condition, then ask for the work needed to satisfy it. "
        "Keep the requested capability and deliverable concrete."
    ),
    "terse_natural": (
        "Write a natural request of 2-4 sentences. Avoid benchmark boilerplate and numbered steps while retaining enough concrete "
        "context to identify the capability."
    ),
}


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


def query_digest(row: dict[str, Any]) -> str:
    query = str(row.get("query") or row.get("query_text") or "")
    if not query:
        raise ValueError(
            f"row has no query text: index={row.get('index')} source_index={row.get('source_index')}"
        )
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def selected_single_skill(row: dict[str, Any]) -> str | None:
    selected = [str(value) for value in row.get("selected_skill_ids", []) or []]
    if len(selected) != 1:
        return None
    positive = str(row.get("positive_skill_id") or selected[0])
    if positive != selected[0]:
        raise ValueError(f"positive/selected mismatch for row index={row.get('index')}")
    return positive


def target_ids_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    targets = [str(row.get("skill_id") or row.get("positive_skill_id") or "") for row in rows]
    if any(not target for target in targets):
        raise ValueError("target rows must contain skill_id")
    if len(targets) != len(set(targets)):
        raise ValueError("target skill ids must be unique")
    return targets


def coverage_row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    source_index = str(row.get("source_index") or row.get("index") or "")
    if not source_index.startswith(COVERAGE_PREFIX):
        raise ValueError(f"invalid diverse-coverage source index: {source_index!r}")
    target_from_index, separator, mode_from_index = source_index[len(COVERAGE_PREFIX) :].rpartition(":")
    if not separator or not target_from_index or mode_from_index not in COVERAGE_MODES:
        raise ValueError(f"invalid diverse-coverage source index: {source_index!r}")
    positive = selected_single_skill(row)
    if positive != target_from_index:
        raise ValueError(
            f"coverage source/positive mismatch: source={source_index} positive={positive}"
        )
    mode = str((row.get("pseudo_query") or {}).get("mode") or "")
    if mode != mode_from_index:
        raise ValueError(
            f"coverage source/mode mismatch: source={source_index} metadata_mode={mode!r}"
        )
    return source_index, positive, mode


def validate_coverage_closure(
    rows: list[dict[str, Any]],
    *,
    expected_targets: int,
    views_per_target: int,
) -> dict[str, list[str]]:
    if views_per_target != len(COVERAGE_MODES):
        raise ValueError(
            f"views_per_target must equal the prescribed mode count: "
            f"{views_per_target} != {len(COVERAGE_MODES)}"
        )
    by_target: dict[str, list[str]] = defaultdict(list)
    source_indices: set[str] = set()
    for row in rows:
        source_index, target, mode = coverage_row_identity(row)
        if source_index in source_indices:
            raise ValueError(f"duplicate diverse-coverage source index: {source_index}")
        source_indices.add(source_index)
        query_digest(row)
        by_target[target].append(mode)
    if len(by_target) != expected_targets:
        raise ValueError(
            f"coverage target count mismatch: observed={len(by_target)} expected={expected_targets}"
        )
    expected_modes = set(COVERAGE_MODES)
    bad_modes = {
        target: modes
        for target, modes in by_target.items()
        if len(modes) != views_per_target or set(modes) != expected_modes
    }
    if bad_modes:
        raise ValueError(f"coverage mode closure failed: {list(bad_modes.items())[:10]}")
    expected_rows = expected_targets * views_per_target
    if len(rows) != expected_rows:
        raise ValueError(
            f"coverage row count mismatch: observed={len(rows)} expected={expected_rows}"
        )
    return dict(by_target)


def task5_owner(row_index: str) -> str | None:
    if not row_index.startswith(TASK5_PREFIX):
        return None
    owner, separator, rank = row_index[len(TASK5_PREFIX) :].rpartition(":")
    if not separator or not owner or not rank.isdigit():
        raise ValueError(f"invalid task5 pseudo index: {row_index}")
    return owner


def collect_existing_views(
    rows: list[dict[str, Any]],
    targets: list[str],
    *,
    task5_per_skill: int,
    mode3_per_skill: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]], int]:
    target_set = set(targets)
    task5: dict[str, list[tuple[str, str]]] = defaultdict(list)
    mode3: dict[str, list[tuple[str, str]]] = defaultdict(list)
    alias_task5_rows = 0
    for row in rows:
        positive = selected_single_skill(row)
        if positive not in target_set:
            continue
        index = str(row.get("index") or "")
        query = str(row.get("query") or "").strip()
        owner = task5_owner(index)
        if owner is not None:
            if owner == positive:
                task5[positive].append((index, query))
            else:
                alias_task5_rows += 1
        elif index.startswith(MODE3_PREFIX):
            mode3[positive].append((index, query))

    queries: dict[str, list[str]] = {}
    indices: dict[str, list[str]] = {}
    for target in targets:
        task_rows = sorted(task5[target])
        mode_rows = sorted(mode3[target])
        if len(task_rows) != task5_per_skill or len(mode_rows) != mode3_per_skill:
            raise ValueError(
                f"existing-view mismatch for {target}: task5={len(task_rows)}/{task5_per_skill} "
                f"mode3={len(mode_rows)}/{mode3_per_skill}"
            )
        combined = [*task_rows, *mode_rows]
        if any(not query for _, query in combined):
            raise ValueError(f"empty existing query for {target}")
        if len({query.strip().lower() for _, query in combined}) != len(combined):
            raise ValueError(f"duplicate existing query for {target}")
        indices[target] = [index for index, _ in combined]
        queries[target] = [query for _, query in combined]
    return queries, indices, alias_task5_rows


def skill_excerpt(skill: dict[str, Any], body_chars: int) -> str:
    description = str(skill.get("description") or skill.get("task_description") or "")[:400]
    body = str(skill.get("body") or "")[: min(max(body_chars, 0), 3000)]
    return f"Name: {skill_name(skill)}\nDescription: {description}\nBody:\n{body}"


def existing_query_excerpt(query: str, max_chars: int = 450) -> str:
    normalized = re.sub(r"\s+", " ", query).strip()
    return normalized if len(normalized) <= max_chars else normalized[:max_chars].rstrip() + "…"


def build_coverage_messages(
    skill: dict[str, Any],
    existing_queries: list[str],
    *,
    body_chars: int,
) -> list[dict[str, str]]:
    numbered_modes = "\n".join(
        f"{position}. `{mode}`: {MODE_REQUIREMENTS[mode]}"
        for position, mode in enumerate(COVERAGE_MODES, start=1)
    )
    existing = "\n".join(
        f"- Existing view {position}: {existing_query_excerpt(query)}"
        for position, query in enumerate(existing_queries, start=1)
    )
    prompt = (
        "Generate scenario-diverse synthetic retrieval queries for a single-skill generative retriever. Use only training-side "
        "information below.\n\n"
        "Private skill document:\n"
        f"{skill_excerpt(skill, body_chars)}\n\n"
        "Existing training views for this same document:\n"
        f"{existing}\n\n"
        f"Return exactly {len(COVERAGE_MODES)} new queries, one for each mode:\n{numbered_modes}\n\n"
        "Shared requirements:\n"
        "- Each query must require exactly one specialized capability and be fully supported by the private document alone.\n"
        "- Change the situation, intent framing, and artifact flow relative to every existing view; do not merely paraphrase one.\n"
        "- The six new queries must also be materially different from one another.\n"
        "- Do not mention the skill name, skill id, retrieval, private documents, or the existence of a skill.\n"
        "- Do not invent unsupported tools, APIs, formats, measurements, or domain facts. Use neutral placeholders when needed.\n"
        "- For every mode except `terse_natural`, write 400-1,500 characters, include `You need to:` with 2-5 numbered steps, "
        "and end with a deliverable plus a supported verification criterion.\n"
        "- For `terse_natural`, write 120-500 characters in 2-4 sentences without numbered steps.\n\n"
        "Return JSON only as an array in this exact schema:\n"
        '[{"mode":"implicit_outcome","query":"..."}, ...]'
    )
    return [{"role": "user", "content": prompt}]


def response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    return str(((choices[0] or {}).get("message") or {}).get("content") or "").strip()


def parse_mode_queries(text: str) -> list[dict[str, str]]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("queries") or payload.get("items")
    if not isinstance(payload, list):
        raise ValueError("response must be a JSON array")
    parsed: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("every response item must be an object")
        parsed.append({"mode": str(item.get("mode") or "").strip(), "query": str(item.get("query") or "").strip()})
    return parsed


def normalize_coverage_query_layout(query: str, mode: str) -> str:
    if mode == "terse_natural":
        return query.strip()
    normalized = normalize_task_spec_layout(query)
    if "you need to:" not in normalized.lower():
        first_item = re.search(r"(?<!\S)1[.)]\s+", normalized)
        if first_item is not None:
            normalized = (
                f"{normalized[: first_item.start()].rstrip()}\n\nYou need to:\n"
                f"{normalized[first_item.start() :].lstrip()}"
            )
    return normalize_task_spec_layout(normalized)


def build_generation_record(
    *,
    target: str,
    skill: dict[str, Any],
    existing_queries: list[str],
    response: dict[str, Any],
    max_query_chars: int,
    existing_jaccard_threshold: float,
    pairwise_jaccard_threshold: float,
) -> dict[str, Any]:
    reject_reasons: list[str] = []
    query_records: list[dict[str, Any]] = []
    try:
        parsed = parse_mode_queries(response_content(response))
    except (json.JSONDecodeError, ValueError) as exc:
        parsed = []
        reject_reasons.append(f"parse:{type(exc).__name__}")
    observed_modes = [item["mode"] for item in parsed]
    if observed_modes != list(COVERAGE_MODES):
        reject_reasons.append(f"mode_order:{observed_modes}")

    existing_tokens = [token_set(query) for query in existing_queries]
    accepted_tokens: list[set[str]] = []
    seen_texts: set[str] = set()
    for item in parsed:
        mode = item["mode"]
        query = normalize_coverage_query_layout(item["query"], mode)
        sanitized, sanitizations = sanitize_skill_references(query, skill)
        query = canonicalize_query(sanitized, max_query_chars)
        item_reasons: list[str] = []
        minimum_chars = 120 if mode == "terse_natural" else 300
        maximum_chars = 500 if mode == "terse_natural" else max_query_chars
        if len(query) < minimum_chars:
            item_reasons.append(f"short:{len(query)}<{minimum_chars}")
        if len(query) > maximum_chars:
            item_reasons.append(f"long:{len(query)}>{maximum_chars}")
        has_task_marker = "you need to:" in query.lower()
        has_numbered_steps = bool(re.search(r"(^|\n)\s*1[.)]\s", query))
        if mode == "terse_natural" and re.search(r"(^|\n)\s*\d+[.)]\s", query):
            item_reasons.append("terse_has_numbered_steps")
        item_reasons.extend(f"leak:{reason}" for reason in exact_leak_reasons(query, skill, []))
        normalized_key = re.sub(r"\s+", " ", query).strip().lower()
        if not normalized_key or normalized_key in seen_texts:
            item_reasons.append("duplicate_query")
        candidate_tokens = token_set(query)
        max_existing = max((jaccard(candidate_tokens, ref) for ref in existing_tokens), default=0.0)
        max_pairwise = max((jaccard(candidate_tokens, ref) for ref in accepted_tokens), default=0.0)
        if max_existing > existing_jaccard_threshold:
            item_reasons.append(f"existing_jaccard:{max_existing:.6f}>{existing_jaccard_threshold:.6f}")
        if max_pairwise > pairwise_jaccard_threshold:
            item_reasons.append(f"pairwise_jaccard:{max_pairwise:.6f}>{pairwise_jaccard_threshold:.6f}")
        if item_reasons:
            reject_reasons.extend(f"{mode}:{reason}" for reason in item_reasons)
        else:
            seen_texts.add(normalized_key)
            accepted_tokens.append(candidate_tokens)
        query_records.append(
            {
                "mode": mode,
                "query": query,
                "max_existing_jaccard": max_existing,
                "max_pairwise_jaccard": max_pairwise,
                "has_task_marker": has_task_marker,
                "has_numbered_steps": has_numbered_steps,
                "sanitized_reference_types": sorted(set(sanitizations)),
                "reject_reasons": item_reasons,
            }
        )

    if len(query_records) != len(COVERAGE_MODES):
        reject_reasons.append(f"query_count:{len(query_records)}!={len(COVERAGE_MODES)}")
    return {
        "schema_version": 1,
        "work_key": target,
        "target_skill_id": target,
        "target_skill_name": skill_name(skill),
        "status": "complete" if not reject_reasons else "invalid_generation",
        "prompt_version": PROMPT_VERSION,
        "response_model": str(response.get("model") or ""),
        "usage": response.get("usage") or {},
        "queries": query_records,
        "reject_reasons": reject_reasons,
    }


def build_error_record(target: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "work_key": target,
        "target_skill_id": target,
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


def materialize(
    *,
    output_dir: Path,
    targets: list[str],
    existing_indices: dict[str, list[str]],
    profile: dict[str, Any],
    run_config: dict[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    records_path = output_dir / "coverage_generations.jsonl"
    records = load_jsonl(records_path) if records_path.exists() else []
    latest = latest_records(records)
    attempts = Counter(str(record.get("work_key") or "") for record in records)
    selected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for target in targets:
        record = latest.get(target)
        if record is None or record.get("status") != "complete":
            failures.append(
                {
                    "work_key": target,
                    "target_skill_id": target,
                    "status": str((record or {}).get("status") or "missing"),
                    "attempts": attempts[target],
                    "retryable": attempts[target] < max_attempts,
                }
            )
            continue
        queries = record.get("queries") or []
        for rank, mode in enumerate(COVERAGE_MODES):
            query_record = queries[rank]
            if query_record.get("mode") != mode or query_record.get("reject_reasons"):
                raise ValueError(f"invalid complete query record for {target}:{mode}")
            query = str(query_record["query"])
            selected.append(
                {
                    "index": f"{COVERAGE_PREFIX}{target}:{mode}",
                    "source": COVERAGE_SOURCE,
                    "query": query,
                    "query_text": f"{QUERY_INSTRUCTION}{query}",
                    "target_skill_count": 1,
                    "selected_skill_ids": [target],
                    "positive_skill_id": target,
                    "pseudo_query": {
                        "mode": mode,
                        "rank": rank,
                        "clean_catalog": True,
                        "selection_basis": SELECTION_BASIS,
                        "prompt_version": PROMPT_VERSION,
                        "materialization_version": MATERIALIZATION_VERSION,
                        "generator_model": str(record.get("response_model") or profile.get("model") or ""),
                        "existing_view_indices": existing_indices[target],
                        "max_existing_jaccard": query_record["max_existing_jaccard"],
                        "max_pairwise_jaccard": query_record["max_pairwise_jaccard"],
                        "has_task_marker": bool(
                            query_record.get("has_task_marker", "you need to:" in query.lower())
                        ),
                        "has_numbered_steps": bool(
                            query_record.get(
                                "has_numbered_steps",
                                re.search(r"(^|\n)\s*1[.)]\s", query),
                            )
                        ),
                        "reference_sanitized": bool(query_record.get("sanitized_reference_types")),
                        "sanitized_reference_types": query_record.get("sanitized_reference_types") or [],
                    },
                }
            )

    expected_rows = len(targets) * len(COVERAGE_MODES)
    indices = [str(row["index"]) for row in selected]
    queries = [re.sub(r"\s+", " ", str(row["query"])).strip().lower() for row in selected]
    if len(indices) != len(set(indices)):
        raise ValueError("materialized coverage indices are not unique")
    if len(queries) != len(set(queries)):
        raise ValueError("materialized coverage query texts are not globally unique")
    selected_path = output_dir / "selected_diverse_coverage.jsonl"
    failures_path = output_dir / "failures.jsonl"
    write_jsonl(selected_path, selected)
    write_jsonl(failures_path, failures)
    manifest = {
        "format": "diverse_coverage_generation_v1",
        "complete": not failures and len(selected) == expected_rows,
        "selection_basis": SELECTION_BASIS,
        "prompt_version": PROMPT_VERSION,
        "materialization_version": MATERIALIZATION_VERSION,
        "run_config": run_config,
        "profile": public_profile(profile),
        "generation_checkpoint": str(records_path),
        "selected_rows_path": str(selected_path),
        "failures_path": str(failures_path),
        "target_skill_count": len(targets),
        "modes": list(COVERAGE_MODES),
        "expected_rows": expected_rows,
        "selected_rows": len(selected),
        "failed_targets": len(failures),
        "checkpoint_records": len(records),
        "all_attempt_usage": aggregate_usage(records),
        "latest_record_usage": aggregate_usage(list(latest.values())),
        "selected_query_stats": selected_query_stats(selected),
    }
    if selected_path.exists():
        manifest["selected_rows_sha256"] = file_sha256(selected_path)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def command_preview(args: argparse.Namespace) -> None:
    targets = target_ids_from_rows(load_jsonl(args.target_skills))[: args.limit]
    catalog = {skill_id(row): row for row in load_jsonl(args.skills_jsonl)}
    existing, _, _ = collect_existing_views(
        load_jsonl(args.augmented_rows),
        target_ids_from_rows(load_jsonl(args.target_skills)),
        task5_per_skill=args.task5_per_skill,
        mode3_per_skill=args.mode3_per_skill,
    )
    previews = [
        {"target_skill_id": target, "messages": build_coverage_messages(catalog[target], existing[target], body_chars=args.body_chars)}
        for target in targets
    ]
    write_jsonl(args.output, previews)
    print(f"saved prompt previews={len(previews)} output={args.output}")


def command_generate(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.max_attempts < 1:
        raise SystemExit("--workers and --max-attempts must be positive")
    targets = target_ids_from_rows(load_jsonl(args.target_skills))
    if args.limit_skills > 0:
        targets = targets[: args.limit_skills]
    if len(targets) != args.expected_targets and args.limit_skills < 1:
        raise SystemExit(f"target count mismatch: observed={len(targets)} expected={args.expected_targets}")
    catalog = {skill_id(row): row for row in load_jsonl(args.skills_jsonl)}
    missing = sorted(set(targets) - set(catalog))
    if missing:
        raise SystemExit(f"target skills missing from catalog: {missing[:10]}")
    existing_queries, existing_indices, alias_rows = collect_existing_views(
        load_jsonl(args.augmented_rows),
        targets,
        task5_per_skill=args.task5_per_skill,
        mode3_per_skill=args.mode3_per_skill,
    )
    profile = load_profile(args.profile)
    validate_clean_generation_profile(profile)
    configure_profile_transport(profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": 1,
        "selection_basis": SELECTION_BASIS,
        "prompt_version": PROMPT_VERSION,
        "target_skills": str(args.target_skills),
        "target_skills_sha256": file_sha256(args.target_skills),
        "skills_jsonl": str(args.skills_jsonl),
        "skills_sha256": file_sha256(args.skills_jsonl),
        "augmented_rows": str(args.augmented_rows),
        "augmented_rows_sha256": file_sha256(args.augmented_rows),
        "target_skill_count": len(targets),
        "modes": list(COVERAGE_MODES),
        "views_per_target": len(COVERAGE_MODES),
        "existing_task5_per_skill": args.task5_per_skill,
        "existing_mode3_per_skill": args.mode3_per_skill,
        "discarded_alias_task5_rows": alias_rows,
        "body_chars": args.body_chars,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_query_chars": args.max_query_chars,
        "existing_jaccard_threshold": args.existing_jaccard_threshold,
        "pairwise_jaccard_threshold": args.pairwise_jaccard_threshold,
        "model": profile["model"],
    }
    run_config_path = args.output_dir / "run_config.json"
    if run_config_path.exists():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise SystemExit("generation run config differs from the existing resumable output")
    else:
        write_json(run_config_path, run_config)

    records_path = args.output_dir / "coverage_generations.jsonl"
    records = load_jsonl(records_path) if records_path.exists() else []
    latest = latest_records(records)
    attempts = Counter(str(record.get("work_key") or "") for record in records)

    def needs_generation(target: str) -> bool:
        return str((latest.get(target) or {}).get("status") or "") != "complete" and attempts[target] < args.max_attempts

    pending = [target for target in targets if needs_generation(target)]
    print(
        f"coverage generation targets={len(targets)} pending={len(pending)} "
        f"complete={len(targets) - len(pending)} workers={args.workers}",
        flush=True,
    )
    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed = 0
    round_total = len(pending)

    def run_target(target: str) -> dict[str, Any]:
        nonlocal completed
        skill = catalog[target]
        try:
            response = call_query_generator(
                profile,
                build_coverage_messages(skill, existing_queries[target], body_chars=args.body_chars),
                args.temperature,
                args.top_p,
                args.timeout,
                args.max_tokens,
                n=1,
            )
            record = build_generation_record(
                target=target,
                skill=skill,
                existing_queries=existing_queries[target],
                response=response,
                max_query_chars=args.max_query_chars,
                existing_jaccard_threshold=args.existing_jaccard_threshold,
                pairwise_jaccard_threshold=args.pairwise_jaccard_threshold,
            )
        except Exception as exc:  # Resumable transport and parse failure checkpoint.
            record = build_error_record(target, exc)
        append_jsonl(records_path, record, write_lock)
        with progress_lock:
            completed += 1
            if completed == round_total or completed % args.progress_every == 0:
                print(f"progress={completed}/{round_total} target={target} status={record['status']}", flush=True)
        return record

    for attempt_round in range(args.max_attempts):
        if not pending:
            break
        completed = 0
        round_total = len(pending)
        if args.workers == 1:
            for target in pending:
                run_target(target)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(run_target, pending))
        records = load_jsonl(records_path)
        latest = latest_records(records)
        attempts = Counter(str(record.get("work_key") or "") for record in records)
        pending = [target for target in targets if needs_generation(target)]
        if pending:
            print(f"retry_round={attempt_round + 1} pending={len(pending)}", flush=True)

    manifest = materialize(
        output_dir=args.output_dir,
        targets=targets,
        existing_indices=existing_indices,
        profile=profile,
        run_config=run_config,
        max_attempts=args.max_attempts,
    )
    print(
        f"selected_rows={manifest['selected_rows']}/{manifest['expected_rows']} "
        f"failed_targets={manifest['failed_targets']}",
        flush=True,
    )
    if args.require_complete and not manifest["complete"]:
        raise SystemExit(2)


def prepare_mining_rows(
    *,
    selected_rows_path: Path,
    generation_manifest_path: Path,
    output_dir: Path,
    expected_targets: int,
    views_per_target: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite mining input directory: {output_dir}")
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    selected_sha256 = file_sha256(selected_rows_path)
    required_manifest = {
        "complete": True,
        "target_skill_count": expected_targets,
        "selected_rows": expected_targets * views_per_target,
        "modes": list(COVERAGE_MODES),
        "selected_rows_sha256": selected_sha256,
    }
    drift = {
        key: (expected, generation_manifest.get(key))
        for key, expected in required_manifest.items()
        if generation_manifest.get(key) != expected
    }
    if drift:
        raise ValueError(f"generation manifest drifted: {drift}")

    rows = load_jsonl(selected_rows_path)
    validate_coverage_closure(
        rows,
        expected_targets=expected_targets,
        views_per_target=views_per_target,
    )
    mining_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        source_index, _, mode = coverage_row_identity(row)
        mining_rows.append(
            {
                **row,
                "index": ordinal,
                "source_index": source_index,
                "coverage_mining": {
                    "source_kind": "diverse_coverage",
                    "selection_basis": SELECTION_BASIS,
                    "mode": mode,
                },
            }
        )
    output_path = output_dir / "mining_rows.jsonl"
    write_jsonl(output_path, mining_rows)
    manifest = {
        "format": "diverse_coverage_mining_rows_v1",
        "selection_basis": SELECTION_BASIS,
        "selected_rows": str(selected_rows_path),
        "selected_rows_sha256": selected_sha256,
        "generation_manifest": str(generation_manifest_path),
        "generation_manifest_sha256": file_sha256(generation_manifest_path),
        "target_skill_count": expected_targets,
        "views_per_target": views_per_target,
        "modes": list(COVERAGE_MODES),
        "row_count": len(mining_rows),
        "index_contract": "zero_based_contiguous_row_ordinal",
        "source_index_contract": "original_diverse_coverage_index",
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def filter_mined_errors(
    *,
    mining_rows_path: Path,
    candidates_path: Path,
    candidate_manifest_path: Path,
    prediction_paths: list[Path],
    output_dir: Path,
    expected_rows: int,
    expected_targets: int,
    views_per_target: int,
    expected_negatives: int,
    expected_beams: int,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite error-filtered directory: {output_dir}")
    rows = load_jsonl(mining_rows_path)
    if len(rows) != expected_rows:
        raise ValueError(f"mining row count mismatch: observed={len(rows)} expected={expected_rows}")
    validate_coverage_closure(
        rows,
        expected_targets=expected_targets,
        views_per_target=views_per_target,
    )
    mining_by_index: dict[int, dict[str, Any]] = {}
    for expected_index, row in enumerate(rows):
        row_index = int(row.get("index"))
        if row_index != expected_index:
            raise ValueError(
                f"mining row indices must be contiguous: ordinal={expected_index} index={row_index}"
            )
        mining_by_index[row_index] = row

    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    mining_rows_sha256 = file_sha256(mining_rows_path)
    candidates_sha256 = file_sha256(candidates_path)
    required_candidate_manifest = {
        "rows_sha256": mining_rows_sha256,
        "output_sha256": candidates_sha256,
        "mining_checkpoint_sha256": expected_checkpoint_sha256,
    }
    candidate_drift = {
        key: (expected, candidate_manifest.get(key))
        for key, expected in required_candidate_manifest.items()
        if candidate_manifest.get(key) != expected
    }
    mining_protocol = candidate_manifest.get("mining_protocol") or {}
    if mining_protocol.get("num_beams") != expected_beams:
        candidate_drift["mining_protocol.num_beams"] = (
            expected_beams,
            mining_protocol.get("num_beams"),
        )
    audit = candidate_manifest.get("audit") or {}
    required_audit = {
        "rows": expected_rows,
        "training_rows": expected_rows,
        "negatives_per_row": expected_negatives,
        "known_unique_non_gold_negatives": True,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            candidate_drift[f"audit.{key}"] = (expected, audit.get(key))
    if candidate_drift:
        raise ValueError(f"candidate manifest drifted: {candidate_drift}")

    expected_prediction_hashes = sorted(
        str(item.get("predictions_sha256") or "")
        for item in candidate_manifest.get("prediction_shards", [])
    )
    observed_prediction_hashes = sorted(file_sha256(path) for path in prediction_paths)
    if observed_prediction_hashes != expected_prediction_hashes:
        raise ValueError("prediction files do not match the compacted candidate manifest")

    candidates_by_index: dict[int, dict[str, Any]] = {}
    for candidate in load_jsonl(candidates_path):
        row_index = int(candidate.get("row_index"))
        if row_index in candidates_by_index or row_index not in mining_by_index:
            raise ValueError(f"duplicate or unknown candidate row_index={row_index}")
        row = mining_by_index[row_index]
        source_index, positive, _ = coverage_row_identity(row)
        negatives = [str(value) for value in candidate.get("negative_skill_ids", []) or []]
        if str(candidate.get("source_index") or "") != source_index:
            raise ValueError(f"candidate source mismatch at row_index={row_index}")
        if str(candidate.get("positive_skill_id") or "") != positive:
            raise ValueError(f"candidate positive mismatch at row_index={row_index}")
        if str(candidate.get("query_sha256") or "") != query_digest(row):
            raise ValueError(f"candidate query mismatch at row_index={row_index}")
        if len(negatives) != expected_negatives or len(negatives) != len(set(negatives)):
            raise ValueError(f"candidate negative count/uniqueness failed at row_index={row_index}")
        if positive in negatives:
            raise ValueError(f"candidate negatives contain gold at row_index={row_index}")
        candidates_by_index[row_index] = candidate
    if set(candidates_by_index) != set(mining_by_index):
        raise ValueError("candidate/mining row closure failed")

    predictions_by_index: dict[int, dict[str, Any]] = {}
    for prediction_path in prediction_paths:
        for prediction in load_jsonl(prediction_path):
            row_index = int(prediction.get("index"))
            if row_index in predictions_by_index or row_index not in mining_by_index:
                raise ValueError(f"duplicate or unknown prediction row_index={row_index}")
            row = mining_by_index[row_index]
            _, positive, _ = coverage_row_identity(row)
            gold = [str(value) for value in prediction.get("gold_skill_ids", []) or []]
            ranked = [str(value) for value in prediction.get("ranked_skill_ids", []) or []]
            if gold != [positive] or not ranked:
                raise ValueError(f"prediction gold/ranked closure failed at row_index={row_index}")
            if bool(prediction.get("invalid")):
                raise ValueError(f"invalid decoded prediction at row_index={row_index}")
            if query_digest(prediction) != query_digest(row):
                raise ValueError(f"prediction query mismatch at row_index={row_index}")
            candidate = candidates_by_index[row_index]
            expected_candidate_order = [sid for sid in ranked if sid != positive][:expected_negatives]
            if expected_candidate_order != candidate["negative_skill_ids"]:
                raise ValueError(f"prediction/candidate order mismatch at row_index={row_index}")
            predictions_by_index[row_index] = prediction
    if set(predictions_by_index) != set(mining_by_index):
        raise ValueError("prediction/mining row closure failed")

    outcomes: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    error_candidates: list[dict[str, Any]] = []
    per_mode_total: Counter[str] = Counter()
    per_mode_errors: Counter[str] = Counter()
    per_target_total: Counter[str] = Counter()
    per_target_errors: Counter[str] = Counter()
    per_target_modes: dict[str, list[str]] = defaultdict(list)
    per_target_confusers: dict[str, Counter[str]] = defaultdict(Counter)
    gold_in_beam = 0
    for row_index in sorted(mining_by_index):
        row = mining_by_index[row_index]
        source_index, positive, mode = coverage_row_identity(row)
        prediction = predictions_by_index[row_index]
        ranked = [str(value) for value in prediction["ranked_skill_ids"]]
        gold_rank = ranked.index(positive) + 1 if positive in ranked else None
        if gold_rank is not None:
            gold_in_beam += 1
        gold_top1 = ranked[0] == positive
        top1_confuser = None if gold_top1 else ranked[0]
        per_mode_total[mode] += 1
        per_target_total[positive] += 1
        if not gold_top1:
            per_mode_errors[mode] += 1
            per_target_errors[positive] += 1
            per_target_modes[positive].append(mode)
            per_target_confusers[positive][str(top1_confuser)] += 1
        outcomes.append(
            {
                "row_index": row_index,
                "source_index": source_index,
                "target_skill_id": positive,
                "mode": mode,
                "gold_top1": gold_top1,
                "gold_rank": gold_rank,
                "top1_skill_id": ranked[0],
            }
        )
        if gold_top1:
            continue
        filtered_index = len(error_rows)
        error_rows.append(
            {
                **row,
                "index": source_index,
                "source_index": source_index,
                "error_filtered_mining": {
                    "source_row_index": row_index,
                    "mining_checkpoint_sha256": expected_checkpoint_sha256,
                    "num_beams": expected_beams,
                    "gold_rank": gold_rank,
                    "top1_confuser_skill_id": top1_confuser,
                },
            }
        )
        source_candidate = candidates_by_index[row_index]
        error_candidates.append(
            {
                "row_index": filtered_index,
                "source_index": source_index,
                "query_sha256": query_digest(row),
                "positive_skill_id": positive,
                "negative_skill_ids": source_candidate["negative_skill_ids"],
                "source_mining_row_index": row_index,
                "top1_confuser_skill_id": top1_confuser,
            }
        )
    if not error_rows:
        raise ValueError("error filtering retained zero rows")

    target_summaries = []
    for target in sorted(per_target_total):
        confusers = per_target_confusers[target]
        target_summaries.append(
            {
                "target_skill_id": target,
                "total_queries": per_target_total[target],
                "gold_top1_queries": per_target_total[target] - per_target_errors[target],
                "error_queries": per_target_errors[target],
                "error_modes": sorted(per_target_modes[target]),
                "top1_confusers": [
                    {"skill_id": skill, "count": count}
                    for skill, count in sorted(confusers.items(), key=lambda item: (-item[1], item[0]))
                ],
            }
        )
    error_histogram = Counter(per_target_errors[target] for target in per_target_total)
    by_mode = {
        mode: {
            "total": per_mode_total[mode],
            "errors": per_mode_errors[mode],
            "gold_top1": per_mode_total[mode] - per_mode_errors[mode],
            "error_rate": per_mode_errors[mode] / per_mode_total[mode],
        }
        for mode in COVERAGE_MODES
    }

    output_dir.mkdir(parents=True)
    error_rows_path = output_dir / "error_rows.jsonl"
    error_candidates_path = output_dir / "error_candidates.jsonl"
    outcomes_path = output_dir / "all_query_outcomes.jsonl"
    target_summary_path = output_dir / "target_error_summary.jsonl"
    write_jsonl(error_rows_path, error_rows)
    write_jsonl(error_candidates_path, error_candidates)
    write_jsonl(outcomes_path, outcomes)
    write_jsonl(target_summary_path, target_summaries)
    manifest = {
        "format": "diverse_coverage_error_filtered_v1",
        "selection_basis": SELECTION_BASIS,
        "mining_rows": str(mining_rows_path),
        "mining_rows_sha256": mining_rows_sha256,
        "candidates": str(candidates_path),
        "candidates_sha256": candidates_sha256,
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "prediction_shards": [
            {"path": str(path), "sha256": file_sha256(path)} for path in prediction_paths
        ],
        "mining_checkpoint_sha256": expected_checkpoint_sha256,
        "mining_protocol": {"num_beams": expected_beams},
        "input_rows": len(rows),
        "target_skill_count": len(per_target_total),
        "views_per_target": views_per_target,
        "gold_top1_rows": len(rows) - len(error_rows),
        "gold_in_beam_rows": gold_in_beam,
        "gold_missing_from_beam_rows": len(rows) - gold_in_beam,
        "error_rows": len(error_rows),
        "targets_with_errors": sum(count > 0 for count in per_target_errors.values()),
        "unique_top1_error_pairs": sum(len(values) for values in per_target_confusers.values()),
        "by_mode": by_mode,
        "error_count_per_target_histogram": {
            str(count): targets for count, targets in sorted(error_histogram.items())
        },
        "outputs": {
            "error_rows": str(error_rows_path),
            "error_rows_sha256": file_sha256(error_rows_path),
            "error_candidates": str(error_candidates_path),
            "error_candidates_sha256": file_sha256(error_candidates_path),
            "all_query_outcomes": str(outcomes_path),
            "all_query_outcomes_sha256": file_sha256(outcomes_path),
            "target_error_summary": str(target_summary_path),
            "target_error_summary_sha256": file_sha256(target_summary_path),
        },
        "candidate_join_contract": "zero_based_error_row_ordinal",
        "query_binding": "sha256",
        "negatives_per_error_row": expected_negatives,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def command_prepare_mining(args: argparse.Namespace) -> None:
    manifest = prepare_mining_rows(
        selected_rows_path=args.selected_rows,
        generation_manifest_path=args.generation_manifest,
        output_dir=args.output_dir,
        expected_targets=args.expected_targets,
        views_per_target=args.views_per_target,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def command_filter_errors(args: argparse.Namespace) -> None:
    manifest = filter_mined_errors(
        mining_rows_path=args.mining_rows,
        candidates_path=args.candidates,
        candidate_manifest_path=args.candidate_manifest,
        prediction_paths=args.predictions,
        output_dir=args.output_dir,
        expected_rows=args.expected_rows,
        expected_targets=args.expected_targets,
        views_per_target=args.views_per_target,
        expected_negatives=args.expected_negatives,
        expected_beams=args.expected_beams,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def add_shared_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-skills", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--augmented-rows", type=Path, required=True)
    parser.add_argument("--task5-per-skill", type=int, default=5)
    parser.add_argument("--mode3-per-skill", type=int, default=3)
    parser.add_argument("--body-chars", type=int, default=2500)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser("preview")
    add_shared_inputs(preview)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--limit", type=int, default=3)
    preview.set_defaults(func=command_preview)

    generate = subparsers.add_parser("generate")
    add_shared_inputs(generate)
    generate.add_argument("--profile", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--expected-targets", type=int, default=1722)
    generate.add_argument("--limit-skills", type=int, default=-1)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--max-tokens", type=int, default=4096)
    generate.add_argument("--max-query-chars", type=int, default=1500)
    generate.add_argument("--existing-jaccard-threshold", type=float, default=0.78)
    generate.add_argument("--pairwise-jaccard-threshold", type=float, default=0.72)
    generate.add_argument("--timeout", type=int, default=240)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--progress-every", type=int, default=25)
    generate.add_argument("--max-attempts", type=int, default=3)
    generate.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    generate.set_defaults(func=command_generate)

    prepare = subparsers.add_parser("prepare-mining")
    prepare.add_argument("--selected-rows", type=Path, required=True)
    prepare.add_argument("--generation-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--expected-targets", type=int, default=1722)
    prepare.add_argument("--views-per-target", type=int, default=len(COVERAGE_MODES))
    prepare.set_defaults(func=command_prepare_mining)

    filter_errors = subparsers.add_parser("filter-errors")
    filter_errors.add_argument("--mining-rows", type=Path, required=True)
    filter_errors.add_argument("--candidates", type=Path, required=True)
    filter_errors.add_argument("--candidate-manifest", type=Path, required=True)
    filter_errors.add_argument("--predictions", type=Path, action="append", required=True)
    filter_errors.add_argument("--output-dir", type=Path, required=True)
    filter_errors.add_argument("--expected-rows", type=int, default=1722 * len(COVERAGE_MODES))
    filter_errors.add_argument("--expected-targets", type=int, default=1722)
    filter_errors.add_argument("--views-per-target", type=int, default=len(COVERAGE_MODES))
    filter_errors.add_argument("--expected-negatives", type=int, default=10)
    filter_errors.add_argument("--expected-beams", type=int, default=50)
    filter_errors.add_argument("--expected-checkpoint-sha256", required=True)
    filter_errors.set_defaults(func=command_filter_errors)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
