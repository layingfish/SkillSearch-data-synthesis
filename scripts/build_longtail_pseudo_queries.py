#!/usr/bin/env python3
"""Generate and select pseudo queries for SkillGR training and diagnostics.

This script follows the DSI-QG pattern at small scale:

skill document -> many candidate pseudo queries -> filter/rank/select -> query rows

``generate-catalog`` is the clean full-catalog path and reads only training skill
documents. The legacy ``generate`` command deliberately treats official-single
GT skills as a transductive diagnostic set; never mix its outputs into clean runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import generate_single_skill_probe as single  # noqa: E402


SKILLROUTER_QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant skill document "
    "that would help an agent complete the task\nQuery:"
)
TOKEN_RE = re.compile(r"[a-z0-9_./+-]+", re.I)
ARTIFACT_RE = re.compile(
    r"(`[^`]+`|/[a-z0-9_./{}+-]+|\b[a-z0-9_-]+\.(?:pdf|csv|json|jsonl|yaml|yml|xlsx|xls|txt|md|py|js|ts|stl|mp4|wav|rttm|bib|xml|html)\b)",
    re.I,
)
CONSTRAINT_RE = re.compile(
    r"\b(schema|json|yaml|csv|threshold|accuracy|validate|validation|must|within|format|field|"
    r"output|save|report|metric|constraint|criterion|criteria)\b",
    re.I,
)
CLEAN_CATALOG_SOURCE = "full_catalog_task_style_pseudo_query"
CLEAN_PROMPT_VERSION = "task_style_v1_20260713"
CLEAN_SELECTION_VERSION = "task_spec_overrides_v6_20260713"
CLEAN_TERMINAL_STATUSES = {"complete", "insufficient_candidates"}
TRUNCATED_CANDIDATE_PENALTY = 0.75
ISOLATED_TASK_MODES = ("outcome_first", "workflow_embedded", "constraint_failure")
ISOLATED_SOURCE = "zero_multi_cooccurrence_task_mode_pseudo_query"
ISOLATED_PROMPT_VERSION = "zero_multi_task_modes_v2_20260714"
ISOLATED_MATERIALIZATION_VERSION = "task_layout_v2_20260714"
ISOLATED_TERMINAL_STATUSES = {"complete", "invalid_generation"}


@dataclass
class OfficialSingleTarget:
    skill_id: str
    official_indices: list[str]
    official_queries: list[str]
    train_frequency: int


@dataclass
class CandidateQuery:
    query_id: str
    mode: str
    text: str
    source_rank: int
    metadata: dict[str, Any]
    score: float = 0.0


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "")


def skill_name(row: dict[str, Any]) -> str:
    return str(row.get("skill_name") or row.get("name") or skill_id(row).rsplit("/", 1)[-1])


def selected_skill_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("selected_skill_ids")
    if not ids and isinstance(row.get("generated"), dict):
        ids = row["generated"].get("selected_skill_ids")
    return [str(sid) for sid in (ids or []) if sid]


def collect_train_frequencies(rows: list[dict[str, Any]]) -> Counter[str]:
    freq: Counter[str] = Counter()
    for row in rows:
        freq.update(selected_skill_ids(row))
    return freq


def collect_multi_row_frequencies(rows: list[dict[str, Any]]) -> Counter[str]:
    """Count rows in which each skill co-occurs with at least one other skill."""
    frequencies: Counter[str] = Counter()
    for row in rows:
        ids = list(dict.fromkeys(selected_skill_ids(row)))
        if len(ids) > 1:
            frequencies.update(ids)
    return frequencies


def select_zero_multi_cooccurrence_skills(
    catalog_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    limit_skills: int = -1,
) -> list[dict[str, Any]]:
    """Select catalog skills absent from every multi-skill training row."""
    catalog_skills = select_catalog_skills(catalog_rows, skill_ids_file=None, limit_skills=-1)
    catalog_ids = {skill_id(skill) for skill in catalog_skills}
    unknown_ids = sorted(
        {
            sid
            for row in train_rows
            for sid in selected_skill_ids(row)
            if sid not in catalog_ids
        }
    )
    if unknown_ids:
        raise ValueError(f"training rows contain skills missing from the catalog: {unknown_ids[:10]}")
    multi_frequencies = collect_multi_row_frequencies(train_rows)
    selected = [skill for skill in catalog_skills if multi_frequencies.get(skill_id(skill), 0) == 0]
    if limit_skills > 0:
        selected = selected[:limit_skills]
    return selected


def extract_official_single_targets(
    official_rows: list[dict[str, Any]],
    train_frequencies: Counter[str],
    max_train_frequency: int,
) -> list[OfficialSingleTarget]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row_index, row in enumerate(official_rows):
        ids = selected_skill_ids(row)
        if len(ids) != 1:
            continue
        sid = ids[0]
        if train_frequencies.get(sid, 0) > max_train_frequency:
            continue
        if sid not in grouped:
            grouped[sid] = {"indices": [], "queries": []}
            order.append(sid)
        grouped[sid]["indices"].append(str(row.get("index", row_index)))
        grouped[sid]["queries"].append(str(row.get("query") or ""))
    return [
        OfficialSingleTarget(
            skill_id=sid,
            official_indices=list(grouped[sid]["indices"]),
            official_queries=list(grouped[sid]["queries"]),
            train_frequency=int(train_frequencies.get(sid, 0)),
        )
        for sid in order
    ]


def load_skill_lookup(path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sid = skill_id(row)
        if sid:
            lookup[sid] = row
    return lookup


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_query_text(text: str) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def tokens(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(str(text))]


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text)))


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = tokens(text)
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def max_ngram_jaccard(text: str, references: list[str], n: int) -> float:
    mine = ngrams(text, n)
    if not mine:
        return 0.0
    best = 0.0
    for ref in references:
        ref_ngrams = ngrams(ref, n)
        if ref_ngrams:
            best = max(best, jaccard(mine, ref_ngrams))
    return best


def skill_document_text(skill: dict[str, Any], body_chars: int = 6000) -> str:
    parts = [
        skill_name(skill),
        str(skill.get("description") or skill.get("task_description") or ""),
        str(skill.get("body") or "")[:body_chars],
        " ".join(str(value) for value in (skill.get("task_object_types") or [])),
        " ".join(str(value) for value in (skill.get("input_artifact_types") or [])),
        " ".join(str(value) for value in (skill.get("output_artifact_types") or [])),
    ]
    return normalize_ws(" | ".join(part for part in parts if part))


def exact_leak_reasons(
    query: str,
    skill: dict[str, Any],
    official_queries: list[str],
    official_ngram_size: int = 8,
    official_ngram_jaccard_threshold: float = 0.18,
) -> list[str]:
    text = str(query)
    lowered = text.lower()
    reasons: list[str] = []
    sid = skill_id(skill).lower()
    if sid and sid in lowered:
        reasons.append("skill_id")
    name = skill_name(skill).strip().lower()
    explicit_name_pattern = re.compile(
        rf"(?:\b(?:skill|plugin)\s+(?:named\s+|called\s+)?[`\"']?{re.escape(name)}\b|"
        rf"\b{re.escape(name)}[`\"']?\s+(?:skill|plugin)\b)",
        re.I,
    ) if name else None
    if len(name) >= 6 and name in lowered and (
        any(separator in name for separator in ("/", "-", "_"))
        or (explicit_name_pattern is not None and explicit_name_pattern.search(text))
    ):
        reasons.append("skill_name")
    basename = sid.rsplit("/", 1)[-1].replace("-", " ").strip()
    explicit_basename_pattern = re.compile(
        rf"(?:\b(?:skill|plugin)\s+(?:named\s+|called\s+)?[`\"']?{re.escape(basename)}\b|"
        rf"\b{re.escape(basename)}[`\"']?\s+(?:skill|plugin)\b)",
        re.I,
    ) if basename else None
    if (
        len(basename) >= 10
        and basename in lowered
        and explicit_basename_pattern is not None
        and explicit_basename_pattern.search(text)
    ):
        reasons.append("skill_basename")
    if official_queries and max_ngram_jaccard(text, official_queries, official_ngram_size) >= official_ngram_jaccard_threshold:
        reasons.append("official_ngram_overlap")
    return reasons


def sanitize_skill_references(query: str, skill: dict[str, Any]) -> tuple[str, list[str]]:
    text = normalize_query_text(query)
    edits: list[str] = []
    token_chars = r"A-Za-z0-9_./-"

    def replace_identifier_token(value: str, identifier: str, edit_type: str) -> str:
        if not identifier:
            return value
        pattern = re.compile(
            rf"(?<![{token_chars}])[{token_chars}]*{re.escape(identifier)}[{token_chars}]*(?![{token_chars}])",
            re.I,
        )
        replaced, count = pattern.subn("<PROJECT_COMMAND>", value)
        if count:
            edits.extend([edit_type] * count)
        return replaced

    sid = skill_id(skill).strip()
    text = replace_identifier_token(text, sid, "skill_id_token")

    name = skill_name(skill).strip()
    if name and any(separator in name for separator in ("/", "-", "_")):
        text = replace_identifier_token(text, name, "skill_name_token")

    contextual_names = [name, sid.rsplit("/", 1)[-1].replace("-", " ").strip()]
    for contextual_name in dict.fromkeys(item for item in contextual_names if len(item) >= 6):
        patterns = [
            re.compile(
                rf"\b(?:skill|plugin)\s+(?:named\s+|called\s+)?[`\"']?{re.escape(contextual_name)}[`\"']?\b",
                re.I,
            ),
            re.compile(
                rf"\b{re.escape(contextual_name)}[`\"']?\s+(?:skill|plugin)\b",
                re.I,
            ),
        ]
        for pattern in patterns:
            text, count = pattern.subn("<PROJECT_CAPABILITY>", text)
            if count:
                edits.extend(["explicit_capability_reference"] * count)
    return normalize_query_text(text), edits


def parse_generated_queries(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.I).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    queries: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                queries.append(item)
            elif isinstance(item, dict):
                value = item.get("query") or item.get("text") or item.get("task")
                if value:
                    queries.append(str(value))
    elif isinstance(payload, dict):
        direct = payload.get("query") or payload.get("text") or payload.get("task")
        values = payload.get("queries") or payload.get("pseudo_queries") or payload.get("items")
        if direct:
            queries.append(str(direct))
        elif isinstance(values, list):
            for item in values:
                if isinstance(item, str):
                    queries.append(item)
                elif isinstance(item, dict) and (item.get("query") or item.get("text")):
                    queries.append(str(item.get("query") or item.get("text")))
    if not queries:
        for line in raw.splitlines():
            line = line.strip()
            line = re.sub(r"^\s*(?:[-*]|\d+[\).])\s*", "", line)
            if line:
                queries.append(line)
    return [normalize_query_text(q) for q in queries if normalize_query_text(q)]


def candidate_features(query: str) -> dict[str, Any]:
    wc = word_count(query)
    artifact_count = len(ARTIFACT_RE.findall(query))
    constraint_count = len(CONSTRAINT_RE.findall(query))
    has_numbered_steps = bool(re.search(r"(^|\n)\s*\d+[\).]", query))
    has_code_fence = "```" in query
    return {
        "word_count": wc,
        "artifact_count": artifact_count,
        "constraint_count": constraint_count,
        "has_numbered_steps": has_numbered_steps,
        "has_code_fence": has_code_fence,
    }


def relevance_score(query: str, skill_text: str, mode: str) -> float:
    q_tokens = token_set(query)
    s_tokens = token_set(skill_text)
    overlap = jaccard(q_tokens, s_tokens)
    features = candidate_features(query)
    wc = features["word_count"]
    if mode == "task_spec":
        length_score = 1.0 if 80 <= wc <= 420 else 0.6 if 45 <= wc <= 520 else 0.1
        artifact_bonus = min(0.3, 0.08 * features["artifact_count"])
        constraint_bonus = min(0.25, 0.04 * features["constraint_count"])
        structure_bonus = 0.08 if features["has_numbered_steps"] or features["has_code_fence"] else 0.0
        structure_bonus += 0.12 if "you need to:" in query.lower() else 0.0
    else:
        length_score = 1.0 if 18 <= wc <= 140 else 0.6 if 10 <= wc <= 220 else 0.1
        artifact_bonus = min(0.12, 0.04 * features["artifact_count"])
        constraint_bonus = min(0.10, 0.025 * features["constraint_count"])
        structure_bonus = 0.0
    return overlap * 3.0 + length_score + artifact_bonus + constraint_bonus + structure_bonus


def select_diverse_queries(candidates: list[CandidateQuery], skill_text: str, keep: int) -> list[CandidateQuery]:
    scored: list[CandidateQuery] = []
    seen_texts: set[str] = set()
    for candidate in candidates:
        text_key = normalize_ws(candidate.text).lower()
        if not text_key or text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        score = relevance_score(candidate.text, skill_text, candidate.mode)
        if candidate.metadata.get("was_truncated"):
            score -= TRUNCATED_CANDIDATE_PENALTY
        scored.append(
            CandidateQuery(
                candidate.query_id,
                candidate.mode,
                candidate.text,
                candidate.source_rank,
                dict(candidate.metadata),
                score=score,
            )
        )
    if scored and all(candidate.mode == "task_spec" for candidate in scored):
        task_formatted = [candidate for candidate in scored if "you need to:" in candidate.text.lower()]
        if len(task_formatted) >= keep:
            scored = task_formatted
        untruncated = [candidate for candidate in scored if not candidate.metadata.get("was_truncated")]
        if len(untruncated) >= keep:
            scored = untruncated
    selected: list[CandidateQuery] = []
    while scored and len(selected) < keep:
        best_index = 0
        best_value = -1e9
        for idx, candidate in enumerate(scored):
            max_similarity = 0.0
            if selected:
                cand_tokens = token_set(candidate.text)
                max_similarity = max(jaccard(cand_tokens, token_set(item.text)) for item in selected)
            value = candidate.score - (2.0 * max_similarity) - (0.01 * candidate.source_rank)
            if value > best_value:
                best_value = value
                best_index = idx
        chosen = scored.pop(best_index)
        selected.append(chosen)
    return selected


def build_pseudo_row(
    query: str,
    skill_id: str,
    mode: str,
    rank: int,
    source_candidate_count: int,
    official_indices: list[str],
    train_frequency: int,
    score: float | None = None,
) -> dict[str, Any]:
    clean_query = normalize_ws(query)
    return {
        "index": f"pseudo:{mode}:{skill_id}:{rank}",
        "source": "official_single_longtail_pseudo_query",
        "query": clean_query,
        "query_text": f"{SKILLROUTER_QUERY_INSTRUCTION}{clean_query}",
        "target_skill_count": 1,
        "selected_skill_ids": [skill_id],
        "positive_skill_id": skill_id,
        "pseudo_query": {
            "mode": mode,
            "rank": rank,
            "source_candidate_count": source_candidate_count,
            "selection_score": score,
            "official_single_indices": official_indices,
            "train_frequency": train_frequency,
            "transductive_diagnostic": True,
        },
    }


def write_augmented_rows(
    base_paths: list[Path],
    pseudo_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
) -> None:
    base_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    for path in base_paths:
        base_rows.extend(load_jsonl(path))
    for path in pseudo_paths:
        pseudo_rows.extend(load_jsonl(path))
    write_jsonl(output_path, [*base_rows, *pseudo_rows])
    by_mode = Counter(str((row.get("pseudo_query") or {}).get("mode", "unknown")) for row in pseudo_rows)
    write_json(
        manifest_path,
        {
            "base_paths": [str(path) for path in base_paths],
            "pseudo_paths": [str(path) for path in pseudo_paths],
            "output_path": str(output_path),
            "base_rows": len(base_rows),
            "pseudo_rows": len(pseudo_rows),
            "total_rows": len(base_rows) + len(pseudo_rows),
            "pseudo_rows_by_mode": dict(sorted(by_mode.items())),
            "warning": "Transductive diagnostic data: uses official single GT skill ids.",
        },
    )


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    for key in ["base_url", "api_key", "model"]:
        if not profile.get(key):
            raise SystemExit(f"profile missing required field: {key}")
    return profile


def configure_profile_transport(profile: dict[str, Any]) -> None:
    proxy_url = str(profile.get("proxy_url") or "").strip()
    if proxy_url:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
        urllib.request.install_opener(opener)
    extra_headers = profile.get("extra_headers") or {}
    if isinstance(extra_headers, dict):
        single.DEFAULT_EXTRA_HEADERS = {str(k): str(v) for k, v in extra_headers.items()}


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    out = dict(profile)
    if "api_key" in out:
        out["api_key"] = "<redacted>"
    return out


def build_generation_messages(skill: dict[str, Any], mode: str, count: int, body_chars: int) -> list[dict[str, str]]:
    internal_name = skill_name(skill)
    description = str(skill.get("description") or skill.get("task_description") or "")[:300]
    body = str(skill.get("body") or "")[: min(max(body_chars, 0), 2500)]
    if mode == "task_spec":
        requirements = (
            "- Write a task that the user wants an AI agent to execute, not a search query, skill summary, "
            "tutorial request, or request for an explanation.\n"
            "- The task must be solvable using this skill alone. Do not add work that requires another specialized capability.\n"
            "- Open with one or two sentences establishing a concrete situation and the available input, artifact, or current state.\n"
            "- Then write `You need to:` followed by 2-5 numbered requirements in execution order. For an atomic skill, "
            "use preparation, the core action, and verification rather than inventing extra capabilities.\n"
            "- State an explicit deliverable or final outcome, followed by a supported acceptance or verification criterion.\n"
            "- You may invent neutral filenames, paths, row counts, and placeholder values solely to instantiate a supported workflow. "
            "Every domain-specific entity name, identifier, measurement, threshold, API, schema, command, and factual claim must be "
            "supported by the skill document. If a required specialized value is absent, use an explicit placeholder such as "
            "`<PDB_ID>` or say that it is user-provided; never fabricate it.\n"
            "- Preserve factual and operational grounding. Do not invent performance claims or evaluation procedures that the skill "
            "document does not support.\n"
            "- Do not mention the skill name, skill id, retrieval, or the existence of a skill.\n"
            "- Format each request as a multi-line benchmark task specification. Aim for 800-1,500 characters, but never pad it "
            "with unsupported requirements merely to reach the target length."
        )
    elif mode == "generic":
        requirements = (
            "- Every request must clearly require the capability in the reference document.\n"
            "- Write natural, concrete user requests rather than benchmark-style task specifications.\n"
            "- Vary the intent or context, not just wording.\n"
            "- Do not invent unsupported tools, formats, or domain facts.\n"
            "- Do not mention the skill name, skill id, or the existence of a skill.\n"
            "- Keep each request within 1,500 characters."
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    prompt = (
        "You generate synthetic retrieval queries for training a single-skill generative retriever.\n\n"
        "Use only the following private skill document as capability and factual grounding:\n"
        f"Skill name: {internal_name}\n"
        f"Description: {description}\n"
        f"Body:\n{body}\n\n"
        f"Generate exactly {count} self-contained, realistic, execution-oriented task request(s).\n\n"
        "Requirements:\n"
        f"{requirements}\n\n"
        "Return JSON only, as an array of objects with this schema:\n"
        '[{"query": "..."}]'
    )
    return [{"role": "user", "content": prompt}]


def build_isolated_mode_messages(
    skill: dict[str, Any],
    mode: str,
    body_chars: int,
) -> list[dict[str, str]]:
    if mode not in ISOLATED_TASK_MODES:
        raise ValueError(f"unknown isolated task mode: {mode}")
    mode_instruction = {
        "outcome_first": (
            "The opening sentence must foreground the concrete end result the user needs, not a failed attempt or a list of "
            "implementation steps. Keep the target capability implicit: describe the input state, desired outcome, deliverable, "
            "and verification without naming or defining the capability. Do not create diversity merely by changing entity names."
        ),
        "workflow_embedded": (
            "The opening sentence must establish a broader practical workflow and the artifact already available at this stage. "
            "Place the target capability as its decisive specialized step. Surrounding steps may only cover ordinary input handling, "
            "saving, or verification; do not introduce a second specialized capability or a second retrievable skill. Do not create "
            "diversity merely by changing entity names."
        ),
        "constraint_failure": (
            "The first sentence must explicitly describe a failed prior attempt, an edge case, or a hard operational constraint that "
            "makes the target capability necessary; a generic new implementation request does not satisfy this mode. The failure or "
            "constraint must be supported by the document or stated generically. Do not invent tool-specific error messages, APIs, "
            "thresholds, or factual claims, and do not create diversity merely by changing entity names."
        ),
    }[mode]
    internal_name = skill_name(skill)
    description = str(skill.get("description") or skill.get("task_description") or "")[:300]
    body = str(skill.get("body") or "")[: min(max(body_chars, 0), 2500)]
    prompt = (
        "Generate one synthetic retrieval query for training a single-skill generative retriever. "
        "The query must read like an executable benchmark task written by a real user.\n\n"
        "Use only this private skill document as capability and factual grounding:\n"
        f"Skill name: {internal_name}\n"
        f"Description: {description}\n"
        f"Body:\n{body}\n\n"
        f"Diversity mode: {mode}\n{mode_instruction}\n\n"
        "Requirements:\n"
        "- The task must be solvable using this skill alone and must have exactly one specialized retrieval target.\n"
        "- Open with one or two sentences giving a concrete situation and available input, artifact, or current state.\n"
        "- Then write `You need to:` followed by 2-5 numbered requirements in execution order.\n"
        "- End with an explicit deliverable and a supported acceptance or verification criterion.\n"
        "- Use neutral placeholders for specialized values absent from the document; never fabricate them.\n"
        "- Do not mention the skill name, skill id, retrieval, or the existence of a skill.\n"
        "- Preserve factual and operational grounding. Do not request unsupported tools, APIs, formats, or evaluations.\n"
        "- Aim for 800-1,500 characters, but do not pad the task with unsupported work.\n\n"
        "Return JSON only with exactly this schema:\n"
        '{"query": "..."}'
    )
    return [{"role": "user", "content": prompt}]


def call_query_generator(
    profile: dict[str, Any],
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    timeout: int,
    max_tokens: int,
    n: int = 1,
) -> dict[str, Any]:
    return single.call_chat_completion(
        profile["base_url"],
        profile["api_key"],
        profile["model"],
        messages,
        temperature,
        top_p,
        timeout,
        use_schema=False,
        max_tokens=max_tokens,
        n=n,
    )


def response_content(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"] or "").strip()


def response_choice_contents(response: dict[str, Any]) -> list[tuple[int, str]]:
    contents: list[tuple[int, str]] = []
    for position, choice in enumerate(response.get("choices") or []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        content = normalize_query_text(message.get("content") or "")
        choice_index = choice.get("index")
        if not isinstance(choice_index, int):
            choice_index = position
        contents.append((choice_index, content))
    return contents


def decode_json_string_fragment(raw: str, start: int) -> tuple[str, bool]:
    output: list[str] = []
    index = start
    escapes = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
    while index < len(raw):
        char = raw[index]
        if char == '"':
            return "".join(output), True
        if char != "\\":
            output.append(char)
            index += 1
            continue
        index += 1
        if index >= len(raw):
            break
        escaped = raw[index]
        if escaped == "u":
            code = raw[index + 1 : index + 5]
            if len(code) != 4 or not re.fullmatch(r"[0-9a-fA-F]{4}", code):
                break
            output.append(chr(int(code, 16)))
            index += 5
            continue
        output.append(escapes.get(escaped, escaped))
        index += 1
    return "".join(output), False


def recover_query_from_json_prefix(text: str) -> tuple[str, bool] | None:
    candidate = normalize_query_text(text)
    if not candidate.startswith(("[", "{")):
        return None
    match = re.search(r'"(?:query|text|task)"\s*:\s*"', candidate, flags=re.I)
    if match is None:
        return None
    query, closed = decode_json_string_fragment(candidate, match.end())
    query = normalize_query_text(query)
    return (query, closed) if query else None


def parse_single_generated_query_with_mode(text: str) -> tuple[str, str]:
    raw = normalize_query_text(text)
    if not raw:
        return "", "empty"
    candidate = raw
    if raw.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
        candidate = re.sub(r"```$", "", candidate).strip()
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        recovered = recover_query_from_json_prefix(candidate)
        if recovered is not None:
            query, closed = recovered
            return query, "json_prefix_complete" if closed else "json_prefix_truncated"
        return raw, "raw"
    parsed = parse_generated_queries(candidate)
    return (parsed[0], "json") if parsed else ("", "json_empty")


def parse_single_generated_query(text: str) -> str:
    query, _ = parse_single_generated_query_with_mode(text)
    return query


def build_candidates_from_response(
    response: dict[str, Any],
    mode: str,
    query_id_prefix: str,
    start_rank: int,
    metadata: dict[str, Any],
) -> list[CandidateQuery]:
    queries = parse_generated_queries(response_content(response))
    return [
        CandidateQuery(
            query_id=f"{query_id_prefix}:{start_rank + idx}",
            mode=mode,
            text=query,
            source_rank=start_rank + idx,
            metadata={**metadata, "raw_response_model": response.get("model")},
        )
        for idx, query in enumerate(queries)
    ]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonicalize_query(text: str, max_chars: int) -> str:
    query = normalize_query_text(text)
    if max_chars > 0 and len(query) > max_chars:
        prefix = query[:max_chars]
        minimum_boundary = int(max_chars * 0.8)
        boundaries = [
            match.end()
            for match in re.finditer(r"[.!?](?=\s|$)", prefix)
            if match.end() >= minimum_boundary
        ]
        query = prefix[: max(boundaries)].rstrip() if boundaries else prefix.rstrip()
    return query


def normalize_task_spec_layout(text: str) -> str:
    query = normalize_query_text(text)
    query = re.sub(r"\bVoc(?:ê|e) precisa\s*:", "You need to:", query, count=1, flags=re.I)
    query = re.sub(r"\bYou need\s*:", "You need to:", query, count=1, flags=re.I)
    if "you need to:" not in query.lower():
        first_item = re.search(r"(?:^|\n\s*\n)\s*(?=1[.)]\s)", query)
        if first_item is not None:
            query = f"{query[: first_item.start()].rstrip()}\n\nYou need to:\n{query[first_item.end() :].lstrip()}"
    query = re.sub(r"(You need to:)\s*(?=\d+[.)]\s)", r"\1\n", query, count=1, flags=re.I)
    marker = re.search(r"\bYou need to:\s*", query, flags=re.I)
    if marker is not None:
        prefix = query[: marker.end()].rstrip()
        body = re.sub(r"(?<!\n)\s+(?=\d+[.)]\s)", "\n", query[marker.end() :].lstrip())
        query = f"{prefix}\n{body}"
    return normalize_query_text(query)


def apply_task_spec_override(query: str, skill: dict[str, Any]) -> tuple[str, str | None]:
    text = normalize_query_text(query)
    if "you need to:" in text.lower():
        return text, None
    sid = skill_id(skill)
    if sid == "other/optimizing-query-text":
        wrapped = (
            "A Snowflake SQL statement needs to be optimized before production use. The current statement is:\n\n"
            f"```sql\n{text}\n```\n\n"
            "You need to:\n"
            "1. Rewrite the statement only where a documented safe optimization applies.\n"
            "2. Preserve the exact output columns and aliases, row set, ordering, and LIMIT behavior.\n"
            "3. If identical results cannot be guaranteed, leave the original statement unchanged.\n"
            "4. Return only the executable optimized SQL, without markdown or an explanation.\n\n"
            "Deliverable: One Snowflake SQL statement ready to execute.\n"
            "Acceptance criteria: The statement is semantically identical to the input and applies no unsafe optimization."
        )
        return wrapped, "snowflake_sql_input"
    if sid == "other/galaxy-query-generation":
        wrapped = (
            "You are adding a benchmark item for a Galaxy tutorial step. The intended user scenario is:\n\n"
            f"{text}\n\n"
            "The relevant tutorial context and tool metadata are available under `<TUTORIAL_ID>`.\n\n"
            "You need to:\n"
            "1. Read the tutorial around the corresponding tool step and verify the real user intent.\n"
            "2. Write one natural English query that asks for a Galaxy tool recommendation, states the data type and goal, "
            "and does not expose a tool name, tool ID, tutorial, filename, accession, or URL.\n"
            "3. Resolve exactly one stable Toolshed GUID or supported Galaxy internal tool ID for that query.\n"
            "4. Append the query and its `- tool:` line to `data/benchmark/v1_items.jsonl`.\n\n"
            "Deliverable: One new benchmark item containing the query and exactly one stable tool identifier.\n"
            "Acceptance criteria: The query and tool ID are consistent, the query follows the repository rules, and the item "
            "has no tool leakage or dataset-specific identifier."
        )
        return wrapped, "galaxy_benchmark_intent"
    return text, None


def validate_clean_generation_profile(profile: dict[str, Any]) -> None:
    profile_text = " ".join(
        str(profile.get(key) or "").lower()
        for key in ("name", "base_url", "model", "purpose")
    )
    if "openrouter" in profile_text:
        raise ValueError("OpenRouter is disabled for clean catalog generation; use the Huayan profile")


def build_catalog_pseudo_row(
    query: str,
    sid: str,
    rank: int,
    source_candidate_count: int,
    score: float,
    candidate_index: int,
    generator_model: str,
    source_was_truncated: bool,
    sanitized_reference_types: list[str] | None = None,
    task_spec_override: str | None = None,
) -> dict[str, Any]:
    clean_query = normalize_query_text(query)
    sanitized_types = sorted(set(sanitized_reference_types or []))
    return {
        "index": f"pseudo:catalog_task:{sid}:{rank}",
        "source": CLEAN_CATALOG_SOURCE,
        "query": clean_query,
        "query_text": f"{SKILLROUTER_QUERY_INSTRUCTION}{clean_query}",
        "target_skill_count": 1,
        "selected_skill_ids": [sid],
        "positive_skill_id": sid,
        "pseudo_query": {
            "mode": "task_spec",
            "rank": rank,
            "source_candidate_count": source_candidate_count,
            "selection_score": score,
            "candidate_index": candidate_index,
            "clean_catalog": True,
            "prompt_version": CLEAN_PROMPT_VERSION,
            "generator_model": generator_model,
            "source_was_truncated": source_was_truncated,
            "reference_sanitized": bool(sanitized_types),
            "sanitized_reference_types": sanitized_types,
            "task_spec_override": task_spec_override,
        },
    }


def build_catalog_generation_record(
    skill: dict[str, Any],
    response: dict[str, Any],
    requested_candidates: int,
    keep: int,
    max_query_chars: int,
) -> dict[str, Any]:
    sid = skill_id(skill)
    model = str(response.get("model") or "")
    candidates: list[CandidateQuery] = []
    choice_rows: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    contents = response_choice_contents(response)[:requested_candidates]

    for position, (choice_index, raw_content) in enumerate(contents):
        parsed_query, parse_mode = parse_single_generated_query_with_mode(raw_content)
        task_query, task_spec_override = apply_task_spec_override(parsed_query, skill)
        sanitized_query, sanitized_reference_types = sanitize_skill_references(task_query, skill)
        query = canonicalize_query(sanitized_query, max_query_chars)
        query_id = f"{sid}:task_spec:{position}"
        reasons: list[str] = []
        if not query:
            reasons.append("empty")
        else:
            reasons.extend(exact_leak_reasons(query, skill, official_queries=[]))
            text_key = normalize_ws(query).lower()
            if text_key in seen_texts:
                reasons.append("exact_duplicate")
            elif not reasons:
                seen_texts.add(text_key)

        was_truncated = bool(sanitized_query and len(sanitized_query) > len(query))
        metadata = {
            "choice_index": choice_index,
            "raw_response_model": model,
            "raw_character_count": len(sanitized_query),
            "canonical_character_count": len(query),
            "was_truncated": was_truncated,
            "sanitized_reference_types": sorted(set(sanitized_reference_types)),
            "parse_mode": parse_mode,
            "task_spec_override": task_spec_override,
        }
        choice_row = {
            "query_id": query_id,
            "choice_index": choice_index,
            "query": query,
            "source_rank": position,
            "features": candidate_features(query),
            "reject_reasons": reasons,
            "metadata": metadata,
            "selected": False,
            "selection_score": None,
        }
        if was_truncated:
            choice_row["raw_query"] = sanitized_query
        elif not query and raw_content:
            choice_row["unparsed_content"] = raw_content
        choice_rows.append(choice_row)
        if not reasons:
            candidates.append(
                CandidateQuery(
                    query_id=query_id,
                    mode="task_spec",
                    text=query,
                    source_rank=position,
                    metadata=metadata,
                )
            )

    selected = select_diverse_queries(candidates, skill_document_text(skill), keep)
    selected_by_id = {candidate.query_id: candidate for candidate in selected}
    for row in choice_rows:
        selected_candidate = selected_by_id.get(row["query_id"])
        if selected_candidate is not None:
            row["selected"] = True
            row["selection_score"] = selected_candidate.score

    selected_queries = [
        {
            "query_id": candidate.query_id,
            "source_rank": candidate.source_rank,
            "choice_index": int(candidate.metadata["choice_index"]),
            "selection_score": candidate.score,
            "source_was_truncated": bool(candidate.metadata["was_truncated"]),
            "sanitized_reference_types": list(candidate.metadata.get("sanitized_reference_types") or []),
            "task_spec_override": candidate.metadata.get("task_spec_override"),
        }
        for candidate in selected
    ]
    status = "complete" if len(selected_queries) == keep else "insufficient_candidates"
    return {
        "schema_version": 1,
        "skill_id": sid,
        "skill_name": skill_name(skill),
        "status": status,
        "created_at": utc_timestamp(),
        "prompt_version": CLEAN_PROMPT_VERSION,
        "selection_version": CLEAN_SELECTION_VERSION,
        "requested_candidates": requested_candidates,
        "returned_choices": len(contents),
        "accepted_candidates": len(candidates),
        "selected_count": len(selected_queries),
        "response_model": model,
        "usage": response.get("usage") or {},
        "choices": choice_rows,
        "selected_queries": selected_queries,
    }


def isolated_work_key(sid: str, mode: str) -> str:
    return f"{sid}\t{mode}"


def build_isolated_generation_record(
    skill: dict[str, Any],
    mode: str,
    response: dict[str, Any],
    max_query_chars: int,
) -> dict[str, Any]:
    sid = skill_id(skill)
    contents = response_choice_contents(response)
    raw_content = contents[0][1] if contents else ""
    parsed_query, parse_mode = parse_single_generated_query_with_mode(raw_content)
    task_query, task_spec_override = apply_task_spec_override(parsed_query, skill)
    normalized_task_query = normalize_task_spec_layout(task_query)
    sanitized_query, sanitized_reference_types = sanitize_skill_references(normalized_task_query, skill)
    query = canonicalize_query(sanitized_query, max_query_chars)
    reject_reasons = [] if query else ["empty"]
    if query:
        reject_reasons.extend(exact_leak_reasons(query, skill, official_queries=[]))
    was_truncated = bool(sanitized_query and len(sanitized_query) > len(query))
    return {
        "schema_version": 1,
        "work_key": isolated_work_key(sid, mode),
        "skill_id": sid,
        "skill_name": skill_name(skill),
        "mode": mode,
        "status": "complete" if not reject_reasons else "invalid_generation",
        "created_at": utc_timestamp(),
        "prompt_version": ISOLATED_PROMPT_VERSION,
        "response_model": str(response.get("model") or ""),
        "usage": response.get("usage") or {},
        "query": query,
        "raw_query": sanitized_query if was_truncated else None,
        "parse_mode": parse_mode,
        "source_was_truncated": was_truncated,
        "sanitized_reference_types": sorted(set(sanitized_reference_types)),
        "task_spec_override": task_spec_override,
        "reject_reasons": reject_reasons,
    }


def build_isolated_api_error_record(skill: dict[str, Any], mode: str, exc: Exception) -> dict[str, Any]:
    sid = skill_id(skill)
    return {
        "schema_version": 1,
        "work_key": isolated_work_key(sid, mode),
        "skill_id": sid,
        "skill_name": skill_name(skill),
        "mode": mode,
        "status": "api_error",
        "created_at": utc_timestamp(),
        "prompt_version": ISOLATED_PROMPT_VERSION,
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
    }


def build_isolated_pseudo_row(record: dict[str, Any], generator_model: str) -> dict[str, Any]:
    sid = str(record["skill_id"])
    mode = str(record["mode"])
    query = normalize_task_spec_layout(record["query"])
    sanitized_types = sorted(set(record.get("sanitized_reference_types") or []))
    return {
        "index": f"pseudo:zero_multi_mode:{mode}:{sid}",
        "source": ISOLATED_SOURCE,
        "query": query,
        "query_text": f"{SKILLROUTER_QUERY_INSTRUCTION}{query}",
        "target_skill_count": 1,
        "selected_skill_ids": [sid],
        "positive_skill_id": sid,
        "pseudo_query": {
            "mode": mode,
            "rank": 0,
            "clean_catalog": True,
            "selection_basis": "zero_multi_row_cooccurrence",
            "prompt_version": ISOLATED_PROMPT_VERSION,
            "materialization_version": ISOLATED_MATERIALIZATION_VERSION,
            "generator_model": str(record.get("response_model") or generator_model),
            "parse_mode": str(record.get("parse_mode") or ""),
            "source_was_truncated": bool(record.get("source_was_truncated")),
            "reference_sanitized": bool(sanitized_types),
            "sanitized_reference_types": sanitized_types,
            "task_spec_override": record.get("task_spec_override"),
        },
    }


def reselect_catalog_generation_record(
    skill: dict[str, Any],
    record: dict[str, Any],
    keep: int,
    max_query_chars: int = 1500,
) -> dict[str, Any]:
    if record.get("status") == "api_error" or not isinstance(record.get("choices"), list):
        return record

    accepted: list[CandidateQuery] = []
    seen_texts: set[str] = set()
    choice_rows: list[dict[str, Any]] = []
    for position, original in enumerate(record.get("choices") or []):
        row = dict(original)
        source_query = normalize_query_text(row.get("raw_query") or row.get("query") or "")
        recovered_source, parse_mode = parse_single_generated_query_with_mode(source_query)
        task_source, new_task_override = apply_task_spec_override(recovered_source, skill)
        sanitized_source, new_sanitizations = sanitize_skill_references(task_source, skill)
        query = canonicalize_query(sanitized_source, max_query_chars)
        row["query"] = query
        was_truncated = bool(sanitized_source and len(sanitized_source) > len(query))
        metadata = dict(row.get("metadata") or {})
        sanitized_types = sorted(
            set(metadata.get("sanitized_reference_types") or []) | set(new_sanitizations)
        )
        metadata["raw_character_count"] = len(sanitized_source)
        metadata["canonical_character_count"] = len(query)
        metadata["was_truncated"] = was_truncated
        metadata["sanitized_reference_types"] = sanitized_types
        if parse_mode != "raw" or not metadata.get("parse_mode"):
            metadata["parse_mode"] = parse_mode
        if new_task_override is not None:
            metadata["task_spec_override"] = new_task_override
        row["metadata"] = metadata
        if was_truncated:
            row["raw_query"] = sanitized_source
        else:
            row.pop("raw_query", None)
        reasons: list[str] = []
        if not query:
            reasons.append("empty")
        else:
            reasons.extend(exact_leak_reasons(query, skill, official_queries=[]))
            text_key = normalize_ws(query).lower()
            if text_key in seen_texts:
                reasons.append("exact_duplicate")
            elif not reasons:
                seen_texts.add(text_key)
        row["reject_reasons"] = reasons
        row["selected"] = False
        row["selection_score"] = None
        choice_rows.append(row)
        if not reasons:
            accepted.append(
                CandidateQuery(
                    query_id=str(row.get("query_id") or f"{skill_id(skill)}:task_spec:{position}"),
                    mode="task_spec",
                    text=query,
                    source_rank=int(row.get("source_rank", position)),
                    metadata=metadata,
                )
            )

    selected = select_diverse_queries(accepted, skill_document_text(skill), keep)
    selected_by_id = {candidate.query_id: candidate for candidate in selected}
    for row in choice_rows:
        candidate = selected_by_id.get(str(row.get("query_id") or ""))
        if candidate is not None:
            row["selected"] = True
            row["selection_score"] = candidate.score

    out = dict(record)
    out["selection_version"] = CLEAN_SELECTION_VERSION
    out["accepted_candidates"] = len(accepted)
    out["selected_count"] = len(selected)
    out["status"] = "complete" if len(selected) == keep else "insufficient_candidates"
    out["choices"] = choice_rows
    out["selected_queries"] = [
        {
            "query_id": candidate.query_id,
            "source_rank": candidate.source_rank,
            "choice_index": int(candidate.metadata.get("choice_index", candidate.source_rank)),
            "selection_score": candidate.score,
            "source_was_truncated": bool(candidate.metadata.get("was_truncated")),
            "sanitized_reference_types": list(candidate.metadata.get("sanitized_reference_types") or []),
            "task_spec_override": candidate.metadata.get("task_spec_override"),
        }
        for candidate in selected
    ]
    return out


def build_catalog_api_error_record(skill: dict[str, Any], requested_candidates: int, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "skill_id": skill_id(skill),
        "skill_name": skill_name(skill),
        "status": "api_error",
        "created_at": utc_timestamp(),
        "prompt_version": CLEAN_PROMPT_VERSION,
        "selection_version": CLEAN_SELECTION_VERSION,
        "requested_candidates": requested_candidates,
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
    }


def load_generation_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    malformed: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed.append(line_number)
    if malformed:
        print(
            f"warning: ignored malformed checkpoint lines in {path}: {malformed[:10]}",
            file=sys.stderr,
        )
    return rows


def latest_generation_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in records:
        sid = str(row.get("skill_id") or "")
        if sid:
            latest[sid] = row
    return latest


def latest_isolated_generation_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in records:
        sid = str(row.get("skill_id") or "")
        mode = str(row.get("mode") or "")
        if sid and mode:
            latest[isolated_work_key(sid, mode)] = row
    return latest


def load_skill_id_filter(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        return [str(item) for item in payload if str(item).strip()]
    ids: list[str] = []
    for line in raw.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            row = json.loads(value)
        except json.JSONDecodeError:
            ids.append(value)
        else:
            if isinstance(row, str):
                ids.append(row)
            elif isinstance(row, dict) and skill_id(row):
                ids.append(skill_id(row))
            else:
                raise ValueError(f"cannot read skill id from {path}: {value[:120]}")
    return ids


def select_catalog_skills(
    catalog_rows: list[dict[str, Any]],
    skill_ids_file: Path | None,
    limit_skills: int,
) -> list[dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in catalog_rows:
        sid = skill_id(row)
        if not sid:
            raise ValueError("catalog contains a row without skill_id")
        if sid in lookup:
            raise ValueError(f"catalog contains duplicate skill_id: {sid}")
        lookup[sid] = row
        order.append(sid)

    requested_ids = load_skill_id_filter(skill_ids_file)
    if requested_ids is not None:
        duplicate_ids = [sid for sid, count in Counter(requested_ids).items() if count > 1]
        if duplicate_ids:
            raise ValueError(f"skill id filter contains duplicates: {duplicate_ids[:10]}")
        missing = [sid for sid in requested_ids if sid not in lookup]
        if missing:
            raise ValueError(f"skill id filter is missing from catalog: {missing[:10]}")
        order = requested_ids
    if limit_skills > 0:
        order = order[:limit_skills]
    return [lookup[sid] for sid in order]


def catalog_skill_digest(skills: list[dict[str, Any]]) -> str:
    payload = "\n".join(skill_id(skill) for skill in skills).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_catalog_run_config(
    args: argparse.Namespace,
    profile: dict[str, Any],
    skills: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "clean_catalog": True,
        "prompt_version": CLEAN_PROMPT_VERSION,
        "skills_jsonl": str(args.skills_jsonl),
        "skill_count": len(skills),
        "skill_ids_sha256": catalog_skill_digest(skills),
        "profile_name": str(profile.get("name") or ""),
        "base_url": str(profile.get("base_url") or ""),
        "model": str(profile.get("model") or ""),
        "candidate_count": args.candidate_count,
        "keep_per_skill": args.keep_per_skill,
        "body_chars": args.body_chars,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_query_chars": args.max_query_chars,
    }


def isolated_skill_document_digest(skills: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(skill, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for skill in skills
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_isolated_run_config(
    args: argparse.Namespace,
    profile: dict[str, Any],
    skills: list[dict[str, Any]],
    modes: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "clean_catalog": True,
        "selection_basis": "zero_multi_row_cooccurrence",
        "prompt_version": ISOLATED_PROMPT_VERSION,
        "train_rows": str(args.train_rows),
        "skills_jsonl": str(args.skills_jsonl),
        "target_skill_count": len(skills),
        "target_skill_ids_sha256": catalog_skill_digest(skills),
        "target_skill_documents_sha256": isolated_skill_document_digest(skills),
        "modes": modes,
        "request_count": len(skills) * len(modes),
        "profile_name": str(profile.get("name") or ""),
        "base_url": str(profile.get("base_url") or ""),
        "model": str(profile.get("model") or ""),
        "body_chars": args.body_chars,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_query_chars": args.max_query_chars,
    }


def ensure_catalog_run_config(path: Path, config: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            changed = sorted(key for key in set(existing) | set(config) if existing.get(key) != config.get(key))
            raise ValueError(
                f"refusing to resume with a different run configuration ({', '.join(changed)}); "
                "use a new output directory"
            )
        return
    write_json(path, config)


def aggregate_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in records:
        usage = row.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[str(key)] += value
    return dict(sorted(totals.items()))


def forbidden_clean_metadata_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text.startswith("official") or "transductive" in key_text:
                found.append(str(key))
            found.extend(forbidden_clean_metadata_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(forbidden_clean_metadata_keys(item))
    return found


def validate_clean_catalog_pseudo_rows(
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    queries_per_skill: int,
    max_query_chars: int,
) -> None:
    skill_lookup = {skill_id(skill): skill for skill in skills}
    counts: Counter[str] = Counter()
    query_keys: dict[str, set[str]] = defaultdict(set)
    indices: set[str] = set()
    issues: list[str] = []

    for row_number, row in enumerate(rows, 1):
        ids = selected_skill_ids(row)
        sid = ids[0] if len(ids) == 1 else ""
        query = normalize_query_text(row.get("query") or "")
        index = str(row.get("index") or "")
        metadata = row.get("pseudo_query") or {}
        if len(ids) != 1 or row.get("target_skill_count") != 1 or row.get("positive_skill_id") != sid:
            issues.append(f"row {row_number}: invalid single-skill labels")
        if sid not in skill_lookup:
            issues.append(f"row {row_number}: unknown skill_id {sid!r}")
        if row.get("source") != CLEAN_CATALOG_SOURCE or metadata.get("clean_catalog") is not True:
            issues.append(f"row {row_number}: invalid clean catalog provenance")
        forbidden = forbidden_clean_metadata_keys(metadata)
        if forbidden:
            issues.append(f"row {row_number}: forbidden metadata keys {forbidden[:3]}")
        if not query or (max_query_chars > 0 and len(query) > max_query_chars):
            issues.append(f"row {row_number}: invalid query length {len(query)}")
        stripped_query = query.lstrip()
        if stripped_query.startswith("```json") or re.match(r'^[\[{]\s*[\[{\"]?\s*"query"', stripped_query, re.I):
            issues.append(f"row {row_number}: literal JSON wrapper in query")
        if re.match(r"^(?:SELECT|WITH|INSERT|UPDATE|DELETE)\b", stripped_query, re.I):
            issues.append(f"row {row_number}: bare SQL is not a task specification")
        if sid in {"other/optimizing-query-text", "other/galaxy-query-generation"} and "you need to:" not in query.lower():
            issues.append(f"row {row_number}: missing required task-spec override for {sid}")
        if row.get("query_text") != f"{SKILLROUTER_QUERY_INSTRUCTION}{query}":
            issues.append(f"row {row_number}: query_text mismatch")
        if index in indices:
            issues.append(f"row {row_number}: duplicate index {index}")
        indices.add(index)
        if sid in skill_lookup:
            leak_reasons = exact_leak_reasons(query, skill_lookup[sid], official_queries=[])
            if leak_reasons:
                issues.append(f"row {row_number}: skill leak {leak_reasons}")
        query_key = normalize_ws(query).lower()
        if sid and query_key in query_keys[sid]:
            issues.append(f"row {row_number}: duplicate query for {sid}")
        query_keys[sid].add(query_key)
        counts[sid] += 1

    expected_ids = set(skill_lookup)
    actual_ids = {sid for sid, count in counts.items() if count > 0}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        issues.append(f"skill coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")
    wrong_counts = {sid: counts.get(sid, 0) for sid in expected_ids if counts.get(sid, 0) != queries_per_skill}
    if wrong_counts:
        issues.append(f"wrong per-skill query counts: {dict(list(sorted(wrong_counts.items()))[:5])}")
    if issues:
        raise ValueError("clean catalog pseudo-row validation failed:\n- " + "\n- ".join(issues[:20]))


def validate_isolated_mode_pseudo_rows(
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    modes: list[str],
    max_query_chars: int,
    base_rows: list[dict[str, Any]] | None = None,
) -> None:
    skill_lookup = {skill_id(skill): skill for skill in skills}
    expected_pairs = {(sid, mode) for sid in skill_lookup for mode in modes}
    actual_pairs: set[tuple[str, str]] = set()
    indices: set[str] = set()
    query_keys: set[str] = {
        normalize_ws(str(row.get("query") or "")).lower()
        for row in (base_rows or [])
        if normalize_ws(str(row.get("query") or ""))
    }
    issues: list[str] = []

    for row_number, row in enumerate(rows, 1):
        ids = selected_skill_ids(row)
        sid = ids[0] if len(ids) == 1 else ""
        metadata = row.get("pseudo_query") or {}
        mode = str(metadata.get("mode") or "")
        query = normalize_query_text(row.get("query") or "")
        index = str(row.get("index") or "")
        pair = (sid, mode)
        if len(ids) != 1 or row.get("target_skill_count") != 1 or row.get("positive_skill_id") != sid:
            issues.append(f"row {row_number}: invalid single-skill labels")
        if sid not in skill_lookup:
            issues.append(f"row {row_number}: unknown target skill {sid!r}")
        if mode not in modes:
            issues.append(f"row {row_number}: unknown task mode {mode!r}")
        if pair in actual_pairs:
            issues.append(f"row {row_number}: duplicate skill-mode pair {pair}")
        actual_pairs.add(pair)
        if row.get("source") != ISOLATED_SOURCE:
            issues.append(f"row {row_number}: invalid source")
        if metadata.get("clean_catalog") is not True:
            issues.append(f"row {row_number}: missing clean provenance")
        if metadata.get("selection_basis") != "zero_multi_row_cooccurrence":
            issues.append(f"row {row_number}: invalid selection basis")
        if metadata.get("prompt_version") != ISOLATED_PROMPT_VERSION:
            issues.append(f"row {row_number}: invalid prompt version")
        if metadata.get("materialization_version") != ISOLATED_MATERIALIZATION_VERSION:
            issues.append(f"row {row_number}: invalid materialization version")
        forbidden = forbidden_clean_metadata_keys(metadata)
        if forbidden:
            issues.append(f"row {row_number}: forbidden metadata keys {forbidden[:3]}")
        if not query or len(query) > max_query_chars:
            issues.append(f"row {row_number}: invalid query length {len(query)}")
        if "you need to:" not in query.lower():
            issues.append(f"row {row_number}: missing task-spec marker")
        if row.get("query_text") != f"{SKILLROUTER_QUERY_INSTRUCTION}{query}":
            issues.append(f"row {row_number}: query_text mismatch")
        if index in indices:
            issues.append(f"row {row_number}: duplicate index {index}")
        indices.add(index)
        if sid in skill_lookup:
            leak_reasons = exact_leak_reasons(query, skill_lookup[sid], official_queries=[])
            if leak_reasons:
                issues.append(f"row {row_number}: skill leak {leak_reasons}")
        query_key = normalize_ws(query).lower()
        if query_key and query_key in query_keys:
            issues.append(f"row {row_number}: exact query duplicate")
        query_keys.add(query_key)

    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        issues.append(f"skill-mode coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if issues:
        raise ValueError("isolated mode pseudo-row validation failed:\n- " + "\n- ".join(issues[:20]))


def selected_query_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    lengths = [len(str(row.get("query") or "")) for row in rows]
    words = [word_count(str(row.get("query") or "")) for row in rows]
    numbered = [bool(re.search(r"(^|\n)\s*\d+[\).]", str(row.get("query") or ""))) for row in rows]
    task_markers = ["you need to:" in str(row.get("query") or "").lower() for row in rows]
    truncated = [bool((row.get("pseudo_query") or {}).get("source_was_truncated")) for row in rows]
    sanitized = [bool((row.get("pseudo_query") or {}).get("reference_sanitized")) for row in rows]
    overridden = [bool((row.get("pseudo_query") or {}).get("task_spec_override")) for row in rows]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[selected_skill_ids(row)[0]].append(row)
    pairwise: list[float] = []
    per_skill_max: list[float] = []
    for skill_rows in grouped.values():
        values: list[float] = []
        for left in range(len(skill_rows)):
            for right in range(left + 1, len(skill_rows)):
                values.append(jaccard(token_set(skill_rows[left]["query"]), token_set(skill_rows[right]["query"])))
        pairwise.extend(values)
        if values:
            per_skill_max.append(max(values))
    return {
        "count": len(rows),
        "character_count": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "word_count": {
            "min": min(words),
            "max": max(words),
            "mean": sum(words) / len(words),
        },
        "numbered_requirements_rate": sum(numbered) / len(numbered),
        "you_need_to_rate": sum(task_markers) / len(task_markers),
        "source_truncated_rate": sum(truncated) / len(truncated),
        "reference_sanitized_rate": sum(sanitized) / len(sanitized),
        "task_spec_override_rate": sum(overridden) / len(overridden),
        "pairwise_token_jaccard_mean": sum(pairwise) / len(pairwise) if pairwise else None,
        "per_skill_max_token_jaccard_mean": sum(per_skill_max) / len(per_skill_max) if per_skill_max else None,
    }


def materialize_catalog_outputs(
    output_dir: Path,
    skills: list[dict[str, Any]],
    profile: dict[str, Any],
    run_config: dict[str, Any],
    keep_per_skill: int,
    max_query_chars: int,
) -> dict[str, Any]:
    generations_path = output_dir / "catalog_generations.jsonl"
    all_records = load_generation_records(generations_path)
    latest = latest_generation_records(all_records)
    selected_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    complete_skills: list[dict[str, Any]] = []

    for skill in skills:
        sid = skill_id(skill)
        record = latest.get(sid)
        if record is not None:
            record = reselect_catalog_generation_record(skill, record, keep_per_skill, max_query_chars)
            latest[sid] = record
        if record is None or record.get("status") != "complete":
            failure = {
                "skill_id": sid,
                "skill_name": skill_name(skill),
                "status": str((record or {}).get("status") or "missing"),
            }
            for key in ("returned_choices", "accepted_candidates", "selected_count", "error_type", "error"):
                if record is not None and key in record:
                    failure[key] = record[key]
            failures.append(failure)
            continue

        selected = record.get("selected_queries") or []
        if len(selected) != keep_per_skill:
            failures.append(
                {
                    "skill_id": sid,
                    "skill_name": skill_name(skill),
                    "status": "invalid_complete_record",
                    "selected_count": len(selected),
                }
            )
            continue
        choice_lookup = {str(item.get("query_id") or ""): item for item in (record.get("choices") or [])}
        if any(str(item.get("query_id") or "") not in choice_lookup for item in selected):
            failures.append(
                {
                    "skill_id": sid,
                    "skill_name": skill_name(skill),
                    "status": "invalid_complete_record",
                    "error": "selected query is missing from choices",
                }
            )
            continue
        complete_skills.append(skill)
        for rank, item in enumerate(selected):
            choice = choice_lookup[str(item["query_id"])]
            selected_rows.append(
                build_catalog_pseudo_row(
                    query=str(choice["query"]),
                    sid=sid,
                    rank=rank,
                    source_candidate_count=int(record.get("returned_choices") or 0),
                    score=float(item["selection_score"]),
                    candidate_index=int(item["choice_index"]),
                    generator_model=str(record.get("response_model") or profile.get("model") or ""),
                    source_was_truncated=bool(item.get("source_was_truncated")),
                    sanitized_reference_types=list(item.get("sanitized_reference_types") or []),
                    task_spec_override=item.get("task_spec_override"),
                )
            )

    if complete_skills:
        validate_clean_catalog_pseudo_rows(
            selected_rows,
            complete_skills,
            queries_per_skill=keep_per_skill,
            max_query_chars=max_query_chars,
        )
    selected_path = output_dir / f"selected_task{keep_per_skill}.jsonl"
    failures_path = output_dir / "failures.jsonl"
    write_jsonl(selected_path, selected_rows)
    write_jsonl(failures_path, failures)

    status_counts = Counter(str(row.get("status") or "unknown") for row in latest.values())
    manifest = {
        "clean_catalog": True,
        "complete": not failures and len(selected_rows) == len(skills) * keep_per_skill,
        "selection_version": CLEAN_SELECTION_VERSION,
        "run_config": run_config,
        "profile": public_profile(profile),
        "generation_checkpoint": str(generations_path),
        "selected_rows_path": str(selected_path),
        "failures_path": str(failures_path),
        "target_skill_count": len(skills),
        "complete_skill_count": len(complete_skills),
        "failed_skill_count": len(failures),
        "expected_selected_rows": len(skills) * keep_per_skill,
        "selected_rows": len(selected_rows),
        "checkpoint_records": len(all_records),
        "latest_status_counts": dict(sorted(status_counts.items())),
        "all_attempt_usage": aggregate_usage(all_records),
        "latest_record_usage": aggregate_usage(list(latest.values())),
        "selected_query_stats": selected_query_stats(selected_rows),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def command_generate_catalog(args: argparse.Namespace) -> None:
    if args.candidate_count < args.keep_per_skill:
        raise SystemExit("--candidate-count must be >= --keep-per-skill")
    if args.keep_per_skill < 1:
        raise SystemExit("--keep-per-skill must be >= 1")
    if args.max_query_chars < 1:
        raise SystemExit("--max-query-chars must be >= 1")

    catalog_rows = load_jsonl(args.skills_jsonl)
    skills = select_catalog_skills(catalog_rows, args.skill_ids_file, args.limit_skills)
    if not skills:
        raise SystemExit("no catalog skills selected")
    profile = load_profile(args.profile)
    try:
        validate_clean_generation_profile(profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    configure_profile_transport(profile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_catalog_run_config(args, profile, skills)
    try:
        ensure_catalog_run_config(args.output_dir / "run_config.json", run_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    generations_path = args.output_dir / "catalog_generations.jsonl"
    existing_records = load_generation_records(generations_path)
    latest = latest_generation_records(existing_records)
    pending = [
        skill
        for skill in skills
        if str((latest.get(skill_id(skill)) or {}).get("status") or "") not in CLEAN_TERMINAL_STATUSES
    ]
    skipped = len(skills) - len(pending)
    print(
        f"clean catalog generation: skills={len(skills)} pending={len(pending)} resumed={skipped} "
        f"n={args.candidate_count} keep={args.keep_per_skill}",
        flush=True,
    )

    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed = 0

    def run_skill(skill: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        try:
            messages = build_generation_messages(skill, "task_spec", 1, args.body_chars)
            response = call_query_generator(
                profile,
                messages,
                args.temperature,
                args.top_p,
                args.timeout,
                args.max_tokens,
                n=args.candidate_count,
            )
            record = build_catalog_generation_record(
                skill,
                response,
                requested_candidates=args.candidate_count,
                keep=args.keep_per_skill,
                max_query_chars=args.max_query_chars,
            )
        except Exception as exc:  # Keep the checkpoint usable after isolated transport failures.
            record = build_catalog_api_error_record(skill, args.candidate_count, exc)
        append_jsonl(generations_path, record, write_lock)
        with progress_lock:
            completed += 1
            if completed == len(pending) or completed % args.progress_every == 0:
                print(
                    f"progress={completed}/{len(pending)} skill_id={skill_id(skill)} status={record['status']}",
                    flush=True,
                )
        return record

    if args.workers == 1:
        for skill in pending:
            run_skill(skill)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(run_skill, pending))

    manifest = materialize_catalog_outputs(
        args.output_dir,
        skills,
        profile,
        run_config,
        args.keep_per_skill,
        args.max_query_chars,
    )
    print(
        f"saved clean catalog outputs to {args.output_dir}; "
        f"complete_skills={manifest['complete_skill_count']}/{manifest['target_skill_count']} "
        f"selected_rows={manifest['selected_rows']}",
        flush=True,
    )
    if args.require_complete and not manifest["complete"]:
        raise SystemExit(2)


def materialize_isolated_outputs(
    output_dir: Path,
    skills: list[dict[str, Any]],
    modes: list[str],
    profile: dict[str, Any],
    run_config: dict[str, Any],
    max_query_chars: int,
) -> dict[str, Any]:
    generations_path = output_dir / "isolated_mode_generations.jsonl"
    all_records = load_generation_records(generations_path)
    latest = latest_isolated_generation_records(all_records)
    selected_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for skill in skills:
        sid = skill_id(skill)
        for mode in modes:
            record = latest.get(isolated_work_key(sid, mode))
            if record is None or record.get("status") != "complete":
                failure = {
                    "skill_id": sid,
                    "skill_name": skill_name(skill),
                    "mode": mode,
                    "status": str((record or {}).get("status") or "missing"),
                }
                for key in ("error_type", "error", "reject_reasons"):
                    if record is not None and key in record:
                        failure[key] = record[key]
                failures.append(failure)
                continue
            selected_rows.append(build_isolated_pseudo_row(record, str(profile.get("model") or "")))

    validation_error = None
    if not failures:
        try:
            validate_isolated_mode_pseudo_rows(selected_rows, skills, modes, max_query_chars)
        except ValueError as exc:
            validation_error = str(exc)
    selected_path = output_dir / f"selected_mode{len(modes)}.jsonl"
    failures_path = output_dir / "failures.jsonl"
    write_jsonl(selected_path, selected_rows)
    write_jsonl(failures_path, failures)
    write_jsonl(
        output_dir / "target_skills.jsonl",
        [
            {
                "skill_id": skill_id(skill),
                "skill_name": skill_name(skill),
                "multi_row_frequency": 0,
            }
            for skill in skills
        ],
    )
    status_counts = Counter(str(row.get("status") or "unknown") for row in latest.values())
    expected_rows = len(skills) * len(modes)
    manifest = {
        "clean_catalog": True,
        "complete": not failures and validation_error is None and len(selected_rows) == expected_rows,
        "selection_basis": "zero_multi_row_cooccurrence",
        "prompt_version": ISOLATED_PROMPT_VERSION,
        "materialization_version": ISOLATED_MATERIALIZATION_VERSION,
        "run_config": run_config,
        "profile": public_profile(profile),
        "generation_checkpoint": str(generations_path),
        "selected_rows_path": str(selected_path),
        "failures_path": str(failures_path),
        "target_skills_path": str(output_dir / "target_skills.jsonl"),
        "target_skill_count": len(skills),
        "modes": modes,
        "expected_selected_rows": expected_rows,
        "selected_rows": len(selected_rows),
        "failed_requests": len(failures),
        "validation_error": validation_error,
        "checkpoint_records": len(all_records),
        "latest_status_counts": dict(sorted(status_counts.items())),
        "all_attempt_usage": aggregate_usage(all_records),
        "latest_record_usage": aggregate_usage(list(latest.values())),
        "selected_query_stats": selected_query_stats(selected_rows),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def command_generate_isolated_modes(args: argparse.Namespace) -> None:
    if args.max_query_chars < 1:
        raise SystemExit("--max-query-chars must be >= 1")
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    if not modes or len(modes) != len(set(modes)):
        raise SystemExit("--modes must contain distinct task modes")
    unknown_modes = [mode for mode in modes if mode not in ISOLATED_TASK_MODES]
    if unknown_modes:
        raise SystemExit(f"unknown isolated task modes: {unknown_modes}")

    catalog_rows = load_jsonl(args.skills_jsonl)
    train_rows = load_jsonl(args.train_rows)
    try:
        skills = select_zero_multi_cooccurrence_skills(catalog_rows, train_rows, args.limit_skills)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not skills:
        raise SystemExit("no zero-multi-cooccurrence catalog skills selected")
    profile = load_profile(args.profile)
    try:
        validate_clean_generation_profile(profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    configure_profile_transport(profile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_isolated_run_config(args, profile, skills, modes)
    try:
        ensure_catalog_run_config(args.output_dir / "run_config.json", run_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    generations_path = args.output_dir / "isolated_mode_generations.jsonl"
    latest = latest_isolated_generation_records(load_generation_records(generations_path))
    work = [
        (skill, mode)
        for skill in skills
        for mode in modes
        if str((latest.get(isolated_work_key(skill_id(skill), mode)) or {}).get("status") or "")
        not in ISOLATED_TERMINAL_STATUSES
    ]
    print(
        f"isolated mode generation: skills={len(skills)} modes={len(modes)} "
        f"requests={len(skills) * len(modes)} pending={len(work)} resumed={len(skills) * len(modes) - len(work)}",
        flush=True,
    )
    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed = 0

    def run_item(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
        nonlocal completed
        skill, mode = item
        try:
            messages = build_isolated_mode_messages(skill, mode, args.body_chars)
            response = call_query_generator(
                profile,
                messages,
                args.temperature,
                args.top_p,
                args.timeout,
                args.max_tokens,
                n=1,
            )
            record = build_isolated_generation_record(skill, mode, response, args.max_query_chars)
        except Exception as exc:  # Only transport failures are retried when the command resumes.
            record = build_isolated_api_error_record(skill, mode, exc)
        append_jsonl(generations_path, record, write_lock)
        with progress_lock:
            completed += 1
            if completed == len(work) or completed % args.progress_every == 0:
                print(
                    f"progress={completed}/{len(work)} skill_id={skill_id(skill)} mode={mode} status={record['status']}",
                    flush=True,
                )
        return record

    if args.workers == 1:
        for item in work:
            run_item(item)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(run_item, work))

    manifest = materialize_isolated_outputs(
        args.output_dir,
        skills,
        modes,
        profile,
        run_config,
        args.max_query_chars,
    )
    print(
        f"saved isolated mode outputs to {args.output_dir}; "
        f"selected_rows={manifest['selected_rows']}/{manifest['expected_selected_rows']} "
        f"failed_requests={manifest['failed_requests']}",
        flush=True,
    )
    if args.require_complete and not manifest["complete"]:
        raise SystemExit(2)


def write_clean_augmented_rows(
    base_paths: list[Path],
    pseudo_paths: list[Path],
    skills_path: Path,
    output_path: Path,
    manifest_path: Path,
    queries_per_skill: int,
    max_query_chars: int,
) -> None:
    base_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    for path in base_paths:
        base_rows.extend(load_jsonl(path))
    for path in pseudo_paths:
        pseudo_rows.extend(load_jsonl(path))
    skills = load_jsonl(skills_path)
    validate_clean_catalog_pseudo_rows(pseudo_rows, skills, queries_per_skill, max_query_chars)

    all_indices: set[str] = set()
    duplicate_indices: list[str] = []
    for row in [*base_rows, *pseudo_rows]:
        index = str(row.get("index") or "")
        if index and index in all_indices:
            duplicate_indices.append(index)
        all_indices.add(index)
    if duplicate_indices:
        raise ValueError(f"duplicate indices across augmented rows: {duplicate_indices[:10]}")

    write_jsonl(output_path, [*base_rows, *pseudo_rows])
    write_json(
        manifest_path,
        {
            "clean_catalog": True,
            "base_paths": [str(path) for path in base_paths],
            "pseudo_paths": [str(path) for path in pseudo_paths],
            "skills_jsonl": str(skills_path),
            "output_path": str(output_path),
            "catalog_skills": len(skills),
            "queries_per_skill": queries_per_skill,
            "base_rows": len(base_rows),
            "pseudo_rows": len(pseudo_rows),
            "total_rows": len(base_rows) + len(pseudo_rows),
            "pseudo_source": CLEAN_CATALOG_SOURCE,
        },
    )


def command_augment_catalog(args: argparse.Namespace) -> None:
    try:
        write_clean_augmented_rows(
            args.base_rows,
            args.pseudo_rows,
            args.skills_jsonl,
            args.output,
            args.manifest,
            args.queries_per_skill,
            args.max_query_chars,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"saved clean augmented rows to {args.output}")


def write_isolated_augmented_rows(
    base_paths: list[Path],
    pseudo_paths: list[Path],
    skills_path: Path,
    output_path: Path,
    manifest_path: Path,
    modes: list[str],
    max_query_chars: int,
) -> None:
    base_rows = [row for path in base_paths for row in load_jsonl(path)]
    pseudo_rows = [row for path in pseudo_paths for row in load_jsonl(path)]
    catalog_rows = load_jsonl(skills_path)
    targets = select_zero_multi_cooccurrence_skills(catalog_rows, base_rows)
    validate_isolated_mode_pseudo_rows(
        pseudo_rows,
        targets,
        modes,
        max_query_chars,
        base_rows=base_rows,
    )

    all_indices: set[str] = set()
    duplicate_indices: list[str] = []
    for row in [*base_rows, *pseudo_rows]:
        index = str(row.get("index") or "")
        if index and index in all_indices:
            duplicate_indices.append(index)
        all_indices.add(index)
    if duplicate_indices:
        raise ValueError(f"duplicate indices across augmented rows: {duplicate_indices[:10]}")

    write_jsonl(output_path, [*base_rows, *pseudo_rows])
    write_json(
        manifest_path,
        {
            "clean_catalog": True,
            "selection_basis": "zero_multi_row_cooccurrence",
            "base_paths": [str(path) for path in base_paths],
            "pseudo_paths": [str(path) for path in pseudo_paths],
            "skills_jsonl": str(skills_path),
            "output_path": str(output_path),
            "target_skills": len(targets),
            "modes": modes,
            "base_rows": len(base_rows),
            "pseudo_rows": len(pseudo_rows),
            "total_rows": len(base_rows) + len(pseudo_rows),
            "pseudo_source": ISOLATED_SOURCE,
        },
    )


def command_augment_isolated_modes(args: argparse.Namespace) -> None:
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    if not modes or any(mode not in ISOLATED_TASK_MODES for mode in modes):
        raise SystemExit(f"--modes must be selected from {ISOLATED_TASK_MODES}")
    try:
        write_isolated_augmented_rows(
            args.base_rows,
            args.pseudo_rows,
            args.skills_jsonl,
            args.output,
            args.manifest,
            modes,
            args.max_query_chars,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"saved isolated-mode augmented rows to {args.output}")


def materialize_selected_rows(
    skill: dict[str, Any],
    target: OfficialSingleTarget,
    mode: str,
    candidates: list[CandidateQuery],
    keep: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    skill_text = skill_document_text(skill)
    accepted: list[CandidateQuery] = []
    audit_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        reasons = exact_leak_reasons(candidate.text, skill, target.official_queries)
        features = candidate_features(candidate.text)
        audit_rows.append(
            {
                "skill_id": target.skill_id,
                "mode": mode,
                "query_id": candidate.query_id,
                "query": candidate.text,
                "source_rank": candidate.source_rank,
                "features": features,
                "reject_reasons": reasons,
                "metadata": candidate.metadata,
            }
        )
        if not reasons:
            accepted.append(candidate)
    selected = select_diverse_queries(accepted, skill_text, keep)
    selected_ids = {item.query_id for item in selected}
    for row in audit_rows:
        row["selected"] = row["query_id"] in selected_ids
        row["selection_score"] = next((item.score for item in selected if item.query_id == row["query_id"]), None)
    rows = [
        build_pseudo_row(
            item.text,
            target.skill_id,
            mode,
            rank,
            source_candidate_count=len(candidates),
            official_indices=target.official_indices,
            train_frequency=target.train_frequency,
            score=item.score,
        )
        for rank, item in enumerate(selected)
    ]
    return rows, audit_rows


def make_mixed_rows(generic_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], generic_keep: int, task_keep: int) -> list[dict[str, Any]]:
    grouped_generic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generic_rows:
        grouped_generic[row["selected_skill_ids"][0]].append(row)
    for row in task_rows:
        grouped_task[row["selected_skill_ids"][0]].append(row)
    mixed: list[dict[str, Any]] = []
    for sid in sorted(set(grouped_generic) | set(grouped_task)):
        chosen = [*grouped_generic.get(sid, [])[:generic_keep], *grouped_task.get(sid, [])[:task_keep]]
        for rank, row in enumerate(chosen):
            out = dict(row)
            out["index"] = f"pseudo:mixed:{sid}:{rank}"
            out["pseudo_query"] = dict(row.get("pseudo_query") or {})
            out["pseudo_query"]["mode"] = "mixed"
            out["pseudo_query"]["source_mode"] = (row.get("pseudo_query") or {}).get("mode")
            out["pseudo_query"]["rank"] = rank
            mixed.append(out)
    return mixed


def raw_response_path(output_dir: Path, sid: str, mode: str, call_index: int) -> Path:
    safe_sid = re.sub(r"[^A-Za-z0-9_.-]+", "__", sid)
    return output_dir / "raw_responses" / mode / f"{safe_sid}.{call_index:03d}.json"


def generate_for_target_mode(
    target: OfficialSingleTarget,
    skill: dict[str, Any],
    mode: str,
    profile: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_candidates: list[CandidateQuery] = []
    calls = int(math.ceil(args.candidate_count / args.queries_per_call))
    for call_index in range(calls):
        remaining = args.candidate_count - len(all_candidates)
        count = max(1, min(args.queries_per_call, remaining))
        messages = build_generation_messages(skill, mode, count, args.body_chars)
        response = call_query_generator(profile, messages, args.temperature, args.top_p, args.timeout, args.max_tokens)
        raw_path = raw_response_path(args.output_dir, target.skill_id, mode, call_index)
        write_json(
            raw_path,
            {
                "skill_id": target.skill_id,
                "mode": mode,
                "call_index": call_index,
                "messages": messages,
                "response": response,
            },
        )
        candidates = build_candidates_from_response(
            response,
            mode,
            query_id_prefix=f"{target.skill_id}:{mode}:{call_index}",
            start_rank=len(all_candidates),
            metadata={"raw_response_path": str(raw_path)},
        )
        all_candidates.extend(candidates)
    return materialize_selected_rows(skill, target, mode, all_candidates, args.keep_per_mode)


def command_generate(args: argparse.Namespace) -> None:
    official_rows = load_jsonl(args.official_rows)
    train_rows = load_jsonl(args.train_rows)
    train_freq = collect_train_frequencies(train_rows)
    targets = extract_official_single_targets(official_rows, train_freq, args.max_train_frequency)
    skill_lookup = load_skill_lookup(args.skills_jsonl)
    missing = [target.skill_id for target in targets if target.skill_id not in skill_lookup]
    if missing and args.fail_on_missing_skills:
        raise SystemExit(f"missing skills in --skills-jsonl: {missing[:10]}")
    if missing:
        targets = [target for target in targets if target.skill_id in skill_lookup]
    if args.limit_skills > 0:
        targets = targets[: args.limit_skills]
    profile = load_profile(args.profile)
    configure_profile_transport(profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    for mode in modes:
        if mode not in {"generic", "task_spec"}:
            raise SystemExit(f"unknown mode in --modes: {mode}")

    selected_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    audit_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}

    def run_item(item: tuple[OfficialSingleTarget, str]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        target, mode = item
        rows, audit = generate_for_target_mode(target, skill_lookup[target.skill_id], mode, profile, args)
        return mode, rows, audit

    work = [(target, mode) for target in targets for mode in modes]
    if args.workers == 1:
        results = [run_item(item) for item in work]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(run_item, work))
    for mode, rows, audit in results:
        selected_by_mode[mode].extend(rows)
        audit_by_mode[mode].extend(audit)

    for mode in modes:
        write_jsonl(args.output_dir / f"selected_{mode}{args.keep_per_mode}.jsonl", selected_by_mode[mode])
        write_jsonl(args.output_dir / f"candidates_{mode}.audit.jsonl", audit_by_mode[mode])
    mixed_rows: list[dict[str, Any]] = []
    if "generic" in selected_by_mode and "task_spec" in selected_by_mode and args.write_mixed:
        mixed_rows = make_mixed_rows(
            selected_by_mode["generic"],
            selected_by_mode["task_spec"],
            args.mixed_generic_keep,
            args.mixed_task_keep,
        )
        write_jsonl(args.output_dir / f"selected_mixed{args.mixed_generic_keep + args.mixed_task_keep}.jsonl", mixed_rows)

    short_targets = [
        {
            "skill_id": target.skill_id,
            "official_indices": target.official_indices,
            "train_frequency": target.train_frequency,
        }
        for target in targets
    ]
    write_json(
        args.output_dir / "manifest.json",
        {
            "profile": public_profile(profile),
            "official_rows": str(args.official_rows),
            "train_rows": str(args.train_rows),
            "skills_jsonl": str(args.skills_jsonl),
            "max_train_frequency": args.max_train_frequency,
            "targets": short_targets,
            "target_count": len(targets),
            "missing_target_skills": missing,
            "modes": modes,
            "candidate_count": args.candidate_count,
            "queries_per_call": args.queries_per_call,
            "keep_per_mode": args.keep_per_mode,
            "rows_by_mode": {mode: len(rows) for mode, rows in selected_by_mode.items()},
            "mixed_rows": len(mixed_rows),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "warning": "Transductive diagnostic: target skills are official single GT skills; do not treat as clean benchmark training.",
        },
    )
    print(f"saved pseudo-query outputs to {args.output_dir}")


def command_augment(args: argparse.Namespace) -> None:
    write_augmented_rows(args.base_rows, args.pseudo_rows, args.output, args.manifest)
    print(f"saved augmented rows to {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "generate-catalog",
        help="Clean fixed-budget generation from training catalog skills only.",
    )
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--skill-ids-file", type=Path)
    p.add_argument("--limit-skills", type=int, default=-1)
    p.add_argument("--candidate-count", type=int, default=8)
    p.add_argument("--keep-per-skill", type=int, default=5)
    p.add_argument("--body-chars", type=int, default=2500)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-query-chars", type=int, default=1500)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=command_generate_catalog)

    p = sub.add_parser(
        "generate-isolated-modes",
        help="Generate three clean task modes for skills absent from all multi-skill training rows.",
    )
    p.add_argument("--train-rows", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--modes", default=",".join(ISOLATED_TASK_MODES))
    p.add_argument("--limit-skills", type=int, default=-1)
    p.add_argument("--body-chars", type=int, default=2500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-query-chars", type=int, default=1500)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=command_generate_isolated_modes)

    p = sub.add_parser("generate")
    p.add_argument("--official-rows", type=Path, required=True)
    p.add_argument("--train-rows", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--modes", default="generic,task_spec")
    p.add_argument("--max-train-frequency", type=int, default=1)
    p.add_argument("--candidate-count", type=int, default=24)
    p.add_argument("--queries-per-call", type=int, default=6)
    p.add_argument("--keep-per-mode", type=int, default=8)
    p.add_argument("--limit-skills", type=int, default=-1)
    p.add_argument("--body-chars", type=int, default=2500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=3500)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--fail-on-missing-skills", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--write-mixed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mixed-generic-keep", type=int, default=4)
    p.add_argument("--mixed-task-keep", type=int, default=4)
    p.set_defaults(func=command_generate)

    p = sub.add_parser("augment")
    p.add_argument("--base-rows", type=Path, nargs="+", required=True)
    p.add_argument("--pseudo-rows", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.set_defaults(func=command_augment)

    p = sub.add_parser(
        "augment-catalog",
        help="Validate and append clean full-catalog pseudo rows to decoder training rows.",
    )
    p.add_argument("--base-rows", type=Path, nargs="+", required=True)
    p.add_argument("--pseudo-rows", type=Path, nargs="+", required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--queries-per-skill", type=int, default=5)
    p.add_argument("--max-query-chars", type=int, default=1500)
    p.set_defaults(func=command_augment_catalog)

    p = sub.add_parser(
        "augment-isolated-modes",
        help="Validate and append clean zero-multi-cooccurrence task-mode rows.",
    )
    p.add_argument("--base-rows", type=Path, nargs="+", required=True)
    p.add_argument("--pseudo-rows", type=Path, nargs="+", required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--modes", default=",".join(ISOLATED_TASK_MODES))
    p.add_argument("--max-query-chars", type=int, default=1500)
    p.set_defaults(func=command_augment_isolated_modes)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if hasattr(args, "workers") and args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if hasattr(args, "progress_every") and args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()
