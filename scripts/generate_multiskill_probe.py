#!/usr/bin/env python3
"""Generate a small GPT-based probe set from high-precision skill panels.

This script is intentionally small and API-provider agnostic. It reads an
OpenAI-compatible API key from the environment and never stores secrets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = [
    ("development/devops_deployment/observability_and_monitoring", "observability_stack"),
    ("content_creation/document_authoring/project_and_codebase_documentation", "documentation_site"),
    ("automation/project_management_automation", "issue_or_project"),
    ("automation/browser_web_automation", "web_page"),
    ("automation/productivity_task_management/email_management", "issue_or_project"),
]

DEFAULT_EXTRA_HEADERS: dict[str, str] = {}

NARROW_CONTEXT_TERMS = [
    "bitcoin",
    "gohome",
    "kfg",
    "openshift",
    "specweave",
]

QUERY_REQUIRED_CONTEXT_TERMS = [
    "bitcoin",
    "codex",
    "gohome",
    "jira",
    "kfg",
    "openai",
    "openshift",
    "solidstart",
    "specweave",
    "wheels",
]

NAMED_CONTEXT_TERMS = {
    "airtable",
    "anki",
    "aws",
    "azure",
    "axiom",
    "docker",
    "figma",
    "firebase",
    "github",
    "gitlab",
    "google",
    "grafana",
    "jira",
    "kubernetes",
    "linear",
    "notion",
    "openai",
    "postgres",
    "postgresql",
    "prometheus",
    "redis",
    "salesforce",
    "sendgrid",
    "sentry",
    "slack",
    "stripe",
    "vercel",
}

SELECTED_CONTEXT_REQUIREMENTS = [
    (["data drift", "concept drift", "mlflow", "model performance"], ["model", "ml", "machine learning", "drift", "mlflow"]),
    (["mongodb"], ["mongodb", "mongo", "polyglot"]),
    (["prisma"], ["prisma", "polyglot"]),
    (["pydanticai", "pydantic ai", "pydantic"], ["pydantic", "pydanticai", "agent", "python", "polyglot"]),
    (["salesforce"], ["salesforce"]),
    (["sendgrid"], ["sendgrid"]),
    (["spring boot", "spring"], ["spring"]),
    (["sql database specialist"], ["sql", "relational", "postgres", "postgresql", "mysql", "polyglot"]),
    (["sqlmodel"], ["sqlmodel", "postgres", "postgresql", "sql", "polyglot"]),
    (["stripe"], ["stripe"]),
    (["yarp"], ["yarp", "reverse proxy"]),
]

INCOMPATIBLE_CONTEXT_PAIRS = [
    (["spring"], ["pydantic"], ["polyglot", "multi service", "microservice", "java and python", "python service"]),
    (["mongodb"], ["sqlmodel", "postgresql", "postgres", "sql database"], ["polyglot", "multi database", "multiple database", "hybrid", "migration"]),
]

SIMILARITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "for",
    "from",
    "in",
    "of",
    "or",
    "the",
    "to",
    "use",
    "used",
    "using",
    "when",
    "with",
}


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_skill_lookup(path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sid = str(row.get("skill_id") or row.get("id") or "")
        if sid:
            lookup[sid] = row
    return lookup


def choose_panels(panels: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used = set()
    for task_space, anchor in DEFAULT_TARGETS:
        for panel in panels:
            key = (panel.get("task_space_id"), panel.get("anchor_task_object"))
            if key == (task_space, anchor) and panel.get("viable"):
                chosen.append(panel)
                used.add(id(panel))
                break
        if len(chosen) >= limit:
            return chosen

    remaining = [
        panel
        for panel in panels
        if panel.get("viable") and id(panel) not in used
    ]
    remaining.sort(
        key=lambda p: (
            p.get("metrics", {}).get("operation_role_count", 0),
            p.get("metrics", {}).get("operation_type_count", 0),
            p.get("metrics", {}).get("dominant_task_object_share", 0),
        ),
        reverse=True,
    )
    return (chosen + remaining)[:limit]


def panel_passes_strict_filter(
    panel: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    max_skills: int,
    target_skill_count: int,
    strict_candidate_filter: bool,
) -> bool:
    """High-precision panel gate for harder 4-skill generation."""
    if not panel.get("viable"):
        return False
    metrics = panel.get("metrics") or {}
    if int(metrics.get("operation_role_count") or 0) < target_skill_count:
        return False
    if int(metrics.get("operation_type_count") or 0) < target_skill_count:
        return False
    if float(metrics.get("max_role_share") or 1.0) > 0.35:
        return False
    if float(metrics.get("max_type_share") or 1.0) > 0.35:
        return False

    candidate_skills = choose_candidate_skills(
        panel,
        lookup,
        max_skills,
        target_skill_count,
        strict_candidate_filter,
    )
    if len(candidate_skills) < max(target_skill_count + 4, 8):
        return False
    roles = {
        str(skill.get("operation_role") or "")
        for skill in candidate_skills
        if skill.get("operation_role")
    }
    types = {
        str(skill.get("operation_type") or "")
        for skill in candidate_skills
        if skill.get("operation_type")
    }
    return len(roles) >= target_skill_count and len(types) >= target_skill_count


def compact_skill(panel_skill: dict[str, Any], lookup: dict[str, dict[str, Any]], desc_chars: int) -> dict[str, Any]:
    sid = str(panel_skill.get("skill_id") or "")
    full = lookup.get(sid, {})
    desc = str(full.get("description") or full.get("task_description") or "")[:desc_chars]
    return {
        "skill_id": sid,
        "name": panel_skill.get("skill_name"),
        "description": desc,
        "operation": f"{panel_skill.get('operation_role')}/{panel_skill.get('operation_type')}",
        "task_objects": panel_skill.get("task_object_types") or [],
        "object_signal_source": panel_skill.get("task_object_signal_source"),
        "input_artifacts": panel_skill.get("input_artifact_types") or [],
        "output_artifacts": panel_skill.get("output_artifact_types") or [],
        "modalities": panel_skill.get("modality_types") or [],
    }


def skill_text_for_filter(panel_skill: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> str:
    sid = str(panel_skill.get("skill_id") or "")
    full = lookup.get(sid, {})
    return normalize(" ".join([
        sid,
        str(panel_skill.get("skill_name") or full.get("skill_name") or full.get("name") or ""),
        str(full.get("description") or full.get("task_description") or ""),
        str(panel_skill.get("operation_role") or ""),
        str(panel_skill.get("operation_type") or ""),
        " ".join(panel_skill.get("task_object_types") or []),
    ]))


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize(text).split()
        if len(token) >= 4 and token not in SIMILARITY_STOPWORDS
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_coverage(claim: set[str], evidence: set[str]) -> float:
    if not claim:
        return 1.0
    return len(claim & evidence) / len(claim)


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def selected_context_warnings(selected_text: str, query_text: str) -> list[str]:
    warnings: list[str] = []
    for selected_terms, query_terms in SELECTED_CONTEXT_REQUIREMENTS:
        if contains_any(selected_text, selected_terms) and not contains_any(query_text, query_terms):
            warnings.append("selected_technology_requires_query_context")
            break
    for left_terms, right_terms, query_terms in INCOMPATIBLE_CONTEXT_PAIRS:
        if (
            contains_any(selected_text, left_terms)
            and contains_any(selected_text, right_terms)
            and not contains_any(query_text, query_terms)
        ):
            warnings.append("mixed_technology_stack_without_query_context")
            break
    return warnings


def choose_candidate_skills(
    panel: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    max_skills: int,
    target_skill_count: int,
    strict_candidate_filter: bool,
) -> list[dict[str, Any]]:
    raw_skills = list(panel.get("skills") or [])
    if not strict_candidate_filter:
        return raw_skills[:max_skills]

    panel_context = normalize(f"{panel.get('task_space_id')} {panel.get('anchor_task_object')}")
    kept: list[dict[str, Any]] = []
    kept_tokens: list[set[str]] = []
    for skill in raw_skills:
        text = skill_text_for_filter(skill, lookup)
        if any(term in text and term not in panel_context for term in NARROW_CONTEXT_TERMS):
            continue
        tokens = content_tokens(text)
        if any(jaccard(tokens, existing) >= 0.30 for existing in kept_tokens):
            continue
        kept.append(skill)
        kept_tokens.append(tokens)
        if len(kept) >= max_skills:
            break

    # If filtering is too aggressive, keep the original panel rather than
    # starving the generator. Quality is better handled by resampling.
    if len(kept) < max(target_skill_count + 2, 6):
        return raw_skills[:max_skills]
    return kept


def build_messages(
    panel: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    max_skills: int,
    desc_chars: int,
    target_skill_count: int,
    query_style: str,
    candidate_index: int,
    best_of_n: int,
    strict_candidate_filter: bool,
) -> list[dict[str, str]]:
    candidate_panel_skills = choose_candidate_skills(
        panel,
        lookup,
        max_skills,
        target_skill_count,
        strict_candidate_filter,
    )
    skills = [
        compact_skill(skill, lookup, desc_chars)
        for skill in candidate_panel_skills
    ]
    candidate_specs = []
    for number, skill in enumerate(skills, 1):
        skill_name = str(skill.get("name") or skill.get("skill_id") or "")
        category = str(panel.get("task_space_id") or skill.get("operation") or "uncategorized")
        body_preview = "; ".join([
            f"Operation: {skill.get('operation')}",
            f"Task objects: {', '.join(skill.get('task_objects') or [])}",
            f"Inputs: {', '.join(skill.get('input_artifacts') or [])}",
            f"Outputs: {', '.join(skill.get('output_artifacts') or [])}",
            f"Modalities: {', '.join(skill.get('modalities') or [])}",
        ])
        candidate_specs.append(
            f"{number}. Skill id: {skill.get('skill_id')}\n"
            f"   Skill name: {skill_name}\n"
            f"   Category: {category}\n"
            f"   Description: {skill.get('description')}\n"
            f"   Skill body: {body_preview}"
        )
    system = (
        "You are an experienced user of AI assistants. You write clear, "
        "realistic task requests that describe what you need to accomplish."
    )
    user = (
        "Given the following candidate skill specifications, first choose exactly "
        f"{target_skill_count} skills that would be naturally useful together for one realistic task. "
        "Then write a realistic task description that someone would ask an AI assistant to help with. "
        "The task should naturally require the capabilities described in all selected skills.\n\n"
        f"Candidate skills:\n{chr(10).join(candidate_specs)}\n\n"
        "Requirements:\n"
        "(1) Describe a concrete scenario with specific inputs/outputs.\n"
        "(2) Include enough detail that every selected skill would be clearly useful.\n"
        f"(3) Select exactly {target_skill_count} skills, no fewer and no more.\n"
        "(4) The selected skills should have complementary, non-overlapping roles in one plausible workflow.\n"
        "(5) Do NOT mention any selected skill name anywhere in the task description.\n"
        "(6) In the selected skill ids line, copy the exact full Skill id strings from the candidate list. "
        "Do not use candidate numbers, skill names, shortened ids, or ids without their namespace prefix.\n\n"
        "Output ONLY the following two lines:\n"
        "Selected skill ids: <comma-separated selected skill ids>\n"
        "Task description: <task description>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    timeout: int,
    use_schema: bool,
    target_skill_count: int,
    max_tokens: int,
) -> dict[str, Any]:
    api_base = api_base.rstrip("/")
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens > 0:
        body["max_tokens"] = max_tokens
    if top_p > 0:
        body["top_p"] = top_p

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "SkillSearch-workflow-first/0.1",
    }
    headers.update(DEFAULT_EXTRA_HEADERS)

    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    retryable_http_codes = {403, 408, 409, 425, 429, 500, 502, 503, 504}
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in retryable_http_codes or attempt == 3:
                raise
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt == 3:
                raise
        time.sleep(min(12.0, 1.5 * (2 ** attempt)))
    assert last_exc is not None
    raise last_exc


def parse_text_selection_payload(content: str, target_skill_count: int) -> dict[str, Any]:
    text = str(content or "").strip()
    selected_line = ""
    query_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("selected skill ids:") or lowered.startswith("selected_skill_ids:"):
            selected_line = stripped.split(":", 1)[1].strip()
        elif lowered.startswith("task description:") or lowered.startswith("query:"):
            query_line = stripped.split(":", 1)[1].strip()

    if not selected_line:
        bracketed = re.search(r"\[([^\]]+)\]", text)
        if bracketed:
            selected_line = bracketed.group(1)
    selected_ids = [
        item.strip().strip("\"'")
        for item in re.split(r"[,;\n]+", selected_line)
        if item.strip().strip("\"'")
    ]
    if not query_line:
        non_id_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().lower().startswith(("selected skill ids:", "selected_skill_ids:"))
        ]
        query_line = " ".join(non_id_lines).strip()
    return {
        "query": query_line,
        "target_skill_count": target_skill_count,
        "selected_skill_ids": selected_ids,
        "selected_skills": [{"skill_id": sid} for sid in selected_ids],
    }


def parse_content(response: dict[str, Any], target_skill_count: int) -> dict[str, Any]:
    content = response["choices"][0]["message"]["content"]
    return parse_text_selection_payload(content, target_skill_count)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def generated_ids(generated: dict[str, Any]) -> list[str]:
    return [str(sid) for sid in (generated.get("selected_skill_ids") or [])]


def skill_lookup_from_panel(
    panel: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    max_skills: int,
    target_skill_count: int,
    strict_candidate_filter: bool,
) -> dict[str, dict[str, Any]]:
    return {
        str(skill.get("skill_id")): skill
        for skill in choose_candidate_skills(
            panel,
            lookup,
            max_skills,
            target_skill_count,
            strict_candidate_filter,
        )
        if skill.get("skill_id")
    }


def overlap_level(generated: dict[str, Any]) -> str:
    risk = generated.get("overlap_risk") or {}
    if isinstance(risk, dict):
        return normalize(str(risk.get("level") or ""))
    return normalize(str(risk))


def score_generated(
    generated: dict[str, Any],
    panel: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    max_skills: int,
    target_skill_count: int,
    strict_candidate_filter: bool,
) -> dict[str, Any]:
    by_id = skill_lookup_from_panel(
        panel,
        lookup,
        max_skills,
        target_skill_count,
        strict_candidate_filter,
    )
    candidate_ids = set(by_id)
    selected = generated_ids(generated)
    selected_set = set(selected)
    selected_skills = generated.get("selected_skills") or []
    query = str(generated.get("query") or "")
    qnorm = normalize(query)

    hard_issues: list[str] = []
    quality_warnings: list[str] = []

    if int(generated.get("target_skill_count") or 0) != target_skill_count:
        hard_issues.append("wrong_target_skill_count")
    if len(selected) != target_skill_count:
        hard_issues.append("wrong_selected_count")
    if len(selected_set) != len(selected):
        hard_issues.append("duplicate_selected_ids")
    if any(sid not in candidate_ids for sid in selected):
        hard_issues.append("selected_id_not_in_panel")
    if selected_skills and len(selected_skills) != len(selected):
        hard_issues.append("selected_skills_count_mismatch")
    if selected_skills and [str(skill.get("skill_id")) for skill in selected_skills] != selected:
        quality_warnings.append("selected_order_or_id_mismatch")
    if word_count(query) < 18:
        quality_warnings.append("query_too_short")
    if word_count(query) > 160:
        quality_warnings.append("query_too_long")
    if any(term in qnorm for term in ["candidate panel", "skill id", "selected skill", "operation label"]):
        hard_issues.append("query_mentions_generation_artifact")
    if any(sid.lower() in query.lower() for sid in selected):
        hard_issues.append("query_mentions_skill_id")

    roles = [str(by_id.get(sid, {}).get("operation_role") or "") for sid in selected if sid in by_id]
    types = [str(by_id.get(sid, {}).get("operation_type") or "") for sid in selected if sid in by_id]
    if roles and len(set(roles)) < min(target_skill_count, 2):
        quality_warnings.append("low_operation_role_diversity")
    if types and len(set(types)) < min(target_skill_count, len(types)):
        quality_warnings.append("duplicate_operation_type")
    if target_skill_count >= 4:
        validation_types = [
            str(by_id.get(sid, {}).get("operation_type") or "")
            for sid in selected
            if str(by_id.get(sid, {}).get("operation_role") or "") == "validate"
        ]
        if len(validation_types) > 1 and "validation_check" in validation_types:
            quality_warnings.append("generic_validation_overlap_in_4skill")

    selected_full_texts = [
        skill_text_for_filter(by_id.get(sid, {}), lookup)
        for sid in selected
        if sid in by_id
    ]
    selected_full_text = " ".join(selected_full_texts)
    query_terms = set(qnorm.split())
    unsupported_named_terms = [
        term
        for term in sorted(NAMED_CONTEXT_TERMS & query_terms)
        if term not in selected_full_text
    ]
    if unsupported_named_terms:
        quality_warnings.append("unsupported_named_context_in_query")
    selected_token_sets = [content_tokens(text) for text in selected_full_texts]
    for i, tokens_a in enumerate(selected_token_sets):
        for tokens_b in selected_token_sets[i + 1:]:
            if jaccard(tokens_a, tokens_b) >= 0.30:
                quality_warnings.append("near_duplicate_selected_skills")
                break
        if "near_duplicate_selected_skills" in quality_warnings:
            break

    for sid in selected:
        skill = by_id.get(sid, {})
        skill_text = normalize(" ".join([
            str(skill.get("skill_id") or ""),
            str(skill.get("skill_name") or ""),
            " ".join(skill.get("task_object_types") or []),
            " ".join(skill.get("input_artifact_types") or []),
            " ".join(skill.get("output_artifact_types") or []),
        ]))
        if any(term in skill_text and term not in qnorm for term in NARROW_CONTEXT_TERMS):
            quality_warnings.append("narrow_context_skill_without_query_context")
            break
    if any(term in selected_full_text and term not in qnorm for term in QUERY_REQUIRED_CONTEXT_TERMS):
        quality_warnings.append("selected_context_requires_query_context")
    quality_warnings.extend(selected_context_warnings(selected_full_text, qnorm))

    for selected_skill in selected_skills:
        if not isinstance(selected_skill, dict):
            continue
        sid = str(selected_skill.get("skill_id") or "")
        if sid not in by_id:
            continue
        claim_tokens = content_tokens(" ".join([
            str(selected_skill.get("role_in_workflow") or ""),
            str(selected_skill.get("unique_contribution") or ""),
            str(selected_skill.get("evidence_from_skill_description") or ""),
        ]))
        evidence_tokens = content_tokens(skill_text_for_filter(by_id[sid], lookup))
        if len(claim_tokens) >= 5 and token_coverage(claim_tokens, evidence_tokens) < 0.12:
            quality_warnings.append("unsupported_skill_claim_by_evidence")
            break

    graph = generated.get("latent_workflow_graph") or {}
    node_ids = [str(node.get("skill_id")) for node in (graph.get("nodes") or []) if isinstance(node, dict)]
    edge_ids: set[str] = set()
    invalid_edges = 0
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            invalid_edges += 1
            continue
        src = str(edge.get("from_skill_id") or "")
        dst = str(edge.get("to_skill_id") or "")
        edge_ids.update([src, dst])
        if src not in selected_set or dst not in selected_set or src == dst:
            invalid_edges += 1
    if graph:
        if set(node_ids) != selected_set:
            hard_issues.append("workflow_graph_nodes_do_not_match_selection")
        if invalid_edges:
            hard_issues.append("workflow_graph_has_invalid_edges")
        if len(graph.get("edges") or []) < max(1, target_skill_count - 1):
            quality_warnings.append("workflow_graph_too_sparse")
        if selected_set and not selected_set.issubset(set(node_ids) | edge_ids):
            quality_warnings.append("workflow_graph_does_not_cover_selection")

    necessity_ids = [
        str(item.get("skill_id"))
        for item in (generated.get("skill_necessity_checks") or [])
        if isinstance(item, dict)
    ]
    if generated.get("skill_necessity_checks") and set(necessity_ids) != selected_set:
        hard_issues.append("necessity_checks_do_not_match_selection")
    for item in generated.get("skill_necessity_checks") or []:
        if not isinstance(item, dict):
            hard_issues.append("malformed_necessity_check")
            continue
        if word_count(str(item.get("what_breaks_if_removed") or "")) < 5:
            quality_warnings.append("thin_leave_one_out_check")
            break

    risk_level = overlap_level(generated)
    if "high" in risk_level:
        hard_issues.append("high_overlap_risk")
    elif "medium" in risk_level:
        quality_warnings.append("medium_overlap_risk")
    elif risk_level and "low" not in risk_level:
        quality_warnings.append("unclear_overlap_risk")

    weakness = normalize(str(generated.get("potential_weakness") or ""))
    for term in ["unsupported", "unclear", "ambiguous", "overlap", "not fully compatible"]:
        if term in weakness:
            quality_warnings.append("self_reported_quality_risk")
            break

    if "evidence_consistency_check" in generated and word_count(str(generated.get("evidence_consistency_check") or "")) < 6:
        quality_warnings.append("thin_evidence_consistency_check")

    if hard_issues:
        score = -1000 - 50 * len(hard_issues) - 5 * len(quality_warnings)
    else:
        score = 100
        score += 4 * min(len(set(roles)), target_skill_count)
        score += 5 * min(len(set(types)), target_skill_count)
        if target_skill_count >= 4 and roles and len(set(roles)) == target_skill_count:
            score += 10
        score += 8 if "low" in risk_level else 0
        score += 6 if 35 <= word_count(query) <= 120 else 0
        score += 5 if len(graph.get("edges") or []) >= target_skill_count - 1 else 0
        score -= 12 * len(quality_warnings)

    return {
        "score": score,
        "hard_issues": hard_issues,
        "quality_warnings": sorted(set(quality_warnings)),
        "selected_skill_ids": selected,
        "role_count": len(set(roles)) if roles else 0,
        "operation_type_count": len(set(types)) if types else 0,
        "query_word_count": word_count(query),
        "overlap_risk_level": risk_level,
    }


def generate_candidate(
    *,
    panel_index: int,
    candidate_index: int,
    panel: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    messages = build_messages(
        panel,
        lookup,
        args.max_skills,
        args.desc_chars,
        args.target_skill_count,
        args.query_style,
        candidate_index,
        args.best_of_n,
        args.strict_candidate_filter,
    )
    try:
        try:
            response = call_chat_completion(
                args.api_base,
                api_key,
                args.model,
                messages,
                args.temperature,
                args.top_p,
                args.timeout,
                use_schema=not args.no_schema,
                target_skill_count=args.target_skill_count,
                max_tokens=args.max_tokens,
            )
        except urllib.error.HTTPError as exc:
            if not args.no_schema:
                response = call_chat_completion(
                    args.api_base,
                    api_key,
                    args.model,
                    messages,
                    args.temperature,
                    args.top_p,
                    args.timeout,
                    use_schema=False,
                    target_skill_count=args.target_skill_count,
                    max_tokens=args.max_tokens,
                )
            else:
                raise exc
        generated = parse_content(response, args.target_skill_count)
        return {
            "panel_index": panel_index,
            "candidate_index": candidate_index,
            "generated": generated,
            "evaluation": score_generated(
                generated,
                panel,
                lookup,
                args.max_skills,
                args.target_skill_count,
                args.strict_candidate_filter,
            ),
            "usage": response.get("usage"),
            "model": response.get("model", args.model),
        }
    except Exception as exc:
        return {
            "panel_index": panel_index,
            "candidate_index": candidate_index,
            "error": f"{type(exc).__name__}: {exc}",
            "evaluation": {
                "score": -2000,
                "hard_issues": ["generation_error"],
                "quality_warnings": [],
            },
        }


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    generated = candidate.get("generated") or {}
    return {
        "candidate_index": candidate.get("candidate_index"),
        "score": candidate.get("evaluation", {}).get("score"),
        "hard_issues": candidate.get("evaluation", {}).get("hard_issues"),
        "quality_warnings": candidate.get("evaluation", {}).get("quality_warnings"),
        "selected_skill_ids": generated.get("selected_skill_ids"),
        "query": generated.get("query"),
        "usage": candidate.get("usage"),
        "error": candidate.get("error"),
    }


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[bool, bool, int, int]:
    evaluation = candidate.get("evaluation", {})
    return (
        not bool(evaluation.get("hard_issues")),
        not bool(evaluation.get("quality_warnings")),
        int(evaluation.get("score", -9999)),
        -int(candidate.get("candidate_index") or 0),
    )


def row_candidate_skill_ids(
    panel: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(skill.get("skill_id"))
        for skill in choose_candidate_skills(
            panel,
            lookup,
            args.max_skills,
            args.target_skill_count,
            args.strict_candidate_filter,
        )
        if skill.get("skill_id")
    ]


def _candidate_process_worker(queue: mp.Queue, kwargs: dict[str, Any]) -> None:
    try:
        queue.put(generate_candidate(**kwargs))
    except BaseException as exc:
        queue.put({
            "panel_index": kwargs.get("panel_index"),
            "candidate_index": kwargs.get("candidate_index"),
            "error": f"{type(exc).__name__}: {exc}",
            "evaluation": {
                "score": -2000,
                "hard_issues": ["generation_error"],
                "quality_warnings": [],
            },
        })


def generate_candidate_with_process_timeout(kwargs: dict[str, Any], timeout: int) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_candidate_process_worker, args=(queue, kwargs))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        return {
            "panel_index": kwargs.get("panel_index"),
            "candidate_index": kwargs.get("candidate_index"),
            "error": f"TimeoutError: hard API-call timeout after {timeout}s",
            "evaluation": {
                "score": -2000,
                "hard_issues": ["generation_error"],
                "quality_warnings": [],
            },
        }
    if not queue.empty():
        return queue.get()
    return {
        "panel_index": kwargs.get("panel_index"),
        "candidate_index": kwargs.get("candidate_index"),
        "error": f"ProcessError: worker exited with code {process.exitcode} without a result",
        "evaluation": {
            "score": -2000,
            "hard_issues": ["generation_error"],
            "quality_warnings": [],
        },
    }


def candidate_kwargs_for(
    index: int,
    candidate_index: int,
    panel: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    return {
        "panel_index": index,
        "candidate_index": candidate_index,
        "panel": panel,
        "lookup": lookup,
        "args": args,
        "api_key": api_key,
    }


def run_candidate(candidate_kwargs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.hard_timeout > 0:
        return generate_candidate_with_process_timeout(candidate_kwargs, args.hard_timeout)
    return generate_candidate(**candidate_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", required=True, type=Path)
    parser.add_argument("--skills", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("CHATANYWHERE_MODEL", "gpt-4o"))
    parser.add_argument("--api-base", default=os.environ.get("CHATANYWHERE_API_BASE", "https://api.chatanywhere.tech/v1/"))
    parser.add_argument("--api-key-env", default="CHATANYWHERE_API_KEY")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-skills", type=int, default=12)
    parser.add_argument("--desc-chars", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.0, help="set >0 to pass top_p to the chat completion API")
    parser.add_argument("--target-skill-count", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--query-style", choices=["concise", "scenario", "indirect"], default="scenario")
    parser.add_argument("--best-of-n", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--strict-candidate-filter", action="store_true")
    parser.add_argument("--strict-panel-filter", action="store_true")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--hard-timeout", type=int, default=0, help="main-thread hard timeout per candidate when --workers=1")
    parser.add_argument("--max-tokens", type=int, default=1800, help="maximum completion tokens per API call; set <=0 to omit")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--no-schema", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--drop-non-ok", action="store_true", help="skip rows whose best candidate has hard issues or quality warnings")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")

    lookup = load_skill_lookup(args.skills)
    all_panels = load_jsonl(args.panels)
    if args.strict_panel_filter:
        all_panels = [
            panel
            for panel in all_panels
            if panel_passes_strict_filter(
                panel,
                lookup,
                args.max_skills,
                args.target_skill_count,
                args.strict_candidate_filter,
            )
        ]
        if not all_panels:
            raise SystemExit("--strict-panel-filter removed all panels")
    panels = choose_panels(all_panels, args.limit)
    outputs: list[dict[str, Any]] = []
    if args.append and args.output.exists():
        args.output.unlink()
    if args.best_of_n < 1:
        raise SystemExit("--best-of-n must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    for index, panel in enumerate(panels, 1):
        candidates: list[dict[str, Any]] = []
        if args.workers == 1:
            for candidate_index in range(1, args.best_of_n + 1):
                candidate_kwargs = candidate_kwargs_for(index, candidate_index, panel, lookup, args, api_key)
                candidates.append(run_candidate(candidate_kwargs, args))
                if args.sleep:
                    time.sleep(args.sleep)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        run_candidate,
                        candidate_kwargs_for(index, candidate_index, panel, lookup, args, api_key),
                        args,
                    )
                    for candidate_index in range(1, args.best_of_n + 1)
                ]
                for future in concurrent.futures.as_completed(futures):
                    candidates.append(future.result())
                    if args.sleep:
                        time.sleep(args.sleep)

        candidates.sort(key=candidate_sort_key, reverse=True)
        best = candidates[0]
        if best.get("error") and not args.continue_on_error:
            raise RuntimeError(best["error"])
        best_eval = best.get("evaluation", {})
        if args.drop_non_ok and (
            best_eval.get("hard_issues") or best_eval.get("quality_warnings")
        ):
            print(json.dumps({
                "index": index,
                "task_space_id": panel.get("task_space_id"),
                "skipped": True,
                "reason": "best_candidate_not_ok",
                "score": best_eval.get("score"),
                "hard_issues": best_eval.get("hard_issues"),
                "quality_warnings": best_eval.get("quality_warnings"),
            }, ensure_ascii=False))
            continue

        row = {
            "index": index,
            "task_space_id": panel.get("task_space_id"),
            "anchor_task_object": panel.get("anchor_task_object"),
            "panel_metrics": panel.get("metrics"),
            "candidate_skill_ids": row_candidate_skill_ids(panel, lookup, args),
            "generated": best.get("generated"),
            "usage": best.get("usage"),
            "model": best.get("model", args.model),
            "target_skill_count": args.target_skill_count,
            "query_style": args.query_style,
            "best_of_n": args.best_of_n,
            "chosen_candidate_index": best.get("candidate_index"),
            "chosen_evaluation": best.get("evaluation"),
            "candidate_summaries": [summarize_candidate(candidate) for candidate in candidates],
        }
        if best.get("error"):
            row["error"] = best["error"]
        if args.keep_candidates:
            row["candidates"] = candidates

        outputs.append(row)
        if args.append:
            append_jsonl(args.output, row)
        print(json.dumps({
            "index": index,
            "task_space_id": panel.get("task_space_id"),
            "selected_skill_ids": (best.get("generated") or {}).get("selected_skill_ids"),
            "score": best.get("evaluation", {}).get("score"),
            "hard_issues": best.get("evaluation", {}).get("hard_issues"),
            "quality_warnings": best.get("evaluation", {}).get("quality_warnings"),
            "chosen_candidate_index": best.get("candidate_index"),
        }, ensure_ascii=False))

    if not args.append:
        write_jsonl(args.output, outputs)


if __name__ == "__main__":
    main()
