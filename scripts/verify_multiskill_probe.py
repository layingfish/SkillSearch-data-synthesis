#!/usr/bin/env python3
"""LLM verifier for generated multi-skill training samples.

The verifier is deliberately weaker than ToolLLM's real API execution. Skills
are often guides or references rather than callable APIs, so we ask the model to
perform a grounded pseudo-execution audit instead.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


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


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_skill_lookup(path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sid = str(row.get("skill_id") or row.get("id") or "")
        if sid:
            lookup[sid] = row
    return lookup


def load_panel_lookup(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("task_space_id")), str(row.get("anchor_task_object"))): row
        for row in load_jsonl(path)
    }


def compact_skill(
    skill_id: str,
    panel_skill: dict[str, Any] | None,
    full_skill: dict[str, Any] | None,
    desc_chars: int,
) -> dict[str, Any]:
    panel_skill = panel_skill or {}
    full_skill = full_skill or {}
    desc = str(full_skill.get("description") or full_skill.get("task_description") or panel_skill.get("description") or "")
    return {
        "skill_id": skill_id,
        "name": panel_skill.get("skill_name") or full_skill.get("skill_name") or full_skill.get("name"),
        "description": desc[:desc_chars],
        "operation_role": panel_skill.get("operation_role") or full_skill.get("operation_role"),
        "operation_type": panel_skill.get("operation_type") or full_skill.get("operation_type"),
        "skill_kind": panel_skill.get("skill_kind") or full_skill.get("skill_kind"),
        "scope_level": panel_skill.get("scope_level") or full_skill.get("scope_level"),
        "task_objects": panel_skill.get("task_object_types") or full_skill.get("task_object_types") or [],
        "input_artifacts": panel_skill.get("input_artifact_types") or full_skill.get("input_artifact_types") or [],
        "output_artifacts": panel_skill.get("output_artifact_types") or full_skill.get("output_artifact_types") or [],
    }


def make_verifier_schema(target_skill_count: int, max_distractors: int) -> dict[str, Any]:
    return {
        "name": "multiskill_semantic_verification",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verdict": {"type": "string", "enum": ["accept", "revise", "reject"]},
                "final_score": {"type": "number"},
                "sufficiency": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rating": {"type": "string", "enum": ["pass", "weak", "fail"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["rating", "reason"],
                },
                "minimality": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rating": {"type": "string", "enum": ["pass", "weak", "fail"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["rating", "reason"],
                },
                "query_naturalness": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rating": {"type": "string", "enum": ["natural", "too_tool_specific", "too_generic", "unnatural"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["rating", "reason"],
                },
                "requirement_coverage": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "requirement": {"type": "string"},
                            "source": {"type": "string", "enum": ["query", "desired_output", "input_constraint"]},
                            "covered_by_skill_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 0,
                                "maxItems": target_skill_count,
                            },
                            "coverage": {"type": "string", "enum": ["direct", "indirect", "missing"]},
                            "evidence": {"type": "string"},
                            "gap": {"type": "string"},
                        },
                        "required": [
                            "requirement",
                            "source",
                            "covered_by_skill_ids",
                            "coverage",
                            "evidence",
                            "gap",
                        ],
                    },
                    "minItems": 1,
                    "maxItems": 10,
                },
                "pseudo_execution_trace": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "step_index": {"type": "integer"},
                            "skill_id": {"type": "string"},
                            "expected_input": {"type": "string"},
                            "expected_output": {"type": "string"},
                            "success_condition": {"type": "string"},
                        },
                        "required": [
                            "step_index",
                            "skill_id",
                            "expected_input",
                            "expected_output",
                            "success_condition",
                        ],
                    },
                    "minItems": target_skill_count,
                    "maxItems": target_skill_count,
                },
                "selected_skill_checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "skill_id": {"type": "string"},
                            "necessity": {"type": "string", "enum": ["required", "useful", "redundant", "unsupported"]},
                            "evidence_grounding": {"type": "string", "enum": ["pass", "weak", "fail"]},
                            "unique_contribution": {"type": "string", "enum": ["pass", "weak", "fail"]},
                            "remove_effect": {"type": "string"},
                            "concern": {"type": "string"},
                        },
                        "required": [
                            "skill_id",
                            "necessity",
                            "evidence_grounding",
                            "unique_contribution",
                            "remove_effect",
                            "concern",
                        ],
                    },
                    "minItems": target_skill_count,
                    "maxItems": target_skill_count,
                },
                "overlap_checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "skill_a": {"type": "string"},
                            "skill_b": {"type": "string"},
                            "level": {"type": "string", "enum": ["none", "low", "medium", "high"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["skill_a", "skill_b", "level", "reason"],
                    },
                    "minItems": 0,
                    "maxItems": 10,
                },
                "unsupported_claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "skill_id": {"type": "string"},
                            "claim": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["skill_id", "claim", "reason"],
                    },
                    "minItems": 0,
                    "maxItems": 8,
                },
                "better_replacements_from_panel": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "replace_skill_id": {"type": "string"},
                            "with_skill_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["replace_skill_id", "with_skill_id", "reason"],
                    },
                    "minItems": 0,
                    "maxItems": min(max_distractors, 5),
                },
                "diversity_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                "decision_reason": {"type": "string"},
                "recommended_action": {"type": "string", "enum": ["keep", "resample", "revise_prompt"]},
            },
            "required": [
                "verdict",
                "final_score",
                "sufficiency",
                "minimality",
                "query_naturalness",
                "requirement_coverage",
                "pseudo_execution_trace",
                "selected_skill_checks",
                "overlap_checks",
                "unsupported_claims",
                "better_replacements_from_panel",
                "diversity_risk",
                "decision_reason",
                "recommended_action",
            ],
        },
        "strict": True,
    }


def choose_distractors(
    row: dict[str, Any],
    selected_ids: set[str],
    panel_skills: list[dict[str, Any]],
    max_distractors: int,
) -> list[str]:
    ordered = [str(sid) for sid in (row.get("candidate_skill_ids") or []) if sid]
    if not ordered:
        ordered = [str(skill.get("skill_id")) for skill in panel_skills if skill.get("skill_id")]
    out: list[str] = []
    for sid in ordered:
        if sid and sid not in selected_ids and sid not in out:
            out.append(sid)
        if len(out) >= max_distractors:
            break
    return out


def build_payload(
    row: dict[str, Any],
    panels: dict[tuple[str, str], dict[str, Any]],
    skills: dict[str, dict[str, Any]],
    desc_chars: int,
    max_distractors: int,
) -> dict[str, Any]:
    generated = row.get("generated") or {}
    selected_ids = [str(sid) for sid in (generated.get("selected_skill_ids") or [])]
    selected_set = set(selected_ids)
    panel = panels.get((str(row.get("task_space_id")), str(row.get("anchor_task_object"))), {})
    panel_skills = panel.get("skills") or []
    panel_by_id = {
        str(skill.get("skill_id")): skill
        for skill in panel_skills
        if skill.get("skill_id")
    }
    distractor_ids = choose_distractors(row, selected_set, panel_skills, max_distractors)
    generated_selected = {
        str(item.get("skill_id")): item
        for item in (generated.get("selected_skills") or [])
        if isinstance(item, dict)
    }
    return {
        "task_space_id": row.get("task_space_id"),
        "anchor_task_object": row.get("anchor_task_object"),
        "target_skill_count": row.get("target_skill_count") or generated.get("target_skill_count"),
        "query": generated.get("query"),
        "input_artifacts": generated.get("input_artifacts"),
        "desired_outputs": generated.get("desired_outputs"),
        "selected_skills": [
            {
                **compact_skill(sid, panel_by_id.get(sid), skills.get(sid), desc_chars),
                "generator_claim": generated_selected.get(sid, {}),
            }
            for sid in selected_ids
        ],
        "distractor_skills_from_same_panel": [
            compact_skill(sid, panel_by_id.get(sid), skills.get(sid), desc_chars)
            for sid in distractor_ids
        ],
        "generator_workflow": {
            "latent_workflow_graph": generated.get("latent_workflow_graph"),
            "workflow_steps": generated.get("workflow_steps"),
            "skill_necessity_checks": generated.get("skill_necessity_checks"),
            "why_collaborative": generated.get("why_collaborative"),
            "non_overlap_check": generated.get("non_overlap_check"),
            "overlap_risk": generated.get("overlap_risk"),
            "evidence_consistency_check": generated.get("evidence_consistency_check"),
            "potential_weakness": generated.get("potential_weakness"),
        },
    }


def build_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    target = int(payload.get("target_skill_count") or len(payload.get("selected_skills") or []))
    system = f"""You are a semantic verifier for multi-skill retrieval training data.

The generator has written one user query and selected exactly {target} skills from a redundant panel. Your job is to audit whether this is a high-quality collaborative training example.

You are NOT required to prove the selected skills are the only possible answer. Accept the sample if the selected skills are a grounded, minimal, sufficient, non-overlapping solution.

Use a loose ToolLLM-style validation: do not call real APIs, but write a pseudo-execution trace showing how each selected skill would consume an input artifact, produce an output artifact, and pass a success condition.

Before giving the final verdict, extract the explicit requirements from the query and desired outputs. For each requirement, decide whether it has direct, indirect, or missing coverage.

Direct coverage requires explicit support in the selected skill's description, operation type/role, or output artifacts. Do not let a broad artifact such as "documentation" satisfy a specific requirement such as "validation results", "consistency report", "MkDocs site configuration", "structured logging standardization", or "Prometheus monitoring setup". Specific requirements need specific evidence.

If any explicit desired output has missing coverage, the sample is not sufficient. If coverage is only indirect, mark sufficiency or minimality as weak unless another selected skill directly covers it.

Reject or ask for resampling when:
1. The selected skills cannot together satisfy the query.
2. Any selected skill is unsupported by its description.
3. A selected skill is merely useful but not necessary.
4. Two selected skills substantially overlap.
5. A distractor in the same panel is clearly a better replacement.

Be strict about evidence, but do not penalize natural mentions of real products or artifacts such as Prometheus, Axiom logs, MkDocs, GitHub, Jira, or Sentry.
"""
    user = (
        "Verify this generated multi-skill training sample. Return only valid JSON matching the schema.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: int,
    use_schema: bool,
    target_skill_count: int,
    max_distractors: int,
) -> dict[str, Any]:
    api_base = api_base.rstrip("/")
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if use_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": make_verifier_schema(target_skill_count, max_distractors),
        }
    else:
        body["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_content(response: dict[str, Any]) -> dict[str, Any]:
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)


def normalize_verdict(verifier: dict[str, Any]) -> dict[str, Any]:
    verdict = str(verifier.get("verdict") or "").lower()
    score = float(verifier.get("final_score") or 0)
    if 0 < score <= 1:
        score *= 100
    elif 1 < score <= 10:
        score *= 10
    sufficiency = normalize(str((verifier.get("sufficiency") or {}).get("rating") or ""))
    minimality = normalize(str((verifier.get("minimality") or {}).get("rating") or ""))
    naturalness = normalize(str((verifier.get("query_naturalness") or {}).get("rating") or ""))
    diversity_risk = normalize(str(verifier.get("diversity_risk") or ""))
    selected_checks = verifier.get("selected_skill_checks") or []
    overlap_checks = verifier.get("overlap_checks") or []
    unsupported = verifier.get("unsupported_claims") or []
    replacements = verifier.get("better_replacements_from_panel") or []
    requirement_coverage = verifier.get("requirement_coverage") or []

    hard_reasons: list[str] = []
    soft_reasons: list[str] = []
    if verdict == "reject":
        hard_reasons.append("verifier_reject")
    if sufficiency == "fail":
        hard_reasons.append("sufficiency_fail")
    elif sufficiency == "weak":
        soft_reasons.append("sufficiency_weak")
    if minimality == "fail":
        hard_reasons.append("minimality_fail")
    elif minimality == "weak":
        soft_reasons.append("minimality_weak")
    if naturalness in {"unnatural", "too_generic"}:
        hard_reasons.append(f"query_{naturalness}")
    elif naturalness == "too_tool_specific":
        soft_reasons.append("query_too_tool_specific")
    if diversity_risk == "high":
        hard_reasons.append("diversity_risk_high")
    elif diversity_risk == "medium":
        soft_reasons.append("diversity_risk_medium")
    if score < 65:
        hard_reasons.append("score_below_65")
    elif score < 80:
        soft_reasons.append("score_below_80")
    if unsupported:
        hard_reasons.append("unsupported_claims")
    if replacements:
        soft_reasons.append("better_replacements_suggested")
    if any(normalize(str(item.get("coverage") or "")) == "missing" for item in requirement_coverage):
        hard_reasons.append("requirement_missing")
    if any(normalize(str(item.get("coverage") or "")) == "indirect" for item in requirement_coverage):
        soft_reasons.append("requirement_indirect")

    for check in selected_checks:
        necessity = normalize(str(check.get("necessity") or ""))
        grounding = normalize(str(check.get("evidence_grounding") or ""))
        unique = normalize(str(check.get("unique_contribution") or ""))
        if necessity in {"redundant", "unsupported"}:
            hard_reasons.append(f"skill_{necessity}")
        elif necessity == "useful":
            soft_reasons.append("skill_only_useful")
        if grounding == "fail":
            hard_reasons.append("evidence_grounding_fail")
        elif grounding == "weak":
            soft_reasons.append("evidence_grounding_weak")
        if unique == "fail":
            hard_reasons.append("unique_contribution_fail")
        elif unique == "weak":
            soft_reasons.append("unique_contribution_weak")

    if any(normalize(str(item.get("level") or "")) == "high" for item in overlap_checks):
        hard_reasons.append("high_overlap")
    if any(normalize(str(item.get("level") or "")) == "medium" for item in overlap_checks):
        soft_reasons.append("medium_overlap")

    hard_reasons = sorted(set(hard_reasons))
    soft_reasons = sorted(set(soft_reasons))
    accepted = verdict == "accept" and not hard_reasons and score >= 80
    return {
        "accepted": accepted,
        "hard_reasons": hard_reasons,
        "soft_reasons": soft_reasons,
        "verdict": verdict,
        "score": score,
    }


def skill_text(skill: dict[str, Any]) -> str:
    fields = [
        skill.get("skill_id"),
        skill.get("name"),
        skill.get("description"),
        skill.get("operation_role"),
        skill.get("operation_type"),
        skill.get("skill_kind"),
        " ".join(skill.get("input_artifacts") or []),
        " ".join(skill.get("output_artifacts") or []),
    ]
    return normalize(" ".join(str(field or "") for field in fields))


def any_selected_skill(payload: dict[str, Any], predicate: Any) -> bool:
    return any(predicate(skill, skill_text(skill)) for skill in payload.get("selected_skills") or [])


def semantic_guardrails(payload: dict[str, Any]) -> dict[str, list[str]]:
    query = normalize(str(payload.get("query") or ""))
    desired = normalize(" ".join(str(item or "") for item in (payload.get("desired_outputs") or [])))
    text = f"{query} {desired}"
    hard: list[str] = []
    soft: list[str] = []

    # Specific requested products or concrete outputs should have directly
    # matching selected capabilities. These are deliberately conservative:
    # they only trigger when the user/query explicitly names the requirement.
    if "prometheus" in text and not any_selected_skill(payload, lambda _s, st: "prometheus" in st):
        hard.append("missing_prometheus_capability")
    if "mkdocs" in text and not any_selected_skill(payload, lambda _s, st: "mkdocs" in st or "material theme" in st):
        hard.append("missing_mkdocs_capability")
    if "axiom" in text and not any_selected_skill(payload, lambda _s, st: "axiom" in st):
        hard.append("missing_axiom_capability")

    if ("project index" in text or "structure index" in text or "structured index" in text) and not any_selected_skill(
        payload,
        lambda _s, st: "index" in st or "sampling or indexing" in st or "sampling_or_indexing" in st,
    ):
        hard.append("missing_indexing_capability")

    if ("structured logging" in text or "log standard" in text or "standardized log" in text) and not any_selected_skill(
        payload,
        lambda _s, st: "structured logging" in st or "standardiz" in st or "correlation id" in st,
    ):
        hard.append("missing_structured_logging_capability")

    validation_terms = [
        "validation result",
        "validation results",
        "validated",
        "validate the",
        "validate documentation",
        "validating",
    ]
    if any(term in text for term in validation_terms):
        has_direct_validation = any_selected_skill(
            payload,
            lambda skill, st: (
                normalize(str(skill.get("operation_role") or "")) == "validate"
                or "validation result" in normalize(" ".join(skill.get("output_artifacts") or []))
                or any(
                    term in normalize(str(skill.get("operation_type") or ""))
                    for term in ["review", "evaluation", "validation"]
                )
            ),
        )
        if not has_direct_validation:
            hard.append("missing_direct_validation_capability")

    return {"hard": sorted(set(hard)), "soft": sorted(set(soft))}


def merge_guardrails(summary: dict[str, Any], guardrails: dict[str, list[str]]) -> dict[str, Any]:
    hard = sorted(set(list(summary.get("hard_reasons") or []) + list(guardrails.get("hard") or [])))
    soft = sorted(set(list(summary.get("soft_reasons") or []) + list(guardrails.get("soft") or [])))
    out = dict(summary)
    out["hard_reasons"] = hard
    out["soft_reasons"] = soft
    out["accepted"] = bool(out.get("verdict") == "accept" and not hard and float(out.get("score") or 0) >= 80)
    return out


def verify_one(
    *,
    row: dict[str, Any],
    panels: dict[tuple[str, str], dict[str, Any]],
    skills: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    target = int(row.get("target_skill_count") or (row.get("generated") or {}).get("target_skill_count") or 0)
    if "error" in row or not row.get("generated"):
        return {
            "index": row.get("index"),
            "task_space_id": row.get("task_space_id"),
            "anchor_task_object": row.get("anchor_task_object"),
            "target_skill_count": target,
            "error": row.get("error") or "missing generated payload",
            "verifier_summary": {
                "accepted": False,
                "hard_reasons": ["generation_error"],
                "soft_reasons": [],
                "verdict": "reject",
                "score": 0,
            },
        }

    payload = build_payload(row, panels, skills, args.desc_chars, args.max_distractors)
    messages = build_messages(payload)
    try:
        try:
            response = call_chat_completion(
                args.api_base,
                api_key,
                args.model,
                messages,
                args.temperature,
                args.timeout,
                use_schema=not args.no_schema,
                target_skill_count=target,
                max_distractors=args.max_distractors,
            )
        except urllib.error.HTTPError as exc:
            if not args.no_schema:
                response = call_chat_completion(
                    args.api_base,
                    api_key,
                    args.model,
                    messages,
                    args.temperature,
                    args.timeout,
                    use_schema=False,
                    target_skill_count=target,
                    max_distractors=args.max_distractors,
                )
            else:
                raise exc
        verifier = parse_content(response)
        summary = merge_guardrails(normalize_verdict(verifier), semantic_guardrails(payload))
        return {
            "index": row.get("index"),
            "task_space_id": row.get("task_space_id"),
            "anchor_task_object": row.get("anchor_task_object"),
            "target_skill_count": target,
            "selected_skill_ids": (row.get("generated") or {}).get("selected_skill_ids"),
            "query": (row.get("generated") or {}).get("query"),
            "verifier": verifier,
            "verifier_summary": summary,
            "usage": response.get("usage"),
            "model": response.get("model", args.model),
        }
    except Exception as exc:
        return {
            "index": row.get("index"),
            "task_space_id": row.get("task_space_id"),
            "anchor_task_object": row.get("anchor_task_object"),
            "target_skill_count": target,
            "selected_skill_ids": (row.get("generated") or {}).get("selected_skill_ids"),
            "query": (row.get("generated") or {}).get("query"),
            "error": f"{type(exc).__name__}: {exc}",
            "verifier_summary": {
                "accepted": False,
                "hard_reasons": ["verifier_error"],
                "soft_reasons": [],
                "verdict": "reject",
                "score": 0,
            },
        }


def _verify_process_worker(queue: mp.Queue, kwargs: dict[str, Any]) -> None:
    try:
        queue.put(verify_one(**kwargs))
    except BaseException as exc:
        row = kwargs.get("row") or {}
        queue.put({
            "index": row.get("index"),
            "task_space_id": row.get("task_space_id"),
            "anchor_task_object": row.get("anchor_task_object"),
            "target_skill_count": row.get("target_skill_count"),
            "error": f"{type(exc).__name__}: {exc}",
            "verifier_summary": {
                "accepted": False,
                "hard_reasons": ["verifier_error"],
                "soft_reasons": [],
                "verdict": "reject",
                "score": 0,
            },
        })


def verify_with_process_timeout(kwargs: dict[str, Any], timeout: int) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_verify_process_worker, args=(queue, kwargs))
    process.start()
    process.join(timeout)
    row = kwargs.get("row") or {}
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        return {
            "index": row.get("index"),
            "task_space_id": row.get("task_space_id"),
            "anchor_task_object": row.get("anchor_task_object"),
            "target_skill_count": row.get("target_skill_count"),
            "selected_skill_ids": (row.get("generated") or {}).get("selected_skill_ids"),
            "query": (row.get("generated") or {}).get("query"),
            "error": f"TimeoutError: hard verifier timeout after {timeout}s",
            "verifier_summary": {
                "accepted": False,
                "hard_reasons": ["verifier_timeout"],
                "soft_reasons": [],
                "verdict": "reject",
                "score": 0,
            },
        }
    if not queue.empty():
        return queue.get()
    return {
        "index": row.get("index"),
        "task_space_id": row.get("task_space_id"),
        "anchor_task_object": row.get("anchor_task_object"),
        "target_skill_count": row.get("target_skill_count"),
        "selected_skill_ids": (row.get("generated") or {}).get("selected_skill_ids"),
        "query": (row.get("generated") or {}).get("query"),
        "error": f"ProcessError: verifier worker exited with code {process.exitcode} without a result",
        "verifier_summary": {
            "accepted": False,
            "hard_reasons": ["verifier_error"],
            "soft_reasons": [],
            "verdict": "reject",
            "score": 0,
        },
    }


def write_report(path: Path, results: list[dict[str, Any]]) -> None:
    verdict_counts = Counter((row.get("verifier_summary") or {}).get("verdict") for row in results)
    hard_counts = Counter(
        reason
        for row in results
        for reason in (row.get("verifier_summary") or {}).get("hard_reasons", [])
    )
    soft_counts = Counter(
        reason
        for row in results
        for reason in (row.get("verifier_summary") or {}).get("soft_reasons", [])
    )
    by_target: dict[int, list[dict[str, Any]]] = {}
    for row in results:
        by_target.setdefault(int(row.get("target_skill_count") or 0), []).append(row)

    accepted = sum(1 for row in results if (row.get("verifier_summary") or {}).get("accepted"))
    lines = [
        "# Multi-Skill Semantic Verifier Report",
        "",
        f"- rows: {len(results)}",
        f"- accepted: {accepted}/{len(results)}" if results else "- accepted: 0/0",
        "",
        "## By Target",
        "",
    ]
    for target, items in sorted(by_target.items()):
        acc = sum(1 for item in items if (item.get("verifier_summary") or {}).get("accepted"))
        avg = sum(float((item.get("verifier_summary") or {}).get("score") or 0) for item in items) / len(items)
        lines.append(f"- {target}-skill: accepted {acc}/{len(items)}, avg score {avg:.2f}")
    lines.extend(["", "## Verdicts", ""])
    for verdict, count in verdict_counts.most_common():
        lines.append(f"- `{verdict}`: {count}")
    lines.extend(["", "## Hard Reasons", ""])
    for reason, count in hard_counts.most_common():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Soft Reasons", ""])
    for reason, count in soft_counts.most_common():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Samples", ""])
    for row in results:
        summary = row.get("verifier_summary") or {}
        lines.append(f"### target={row.get('target_skill_count')} index={row.get('index')} verdict={summary.get('verdict')} score={summary.get('score')}")
        lines.append(f"- accepted: {summary.get('accepted')}")
        lines.append(f"- task_space: `{row.get('task_space_id')}`")
        lines.append(f"- selected: `{row.get('selected_skill_ids')}`")
        lines.append(f"- hard reasons: `{summary.get('hard_reasons')}`")
        lines.append(f"- soft reasons: `{summary.get('soft_reasons')}`")
        if row.get("verifier"):
            verifier = row["verifier"]
            lines.append(f"- sufficiency: `{(verifier.get('sufficiency') or {}).get('rating')}` - {(verifier.get('sufficiency') or {}).get('reason')}")
            lines.append(f"- minimality: `{(verifier.get('minimality') or {}).get('rating')}` - {(verifier.get('minimality') or {}).get('reason')}")
            coverage = verifier.get("requirement_coverage") or []
            if coverage:
                compact = [
                    {
                        "requirement": item.get("requirement"),
                        "coverage": item.get("coverage"),
                        "covered_by": item.get("covered_by_skill_ids"),
                    }
                    for item in coverage
                ]
                lines.append(f"- requirement coverage: `{compact}`")
            lines.append(f"- decision: {verifier.get('decision_reason')}")
        if row.get("error"):
            lines.append(f"- error: {row.get('error')}")
        lines.append(f"- query: {row.get('query')}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--panels", required=True, type=Path)
    parser.add_argument("--skills", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("CHATANYWHERE_MODEL", "gpt-4o"))
    parser.add_argument("--api-base", default=os.environ.get("CHATANYWHERE_API_BASE", "https://api.chatanywhere.tech/v1/"))
    parser.add_argument("--api-key-env", default="CHATANYWHERE_API_KEY")
    parser.add_argument("--desc-chars", type=int, default=320)
    parser.add_argument("--max-distractors", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--hard-timeout", type=int, default=110)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-schema", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")

    panels = load_panel_lookup(args.panels)
    skills = load_skill_lookup(args.skills)
    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        for row in load_jsonl(path):
            row = dict(row)
            row["_source_file"] = str(path)
            rows.append(row)
    if args.limit > 0:
        rows = rows[: args.limit]

    results: list[dict[str, Any]] = []
    for row in rows:
        kwargs = {
            "row": row,
            "panels": panels,
            "skills": skills,
            "args": args,
            "api_key": api_key,
        }
        if args.hard_timeout > 0:
            result = verify_with_process_timeout(kwargs, args.hard_timeout)
        else:
            result = verify_one(**kwargs)
        result["source_file"] = row.get("_source_file")
        results.append(result)
        print(json.dumps({
            "target": result.get("target_skill_count"),
            "index": result.get("index"),
            "accepted": (result.get("verifier_summary") or {}).get("accepted"),
            "verdict": (result.get("verifier_summary") or {}).get("verdict"),
            "score": (result.get("verifier_summary") or {}).get("score"),
            "hard_reasons": (result.get("verifier_summary") or {}).get("hard_reasons"),
            "soft_reasons": (result.get("verifier_summary") or {}).get("soft_reasons"),
        }, ensure_ascii=False))
        if args.sleep:
            time.sleep(args.sleep)

    write_jsonl(args.output, results)
    write_report(args.report, results)


if __name__ == "__main__":
    main()
