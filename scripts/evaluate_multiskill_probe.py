#!/usr/bin/env python3
"""Evaluate generated multi-skill probe samples with deterministic checks."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BAD_WEAKNESS_TERMS = [
    "overlap",
    "overlapping",
    "unsupported",
    "not fully compatible",
    "might require skills that are not",
    "unclear",
    "ambiguous",
]

NATURAL_TOOL_NAMES = {
    "aws",
    "azure",
    "axiom",
    "docker",
    "github",
    "gitlab",
    "grafana",
    "jira",
    "kubernetes",
    "linear",
    "material",
    "mkdocs",
    "mysql",
    "notion",
    "postgres",
    "postgresql",
    "prometheus",
    "redis",
    "salesforce",
    "sentry",
    "slack",
}

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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_skill_lookup(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sid = str(row.get("skill_id") or row.get("id") or "")
        if sid:
            lookup[sid] = row
    return lookup


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokenish_name(name: str) -> str:
    return normalize(name)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
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
    return {
        token
        for token in normalize(text).split()
        if len(token) >= 4 and token not in stopwords
    }


def token_coverage(claim: set[str], evidence: set[str]) -> float:
    if not claim:
        return 1.0
    return len(claim & evidence) / len(claim)


def skill_text(skill: dict[str, Any], full: dict[str, Any] | None = None) -> str:
    full = full or {}
    return normalize(" ".join([
        str(skill.get("skill_id") or full.get("skill_id") or ""),
        str(skill.get("skill_name") or full.get("skill_name") or full.get("name") or ""),
        str(full.get("description") or full.get("task_description") or ""),
        str(skill.get("operation_role") or full.get("operation_role") or ""),
        str(skill.get("operation_type") or full.get("operation_type") or ""),
        " ".join(skill.get("task_object_types") or full.get("task_object_types") or []),
        " ".join(skill.get("input_artifact_types") or full.get("input_artifact_types") or []),
        " ".join(skill.get("output_artifact_types") or full.get("output_artifact_types") or []),
    ]))


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


def is_suspicious_skill_name(name: str, skill_id: str) -> bool:
    norm = tokenish_name(name)
    if not norm or norm in NATURAL_TOOL_NAMES:
        return False
    if str(skill_id).startswith("product/"):
        return False
    if any(ch in name for ch in ["/", "_"]):
        return True
    if any(ch.isdigit() for ch in name):
        return True
    if "-" in name:
        parts = [part for part in re.split(r"-+", name.lower()) if part]
        if len(parts) >= 4:
            return True
        if any(len(part) >= 14 for part in parts):
            return True
    return False


def overlap_level(generated: dict[str, Any]) -> str:
    risk = generated.get("overlap_risk") or {}
    if isinstance(risk, dict):
        return normalize(str(risk.get("level") or ""))
    return normalize(str(risk))


def evaluate_row(row: dict[str, Any], full_lookup: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    if "error" in row:
        return {
            "ok": False,
            "hard_ok": False,
            "quality_ok": False,
            "hard_issues": ["generation_error"],
            "quality_warnings": [],
            "quality_score": -100,
            "error": row["error"],
            "target": int(row.get("target_skill_count") or 0),
            "selected_count": 0,
            "selected_ids": [],
            "query": "",
            "weakness": None,
            "leaked_names": [],
            "role_count": None,
            "type_count": None,
            "overlap_risk_level": "",
            "workflow_shape": None,
        }

    generated = row.get("generated") or {}
    selected = list(generated.get("selected_skill_ids") or [])
    target = int(row.get("target_skill_count") or generated.get("target_skill_count") or 0)
    candidate_ids = set(row.get("candidate_skill_ids") or [])
    selected_skills = generated.get("selected_skills") or []
    query = str(generated.get("query") or "")
    qnorm = normalize(query)
    weakness = normalize(str(generated.get("potential_weakness") or ""))

    hard_issues: list[str] = []
    quality_warnings: list[str] = []

    if not generated:
        hard_issues.append("missing_generated_payload")
    if len(selected) != target:
        hard_issues.append("wrong_selected_count")
    if len(set(selected)) != len(selected):
        hard_issues.append("duplicate_selected_ids")
    if any(sid not in candidate_ids for sid in selected):
        hard_issues.append("selected_id_not_in_panel")
    if selected_skills and len(selected_skills) != len(selected):
        hard_issues.append("selected_skills_count_mismatch")
    if selected_skills and [s.get("skill_id") for s in selected_skills] != selected:
        quality_warnings.append("selected_order_or_id_mismatch")
    if word_count(query) < 18:
        quality_warnings.append("query_too_short")
    if word_count(query) > 160:
        quality_warnings.append("query_too_long")
    if any(term in qnorm for term in ["candidate panel", "skill id", "selected skill", "operation label"]):
        hard_issues.append("query_mentions_generation_artifact")
    if any(str(sid).lower() in query.lower() for sid in selected):
        hard_issues.append("query_mentions_skill_id")

    by_id = {
        str(skill.get("skill_id")): skill
        for skill in (row.get("panel_skills") or [])
    }
    full_lookup = full_lookup or {}
    # Older probe rows do not carry panel_skills. The caller can enrich rows;
    # if not, skip role/type checks rather than fabricating evidence.
    roles: list[str] = []
    types: list[str] = []
    leaked_names: list[str] = []
    for sid in selected:
        skill = by_id.get(sid)
        if not skill:
            continue
        roles.append(str(skill.get("operation_role")))
        types.append(str(skill.get("operation_type")))
        name = tokenish_name(str(skill.get("skill_name") or ""))
        if name and len(name) >= 4 and name in qnorm and is_suspicious_skill_name(str(skill.get("skill_name") or ""), sid):
            leaked_names.append(str(skill.get("skill_name")))
    if roles and len(set(roles)) < min(len(roles), 2):
        quality_warnings.append("low_role_diversity")
    if types and len(set(types)) < len(types):
        quality_warnings.append("duplicate_operation_type")
    if target >= 4:
        validation_types = [
            str(by_id.get(sid, {}).get("operation_type") or "")
            for sid in selected
            if str(by_id.get(sid, {}).get("operation_role") or "") == "validate"
        ]
        if len(validation_types) > 1 and "validation_check" in validation_types:
            quality_warnings.append("generic_validation_overlap_in_4skill")
    if leaked_names:
        quality_warnings.append("query_mentions_suspicious_skill_name")

    selected_skill_text = normalize(" ".join(
        skill_text(by_id.get(sid, {}), full_lookup.get(sid))
        for sid in selected
    ))
    query_terms = set(qnorm.split())
    if any(term not in selected_skill_text for term in NAMED_CONTEXT_TERMS & query_terms):
        quality_warnings.append("unsupported_named_context_in_query")
    if any(term in selected_skill_text and term not in qnorm for term in QUERY_REQUIRED_CONTEXT_TERMS):
        quality_warnings.append("selected_context_requires_query_context")
    quality_warnings.extend(selected_context_warnings(selected_skill_text, qnorm))

    for sid in selected:
        skill = by_id.get(sid, {})
        text = skill_text(skill, full_lookup.get(sid))
        if any(term in text and term not in qnorm for term in NARROW_CONTEXT_TERMS):
            quality_warnings.append("narrow_context_skill_without_query_context")
            break

    for selected_skill in selected_skills:
        if not isinstance(selected_skill, dict):
            continue
        sid = str(selected_skill.get("skill_id") or "")
        if sid not in by_id and sid not in full_lookup:
            continue
        claim_tokens = content_tokens(" ".join([
            str(selected_skill.get("role_in_workflow") or ""),
            str(selected_skill.get("unique_contribution") or ""),
            str(selected_skill.get("evidence_from_skill_description") or ""),
        ]))
        evidence_tokens = content_tokens(skill_text(by_id.get(sid, {}), full_lookup.get(sid)))
        if len(claim_tokens) >= 5 and token_coverage(claim_tokens, evidence_tokens) < 0.12:
            quality_warnings.append("unsupported_skill_claim_by_evidence")
            break

    selected_set = set(selected)
    graph = generated.get("latent_workflow_graph") or {}
    node_ids = [
        str(node.get("skill_id"))
        for node in (graph.get("nodes") or [])
        if isinstance(node, dict)
    ]
    invalid_edges = 0
    edge_ids: set[str] = set()
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
        if len(graph.get("edges") or []) < max(1, target - 1):
            quality_warnings.append("workflow_graph_too_sparse")
        if selected_set and not selected_set.issubset(set(node_ids) | edge_ids):
            quality_warnings.append("workflow_graph_does_not_cover_selection")

    necessity_ids = [
        str(item.get("skill_id"))
        for item in (generated.get("skill_necessity_checks") or [])
        if isinstance(item, dict)
    ]
    if generated.get("skill_necessity_checks"):
        if set(necessity_ids) != selected_set:
            hard_issues.append("necessity_checks_do_not_match_selection")
        for item in generated.get("skill_necessity_checks") or []:
            if not isinstance(item, dict):
                hard_issues.append("malformed_necessity_check")
                continue
            if word_count(str(item.get("what_breaks_if_removed") or "")) < 5:
                quality_warnings.append("thin_leave_one_out_check")
                break

    risk_level = overlap_level(generated)
    if risk_level:
        if "high" in risk_level:
            hard_issues.append("high_overlap_risk")
        elif "medium" in risk_level:
            quality_warnings.append("medium_overlap_risk")
        elif "low" not in risk_level:
            quality_warnings.append("unclear_overlap_risk")

    if any(term in weakness for term in BAD_WEAKNESS_TERMS):
        quality_warnings.append("self_reported_risk")
    if "evidence_consistency_check" in generated and word_count(str(generated.get("evidence_consistency_check") or "")) < 6:
        quality_warnings.append("thin_evidence_consistency_check")

    hard_issues = sorted(set(hard_issues))
    quality_warnings = sorted(set(quality_warnings))
    quality_score = 100
    quality_score -= 100 * len(hard_issues)
    quality_score -= 10 * len(quality_warnings)
    if roles:
        quality_score += 4 * min(len(set(roles)), target)
    if target >= 4 and roles and len(set(roles)) == target:
        quality_score += 10
    if types:
        quality_score += 5 * min(len(set(types)), target)
    if "low" in risk_level:
        quality_score += 8
    if 35 <= word_count(query) <= 120:
        quality_score += 6
    if graph and len(graph.get("edges") or []) >= max(1, target - 1):
        quality_score += 5

    return {
        "ok": not hard_issues and not quality_warnings,
        "hard_ok": not hard_issues,
        "quality_ok": not quality_warnings,
        "hard_issues": hard_issues,
        "quality_warnings": quality_warnings,
        "quality_score": quality_score,
        "target": target,
        "selected_count": len(selected),
        "selected_ids": selected,
        "query": query,
        "weakness": generated.get("potential_weakness"),
        "leaked_names": leaked_names,
        "role_count": len(set(roles)) if roles else None,
        "type_count": len(set(types)) if types else None,
        "overlap_risk_level": risk_level,
        "workflow_shape": (graph.get("workflow_shape") if isinstance(graph, dict) else None),
    }


def enrich_with_panel_skills(rows: list[dict[str, Any]], panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    panel_by_key = {
        (panel.get("task_space_id"), panel.get("anchor_task_object")): panel
        for panel in panels
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        panel = panel_by_key.get((row.get("task_space_id"), row.get("anchor_task_object")))
        if panel:
            row["panel_skills"] = panel.get("skills") or []
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--panels", required=True, type=Path)
    parser.add_argument("--skills", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()

    panels = load_jsonl(args.panels)
    full_lookup = load_skill_lookup(args.skills)
    all_results: list[dict[str, Any]] = []
    for path in args.inputs:
        rows = enrich_with_panel_skills(load_jsonl(path), panels)
        for row in rows:
            result = evaluate_row(row, full_lookup)
            result["file"] = str(path)
            result["index"] = row.get("index")
            result["task_space_id"] = row.get("task_space_id")
            result["anchor_task_object"] = row.get("anchor_task_object")
            result["chosen_candidate_index"] = row.get("chosen_candidate_index")
            result["chosen_evaluation"] = row.get("chosen_evaluation")
            all_results.append(result)

    hard_issue_counts = Counter(issue for result in all_results for issue in result["hard_issues"])
    warning_counts = Counter(issue for result in all_results for issue in result["quality_warnings"])
    by_target: dict[int, list[dict[str, Any]]] = {}
    for result in all_results:
        by_target.setdefault(int(result.get("target") or 0), []).append(result)

    summary = {
        "rows": len(all_results),
        "ok_rows": sum(1 for result in all_results if result["ok"]),
        "hard_ok_rows": sum(1 for result in all_results if result["hard_ok"]),
        "quality_ok_rows": sum(1 for result in all_results if result["quality_ok"]),
        "ok_rate": round(sum(1 for result in all_results if result["ok"]) / len(all_results), 6) if all_results else 0.0,
        "hard_ok_rate": round(sum(1 for result in all_results if result["hard_ok"]) / len(all_results), 6) if all_results else 0.0,
        "quality_ok_rate": round(sum(1 for result in all_results if result["quality_ok"]) / len(all_results), 6) if all_results else 0.0,
        "hard_issue_counts": hard_issue_counts.most_common(),
        "warning_counts": warning_counts.most_common(),
        "by_target": {
            str(target): {
                "rows": len(items),
                "ok_rows": sum(1 for item in items if item["ok"]),
                "hard_ok_rows": sum(1 for item in items if item["hard_ok"]),
                "quality_ok_rows": sum(1 for item in items if item["quality_ok"]),
                "ok_rate": round(sum(1 for item in items if item["ok"]) / len(items), 6) if items else 0.0,
                "hard_ok_rate": round(sum(1 for item in items if item["hard_ok"]) / len(items), 6) if items else 0.0,
                "quality_ok_rate": round(sum(1 for item in items if item["quality_ok"]) / len(items), 6) if items else 0.0,
                "avg_quality_score": round(sum(item["quality_score"] for item in items) / len(items), 3) if items else 0.0,
            }
            for target, items in sorted(by_target.items())
        },
        "results": all_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Multi-Skill Smoke Test Evaluation",
        "",
        f"- rows: {summary['rows']}",
        f"- ok rows: {summary['ok_rows']}",
        f"- hard-ok rows: {summary['hard_ok_rows']}",
        f"- quality-ok rows: {summary['quality_ok_rows']}",
        f"- ok rate: {summary['ok_rate']}",
        f"- hard-ok rate: {summary['hard_ok_rate']}",
        f"- quality-ok rate: {summary['quality_ok_rate']}",
        "",
        "## By Target",
        "",
    ]
    for target, stats in summary["by_target"].items():
        lines.append(
            f"- {target}-skill: ok {stats['ok_rows']}/{stats['rows']} ({stats['ok_rate']}), "
            f"hard-ok {stats['hard_ok_rows']}/{stats['rows']} ({stats['hard_ok_rate']}), "
            f"quality-ok {stats['quality_ok_rows']}/{stats['rows']} ({stats['quality_ok_rate']}), "
            f"avg quality score {stats['avg_quality_score']}"
        )
    lines.extend(["", "## Hard Issues", ""])
    for issue, count in summary["hard_issue_counts"]:
        lines.append(f"- `{issue}`: {count}")
    lines.extend(["", "## Quality Warnings", ""])
    for issue, count in summary["warning_counts"]:
        lines.append(f"- `{issue}`: {count}")
    lines.extend(["", "## Flagged Samples", ""])
    for result in all_results:
        if not result["ok"]:
            lines.append(f"### {Path(result['file']).name} #{result['index']} target={result['target']}")
            lines.append(f"- task_space: `{result['task_space_id']}`")
            lines.append(f"- hard issues: {result['hard_issues']}")
            lines.append(f"- quality warnings: {result['quality_warnings']}")
            lines.append(f"- quality score: {result['quality_score']}")
            lines.append(f"- workflow shape: {result['workflow_shape']}")
            lines.append(f"- selected: {result['selected_ids']}")
            lines.append(f"- query: {result['query']}")
            lines.append(f"- weakness: {result['weakness']}")
            lines.append("")

    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["rows", "ok_rows", "hard_ok_rows", "quality_ok_rows", "ok_rate", "hard_ok_rate", "quality_ok_rate", "hard_issue_counts", "warning_counts", "by_target"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
