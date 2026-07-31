#!/usr/bin/env python3
"""Build strict k-skill top-up inputs for balanced SkillRouter data.

This script does not call any model. It samples replay-compatible candidate
inputs from panel-classified skills, with a stricter compatibility gate than the
original panel simulator. The intended first use is k=4 top-up, where weak
composition was the main failure mode in manual audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


GENERIC_OBJECTS = {"codebase", "unknown"}
GENERIC_ARTIFACTS = {
    "requirements",
    "source_artifact",
    "input_artifact",
    "external_resource",
    "candidate_output",
    "runtime_target",
    "source_code",
    "documentation",
    "configuration",
    "report",
    "loaded_data",
    "extracted_content",
    "transformed_artifact",
    "analysis_result",
    "validation_result",
    "generated_artifact",
    "integrated_system",
    "operation_result",
}
RISK_FLAGS = {
    "broad_skill",
    "context_only_task_object",
    "coarse_operation_type",
    "generic_operation_label",
    "large_bucket_without_refinement",
    "weak_modality_signal",
    "workflow_like_skill",
}

GENERIC_NAME_TOKENS = {
    "agent",
    "ai",
    "assistant",
    "builder",
    "checker",
    "code",
    "coder",
    "create",
    "creator",
    "dev",
    "developer",
    "generator",
    "guide",
    "helper",
    "manager",
    "pattern",
    "patterns",
    "plugin",
    "review",
    "skill",
    "skills",
    "tool",
    "tools",
}

FAMILY_HINTS: dict[str, list[str]] = {
    "accessibility": ["accessibility", "a11y", "axe"],
    "api_design": ["api design", "openapi", "endpoint design"],
    "code_review": ["code review", "reviewer", "review gate", "review comments"],
    "debugging": ["debug", "debugger", "root cause", "five whys"],
    "document_conversion": ["docx", "markdown to html", "md2pdf", "pandoc", "convert"],
    "frontend_review": ["frontend review", "frontend reviewer"],
    "google_workspace": ["google workspace", "google docs", "google sheets", "google cli"],
    "image_generation": ["image generation", "imagegen", "nano banana", "illustrator"],
    "mocking": ["mock", "wiremock", "msw", "stubbing"],
    "planning": ["plan writer", "create plan", "plan validate", "task plan"],
    "playwright": ["playwright", "browser testing", "chrome devtools"],
    "postgres": ["postgres", "postgresql"],
    "release_automation": ["release", "changelog", "release notes"],
    "research": ["research", "researcher", "websearch", "arxiv"],
    "secret_scanning": ["secret", "secrets", "keyvault"],
    "security_audit": ["security audit", "vulnerability", "threat model"],
    "supabase": ["supabase"],
    "testing": ["testing", "test suite", "unit test", "pytest"],
    "voice_tts": ["voice", "tts", "speaker", "diarization"],
    "weather": ["weather"],
}

TECH_TERMS: dict[str, list[str]] = {
    "angular": ["angular"],
    "aws": ["aws", "aurora", "rds", "lambda", "cloudformation", "cloudwatch"],
    "azure": ["azure"],
    "django": ["django"],
    "expo": ["expo", "expo router"],
    "express": ["express", "node express"],
    "fastapi": ["fastapi"],
    "firebase": ["firebase", "firestore"],
    "flutter": ["flutter"],
    "laravel": ["laravel", "filament"],
    "nextjs": ["nextjs", "next.js", "next js"],
    "node": ["node", "nodejs", "node.js", "typescript", "javascript"],
    "postgres": ["postgres", "postgresql"],
    "python": ["python", "pytest", "pydantic"],
    "rails": ["rails", "ruby on rails"],
    "react": ["react", "react native"],
    "svelte": ["svelte", "sveltekit"],
    "supabase": ["supabase"],
    "swift": ["swift", "swiftui", "ios", "macos"],
    "vue": ["vue", "nuxt"],
}

INCOMPATIBLE_TECH_GROUPS = {
    frozenset({"angular", "react"}),
    frozenset({"angular", "svelte"}),
    frozenset({"angular", "vue"}),
    frozenset({"django", "fastapi"}),
    frozenset({"django", "rails"}),
    frozenset({"expo", "flutter"}),
    frozenset({"expo", "swift"}),
    frozenset({"fastapi", "rails"}),
    frozenset({"firebase", "supabase"}),
    frozenset({"flutter", "react"}),
    frozenset({"flutter", "swift"}),
    frozenset({"laravel", "rails"}),
    frozenset({"nextjs", "svelte"}),
    frozenset({"nextjs", "vue"}),
    frozenset({"rails", "node"}),
    frozenset({"react", "svelte"}),
    frozenset({"react", "vue"}),
    frozenset({"svelte", "vue"}),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9.+#-]+", " ", str(text or "").lower()).strip()


def skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "")


def row_text(row: dict[str, Any]) -> str:
    return normalize(
        " ".join(
            [
                skill_id(row),
                str(row.get("skill_name") or row.get("name") or ""),
                str(row.get("description") or ""),
                str(row.get("task_description") or ""),
                str(row.get("domain") or ""),
                str(row.get("namespace_prefix") or ""),
                str(row.get("operation_role") or ""),
                str(row.get("operation_type") or ""),
                " ".join(row.get("task_object_types") or []),
                " ".join(row.get("input_artifact_types") or []),
                " ".join(row.get("output_artifact_types") or []),
            ]
        )
    )


def tech_groups(text: str) -> set[str]:
    groups: set[str] = set()
    padded = f" {text} "
    for group, terms in TECH_TERMS.items():
        if any(f" {normalize(term)} " in padded for term in terms):
            groups.add(group)
    return groups


def family_labels(text: str) -> set[str]:
    labels: set[str] = set()
    padded = f" {text} "
    for label, hints in FAMILY_HINTS.items():
        for hint in hints:
            if f" {normalize(hint)} " in padded:
                labels.add(label)
                break
    return labels


def important_name_tokens(row: dict[str, Any]) -> set[str]:
    text = normalize(
        " ".join(
            [
                skill_id(row).rsplit("/", 1)[-1],
                str(row.get("skill_name") or row.get("name") or ""),
            ]
        )
    )
    return {
        token
        for token in text.split()
        if len(token) >= 4 and token not in GENERIC_NAME_TOKENS
    }


def incompatible_tech(groups: set[str]) -> bool:
    return any(pair <= groups for pair in INCOMPATIBLE_TECH_GROUPS)


def non_generic_objects(row: dict[str, Any]) -> set[str]:
    return {str(obj) for obj in (row.get("task_object_types") or []) if obj and obj not in GENERIC_OBJECTS}


def non_generic_inputs(row: dict[str, Any]) -> set[str]:
    return {str(obj) for obj in (row.get("input_artifact_types") or []) if obj and obj not in GENERIC_ARTIFACTS}


def non_generic_outputs(row: dict[str, Any]) -> set[str]:
    return {str(obj) for obj in (row.get("output_artifact_types") or []) if obj and obj not in GENERIC_ARTIFACTS}


def substitute_group(row: dict[str, Any], tech: set[str]) -> str:
    obj = str(row.get("primary_task_object_type") or next(iter(sorted(non_generic_objects(row))), "unknown"))
    op = str(row.get("operation_type") or row.get("operation_role") or "unknown")
    tech_key = "+".join(sorted(tech)) if tech else str(row.get("namespace_prefix") or "")
    return normalize(f"{obj}|{op}|{tech_key}")


def usable(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if not skill_id(row):
        return False
    if row.get("operation_role") in {"", None, "general_capability"}:
        return False
    flags = set(row.get("quality_flags") or [])
    if flags & set(args.exclude_flag):
        return False
    conf = row.get("classification_confidence") or {}
    for key, threshold in [
        ("task_space", args.min_task_space_confidence),
        ("operation", args.min_operation_confidence),
        ("task_object", args.min_task_object_confidence),
    ]:
        try:
            if float(conf.get(key, 0.0)) < threshold:
                return False
        except (TypeError, ValueError):
            return False
    return True


def card(row: dict[str, Any]) -> dict[str, Any]:
    text = row_text(row)
    tech = tech_groups(text)
    return {
        "skill_id": skill_id(row),
        "skill_name": row.get("skill_name") or row.get("name") or skill_id(row).rsplit("/", 1)[-1],
        "source_task_space_id": row.get("source_task_space_id") or row.get("task_space_id"),
        "task_space_id": row.get("task_space_id"),
        "primary_task_object_type": row.get("primary_task_object_type"),
        "task_objects": sorted(non_generic_objects(row)),
        "operation_role": row.get("operation_role"),
        "operation_type": row.get("operation_type"),
        "input_artifacts": sorted(non_generic_inputs(row)),
        "output_artifacts": sorted(non_generic_outputs(row)),
        "tech_groups": sorted(tech),
        "family_labels": sorted(family_labels(text)),
        "name_tokens": sorted(important_name_tokens(row)),
        "substitute_group": substitute_group(row, tech),
        "quality_flags": row.get("quality_flags") or [],
        "description": row.get("description") or row.get("task_description") or "",
        "_row": row,
    }


def artifact_bridge(cards: list[dict[str, Any]]) -> int:
    count = 0
    for i, src in enumerate(cards):
        outputs = set(src["output_artifacts"])
        if not outputs:
            continue
        for j, dst in enumerate(cards):
            if i != j and outputs & set(dst["input_artifacts"]):
                count += 1
    return count


def shared_objects(cards: list[dict[str, Any]]) -> set[str]:
    object_sets = [set(item["task_objects"]) for item in cards if item["task_objects"]]
    if not object_sets:
        return set()
    return set.intersection(*object_sets) if all(object_sets) else set()


def has_pairwise_object_overlap(cards: list[dict[str, Any]]) -> bool:
    overlaps = 0
    for left, right in combinations(cards, 2):
        if set(left["task_objects"]) & set(right["task_objects"]):
            overlaps += 1
    return overlaps >= len(cards) - 1


def compatible(cards: list[dict[str, Any]], k: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ids = [item["skill_id"] for item in cards]
    if len(set(ids)) != len(ids):
        reasons.append("duplicate_skill_id")
    roles = {str(item["operation_role"]) for item in cards if item.get("operation_role")}
    if len(roles) < min(k, 3):
        reasons.append("low_role_diversity")
    substitute_groups = [item["substitute_group"] for item in cards]
    if len(set(substitute_groups)) != len(substitute_groups):
        reasons.append("substitute_group_collision")
    tech = {group for item in cards for group in item["tech_groups"]}
    if incompatible_tech(tech):
        reasons.append("incompatible_tech_groups")
    if not (shared_objects(cards) or artifact_bridge(cards) > 0 or has_pairwise_object_overlap(cards)):
        reasons.append("weak_shared_context")
    return not reasons, reasons


def group_cards(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in cards:
        key = str(item.get("source_task_space_id") or item.get("task_space_id") or "unknown")
        obj = str(item.get("primary_task_object_type") or "")
        if obj and obj not in GENERIC_OBJECTS:
            key = f"{key}::{obj}"
        groups[key].append(item)
    return groups


def fingerprint(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_excluded_sets(path: Path | None) -> set[tuple[str, ...]]:
    if not path:
        return set()
    excluded: set[tuple[str, ...]] = set()
    for row in load_jsonl(path):
        ids = (row.get("generated") or {}).get("selected_skill_ids") or row.get("selected_skill_ids") or []
        if ids:
            excluded.add(tuple(sorted(str(sid) for sid in ids)))
    return excluded


def sample_inputs(cards: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(args.seed)
    groups = group_cards(cards)
    group_items = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    excluded_sets = load_excluded_sets(args.exclude_selected_sets)
    seen_sets = set(excluded_sets)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    index = 1

    for _round in range(args.max_rounds):
        for group_key, group in group_items:
            if len(rows) >= args.limit:
                return rows, skipped
            if len(group) < args.target_skill_count:
                skipped["group_too_small"] += 1
                continue
            by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
            shuffled = list(group)
            rng.shuffle(shuffled)
            for item in shuffled:
                by_role[str(item.get("operation_role") or "unknown")].append(item)
            role_order = sorted(by_role, key=lambda role: len(by_role[role]), reverse=True)
            attempts = min(args.group_attempts, max(1, len(group)))
            for _attempt in range(attempts):
                chosen: list[dict[str, Any]] = []
                used_ids: set[str] = set()
                # Prefer one item per role first.
                for role in role_order:
                    rng.shuffle(by_role[role])
                    for item in by_role[role]:
                        if item["skill_id"] not in used_ids:
                            chosen.append(item)
                            used_ids.add(item["skill_id"])
                            break
                    if len(chosen) >= args.target_skill_count:
                        break
                # Fill from group if role buckets were insufficient.
                if len(chosen) < args.target_skill_count:
                    for item in shuffled:
                        if item["skill_id"] not in used_ids:
                            chosen.append(item)
                            used_ids.add(item["skill_id"])
                        if len(chosen) >= args.target_skill_count:
                            break
                if len(chosen) != args.target_skill_count:
                    skipped["not_enough_distinct"] += 1
                    continue
                ok, reasons = compatible(chosen, args.target_skill_count)
                if not ok:
                    skipped.update(reasons)
                    continue
                ids = [item["skill_id"] for item in chosen]
                key = tuple(sorted(ids))
                if key in seen_sets:
                    skipped["duplicate_selected_set"] += 1
                    continue
                seen_sets.add(key)
                rows.append(
                    {
                        "index": index,
                        "target_skill_count": args.target_skill_count,
                        "candidate_skill_ids": ids,
                        "task_space_id": str(chosen[0].get("source_task_space_id") or chosen[0].get("task_space_id")),
                        "anchor_task_object": str(chosen[0].get("primary_task_object_type") or ""),
                        "panel_fingerprint": fingerprint(ids),
                        "panel_source_tier": "compatibility_topup_v1",
                        "sampling": {
                            "source": "build_balanced_topup_inputs",
                            "seed": args.seed,
                            "group_key": group_key,
                            "role_count": len({str(item.get("operation_role")) for item in chosen}),
                            "operation_type_count": len({str(item.get("operation_type")) for item in chosen}),
                            "shared_objects": sorted(shared_objects(chosen)),
                            "artifact_bridge_count": artifact_bridge(chosen),
                            "tech_groups": sorted({group for item in chosen for group in item["tech_groups"]}),
                        },
                        "composition_constraints": {
                            "strict_same_k": True,
                            "no_substitute_group_collision": True,
                            "no_incompatible_tech_groups": True,
                            "selected_skill_summaries": [
                                {
                                    "skill_id": item["skill_id"],
                                    "skill_name": item["skill_name"],
                                    "operation_role": item["operation_role"],
                                    "operation_type": item["operation_type"],
                                    "task_objects": item["task_objects"],
                                    "input_artifacts": item["input_artifacts"],
                                    "output_artifacts": item["output_artifacts"],
                                    "tech_groups": item["tech_groups"],
                                    "description": str(item.get("description") or "")[:220],
                                }
                                for item in chosen
                            ],
                        },
                    }
                )
                index += 1
                break
    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--target-skill-count", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--max-rounds", type=int, default=80)
    parser.add_argument("--group-attempts", type=int, default=12)
    parser.add_argument("--exclude-selected-sets", type=Path)
    parser.add_argument("--min-task-space-confidence", type=float, default=0.75)
    parser.add_argument("--min-operation-confidence", type=float, default=0.35)
    parser.add_argument("--min-task-object-confidence", type=float, default=0.45)
    parser.add_argument(
        "--exclude-flag",
        action="append",
        default=sorted(RISK_FLAGS),
        help="quality flag to exclude; repeatable",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.skills)
    cards = [card(row) for row in rows if usable(row, args)]
    sampled, skipped = sample_inputs(cards, args)
    write_jsonl(args.output, sampled)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "skills": str(args.skills),
                "output": str(args.output),
                "target_skill_count": args.target_skill_count,
                "limit": args.limit,
                "sampled": len(sampled),
                "usable_cards": len(cards),
                "skipped": dict(skipped.most_common()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"sampled={len(sampled)} output={args.output}")


if __name__ == "__main__":
    main()
