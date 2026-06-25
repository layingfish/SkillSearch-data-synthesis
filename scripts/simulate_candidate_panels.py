#!/usr/bin/env python3
"""Simulate ToolLLM-style candidate panels from panel-oriented labels.

The simulator does not generate queries. It checks whether classification labels
can produce redundant but coherent panels that an LLM could later use to choose
2/3/4 skills for a synthesized instruction.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


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

GENERIC_TASK_OBJECTS = {
    "codebase",
    "unknown",
}

LOW_RISK_FLAGS = {
    "repaired_coarse_task_space",
    "operation_reassigned_by_text",
    "generic_operation_repaired",
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


def confidence(row: dict[str, Any], key: str) -> float:
    conf = row.get("classification_confidence") or {}
    try:
        return float(conf.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def usable(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if row.get("operation_role") == "general_capability":
        return False
    if confidence(row, "task_space") < args.min_task_confidence:
        return False
    if confidence(row, "operation") < args.min_operation_confidence:
        return False
    if confidence(row, "artifact") < args.min_artifact_confidence:
        return False
    if confidence(row, "task_object") < args.min_task_object_confidence:
        return False
    flags = set(row.get("quality_flags") or [])
    if set(args.exclude_flag) & flags:
        return False
    return True


def group_key(row: dict[str, Any], args: argparse.Namespace) -> str:
    if args.group_field == "task_space_parent_id":
        path = row.get("task_space_path") or []
        if len(path) >= 2:
            return "/".join(str(part) for part in path[:2])
    return str(row.get(args.group_field) or row.get("task_space_id") or "unknown")


def task_space_groups(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if usable(row, args):
            groups[group_key(row, args)].append(row)
    return groups


def score_skill_for_panel(row: dict[str, Any]) -> tuple[float, int, str]:
    conf = row.get("classification_confidence") or {}
    avg_conf = (
        float(conf.get("task_space", 0.0))
        + float(conf.get("operation", 0.0))
        + float(conf.get("artifact", 0.0))
    ) / 3
    flag_penalty = len(row.get("quality_flags") or [])
    return (avg_conf, -flag_penalty, str(row.get("skill_id") or row.get("id") or ""))


def anchor_group(group: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    if not args.anchor_task_object:
        return group, "none"

    object_counts = Counter(
        obj
        for row in group
        for obj in (row.get("task_object_types") or [])
        if obj not in GENERIC_TASK_OBJECTS
    )
    candidates: list[tuple[int, int, int, str, list[dict[str, Any]]]] = []
    for obj, _ in object_counts.most_common():
        anchored = [row for row in group if obj in set(row.get("task_object_types") or [])]
        if len(anchored) < args.min_anchor_group_size:
            continue
        role_count = len({str(row.get("operation_role")) for row in anchored})
        type_count = len({str(row.get("operation_type")) for row in anchored})
        if role_count < args.min_roles or type_count < args.min_operation_types:
            continue
        candidates.append((role_count, type_count, len(anchored), obj, anchored))

    if not candidates:
        return group, "none"
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    _, _, _, obj, anchored = candidates[0]
    return anchored, obj


def pick_panel(group: list[dict[str, Any]], args: argparse.Namespace, rng: random.Random) -> tuple[list[dict[str, Any]], str]:
    group, anchor = anchor_group(group, args)
    by_role_type: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in group:
        by_role_type[str(row.get("operation_role"))][str(row.get("operation_type"))].append(row)

    role_order = sorted(
        by_role_type,
        key=lambda role: (len(by_role_type[role]), sum(len(v) for v in by_role_type[role].values()), role),
        reverse=True,
    )
    if len(role_order) < args.min_roles:
        return [], anchor

    selected_roles = role_order[: max(args.min_roles, min(args.max_roles, len(role_order)))]
    panel: list[dict[str, Any]] = []
    role_cursors: dict[str, int] = {role: 0 for role in selected_roles}
    type_items: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for role in selected_roles:
        operation_types = sorted(
            by_role_type[role],
            key=lambda op_type: (len(by_role_type[role][op_type]), op_type),
            reverse=True,
        )
        for op_type in operation_types:
            items = sorted(by_role_type[role][op_type], key=score_skill_for_panel, reverse=True)
            if args.shuffle_within_bucket:
                top = items[: min(len(items), args.shuffle_top_k)]
                rng.shuffle(top)
                items = top + items[len(top):]
            type_items[(role, op_type)] = items[: args.max_per_operation_type]

    while len(panel) < args.panel_size:
        progressed = False
        for role in selected_roles:
            types = [op_type for r, op_type in type_items if r == role]
            if not types:
                continue
            op_type = types[role_cursors[role] % len(types)]
            role_cursors[role] += 1
            items = type_items.get((role, op_type), [])
            if not items:
                continue
            panel.append(items.pop(0))
            progressed = True
            if len(panel) >= args.panel_size:
                break
        if not progressed:
            break
    return panel, anchor


def artifact_bridge_count(panel: list[dict[str, Any]]) -> int:
    count = 0
    for i, src in enumerate(panel):
        outputs = set(src.get("output_artifact_types") or []) - GENERIC_ARTIFACTS
        if not outputs:
            continue
        for j, dst in enumerate(panel):
            if i == j:
                continue
            inputs = set(dst.get("input_artifact_types") or []) - GENERIC_ARTIFACTS
            if outputs & inputs:
                count += 1
    return count


def task_object_bridge_count(panel: list[dict[str, Any]]) -> int:
    count = 0
    object_sets = [
        set(row.get("task_object_types") or []) - GENERIC_TASK_OBJECTS
        for row in panel
    ]
    for i, src_objects in enumerate(object_sets):
        if not src_objects:
            continue
        for j, dst_objects in enumerate(object_sets):
            if i != j and src_objects & dst_objects:
                count += 1
    return count


def panel_metrics(panel: list[dict[str, Any]], target_size: int) -> dict[str, Any]:
    roles = Counter(str(row.get("operation_role")) for row in panel)
    types = Counter(str(row.get("operation_type")) for row in panel)
    flags = Counter(flag for row in panel for flag in (row.get("quality_flags") or []))
    risk_flags = Counter(
        flag
        for row in panel
        for flag in (row.get("quality_flags") or [])
        if flag not in LOW_RISK_FLAGS
    )
    modalities = Counter(modality for row in panel for modality in (row.get("modality_types") or []))
    kinds = Counter(str(row.get("skill_kind")) for row in panel)
    task_objects = Counter(
        obj
        for row in panel
        for obj in (row.get("task_object_types") or [])
        if obj not in GENERIC_TASK_OBJECTS
    )
    task_object_families = Counter(
        family
        for row in panel
        for family in (row.get("task_object_families") or [])
    )
    confidences = [
        mean([
            confidence(row, "task_space"),
            confidence(row, "operation"),
            confidence(row, "artifact"),
            confidence(row, "task_object"),
        ])
        for row in panel
    ]
    size = len(panel)
    dominant_task_object, dominant_task_object_count = task_objects.most_common(1)[0] if task_objects else ("unknown", 0)
    return {
        "size": size,
        "size_ratio": round(size / target_size, 6) if target_size else 0.0,
        "operation_role_count": len(roles),
        "operation_type_count": len(types),
        "max_role_share": round(max(roles.values()) / size, 6) if size else 0.0,
        "max_type_share": round(max(types.values()) / size, 6) if size else 0.0,
        "quality_flag_rate": round(sum(risk_flags.values()) / size, 6) if size else 0.0,
        "audit_flag_rate": round(sum(flags.values()) / size, 6) if size else 0.0,
        "artifact_bridge_count": artifact_bridge_count(panel),
        "task_object_bridge_count": task_object_bridge_count(panel),
        "task_object_count": len(task_objects),
        "task_object_family_count": len(task_object_families),
        "dominant_task_object": dominant_task_object,
        "dominant_task_object_share": round(dominant_task_object_count / size, 6) if size else 0.0,
        "modality_count": len(modalities),
        "skill_kind_count": len(kinds),
        "avg_confidence": round(mean(confidences), 6) if confidences else 0.0,
        "roles": dict(roles.most_common()),
        "types": dict(types.most_common()),
        "flags": dict(flags.most_common()),
        "risk_flags": dict(risk_flags.most_common()),
        "modalities": dict(modalities.most_common()),
        "task_objects": dict(task_objects.most_common()),
        "task_object_families": dict(task_object_families.most_common()),
    }


def viability(metrics: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics["size"] < args.panel_size:
        reasons.append("too_small")
    if metrics["operation_role_count"] < args.min_roles:
        reasons.append("too_few_roles")
    if metrics["operation_type_count"] < args.min_operation_types:
        reasons.append("too_few_operation_types")
    if metrics["max_type_share"] > args.max_type_share:
        reasons.append("type_dominance")
    if metrics["quality_flag_rate"] > args.max_quality_flag_rate:
        reasons.append("too_many_flags")
    if metrics["avg_confidence"] < args.min_avg_confidence:
        reasons.append("low_confidence")
    if metrics["artifact_bridge_count"] < args.min_artifact_bridges:
        reasons.append("weak_artifact_connectivity")
    if metrics["task_object_bridge_count"] < args.min_task_object_bridges:
        reasons.append("weak_task_object_connectivity")
    if metrics["dominant_task_object_share"] < args.min_dominant_task_object_share:
        reasons.append("no_task_object_anchor")
    if args.max_task_object_families > 0 and metrics["task_object_family_count"] > args.max_task_object_families:
        reasons.append("too_many_task_object_families")
    return not reasons, reasons


def summarize_panel_record(task_space_id: str, anchor: str, panel: list[dict[str, Any]], metrics: dict[str, Any], ok: bool, reasons: list[str]) -> dict[str, Any]:
    return {
        "task_space_id": task_space_id,
        "anchor_task_object": anchor,
        "viable": ok,
        "failure_reasons": reasons,
        "metrics": metrics,
        "skills": [
            {
                "skill_id": row.get("skill_id") or row.get("id"),
                "skill_name": row.get("skill_name") or row.get("name"),
                "operation_role": row.get("operation_role"),
                "operation_type": row.get("operation_type"),
                "task_object_types": row.get("task_object_types"),
                "local_task_object_types": row.get("local_task_object_types"),
                "context_task_object_types": row.get("context_task_object_types"),
                "task_object_signal_source": row.get("task_object_signal_source"),
                "task_object_families": row.get("task_object_families"),
                "input_artifact_types": row.get("input_artifact_types"),
                "output_artifact_types": row.get("output_artifact_types"),
                "modality_types": row.get("modality_types"),
                "quality_flags": row.get("quality_flags"),
            }
            for row in panel
        ],
    }


def simulate(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    groups = task_space_groups(rows, args)
    panels: list[dict[str, Any]] = []
    ineligible = 0
    for task_space_id, group in sorted(groups.items()):
        if len(group) < args.min_task_space_size:
            ineligible += 1
            continue
        for _ in range(args.panels_per_task_space):
            panel, anchor = pick_panel(group, args, rng)
            metrics = panel_metrics(panel, args.panel_size)
            ok, reasons = viability(metrics, args)
            if args.require_anchor and anchor == "none":
                reasons.append("no_selected_anchor")
                ok = False
            panels.append(summarize_panel_record(task_space_id, anchor, panel, metrics, ok, reasons))

    reason_counts = Counter(reason for panel in panels for reason in panel["failure_reasons"])
    viable_panels = [panel for panel in panels if panel["viable"]]
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    summary = {
        "input_rows": len(rows),
        "usable_rows": sum(len(group) for group in groups.values()),
        "task_spaces_with_usable_rows": len(groups),
        "ineligible_small_task_spaces": ineligible,
        "panels": len(panels),
        "viable_panels": len(viable_panels),
        "viable_panel_rate": round(len(viable_panels) / len(panels), 6) if panels else 0.0,
        "failure_reasons": reason_counts.most_common(),
        "avg_panel_size": round(mean(panel["metrics"]["size"] for panel in panels), 6) if panels else 0.0,
        "avg_operation_roles": round(mean(panel["metrics"]["operation_role_count"] for panel in panels), 6) if panels else 0.0,
        "avg_operation_types": round(mean(panel["metrics"]["operation_type_count"] for panel in panels), 6) if panels else 0.0,
        "avg_artifact_bridges": round(mean(panel["metrics"]["artifact_bridge_count"] for panel in panels), 6) if panels else 0.0,
        "avg_task_object_bridges": round(mean(panel["metrics"]["task_object_bridge_count"] for panel in panels), 6) if panels else 0.0,
        "avg_task_object_count": round(mean(panel["metrics"]["task_object_count"] for panel in panels), 6) if panels else 0.0,
        "avg_dominant_task_object_share": round(mean(panel["metrics"]["dominant_task_object_share"] for panel in panels), 6) if panels else 0.0,
        "anchored_panel_rate": round(sum(1 for panel in panels if panel.get("anchor_task_object") != "none") / len(panels), 6) if panels else 0.0,
        "avg_flag_rate": round(mean(panel["metrics"]["quality_flag_rate"] for panel in panels), 6) if panels else 0.0,
        "args": serialized_args,
    }
    return panels, summary


def report(summary: dict[str, Any], panels: list[dict[str, Any]], max_examples: int) -> str:
    lines = [
        "# Candidate Panel Simulation",
        "",
        "This evaluates whether the current classification can construct ToolLLM-style candidate panels.",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "input_rows",
        "usable_rows",
        "task_spaces_with_usable_rows",
        "ineligible_small_task_spaces",
        "panels",
        "viable_panels",
        "viable_panel_rate",
        "avg_panel_size",
        "avg_operation_roles",
        "avg_operation_types",
        "avg_artifact_bridges",
        "avg_task_object_bridges",
        "avg_task_object_count",
        "avg_dominant_task_object_share",
        "anchored_panel_rate",
        "avg_flag_rate",
    ]:
        lines.append(f"| {key} | {summary[key]} |")
    lines.extend(["", "## Failure Reasons", ""])
    for reason, count in summary["failure_reasons"]:
        lines.append(f"- `{reason}`: {count}")

    good = [panel for panel in panels if panel["viable"]][:max_examples]
    bad = [panel for panel in panels if not panel["viable"]][:max_examples]
    lines.extend(["", "## Viable Examples", ""])
    for panel in good:
        lines.append(f"### {panel['task_space_id']} anchor={panel.get('anchor_task_object')}")
        lines.append(f"- metrics: {panel['metrics']}")
        for skill in panel["skills"][:12]:
            lines.append(f"- `{skill['operation_role']}/{skill['operation_type']}` objs={skill.get('task_object_types')} source={skill.get('task_object_signal_source')} | {skill['skill_id']} | {skill['skill_name']}")
        lines.append("")
    lines.extend(["", "## Non-Viable Examples", ""])
    for panel in bad:
        lines.append(f"### {panel['task_space_id']} anchor={panel.get('anchor_task_object')} reasons={panel['failure_reasons']}")
        lines.append(f"- metrics: {panel['metrics']}")
        for skill in panel["skills"][:12]:
            lines.append(f"- `{skill['operation_role']}/{skill['operation_type']}` objs={skill.get('task_object_types')} source={skill.get('task_object_signal_source')} | {skill['skill_id']} | {skill['skill_name']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--panel-jsonl", type=Path)
    parser.add_argument("--panel-size", type=int, default=15)
    parser.add_argument("--panels-per-task-space", type=int, default=1)
    parser.add_argument("--min-task-space-size", type=int, default=20)
    parser.add_argument("--group-field", default="task_space_id")
    parser.add_argument("--min-roles", type=int, default=4)
    parser.add_argument("--max-roles", type=int, default=6)
    parser.add_argument("--min-operation-types", type=int, default=8)
    parser.add_argument("--max-per-operation-type", type=int, default=3)
    parser.add_argument("--max-type-share", type=float, default=0.35)
    parser.add_argument("--max-quality-flag-rate", type=float, default=1.6)
    parser.add_argument("--min-artifact-bridges", type=int, default=1)
    parser.add_argument("--min-task-object-bridges", type=int, default=0)
    parser.add_argument("--min-dominant-task-object-share", type=float, default=0.0)
    parser.add_argument("--max-task-object-families", type=int, default=0)
    parser.add_argument("--anchor-task-object", action="store_true")
    parser.add_argument("--require-anchor", action="store_true")
    parser.add_argument("--min-anchor-group-size", type=int, default=20)
    parser.add_argument("--min-avg-confidence", type=float, default=0.68)
    parser.add_argument("--min-task-confidence", type=float, default=0.0)
    parser.add_argument("--min-operation-confidence", type=float, default=0.55)
    parser.add_argument("--min-artifact-confidence", type=float, default=0.0)
    parser.add_argument("--min-task-object-confidence", type=float, default=0.0)
    parser.add_argument("--exclude-flag", action="append", default=["generic_operation_label"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle-within-bucket", action="store_true")
    parser.add_argument("--shuffle-top-k", type=int, default=20)
    parser.add_argument("--max-report-examples", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    panels, summary = simulate(rows, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_report.write_text(report(summary, panels, args.max_report_examples), encoding="utf-8")
    if args.panel_jsonl is not None:
        write_jsonl(args.panel_jsonl, panels)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
