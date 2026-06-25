#!/usr/bin/env python3
"""Evaluate generated single-skill probe samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from generate_single_skill_probe import load_jsonl, score_generated, skill_id


def load_skill_lookup(path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sid = skill_id(row)
        if sid:
            lookup[sid] = row
    return lookup


def evaluate_row(row: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "error" in row:
        return {
            "ok": False,
            "hard_ok": False,
            "quality_ok": False,
            "hard_issues": ["generation_error"],
            "quality_warnings": [],
            "quality_score": -100,
            "target": 1,
            "selected_ids": [],
            "query": "",
            "skill_id": row.get("skill_id"),
            "task_space_id": row.get("task_space_id"),
            "chosen_candidate_index": row.get("chosen_candidate_index"),
        }
    sid = str(row.get("skill_id") or "")
    skill = lookup.get(sid, {"skill_id": sid, "skill_name": row.get("skill_name")})
    evaluation = score_generated(row.get("generated") or {}, skill)
    score = int(evaluation.get("score") or 0)
    return {
        "ok": not evaluation["hard_issues"] and not evaluation["quality_warnings"],
        "hard_ok": not evaluation["hard_issues"],
        "quality_ok": not evaluation["quality_warnings"],
        "hard_issues": evaluation["hard_issues"],
        "quality_warnings": evaluation["quality_warnings"],
        "quality_score": score,
        "target": 1,
        "selected_ids": evaluation.get("selected_skill_ids") or [],
        "query": (row.get("generated") or {}).get("query") or "",
        "skill_id": sid,
        "task_space_id": row.get("task_space_id"),
        "chosen_candidate_index": row.get("chosen_candidate_index"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--skills", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()

    lookup = load_skill_lookup(args.skills)
    results: list[dict[str, Any]] = []
    for path in args.inputs:
        for row in load_jsonl(path):
            result = evaluate_row(row, lookup)
            result["file"] = str(path)
            result["index"] = row.get("index")
            result["chosen_evaluation"] = row.get("chosen_evaluation")
            results.append(result)

    hard_issue_counts = Counter(issue for result in results for issue in result["hard_issues"])
    warning_counts = Counter(issue for result in results for issue in result["quality_warnings"])
    summary = {
        "rows": len(results),
        "ok_rows": sum(1 for result in results if result["ok"]),
        "hard_ok_rows": sum(1 for result in results if result["hard_ok"]),
        "quality_ok_rows": sum(1 for result in results if result["quality_ok"]),
        "ok_rate": round(sum(1 for result in results if result["ok"]) / len(results), 6) if results else 0.0,
        "hard_ok_rate": round(sum(1 for result in results if result["hard_ok"]) / len(results), 6) if results else 0.0,
        "quality_ok_rate": round(sum(1 for result in results if result["quality_ok"]) / len(results), 6) if results else 0.0,
        "avg_quality_score": round(sum(result["quality_score"] for result in results) / len(results), 3) if results else 0.0,
        "hard_issue_counts": hard_issue_counts.most_common(),
        "warning_counts": warning_counts.most_common(),
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Single-Skill Pilot Evaluation",
        "",
        f"- rows: {summary['rows']}",
        f"- ok rows: {summary['ok_rows']}",
        f"- hard-ok rows: {summary['hard_ok_rows']}",
        f"- quality-ok rows: {summary['quality_ok_rows']}",
        f"- ok rate: {summary['ok_rate']}",
        f"- hard-ok rate: {summary['hard_ok_rate']}",
        f"- quality-ok rate: {summary['quality_ok_rate']}",
        f"- avg quality score: {summary['avg_quality_score']}",
        "",
        "## Hard Issues",
        "",
    ]
    for issue, count in summary["hard_issue_counts"]:
        lines.append(f"- `{issue}`: {count}")
    lines.extend(["", "## Quality Warnings", ""])
    for issue, count in summary["warning_counts"]:
        lines.append(f"- `{issue}`: {count}")
    lines.extend(["", "## Flagged Samples", ""])
    for result in results:
        if not result["ok"]:
            lines.append(f"### {Path(result['file']).name} #{result['index']} target=1")
            lines.append(f"- skill: `{result['skill_id']}`")
            lines.append(f"- task_space: `{result['task_space_id']}`")
            lines.append(f"- hard issues: {result['hard_issues']}")
            lines.append(f"- quality warnings: {result['quality_warnings']}")
            lines.append(f"- quality score: {result['quality_score']}")
            lines.append(f"- query: {result['query']}")
            lines.append("")

    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in [
        "rows",
        "ok_rows",
        "hard_ok_rows",
        "quality_ok_rows",
        "ok_rate",
        "hard_ok_rate",
        "quality_ok_rate",
        "avg_quality_score",
        "hard_issue_counts",
        "warning_counts",
    ]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
