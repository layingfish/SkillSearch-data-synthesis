#!/usr/bin/env python3
"""Re-select best-of-N multi-skill candidates with the current scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_multiskill_probe import candidate_sort_key, load_jsonl, load_skill_lookup, row_candidate_skill_ids, score_generated


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--panels", required=True, type=Path)
    parser.add_argument("--skills", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-skill-count", type=int, default=4)
    parser.add_argument("--max-skills", type=int, default=15)
    parser.add_argument("--strict-candidate-filter", action="store_true")
    parser.add_argument("--drop-non-ok", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    panels = load_jsonl(args.panels)
    panel_by_key = {
        (panel.get("task_space_id"), panel.get("anchor_task_object")): panel
        for panel in panels
    }
    lookup = load_skill_lookup(args.skills)

    rescored_rows: list[dict[str, Any]] = []
    for row in rows:
        panel = panel_by_key.get((row.get("task_space_id"), row.get("anchor_task_object")))
        if not panel or not row.get("candidates"):
            continue
        rescored_candidates: list[dict[str, Any]] = []
        for candidate in row["candidates"]:
            candidate = dict(candidate)
            candidate["evaluation"] = score_generated(
                candidate.get("generated") or {},
                panel,
                lookup,
                args.max_skills,
                args.target_skill_count,
                args.strict_candidate_filter,
            )
            rescored_candidates.append(candidate)
        rescored_candidates.sort(key=candidate_sort_key, reverse=True)
        best = rescored_candidates[0]
        best_eval = best.get("evaluation", {})
        if args.drop_non_ok and (
            best_eval.get("hard_issues") or best_eval.get("quality_warnings")
        ):
            continue

        new_row = dict(row)
        new_row["generated"] = best.get("generated")
        new_row["usage"] = best.get("usage")
        new_row["model"] = best.get("model", row.get("model"))
        new_row["chosen_candidate_index"] = best.get("candidate_index")
        new_row["chosen_evaluation"] = best_eval
        new_row["candidate_skill_ids"] = row_candidate_skill_ids(panel, lookup, argparse.Namespace(
            max_skills=args.max_skills,
            target_skill_count=args.target_skill_count,
            strict_candidate_filter=args.strict_candidate_filter,
        ))
        new_row["candidate_summaries"] = [
            {
                "candidate_index": candidate.get("candidate_index"),
                "score": candidate.get("evaluation", {}).get("score"),
                "hard_issues": candidate.get("evaluation", {}).get("hard_issues"),
                "quality_warnings": candidate.get("evaluation", {}).get("quality_warnings"),
                "selected_skill_ids": (candidate.get("generated") or {}).get("selected_skill_ids"),
                "query": (candidate.get("generated") or {}).get("query"),
                "usage": candidate.get("usage"),
                "error": candidate.get("error"),
            }
            for candidate in rescored_candidates
        ]
        new_row["candidates"] = rescored_candidates
        rescored_rows.append(new_row)
        print(json.dumps({
            "index": new_row.get("index"),
            "task_space_id": new_row.get("task_space_id"),
            "selected_skill_ids": (new_row.get("generated") or {}).get("selected_skill_ids"),
            "score": best_eval.get("score"),
            "hard_issues": best_eval.get("hard_issues"),
            "quality_warnings": best_eval.get("quality_warnings"),
            "chosen_candidate_index": best.get("candidate_index"),
        }, ensure_ascii=False))

    write_jsonl(args.output, rescored_rows)


if __name__ == "__main__":
    main()
