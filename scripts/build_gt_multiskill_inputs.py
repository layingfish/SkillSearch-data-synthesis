#!/usr/bin/env python3
"""Build GT-anchored multi-skill replay inputs for second-stage synthesis.

The script uses official SkillRouter gold skill sets as composition anchors,
but it does not copy benchmark queries into the replay prompt. Each output row
contains only skill ids plus lightweight provenance and generation guidance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


VARIANT_GUIDANCE = [
    (
        "project_request",
        "Write a new project request from a user who needs these capabilities together. "
        "Use concrete inputs and expected outputs, but do not copy or paraphrase any benchmark task wording.",
    ),
    (
        "debug_or_repair",
        "Write a debugging, repair, or improvement request where these capabilities are jointly needed. "
        "Make the failure mode or quality target concrete, without copying benchmark task wording.",
    ),
    (
        "evaluation_or_reporting",
        "Write an evaluation, validation, or reporting request where the selected skills each provide a necessary part of the solution. "
        "Use a different scenario from the official benchmark query.",
    ),
]


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


def skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "")


def load_skill_ids(path: Path) -> set[str]:
    return {sid for row in load_jsonl(path) if (sid := skill_id(row))}


def task_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id") or row.get("task_id") or row.get("id") or "")


def fingerprint(items: Iterable[str]) -> str:
    payload = "\n".join(sorted(items)).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def combo_type(gold_ids: list[str], combo: tuple[str, ...]) -> str:
    if len(combo) == len(gold_ids) and set(combo) == set(gold_ids):
        return "full_official_gt_set"
    return "official_gt_subset"


def iter_task_combos(
    row: dict[str, Any],
    *,
    min_k: int,
    max_k: int,
    policy: str,
) -> Iterable[tuple[str, ...]]:
    gold_ids = [str(sid) for sid in (row.get("gold_skill_ids") or []) if sid]
    if len(gold_ids) < min_k:
        return
    upper = min(max_k, len(gold_ids))
    if policy == "full_or_windows":
        if min_k <= len(gold_ids) <= max_k:
            yield tuple(gold_ids)
            return
        for k in range(min_k, upper + 1):
            for start in range(0, len(gold_ids) - k + 1):
                yield tuple(gold_ids[start : start + k])
        return
    if policy != "all_subsets":
        raise ValueError(f"unknown combo policy: {policy}")
    for k in range(min_k, upper + 1):
        for combo in itertools.combinations(gold_ids, k):
            yield tuple(combo)


def make_generation_guidance(row: dict[str, Any], variant_name: str, variant_text: str) -> str:
    domain = str(row.get("domain") or "").strip()
    difficulty = str(row.get("difficulty") or "").strip()
    tags = row.get("tags") or []
    tag_text = ", ".join(str(tag) for tag in tags[:4])
    context_bits = []
    if domain:
        context_bits.append(f"Domain hint: {domain}.")
    if difficulty:
        context_bits.append(f"Difficulty hint: {difficulty}.")
    if tag_text:
        context_bits.append(f"Optional topic hints: {tag_text}.")
    context = " ".join(context_bits)
    return (
        f"{variant_text} {context} "
        "The task description must be an independent synthetic user query. "
        "Do not mention SkillRouter, gold skills, candidate panels, skill ids, selected skills, or benchmark metadata."
    ).strip()


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qrels = load_jsonl(args.qrels)
    available_skill_ids = load_skill_ids(args.skills)
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    missing: Counter[str] = Counter()
    combo_seen: set[tuple[str, str, tuple[str, ...], int]] = set()

    for qrel in qrels:
        tid = task_id(qrel)
        gold_ids = [str(sid) for sid in (qrel.get("gold_skill_ids") or []) if sid]
        absent = [sid for sid in gold_ids if sid not in available_skill_ids]
        if absent:
            missing.update(absent)
            continue
        if len(gold_ids) < args.min_k:
            continue
        combos = list(iter_task_combos(qrel, min_k=args.min_k, max_k=args.max_k, policy=args.combo_policy))
        if args.max_combos_per_task > 0 and len(combos) > args.max_combos_per_task:
            rng.shuffle(combos)
            combos = combos[: args.max_combos_per_task]
        for combo in combos:
            sorted_combo = tuple(sorted(combo))
            for variant_index in range(args.variants_per_combo):
                variant_name, variant_text = VARIANT_GUIDANCE[variant_index % len(VARIANT_GUIDANCE)]
                seen_key = (tid, variant_name, sorted_combo, variant_index)
                if seen_key in combo_seen:
                    continue
                combo_seen.add(seen_key)
                rows.append(
                    {
                        "index": len(rows) + 1,
                        "target_skill_count": len(combo),
                        "candidate_skill_ids": list(combo),
                        "task_space_id": f"skillrouter_gt/{qrel.get('domain') or 'unknown'}",
                        "anchor_task_object": tid,
                        "panel_fingerprint": fingerprint(combo),
                        "panel_source_tier": "skillrouter_gt_multiskill_v1",
                        "generation_guidance": make_generation_guidance(qrel, variant_name, variant_text),
                        "sampling": {
                            "source": "build_gt_multiskill_inputs",
                            "seed": args.seed,
                            "combo_policy": args.combo_policy,
                            "variant_name": variant_name,
                            "variant_index": variant_index,
                        },
                        "official_task": {
                            "query_id": tid,
                            "domain": qrel.get("domain"),
                            "difficulty": qrel.get("difficulty"),
                            "num_skills": qrel.get("num_skills") or len(gold_ids),
                            "combo_type": combo_type(gold_ids, combo),
                            "gold_skill_count": len(gold_ids),
                            "tags": qrel.get("tags") or [],
                        },
                    }
                )

    if args.shuffle:
        rng.shuffle(rows)
        for index, row in enumerate(rows, 1):
            row["index"] = index

    if args.total > 0 and len(rows) > args.total:
        rows = rows[: args.total]
        for index, row in enumerate(rows, 1):
            row["index"] = index

    summary = {
        "qrels": args.qrels.as_posix(),
        "skills": args.skills.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "seed": args.seed,
        "combo_policy": args.combo_policy,
        "variants_per_combo": args.variants_per_combo,
        "total_limit": args.total,
        "rows": len(rows),
        "by_target_skill_count": dict(sorted(Counter(row["target_skill_count"] for row in rows).items())),
        "by_combo_type": dict(sorted(Counter(row["official_task"]["combo_type"] for row in rows).items())),
        "by_variant": dict(sorted(Counter(row["sampling"]["variant_name"] for row in rows).items())),
        "covered_official_tasks": len({row["official_task"]["query_id"] for row in rows}),
        "covered_gt_skills": len({sid for row in rows for sid in row["candidate_skill_ids"]}),
        "missing_gt_skills": dict(missing.most_common()),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, default=Path("/root/workplace/SkillSearch/data/skillrouter_eval/all_qrels.jsonl"))
    parser.add_argument("--skills", type=Path, default=Path("outputs/eval/panel_classification_v2/easy_depth3_panel_classified.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--min-k", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--max-k", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--combo-policy", choices=["all_subsets", "full_or_windows"], default="all_subsets")
    parser.add_argument("--variants-per-combo", type=int, default=3)
    parser.add_argument("--max-combos-per-task", type=int, default=0)
    parser.add_argument("--total", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    if args.min_k > args.max_k:
        raise SystemExit("--min-k must be <= --max-k")
    if args.variants_per_combo < 1:
        raise SystemExit("--variants-per-combo must be >= 1")

    rows, summary = build_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_k: dict[int, list[dict[str, Any]]] = {2: [], 3: [], 4: []}
    for row in rows:
        by_k[int(row["target_skill_count"])].append(row)
    outputs: dict[str, dict[str, Any]] = {}
    for k in range(args.min_k, args.max_k + 1):
        output = args.output_dir / f"input_{k}skill.jsonl"
        written = write_jsonl(output, by_k.get(k, []))
        outputs[str(k)] = {"path": output.as_posix(), "written": written}
        print(json.dumps({"k": k, "written": written, "output": output.as_posix()}, ensure_ascii=False))

    summary["outputs"] = outputs
    summary_path = args.summary or (args.output_dir / "gt_multiskill_inputs_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
