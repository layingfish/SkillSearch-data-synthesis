#!/usr/bin/env python3
"""Build replay inputs with a configurable bell-shaped skill-count distribution."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import build_balanced_topup_inputs as topup
import generate_single_skill_probe as single


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


def compute_cardinality_quotas(
    *,
    total: int,
    mu: float,
    sigma: float,
    min_k: int = 1,
    max_k: int = 4,
) -> dict[int, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if min_k < 1 or max_k < min_k:
        raise ValueError("invalid k range")

    ks = list(range(min_k, max_k + 1))
    weights = {
        k: math.exp(-0.5 * ((k - mu) / sigma) ** 2)
        for k in ks
    }
    weight_sum = sum(weights.values())
    raw = {k: total * weights[k] / weight_sum for k in ks}
    quotas = {k: int(math.floor(value)) for k, value in raw.items()}
    remaining = total - sum(quotas.values())
    remainders = sorted(
        ((raw[k] - quotas[k], -abs(k - mu), k) for k in ks),
        reverse=True,
    )
    for _, _, k in remainders[:remaining]:
        quotas[k] += 1
    return quotas


def single_input_row(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    return {
        "index": index,
        "skill_id": single.skill_id(row),
        "skill_name": row.get("skill_name") or row.get("name"),
        "task_space_id": row.get("task_space_id"),
        "operation_role": row.get("operation_role"),
        "operation_type": row.get("operation_type"),
        "target_skill_count": 1,
        "sampling": {
            "source": "build_normal_skillrouter_inputs",
            "seed": seed,
            "tier": "single_skill_stratified",
        },
    }


def write_single_skill_inputs(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    limit: int,
    seed: int,
) -> int:
    chosen = single.choose_skills(rows, limit, seed)
    payloads = [
        single_input_row(row, index, seed)
        for index, row in enumerate(chosen, 1)
    ]
    return write_jsonl(output, payloads)


def build_multi_skill_inputs(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    target_skill_count: int,
    limit: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[int, Counter[str]]:
    sample_args = argparse.Namespace(
        target_skill_count=target_skill_count,
        limit=limit,
        seed=seed,
        max_rounds=args.max_rounds,
        group_attempts=args.group_attempts,
        exclude_selected_sets=args.exclude_selected_sets,
        min_task_space_confidence=args.min_task_space_confidence,
        min_operation_confidence=args.min_operation_confidence,
        min_task_object_confidence=args.min_task_object_confidence,
        exclude_flag=args.exclude_flag,
    )
    cards = [topup.card(row) for row in rows if topup.usable(row, sample_args)]
    sampled, skipped = topup.sample_inputs(cards, sample_args)
    written = write_jsonl(output, sampled)
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--mu", type=float, default=2.5)
    parser.add_argument("--sigma", type=float, default=0.9)
    parser.add_argument("--min-k", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--max-k", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--max-rounds", type=int, default=240)
    parser.add_argument("--group-attempts", type=int, default=20)
    parser.add_argument("--exclude-selected-sets", type=Path)
    parser.add_argument("--min-task-space-confidence", type=float, default=0.75)
    parser.add_argument("--min-operation-confidence", type=float, default=0.35)
    parser.add_argument("--min-task-object-confidence", type=float, default=0.45)
    parser.add_argument(
        "--exclude-flag",
        action="append",
        default=sorted(topup.RISK_FLAGS),
        help="quality flag to exclude for multi-skill panel sampling; repeatable",
    )
    args = parser.parse_args()

    quotas = compute_cardinality_quotas(
        total=args.total,
        mu=args.mu,
        sigma=args.sigma,
        min_k=args.min_k,
        max_k=args.max_k,
    )
    rows = load_jsonl(args.skills)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {}
    skipped_by_k: dict[str, dict[str, int]] = {}
    for k in range(args.min_k, args.max_k + 1):
        limit = quotas.get(k, 0)
        output = args.output_dir / f"input_{k}skill.jsonl"
        seed = args.seed + k
        if k == 1:
            written = write_single_skill_inputs(rows, output, limit=limit, seed=seed)
            skipped = Counter()
        else:
            written, skipped = build_multi_skill_inputs(
                rows,
                output,
                target_skill_count=k,
                limit=limit,
                seed=seed,
                args=args,
            )
        outputs[str(k)] = {
            "path": output.as_posix(),
            "quota": limit,
            "written": written,
        }
        skipped_by_k[str(k)] = dict(skipped.most_common())
        print(json.dumps({"k": k, "quota": limit, "written": written, "output": output.as_posix()}))

    summary_path = args.summary or (args.output_dir / "normal_inputs_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "skills": args.skills.as_posix(),
                "output_dir": args.output_dir.as_posix(),
                "total": args.total,
                "mu": args.mu,
                "sigma": args.sigma,
                "min_k": args.min_k,
                "max_k": args.max_k,
                "seed": args.seed,
                "quotas": quotas,
                "outputs": outputs,
                "skipped_by_k": skipped_by_k,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
