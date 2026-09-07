#!/usr/bin/env python3
"""Select one OPQ ceiling candidate from Dev macro Recall only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SELECTION_K = (10, 5, 1, 50)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dev_key(metrics_path: Path) -> tuple[float, ...]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("recall_aggregation") != "macro":
        raise ValueError(f"not macro Recall metrics: {metrics_path}")
    overall = metrics.get("overall")
    if not isinstance(overall, dict):
        raise ValueError(f"missing overall metrics: {metrics_path}")
    values = []
    for k in SELECTION_K:
        record = overall.get(f"@{k}")
        if not isinstance(record, dict) or "macro_recall" not in record:
            raise ValueError(f"missing overall @{k} macro_recall: {metrics_path}")
        value = float(record["macro_recall"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid Recall@{k}={value}: {metrics_path}")
        values.append(value)
    return tuple(values)


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    names = [str(candidate["name"]) for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")

    audited = []
    for order, candidate in enumerate(candidates):
        metrics_path = Path(candidate["metrics_path"])
        checkpoint_path = Path(candidate["checkpoint_path"])
        semantic_ids_path = Path(candidate["semantic_ids_path"])
        for required in (metrics_path, checkpoint_path, semantic_ids_path):
            if not required.is_file():
                raise ValueError(f"candidate artifact is missing: {required}")
        key = load_dev_key(metrics_path)
        audited.append(
            {
                **candidate,
                "order": order,
                "selection_key": list(key),
                "dev_metrics_sha256": file_sha256(metrics_path),
            }
        )

    # Earlier input order is the deterministic final tie-break. The runner
    # supplies candidates from simpler to more specialized architecture.
    winner = max(
        audited,
        key=lambda candidate: (
            *candidate["selection_key"],
            -candidate["order"],
        ),
    )
    return {
        "status": "complete",
        "selection_split": "dev_rows_001",
        "official_metrics_read": False,
        "metric": "macro_recall",
        "selection_order": [f"R@{k}" for k in SELECTION_K],
        "tie_break": "earlier_candidate_order",
        "winner": winner,
        "candidates": audited,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidates = json.loads(args.registry.read_text(encoding="utf-8"))
    if not isinstance(candidates, list):
        raise SystemExit("candidate registry must contain a JSON list")
    result = select_candidate(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
