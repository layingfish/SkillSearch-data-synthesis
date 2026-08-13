#!/usr/bin/env python3
"""Select sequence-contrastive checkpoints from Official core75 evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from skill_gr_pipeline import file_sha256


K_VALUES = (1, 10, 50)


def read_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("overall", {}).get("rows", 0)) <= 0:
        raise ValueError(f"metrics have no evaluated rows: {path}")
    return payload


def recall_vector(payload: dict[str, Any]) -> dict[str, float]:
    return {
        f"r@{k}": float(payload["overall"][f"@{k}"]["recall"])
        for k in K_VALUES
    }


def load_evaluation(directory: Path) -> dict[str, Any]:
    provenance_path = directory / "checkpoint_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    metrics = {
        subset: read_metrics(directory / f"ranked_metrics_{subset}.json")
        for subset in ("all", "single", "multi")
    }
    expected_rows = {"all": 75, "single": 24, "multi": 51}
    observed_rows = {
        subset: int(payload["overall"]["rows"])
        for subset, payload in metrics.items()
    }
    if observed_rows != expected_rows:
        raise ValueError(
            f"Official core75 subset sizes drifted in {directory}: {observed_rows}"
        )
    return {
        "label": str(provenance["label"]),
        "step": int(provenance["step"]),
        "checkpoint": str(provenance["checkpoint"]),
        "metrics": {subset: recall_vector(payload) for subset, payload in metrics.items()},
        "predictions": str(directory / "predictions.jsonl"),
    }


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        row["metrics"]["single"]["r@1"],
        row["metrics"]["all"]["r@1"],
        row["metrics"]["all"]["r@10"],
        row["metrics"]["all"]["r@50"],
        -int(row["step"]),
    )


def select_rows(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    train_rows = [row for row in rows if row["label"] != "init"]
    if not train_rows:
        raise ValueError("checkpoint sweep has no trained checkpoints")
    single_top1_best = max(train_rows, key=selection_key)
    guarded = [
        row
        for row in train_rows
        if row["metrics"]["all"]["r@10"] >= baseline["metrics"]["all"]["r@10"]
        and row["metrics"]["all"]["r@50"] >= baseline["metrics"]["all"]["r@50"]
    ]
    return {
        "single_top1_best": single_top1_best,
        "deep_recall_guarded_best": max(guarded, key=selection_key) if guarded else baseline,
    }


def metric_delta(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        subset: {
            key: row["metrics"][subset][key] - baseline["metrics"][subset][key]
            for key in ("r@1", "r@10", "r@50")
        }
        for subset in ("all", "single", "multi")
    }


def write_selected_link(selected_root: Path, label: str, checkpoint: Path) -> Path:
    link = selected_root / label
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace selected checkpoint link: {link}")
    link.symlink_to(checkpoint.resolve(strict=True), target_is_directory=True)
    return link


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite summary: {args.output}")
    directories = sorted(
        path for path in args.eval_root.iterdir() if (path / "checkpoint_provenance.json").is_file()
    )
    rows = [load_evaluation(path) for path in directories]
    baseline_matches = [row for row in rows if row["label"] == "init"]
    if len(baseline_matches) != 1:
        raise ValueError(f"expected one init evaluation, got {len(baseline_matches)}")
    baseline = baseline_matches[0]
    expected_baseline = {
        "all": {"r@1": 0.28257142857142853, "r@10": 0.7, "r@50": 0.7511111111111111},
        "single": {"r@1": 0.4166666666666667, "r@10": 0.5, "r@50": 0.5833333333333334},
        "multi": {"r@1": 0.21946778711484594, "r@10": 0.7941176470588235, "r@50": 0.8300653594771242},
    }
    if baseline["metrics"] != expected_baseline:
        raise ValueError(
            "cum_002@28,400 anchor metrics changed under the current evaluator: "
            f"expected={expected_baseline}, observed={baseline['metrics']}"
        )
    selections = select_rows(rows, baseline)
    args.selected_root.mkdir(parents=True, exist_ok=False)
    selected_payload: dict[str, Any] = {}
    for label, row in selections.items():
        checkpoint = Path(row["checkpoint"])
        link = write_selected_link(args.selected_root, label, checkpoint)
        selected_payload[label] = {
            **row,
            "delta_vs_init": metric_delta(row, baseline),
            "checkpoint_sha256": file_sha256(checkpoint / "structured_decoder.pt"),
            "selected_link": str(link),
        }
    summary = {
        "format": "sequence_contrastive_checkpoint_sweep_v1",
        "protocol": {
            "rows": "Official core75",
            "query_count": 75,
            "single_query_count": 24,
            "multi_query_count": 51,
            "num_beams": 50,
            "rank_aggregation": "beam_order",
            "decoder_head": "ce",
            "score_transform": "logprob",
            "k_values": list(K_VALUES),
            "checkpoint_selection_data": "Official core75",
        },
        "baseline": baseline,
        "evaluations": sorted(rows, key=lambda row: row["step"]),
        "selections": selected_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(selected_payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
