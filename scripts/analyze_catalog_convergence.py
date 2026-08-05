#!/usr/bin/env python3
"""Select the first catalog checkpoint with a stable teacher-forced plateau."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_catalog_convergence(
    checkpoints: list[dict[str, Any]],
    *,
    checkpoint_interval: int,
    ce_relative_threshold: float,
    sid_exact_absolute_threshold: float,
    consecutive_stable_intervals: int,
) -> dict[str, Any]:
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if consecutive_stable_intervals <= 0:
        raise ValueError("consecutive stable intervals must be positive")
    if ce_relative_threshold < 0 or sid_exact_absolute_threshold < 0:
        raise ValueError("convergence thresholds must be non-negative")

    normalized = [
        {
            "epoch": int(row["epoch"]),
            "teacher_forced_ce": float(row["teacher_forced_ce"]),
            "teacher_forced_sid_exact_accuracy": float(
                row["teacher_forced_sid_exact_accuracy"]
            ),
        }
        for row in checkpoints
    ]
    epochs = [row["epoch"] for row in normalized]
    if epochs != sorted(set(epochs)):
        raise ValueError("checkpoint epochs must be unique and strictly increasing")
    if any(epoch <= 0 or epoch % checkpoint_interval != 0 for epoch in epochs):
        raise ValueError("checkpoint epochs must be positive multiples of the interval")
    if any(
        current - previous != checkpoint_interval
        for previous, current in zip(epochs, epochs[1:])
    ):
        raise ValueError("checkpoint epochs must be contiguous at the configured interval")

    intervals = []
    for previous, current in zip(normalized, normalized[1:]):
        intervals.append(
            {
                "from_epoch": previous["epoch"],
                "to_epoch": current["epoch"],
                "ce_relative_change": abs(
                    current["teacher_forced_ce"] - previous["teacher_forced_ce"]
                )
                / max(abs(previous["teacher_forced_ce"]), 1e-12),
                "sid_exact_absolute_change": abs(
                    current["teacher_forced_sid_exact_accuracy"]
                    - previous["teacher_forced_sid_exact_accuracy"]
                ),
            }
        )

    selected_epoch = None
    window_size = consecutive_stable_intervals
    for end in range(window_size - 1, len(intervals)):
        window = intervals[end - window_size + 1 : end + 1]
        if all(
            row["ce_relative_change"] < ce_relative_threshold
            and row["sid_exact_absolute_change"] < sid_exact_absolute_threshold
            for row in window
        ):
            selected_epoch = int(window[-1]["to_epoch"])
            break

    return {
        "converged": selected_epoch is not None,
        "selected_epoch": selected_epoch,
        "criterion": {
            "checkpoint_interval_epochs": checkpoint_interval,
            "consecutive_stable_intervals": consecutive_stable_intervals,
            "ce_relative_change_below": ce_relative_threshold,
            "sid_exact_absolute_change_below": sid_exact_absolute_threshold,
        },
        "checkpoints": normalized,
        "intervals": intervals,
    }


def load_checkpoint_metrics(
    metrics_dir: Path,
    checkpoint_interval: int,
    max_epoch: int,
) -> list[dict[str, Any]]:
    if max_epoch <= 0 or max_epoch % checkpoint_interval != 0:
        raise ValueError("max epoch must be a positive multiple of the checkpoint interval")
    checkpoints = []
    for epoch in range(checkpoint_interval, max_epoch + 1, checkpoint_interval):
        path = metrics_dir / f"epoch{epoch}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoints.append(
            {
                "epoch": epoch,
                "teacher_forced_ce": payload["teacher_forced_ce"],
                "teacher_forced_sid_exact_accuracy": payload[
                    "teacher_forced_sid_exact_accuracy"
                ],
            }
        )
    return checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--max-epoch", type=int, required=True)
    parser.add_argument("--ce-relative-threshold", type=float, required=True)
    parser.add_argument("--sid-exact-absolute-threshold", type=float, required=True)
    parser.add_argument("--consecutive-stable-intervals", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoints = load_checkpoint_metrics(
        args.metrics_dir,
        args.checkpoint_interval,
        args.max_epoch,
    )
    result = summarize_catalog_convergence(
        checkpoints,
        checkpoint_interval=args.checkpoint_interval,
        ce_relative_threshold=args.ce_relative_threshold,
        sid_exact_absolute_threshold=args.sid_exact_absolute_threshold,
        consecutive_stable_intervals=args.consecutive_stable_intervals,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
