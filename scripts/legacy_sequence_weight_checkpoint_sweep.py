#!/usr/bin/env python3
"""Evaluate and select every checkpoint in the legacy prefix-weight sweep."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from legacy_core75_checkpoint_oracle import (
        _checkpoint_state_summary,
        _claim_ready,
        _copy_tree_atomic,
        _evaluate_checkpoint,
        _file_sha256,
        _metric_recall,
        _safe_remove_tree,
        _step_name,
        _subset_exact,
        _write_json_atomic,
        oracle_eval_steps,
        oracle_selection_key,
    )
except ModuleNotFoundError:
    from scripts.legacy_core75_checkpoint_oracle import (
        _checkpoint_state_summary,
        _claim_ready,
        _copy_tree_atomic,
        _evaluate_checkpoint,
        _file_sha256,
        _metric_recall,
        _safe_remove_tree,
        _step_name,
        _subset_exact,
        _write_json_atomic,
        oracle_eval_steps,
        oracle_selection_key,
    )


DEFAULT_JOINT_MIN_R10 = 0.69
DEFAULT_JOINT_MIN_R50 = 0.75
DEFAULT_NEW_ARMS = (
    "cum_002",
    "seq_001",
    "seq_002",
    "seq_004",
    "seq_006",
    "seq_008",
    "seq_010",
)


def joint_eligible(result: dict[str, Any], *, min_r10: float, min_r50: float) -> bool:
    return (
        _metric_recall(result, 10) >= min_r10 and _metric_recall(result, 50) >= min_r50
    )


def select_best(
    results: list[dict[str, Any]],
    *,
    min_r10: float | None = None,
    min_r50: float | None = None,
) -> dict[str, Any] | None:
    candidates = results
    if min_r10 is not None or min_r50 is not None:
        if min_r10 is None or min_r50 is None:
            raise ValueError("joint selection requires both R@10 and R@50 thresholds")
        candidates = [
            result
            for result in results
            if joint_eligible(result, min_r10=min_r10, min_r50=min_r50)
        ]
    return max(candidates, key=oracle_selection_key) if candidates else None


def _hardlink_tree_atomic(source: Path, destination: Path) -> None:
    """Clone a retained checkpoint on the same filesystem without duplicating bytes."""
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite retained checkpoint: {destination}"
        )
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary retained checkpoint exists: {temporary}")
    try:
        shutil.copytree(
            source,
            temporary,
            symlinks=True,
            copy_function=os.link,
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _replace_retained(
    candidate: Path,
    *,
    top1_path: Path,
    joint_path: Path,
    improve_top1: bool,
    improve_joint: bool,
    queue_root: Path,
) -> None:
    if not improve_top1 and not improve_joint:
        _safe_remove_tree(candidate, queue_root)
        return
    if improve_top1 and top1_path.exists():
        _safe_remove_tree(top1_path, queue_root)
    if improve_joint and joint_path.exists():
        _safe_remove_tree(joint_path, queue_root)
    if improve_top1 and improve_joint:
        os.replace(candidate, top1_path)
        _hardlink_tree_atomic(top1_path, joint_path)
    elif improve_top1:
        os.replace(candidate, top1_path)
    else:
        os.replace(candidate, joint_path)


def _candidate_record(
    *, step: int, key: tuple[float, float, float, int], path: Path
) -> dict[str, Any]:
    return {"step": step, "selection_key": list(key), "checkpoint": str(path)}


def _worker_payload(
    *,
    args: argparse.Namespace,
    processed: list[int],
    top1_record: dict[str, Any] | None,
    joint_record: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "arm": args.arm,
        "worker_id": args.worker_id,
        "processed_steps": processed,
        "top1_best": top1_record,
        "joint_best": joint_record,
        "joint_thresholds": {
            "r10": args.joint_min_r10,
            "r50": args.joint_min_r50,
        },
    }


def command_worker(args: argparse.Namespace) -> None:
    args.queue_root = args.queue_root.resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_root = args.output_root / "workers" / f"worker_{args.worker_id:02d}"
    worker_root.mkdir(parents=True, exist_ok=False)
    top1_path = args.queue_root / f"worker_{args.worker_id:02d}_top1_best"
    joint_path = args.queue_root / f"worker_{args.worker_id:02d}_joint_best"
    top1_key: tuple[float, float, float, int] | None = None
    joint_key: tuple[float, float, float, int] | None = None
    top1_record: dict[str, Any] | None = None
    joint_record: dict[str, Any] | None = None
    processed: list[int] = []
    current_claim: Path | None = None
    try:
        while True:
            current_claim = _claim_ready(args.queue_root, args.worker_id)
            if current_claim is None:
                if (args.queue_root / "producer.failed.json").is_file():
                    raise RuntimeError("checkpoint producer failed")
                if (args.queue_root / "producer.done").is_file():
                    break
                time.sleep(args.poll_seconds)
                continue
            ready = json.loads(current_claim.read_text(encoding="utf-8"))
            result = _evaluate_checkpoint(args, ready)
            result["artifact_format"] = "legacy_sequence_weight_checkpoint_step_v1"
            result["arm"] = args.arm
            step = int(result["step"])
            result_path = (
                args.output_root / "steps" / _step_name(step) / "oracle_result.json"
            )
            _write_json_atomic(result_path, result)
            key = oracle_selection_key(result)
            improve_top1 = top1_key is None or key > top1_key
            eligible = joint_eligible(
                result,
                min_r10=args.joint_min_r10,
                min_r50=args.joint_min_r50,
            )
            improve_joint = eligible and (joint_key is None or key > joint_key)
            candidate = Path(ready["checkpoint_dir"])
            _replace_retained(
                candidate,
                top1_path=top1_path,
                joint_path=joint_path,
                improve_top1=improve_top1,
                improve_joint=improve_joint,
                queue_root=args.queue_root,
            )
            if improve_top1:
                top1_key = key
                top1_record = _candidate_record(step=step, key=key, path=top1_path)
            if improve_joint:
                joint_key = key
                joint_record = _candidate_record(step=step, key=key, path=joint_path)

            processed.append(step)
            current_claim.unlink()
            current_claim = None
            payload = _worker_payload(
                args=args,
                processed=processed,
                top1_record=top1_record,
                joint_record=joint_record,
            )
            _write_json_atomic(worker_root / "progress.json", payload)
            print(
                f"sweep_worker={args.worker_id} arm={args.arm} step={step} "
                f"r1={_metric_recall(result, 1):.9f} "
                f"top1_best={top1_record['step'] if top1_record else None} "
                f"joint_best={joint_record['step'] if joint_record else None}",
                flush=True,
            )
    except BaseException as exc:
        failure = {
            "arm": args.arm,
            "worker_id": args.worker_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claim": str(current_claim) if current_claim is not None else None,
            "processed_steps": processed,
        }
        _write_json_atomic(worker_root / "failed.json", failure)
        _write_json_atomic(args.queue_root / "worker.failed.json", failure)
        raise

    if top1_record is None or not top1_path.is_dir():
        raise RuntimeError(f"worker {args.worker_id} processed no checkpoints")
    _write_json_atomic(
        worker_root / "done.json",
        _worker_payload(
            args=args,
            processed=processed,
            top1_record=top1_record,
            joint_record=joint_record,
        ),
    )


def _load_results(
    arm_root: Path, expected_steps: list[int], arm: str
) -> list[dict[str, Any]]:
    paths = sorted((arm_root / "steps").glob("step_*/oracle_result.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_step = {int(result["step"]): result for result in results}
    if len(by_step) != len(results):
        raise ValueError(f"duplicate checkpoint result for arm {arm}")
    if sorted(by_step) != expected_steps:
        raise ValueError(
            f"checkpoint coverage mismatch for {arm}: expected={len(expected_steps)}, "
            f"observed={len(by_step)}"
        )
    if any(result.get("arm") != arm for result in results):
        raise ValueError(f"result arm mismatch for {arm}")
    return results


def _retained_source(
    workers: list[dict[str, Any]], *, selection: str, selected_step: int
) -> Path:
    matches = [
        worker[selection]
        for worker in workers
        if worker.get(selection) is not None
        and int(worker[selection]["step"]) == selected_step
    ]
    if len(matches) != 1:
        raise ValueError(
            f"selected {selection} step={selected_step} is retained by {len(matches)} workers"
        )
    path = Path(matches[0]["checkpoint"])
    if not path.is_dir():
        raise FileNotFoundError(f"retained checkpoint is missing: {path}")
    return path


def _selection_payload(
    *,
    selected: dict[str, Any],
    checkpoint: dict[str, Any],
    thresholds: dict[str, float] | None,
) -> dict[str, Any]:
    return {
        **selected,
        "selection_key": list(oracle_selection_key(selected)),
        "thresholds": thresholds,
        "checkpoint": checkpoint,
    }


def _historical_anchor(
    *,
    current_arm: dict[str, Any],
    current_loss_history: Path,
    results_by_step: dict[int, dict[str, Any]],
    historical_root: Path,
) -> dict[str, Any]:
    historical_arm_path = historical_root / "full_query_arm.json"
    historical_result_path = historical_root / "result_summary.json"
    historical_loss_history = historical_root / "train" / "loss_history.jsonl"
    historical_arm = json.loads(historical_arm_path.read_text(encoding="utf-8"))
    historical_result = json.loads(historical_result_path.read_text(encoding="utf-8"))
    step = int(historical_result["training"]["query_best_step"])
    result = results_by_step[step]
    checks = {
        "loss_history_sha256": _file_sha256(current_loss_history)
        == _file_sha256(historical_loss_history),
        "best_step": int(current_arm["training"]["history"]["best_step"]) == step,
        "best_dev_loss": float(current_arm["training"]["history"]["best_dev_loss"])
        == float(historical_result["training"]["query_best_dev_loss"]),
        "final_model_state": current_arm["training"]["final_live_model_state_sha256"]
        == historical_result["training"]["final_model_state_sha256"],
        "prediction": result["predictions_sha256"]
        == historical_result["evaluation"]["predictions_sha256"],
        "overall_metrics": _subset_exact(
            result["overall"], historical_result["overall"]
        ),
        "single_metrics": _subset_exact(result["single"], historical_result["single"]),
        "multi_metrics": _subset_exact(result["multi"], historical_result["multi"]),
        "catalog_boundary": current_arm["catalog_boundary"]
        == historical_arm["catalog_boundary"],
        "query_step_1": current_arm["query_step_1"] == historical_arm["query_step_1"],
    }
    if not all(checks.values()):
        raise ValueError(f"historical anchor failed: {checks}")
    return {
        "root": str(historical_root),
        "artifact_sha256": {
            "full_query_arm": _file_sha256(historical_arm_path),
            "result_summary": _file_sha256(historical_result_path),
            "loss_history": _file_sha256(historical_loss_history),
        },
        "step": step,
        "checks": checks,
        "passed": True,
    }


def command_analyze_arm(args: argparse.Namespace) -> None:
    expected_steps = oracle_eval_steps(args.query_steps, args.eval_steps)
    results = _load_results(args.arm_root, expected_steps, args.arm)
    results_by_step = {int(result["step"]): result for result in results}
    loss_history = args.arm_root / "train" / "loss_history.jsonl"
    loss_events = [
        json.loads(line)
        for line in loss_history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev_by_step = {
        int(row["step"]): float(row["dev_loss"])
        for row in loss_events
        if row.get("stage") == "query" and row.get("event") == "eval"
    }
    if sorted(dev_by_step) != expected_steps:
        raise ValueError(f"dev-loss coverage mismatch for {args.arm}")
    if any(
        float(results_by_step[step]["dev_loss"]) != dev_by_step[step]
        for step in expected_steps
    ):
        raise ValueError(f"checkpoint/dev-loss alignment mismatch for {args.arm}")

    worker_paths = sorted((args.arm_root / "workers").glob("worker_*/done.json"))
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in worker_paths]
    if len(workers) != args.worker_count:
        raise ValueError(
            f"worker count mismatch for {args.arm}: expected={args.worker_count}, "
            f"observed={len(workers)}"
        )
    processed = sorted(step for worker in workers for step in worker["processed_steps"])
    if processed != expected_steps:
        raise ValueError(
            f"workers do not cover checkpoint steps exactly for {args.arm}"
        )

    top1 = select_best(results)
    joint = select_best(
        results,
        min_r10=args.joint_min_r10,
        min_r50=args.joint_min_r50,
    )
    if top1 is None:
        raise RuntimeError(f"no top1 selection for {args.arm}")

    top1_source = _retained_source(
        workers, selection="top1_best", selected_step=int(top1["step"])
    )
    top1_dir = args.arm_root / "top1_best"
    _copy_tree_atomic(top1_source, top1_dir)
    shutil.copy2(
        args.arm_root / "full_query_arm.json", top1_dir / "training_provenance.json"
    )
    top1_checkpoint = _checkpoint_state_summary(top1_dir)
    if int(top1_checkpoint["oracle_step"]) != int(top1["step"]):
        raise ValueError(f"top1 checkpoint step mismatch for {args.arm}")

    joint_payload = None
    if joint is not None:
        joint_source = _retained_source(
            workers, selection="joint_best", selected_step=int(joint["step"])
        )
        joint_dir = args.arm_root / "joint_best"
        _copy_tree_atomic(joint_source, joint_dir)
        shutil.copy2(
            args.arm_root / "full_query_arm.json",
            joint_dir / "training_provenance.json",
        )
        joint_checkpoint = _checkpoint_state_summary(joint_dir)
        if int(joint_checkpoint["oracle_step"]) != int(joint["step"]):
            raise ValueError(f"joint checkpoint step mismatch for {args.arm}")
        joint_payload = _selection_payload(
            selected=joint,
            checkpoint=joint_checkpoint,
            thresholds={"r10": args.joint_min_r10, "r50": args.joint_min_r50},
        )

    training_arm = json.loads(
        (args.arm_root / "full_query_arm.json").read_text(encoding="utf-8")
    )
    condition = training_arm["condition"]
    prefix = condition["prefix_ranking"]
    contract_checks = {
        "arm": condition["arm"] == args.arm,
        "scope": prefix["scope"] == args.scope,
        "weight": float(prefix["weight"]) == args.weight,
        "grouping": prefix["grouping"] == "fixed",
        "catalog_steps": int(condition["catalog_steps"]) == 2730,
        "query_steps": int(condition["query_steps"]) == args.query_steps,
        "eval_steps": int(condition["eval_steps"]) == args.eval_steps,
        "batch_size": int(condition["batch_size"]) == 16,
        "seed": int(condition["seed"]) == 20260626,
        "query_lr": float(condition["query_lr"]) == 0.0003,
        "dropout": float(condition["dropout_rate"]) == 0.1,
        "workspace": condition["requested_controls"]["cublas_workspace_config"]
        == ":16:8",
    }
    if not all(contract_checks.values()):
        raise ValueError(f"arm contract failed for {args.arm}: {contract_checks}")
    historical = (
        _historical_anchor(
            current_arm=training_arm,
            current_loss_history=loss_history,
            results_by_step=results_by_step,
            historical_root=args.historical_arm_root,
        )
        if args.historical_arm_root is not None
        else None
    )

    ordered = sorted(results, key=oracle_selection_key, reverse=True)
    summary = {
        "artifact_format": "legacy_sequence_weight_checkpoint_arm_summary_v1",
        "arm": args.arm,
        "scope": args.scope,
        "weight": args.weight,
        "protocol": {
            "training": {
                "catalog_steps": 2730,
                "query_steps": args.query_steps,
                "eval_steps": args.eval_steps,
                "eval_point_count": len(expected_steps),
                "batch_size": 16,
                "seed": 20260626,
                "lr": 0.0003,
                "dropout": 0.1,
                "outer_precision": "bf16",
                "cublas_workspace_config": ":16:8",
            },
            "evaluation": "Official core75=75 queries, beam=50, beam_order, CE log-prob",
            "status": "test_selected_performance_search",
        },
        "contract": {"checks": contract_checks, "passed": True},
        "coverage": {
            "expected_steps": expected_steps,
            "result_count": len(results),
            "worker_count": len(workers),
            "passed": True,
        },
        "top1_best": _selection_payload(
            selected=top1,
            checkpoint=top1_checkpoint,
            thresholds=None,
        ),
        "joint_best": joint_payload,
        "historical_anchor": historical,
        "top10": [
            {
                "step": int(result["step"]),
                "dev_loss": float(result["dev_loss"]),
                "r1": _metric_recall(result, 1),
                "r10": _metric_recall(result, 10),
                "r50": _metric_recall(result, 50),
            }
            for result in ordered[:10]
        ],
        "results": results,
    }
    _write_json_atomic(args.arm_root / "checkpoint_sweep_summary.json", summary)
    (args.arm_root / "arm.done").touch()
    print(
        json.dumps(
            {
                "arm": args.arm,
                "top1_step": int(top1["step"]),
                "top1_r1": _metric_recall(top1, 1),
                "joint_step": int(joint["step"]) if joint is not None else None,
                "joint_r1": _metric_recall(joint, 1) if joint is not None else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _selection_source(selection: dict[str, Any]) -> Path:
    return Path(selection["checkpoint"]["path"]).parent


def _matrix_winner(arms: list[dict[str, Any]], selection: str) -> dict[str, Any]:
    candidates = [arm for arm in arms if arm.get(selection) is not None]
    if not candidates:
        raise RuntimeError(f"no matrix candidate for {selection}")
    order = {arm["arm"]: index for index, arm in enumerate(arms)}
    return max(
        candidates,
        key=lambda arm: (*oracle_selection_key(arm[selection]), -order[arm["arm"]]),
    )


def command_analyze_matrix(args: argparse.Namespace) -> None:
    control_summary_path = args.control_root / "checkpoint_oracle_summary.json"
    control_summary = json.loads(control_summary_path.read_text(encoding="utf-8"))
    control_selected = control_summary["selected"]
    if not joint_eligible(
        control_selected,
        min_r10=args.joint_min_r10,
        min_r50=args.joint_min_r50,
    ):
        raise ValueError("reused control top1-best does not satisfy joint thresholds")
    if not (
        control_summary["coverage"]["passed"]
        and control_summary["anchor"]["passed"]
        and control_summary["training_invariant"]["passed"]
    ):
        raise ValueError("reused control oracle failed its recorded invariants")

    arms: list[dict[str, Any]] = [
        {
            "arm": "control",
            "scope": "disabled",
            "weight": 0.0,
            "source_summary": str(control_summary_path),
            "source_summary_sha256": _file_sha256(control_summary_path),
            "top1_best": control_selected,
            "joint_best": control_selected,
            "historical_anchor_passed": True,
        }
    ]
    for arm_name in args.expected_new_arms.split(","):
        path = args.root / "arms" / arm_name / "checkpoint_sweep_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["arm"] != arm_name or not summary["coverage"]["passed"]:
            raise ValueError(f"invalid arm summary: {path}")
        if arm_name in {"cum_002", "seq_010"} and not (
            summary.get("historical_anchor") and summary["historical_anchor"]["passed"]
        ):
            raise ValueError(f"required historical anchor did not pass for {arm_name}")
        arms.append(
            {
                "arm": arm_name,
                "scope": summary["scope"],
                "weight": summary["weight"],
                "source_summary": str(path),
                "source_summary_sha256": _file_sha256(path),
                "top1_best": summary["top1_best"],
                "joint_best": summary["joint_best"],
                "historical_anchor_passed": (
                    summary["historical_anchor"]["passed"]
                    if summary.get("historical_anchor")
                    else None
                ),
            }
        )

    top1_arm = _matrix_winner(arms, "top1_best")
    joint_arm = _matrix_winner(arms, "joint_best")
    selected_root = args.root / "selected"
    selected_root.mkdir(parents=True, exist_ok=False)
    top1_dir = selected_root / "top1_best"
    joint_dir = selected_root / "joint_best"
    _copy_tree_atomic(_selection_source(top1_arm["top1_best"]), top1_dir)
    _copy_tree_atomic(_selection_source(joint_arm["joint_best"]), joint_dir)

    top1_checkpoint = _checkpoint_state_summary(top1_dir)
    joint_checkpoint = _checkpoint_state_summary(joint_dir)
    summary = {
        "artifact_format": "legacy_sequence_weight_checkpoint_sweep_summary_v1",
        "protocol": {
            "evaluation": "Official core75=75 queries, beam=50, beam_order, CE log-prob",
            "checkpoint_points_per_new_arm": 195,
            "top1_selection": "maximize Overall R@1, R@10, R@50, then prefer earlier step",
            "joint_selection": {
                "minimum_r10": args.joint_min_r10,
                "minimum_r50": args.joint_min_r50,
                "eligible_ranking": "maximize Overall R@1, R@10, R@50, then prefer earlier step",
            },
            "status": "test_selected_performance_search",
        },
        "control_oracle_sha256": _file_sha256(control_summary_path),
        "arms": arms,
        "selected": {
            "top1_best": {
                "arm": top1_arm["arm"],
                "result": top1_arm["top1_best"],
                "checkpoint": top1_checkpoint,
            },
            "joint_best": {
                "arm": joint_arm["arm"],
                "result": joint_arm["joint_best"],
                "checkpoint": joint_checkpoint,
            },
        },
    }
    _write_json_atomic(
        args.root / "sequence_weight_checkpoint_sweep_summary.json", summary
    )
    (args.root / "sequence_weight_checkpoint_sweep.done").touch()
    print(
        json.dumps(
            {
                "top1_arm": top1_arm["arm"],
                "top1_step": top1_arm["top1_best"]["step"],
                "top1_r1": _metric_recall(top1_arm["top1_best"], 1),
                "joint_arm": joint_arm["arm"],
                "joint_step": joint_arm["joint_best"]["step"],
                "joint_r1": _metric_recall(joint_arm["joint_best"], 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--arm", required=True)
    worker.add_argument("--worker-id", type=int, required=True)
    worker.add_argument("--queue-root", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--evaluator", type=Path, required=True)
    worker.add_argument("--rows", type=Path, required=True)
    worker.add_argument("--semantic-ids", type=Path, required=True)
    worker.add_argument("--joint-min-r10", type=float, default=DEFAULT_JOINT_MIN_R10)
    worker.add_argument("--joint-min-r50", type=float, default=DEFAULT_JOINT_MIN_R50)
    worker.add_argument("--poll-seconds", type=float, default=0.5)
    worker.set_defaults(func=command_worker)

    arm = subparsers.add_parser("analyze-arm")
    arm.add_argument("--arm", required=True)
    arm.add_argument("--scope", required=True)
    arm.add_argument("--weight", type=float, required=True)
    arm.add_argument("--arm-root", type=Path, required=True)
    arm.add_argument("--historical-arm-root", type=Path)
    arm.add_argument("--query-steps", type=int, default=38805)
    arm.add_argument("--eval-steps", type=int, default=200)
    arm.add_argument("--worker-count", type=int, default=7)
    arm.add_argument("--joint-min-r10", type=float, default=DEFAULT_JOINT_MIN_R10)
    arm.add_argument("--joint-min-r50", type=float, default=DEFAULT_JOINT_MIN_R50)
    arm.set_defaults(func=command_analyze_arm)

    matrix = subparsers.add_parser("analyze-matrix")
    matrix.add_argument("--root", type=Path, required=True)
    matrix.add_argument("--control-root", type=Path, required=True)
    matrix.add_argument("--expected-new-arms", default=",".join(DEFAULT_NEW_ARMS))
    matrix.add_argument("--joint-min-r10", type=float, default=DEFAULT_JOINT_MIN_R10)
    matrix.add_argument("--joint-min-r50", type=float, default=DEFAULT_JOINT_MIN_R50)
    matrix.set_defaults(func=command_analyze_matrix)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
