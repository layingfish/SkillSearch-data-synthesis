#!/usr/bin/env python3
"""Produce, evaluate, and audit eval-step checkpoints for the legacy core75 oracle."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_HISTORICAL_ARM_SHA256 = (
    "64d31c5edb39aaabe0c085b30d1e19ade43ccab79cbc241b3b654ca85646fd8b"
)
EXPECTED_HISTORICAL_RESULT_SHA256 = (
    "996243be55e12fa53bcade55654bc46459491079cbab2842f37f7d83e9e22ec8"
)
EXPECTED_ANCHOR_STEP = 38200
EXPECTED_ANCHOR_DEV_LOSS = 0.7701343703584546
EXPECTED_ANCHOR_MODEL_STATE_SHA256 = (
    "bbec7389f56b36a24db000111adf8ce99e3bf712609cfcd1562c64c9910172f8"
)
EXPECTED_ANCHOR_PREDICTIONS_SHA256 = (
    "3663a15ad9859abc76141d815f0da3c88bd9b3d5d9e914f09a3d8b0a3ed843f4"
)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def oracle_eval_steps(query_steps: int, eval_steps: int) -> list[int]:
    steps = list(range(eval_steps, query_steps + 1, eval_steps))
    if not steps or steps[-1] != query_steps:
        steps.append(query_steps)
    return steps


def _step_name(step: int) -> str:
    return f"step_{step:08d}"


def _metric_recall(metrics: dict[str, Any], cutoff: int) -> float:
    return float(metrics["overall"][f"@{cutoff}"]["recall"])


def oracle_selection_key(result: dict[str, Any]) -> tuple[float, float, float, int]:
    overall = result["overall"]
    return (
        float(overall["@1"]["recall"]),
        float(overall["@10"]["recall"]),
        float(overall["@50"]["recall"]),
        -int(result["step"]),
    )


class EvalCheckpointSnapshotQueue:
    """Materialize query-eval model states without changing training RNG state."""

    def __init__(
        self,
        root: Path,
        *,
        expected_steps: list[int],
        max_queued: int,
        capacity_wait_seconds: float = 3600.0,
    ) -> None:
        self.root = root
        self.expected_steps = [int(step) for step in expected_steps]
        self.max_queued = int(max_queued)
        self.capacity_wait_seconds = float(capacity_wait_seconds)
        if not self.root.is_absolute():
            raise ValueError(f"oracle checkpoint queue must be absolute: {self.root}")
        if self.max_queued < 1:
            raise ValueError(f"max queued checkpoints must be positive: {self.max_queued}")
        self.root.mkdir(parents=True, exist_ok=True)
        unexpected = sorted(
            path.name
            for path in self.root.iterdir()
            if path.name.startswith("step_") or path.name.startswith(".step_")
        )
        if unexpected:
            raise FileExistsError(f"oracle checkpoint queue is not empty: {unexpected[:5]}")
        self.pending: dict[str, Any] | None = None
        self.template: dict[str, Any] | None = None
        self.original_saver: Any = None
        self.records: list[dict[str, Any]] = []

    def observe_loss_event(self, row: dict[str, Any]) -> None:
        if row.get("stage") != "query" or row.get("event") != "eval":
            return
        if self.pending is not None:
            raise RuntimeError(
                f"new oracle eval arrived before pending snapshot was flushed: {self.pending}"
            )
        step = int(row["step"])
        dev_loss = float(row["dev_loss"])
        if step not in self.expected_steps:
            raise ValueError(f"unexpected oracle eval step: {step}")
        if not math.isfinite(dev_loss):
            raise ValueError(f"non-finite oracle dev loss at step {step}: {dev_loss}")
        if any(int(record["step"]) == step for record in self.records):
            raise ValueError(f"duplicate oracle eval step: {step}")
        self.pending = {"step": step, "dev_loss": dev_loss}

    def capture_saver_template(
        self, original_saver: Any, save_args: tuple[Any, ...], save_kwargs: dict[str, Any]
    ) -> None:
        if self.template is not None:
            return
        bound = inspect.signature(original_saver).bind_partial(*save_args, **save_kwargs)
        bound.apply_defaults()
        values = bound.arguments
        checkpoint_name = Path(values["checkpoint_dir"]).name
        if checkpoint_name != "best":
            raise RuntimeError(
                "oracle saver template must be captured from the first dev-best save, "
                f"got {checkpoint_name}"
            )
        required = (
            "tokenizer",
            "base_model",
            "codebook_size",
            "depth",
            "dedup_size",
            "seed",
            "training_formulation",
            "semantic_id_mode",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise RuntimeError(f"oracle saver template is incomplete: {missing}")
        extra = dict(values.get("extra") or {})
        extra.pop("best_step", None)
        extra.pop("best_dev_loss", None)
        self.template = {name: values[name] for name in required}
        self.template["extra"] = extra
        self.original_saver = original_saver

    def _queued_checkpoint_count(self) -> int:
        return sum(
            1
            for path in self.root.glob("step_*")
            if path.is_dir() and path.name[5:].isdigit()
        )

    def _wait_for_capacity(self) -> None:
        started = time.monotonic()
        while True:
            failure = self.root / "worker.failed.json"
            if failure.is_file():
                raise RuntimeError(
                    f"oracle evaluator worker failed: {failure.read_text(encoding='utf-8')}"
                )
            if self._queued_checkpoint_count() < self.max_queued:
                return
            if time.monotonic() - started > self.capacity_wait_seconds:
                raise TimeoutError(
                    "timed out waiting for oracle checkpoint queue capacity: "
                    f"max_queued={self.max_queued}"
                )
            time.sleep(0.5)

    def flush(self, model: Any, torch_module: Any, *, reason: str) -> bool:
        if self.pending is None:
            return False
        if self.template is None or self.original_saver is None:
            raise RuntimeError("oracle snapshot requested before saver template capture")
        self._wait_for_capacity()
        pending = dict(self.pending)
        step = int(pending["step"])
        name = _step_name(step)
        checkpoint_dir = self.root / name
        temporary_dir = self.root / f".{name}.tmp.{os.getpid()}"
        ready_path = self.root / f"{name}.ready.json"
        if checkpoint_dir.exists() or temporary_dir.exists() or ready_path.exists():
            raise FileExistsError(f"refusing to overwrite oracle snapshot for step {step}")

        cpu_rng_before = torch_module.random.get_rng_state().clone()
        cuda_rng_before = torch_module.cuda.get_rng_state().clone()
        extra = {
            **self.template["extra"],
            "oracle_step": step,
            "oracle_dev_loss": float(pending["dev_loss"]),
            "oracle_snapshot_reason": reason,
        }
        try:
            self.original_saver(
                temporary_dir,
                model,
                self.template["tokenizer"],
                self.template["base_model"],
                self.template["codebook_size"],
                self.template["depth"],
                self.template["dedup_size"],
                self.template["seed"],
                self.template["training_formulation"],
                extra,
                optimizer=None,
                semantic_id_mode=self.template["semantic_id_mode"],
            )
            cpu_rng_after = torch_module.random.get_rng_state()
            cuda_rng_after = torch_module.cuda.get_rng_state()
            if not torch_module.equal(cpu_rng_before, cpu_rng_after):
                raise RuntimeError(f"oracle snapshot changed CPU RNG at step {step}")
            if not torch_module.equal(cuda_rng_before, cuda_rng_after):
                raise RuntimeError(f"oracle snapshot changed CUDA RNG at step {step}")
            checkpoint_path = temporary_dir / "structured_decoder.pt"
            if not checkpoint_path.is_file():
                raise RuntimeError(f"oracle snapshot is missing model state at step {step}")
            os.replace(temporary_dir, checkpoint_dir)
            record = {
                "step": step,
                "dev_loss": float(pending["dev_loss"]),
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_bytes": int(
                    (checkpoint_dir / "structured_decoder.pt").stat().st_size
                ),
                "reason": reason,
                "cpu_rng_unchanged": True,
                "cuda_rng_unchanged": True,
            }
            _write_json_atomic(ready_path, record)
            self.records.append(record)
            self.pending = None
            print(
                f"oracle_snapshot step={step} dev_loss={pending['dev_loss']:.9f} "
                f"reason={reason} queued={self._queued_checkpoint_count()}",
                flush=True,
            )
            return True
        except BaseException:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            raise

    def validate(self) -> dict[str, Any]:
        if self.pending is not None:
            raise RuntimeError(f"unflushed oracle snapshot: {self.pending}")
        observed_steps = [int(record["step"]) for record in self.records]
        if observed_steps != self.expected_steps:
            raise ValueError(
                "oracle snapshot step mismatch: "
                f"expected_tail={self.expected_steps[-3:]}, observed_tail={observed_steps[-3:]}, "
                f"expected_count={len(self.expected_steps)}, observed_count={len(observed_steps)}"
            )
        if not all(
            record["cpu_rng_unchanged"] and record["cuda_rng_unchanged"]
            for record in self.records
        ):
            raise ValueError("oracle snapshot RNG invariant failed")
        return {
            "enabled": True,
            "queue_root": str(self.root),
            "max_queued": self.max_queued,
            "snapshot_count": len(self.records),
            "steps": observed_steps,
            "records": self.records,
            "rng_invariant_passed": True,
        }


def _claim_ready(queue_root: Path, worker_id: int) -> Path | None:
    for ready in sorted(queue_root.glob("step_*.ready.json")):
        claimed = queue_root / ready.name.replace(
            ".ready.json", f".claimed.worker_{worker_id:02d}.json"
        )
        try:
            os.replace(ready, claimed)
        except FileNotFoundError:
            continue
        return claimed
    return None


def _safe_remove_tree(path: Path, allowed_root: Path) -> None:
    resolved_root = allowed_root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise RuntimeError(f"refusing to remove path outside oracle queue: {resolved_path}")
    shutil.rmtree(resolved_path)


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"command": command}) + "\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _evaluate_checkpoint(args: argparse.Namespace, ready: dict[str, Any]) -> dict[str, Any]:
    step = int(ready["step"])
    checkpoint_dir = Path(ready["checkpoint_dir"])
    if checkpoint_dir.parent.resolve() != args.queue_root.resolve():
        raise ValueError(f"checkpoint is outside queue root: {checkpoint_dir}")
    if checkpoint_dir.name != _step_name(step):
        raise ValueError(f"checkpoint step/path mismatch: step={step}, path={checkpoint_dir}")
    eval_dir = args.output_root / "steps" / _step_name(step)
    if eval_dir.exists():
        raise FileExistsError(f"refusing to overwrite oracle evaluation: {eval_dir}")
    eval_dir.mkdir(parents=True)
    log_path = eval_dir / "eval.log"
    _run_logged(
        [
            sys.executable,
            str(args.evaluator),
            "eval-single-sid-decoder",
            "--rows",
            str(args.rows),
            "--model-dir",
            str(checkpoint_dir),
            "--semantic-ids",
            str(args.semantic_ids),
            "--output-dir",
            str(eval_dir),
            "--split-name",
            "official_core75",
            "--max-source-length",
            "2048",
            "--num-beams",
            "50",
            "--rank-aggregation",
            "beam_order",
            "--decoder-head",
            "ce",
            "--score-transform",
            "logprob",
            "--k-values",
            "1,10,50",
            "--log-every",
            "75",
        ],
        log_path,
    )
    for subset in ("all", "single", "multi"):
        _run_logged(
            [
                sys.executable,
                str(args.evaluator),
                "rescore-ranked",
                "--predictions",
                str(eval_dir / "predictions.jsonl"),
                "--subset",
                subset,
                "--k-values",
                "1,10,50",
                "--output",
                str(eval_dir / f"ranked_metrics_{subset}.json"),
            ],
            log_path,
        )
    result = {
        "artifact_format": "legacy_core75_checkpoint_oracle_step_v1",
        "step": step,
        "dev_loss": float(ready["dev_loss"]),
        "worker_id": int(args.worker_id),
        "checkpoint_bytes": int(ready["checkpoint_bytes"]),
        "snapshot_reason": str(ready["reason"]),
        "overall": json.loads(
            (eval_dir / "ranked_metrics_all.json").read_text(encoding="utf-8")
        )["overall"],
        "single": json.loads(
            (eval_dir / "ranked_metrics_single.json").read_text(encoding="utf-8")
        )["overall"],
        "multi": json.loads(
            (eval_dir / "ranked_metrics_multi.json").read_text(encoding="utf-8")
        )["overall"],
        "predictions_sha256": _file_sha256(eval_dir / "predictions.jsonl"),
        "evaluator_sha256": _file_sha256(args.evaluator),
        "protocol": "Official core75, 75 queries, beam=50, beam_order, CE log-prob",
    }
    _write_json_atomic(eval_dir / "oracle_result.json", result)
    return result


def _copy_tree_atomic(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite retained checkpoint: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary retained checkpoint exists: {temporary}")
    shutil.copytree(source, temporary, symlinks=True)
    os.replace(temporary, destination)


def command_worker(args: argparse.Namespace) -> None:
    args.queue_root = args.queue_root.resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_root = args.output_root / "workers" / f"worker_{args.worker_id:02d}"
    worker_root.mkdir(parents=True, exist_ok=False)
    local_best = args.queue_root / f"worker_{args.worker_id:02d}_best"
    best_key: tuple[float, float, float, int] | None = None
    best_step: int | None = None
    processed: list[int] = []
    current_claim: Path | None = None
    try:
        while True:
            current_claim = _claim_ready(args.queue_root, args.worker_id)
            if current_claim is None:
                if (args.queue_root / "producer.failed.json").is_file():
                    raise RuntimeError("oracle checkpoint producer failed")
                if (args.queue_root / "producer.done").is_file():
                    break
                time.sleep(args.poll_seconds)
                continue
            ready = json.loads(current_claim.read_text(encoding="utf-8"))
            result = _evaluate_checkpoint(args, ready)
            step = int(result["step"])
            candidate = Path(ready["checkpoint_dir"])
            key = oracle_selection_key(result)

            if step == args.anchor_step:
                retained_anchor = (
                    args.output_root / "retained" / _step_name(args.anchor_step)
                )
                retained_anchor.parent.mkdir(parents=True, exist_ok=True)
                _copy_tree_atomic(candidate, retained_anchor)

            if best_key is None or key > best_key:
                if local_best.exists():
                    _safe_remove_tree(local_best, args.queue_root)
                os.replace(candidate, local_best)
                best_key = key
                best_step = step
            else:
                _safe_remove_tree(candidate, args.queue_root)

            processed.append(step)
            current_claim.unlink()
            current_claim = None
            _write_json_atomic(
                worker_root / "progress.json",
                {
                    "worker_id": args.worker_id,
                    "processed_steps": processed,
                    "local_best_step": best_step,
                    "local_best_key": list(best_key) if best_key is not None else None,
                    "local_best_checkpoint": str(local_best) if best_step is not None else None,
                },
            )
            print(
                f"oracle_worker={args.worker_id} step={step} "
                f"r1={_metric_recall(result, 1):.9f} local_best={best_step}",
                flush=True,
            )
    except BaseException as exc:
        failure = {
            "worker_id": args.worker_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claim": str(current_claim) if current_claim is not None else None,
            "processed_steps": processed,
        }
        _write_json_atomic(worker_root / "failed.json", failure)
        _write_json_atomic(args.queue_root / "worker.failed.json", failure)
        raise

    if best_step is None or best_key is None or not local_best.is_dir():
        raise RuntimeError(f"oracle worker {args.worker_id} processed no checkpoints")
    _write_json_atomic(
        worker_root / "done.json",
        {
            "worker_id": args.worker_id,
            "processed_steps": processed,
            "local_best_step": best_step,
            "local_best_key": list(best_key),
            "local_best_checkpoint": str(local_best),
        },
    )


def _subset_exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for cutoff in (1, 10, 50):
        if float(left[f"@{cutoff}"]["recall"]) != float(
            right[f"@{cutoff}"]["recall"]
        ):
            return False
    return int(left["rows"]) == int(right["rows"]) and int(
        left["total_gold"]
    ) == int(right["total_gold"])


def _checkpoint_state_summary(checkpoint_dir: Path) -> dict[str, Any]:
    import torch

    try:
        from legacy_baseline_parity_probe import _hash_tensor_state
    except ModuleNotFoundError:
        from scripts.legacy_baseline_parity_probe import _hash_tensor_state

    checkpoint_path = checkpoint_dir / "structured_decoder.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "path": str(checkpoint_path),
        "file_sha256": _file_sha256(checkpoint_path),
        "model_state_sha256": _hash_tensor_state(payload["state_dict"]),
        "oracle_step": payload.get("oracle_step"),
        "oracle_dev_loss": payload.get("oracle_dev_loss"),
        "optimizer_state_present": "optimizer_state_dict" in payload,
    }


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (lvalue - left_mean) * (rvalue - right_mean)
        for lvalue, rvalue in zip(left, right)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def command_analyze(args: argparse.Namespace) -> None:
    if args.anchor_step != EXPECTED_ANCHOR_STEP:
        raise ValueError(
            f"historical anchor step must be {EXPECTED_ANCHOR_STEP}: {args.anchor_step}"
        )
    expected_steps = oracle_eval_steps(args.query_steps, args.eval_steps)
    result_paths = sorted((args.output_root / "steps").glob("step_*/oracle_result.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    results_by_step = {int(result["step"]): result for result in results}
    if len(results_by_step) != len(results):
        raise ValueError("duplicate oracle result step")
    if sorted(results_by_step) != expected_steps:
        raise ValueError(
            "oracle result coverage mismatch: "
            f"expected_count={len(expected_steps)}, observed_count={len(results_by_step)}, "
            f"missing={sorted(set(expected_steps) - set(results_by_step))[:10]}"
        )

    training_loss_history_path = args.train_arm_root / "train" / "loss_history.jsonl"
    historical_loss_history_path = (
        args.historical_arm_root / "train" / "loss_history.jsonl"
    )
    loss_events = [
        json.loads(line)
        for line in training_loss_history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev_by_step = {
        int(row["step"]): float(row["dev_loss"])
        for row in loss_events
        if row.get("stage") == "query" and row.get("event") == "eval"
    }
    if sorted(dev_by_step) != expected_steps:
        raise ValueError("training dev-loss coverage does not match oracle cadence")
    if any(
        float(results_by_step[step]["dev_loss"]) != dev_by_step[step]
        for step in expected_steps
    ):
        raise ValueError("oracle result dev loss does not match training history")

    selected = max(results, key=oracle_selection_key)
    selected_step = int(selected["step"])
    worker_done_paths = sorted((args.output_root / "workers").glob("worker_*/done.json"))
    worker_done = [json.loads(path.read_text(encoding="utf-8")) for path in worker_done_paths]
    processed_steps = sorted(
        step for worker in worker_done for step in worker["processed_steps"]
    )
    if processed_steps != expected_steps:
        raise ValueError("worker completion manifests do not cover oracle steps exactly")
    selected_workers = [
        worker for worker in worker_done if int(worker["local_best_step"]) == selected_step
    ]
    if len(selected_workers) != 1:
        raise ValueError(
            f"selected checkpoint is not retained by exactly one worker: {selected_workers}"
        )
    selected_source = Path(selected_workers[0]["local_best_checkpoint"])
    selected_checkpoint_dir = args.output_root / "test_best"
    _copy_tree_atomic(selected_source, selected_checkpoint_dir)
    shutil.copy2(
        args.train_arm_root / "full_query_arm.json",
        selected_checkpoint_dir / "training_provenance.json",
    )

    anchor_result = results_by_step[args.anchor_step]
    anchor_checkpoint_dir = args.output_root / "retained" / _step_name(args.anchor_step)
    selected_checkpoint = _checkpoint_state_summary(selected_checkpoint_dir)
    anchor_checkpoint = _checkpoint_state_summary(anchor_checkpoint_dir)
    if selected_checkpoint["oracle_step"] != selected_step:
        raise ValueError("selected checkpoint metadata step mismatch")
    if anchor_checkpoint["oracle_step"] != args.anchor_step:
        raise ValueError("anchor checkpoint metadata step mismatch")
    if selected_checkpoint["optimizer_state_present"] or anchor_checkpoint[
        "optimizer_state_present"
    ]:
        raise ValueError("oracle checkpoints must be model-only")

    historical_arm_path = args.historical_arm_root / "full_query_arm.json"
    historical_result_path = args.historical_arm_root / "result_summary.json"
    historical_artifact_hashes = {
        "full_query_arm": _file_sha256(historical_arm_path),
        "result_summary": _file_sha256(historical_result_path),
    }
    expected_historical_artifact_hashes = {
        "full_query_arm": EXPECTED_HISTORICAL_ARM_SHA256,
        "result_summary": EXPECTED_HISTORICAL_RESULT_SHA256,
    }
    if historical_artifact_hashes != expected_historical_artifact_hashes:
        raise ValueError(
            "historical control artifact SHA-256 mismatch: "
            f"expected={expected_historical_artifact_hashes}, "
            f"observed={historical_artifact_hashes}"
        )
    historical_arm = json.loads(historical_arm_path.read_text(encoding="utf-8"))
    historical_result = json.loads(historical_result_path.read_text(encoding="utf-8"))
    training_arm = json.loads(
        (args.train_arm_root / "full_query_arm.json").read_text(encoding="utf-8")
    )
    anchor_checks = {
        "known_step": int(anchor_result["step"]) == EXPECTED_ANCHOR_STEP,
        "known_dev_loss": float(anchor_result["dev_loss"])
        == EXPECTED_ANCHOR_DEV_LOSS,
        "known_model_state": anchor_checkpoint["model_state_sha256"]["full"]
        == EXPECTED_ANCHOR_MODEL_STATE_SHA256,
        "known_predictions": anchor_result["predictions_sha256"]
        == EXPECTED_ANCHOR_PREDICTIONS_SHA256,
        "step": int(anchor_result["step"])
        == int(historical_result["training"]["query_best_step"]),
        "dev_loss": float(anchor_result["dev_loss"])
        == float(historical_result["training"]["query_best_dev_loss"]),
        "model_state": anchor_checkpoint["model_state_sha256"]
        == historical_result["training"]["final_model_state_sha256"],
        "predictions": anchor_result["predictions_sha256"]
        == historical_result["evaluation"]["predictions_sha256"],
        "overall_metrics": _subset_exact(anchor_result["overall"], historical_result["overall"]),
        "single_metrics": _subset_exact(anchor_result["single"], historical_result["single"]),
        "multi_metrics": _subset_exact(anchor_result["multi"], historical_result["multi"]),
    }
    training_checks = {
        "loss_history_sha256": _file_sha256(training_loss_history_path)
        == _file_sha256(historical_loss_history_path),
        "catalog_boundary": training_arm["catalog_boundary"]
        == historical_arm["catalog_boundary"],
        "query_step_1": training_arm["query_step_1"]
        == historical_arm["query_step_1"],
        "best_step": training_arm["training"]["history"]["best_step"]
        == historical_arm["training"]["history"]["best_step"],
        "best_dev_loss": training_arm["training"]["history"]["best_dev_loss"]
        == historical_arm["training"]["history"]["best_dev_loss"],
        "final_model_state": training_arm["training"]["final_live_model_state_sha256"]
        == historical_arm["training"]["final_live_model_state_sha256"],
    }
    if not all(anchor_checks.values()) or not all(training_checks.values()):
        raise ValueError(
            f"historical control invariant failed: anchor={anchor_checks}, training={training_checks}"
        )

    ordered = sorted(results, key=oracle_selection_key, reverse=True)
    dev_losses = [float(result["dev_loss"]) for result in results]
    r1_values = [_metric_recall(result, 1) for result in results]
    summary = {
        "artifact_format": "legacy_core75_checkpoint_oracle_summary_v1",
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
            "selection": "maximize Overall R@1, then R@10, then R@50, then prefer earlier step",
            "status": "diagnostic_test_reuse",
        },
        "coverage": {
            "expected_steps": expected_steps,
            "result_count": len(results),
            "worker_count": len(worker_done),
            "passed": True,
        },
        "selected": {
            **selected,
            "selection_key": list(oracle_selection_key(selected)),
            "checkpoint": selected_checkpoint,
            "within_one_point_of_rtx_r1": _metric_recall(selected, 1)
            >= 0.27368253968253965,
            "at_least_rtx_r1": _metric_recall(selected, 1) >= 0.2836825396825397,
        },
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
        "anchor": {
            "step": args.anchor_step,
            "result": anchor_result,
            "checkpoint": anchor_checkpoint,
            "checks": anchor_checks,
            "historical_artifact_sha256": historical_artifact_hashes,
            "passed": True,
        },
        "training_invariant": {
            "checks": training_checks,
            "loss_history_sha256": {
                "observed": _file_sha256(training_loss_history_path),
                "historical": _file_sha256(historical_loss_history_path),
            },
            "passed": True,
        },
        "correlation": {
            "pearson_dev_loss_vs_r1": _pearson(dev_losses, r1_values),
            "spearman_dev_loss_vs_r1": _pearson(
                _average_ranks(dev_losses), _average_ranks(r1_values)
            ),
        },
        "results": results,
    }
    _write_json_atomic(args.output_root / "checkpoint_oracle_summary.json", summary)
    (args.output_root / "checkpoint_oracle.done").touch()
    print(
        json.dumps(
            {
                "selected_step": selected_step,
                "r1": _metric_recall(selected, 1),
                "r10": _metric_recall(selected, 10),
                "r50": _metric_recall(selected, 50),
                "dev_loss": selected["dev_loss"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--worker-id", type=int, required=True)
    worker.add_argument("--queue-root", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--evaluator", type=Path, required=True)
    worker.add_argument("--rows", type=Path, required=True)
    worker.add_argument("--semantic-ids", type=Path, required=True)
    worker.add_argument("--anchor-step", type=int, default=38200)
    worker.add_argument("--poll-seconds", type=float, default=0.5)
    worker.set_defaults(func=command_worker)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output-root", type=Path, required=True)
    analyze.add_argument("--train-arm-root", type=Path, required=True)
    analyze.add_argument("--historical-arm-root", type=Path, required=True)
    analyze.add_argument("--query-steps", type=int, default=38805)
    analyze.add_argument("--eval-steps", type=int, default=200)
    analyze.add_argument("--anchor-step", type=int, default=38200)
    analyze.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
