#!/usr/bin/env python3
"""Run one full archived single-SID arm with a controlled query optimizer boundary."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

try:
    from legacy_baseline_parity_probe import _hash_tensor_state
    from legacy_layerwise_trace_probe import (
        EXPECTED_DROPOUT_MODULES,
        EXPECTED_INITIAL_MODEL_FINGERPRINT,
        EXPECTED_PIPELINE_SHA256,
        load_module,
        stable_json_sha256,
    )
    from legacy_optimizer_boundary_probe import (
        CATALOG_STEPS,
        EXPECTED_CATALOG_FIRST_BATCH_SHA256,
        EXPECTED_CPU_RNG_AFTER_MODEL_SHA256,
        EXPECTED_CUDA_RNG_BEFORE_FORWARD_SHA256,
        EXPECTED_DEFAULT_CATALOG_LOSSES,
        EXPECTED_MANIFEST_ROW_COUNTS,
        EXPECTED_QUERY_FIRST_BATCH_SHA256,
        OPTIMIZER_MODES,
        OPTIONAL_BOOL_CHOICES,
        legacy_model_state_fingerprint,
        optimizer_param_group_snapshot,
        recursive_state_sha256,
        reset_optimizer_in_place,
        row_ids_payload,
        set_optimizer_lr,
        tensor_bytes_sha256,
        tensor_collection_summary,
    )
    from legacy_runtime_matrix_probe import (
        apply_precision_controls,
        file_sha256,
        precision_snapshot,
        runtime_snapshot,
        selected_weight_file,
    )
except ModuleNotFoundError:
    from scripts.legacy_baseline_parity_probe import _hash_tensor_state
    from scripts.legacy_layerwise_trace_probe import (
        EXPECTED_DROPOUT_MODULES,
        EXPECTED_INITIAL_MODEL_FINGERPRINT,
        EXPECTED_PIPELINE_SHA256,
        load_module,
        stable_json_sha256,
    )
    from scripts.legacy_optimizer_boundary_probe import (
        CATALOG_STEPS,
        EXPECTED_CATALOG_FIRST_BATCH_SHA256,
        EXPECTED_CPU_RNG_AFTER_MODEL_SHA256,
        EXPECTED_CUDA_RNG_BEFORE_FORWARD_SHA256,
        EXPECTED_DEFAULT_CATALOG_LOSSES,
        EXPECTED_MANIFEST_ROW_COUNTS,
        EXPECTED_QUERY_FIRST_BATCH_SHA256,
        OPTIMIZER_MODES,
        OPTIONAL_BOOL_CHOICES,
        legacy_model_state_fingerprint,
        optimizer_param_group_snapshot,
        recursive_state_sha256,
        reset_optimizer_in_place,
        row_ids_payload,
        set_optimizer_lr,
        tensor_bytes_sha256,
        tensor_collection_summary,
    )
    from scripts.legacy_runtime_matrix_probe import (
        apply_precision_controls,
        file_sha256,
        precision_snapshot,
        runtime_snapshot,
        selected_weight_file,
    )

try:
    from legacy_ce_prefix_ranking import (
        FrozenC1Candidates,
        GROUPINGS,
        PrefixRankingRuntime,
        SCOPE_DEPTHS,
        attach_prefix_ranking_forward,
    )
except ModuleNotFoundError:
    from scripts.legacy_ce_prefix_ranking import (
        FrozenC1Candidates,
        GROUPINGS,
        PrefixRankingRuntime,
        SCOPE_DEPTHS,
        attach_prefix_ranking_forward,
    )

try:
    from legacy_core75_checkpoint_oracle import EvalCheckpointSnapshotQueue
except ModuleNotFoundError:
    from scripts.legacy_core75_checkpoint_oracle import EvalCheckpointSnapshotQueue


QUERY_STEPS = 38805
EVAL_STEPS = 200
QUERY_DEV_ROWS = EXPECTED_MANIFEST_ROW_COUNTS["query_dev_rows"]


def query_lr_at_step(base_lr: float, query_step: int, warmup_steps: int) -> float:
    if base_lr <= 0.0:
        raise ValueError(f"base lr must be positive, got {base_lr}")
    if query_step <= 0:
        raise ValueError(f"query step must be positive, got {query_step}")
    if warmup_steps < 0:
        raise ValueError(f"warmup steps must be non-negative, got {warmup_steps}")
    if warmup_steps == 0:
        return float(base_lr)
    return float(base_lr) * min(float(query_step) / float(warmup_steps), 1.0)


def clip_query_gradients(model: Any, max_grad_norm: float) -> Any:
    import torch

    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
        raise ValueError(
            f"max grad norm must be finite and positive, got {max_grad_norm}"
        )
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters:
        raise RuntimeError("gradient clipping requested with no trainable gradients")
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_grad_norm,
        norm_type=2.0,
        error_if_nonfinite=True,
        foreach=True,
    )


def expected_eval_steps(query_steps: int, eval_steps: int) -> list[int]:
    steps = list(range(eval_steps, query_steps + 1, eval_steps))
    if not steps or steps[-1] != query_steps:
        steps.append(query_steps)
    return steps


def validate_loss_history(
    events: list[dict[str, Any]],
    *,
    query_lr: float,
    query_steps: int,
    eval_steps: int,
    query_warmup_steps: int = 0,
) -> dict[str, Any]:
    observed_eval_steps = [
        int(row["step"])
        for row in events
        if row.get("stage") == "query" and row.get("event") == "eval"
    ]
    expected_steps = expected_eval_steps(query_steps, eval_steps)
    if observed_eval_steps != expected_steps:
        raise ValueError(
            "query eval cadence mismatch: "
            f"expected_tail={expected_steps[-3:]}, observed_tail={observed_eval_steps[-3:]}"
        )
    catalog_lrs = {
        float(row["lr"])
        for row in events
        if row.get("stage") == "catalog" and row.get("event") == "train"
    }
    query_lr_events = [
        (int(row["step"]), float(row["lr"]))
        for row in events
        if row.get("stage") == "query" and row.get("event") == "train"
    ]
    if not query_lr_events:
        raise ValueError("query lr history is empty")
    query_lrs = {lr for _, lr in query_lr_events}
    if catalog_lrs != {3e-4}:
        raise ValueError(f"catalog lr history mismatch: {catalog_lrs}")
    mismatched_query_lrs = [
        (step, observed_lr, query_lr_at_step(query_lr, step, query_warmup_steps))
        for step, observed_lr in query_lr_events
        if observed_lr != query_lr_at_step(query_lr, step, query_warmup_steps)
    ]
    if mismatched_query_lrs:
        raise ValueError(f"query lr history mismatch: {mismatched_query_lrs[:5]}")
    best = min(
        (
            row
            for row in events
            if row.get("stage") == "query" and row.get("event") == "eval"
        ),
        key=lambda row: float(row["dev_loss"]),
    )
    return {
        "eval_steps": observed_eval_steps,
        "best_step": int(best["step"]),
        "best_dev_loss": float(best["dev_loss"]),
        "catalog_logged_lrs": sorted(catalog_lrs),
        "query_logged_lrs": sorted(query_lrs),
        "query_warmup_steps": query_warmup_steps,
    }


def checkpoint_state_summary(checkpoint_dir: Path) -> dict[str, Any]:
    import torch

    checkpoint_path = checkpoint_dir / "structured_decoder.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_hash = _hash_tensor_state(checkpoint["state_dict"])
    return {
        "path": str(checkpoint_path),
        "file_sha256": file_sha256(checkpoint_path),
        "model_state_sha256": state_hash,
        "optimizer_state_present": "optimizer_state_dict" in checkpoint,
        "selected_checkpoint": checkpoint.get("selected_checkpoint"),
        "best_step": checkpoint.get("best_step"),
        "best_dev_loss": checkpoint.get("best_dev_loss"),
    }


class FullQueryObserver:
    def __init__(
        self,
        *,
        optimizer_mode: str,
        query_lr: float,
        catalog_steps: int,
        query_steps: int,
        eval_steps: int,
        batch_size: int,
        query_warmup_steps: int = 0,
        query_max_grad_norm: float | None = None,
    ) -> None:
        if optimizer_mode not in OPTIMIZER_MODES:
            raise ValueError(f"unsupported optimizer mode: {optimizer_mode}")
        self.optimizer_mode = optimizer_mode
        self.query_lr = float(query_lr)
        self.catalog_steps = int(catalog_steps)
        self.query_steps = int(query_steps)
        self.eval_steps = int(eval_steps)
        self.batch_size = int(batch_size)
        self.query_warmup_steps = int(query_warmup_steps)
        self.query_max_grad_norm = (
            None if query_max_grad_norm is None else float(query_max_grad_norm)
        )
        if self.query_warmup_steps < 0:
            raise ValueError(
                f"query warmup steps must be non-negative, got {self.query_warmup_steps}"
            )
        if self.query_max_grad_norm is not None and (
            not math.isfinite(self.query_max_grad_norm)
            or self.query_max_grad_norm <= 0.0
        ):
            raise ValueError(
                "query max grad norm must be finite and positive, "
                f"got {self.query_max_grad_norm}"
            )
        if self.query_warmup_steps and self.query_max_grad_norm is not None:
            raise ValueError(
                "query warmup and gradient clipping must be tested separately"
            )
        self.model = None
        self.optimizer = None
        self.optimizer_steps = 0
        self.forward_calls = 0
        self.batch_calls = 0
        self.last_scalar_loss: float | None = None
        self.initial: dict[str, Any] = {}
        self.batches: dict[str, Any] = {}
        self.losses: dict[str, float] = {}
        self.catalog: dict[str, Any] = {}
        self.transition: dict[str, Any] = {}
        self.query_step_1: dict[str, Any] = {}
        self.query_lr_trace: dict[str, float] = {}
        self.clipping: dict[str, Any] = {
            "max_grad_norm": self.query_max_grad_norm,
            "norm_type": 2.0,
            "foreach": True,
            "calls": 0,
            "sampled_clipped_calls": 0,
            "pre_clip_norm_samples": {},
        }
        self.checkpoint_saves: list[dict[str, Any]] = []

    def remember_model(self, model: Any) -> Any:
        self.model = model
        original_forward = model.forward

        def observed_forward(*args: Any, **kwargs: Any) -> Any:
            output = original_forward(*args, **kwargs)
            self.forward_calls += 1
            if getattr(output, "ndim", None) != 0:
                raise ValueError("structured decoder forward must return a scalar loss")
            self.last_scalar_loss = float(output.detach().cpu())
            return output

        model.forward = observed_forward
        return model

    def capture_catalog_boundary(self) -> None:
        import torch

        if self.model is None or self.optimizer is None:
            raise RuntimeError(
                "catalog boundary reached before model/optimizer capture"
            )
        if self.catalog:
            raise RuntimeError("catalog boundary captured more than once")
        if self.optimizer_steps != self.catalog_steps:
            raise RuntimeError(
                f"catalog boundary optimizer-step mismatch: {self.optimizer_steps}"
            )
        self.catalog = {
            "model_fingerprint": legacy_model_state_fingerprint(self.model),
            "model_state_sha256": _hash_tensor_state(self.model.state_dict()),
            "optimizer_state_sha256": recursive_state_sha256(
                self.optimizer.state_dict()
            ),
            "optimizer_state_entries": len(self.optimizer.state),
            "optimizer_param_groups": optimizer_param_group_snapshot(self.optimizer),
            "cpu_rng_sha256": tensor_bytes_sha256(torch.random.get_rng_state()),
            "cuda_rng_sha256": tensor_bytes_sha256(torch.cuda.get_rng_state()),
        }

    def wrap_batch_builder(self, original_builder: Any) -> Any:
        def observed_builder(
            batch_rows: list[dict[str, Any]], *args: Any, **kwargs: Any
        ) -> Any:
            if self.optimizer_steps == 0 and "catalog_step_1" not in self.batches:
                payload = row_ids_payload(batch_rows, include_positive_skill_id=False)
                if payload["row_ids_sha256"] != EXPECTED_CATALOG_FIRST_BATCH_SHA256:
                    raise ValueError(
                        f"catalog first-batch row hash mismatch: {payload}"
                    )
                self.batches["catalog_step_1"] = payload
            elif (
                self.optimizer_steps == self.catalog_steps
                and "query_step_1" not in self.batches
            ):
                self.capture_catalog_boundary()
                payload = row_ids_payload(batch_rows, include_positive_skill_id=True)
                if payload["row_ids_sha256"] != EXPECTED_QUERY_FIRST_BATCH_SHA256:
                    raise ValueError(f"query first-batch row hash mismatch: {payload}")
                self.batches["query_step_1"] = payload
            batch = original_builder(batch_rows, *args, **kwargs)
            self.batch_calls += 1
            return batch

        return observed_builder

    def wrap_optimizer(self, optimizer: Any) -> Any:
        import torch

        if self.model is None:
            raise RuntimeError("optimizer constructed before model capture")
        self.optimizer = optimizer
        self.initial = {
            "model_fingerprint": legacy_model_state_fingerprint(self.model),
            "model_state_sha256": _hash_tensor_state(self.model.state_dict()),
            "cpu_rng_sha256": tensor_bytes_sha256(torch.random.get_rng_state()),
            "cuda_rng_sha256": tensor_bytes_sha256(torch.cuda.get_rng_state()),
            "optimizer_param_groups": optimizer_param_group_snapshot(optimizer),
            "optimizer_state_entries": len(optimizer.state),
        }
        torch.cuda.reset_peak_memory_stats()
        original_step = optimizer.step

        def observed_step(*args: Any, **kwargs: Any) -> Any:
            next_step = self.optimizer_steps + 1
            if self.last_scalar_loss is None:
                raise RuntimeError(
                    "optimizer step observed before scalar training loss"
                )
            loss_names = {
                1: "catalog_step_1",
                50: "catalog_step_50",
                self.catalog_steps: f"catalog_step_{self.catalog_steps}",
                self.catalog_steps + 1: "query_step_1",
                self.catalog_steps + self.query_steps: f"query_step_{self.query_steps}",
            }
            if next_step in loss_names:
                self.losses[loss_names[next_step]] = self.last_scalar_loss
            if self.optimizer_steps == self.catalog_steps:
                if not self.catalog:
                    raise RuntimeError(
                        "query optimizer step started before catalog capture"
                    )
                gradients = [
                    (name, parameter.grad)
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.query_step_1["gradient"] = tensor_collection_summary(gradients)
                if self.optimizer_mode == "reset":
                    reset_optimizer_in_place(optimizer, self.query_lr)
                else:
                    set_optimizer_lr(optimizer, self.query_lr)
                self.transition = {
                    "mode": self.optimizer_mode,
                    "query_lr": self.query_lr,
                    "optimizer_state_entries_after_transition": len(optimizer.state),
                    "optimizer_state_sha256_after_transition": recursive_state_sha256(
                        optimizer.state_dict()
                    ),
                    "optimizer_param_groups_after_transition": optimizer_param_group_snapshot(
                        optimizer
                    ),
                }
            query_step = next_step - self.catalog_steps
            if query_step > 0:
                effective_lr = query_lr_at_step(
                    self.query_lr, query_step, self.query_warmup_steps
                )
                if self.query_warmup_steps:
                    set_optimizer_lr(optimizer, effective_lr)
                lr_trace_steps = {1, self.query_steps}
                if self.query_warmup_steps:
                    lr_trace_steps.update(
                        {
                            min(self.query_warmup_steps, self.query_steps),
                            min(self.query_warmup_steps + 1, self.query_steps),
                        }
                    )
                if query_step in lr_trace_steps:
                    self.query_lr_trace[str(query_step)] = effective_lr
                if self.query_max_grad_norm is not None:
                    pre_clip_norm_tensor = clip_query_gradients(
                        self.model, self.query_max_grad_norm
                    )
                    self.clipping["calls"] += 1
                    if query_step in {1, 50, 500, self.query_steps}:
                        pre_clip_norm = float(pre_clip_norm_tensor.detach().cpu())
                        self.clipping["pre_clip_norm_samples"][str(query_step)] = (
                            pre_clip_norm
                        )
                        if pre_clip_norm > self.query_max_grad_norm:
                            self.clipping["sampled_clipped_calls"] += 1
            result = original_step(*args, **kwargs)
            self.optimizer_steps = next_step
            return result

        optimizer.step = observed_step
        return optimizer

    def wrap_checkpoint_saver(self, original_saver: Any) -> Any:
        def controlled_save(*args: Any, **kwargs: Any) -> Any:
            checkpoint_dir = Path(args[0])
            extra = args[9] if len(args) > 9 else kwargs.get("extra")
            name = checkpoint_dir.name
            record = {
                "name": name,
                "path": str(checkpoint_dir),
                "best_step": extra.get("best_step")
                if isinstance(extra, dict)
                else None,
                "optimizer_requested": kwargs.get("optimizer") is not None,
                "optimizer_saved": False,
                "suppressed": name == "last",
            }
            self.checkpoint_saves.append(record)
            if name == "last":
                return None
            if name not in {"best", "final"}:
                raise ValueError(f"unexpected checkpoint directory: {checkpoint_dir}")
            controlled_kwargs = dict(kwargs)
            controlled_kwargs["optimizer"] = None
            return original_saver(*args, **controlled_kwargs)

        return controlled_save

    def validate(self) -> None:
        expected_optimizer_steps = self.catalog_steps + self.query_steps
        if self.optimizer_steps != expected_optimizer_steps:
            raise ValueError(
                f"optimizer step mismatch: expected={expected_optimizer_steps}, "
                f"actual={self.optimizer_steps}"
            )
        eval_batches = len(
            expected_eval_steps(self.query_steps, self.eval_steps)
        ) * math.ceil(QUERY_DEV_ROWS / self.batch_size)
        expected_forward_calls = expected_optimizer_steps + eval_batches
        if (
            self.forward_calls != expected_forward_calls
            or self.batch_calls != expected_forward_calls
        ):
            raise ValueError(
                "forward/batch call mismatch: "
                f"expected={expected_forward_calls}, forward={self.forward_calls}, batch={self.batch_calls}"
            )
        if self.initial.get("model_fingerprint") != EXPECTED_INITIAL_MODEL_FINGERPRINT:
            raise ValueError(f"initial model fingerprint mismatch: {self.initial}")
        if self.initial.get("cpu_rng_sha256") != EXPECTED_CPU_RNG_AFTER_MODEL_SHA256:
            raise ValueError(f"initial CPU RNG mismatch: {self.initial}")
        if (
            self.initial.get("cuda_rng_sha256")
            != EXPECTED_CUDA_RNG_BEFORE_FORWARD_SHA256
        ):
            raise ValueError(f"initial CUDA RNG mismatch: {self.initial}")
        expected_losses = {
            "catalog_step_1",
            "catalog_step_50",
            f"catalog_step_{self.catalog_steps}",
            "query_step_1",
            f"query_step_{self.query_steps}",
        }
        if set(self.losses) != expected_losses:
            raise ValueError(f"loss capture mismatch: {sorted(self.losses)}")
        if set(self.batches) != {"catalog_step_1", "query_step_1"}:
            raise ValueError(f"batch capture mismatch: {self.batches.keys()}")
        if (
            not self.catalog
            or not self.transition
            or set(self.query_step_1) != {"gradient"}
        ):
            raise ValueError("catalog/query transition capture is incomplete")
        transition_entries = self.transition["optimizer_state_entries_after_transition"]
        if self.optimizer_mode == "reset" and transition_entries != 0:
            raise ValueError(
                f"reset optimizer retained {transition_entries} state entries"
            )
        if (
            self.optimizer_mode == "carry"
            and transition_entries != self.catalog["optimizer_state_entries"]
        ):
            raise ValueError("carry optimizer state-entry count changed at transition")
        transition_lrs = {
            float(group["lr"])
            for group in self.transition["optimizer_param_groups_after_transition"]
        }
        if transition_lrs != {self.query_lr}:
            raise ValueError(f"query lr transition mismatch: {transition_lrs}")
        expected_lr_trace_steps = {1, self.query_steps}
        if self.query_warmup_steps:
            expected_lr_trace_steps.update(
                {
                    min(self.query_warmup_steps, self.query_steps),
                    min(self.query_warmup_steps + 1, self.query_steps),
                }
            )
        expected_lr_trace = {
            str(step): query_lr_at_step(self.query_lr, step, self.query_warmup_steps)
            for step in expected_lr_trace_steps
        }
        if self.query_lr_trace != expected_lr_trace:
            raise ValueError(
                f"query lr trace mismatch: expected={expected_lr_trace}, "
                f"observed={self.query_lr_trace}"
            )
        expected_clip_calls = (
            self.query_steps if self.query_max_grad_norm is not None else 0
        )
        if self.clipping["calls"] != expected_clip_calls:
            raise ValueError(
                "query gradient-clipping call mismatch: "
                f"expected={expected_clip_calls}, observed={self.clipping['calls']}"
            )
        if (
            not 0
            <= self.clipping["sampled_clipped_calls"]
            <= len(self.clipping["pre_clip_norm_samples"])
        ):
            raise ValueError(f"invalid clipping counts: {self.clipping}")
        if any(
            not math.isfinite(float(value))
            for value in self.clipping["pre_clip_norm_samples"].values()
        ):
            raise ValueError(f"non-finite clipping norm sample: {self.clipping}")
        save_names = [row["name"] for row in self.checkpoint_saves]
        if (
            save_names.count("best") < 1
            or save_names.count("last") != 1
            or save_names.count("final") != 1
        ):
            raise ValueError(f"checkpoint save policy mismatch: {save_names}")
        if save_names[-2:] != ["last", "final"]:
            raise ValueError(
                f"unexpected final checkpoint save order: {save_names[-2:]}"
            )


def pipeline_training_argv(args: argparse.Namespace, train_dir: Path) -> list[str]:
    return [
        str(args.pipeline),
        "train-single-sid-decoder",
        "--rows",
        str(args.rows),
        "--dev-rows",
        str(args.dev_rows),
        "--semantic-ids",
        str(args.semantic_ids),
        "--skills-jsonl",
        str(args.skills_jsonl),
        "--output-dir",
        str(train_dir),
        "--model",
        str(args.model),
        "--catalog-max-steps",
        str(args.catalog_steps),
        "--catalog-epochs",
        "3",
        "--max-steps",
        str(args.query_steps),
        "--epochs",
        "5",
        "--batch-size",
        str(args.batch_size),
        "--lr",
        "0.0003",
        "--weight-decay",
        "0",
        "--max-source-length",
        str(args.max_source_length),
        "--logging-steps",
        "50",
        "--eval-steps",
        str(args.eval_steps),
        "--seed",
        str(args.seed),
        "--bf16",
        "--load-best-model-at-end",
    ]


def command_run(args: argparse.Namespace) -> None:
    if args.catalog_steps != CATALOG_STEPS or args.query_steps != QUERY_STEPS:
        raise ValueError(
            f"full-query sweep requires catalog/query={CATALOG_STEPS}/{QUERY_STEPS}"
        )
    if args.eval_steps != EVAL_STEPS:
        raise ValueError(f"full-query sweep requires eval_steps={EVAL_STEPS}")
    if args.oracle_checkpoint_queue is None and args.oracle_max_queued != 14:
        raise ValueError(
            "--oracle-max-queued is only valid with --oracle-checkpoint-queue"
        )
    if args.cublas_workspace_config != "auto":
        observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if observed_workspace != args.cublas_workspace_config:
            raise ValueError(
                "CUBLAS_WORKSPACE_CONFIG must be set before Python starts: "
                f"expected={args.cublas_workspace_config}, observed={observed_workspace}"
            )
    if not math.isfinite(args.prefix_rank_weight) or args.prefix_rank_weight < 0.0:
        raise ValueError(
            f"prefix rank weight must be finite and non-negative: {args.prefix_rank_weight}"
        )
    for label, weight in (
        ("single", args.prefix_rank_single_weight),
        ("multi", args.prefix_rank_multi_weight),
    ):
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"{label} prefix rank weight must be finite and non-negative: {weight}"
            )
    if args.prefix_rank_grouping == "fixed":
        if args.prefix_rank_weight > 0.0 and args.prefix_rank_scope == "disabled":
            raise ValueError("positive prefix rank weight requires an active scope")
        if (
            args.prefix_rank_single_weight != 0.0
            or args.prefix_rank_multi_weight != 0.0
        ):
            raise ValueError(
                "fixed prefix ranking does not accept group-specific weights"
            )
        prefix_rank_active = args.prefix_rank_weight > 0.0
    else:
        if args.prefix_rank_scope != "disabled" or args.prefix_rank_weight != 0.0:
            raise ValueError(
                "gold-cardinality prefix ranking requires fixed scope/weight disabled"
            )
        if args.prefix_rank_single_scope == "disabled":
            raise ValueError("gold-cardinality prefix ranking requires a single scope")
        if args.prefix_rank_multi_scope == "disabled":
            raise ValueError("gold-cardinality prefix ranking requires a multi scope")
        prefix_rank_active = (
            args.prefix_rank_single_weight > 0.0 or args.prefix_rank_multi_weight > 0.0
        )
    if (
        not math.isfinite(args.prefix_rank_temperature)
        or args.prefix_rank_temperature <= 0.0
    ):
        raise ValueError(
            "prefix rank temperature must be finite and positive: "
            f"{args.prefix_rank_temperature}"
        )
    candidate_arguments = (
        args.prefix_rank_summary,
        args.prefix_rank_summary_sha256,
        args.prefix_rank_manifest,
        args.prefix_rank_manifest_sha256,
    )
    if any(value is not None for value in candidate_arguments) and not all(
        value is not None for value in candidate_arguments
    ):
        raise ValueError("prefix ranking artifact arguments must be provided together")
    if prefix_rank_active and not all(
        value is not None for value in candidate_arguments
    ):
        raise ValueError(
            "positive prefix rank weight requires a candidate artifact bundle"
        )
    candidate_input_hashes = (
        {
            "summary": file_sha256(args.prefix_rank_summary),
            "manifest": file_sha256(args.prefix_rank_manifest),
        }
        if args.prefix_rank_summary is not None
        else {"summary": None, "manifest": None}
    )
    expected_candidate_hashes = {
        "summary": args.prefix_rank_summary_sha256,
        "manifest": args.prefix_rank_manifest_sha256,
    }
    if candidate_input_hashes != expected_candidate_hashes:
        raise ValueError(
            "prefix ranking artifact SHA-256 mismatch: "
            f"expected={expected_candidate_hashes}, observed={candidate_input_hashes}"
        )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"each full-query arm requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite full-query arm: {args.output_dir}"
        )
    if file_sha256(args.pipeline) != EXPECTED_PIPELINE_SHA256:
        raise ValueError(f"archived pipeline SHA-256 mismatch: {args.pipeline}")

    configured_precision = apply_precision_controls(torch, args)
    observer = FullQueryObserver(
        optimizer_mode=args.optimizer_mode,
        query_lr=args.query_lr,
        catalog_steps=args.catalog_steps,
        query_steps=args.query_steps,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        query_warmup_steps=args.query_warmup_steps,
        query_max_grad_norm=args.query_max_grad_norm,
    )
    dropout_modules: list[Any] = []
    original_dropout = torch.nn.Dropout

    class ControlledDropout(original_dropout):
        def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
            super().__init__(p=args.dropout_rate, inplace=inplace)
            dropout_modules.append(self)

    torch.nn.Dropout = ControlledDropout
    original_adamw = torch.optim.AdamW

    def observed_adamw(*optimizer_args: Any, **optimizer_kwargs: Any) -> Any:
        return observer.wrap_optimizer(
            original_adamw(*optimizer_args, **optimizer_kwargs)
        )

    torch.optim.AdamW = observed_adamw
    pipeline = load_module(args.pipeline, f"full_query_optimizer_lr_{args.arm}")
    original_factory = pipeline.make_structured_t5_sid_model
    original_saver = pipeline.save_structured_decoder_checkpoint
    original_append_loss_history = pipeline.append_loss_history
    snapshot_queue = (
        EvalCheckpointSnapshotQueue(
            args.oracle_checkpoint_queue,
            expected_steps=expected_eval_steps(args.query_steps, args.eval_steps),
            max_queued=args.oracle_max_queued,
        )
        if args.oracle_checkpoint_queue is not None
        else None
    )
    prefix_rank_trace: dict[str, Any] = {}
    prefix_rank_runtime = None
    if prefix_rank_active:
        semantic_ids = pipeline.load_semantic_ids(args.semantic_ids)
        candidates = FrozenC1Candidates.load(
            summary_path=args.prefix_rank_summary,
            manifest_path=args.prefix_rank_manifest,
            semantic_ids=semantic_ids,
            expected_summary_sha256=args.prefix_rank_summary_sha256,
            expected_manifest_sha256=args.prefix_rank_manifest_sha256,
            expected_rows_sha256=file_sha256(args.rows),
            expected_semantic_ids_sha256=file_sha256(args.semantic_ids),
        )
        prefix_rank_runtime = PrefixRankingRuntime(
            candidates=candidates,
            scope=args.prefix_rank_scope,
            epochs=5,
            grouping=args.prefix_rank_grouping,
            single_scope=args.prefix_rank_single_scope,
            multi_scope=args.prefix_rank_multi_scope,
        )
        pipeline.expand_single_sid_query_rows = prefix_rank_runtime.wrap_expander(
            pipeline.expand_single_sid_query_rows
        )

    def observed_factory(*factory_args: Any, **factory_kwargs: Any) -> Any:
        model = original_factory(*factory_args, **factory_kwargs)
        model = attach_prefix_ranking_forward(
            model,
            weight=args.prefix_rank_weight,
            scope=args.prefix_rank_scope,
            temperature=args.prefix_rank_temperature,
            trace=prefix_rank_trace,
            grouping=args.prefix_rank_grouping,
            single_scope=args.prefix_rank_single_scope,
            single_weight=args.prefix_rank_single_weight,
            multi_scope=args.prefix_rank_multi_scope,
            multi_weight=args.prefix_rank_multi_weight,
        )
        return observer.remember_model(model)

    pipeline.make_structured_t5_sid_model = observed_factory
    batch_builder = pipeline.structured_sid_batch
    if prefix_rank_runtime is not None:
        batch_builder = prefix_rank_runtime.wrap_batch_builder(batch_builder)
    observed_batch_builder = observer.wrap_batch_builder(batch_builder)
    if snapshot_queue is not None:

        def snapshot_aware_batch_builder(*batch_args: Any, **batch_kwargs: Any) -> Any:
            snapshot_queue.flush(
                observer.model, torch, reason="before_next_query_batch"
            )
            return observed_batch_builder(*batch_args, **batch_kwargs)

        pipeline.structured_sid_batch = snapshot_aware_batch_builder

        def snapshot_aware_append_loss_history(
            path: Path, row: dict[str, Any]
        ) -> None:
            original_append_loss_history(path, row)
            snapshot_queue.observe_loss_event(row)

        pipeline.append_loss_history = snapshot_aware_append_loss_history
    else:
        pipeline.structured_sid_batch = observed_batch_builder

    controlled_saver = observer.wrap_checkpoint_saver(original_saver)
    if snapshot_queue is not None:

        def snapshot_aware_saver(*save_args: Any, **save_kwargs: Any) -> Any:
            snapshot_queue.capture_saver_template(
                original_saver, save_args, save_kwargs
            )
            checkpoint_name = Path(save_args[0]).name
            if checkpoint_name == "best":
                snapshot_queue.flush(
                    observer.model, torch, reason="before_dev_best_save"
                )
            elif checkpoint_name == "last":
                snapshot_queue.flush(
                    observer.model, torch, reason="before_terminal_best_restore"
                )
            return controlled_saver(*save_args, **save_kwargs)

        pipeline.save_structured_decoder_checkpoint = snapshot_aware_saver
    else:
        pipeline.save_structured_decoder_checkpoint = controlled_saver

    train_dir = args.output_dir / "train"
    original_argv = sys.argv
    try:
        sys.argv = pipeline_training_argv(args, train_dir)
        pipeline.main()
    finally:
        sys.argv = original_argv
        torch.nn.Dropout = original_dropout
        torch.optim.AdamW = original_adamw

    observer.validate()
    snapshot_payload = (
        snapshot_queue.validate()
        if snapshot_queue is not None
        else {"enabled": False}
    )
    prefix_rank_execution = (
        prefix_rank_runtime.validate()
        if prefix_rank_runtime is not None
        else {
            "expander_calls": 0,
            "query_train_batch_calls": 0,
            "query_train_row_presentations": 0,
            "per_row_presentations": 0,
            "eligible_pair_presentations": 0,
            "active_comparisons": 0,
            "active_depths": [],
        }
    )
    expected_trace_steps = (
        {"1", "50", "500", str(args.query_steps), "score_parity"}
        if prefix_rank_active
        else set()
    )
    if set(prefix_rank_trace) != expected_trace_steps:
        raise ValueError(
            f"prefix ranking trace mismatch: observed={sorted(prefix_rank_trace)}"
        )
    if prefix_rank_active:
        score_parity = prefix_rank_trace["score_parity"]
        if args.prefix_rank_grouping == "fixed":
            parity_passed = set(score_parity) == {"positive", "negative"} and all(
                row.get("passed") for row in score_parity.values()
            )
        else:
            parity_passed = set(score_parity) == {"single", "multi"} and all(
                set(group_rows) == {"positive", "negative"}
                and all(row.get("passed") for row in group_rows.values())
                for group_rows in score_parity.values()
            )
        if not parity_passed:
            raise ValueError(f"prefix ranking score parity failed: {score_parity}")
    sampled_losses = [
        row for key, row in prefix_rank_trace.items() if key != "score_parity"
    ]
    if any(not row["finite"] for row in sampled_losses):
        raise ValueError(f"non-finite prefix ranking trace: {prefix_rank_trace}")
    dropout_rates = sorted({float(module.p) for module in dropout_modules})
    if len(dropout_modules) != EXPECTED_DROPOUT_MODULES or dropout_rates != [
        args.dropout_rate
    ]:
        raise ValueError(
            f"dropout override mismatch: count={len(dropout_modules)}, rates={dropout_rates}"
        )
    if args.path == "default":
        for name, expected in EXPECTED_DEFAULT_CATALOG_LOSSES.items():
            if observer.losses[name] != expected:
                raise ValueError(
                    f"default {name} loss mismatch: expected={expected}, observed={observer.losses[name]}"
                )

    manifest_path = train_dir / "single_sid_decoder_train_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        **EXPECTED_MANIFEST_ROW_COUNTS,
        "catalog_steps": args.catalog_steps,
        "query_steps": args.query_steps,
        "batch_size": args.batch_size,
        "max_source_length": args.max_source_length,
        "seed": args.seed,
        "lr": 3e-4,
        "weight_decay": 0.0,
        "load_best_model_at_end": True,
        "loaded_optimizer_state": False,
        "init_model_dir": None,
        "formulation": "catalog indexing followed by query->single SID finetuning",
    }
    observed_manifest = {key: manifest.get(key) for key in expected_manifest}
    if observed_manifest != expected_manifest:
        raise ValueError(
            f"training manifest mismatch: expected={expected_manifest}, observed={observed_manifest}"
        )

    events = [
        json.loads(line)
        for line in (train_dir / "loss_history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    history_summary = validate_loss_history(
        events,
        query_lr=args.query_lr,
        query_steps=args.query_steps,
        eval_steps=args.eval_steps,
        query_warmup_steps=args.query_warmup_steps,
    )
    if manifest["query_best_dev_loss"] != history_summary["best_dev_loss"]:
        raise ValueError(
            "best dev loss mismatch between manifest and history: "
            f"manifest={manifest['query_best_dev_loss']}, history={history_summary['best_dev_loss']}"
        )
    if Path(str(manifest["best_checkpoint"])).name != "best":
        raise ValueError(f"unexpected best checkpoint: {manifest['best_checkpoint']}")
    if (train_dir / "last").exists():
        raise ValueError(
            "last checkpoint should be suppressed for the full-query sweep"
        )

    best_checkpoint = checkpoint_state_summary(train_dir / "best")
    final_checkpoint = checkpoint_state_summary(train_dir / "final")
    if (
        best_checkpoint["optimizer_state_present"]
        or final_checkpoint["optimizer_state_present"]
    ):
        raise ValueError("model-only checkpoint policy was not applied")
    if best_checkpoint["model_state_sha256"] != final_checkpoint["model_state_sha256"]:
        raise ValueError("final model does not match selected best model")
    if observer.model is None:
        raise RuntimeError("pipeline did not expose final model")
    final_live_hash = _hash_tensor_state(observer.model.state_dict())
    if final_live_hash != final_checkpoint["model_state_sha256"]:
        raise ValueError("live final model does not match saved final checkpoint")

    weight_file = selected_weight_file(args.model)
    condition = {
        "arm": args.arm,
        "role": args.role,
        "path": args.path,
        "optimizer_mode": args.optimizer_mode,
        "query_lr": args.query_lr,
        "query_warmup_steps": args.query_warmup_steps,
        "query_max_grad_norm": args.query_max_grad_norm,
        "catalog_steps": args.catalog_steps,
        "query_steps": args.query_steps,
        "eval_steps": args.eval_steps,
        "batch_size": args.batch_size,
        "max_source_length": args.max_source_length,
        "seed": args.seed,
        "catalog_lr": 3e-4,
        "weight_decay": 0.0,
        "dropout_rate": args.dropout_rate,
        "prefix_ranking": {
            "grouping": args.prefix_rank_grouping,
            "scope": args.prefix_rank_scope,
            "weight": args.prefix_rank_weight,
            "single": {
                "scope": args.prefix_rank_single_scope,
                "weight": args.prefix_rank_single_weight,
            },
            "multi": {
                "scope": args.prefix_rank_multi_scope,
                "weight": args.prefix_rank_multi_weight,
            },
            "temperature": args.prefix_rank_temperature,
            "summary_sha256": args.prefix_rank_summary_sha256,
            "manifest_sha256": args.prefix_rank_manifest_sha256,
            "candidate_domain": "teacher_top50_exclude_all_sibling_gt",
            "score": "mean_depth_softplus_negative_minus_positive_cumulative_logprob",
            "conditional_normalization": "mean_over_active_depths_per_row_then_weight_then_mean_over_eligible_rows",
        },
        "outer_precision": "bf16",
        "requested_controls": {
            "bf16_reduction": args.bf16_reduction,
            "tf32_matmul": args.tf32_matmul,
            "tf32_cudnn": args.tf32_cudnn,
            "float32_matmul_precision": args.float32_matmul_precision,
            "preferred_blas": args.preferred_blas,
            "deterministic_algorithms": args.deterministic_algorithms,
            "cublas_workspace_config": args.cublas_workspace_config,
            "cuda_module_loading": args.cuda_module_loading,
        },
    }
    payload = {
        "artifact_format": "legacy_full_query_optimizer_lr_arm_v2",
        "condition": condition,
        "condition_sha256": stable_json_sha256(condition),
        "runtime": runtime_snapshot(torch),
        "precision": precision_snapshot(torch),
        "configured_precision": configured_precision,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "file_sha256": {
            "wrapper": file_sha256(Path(__file__).resolve()),
            "pipeline": file_sha256(args.pipeline),
            "rows": file_sha256(args.rows),
            "dev_rows": file_sha256(args.dev_rows),
            "skills": file_sha256(args.skills_jsonl),
            "semantic_ids": file_sha256(args.semantic_ids),
            "model_weight": file_sha256(weight_file),
            "prefix_rank_summary": candidate_input_hashes["summary"],
            "prefix_rank_manifest": candidate_input_hashes["manifest"],
        },
        "initial": observer.initial,
        "batches": observer.batches,
        "losses": observer.losses,
        "dropout": {
            "requested_rate": args.dropout_rate,
            "module_count": len(dropout_modules),
            "module_rates": dropout_rates,
        },
        "catalog_boundary": observer.catalog,
        "optimizer_transition": observer.transition,
        "query_step_1": observer.query_step_1,
        "query_stability": {
            "lr_trace": observer.query_lr_trace,
            "gradient_clipping": observer.clipping,
        },
        "prefix_ranking": {
            "execution": prefix_rank_execution,
            "trace": prefix_rank_trace,
        },
        "training": {
            "history": history_summary,
            "manifest": manifest,
            "final_live_model_state_sha256": final_live_hash,
        },
        "checkpoints": {
            "policy": "model_only_best_and_final; last_suppressed",
            "save_calls": observer.checkpoint_saves,
            "best": best_checkpoint,
            "final": final_checkpoint,
        },
        "checkpoint_oracle_snapshots": snapshot_payload,
        "execution": {
            "batch_calls": observer.batch_calls,
            "forward_calls": observer.forward_calls,
            "optimizer_steps": observer.optimizer_steps,
        },
        "memory": {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }
    output_path = args.output_dir / "full_query_arm.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, output_path)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "query_lr": args.query_lr,
                "optimizer_mode": args.optimizer_mode,
                "query_warmup_steps": args.query_warmup_steps,
                "query_max_grad_norm": args.query_max_grad_norm,
                "prefix_rank_scope": args.prefix_rank_scope,
                "prefix_rank_weight": args.prefix_rank_weight,
                "prefix_rank_grouping": args.prefix_rank_grouping,
                "prefix_rank_single_scope": args.prefix_rank_single_scope,
                "prefix_rank_single_weight": args.prefix_rank_single_weight,
                "prefix_rank_multi_scope": args.prefix_rank_multi_scope,
                "prefix_rank_multi_weight": args.prefix_rank_multi_weight,
                "best_step": history_summary["best_step"],
                "best_dev_loss": history_summary["best_dev_loss"],
                "final_model_sha256": final_checkpoint["model_state_sha256"]["full"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--role", choices=("selection", "replica", "anchor", "pair"), required=True
    )
    parser.add_argument("--path", choices=("default", "workspace_16"), required=True)
    parser.add_argument("--optimizer-mode", choices=OPTIMIZER_MODES, required=True)
    parser.add_argument("--query-lr", type=float, required=True)
    parser.add_argument("--query-warmup-steps", type=int, default=0)
    parser.add_argument("--query-max-grad-norm", type=float)
    parser.add_argument(
        "--prefix-rank-scope", choices=tuple(SCOPE_DEPTHS), default="disabled"
    )
    parser.add_argument("--prefix-rank-weight", type=float, default=0.0)
    parser.add_argument("--prefix-rank-grouping", choices=GROUPINGS, default="fixed")
    parser.add_argument(
        "--prefix-rank-single-scope",
        choices=tuple(SCOPE_DEPTHS),
        default="sequence_only",
    )
    parser.add_argument("--prefix-rank-single-weight", type=float, default=0.0)
    parser.add_argument(
        "--prefix-rank-multi-scope", choices=tuple(SCOPE_DEPTHS), default="cumulative"
    )
    parser.add_argument("--prefix-rank-multi-weight", type=float, default=0.0)
    parser.add_argument("--prefix-rank-temperature", type=float, default=1.0)
    parser.add_argument("--prefix-rank-summary", type=Path)
    parser.add_argument("--prefix-rank-summary-sha256")
    parser.add_argument("--prefix-rank-manifest", type=Path)
    parser.add_argument("--prefix-rank-manifest-sha256")
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog-steps", type=int, default=CATALOG_STEPS)
    parser.add_argument("--query-steps", type=int, default=QUERY_STEPS)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--oracle-checkpoint-queue", type=Path)
    parser.add_argument("--oracle-max-queued", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument(
        "--bf16-reduction", choices=OPTIONAL_BOOL_CHOICES, default="auto"
    )
    parser.add_argument("--tf32-matmul", choices=OPTIONAL_BOOL_CHOICES, default="auto")
    parser.add_argument("--tf32-cudnn", choices=OPTIONAL_BOOL_CHOICES, default="auto")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("auto", "highest", "high", "medium"),
        default="auto",
    )
    parser.add_argument(
        "--preferred-blas",
        choices=("auto", "default", "cublas", "cublaslt"),
        default="auto",
    )
    parser.add_argument(
        "--deterministic-algorithms", choices=OPTIONAL_BOOL_CHOICES, default="auto"
    )
    parser.add_argument(
        "--cublas-workspace-config",
        choices=("auto", ":4096:8", ":16:8"),
        default="auto",
    )
    parser.add_argument(
        "--cuda-module-loading",
        choices=("auto", "LAZY", "EAGER"),
        default="auto",
    )
    parser.set_defaults(precision="bf16", func=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.query_lr <= 0.0:
        raise ValueError(f"query lr must be positive, got {args.query_lr}")
    if args.query_warmup_steps < 0:
        raise ValueError(
            f"query warmup steps must be non-negative, got {args.query_warmup_steps}"
        )
    if args.query_max_grad_norm is not None and (
        not math.isfinite(args.query_max_grad_norm) or args.query_max_grad_norm <= 0.0
    ):
        raise ValueError(
            "query max grad norm must be finite and positive, "
            f"got {args.query_max_grad_norm}"
        )
    if args.query_warmup_steps and args.query_max_grad_norm is not None:
        raise ValueError("query warmup and gradient clipping must be tested separately")
    if not 0.0 <= args.dropout_rate <= 1.0:
        raise ValueError(f"dropout-rate must be in [0, 1], got {args.dropout_rate}")
    args.func(args)


if __name__ == "__main__":
    main()
