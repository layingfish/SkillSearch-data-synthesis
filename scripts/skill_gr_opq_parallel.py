#!/usr/bin/env python3
"""Architecture-aligned parallel-MTP decoder for fixed OPQ skill DocIDs.

The existing LETTER decoder treats identifiers as autoregressive paths.  OPQ
subquantizers are independent, so this branch predicts every code position
from the same T5 decoder-start representation and ranks the complete catalog
by the mean per-position log probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import skill_gr_pipeline as pipeline
from distributed_utils import (
    DistributedContext,
    all_gather_object,
    barrier,
    destroy_distributed,
    distributed_epoch_batches_no_padding,
    global_normalized_loss,
    initialize_distributed,
    reduce_sum_count,
    shard_with_indices,
    unwrap_model,
    wrap_ddp,
)


LINEAR_HEAD = "linear"
RPG_HEAD = "rpg_cosine_residual"
RPG_CALIBRATED_HEAD = "rpg_cosine_residual_calibrated"
HEAD_TYPES = (LINEAR_HEAD, RPG_HEAD, RPG_CALIBRATED_HEAD)
LINEAR_ARCHITECTURE = "t5_decoder_start_parallel_position_heads_v1"
RPG_ARCHITECTURE = "t5_decoder_start_rpg_cosine_residual_heads_v1"
RPG_CALIBRATED_ARCHITECTURE = (
    "t5_decoder_start_rpg_cosine_residual_position_calibration_v1"
)
# Backward-compatible name used by the completed Phase 11 runner and tests.
ARCHITECTURE = LINEAR_ARCHITECTURE
TRAINING_OBJECTIVE = "mean_position_cross_entropy_mtp_v1"
CATALOG_SCORING = "exact_full_catalog_mean_position_log_probability_v1"
CALIBRATED_TRAINING_OBJECTIVE = (
    "mean_position_ce_plus_exact_catalog_nll_position_calibration_v1"
)
CALIBRATED_CATALOG_SCORING = (
    "exact_full_catalog_learned_weighted_position_log_probability_v1"
)
CHECKPOINT_NAME = "parallel_decoder.pt"


def architecture_for_head(head_type: str) -> str:
    if head_type == LINEAR_HEAD:
        return LINEAR_ARCHITECTURE
    if head_type == RPG_HEAD:
        return RPG_ARCHITECTURE
    if head_type == RPG_CALIBRATED_HEAD:
        return RPG_CALIBRATED_ARCHITECTURE
    raise ValueError(f"unknown OPQ parallel head type: {head_type}")


def training_objective_for_head(head_type: str) -> str:
    if head_type == RPG_CALIBRATED_HEAD:
        return CALIBRATED_TRAINING_OBJECTIVE
    if head_type in (LINEAR_HEAD, RPG_HEAD):
        return TRAINING_OBJECTIVE
    raise ValueError(f"unknown OPQ parallel head type: {head_type}")


def catalog_scoring_for_head(head_type: str) -> str:
    if head_type == RPG_CALIBRATED_HEAD:
        return CALIBRATED_CATALOG_SCORING
    if head_type in (LINEAR_HEAD, RPG_HEAD):
        return CATALOG_SCORING
    raise ValueError(f"unknown OPQ parallel head type: {head_type}")


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_parallel_semantic_ids(
    semantic_ids: dict[str, list[int]],
    semantic_id_mode: str,
    *,
    depth: int,
    codebook_size: int,
    expected_catalog_size: int | None = None,
) -> dict[str, Any]:
    mode = pipeline.normalize_semantic_id_mode(semantic_id_mode)
    if mode != pipeline.SEMANTIC_ID_MODE_LEARNED:
        raise ValueError(
            "parallel OPQ requires fixed learned_codes without an autoregressive dedup suffix; "
            f"got {mode}"
        )
    if not semantic_ids:
        raise ValueError("parallel OPQ semantic IDs are empty")
    if expected_catalog_size is not None and len(semantic_ids) != expected_catalog_size:
        raise ValueError(
            f"parallel OPQ catalog size mismatch: expected={expected_catalog_size}, "
            f"actual={len(semantic_ids)}"
        )
    lengths = {len(values) for values in semantic_ids.values()}
    if lengths != {depth}:
        raise ValueError(
            f"parallel OPQ depth mismatch: expected={depth}, actual={sorted(lengths)}"
        )
    invalid = [
        sid
        for sid, values in semantic_ids.items()
        if any(int(value) < 0 or int(value) >= codebook_size for value in values)
    ]
    if invalid:
        raise ValueError(
            f"parallel OPQ code outside [0,{codebook_size}): {invalid[:5]}"
        )
    tuples = [tuple(int(value) for value in values) for values in semantic_ids.values()]
    unique_count = len(set(tuples))
    if unique_count != len(tuples):
        raise ValueError(
            "parallel exact catalog scoring requires collision-free resolved DocIDs: "
            f"unique={unique_count}, total={len(tuples)}"
        )
    return {
        "semantic_id_mode": mode,
        "depth": depth,
        "codebook_size": codebook_size,
        "skills": len(semantic_ids),
        "unique_full_tuples": unique_count,
        "semantic_ids_sha256": pipeline.semantic_ids_sha256(semantic_ids, mode),
    }


def parallel_mtp_loss_components(logits, labels, sample_weights=None):
    """Return the weighted sum of per-example mean-position CE and its count."""
    import torch
    import torch.nn.functional as F

    if logits.ndim != 3:
        raise ValueError(
            f"parallel logits must be [batch,depth,codes], got {tuple(logits.shape)}"
        )
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"parallel labels must match [batch,depth]: {tuple(labels.shape)} != {tuple(logits.shape[:2])}"
        )
    batch_size, depth, codebook_size = logits.shape
    if depth <= 0 or codebook_size <= 1:
        raise ValueError(
            "parallel logits require positive depth and at least two code values"
        )
    if bool(((labels < 0) | (labels >= codebook_size)).any()):
        raise ValueError("parallel labels contain an out-of-range code")
    if sample_weights is None:
        sample_weights = torch.ones(
            batch_size, dtype=torch.float32, device=logits.device
        )
    if sample_weights.shape != (batch_size,):
        raise ValueError(
            f"sample weights must be [batch], got {tuple(sample_weights.shape)} for batch={batch_size}"
        )
    row_loss = (
        F.cross_entropy(
            logits.float().reshape(batch_size * depth, codebook_size),
            labels.reshape(batch_size * depth),
            reduction="none",
        )
        .reshape(batch_size, depth)
        .mean(dim=1)
    )
    weights = sample_weights.to(device=logits.device, dtype=row_loss.dtype)
    return (row_loss * weights).sum(), weights.sum(), row_loss


def normalized_position_weights(position_weights, *, depth: int, device):
    """Validate and normalize non-negative position weights to sum to one."""
    import torch

    if position_weights is None:
        return torch.full((depth,), 1.0 / float(depth), device=device)
    weights = position_weights.to(device=device, dtype=torch.float32)
    if weights.shape != (depth,):
        raise ValueError(
            f"position weights must be [depth], got {tuple(weights.shape)} for depth={depth}"
        )
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("position weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("position weights must be non-negative")
    total = weights.sum()
    if not bool(total > 0):
        raise ValueError("position weights must have a positive sum")
    return weights / total


def exact_catalog_log_scores(position_logits, catalog_codes, position_weights=None):
    """Score every catalog tuple by normalized weighted position log probability."""
    import torch
    import torch.nn.functional as F

    if position_logits.ndim != 3:
        raise ValueError(
            f"parallel logits must be [batch,depth,codes], got {tuple(position_logits.shape)}"
        )
    if catalog_codes.ndim != 2:
        raise ValueError(
            f"catalog codes must be [items,depth], got {tuple(catalog_codes.shape)}"
        )
    batch_size, depth, codebook_size = position_logits.shape
    if catalog_codes.shape[1] != depth:
        raise ValueError(
            f"catalog/model depth mismatch: catalog={catalog_codes.shape[1]}, model={depth}"
        )
    if bool(((catalog_codes < 0) | (catalog_codes >= codebook_size)).any()):
        raise ValueError("catalog contains an out-of-range code")
    weights = normalized_position_weights(
        position_weights, depth=depth, device=position_logits.device
    )
    log_probs = F.log_softmax(position_logits.float(), dim=-1)
    scores = torch.zeros(
        (batch_size, catalog_codes.shape[0]),
        dtype=torch.float32,
        device=position_logits.device,
    )
    for position in range(depth):
        scores = scores + weights[position] * log_probs[:, position, :].index_select(
            1, catalog_codes[:, position]
        )
    return scores


def exact_catalog_nll_components(
    position_logits,
    catalog_codes,
    target_catalog_indices,
    *,
    position_weights=None,
    sample_weights=None,
):
    """Return weighted full-catalog NLL using the product of position likelihoods."""
    import torch
    import torch.nn.functional as F

    if target_catalog_indices.shape != (position_logits.shape[0],):
        raise ValueError(
            "catalog targets must be [batch], got "
            f"{tuple(target_catalog_indices.shape)} for batch={position_logits.shape[0]}"
        )
    if bool(
        (
            (target_catalog_indices < 0)
            | (target_catalog_indices >= catalog_codes.shape[0])
        ).any()
    ):
        raise ValueError("catalog targets contain an out-of-range item index")
    scores = exact_catalog_log_scores(
        position_logits, catalog_codes, position_weights=position_weights
    )
    # exact_catalog_log_scores is a normalized mean so historical ranking scores
    # keep their scale. Multiplying by depth recovers the product-model logit for
    # a document-level softmax objective.
    row_loss = F.cross_entropy(
        scores * float(position_logits.shape[1]),
        target_catalog_indices,
        reduction="none",
    )
    if sample_weights is None:
        sample_weights = torch.ones_like(row_loss)
    if sample_weights.shape != row_loss.shape:
        raise ValueError(
            f"sample weights must be [batch], got {tuple(sample_weights.shape)}"
        )
    weights = sample_weights.to(device=row_loss.device, dtype=row_loss.dtype)
    return (row_loss * weights).sum(), weights.sum(), row_loss


def position_weight_kl_to_uniform(position_weights):
    """KL(weight || uniform), used only to prevent degenerate slot collapse."""
    import torch

    weights = normalized_position_weights(
        position_weights,
        depth=int(position_weights.numel()),
        device=position_weights.device,
    )
    return torch.sum(weights * torch.log(weights.clamp_min(1e-12) * weights.numel()))


def scheduled_learning_rate(
    *, step: int, total_steps: int, peak_lr: float, min_lr: float, schedule: str
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 1 or step > total_steps:
        raise ValueError(f"step must be in [1,{total_steps}], got {step}")
    if peak_lr <= 0 or min_lr < 0 or min_lr > peak_lr:
        raise ValueError(f"invalid learning-rate bounds: peak={peak_lr} min={min_lr}")
    if schedule == "constant":
        return float(peak_lr)
    if schedule != "cosine":
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    if total_steps == 1:
        return float(min_lr)
    progress = float(step - 1) / float(total_steps - 1)
    return float(
        min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    )


def stable_topk_catalog(
    scores: list[float],
    skill_ids: list[str],
    top_k: int,
) -> tuple[list[str], list[float]]:
    """Rank exact scores descending with skill ID as a deterministic tie break."""
    if len(scores) != len(skill_ids):
        raise ValueError(
            f"score/catalog length mismatch: {len(scores)} != {len(skill_ids)}"
        )
    if top_k <= 0 or top_k > len(skill_ids):
        raise ValueError(f"top_k must be in [1,{len(skill_ids)}], got {top_k}")
    order = sorted(
        range(len(skill_ids)),
        key=lambda index: (-float(scores[index]), skill_ids[index]),
    )
    selected = order[:top_k]
    return [skill_ids[index] for index in selected], [
        float(scores[index]) for index in selected
    ]


def make_parallel_t5_model(
    base_model: str,
    depth: int,
    codebook_size: int,
    *,
    head_type: str = LINEAR_HEAD,
    temperature: float = 0.03,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForSeq2SeqLM

    if head_type not in HEAD_TYPES:
        raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type}")
    if not temperature > 0:
        raise ValueError("temperature must be positive")

    class ResidualProjection(nn.Module):
        """RPG-style position projection, initialized as an identity map."""

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        def forward(self, hidden):
            return hidden + F.silu(self.linear(hidden))

    class ParallelT5OPQModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModelForSeq2SeqLM.from_pretrained(base_model)
            self.depth = int(depth)
            self.codebook_size = int(codebook_size)
            self.head_type = head_type
            self.temperature = float(temperature)
            d_model = int(self.backbone.config.d_model)
            if self.head_type == LINEAR_HEAD:
                self.position_heads = nn.ModuleList(
                    nn.Linear(d_model, self.codebook_size, bias=True)
                    for _ in range(self.depth)
                )
            else:
                self.position_projections = nn.ModuleList(
                    ResidualProjection(d_model) for _ in range(self.depth)
                )
                self.code_prototypes = nn.Parameter(
                    torch.empty(self.depth, self.codebook_size, d_model)
                )
                nn.init.normal_(
                    self.code_prototypes,
                    mean=0.0,
                    std=float(getattr(self.backbone.config, "initializer_factor", 1.0))
                    * 0.02,
                )
                if self.head_type == RPG_CALIBRATED_HEAD:
                    self.position_log_temperatures = nn.Parameter(
                        torch.full((self.depth,), math.log(self.temperature))
                    )
                    self.position_weight_logits = nn.Parameter(torch.zeros(self.depth))

        def position_temperatures(self):
            if self.head_type == RPG_CALIBRATED_HEAD:
                return self.position_log_temperatures.exp().clamp(0.003, 0.3)
            return torch.full(
                (self.depth,),
                self.temperature,
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )

        def position_score_weights(self):
            if self.head_type == RPG_CALIBRATED_HEAD:
                return torch.softmax(self.position_weight_logits.float(), dim=0)
            return torch.full(
                (self.depth,),
                1.0 / float(self.depth),
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )

        def shared_query_representation(self, input_ids, attention_mask):
            encoder_outputs = self.backbone.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            start_id = self.backbone.config.decoder_start_token_id
            if start_id is None:
                start_id = self.backbone.config.pad_token_id
            decoder_input_ids = torch.full(
                (input_ids.shape[0], 1),
                int(start_id),
                dtype=torch.long,
                device=input_ids.device,
            )
            decoder_outputs = self.backbone.decoder(
                input_ids=decoder_input_ids,
                attention_mask=torch.ones_like(decoder_input_ids),
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            return decoder_outputs.last_hidden_state[:, 0, :]

        def forward(
            self, input_ids, attention_mask, *, return_position_weights: bool = False
        ):
            representation = self.shared_query_representation(input_ids, attention_mask)
            if self.head_type == LINEAR_HEAD:
                logits = torch.stack(
                    [head(representation) for head in self.position_heads],
                    dim=1,
                )
            else:
                projected = torch.stack(
                    [head(representation) for head in self.position_projections], dim=1
                )
                projected = F.normalize(projected.float(), dim=-1)
                prototypes = F.normalize(self.code_prototypes.float(), dim=-1)
                similarities = torch.einsum("bpd,pcd->bpc", projected, prototypes)
                logits = similarities / self.position_temperatures()[None, :, None]
            if return_position_weights:
                return logits, self.position_score_weights()
            return logits

    return ParallelT5OPQModel()


def batch_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    semantic_ids: dict[str, list[int]],
    *,
    depth: int,
    max_source_length: int,
    device,
    zero_weight: bool = False,
) -> dict[str, Any]:
    import torch

    if not rows:
        raise ValueError("cannot tokenize an empty parallel-MTP batch")
    sources = [pipeline.decoder_source_text(row) for row in rows]
    if any(not source for source in sources):
        raise ValueError("parallel-MTP batch contains an empty source text")
    selected: list[str] = []
    for row in rows:
        values = [str(value) for value in row.get("selected_skill_ids", [])]
        if len(values) != 1:
            raise ValueError(
                "parallel-MTP preserves expanded single-SID training and requires one target per row; "
                f"row={row.get('index')} targets={len(values)}"
            )
        if values[0] not in semantic_ids:
            raise KeyError(f"missing OPQ identifier for target skill: {values[0]}")
        selected.append(values[0])
    encoded = tokenizer(
        sources,
        padding=True,
        truncation=True,
        max_length=max_source_length,
        return_tensors="pt",
    )
    labels = torch.tensor(
        [[int(value) for value in semantic_ids[sid]] for sid in selected],
        dtype=torch.long,
        device=device,
    )
    if labels.shape[1] != depth:
        raise ValueError(
            f"batch OPQ depth mismatch: expected={depth}, actual={labels.shape[1]}"
        )
    weights = torch.tensor(
        [float(row.get("_distributed_sample_weight", 1.0)) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    if zero_weight:
        weights.zero_()
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels,
        "sample_weights": weights,
        "selected_skill_ids": selected,
    }


def checkpoint_payload(model, metadata: dict[str, Any], **extra: Any) -> dict[str, Any]:
    state = {
        key: value.detach().cpu()
        for key, value in unwrap_model(model).state_dict().items()
    }
    return {
        **metadata,
        **extra,
        "model_state_dict": state,
    }


def load_initializer_checkpoint(
    model,
    checkpoint: dict[str, Any],
    *,
    target_head_type: str,
    base_model: str,
    depth: int,
    codebook_size: int,
    semantic_ids_sha256: str,
) -> dict[str, Any]:
    """Load a same-head continuation or upgrade RPG to calibrated RPG."""
    source_head_type = str(checkpoint.get("head_type", LINEAR_HEAD))
    if checkpoint.get("architecture") != architecture_for_head(source_head_type):
        raise ValueError("initializer checkpoint architecture contract mismatch")
    if checkpoint.get("catalog_scoring") != catalog_scoring_for_head(source_head_type):
        raise ValueError("initializer checkpoint catalog-scoring contract mismatch")
    if int(checkpoint.get("depth", -1)) != depth:
        raise ValueError("initializer checkpoint depth mismatch")
    if int(checkpoint.get("codebook_size", -1)) != codebook_size:
        raise ValueError("initializer checkpoint codebook-size mismatch")
    if checkpoint.get("semantic_ids_sha256") != semantic_ids_sha256:
        raise ValueError("initializer checkpoint semantic-ID hash mismatch")
    if str(checkpoint.get("base_model")) != str(base_model):
        raise ValueError("initializer checkpoint base-model mismatch")
    allowed_upgrade = (
        source_head_type == RPG_HEAD and target_head_type == RPG_CALIBRATED_HEAD
    )
    if source_head_type != target_head_type and not allowed_upgrade:
        raise ValueError(
            f"unsupported initializer head transition: {source_head_type} -> {target_head_type}"
        )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    expected_missing = (
        {"position_log_temperatures", "position_weight_logits"}
        if allowed_upgrade
        else set()
    )
    if missing != expected_missing or unexpected:
        raise ValueError(
            "initializer state-dict mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return {
        "source_head_type": source_head_type,
        "target_head_type": target_head_type,
        "upgrade": bool(allowed_upgrade),
        "source_checkpoint_role": checkpoint.get("checkpoint_role"),
        "source_best_query_step": checkpoint.get("best_query_step"),
    }


def distributed_eval_loss(
    model,
    rows: list[dict[str, Any]],
    tokenizer,
    semantic_ids: dict[str, list[int]],
    *,
    depth: int,
    max_source_length: int,
    batch_size: int,
    context: DistributedContext,
) -> float:
    import torch

    core_model = unwrap_model(model)
    local_rows = [row for _, row in shard_with_indices(rows, context)]
    was_training = core_model.training
    core_model.eval()
    local_sum = torch.zeros((), dtype=torch.float32, device=context.device)
    local_count = torch.zeros((), dtype=torch.float32, device=context.device)
    with torch.no_grad():
        for start in range(0, len(local_rows), batch_size):
            batch = batch_rows(
                local_rows[start : start + batch_size],
                tokenizer,
                semantic_ids,
                depth=depth,
                max_source_length=max_source_length,
                device=context.device,
            )
            logits = core_model(batch["input_ids"], batch["attention_mask"])
            loss_sum, count, _ = parallel_mtp_loss_components(
                logits, batch["labels"], batch["sample_weights"]
            )
            local_sum = local_sum + loss_sum.detach()
            local_count = local_count + count.detach()
    total_sum, total_count = reduce_sum_count(local_sum, local_count, context)
    core_model.train(was_training)
    return total_sum / max(1, total_count)


def train_stage(
    *,
    stage_name: str,
    rows: list[dict[str, Any]],
    epochs: int,
    model,
    optimizer,
    tokenizer,
    semantic_ids: dict[str, list[int]],
    depth: int,
    max_source_length: int,
    batch_size: int,
    bf16: bool,
    seed: int,
    logging_steps: int,
    context: DistributedContext,
    loss_history_path: Path,
    dev_rows: list[dict[str, Any]] | None = None,
    eval_batch_size: int = 16,
    eval_steps: int = 1000,
    checkpoint_metadata: dict[str, Any] | None = None,
    best_state: dict[str, Any] | None = None,
    lr_schedule: str = "constant",
    peak_lr: float = 3e-4,
    min_lr: float = 0.0,
    max_grad_norm: float = 0.0,
    epoch_offset: int = 0,
    catalog_codes=None,
    catalog_skill_index: dict[str, int] | None = None,
    catalog_loss_weight: float = 0.0,
    position_weight_kl_weight: float = 0.0,
) -> dict[str, Any]:
    import torch

    if epochs < 0:
        raise ValueError(f"{stage_name} epochs must be non-negative")
    if epochs == 0:
        return {"steps": 0, "epochs": 0, "last_loss": None, "epoch_metrics": []}
    if epoch_offset < 0:
        raise ValueError("epoch_offset must be non-negative")
    if max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative")
    if catalog_loss_weight < 0 or position_weight_kl_weight < 0:
        raise ValueError("catalog/position regularization weights must be non-negative")
    if catalog_loss_weight > 0 and (
        catalog_codes is None or catalog_skill_index is None
    ):
        raise ValueError("catalog loss requires catalog codes and skill index")
    steps_per_epoch = math.ceil(len(rows) / (batch_size * context.world_size))
    total_steps = epochs * steps_per_epoch
    step = 0
    epoch_metrics: list[dict[str, Any]] = []
    last_global_loss: float | None = None
    last_global_catalog_nll: float | None = None
    last_eval_step = -1
    for epoch in range(epochs):
        local_epoch_sum = torch.zeros((), dtype=torch.float32, device=context.device)
        local_epoch_count = torch.zeros((), dtype=torch.float32, device=context.device)
        local_epoch_catalog_sum = torch.zeros(
            (), dtype=torch.float32, device=context.device
        )
        batches = distributed_epoch_batches_no_padding(
            rows,
            batch_size,
            context,
            seed=seed,
            epoch=epoch + epoch_offset,
        )
        for local_rows in batches:
            step += 1
            current_lr = scheduled_learning_rate(
                step=step,
                total_steps=total_steps,
                peak_lr=peak_lr,
                min_lr=min_lr,
                schedule=lr_schedule,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr
            zero_weight = not local_rows
            effective_rows = local_rows if local_rows else [rows[0]]
            batch = batch_rows(
                effective_rows,
                tokenizer,
                semantic_ids,
                depth=depth,
                max_source_length=max_source_length,
                device=context.device,
                zero_weight=zero_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=bool(bf16 and getattr(context.device, "type", None) == "cuda"),
            ):
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    return_position_weights=catalog_loss_weight > 0,
                )
            if catalog_loss_weight > 0:
                logits, position_weights = outputs
            else:
                logits = outputs
                position_weights = None
            local_loss_sum, local_count, _ = parallel_mtp_loss_components(
                logits, batch["labels"], batch["sample_weights"]
            )
            loss = global_normalized_loss(local_loss_sum, local_count, context)
            local_catalog_sum = torch.zeros_like(local_loss_sum)
            if catalog_loss_weight > 0:
                target_catalog_indices = torch.tensor(
                    [catalog_skill_index[sid] for sid in batch["selected_skill_ids"]],
                    dtype=torch.long,
                    device=context.device,
                )
                local_catalog_sum, catalog_count, _ = exact_catalog_nll_components(
                    logits,
                    catalog_codes,
                    target_catalog_indices,
                    position_weights=position_weights,
                    sample_weights=batch["sample_weights"],
                )
                catalog_loss = global_normalized_loss(
                    local_catalog_sum, catalog_count, context
                )
                position_kl = position_weight_kl_to_uniform(position_weights)
                loss = (
                    loss
                    + catalog_loss_weight * catalog_loss
                    + position_weight_kl_weight * position_kl
                )
            loss.backward()
            grad_norm = None
            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            optimizer.step()
            local_epoch_sum = local_epoch_sum + local_loss_sum.detach()
            local_epoch_count = local_epoch_count + local_count.detach()
            local_epoch_catalog_sum = (
                local_epoch_catalog_sum + local_catalog_sum.detach()
            )
            should_log = step % logging_steps == 0 or step == total_steps
            if should_log:
                global_sum, global_count = reduce_sum_count(
                    local_loss_sum, local_count, context
                )
                last_global_loss = global_sum / max(1, global_count)
                if context.is_main:
                    event = {
                        "stage": stage_name,
                        "event": "step",
                        "step": step,
                        "total_steps": total_steps,
                        "epoch": epoch_offset
                        + epoch
                        + (step - epoch * steps_per_epoch) / steps_per_epoch,
                        "mean_position_ce": last_global_loss,
                        "examples": global_count,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                    if grad_norm is not None:
                        event["grad_norm_before_clip"] = float(grad_norm.detach().cpu())
                    if catalog_loss_weight > 0:
                        global_catalog_sum, global_catalog_count = reduce_sum_count(
                            local_catalog_sum, local_count, context
                        )
                        last_global_catalog_nll = global_catalog_sum / max(
                            1, global_catalog_count
                        )
                        normalized_weights = normalized_position_weights(
                            position_weights.detach(),
                            depth=depth,
                            device=position_weights.device,
                        )
                        event.update(
                            {
                                "exact_catalog_nll": last_global_catalog_nll,
                                "position_weight_min": float(
                                    normalized_weights.min().cpu()
                                ),
                                "position_weight_max": float(
                                    normalized_weights.max().cpu()
                                ),
                            }
                        )
                    append_jsonl(loss_history_path, event)
                    print(json.dumps(event, sort_keys=True), flush=True)
                elif catalog_loss_weight > 0:
                    reduce_sum_count(local_catalog_sum, local_count, context)
            if (
                stage_name == "query"
                and dev_rows
                and eval_steps > 0
                and step % eval_steps == 0
            ):
                dev_loss = distributed_eval_loss(
                    model,
                    dev_rows,
                    tokenizer,
                    semantic_ids,
                    depth=depth,
                    max_source_length=max_source_length,
                    batch_size=eval_batch_size,
                    context=context,
                )
                last_eval_step = step
                improved = best_state is not None and (
                    best_state.get("loss") is None
                    or dev_loss < float(best_state["loss"])
                )
                if context.is_main:
                    event = {
                        "stage": stage_name,
                        "event": "eval",
                        "step": step,
                        "dev_mean_position_ce": dev_loss,
                        "best": bool(improved),
                    }
                    append_jsonl(loss_history_path, event)
                    print(json.dumps(event, sort_keys=True), flush=True)
                    if improved:
                        best_state["loss"] = dev_loss
                        best_state["step"] = step
                        atomic_torch_save(
                            checkpoint_payload(
                                model,
                                checkpoint_metadata or {},
                                checkpoint_role="best",
                                best_dev_loss=dev_loss,
                                best_query_step=step,
                            ),
                            Path(best_state["path"]),
                        )
                barrier(context)
        epoch_sum, epoch_count = reduce_sum_count(
            local_epoch_sum, local_epoch_count, context
        )
        epoch_event = {
            "stage": stage_name,
            "event": "epoch",
            "epoch": epoch_offset + epoch + 1,
            "steps": step,
            "mean_position_ce": epoch_sum / max(1, epoch_count),
            "examples": epoch_count,
        }
        if catalog_loss_weight > 0:
            epoch_catalog_sum, epoch_catalog_count = reduce_sum_count(
                local_epoch_catalog_sum, local_epoch_count, context
            )
            epoch_event["exact_catalog_nll"] = epoch_catalog_sum / max(
                1, epoch_catalog_count
            )
        epoch_metrics.append(epoch_event)
        if context.is_main:
            append_jsonl(loss_history_path, epoch_event)
            print(json.dumps(epoch_event, sort_keys=True), flush=True)
    if stage_name == "query" and dev_rows and last_eval_step != step:
        dev_loss = distributed_eval_loss(
            model,
            dev_rows,
            tokenizer,
            semantic_ids,
            depth=depth,
            max_source_length=max_source_length,
            batch_size=eval_batch_size,
            context=context,
        )
        improved = best_state is not None and (
            best_state.get("loss") is None or dev_loss < float(best_state["loss"])
        )
        if context.is_main:
            event = {
                "stage": stage_name,
                "event": "eval",
                "step": step,
                "dev_mean_position_ce": dev_loss,
                "best": bool(improved),
            }
            append_jsonl(loss_history_path, event)
            print(json.dumps(event, sort_keys=True), flush=True)
            if improved:
                best_state["loss"] = dev_loss
                best_state["step"] = step
                atomic_torch_save(
                    checkpoint_payload(
                        model,
                        checkpoint_metadata or {},
                        checkpoint_role="best",
                        best_dev_loss=dev_loss,
                        best_query_step=step,
                    ),
                    Path(best_state["path"]),
                )
        barrier(context)
    return {
        "steps": step,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "last_loss": last_global_loss,
        "last_catalog_nll": last_global_catalog_nll,
        "epoch_metrics": epoch_metrics,
    }


def command_train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    context = initialize_distributed(cpu=args.cpu)
    try:
        if args.batch_size * context.world_size != args.expected_global_batch_size:
            raise ValueError(
                "parallel OPQ global batch contract mismatch: "
                f"batch={args.batch_size} world={context.world_size} "
                f"actual={args.batch_size * context.world_size} "
                f"expected={args.expected_global_batch_size}"
            )
        if args.head_type == RPG_CALIBRATED_HEAD:
            if args.catalog_loss_weight <= 0:
                raise ValueError(
                    "calibrated RPG head requires positive catalog loss weight"
                )
        elif args.catalog_loss_weight != 0 or args.position_weight_kl_weight != 0:
            raise ValueError(
                "catalog/position calibration losses require the calibrated RPG head"
            )
        torch.manual_seed(args.seed + context.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + context.rank)
        semantic_ids, semantic_id_mode = pipeline.load_semantic_ids_with_mode(
            args.semantic_ids
        )
        sid_audit = validate_parallel_semantic_ids(
            semantic_ids,
            semantic_id_mode,
            depth=args.depth,
            codebook_size=args.codebook_size,
            expected_catalog_size=args.expected_catalog_size,
        )
        source_rows = pipeline.load_jsonl(args.rows)
        dev_source_rows = pipeline.load_jsonl(args.dev_rows)
        query_rows = pipeline.expand_single_sid_query_rows(source_rows, semantic_ids)
        dev_rows = pipeline.expand_single_sid_query_rows(dev_source_rows, semantic_ids)
        skills = pipeline.load_jsonl(args.skills_jsonl)
        catalog_rows = pipeline.build_single_sid_catalog_rows(
            skills, semantic_ids, args.catalog_input_prefix
        )
        if len(catalog_rows) != len(semantic_ids):
            raise ValueError(
                f"canonical skill/OPQ coverage mismatch: rows={len(catalog_rows)} ids={len(semantic_ids)}"
            )
        random.Random(args.seed).shuffle(catalog_rows)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if context.is_main:
            length_audits = {
                "query_train": pipeline.audit_source_token_lengths(
                    "query_train", query_rows, tokenizer, args.max_source_length, False
                ),
                "query_dev": pipeline.audit_source_token_lengths(
                    "query_dev", dev_rows, tokenizer, args.max_source_length, False
                ),
                "catalog": pipeline.audit_source_token_lengths(
                    "catalog", catalog_rows, tokenizer, args.max_source_length, False
                ),
            }
        else:
            length_audits = None
        barrier(context)
        architecture = architecture_for_head(args.head_type)
        training_objective = training_objective_for_head(args.head_type)
        catalog_scoring = catalog_scoring_for_head(args.head_type)
        core_model = make_parallel_t5_model(
            args.model,
            args.depth,
            args.codebook_size,
            head_type=args.head_type,
            temperature=args.temperature,
        )
        initializer_report = None
        if args.initial_checkpoint is not None:
            initializer = torch.load(
                args.initial_checkpoint, map_location="cpu", weights_only=False
            )
            initializer_report = load_initializer_checkpoint(
                core_model,
                initializer,
                target_head_type=args.head_type,
                base_model=str(args.model),
                depth=args.depth,
                codebook_size=args.codebook_size,
                semantic_ids_sha256=sid_audit["semantic_ids_sha256"],
            )
        core_model.to(context.device)
        model = wrap_ddp(core_model, context)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        metadata = {
            "format": "skill_gr_opq_parallel_checkpoint_v1",
            "architecture": architecture,
            "head_type": args.head_type,
            "temperature": args.temperature,
            "training_objective": training_objective,
            "catalog_scoring": catalog_scoring,
            "base_model": str(args.model),
            "depth": args.depth,
            "codebook_size": args.codebook_size,
            "semantic_id_mode": semantic_id_mode,
            "semantic_ids_sha256": sid_audit["semantic_ids_sha256"],
            "semantic_id_count": len(semantic_ids),
            "source_text_policy": pipeline.GR_SOURCE_TEXT_POLICY,
            "max_source_length": args.max_source_length,
            "seed": args.seed,
            "optimizer_schedule": {
                "optimizer": "AdamW",
                "name": args.lr_scheduler,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "catalog_epochs": args.catalog_epochs,
                "query_epochs": args.epochs,
                "query_epoch_offset": args.epoch_offset,
            },
            "position_calibration": {
                "enabled": args.head_type == RPG_CALIBRATED_HEAD,
                "catalog_loss_weight": args.catalog_loss_weight,
                "position_weight_kl_weight": args.position_weight_kl_weight,
                "temperature_initial": args.temperature,
                "temperature_bounds": [0.003, 0.3],
                "position_weight_parameterization": "softmax",
            },
            "ddp": {
                "world_size": context.world_size,
                "per_device_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * context.world_size,
                "sampler": "no_padding_graph_zero_tail_v1",
            },
            "inputs": {
                "rows": {"path": str(args.rows), "sha256": file_sha256(args.rows)},
                "dev_rows": {
                    "path": str(args.dev_rows),
                    "sha256": file_sha256(args.dev_rows),
                },
                "skills_jsonl": {
                    "path": str(args.skills_jsonl),
                    "sha256": file_sha256(args.skills_jsonl),
                },
                "semantic_ids": {
                    "path": str(args.semantic_ids),
                    "file_sha256": file_sha256(args.semantic_ids),
                    "logical_sha256": sid_audit["semantic_ids_sha256"],
                },
                "base_model_safetensors_sha256": file_sha256(
                    Path(args.model) / "model.safetensors"
                ),
            },
        }
        if args.initial_checkpoint is not None:
            metadata["inputs"]["initial_checkpoint"] = {
                "path": str(args.initial_checkpoint),
                "sha256": file_sha256(args.initial_checkpoint),
                "load_report": initializer_report,
            }
        if context.is_main:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            loss_history_path = args.output_dir / "loss_history.jsonl"
            if loss_history_path.exists():
                raise FileExistsError(
                    f"parallel OPQ output is not fresh: {loss_history_path}"
                )
        barrier(context)
        loss_history_path = args.output_dir / "loss_history.jsonl"
        best_path = args.output_dir / "best" / CHECKPOINT_NAME
        best_state: dict[str, Any] = {"loss": None, "step": None, "path": best_path}
        initial_dev_loss = None
        if args.initial_checkpoint is not None:
            initial_dev_loss = distributed_eval_loss(
                model,
                dev_rows,
                tokenizer,
                semantic_ids,
                depth=args.depth,
                max_source_length=args.max_source_length,
                batch_size=args.eval_batch_size,
                context=context,
            )
            if context.is_main:
                best_state["loss"] = initial_dev_loss
                best_state["step"] = 0
                event = {
                    "stage": "query",
                    "event": "eval",
                    "step": 0,
                    "dev_mean_position_ce": initial_dev_loss,
                    "best": True,
                    "initializer": True,
                }
                append_jsonl(loss_history_path, event)
                print(json.dumps(event, sort_keys=True), flush=True)
                atomic_torch_save(
                    checkpoint_payload(
                        model,
                        metadata,
                        checkpoint_role="best",
                        best_dev_loss=initial_dev_loss,
                        best_query_step=0,
                    ),
                    best_path,
                )
            barrier(context)
        catalog_skill_ids = sorted(semantic_ids)
        catalog_skill_index = {
            skill_id: index for index, skill_id in enumerate(catalog_skill_ids)
        }
        catalog_codes = torch.tensor(
            [semantic_ids[skill_id] for skill_id in catalog_skill_ids],
            dtype=torch.long,
            device=context.device,
        )
        catalog_result = train_stage(
            stage_name="catalog",
            rows=catalog_rows,
            epochs=args.catalog_epochs,
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            semantic_ids=semantic_ids,
            depth=args.depth,
            max_source_length=args.max_source_length,
            batch_size=args.batch_size,
            bf16=args.bf16,
            seed=args.seed,
            logging_steps=args.logging_steps,
            context=context,
            loss_history_path=loss_history_path,
            lr_schedule="constant",
            peak_lr=args.lr,
            min_lr=args.lr,
            max_grad_norm=args.max_grad_norm,
        )
        query_result = train_stage(
            stage_name="query",
            rows=query_rows,
            epochs=args.epochs,
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            semantic_ids=semantic_ids,
            depth=args.depth,
            max_source_length=args.max_source_length,
            batch_size=args.batch_size,
            bf16=args.bf16,
            seed=args.seed,
            logging_steps=args.logging_steps,
            context=context,
            loss_history_path=loss_history_path,
            dev_rows=dev_rows,
            eval_batch_size=args.eval_batch_size,
            eval_steps=args.eval_steps,
            checkpoint_metadata=metadata,
            best_state=best_state,
            lr_schedule=args.lr_scheduler,
            peak_lr=args.lr,
            min_lr=args.min_lr,
            max_grad_norm=args.max_grad_norm,
            epoch_offset=args.epoch_offset,
            catalog_codes=catalog_codes,
            catalog_skill_index=catalog_skill_index,
            catalog_loss_weight=args.catalog_loss_weight,
            position_weight_kl_weight=args.position_weight_kl_weight,
        )
        if context.is_main:
            last_path = args.output_dir / "last" / CHECKPOINT_NAME
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    metadata,
                    checkpoint_role="last",
                    last_query_step=query_result["steps"],
                ),
                last_path,
            )
        barrier(context)
        if context.is_main:
            if not best_path.is_file():
                raise RuntimeError(
                    "parallel OPQ training completed without a dev-best checkpoint"
                )
            selected = torch.load(best_path, map_location="cpu", weights_only=False)
            selected["checkpoint_role"] = "final"
            selected["selected_checkpoint"] = "best"
            final_path = args.output_dir / "final" / CHECKPOINT_NAME
            atomic_torch_save(selected, final_path)
            train_manifest = {
                "status": "complete",
                "architecture": architecture,
                "training_objective": training_objective,
                "catalog_scoring": catalog_scoring,
                "semantic_id_audit": sid_audit,
                "source_train_rows": len(source_rows),
                "expanded_query_train_rows": len(query_rows),
                "source_dev_rows": len(dev_source_rows),
                "expanded_query_dev_rows": len(dev_rows),
                "catalog_rows": len(catalog_rows),
                "initial_dev_loss": initial_dev_loss,
                "catalog_result": catalog_result,
                "query_result": query_result,
                "best_dev_loss": best_state["loss"],
                "best_query_step": best_state["step"],
                "best_checkpoint": str(best_path),
                "last_checkpoint": str(last_path),
                "final_checkpoint": str(final_path),
                "length_audits": length_audits,
                "protocol": metadata,
            }
            pipeline.write_json(
                args.output_dir / "parallel_decoder_train_manifest.json", train_manifest
            )
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "final_checkpoint": str(final_path),
                        "best_dev_loss": best_state["loss"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        barrier(context)
    finally:
        destroy_distributed(context)


def command_eval(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    context = initialize_distributed(cpu=args.cpu)
    try:
        semantic_ids, semantic_id_mode = pipeline.load_semantic_ids_with_mode(
            args.semantic_ids
        )
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        depth = int(checkpoint["depth"])
        codebook_size = int(checkpoint["codebook_size"])
        sid_audit = validate_parallel_semantic_ids(
            semantic_ids,
            semantic_id_mode,
            depth=depth,
            codebook_size=codebook_size,
            expected_catalog_size=args.expected_catalog_size,
        )
        head_type = str(checkpoint.get("head_type", LINEAR_HEAD))
        expected_architecture = architecture_for_head(head_type)
        expected_catalog_scoring = catalog_scoring_for_head(head_type)
        if checkpoint.get("architecture") != expected_architecture:
            raise ValueError(
                f"not a parallel OPQ checkpoint: {checkpoint.get('architecture')}"
            )
        if checkpoint.get("catalog_scoring") != expected_catalog_scoring:
            raise ValueError(
                "parallel OPQ checkpoint catalog-scoring contract mismatch"
            )
        if checkpoint.get("semantic_ids_sha256") != sid_audit["semantic_ids_sha256"]:
            raise ValueError("parallel OPQ checkpoint semantic-ID hash mismatch")
        base_model = str(checkpoint["base_model"])
        temperature = float(checkpoint.get("temperature", 0.03))
        model = make_parallel_t5_model(
            base_model,
            depth,
            codebook_size,
            head_type=head_type,
            temperature=temperature,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(context.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        rows = pipeline.load_jsonl(args.rows)
        if not rows:
            raise ValueError(f"{args.split_name} evaluation rows are empty")
        k_values = pipeline.parse_k_values(args.k_values)
        top_k = max(k_values)
        skill_ids = sorted(semantic_ids)
        catalog_codes = torch.tensor(
            [[int(value) for value in semantic_ids[sid]] for sid in skill_ids],
            dtype=torch.long,
            device=context.device,
        )
        local_indexed_rows = shard_with_indices(rows, context)
        local_predictions: list[dict[str, Any]] = []
        with torch.no_grad():
            for start in range(0, len(local_indexed_rows), args.batch_size):
                indexed_batch = local_indexed_rows[start : start + args.batch_size]
                batch_rows_only = [row for _, row in indexed_batch]
                encoded = tokenizer(
                    [pipeline.decoder_source_text(row) for row in batch_rows_only],
                    padding=True,
                    truncation=True,
                    max_length=int(checkpoint["max_source_length"]),
                    return_tensors="pt",
                )
                logits, position_weights = model(
                    encoded["input_ids"].to(context.device),
                    encoded["attention_mask"].to(context.device),
                    return_position_weights=True,
                )
                score_matrix = exact_catalog_log_scores(
                    logits, catalog_codes, position_weights=position_weights
                ).cpu()
                for batch_index, (row_index, row) in enumerate(indexed_batch):
                    ranked_ids, ranked_scores = stable_topk_catalog(
                        score_matrix[batch_index].tolist(), skill_ids, top_k
                    )
                    local_predictions.append(
                        {
                            "_eval_index": row_index,
                            "index": row.get("index", row_index),
                            "query": pipeline.row_query_text(row),
                            "gold_skill_ids": pipeline.prepared_selected_skill_ids(row),
                            "predicted_skill_ids": ranked_ids[:1],
                            "ranked_skill_ids": ranked_ids,
                            "ranked_scores": ranked_scores,
                            "invalid": False,
                            "catalog_items_scored": len(skill_ids),
                        }
                    )
        gathered = all_gather_object(local_predictions, context)
        if context.is_main:
            predictions = [item for rank_rows in gathered for item in rank_rows]
            predictions.sort(key=lambda item: int(item["_eval_index"]))
            actual_indices = [int(item.pop("_eval_index")) for item in predictions]
            if actual_indices != list(range(len(rows))):
                raise RuntimeError(
                    "distributed Official evaluation lost or duplicated rows: "
                    f"expected={len(rows)}, indices={actual_indices[:10]}"
                )
            metrics = pipeline.compute_at_k_metrics(rows, predictions, k_values)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            pipeline.write_jsonl(args.output_dir / "predictions.jsonl", predictions)
            pipeline.write_json(args.output_dir / "ranked_metrics.json", metrics)
            pipeline.write_json(
                args.output_dir / "eval_manifest.json",
                {
                    "status": "complete",
                    "split": args.split_name,
                    "rows": len(rows),
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": file_sha256(args.checkpoint),
                    "semantic_ids": str(args.semantic_ids),
                    "semantic_ids_sha256": sid_audit["semantic_ids_sha256"],
                    "architecture": expected_architecture,
                    "head_type": head_type,
                    "temperature": temperature,
                    "position_temperatures": [
                        float(value)
                        for value in model.position_temperatures().detach().cpu()
                    ],
                    "position_score_weights": [
                        float(value)
                        for value in model.position_score_weights().detach().cpu()
                    ],
                    "catalog_scoring": expected_catalog_scoring,
                    "catalog_items_scored_per_query": len(skill_ids),
                    "ranked_items_retained": top_k,
                    "tie_break": "skill_id_ascending",
                    "k_values": k_values,
                    "ddp_world_size": context.world_size,
                    "precision": "fp32",
                },
            )
            print(json.dumps(metrics["overall"], sort_keys=True), flush=True)
        barrier(context)
    finally:
        destroy_distributed(context)


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--expected-catalog-size", type=int, default=14551)
    parser.add_argument("--cpu", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--rows", type=Path, required=True)
    train.add_argument("--dev-rows", type=Path, required=True)
    train.add_argument("--skills-jsonl", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--depth", type=int, choices=[32, 64], required=True)
    train.add_argument("--codebook-size", type=int, default=256)
    train.add_argument("--catalog-input-prefix", default="skill document: ")
    train.add_argument("--catalog-epochs", type=int, default=3)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--expected-global-batch-size", type=int, default=16)
    train.add_argument("--eval-batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--min-lr", type=float, default=0.0)
    train.add_argument(
        "--lr-scheduler", choices=("constant", "cosine"), default="constant"
    )
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--max-grad-norm", type=float, default=0.0)
    train.add_argument("--epoch-offset", type=int, default=0)
    train.add_argument("--head-type", choices=HEAD_TYPES, default=LINEAR_HEAD)
    train.add_argument("--temperature", type=float, default=0.03)
    train.add_argument("--catalog-loss-weight", type=float, default=0.0)
    train.add_argument("--position-weight-kl-weight", type=float, default=0.0)
    train.add_argument("--initial-checkpoint", type=Path)
    train.add_argument("--max-source-length", type=int, default=2048)
    train.add_argument("--logging-steps", type=int, default=50)
    train.add_argument("--eval-steps", type=int, default=1000)
    train.add_argument("--seed", type=int, default=20260626)
    train.add_argument("--bf16", action="store_true")
    add_common_data_args(train)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--rows", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--split-name", default="official_core75")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--k-values", default="1,5,10,50")
    add_common_data_args(evaluate)
    evaluate.set_defaults(func=command_eval)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
