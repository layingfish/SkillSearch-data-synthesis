#!/usr/bin/env python3
"""Train a query-aligned semantic/CF residual codebook in T5 M0 space."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from distributed_utils import (  # noqa: E402
    barrier,
    destroy_distributed,
    initialize_distributed,
    unwrap_model,
    wrap_ddp,
)
from skill_gr_pipeline import (  # noqa: E402
    SEMANTIC_ID_MODE_LEARNED,
    file_sha256,
    load_jsonl,
    load_ranked_skill_candidates_with_scores,
    ranked_candidates_for_row,
    retrieval_row_keys,
    semantic_id_tokens,
    validate_semantic_ids,
    write_json,
    write_jsonl,
)


ARTIFACT_FORMAT = "query_aligned_collaborative_rq_codebook_v2"


def squared_euclidean_distance(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (
        x.float().square().sum(dim=-1, keepdim=True)
        + centers.float().square().sum(dim=-1).unsqueeze(0)
        - 2.0 * x.float() @ centers.float().t()
    )


def residual_encode(
    x: torch.Tensor,
    codebooks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hard residual codes and their additive reconstruction."""
    residual = x.float()
    selected: list[torch.Tensor] = []
    codes: list[torch.Tensor] = []
    for level in range(codebooks.shape[0]):
        distances = squared_euclidean_distance(residual, codebooks[level])
        level_codes = distances.argmin(dim=-1)
        level_centers = codebooks[level].index_select(0, level_codes)
        codes.append(level_codes)
        selected.append(level_centers)
        residual = residual - level_centers
    return torch.stack(codes, dim=-1), torch.stack(selected, dim=0).sum(dim=0)


def reconstruction_from_codes(codes: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Reconstruct vectors from an explicit, fixed residual-code assignment."""
    if codes.ndim != 2 or codes.shape[1] != codebooks.shape[0]:
        raise ValueError(
            "codes must have shape [items, depth] matching the codebooks, got "
            f"codes={tuple(codes.shape)}, codebooks={tuple(codebooks.shape)}"
        )
    selected = [
        codebooks[level].index_select(0, codes[:, level].long())
        for level in range(codebooks.shape[0])
    ]
    return torch.stack(selected, dim=0).sum(dim=0)


def pairwise_affinity_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match off-diagonal cosine affinities within one sampled skill batch."""
    if reconstruction.shape[0] <= 1:
        return reconstruction.sum() * 0.0
    reconstruction = F.normalize(reconstruction.float(), p=2, dim=-1)
    target = F.normalize(target.float(), p=2, dim=-1)
    predicted = reconstruction @ reconstruction.t()
    expected = target @ target.t()
    mask = ~torch.eye(predicted.shape[0], dtype=torch.bool, device=predicted.device)
    return F.mse_loss(predicted[mask], expected[mask])


def build_legal_children(codes: torch.Tensor) -> list[dict[tuple[int, ...], tuple[int, ...]]]:
    """Build the legal next-code domain for every observed RQ parent prefix."""
    if codes.ndim != 2 or codes.shape[1] <= 0:
        raise ValueError(f"codes must have shape [items, depth], got {tuple(codes.shape)}")
    result: list[dict[tuple[int, ...], tuple[int, ...]]] = []
    codes_cpu = codes.detach().cpu().long().tolist()
    for level in range(codes.shape[1]):
        values: dict[tuple[int, ...], set[int]] = {}
        for code in codes_cpu:
            parent = tuple(int(value) for value in code[:level])
            values.setdefault(parent, set()).add(int(code[level]))
        result.append(
            {parent: tuple(sorted(children)) for parent, children in values.items()}
        )
    return result


def prefix_teacher_distribution_loss(
    query_vector: torch.Tensor,
    candidate_codes: torch.Tensor,
    teacher_scores: torch.Tensor,
    codebooks: torch.Tensor,
    legal_children: list[dict[tuple[int, ...], tuple[int, ...]]],
    *,
    teacher_temperature: float,
    student_temperature: float,
) -> torch.Tensor:
    """Distill conditional teacher mass into residual-code geometry at every level."""
    if teacher_temperature <= 0.0 or student_temperature <= 0.0:
        raise ValueError("teacher and student temperatures must be positive")
    if candidate_codes.ndim != 2 or candidate_codes.shape[0] != teacher_scores.shape[0]:
        raise ValueError("candidate codes and teacher scores must share the candidate axis")
    if candidate_codes.shape[1] != codebooks.shape[0]:
        raise ValueError("candidate code depth does not match the residual codebook")
    if len(legal_children) != codebooks.shape[0]:
        raise ValueError("legal-child depth does not match the residual codebook")
    if candidate_codes.shape[0] < 2:
        return codebooks.sum() * 0.0

    teacher_probabilities = F.softmax(teacher_scores.float() / teacher_temperature, dim=0)
    total = codebooks.new_tensor(0.0, dtype=torch.float32)
    query_vector = query_vector.float()
    for level in range(codebooks.shape[0]):
        parent_groups: dict[tuple[int, ...], list[int]] = {}
        for candidate_index, values in enumerate(candidate_codes[:, :level].detach().cpu().tolist()):
            parent_groups.setdefault(tuple(int(value) for value in values), []).append(candidate_index)
        for parent, member_indices in parent_groups.items():
            members = torch.tensor(member_indices, dtype=torch.long, device=candidate_codes.device)
            parent_mass = teacher_probabilities.index_select(0, members).sum()
            if not bool(parent_mass > 0):
                continue
            children = legal_children[level].get(parent)
            if not children:
                raise ValueError(f"missing legal children for level={level}, parent={parent}")
            child_tensor = torch.tensor(children, dtype=torch.long, device=candidate_codes.device)
            target_mass = torch.zeros(len(children), dtype=torch.float32, device=candidate_codes.device)
            child_to_offset = {value: offset for offset, value in enumerate(children)}
            member_codes = candidate_codes.index_select(0, members)[:, level].detach().cpu().tolist()
            member_mass = teacher_probabilities.index_select(0, members)
            for local_index, child in enumerate(member_codes):
                target_mass[child_to_offset[int(child)]] += member_mass[local_index]
            target_distribution = target_mass / parent_mass.clamp_min(torch.finfo(torch.float32).tiny)

            residual = query_vector
            if parent:
                residual = residual - torch.stack(
                    [codebooks[parent_level, parent_code] for parent_level, parent_code in enumerate(parent)],
                    dim=0,
                ).sum(dim=0)
            child_centers = codebooks[level].index_select(0, child_tensor).float()
            geometry = 2.0 * (child_centers @ residual) - child_centers.square().sum(dim=-1)
            log_probabilities = F.log_softmax(geometry / student_temperature, dim=0)
            total = total - parent_mass * (target_distribution * log_probabilities).sum()
    return total / float(codebooks.shape[0])


def initialize_residual_kmeans(
    embeddings: torch.Tensor,
    *,
    depth: int,
    codebook_size: int,
    iterations: int,
    seed: int,
) -> torch.Tensor:
    """Run deterministic sequential k-means for a residual codebook."""
    if embeddings.shape[0] < codebook_size:
        raise ValueError("k-means requires at least codebook_size skill embeddings")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    residual = embeddings.float().clone()
    codebooks: list[torch.Tensor] = []
    for level in range(depth):
        initial_indices = torch.randperm(embeddings.shape[0], generator=generator)[:codebook_size]
        centers = residual.index_select(0, initial_indices.to(residual.device)).clone()
        for _ in range(iterations):
            assignments = squared_euclidean_distance(residual, centers).argmin(dim=-1)
            sums = torch.zeros_like(centers)
            counts = torch.zeros(codebook_size, dtype=torch.float32, device=residual.device)
            sums.index_add_(0, assignments, residual)
            counts.index_add_(0, assignments, torch.ones_like(assignments, dtype=torch.float32))
            nonempty = counts > 0
            centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        assignments = squared_euclidean_distance(residual, centers).argmin(dim=-1)
        residual = residual - centers.index_select(0, assignments)
        codebooks.append(centers)
    return torch.stack(codebooks, dim=0)


class QueryAlignedResidualCodebook(nn.Module):
    def __init__(self, initial_codebooks: torch.Tensor) -> None:
        super().__init__()
        if initial_codebooks.ndim != 3:
            raise ValueError("initial codebooks must have shape [depth, size, dim]")
        self.codebooks = nn.Parameter(initial_codebooks.float())

    @torch.no_grad()
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        codes, _ = residual_encode(embeddings, self.codebooks)
        return codes

    def forward(
        self,
        skill_embeddings: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        cf_embeddings: torch.Tensor,
        query_vectors: list[torch.Tensor],
        query_candidate_codes: list[torch.Tensor],
        query_teacher_scores: list[torch.Tensor],
        legal_children: list[dict[tuple[int, ...], tuple[int, ...]]],
        *,
        skill_codes: torch.Tensor | None = None,
        semantic_weight: float,
        cf_weight: float,
        prefix_weight: float,
        teacher_temperature: float,
        student_temperature: float,
    ) -> dict[str, torch.Tensor]:
        if skill_codes is None:
            _, reconstruction = residual_encode(skill_embeddings, self.codebooks)
        else:
            reconstruction = reconstruction_from_codes(skill_codes, self.codebooks)
        reconstruction_loss = F.mse_loss(reconstruction, skill_embeddings.float())
        semantic_loss = pairwise_affinity_loss(reconstruction, semantic_embeddings)
        cf_loss = pairwise_affinity_loss(reconstruction, cf_embeddings)
        prefix_losses = [
            prefix_teacher_distribution_loss(
                query_vector,
                candidate_codes,
                teacher_scores,
                self.codebooks,
                legal_children,
                teacher_temperature=teacher_temperature,
                student_temperature=student_temperature,
            )
            for query_vector, candidate_codes, teacher_scores in zip(
                query_vectors, query_candidate_codes, query_teacher_scores
            )
        ]
        prefix_loss = (
            torch.stack(prefix_losses).mean()
            if prefix_losses
            else self.codebooks.sum() * 0.0
        )
        loss = (
            reconstruction_loss
            + float(semantic_weight) * semantic_loss
            + float(cf_weight) * cf_loss
            + float(prefix_weight) * prefix_loss
        )
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "semantic_loss": semantic_loss,
            "cf_loss": cf_loss,
            "prefix_loss": prefix_loss,
        }


def prefix_capacity_audit(codes: torch.Tensor, codebook_size: int) -> list[dict[str, int]]:
    """Report whether every prefix fits in the code space below it."""
    codes_cpu = codes.detach().cpu().long()
    if codes_cpu.ndim != 2 or codes_cpu.shape[1] <= 0:
        raise ValueError(f"codes must have shape [items, depth], got {tuple(codes_cpu.shape)}")
    depth = int(codes_cpu.shape[1])
    audit: list[dict[str, int]] = []
    for prefix_length in range(1, depth + 1):
        counts = Counter(
            tuple(int(value) for value in row[:prefix_length])
            for row in codes_cpu.tolist()
        )
        suffix_capacity = int(codebook_size ** (depth - prefix_length))
        audit.append(
            {
                "prefix_length": prefix_length,
                "suffix_capacity": suffix_capacity,
                "prefix_groups": len(counts),
                "max_prefix_load": max(counts.values(), default=0),
                "overloaded_prefix_groups": sum(
                    count > suffix_capacity for count in counts.values()
                ),
            }
        )
    return audit


def _confidence_ordered_capacity_assignment(
    costs: Any,
    *,
    child_capacity: int,
) -> tuple[Any, int]:
    """Assign rows to children under a hard capacity using deterministic confidence order."""
    import numpy as np

    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] <= 0:
        raise ValueError(f"cost matrix must be two-dimensional, got {values.shape}")
    if child_capacity <= 0 or values.shape[0] > values.shape[1] * child_capacity:
        raise ValueError(
            "child capacity cannot hold all rows: "
            f"rows={values.shape[0]}, children={values.shape[1]}, capacity={child_capacity}"
        )
    preferences = np.argsort(values, axis=1, kind="stable")
    unconstrained = preferences[:, 0]
    counts = np.bincount(unconstrained, minlength=values.shape[1])
    if int(counts.max(initial=0)) <= child_capacity:
        return unconstrained.astype(np.int64, copy=False), 0

    rows = np.arange(values.shape[0], dtype=np.int64)
    if values.shape[1] == 1:
        margins = np.full(values.shape[0], np.inf, dtype=np.float64)
    else:
        margins = values[rows, preferences[:, 1]] - values[rows, preferences[:, 0]]
    order = np.lexsort((rows, -margins))
    remaining = np.full(values.shape[1], child_capacity, dtype=np.int64)
    assigned = np.full(values.shape[0], -1, dtype=np.int64)
    for row in order.tolist():
        for child in preferences[row].tolist():
            if remaining[child] > 0:
                assigned[row] = child
                remaining[child] -= 1
                break
    if bool((assigned < 0).any()):
        raise RuntimeError("capacity assignment left at least one row unassigned")
    return assigned, int(np.count_nonzero(assigned != unconstrained))


def hierarchical_capacity_unique_codes(
    codes: torch.Tensor,
    embeddings: torch.Tensor,
    codebooks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Resolve collisions with a suffix-capacity constraint at every prefix level."""
    import numpy as np

    if codes.ndim != 2 or codebooks.ndim != 3:
        raise ValueError("codes and codebooks must have shapes [items, depth] and [depth, size, dim]")
    if codes.shape[1] != codebooks.shape[0] or embeddings.shape[0] != codes.shape[0]:
        raise ValueError("codes, embeddings, and codebooks do not share item/depth dimensions")
    depth = int(codebooks.shape[0])
    codebook_size = int(codebooks.shape[1])
    if codes.shape[0] > codebook_size**depth:
        raise ValueError(
            f"{codes.shape[0]} items exceed the {codebook_size**depth} available full codes"
        )

    device = embeddings.device
    resolved = torch.zeros_like(codes, dtype=torch.long, device=device)
    residual = embeddings.detach().to(device=device, dtype=torch.float32).clone()
    codebooks_device = codebooks.detach().to(device=device, dtype=torch.float32)
    parent_groups: dict[tuple[int, ...], list[int]] = {(): list(range(codes.shape[0]))}
    level_audit: list[dict[str, Any]] = []
    total_assignment_cost = 0.0
    for level in range(depth):
        all_costs = (
            squared_euclidean_distance(residual, codebooks_device[level])
            .detach()
            .cpu()
            .numpy()
        )
        child_capacity = int(codebook_size ** (depth - level - 1))
        repaired_parent_groups = repaired_items = capacity_reassignments = 0
        next_parent_groups: dict[tuple[int, ...], list[int]] = {}
        for parent in sorted(parent_groups):
            item_indices = parent_groups[parent]
            local_costs = np.asarray(all_costs[item_indices], dtype=np.float64)
            unconstrained = local_costs.argmin(axis=1)
            unconstrained_counts = np.bincount(unconstrained, minlength=codebook_size)
            assigned, changed = _confidence_ordered_capacity_assignment(
                local_costs,
                child_capacity=child_capacity,
            )
            if int(unconstrained_counts.max(initial=0)) > child_capacity:
                repaired_parent_groups += 1
                repaired_items += len(item_indices)
            capacity_reassignments += changed
            total_assignment_cost += float(local_costs[np.arange(len(item_indices)), assigned].sum())
            item_tensor = torch.tensor(item_indices, dtype=torch.long, device=device)
            assigned_tensor = torch.from_numpy(assigned).to(device=device, dtype=torch.long)
            resolved[item_tensor, level] = assigned_tensor
            for item_index, child in zip(item_indices, assigned.tolist()):
                next_parent_groups.setdefault((*parent, int(child)), []).append(item_index)

        selected = codebooks_device[level].index_select(0, resolved[:, level])
        residual = residual - selected
        max_prefix_load = max((len(items) for items in next_parent_groups.values()), default=0)
        if max_prefix_load > child_capacity:
            raise RuntimeError(
                f"level {level} capacity invariant failed: load={max_prefix_load}, capacity={child_capacity}"
            )
        level_audit.append(
            {
                "level": level,
                "child_capacity": child_capacity,
                "parent_groups": len(parent_groups),
                "repaired_parent_groups": repaired_parent_groups,
                "repaired_items": repaired_items,
                "capacity_reassignments": capacity_reassignments,
                "output_prefix_groups": len(next_parent_groups),
                "max_output_prefix_load": max_prefix_load,
            }
        )
        parent_groups = next_parent_groups

    resolved_cpu = resolved.detach().cpu().long()
    counts = Counter(tuple(int(value) for value in row) for row in resolved_cpu.tolist())
    collision_groups = sum(count > 1 for count in counts.values())
    if collision_groups:
        raise RuntimeError(f"hierarchical capacity assignment left {collision_groups} collisions")
    return resolved_cpu, {
        "method": "hierarchical_confidence_ordered_capacity_assignment",
        "assignment_cost": total_assignment_cost,
        "level_audit": level_audit,
        "pre_capacity_audit": prefix_capacity_audit(codes, codebook_size),
        "post_capacity_audit": prefix_capacity_audit(resolved_cpu, codebook_size),
        "post_collision_groups": collision_groups,
    }


def load_cf_skill_ids(path: Path) -> list[str]:
    rows = load_jsonl(path)
    ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
    return [str(row["skill_id"]) for row in ordered]


def build_query_examples(
    rows: list[dict[str, Any]],
    skill_to_index: dict[str, int],
    candidate_map: dict[str, list[str]],
    teacher_score_map: dict[str, dict[str, float]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        scores: dict[str, float] = {}
        for key in retrieval_row_keys(row):
            scores.update(teacher_score_map.get(key, {}))
        positives = [
            str(sid)
            for sid in row.get("selected_skill_ids", []) or []
            if str(sid) in skill_to_index and str(sid) in scores
        ]
        negatives = [
            sid
            for sid in ranked_candidates_for_row(row, candidate_map)
            if sid in skill_to_index and sid in scores and sid not in positives
        ]
        candidate_ids = list(dict.fromkeys([*positives, *negatives]))[:max_candidates]
        if len(candidate_ids) < 2:
            continue
        examples.append(
            {
                "row_index": row_index,
                "candidate_indices": [skill_to_index[sid] for sid in candidate_ids],
                "teacher_scores": [float(scores[sid]) for sid in candidate_ids],
            }
        )
    if not examples:
        raise ValueError("no query has at least two teacher-scored skill candidates")
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relevance-embeddings", type=Path, required=True)
    parser.add_argument("--semantic-embeddings", type=Path, required=True)
    parser.add_argument("--cf-embeddings", type=Path, required=True)
    parser.add_argument("--cf-skill-ids", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--semantic-weight", type=float, default=0.01)
    parser.add_argument("--cf-weight", type=float, default=0.01)
    parser.add_argument("--prefix-weight", type=float, default=0.1)
    parser.add_argument("--teacher-temperature", type=float, default=0.05)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.depth <= 0 or args.codebook_size <= 1:
        raise ValueError("depth must be positive and codebook_size must exceed one")
    for name in ("semantic_weight", "cf_weight", "prefix_weight"):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if args.teacher_temperature <= 0.0 or args.student_temperature <= 0.0:
        raise ValueError("temperatures must be positive")

    context = initialize_distributed(cpu=args.cpu)
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + context.rank)
    device = context.device

    relevance_payload = torch.load(args.relevance_embeddings, map_location="cpu", weights_only=False)
    semantic_payload = torch.load(args.semantic_embeddings, map_location="cpu", weights_only=False)
    query_payload = torch.load(args.query_embeddings, map_location="cpu", weights_only=False)
    relevance_ids = [str(value) for value in relevance_payload["skill_ids"]]
    semantic_ids = [str(value) for value in semantic_payload["skill_ids"]]
    cf_ids = load_cf_skill_ids(args.cf_skill_ids)
    if relevance_ids != semantic_ids or relevance_ids != cf_ids:
        raise ValueError("M0, Qwen, and LightGCN artifacts do not share the exact canonical skill order")
    if relevance_payload.get("artifact_format") != "ripor_m0_skill_embeddings_v1":
        raise ValueError("relevance embeddings must come from train-ripor-relevance-encoder")
    if query_payload.get("artifact_format") != "ripor_m0_query_embeddings_v1":
        raise ValueError("query embeddings must come from train-ripor-relevance-encoder")
    if query_payload.get("source_rows_sha256") != file_sha256(args.rows):
        raise ValueError("query embeddings were produced from a different training-row file")
    source_checkpoint_hash = str(relevance_payload["source_relevance_checkpoint_sha256"])
    if str(query_payload.get("source_relevance_checkpoint_sha256")) != source_checkpoint_hash:
        raise ValueError("query and skill embeddings were produced by different M0 checkpoints")

    relevance = torch.as_tensor(relevance_payload["embeddings"]).float()
    semantic = torch.as_tensor(semantic_payload["embeddings"]).float()
    loaded_cf = torch.load(args.cf_embeddings, map_location="cpu", weights_only=False)
    if isinstance(loaded_cf, dict):
        loaded_cf = loaded_cf.get("embeddings", loaded_cf.get("cf_embeddings", loaded_cf))
    cf = torch.as_tensor(loaded_cf).float().squeeze()
    query_embeddings = torch.as_tensor(query_payload["embeddings"]).float()
    rows = load_jsonl(args.rows)
    if relevance.ndim != 2 or semantic.ndim != 2 or cf.ndim != 2:
        raise ValueError("skill embedding artifacts must be two-dimensional")
    if relevance.shape[0] != len(relevance_ids) or semantic.shape[0] != len(relevance_ids) or cf.shape[0] != len(relevance_ids):
        raise ValueError("skill embedding row count does not match the canonical skill list")
    if query_embeddings.shape != (len(rows), relevance.shape[1]):
        raise ValueError(
            f"query embeddings must have shape {(len(rows), relevance.shape[1])}, got {tuple(query_embeddings.shape)}"
        )

    candidate_map, teacher_score_map = load_ranked_skill_candidates_with_scores(args.hard_negatives)
    skill_to_index = {sid: index for index, sid in enumerate(relevance_ids)}
    query_examples = build_query_examples(
        rows,
        skill_to_index,
        candidate_map,
        teacher_score_map,
        max_candidates=args.max_candidates,
    )

    relevance_device = relevance.to(device)
    semantic_device = semantic.to(device)
    cf_device = cf.to(device)
    query_device = query_embeddings.to(device)
    if context.is_main:
        initial = initialize_residual_kmeans(
            relevance_device,
            depth=args.depth,
            codebook_size=args.codebook_size,
            iterations=args.kmeans_iters,
            seed=args.seed,
        )
    else:
        initial = torch.zeros(
            args.depth,
            args.codebook_size,
            relevance.shape[1],
            dtype=torch.float32,
            device=device,
        )
    if context.enabled:
        import torch.distributed as dist

        dist.broadcast(initial, src=0)

    # The prefix objective must not be allowed to shrink its own label domain by
    # moving every skill into the same branch.  Derive one capacity-safe RQ
    # assignment from the M0 KMeans initialization and keep those discrete
    # labels fixed while optimizing the continuous codebook geometry.
    with torch.no_grad():
        initial_nearest_codes, _ = residual_encode(relevance_device, initial)
    if context.is_main:
        anchored_codes_cpu, collision_audit = hierarchical_capacity_unique_codes(
            initial_nearest_codes,
            relevance_device,
            initial,
        )
        anchored_codes = anchored_codes_cpu.to(device)
    else:
        anchored_codes = torch.zeros(
            relevance.shape[0],
            args.depth,
            dtype=torch.long,
            device=device,
        )
        collision_audit = None
    if context.enabled:
        dist.broadcast(anchored_codes, src=0)
    legal_children = build_legal_children(anchored_codes)

    core_model = QueryAlignedResidualCodebook(initial).to(device)
    model = wrap_ddp(core_model, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    global_skill_batch = args.batch_size * context.world_size
    global_query_batch = args.query_batch_size * context.world_size
    steps_per_epoch = max(
        1,
        math.ceil(len(relevance_ids) / global_skill_batch),
        math.ceil(len(query_examples) / global_query_batch),
    )
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    if total_steps <= 0:
        raise ValueError("query-aligned RQ requires at least one optimizer step")

    skill_generator = torch.Generator(device="cpu").manual_seed(args.seed + 10_000 + context.rank)
    query_generator = torch.Generator(device="cpu").manual_seed(args.seed + 20_000 + context.rank)
    step = 0
    history_path = args.output_dir / "training_history.jsonl"
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if history_path.exists():
            history_path.unlink()
    barrier(context)
    while step < total_steps:
        skill_indices = torch.randint(
            0,
            relevance.shape[0],
            (args.batch_size,),
            generator=skill_generator,
        ).to(device)
        query_offsets = torch.randint(
            0,
            len(query_examples),
            (args.query_batch_size,),
            generator=query_generator,
        ).tolist()
        sampled_queries = [query_examples[int(offset)] for offset in query_offsets]
        query_vectors = [query_device[item["row_index"]] for item in sampled_queries]
        query_candidate_codes = [
            anchored_codes.index_select(
                0,
                torch.tensor(item["candidate_indices"], dtype=torch.long, device=device),
            )
            for item in sampled_queries
        ]
        query_teacher_scores = [
            torch.tensor(item["teacher_scores"], dtype=torch.float32, device=device)
            for item in sampled_queries
        ]

        components = model(
            relevance_device.index_select(0, skill_indices),
            semantic_device.index_select(0, skill_indices),
            cf_device.index_select(0, skill_indices),
            query_vectors,
            query_candidate_codes,
            query_teacher_scores,
            legal_children,
            skill_codes=anchored_codes.index_select(0, skill_indices),
            semantic_weight=args.semantic_weight,
            cf_weight=args.cf_weight,
            prefix_weight=args.prefix_weight,
            teacher_temperature=args.teacher_temperature,
            student_temperature=args.student_temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        components["loss"].backward()
        optimizer.step()
        step += 1
        if step % args.logging_steps == 0 or step == total_steps:
            values = torch.stack(
                [
                    components["loss"].detach(),
                    components["reconstruction_loss"].detach(),
                    components["semantic_loss"].detach(),
                    components["cf_loss"].detach(),
                    components["prefix_loss"].detach(),
                ]
            ).double()
            if context.enabled:
                import torch.distributed as dist

                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values /= context.world_size
            if context.is_main:
                event = {
                    "step": step,
                    "epoch": step / steps_per_epoch,
                    "loss": float(values[0].cpu()),
                    "reconstruction_loss": float(values[1].cpu()),
                    "semantic_loss": float(values[2].cpu()),
                    "cf_loss": float(values[3].cpu()),
                    "prefix_loss": float(values[4].cpu()),
                }
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event, sort_keys=True), flush=True)

    barrier(context)
    if context.is_main:
        trained_codebooks_device = unwrap_model(model).codebooks.detach()
        trained_codebooks = trained_codebooks_device.cpu().float()
        final_nearest_codes, _ = residual_encode(relevance_device, trained_codebooks_device)
        resolved_codes = anchored_codes.detach().cpu().long()
        if collision_audit is None:
            raise RuntimeError("rank 0 is missing the anchored collision audit")
        semantic_id_map = {
            sid: [int(value) for value in resolved_codes[index].tolist()]
            for index, sid in enumerate(relevance_ids)
        }
        validate_semantic_ids(semantic_id_map, SEMANTIC_ID_MODE_LEARNED)
        code_width = max(3, len(str(args.codebook_size - 1)))
        output_rows = [
            {
                "skill_id": sid,
                "rq_codes": semantic_id_map[sid],
                "semantic_id": semantic_id_map[sid],
                "semantic_id_mode": SEMANTIC_ID_MODE_LEARNED,
                "tokens": semantic_id_tokens(
                    semantic_id_map[sid], code_width, SEMANTIC_ID_MODE_LEARNED
                ),
            }
            for sid in sorted(semantic_id_map)
        ]
        m0_provenance = {
            key: relevance_payload.get(key)
            for key in (
                "source_relevance_checkpoint",
                "source_relevance_checkpoint_sha256",
                "source_relevance_model",
                "source_relevance_model_config_sha256",
                "source_relevance_decoder_length",
                "source_relevance_state_keys",
            )
        }
        artifact = {
            "artifact_format": ARTIFACT_FORMAT,
            "codebook_embeddings": trained_codebooks,
            "codebook_size": int(args.codebook_size),
            "depth": int(args.depth),
            "embedding_dim": int(trained_codebooks.shape[-1]),
            "identity_latent": True,
            "anchored_assignments": True,
            "assignment_policy": "capacity-safe initial M0 residual KMeans assignments fixed during geometry optimization",
            "objective": "anchored M0 reconstruction + Qwen/LightGCN affinity + query-prefix distribution",
            "source_embeddings": str(args.relevance_embeddings),
            "semantic_embeddings": str(args.semantic_embeddings),
            "cf_embeddings": str(args.cf_embeddings),
            "query_embeddings": str(args.query_embeddings),
            "hard_negatives": str(args.hard_negatives),
            "source_rows": str(args.rows),
            "source_rows_sha256": file_sha256(args.rows),
            "semantic_embeddings_sha256": file_sha256(args.semantic_embeddings),
            "cf_embeddings_sha256": file_sha256(args.cf_embeddings),
            "query_embeddings_sha256": file_sha256(args.query_embeddings),
            "hard_negatives_sha256": file_sha256(args.hard_negatives),
            **m0_provenance,
        }
        torch.save(artifact, args.output_dir / "codebook_embeddings.pt")
        torch.save(
            {
                "artifact_format": ARTIFACT_FORMAT,
                "state_dict": unwrap_model(model).state_dict(),
                "config": vars(args),
                **m0_provenance,
            },
            args.output_dir / "rqvae.pt",
        )
        write_jsonl(args.output_dir / "semantic_ids.jsonl", output_rows)
        initial_raw_counter = Counter(
            tuple(int(value) for value in row)
            for row in initial_nearest_codes.detach().cpu().tolist()
        )
        final_nearest_counter = Counter(
            tuple(int(value) for value in row)
            for row in final_nearest_codes.detach().cpu().tolist()
        )
        level_usage = [
            len(set(int(value) for value in resolved_codes[:, level].tolist()))
            for level in range(args.depth)
        ]
        write_json(
            args.output_dir / "rqvae_manifest.json",
            {
                "method": "query_aligned_collaborative_residual_quantizer",
                "artifact_format": ARTIFACT_FORMAT,
                "semantic_id_mode": SEMANTIC_ID_MODE_LEARNED,
                "skills": len(relevance_ids),
                "queries": len(rows),
                "teacher_queries": len(query_examples),
                "depth": args.depth,
                "codebook_size": args.codebook_size,
                "embedding_dim": relevance.shape[1],
                "semantic_dim": semantic.shape[1],
                "cf_dim": cf.shape[1],
                "semantic_weight": args.semantic_weight,
                "cf_weight": args.cf_weight,
                "prefix_weight": args.prefix_weight,
                "teacher_temperature": args.teacher_temperature,
                "student_temperature": args.student_temperature,
                "max_candidates": args.max_candidates,
                "batch_size": args.batch_size,
                "query_batch_size": args.query_batch_size,
                "world_size": context.world_size,
                "global_skill_batch_size": args.batch_size * context.world_size,
                "epochs": args.epochs,
                "max_steps": total_steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "kmeans_iters": args.kmeans_iters,
                "seed": args.seed,
                "anchored_assignments": True,
                "assignment_policy": "capacity-safe initial M0 residual KMeans assignments fixed during geometry optimization",
                "pre_resolution_collision_groups": sum(
                    count > 1 for count in initial_raw_counter.values()
                ),
                "collision_resolution": collision_audit,
                "final_nearest_assignment_audit": {
                    "collision_groups": sum(
                        count > 1 for count in final_nearest_counter.values()
                    ),
                    "capacity_audit": prefix_capacity_audit(
                        final_nearest_codes,
                        args.codebook_size,
                    ),
                },
                "level_usage": level_usage,
                "codebook_embeddings": str(args.output_dir / "codebook_embeddings.pt"),
                "semantic_ids": str(args.output_dir / "semantic_ids.jsonl"),
                "training_history": str(history_path),
                **m0_provenance,
            },
        )
        print(f"saved query-aligned residual codebook to {args.output_dir}", flush=True)
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
