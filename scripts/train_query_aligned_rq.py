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


ARTIFACT_FORMAT = "query_aligned_collaborative_rq_codebook_v1"


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
        semantic_weight: float,
        cf_weight: float,
        prefix_weight: float,
        teacher_temperature: float,
        student_temperature: float,
    ) -> dict[str, torch.Tensor]:
        _, reconstruction = residual_encode(skill_embeddings, self.codebooks)
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


def minimum_cost_unique_last_codes(
    codes: torch.Tensor,
    embeddings: torch.Tensor,
    codebooks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Resolve full-code collisions within each parent by final-level distortion."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    resolved = codes.detach().cpu().long().clone()
    codebooks = codebooks.detach().cpu().float()
    embeddings = embeddings.detach().cpu().float()
    residual = embeddings.clone()
    for level in range(codebooks.shape[0] - 1):
        residual = residual - codebooks[level].index_select(0, resolved[:, level])
    final_costs = squared_euclidean_distance(residual, codebooks[-1]).double().numpy()
    prefix_groups: dict[tuple[int, ...], list[int]] = {}
    for item_index, values in enumerate(resolved[:, :-1].tolist()):
        prefix_groups.setdefault(tuple(int(value) for value in values), []).append(item_index)

    repaired_prefixes = repaired_items = changed_items = 0
    assignment_cost = 0.0
    max_repaired_prefix_size = 0
    for prefix in sorted(prefix_groups):
        item_indices = prefix_groups[prefix]
        existing = Counter(int(resolved[item, -1]) for item in item_indices)
        if max(existing.values()) <= 1:
            continue
        if len(item_indices) > codebooks.shape[1]:
            raise ValueError(
                f"prefix {prefix} contains {len(item_indices)} skills but only {codebooks.shape[1]} final codes"
            )
        repaired_prefixes += 1
        repaired_items += len(item_indices)
        max_repaired_prefix_size = max(max_repaired_prefix_size, len(item_indices))
        local_rows, assigned = linear_sum_assignment(np.asarray(final_costs[item_indices]))
        for local_row, code in zip(local_rows.tolist(), assigned.tolist()):
            item_index = item_indices[local_row]
            changed_items += int(int(resolved[item_index, -1]) != int(code))
            resolved[item_index, -1] = int(code)
            assignment_cost += float(final_costs[item_index, code])

    counts = Counter(tuple(int(value) for value in row) for row in resolved.tolist())
    collision_groups = sum(count > 1 for count in counts.values())
    if collision_groups:
        raise RuntimeError(f"minimum-cost final-code assignment left {collision_groups} collisions")
    return resolved, {
        "method": "minimum_cost_unique_last_code",
        "repaired_prefixes": repaired_prefixes,
        "repaired_items": repaired_items,
        "changed_items": changed_items,
        "max_repaired_prefix_size": max_repaired_prefix_size,
        "assignment_cost": assignment_cost,
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
    snapshot_epoch = -1
    all_codes = None
    legal_children = None
    history_path = args.output_dir / "training_history.jsonl"
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if history_path.exists():
            history_path.unlink()
    barrier(context)
    while step < total_steps:
        epoch = step // steps_per_epoch
        if epoch != snapshot_epoch:
            with torch.no_grad():
                all_codes = unwrap_model(model).encode(relevance_device).detach()
            legal_children = build_legal_children(all_codes)
            snapshot_epoch = epoch
        assert all_codes is not None and legal_children is not None

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
            all_codes.index_select(
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
        trained_codebooks = unwrap_model(model).codebooks.detach().cpu().float()
        raw_codes, _ = residual_encode(relevance, trained_codebooks)
        resolved_codes, collision_audit = minimum_cost_unique_last_codes(
            raw_codes,
            relevance,
            trained_codebooks,
        )
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
            "objective": "M0 reconstruction + Qwen/LightGCN affinity + query-prefix distribution",
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
        raw_counter = Counter(tuple(int(value) for value in row) for row in raw_codes.tolist())
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
                "pre_resolution_collision_groups": sum(count > 1 for count in raw_counter.values()),
                "collision_resolution": collision_audit,
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
