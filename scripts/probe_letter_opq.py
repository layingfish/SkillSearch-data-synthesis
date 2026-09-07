#!/usr/bin/env python3
"""Compare fixed-budget PQ/OPQ/RQ identifiers for the LETTER branch.

This is an offline identifier probe, not an end-to-end decoder experiment.  It
keeps the four-token, 256-way (32-bit) LETTER address budget fixed and reports
quantization, collision, neighborhood, and autoregressive-prefix diagnostics.

Standard OPQ is a learned rotation followed by *parallel* product quantizers.
Its code positions are therefore permutation-equivalent for reconstruction but
not for autoregressive decoding.  For D4, this script audits all 24 position
orders and selects one using a catalog-only, pre-registered semantic-neighbor
rule.  Residual RQ and existing LETTER IDs retain their native order.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


SCHEMA_VERSION = "letter_opq_probe_v1"
METHODS = ("pq", "opq", "rq", "opq_rq")


def parse_int_list(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated integer list"
        )
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _matrix_from_payload(
    payload: Any, source: Path
) -> tuple[np.ndarray, list[str] | None, str]:
    matrix_key = "root"
    skill_ids: list[str] | None = None
    if isinstance(payload, dict):
        for candidate in ("skill_ids", "item_ids", "ids"):
            if candidate in payload:
                skill_ids = [str(value) for value in payload[candidate]]
                break
        for candidate in ("embeddings", "cf_embeddings", "item_embeddings"):
            if candidate in payload:
                payload = payload[candidate]
                matrix_key = candidate
                break
        else:
            raise ValueError(
                f"{source} is a mapping but has none of embeddings/cf_embeddings/item_embeddings"
            )
    if hasattr(payload, "detach"):
        payload = payload.detach().cpu().float().numpy()
    matrix = np.asarray(payload)
    if matrix.ndim != 2:
        squeezed = np.squeeze(matrix)
        if squeezed.ndim != 2:
            raise ValueError(
                f"{source} must contain a 2D embedding matrix, got {matrix.shape}"
            )
        matrix = squeezed
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{source} contains non-finite embedding values")
    if skill_ids is not None and len(skill_ids) != len(matrix):
        raise ValueError(
            f"{source} has {len(skill_ids)} skill IDs but {len(matrix)} rows"
        )
    return matrix, skill_ids, matrix_key


def load_embedding_matrix(
    path: Path,
) -> tuple[np.ndarray, list[str] | None, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        payload: Any = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        keys = list(archive.files)
        if len(keys) != 1:
            raise ValueError(f"{path} must contain exactly one NPZ array, got {keys}")
        payload = archive[keys[0]]
    else:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"PyTorch is required to load {path}") from exc
        payload = torch.load(path, map_location="cpu", weights_only=False)
    matrix, skill_ids, matrix_key = _matrix_from_payload(payload, path)
    return (
        matrix,
        skill_ids,
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]),
            "matrix_key": matrix_key,
            "contains_skill_ids": skill_ids is not None,
        },
    )


def load_skill_ids(path: Path) -> list[str]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    skill_ids = [str(row["skill_id"]) for row in rows]
    if len(set(skill_ids)) != len(skill_ids):
        raise ValueError(f"duplicate skill_id in {path}")
    return skill_ids


def align_matrix(
    matrix: np.ndarray,
    matrix_skill_ids: list[str] | None,
    target_skill_ids: list[str],
    source: Path,
) -> tuple[np.ndarray, str]:
    if matrix_skill_ids is None:
        if len(matrix) != len(target_skill_ids):
            raise ValueError(
                f"{source} row count {len(matrix)} != semantic rows {len(target_skill_ids)}"
            )
        return matrix, "row_order_assumed"
    if len(set(matrix_skill_ids)) != len(matrix_skill_ids):
        raise ValueError(f"duplicate skill IDs in {source}")
    index = {skill_id: row for row, skill_id in enumerate(matrix_skill_ids)}
    missing = [skill_id for skill_id in target_skill_ids if skill_id not in index]
    extras = sorted(set(matrix_skill_ids) - set(target_skill_ids))
    if missing or extras:
        raise ValueError(
            f"{source} skill IDs do not match semantic embeddings: "
            f"missing={missing[:3]} extras={extras[:3]}"
        )
    return (
        np.ascontiguousarray(
            matrix[[index[skill_id] for skill_id in target_skill_ids]]
        ),
        "skill_id",
    )


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot L2-normalize a matrix with zero-norm rows")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def entropy_bits(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def effective_count(counts: np.ndarray) -> float:
    return float(2.0 ** entropy_bits(counts))


def collision_summary(codes: np.ndarray) -> dict[str, Any]:
    _, counts = np.unique(np.asarray(codes), axis=0, return_counts=True)
    collided = counts[counts > 1]
    return {
        "unique_tuples": int(len(counts)),
        "collision_groups": int(len(collided)),
        "collided_items": int(collided.sum()) if len(collided) else 0,
        "max_collision": int(counts.max()),
    }


def _within_group_sse_ratio(
    vectors: np.ndarray,
    inverse: np.ndarray,
    counts: np.ndarray,
) -> float:
    vectors = np.asarray(vectors, dtype=np.float32)
    overall_sum = vectors.sum(axis=0, dtype=np.float64)
    total_energy = float(np.sum(vectors * vectors, dtype=np.float64))
    total_centered = total_energy - float(overall_sum @ overall_sum) / len(vectors)
    if total_centered <= 0:
        return 0.0
    group_sums = np.zeros((len(counts), vectors.shape[1]), dtype=np.float32)
    np.add.at(group_sums, inverse, vectors)
    group_energy = float(
        np.sum(
            np.sum(group_sums * group_sums, axis=1, dtype=np.float64)
            / counts.astype(np.float64),
            dtype=np.float64,
        )
    )
    within = max(0.0, total_energy - group_energy)
    return float(within / total_centered)


def prefix_neighbor_recalls(
    codes: np.ndarray,
    sample_indices: np.ndarray,
    neighbors: np.ndarray,
    ks: Iterable[int],
) -> list[dict[str, float]]:
    codes = np.asarray(codes)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if len(sample_indices) != len(neighbors):
        raise ValueError("sample_indices and neighbors must have the same row count")
    result: list[dict[str, float]] = []
    for depth in range(1, codes.shape[1] + 1):
        same = np.all(
            codes[neighbors, :depth] == codes[sample_indices, None, :depth],
            axis=2,
        )
        depth_result: dict[str, float] = {}
        for k in ks:
            actual_k = min(int(k), same.shape[1])
            depth_result[str(k)] = float(same[:, :actual_k].mean())
        result.append(depth_result)
    return result


def audit_code_matrix(
    codes: np.ndarray,
    *,
    codebook_size: int,
    semantic_vectors: np.ndarray | None = None,
    cf_vectors: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    semantic_neighbors: np.ndarray | None = None,
    cf_neighbors: np.ndarray | None = None,
    neighbor_ks: Iterable[int] = (10, 50),
) -> dict[str, Any]:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2 or not len(codes):
        raise ValueError(f"codes must be a non-empty 2D matrix, got {codes.shape}")
    if np.any(codes < 0) or np.any(codes >= codebook_size):
        raise ValueError(f"codes must be in [0,{codebook_size - 1}]")
    neighbor_ks = tuple(int(k) for k in neighbor_ks)
    semantic_recalls = None
    cf_recalls = None
    if sample_indices is not None and semantic_neighbors is not None:
        semantic_recalls = prefix_neighbor_recalls(
            codes, sample_indices, semantic_neighbors, neighbor_ks
        )
    if sample_indices is not None and cf_neighbors is not None:
        cf_recalls = prefix_neighbor_recalls(
            codes, sample_indices, cf_neighbors, neighbor_ks
        )

    positions = []
    for position in range(codes.shape[1]):
        _, counts = np.unique(codes[:, position], return_counts=True)
        positions.append(
            {
                "position": position,
                "used_codes": int(len(counts)),
                "effective_codes": effective_count(counts),
                "entropy_bits": entropy_bits(counts),
                "max_assignment": int(counts.max()),
            }
        )

    prefixes = []
    for depth in range(1, codes.shape[1] + 1):
        _, inverse, counts = np.unique(
            codes[:, :depth], axis=0, return_inverse=True, return_counts=True
        )
        prefix = {
            "depth": depth,
            "unique_prefixes": int(len(counts)),
            "effective_prefixes": effective_count(counts),
            "entropy_bits": entropy_bits(counts),
            "max_group_size": int(counts.max()),
            "p95_group_size": int(np.percentile(counts, 95, method="higher")),
            "singleton_prefix_fraction": float(np.mean(counts == 1)),
            "singleton_item_fraction": float(np.sum(counts == 1) / len(codes)),
        }
        if semantic_vectors is not None:
            prefix["semantic_within_group_sse_ratio"] = _within_group_sse_ratio(
                semantic_vectors, inverse, counts
            )
        if cf_vectors is not None:
            prefix["cf_within_group_sse_ratio"] = _within_group_sse_ratio(
                cf_vectors, inverse, counts
            )
        if semantic_recalls is not None:
            prefix["semantic_neighbor_prefix_recall"] = semantic_recalls[depth - 1]
        if cf_recalls is not None:
            prefix["cf_neighbor_prefix_recall"] = cf_recalls[depth - 1]
        prefixes.append(prefix)
    return {
        "rows": int(len(codes)),
        "depth": int(codes.shape[1]),
        "codebook_size": int(codebook_size),
        "positions": positions,
        "prefixes": prefixes,
        "full_tuple": collision_summary(codes),
    }


def exact_cosine_neighbors(
    faiss: Any,
    vectors: np.ndarray,
    sample_indices: np.ndarray,
    k: int,
) -> np.ndarray:
    if len(vectors) <= 1:
        raise ValueError("nearest-neighbor audit requires at least two vectors")
    k = min(int(k), len(vectors) - 1)
    normalized = normalize_rows(vectors)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    _, candidates = index.search(normalized[sample_indices], min(len(vectors), k + 1))
    neighbors = np.empty((len(sample_indices), k), dtype=np.int64)
    for row, item_index in enumerate(sample_indices.tolist()):
        filtered = [
            int(value) for value in candidates[row].tolist() if int(value) != item_index
        ]
        if len(filtered) < k:
            raise RuntimeError(
                f"could not collect {k} non-self neighbors for row {item_index}"
            )
        neighbors[row] = filtered[:k]
    return neighbors


def reconstruction_metrics(
    original: np.ndarray, reconstructed: np.ndarray
) -> dict[str, float]:
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"reconstruction shape {reconstructed.shape} != input {original.shape}"
        )
    error = np.asarray(original - reconstructed, dtype=np.float32)
    per_vector_sq = np.sum(error * error, axis=1, dtype=np.float64)
    centered = original.astype(np.float64) - original.mean(axis=0, dtype=np.float64)
    centered_energy = float(np.sum(centered * centered, dtype=np.float64))
    input_energy = float(np.sum(original * original, dtype=np.float64))
    original_norm = np.linalg.norm(original, axis=1)
    reconstructed_norm = np.linalg.norm(reconstructed, axis=1)
    denom = original_norm * reconstructed_norm
    cosine = np.divide(
        np.sum(original * reconstructed, axis=1, dtype=np.float64),
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0,
    )
    sse = float(per_vector_sq.sum())
    return {
        "mse_per_dimension": float(np.mean(error * error, dtype=np.float64)),
        "mean_squared_l2_per_vector": float(per_vector_sq.mean()),
        "p95_l2_error": float(np.percentile(np.sqrt(per_vector_sq), 95)),
        "variance_normalized_sse": (
            float(sse / centered_energy) if centered_energy else 0.0
        ),
        "energy_normalized_sse": float(sse / input_energy) if input_energy else 0.0,
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(np.percentile(cosine, 5)),
    }


def neighbor_overlap(
    reference_neighbors: np.ndarray,
    candidate_neighbors: np.ndarray,
    ks: Iterable[int],
) -> dict[str, float]:
    if reference_neighbors.shape != candidate_neighbors.shape:
        raise ValueError("neighbor matrices must have the same shape")
    result = {}
    for k in ks:
        actual_k = min(int(k), reference_neighbors.shape[1])
        overlaps = []
        for reference, candidate in zip(reference_neighbors, candidate_neighbors):
            overlaps.append(
                len(
                    set(reference[:actual_k].tolist())
                    & set(candidate[:actual_k].tolist())
                )
                / actual_k
            )
        result[str(k)] = float(np.mean(overlaps))
    return result


def parallel_order_report(
    codes: np.ndarray,
    sample_indices: np.ndarray,
    semantic_neighbors: np.ndarray,
    order_k: int,
) -> dict[str, Any]:
    depth = codes.shape[1]
    if depth > 7:
        raise ValueError("refusing factorial position audit for depth > 7")
    entries = []
    for permutation in itertools.permutations(range(depth)):
        ordered = codes[:, permutation]
        recalls = prefix_neighbor_recalls(
            ordered,
            sample_indices,
            semantic_neighbors[:, : min(order_k, semantic_neighbors.shape[1])],
            (order_k,),
        )
        score_vector = [
            recalls[level][str(order_k)] for level in range(max(0, depth - 1))
        ]
        unique_prefixes = []
        max_group_sizes = []
        for level in range(1, depth + 1):
            _, counts = np.unique(ordered[:, :level], axis=0, return_counts=True)
            unique_prefixes.append(int(len(counts)))
            max_group_sizes.append(int(counts.max()))
        entries.append(
            {
                "permutation": list(permutation),
                "semantic_neighbor_prefix_recall_at_order_k": score_vector,
                "unique_prefixes": unique_prefixes,
                "max_group_sizes": max_group_sizes,
            }
        )

    def best_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        return (
            tuple(
                -value for value in entry["semantic_neighbor_prefix_recall_at_order_k"]
            ),
            tuple(entry["permutation"]),
        )

    def worst_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        return (
            tuple(entry["semantic_neighbor_prefix_recall_at_order_k"]),
            tuple(entry["permutation"]),
        )

    identity = list(range(depth))
    native = next(entry for entry in entries if entry["permutation"] == identity)
    return {
        "selection_rule": (
            "lexicographically maximize catalog semantic-neighbor prefix recall from depth 1 "
            f"to {max(1, depth - 1)} at k={order_k}; tie-break by lexicographic permutation"
        ),
        "native": native,
        "selected": min(entries, key=best_key),
        "worst": min(entries, key=worst_key),
        "all_permutations": entries,
    }


def apply_position_order(
    codes_native: np.ndarray, permutation: Iterable[int]
) -> np.ndarray:
    permutation = tuple(int(value) for value in permutation)
    return np.ascontiguousarray(codes_native[:, permutation], dtype=np.int64)


def undo_position_order(
    codes_ordered: np.ndarray, permutation: Iterable[int]
) -> np.ndarray:
    permutation = tuple(int(value) for value in permutation)
    native = np.empty_like(codes_ordered)
    for ordered_position, native_position in enumerate(permutation):
        native[:, native_position] = codes_ordered[:, ordered_position]
    return native


def minimum_cost_unique_last_codes(
    codes: np.ndarray,
    last_code_costs: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import linear_sum_assignment

    codes = np.asarray(codes, dtype=np.int64)
    costs = np.asarray(last_code_costs, dtype=np.float64)
    if codes.ndim != 2 or not len(codes):
        raise ValueError("codes must be a non-empty 2D matrix")
    if costs.ndim != 2 or costs.shape[0] != len(codes):
        raise ValueError(
            f"last_code_costs must have shape ({len(codes)}, K), got {costs.shape}"
        )
    if not np.isfinite(costs).all():
        raise ValueError("last_code_costs contains non-finite values")
    prefix_to_items: dict[tuple[int, ...], list[int]] = {}
    for item_index, code in enumerate(codes.tolist()):
        prefix_to_items.setdefault(tuple(code[:-1]), []).append(item_index)
    resolved = codes.copy()
    audit = {
        "method": "minimum_cost_unique_last_code",
        "repaired_prefixes": 0,
        "repaired_items": 0,
        "changed_items": 0,
        "max_repaired_prefix_size": 0,
        "original_assignment_cost_on_repaired_items": 0.0,
        "assignment_cost": 0.0,
    }
    for prefix in sorted(prefix_to_items):
        item_indices = prefix_to_items[prefix]
        counts = Counter(int(codes[item_index, -1]) for item_index in item_indices)
        if max(counts.values()) <= 1:
            continue
        if len(item_indices) > costs.shape[1]:
            raise ValueError(
                f"prefix {prefix} has {len(item_indices)} items but only {costs.shape[1]} last codes"
            )
        audit["repaired_prefixes"] += 1
        audit["repaired_items"] += len(item_indices)
        audit["max_repaired_prefix_size"] = max(
            audit["max_repaired_prefix_size"], len(item_indices)
        )
        original = codes[item_indices, -1]
        audit["original_assignment_cost_on_repaired_items"] += float(
            costs[item_indices, original].sum()
        )
        local_rows, assigned = linear_sum_assignment(costs[item_indices])
        if len(local_rows) != len(item_indices):
            raise RuntimeError(f"incomplete last-code assignment for prefix {prefix}")
        for local_row, assigned_code in zip(local_rows.tolist(), assigned.tolist()):
            item_index = item_indices[local_row]
            audit["changed_items"] += int(codes[item_index, -1] != assigned_code)
            audit["assignment_cost"] += float(costs[item_index, assigned_code])
            resolved[item_index, -1] = int(assigned_code)
    audit["assignment_cost_delta"] = (
        audit["assignment_cost"] - audit["original_assignment_cost_on_repaired_items"]
    )
    post = collision_summary(resolved)
    audit["post_collision_groups"] = post["collision_groups"]
    if post["collision_groups"]:
        raise RuntimeError("minimum-cost assignment left duplicate full code tuples")
    return resolved, audit


def _squared_distances(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    distances = (
        np.sum(vectors * vectors, axis=1, keepdims=True, dtype=np.float64)
        + np.sum(centroids * centroids, axis=1, dtype=np.float64)[None, :]
        - 2.0 * vectors.astype(np.float64) @ centroids.astype(np.float64).T
    )
    return np.maximum(distances, 0.0)


@dataclass
class FittedQuantizer:
    method: str
    codes_native: np.ndarray
    quant_vectors: np.ndarray
    codebooks: np.ndarray
    layout: str
    parallel_positions: bool
    reverse_transform: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def reconstruct_native(self, codes_native: np.ndarray) -> np.ndarray:
        codes_native = np.asarray(codes_native, dtype=np.int64)
        if self.layout == "product":
            quantized = np.concatenate(
                [
                    self.codebooks[level, codes_native[:, level]]
                    for level in range(codes_native.shape[1])
                ],
                axis=1,
            )
        elif self.layout == "residual":
            quantized = np.zeros_like(self.quant_vectors)
            for level in range(codes_native.shape[1]):
                quantized += self.codebooks[level, codes_native[:, level]]
        else:
            raise ValueError(f"unknown quantizer layout: {self.layout}")
        return np.ascontiguousarray(
            self.reverse_transform(np.ascontiguousarray(quantized))
        )

    def level_costs(self, native_level: int) -> np.ndarray:
        if self.layout == "product":
            subdim = self.quant_vectors.shape[1] // self.codes_native.shape[1]
            block = self.quant_vectors[
                :, native_level * subdim : (native_level + 1) * subdim
            ]
            return _squared_distances(block, self.codebooks[native_level])
        residual = self.quant_vectors.copy()
        for level in range(native_level):
            residual -= self.codebooks[level, self.codes_native[:, level]]
        return _squared_distances(residual, self.codebooks[native_level])


def configure_clustering(pq_or_rq: Any, *, seed: int, iterations: int) -> None:
    pq_or_rq.cp.seed = int(seed)
    pq_or_rq.cp.niter = int(iterations)
    pq_or_rq.cp.nredo = 1


def fit_product_quantizer(
    faiss: Any,
    vectors: np.ndarray,
    *,
    depth: int,
    nbits: int,
    seed: int,
    kmeans_iters: int,
    opq_iters: int,
    opq_pq_iters: int,
    optimized: bool,
    verbose: bool,
) -> FittedQuantizer:
    dim = vectors.shape[1]
    if dim % depth:
        raise ValueError(
            f"product quantization requires dim {dim} divisible by depth {depth}"
        )
    pq = faiss.ProductQuantizer(dim, depth, nbits)
    configure_clustering(pq, seed=seed, iterations=kmeans_iters)
    pq.verbose = bool(verbose)
    state: dict[str, np.ndarray] = {}
    if optimized:
        transform = faiss.OPQMatrix(dim, depth)
        transform.niter = int(opq_iters)
        transform.niter_pq_0 = int(kmeans_iters)
        transform.niter_pq = int(opq_pq_iters)
        transform.max_train_points = int(len(vectors))
        transform.verbose = bool(verbose)
        transform.pq = pq
        transform.train(vectors)
        quant_vectors = np.ascontiguousarray(
            transform.apply_py(vectors), dtype=np.float32
        )

        def reverse(values: np.ndarray) -> np.ndarray:
            return transform.reverse_transform(values)

        state["rotation_A"] = faiss.vector_to_array(transform.A).copy()
        state["rotation_b"] = faiss.vector_to_array(transform.b).copy()
        method = "opq"
    else:
        pq.train(vectors)
        quant_vectors = vectors

        def reverse(values: np.ndarray) -> np.ndarray:
            return values

        method = "pq"
    codes = np.asarray(pq.compute_codes(quant_vectors), dtype=np.int64)
    codebook_size = 1 << nbits
    centroids = (
        faiss.vector_to_array(pq.centroids)
        .reshape(depth, codebook_size, dim // depth)
        .copy()
    )
    state["centroids"] = centroids
    return FittedQuantizer(
        method=method,
        codes_native=codes,
        quant_vectors=quant_vectors,
        codebooks=centroids,
        layout="product",
        parallel_positions=True,
        reverse_transform=reverse,
        metadata={
            "optimized_rotation": optimized,
            "kmeans_iters": kmeans_iters,
            "opq_iters": opq_iters if optimized else None,
            "opq_pq_iters": opq_pq_iters if optimized else None,
        },
        state=state,
    )


def fit_residual_quantizer(
    faiss: Any,
    quant_vectors: np.ndarray,
    *,
    depth: int,
    nbits: int,
    seed: int,
    kmeans_iters: int,
    max_beam_size: int,
    method: str = "rq",
    reverse_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    extra_state: dict[str, np.ndarray] | None = None,
    verbose: bool = False,
) -> FittedQuantizer:
    rq = faiss.ResidualQuantizer(quant_vectors.shape[1], depth, nbits)
    configure_clustering(rq, seed=seed, iterations=kmeans_iters)
    rq.max_beam_size = int(max_beam_size)
    rq.train_type = rq.Train_progressive_dim
    rq.verbose = bool(verbose)
    rq.train(quant_vectors)
    codes = np.asarray(rq.compute_codes(quant_vectors), dtype=np.int64)
    codebook_size = 1 << nbits
    codebooks = (
        faiss.vector_to_array(rq.codebooks)
        .reshape(depth, codebook_size, quant_vectors.shape[1])
        .copy()
    )
    state = {"codebooks": codebooks}
    if extra_state:
        state.update(extra_state)
    return FittedQuantizer(
        method=method,
        codes_native=codes,
        quant_vectors=quant_vectors,
        codebooks=codebooks,
        layout="residual",
        parallel_positions=False,
        reverse_transform=reverse_transform or (lambda values: values),
        metadata={
            "kmeans_iters": kmeans_iters,
            "max_beam_size": max_beam_size,
            "train_type": "Train_progressive_dim",
        },
        state=state,
    )


def semantic_ids_sha256(skill_ids: list[str], codes: np.ndarray) -> str:
    canonical = json.dumps(
        sorted(
            (skill_id, [int(value) for value in codes[index]])
            for index, skill_id in enumerate(skill_ids)
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_semantic_ids(
    path: Path,
    skill_ids: list[str],
    codes: np.ndarray,
    raw_codes: np.ndarray,
    method: str,
) -> None:
    width = max(3, len(str(int(codes.max()))))
    rows = []
    for index in sorted(range(len(skill_ids)), key=lambda value: skill_ids[value]):
        semantic_id = [int(value) for value in codes[index]]
        row = {
            "skill_id": skill_ids[index],
            "quantizer_codes": semantic_id,
            "semantic_id": semantic_id,
            "semantic_id_mode": "learned_codes",
            "tokens": [
                f"<RQ{level}_{code:0{width}d}>"
                for level, code in enumerate(semantic_id)
            ],
            "quantizer_method": method,
        }
        if not np.array_equal(codes[index], raw_codes[index]):
            row["pre_assignment_codes"] = [int(value) for value in raw_codes[index]]
        rows.append(row)
    write_jsonl(path, rows)


def load_letter_codes(
    path: Path,
    target_skill_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_skill = {str(row["skill_id"]): row for row in rows}
    if len(by_skill) != len(rows):
        raise ValueError(f"duplicate skill_id in {path}")
    missing = [skill_id for skill_id in target_skill_ids if skill_id not in by_skill]
    extras = sorted(set(by_skill) - set(target_skill_ids))
    if missing or extras:
        raise ValueError(
            f"{path} does not align with embeddings: missing={missing[:3]} extras={extras[:3]}"
        )
    final = np.asarray(
        [by_skill[skill_id]["semantic_id"] for skill_id in target_skill_ids],
        dtype=np.int64,
    )
    raw = np.asarray(
        [
            by_skill[skill_id].get(
                "pre_assignment_rq_codes",
                by_skill[skill_id].get(
                    "pre_assignment_codes", by_skill[skill_id]["semantic_id"]
                ),
            )
            for skill_id in target_skill_ids
        ],
        dtype=np.int64,
    )
    return (
        raw,
        final,
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": len(rows),
        },
    )


def process_fitted_quantizer(
    fit: FittedQuantizer,
    *,
    quantizer_vectors: np.ndarray,
    semantic_audit_vectors: np.ndarray,
    cf_vectors: np.ndarray | None,
    skill_ids: list[str],
    method_dir: Path,
    codebook_size: int,
    sample_indices: np.ndarray,
    semantic_neighbors: np.ndarray,
    cf_neighbors: np.ndarray | None,
    neighbor_ks: list[int],
    order_k: int,
    faiss: Any,
) -> dict[str, Any]:
    if fit.parallel_positions:
        order_report = parallel_order_report(
            fit.codes_native, sample_indices, semantic_neighbors, order_k
        )
        permutation = order_report["selected"]["permutation"]
    else:
        permutation = list(range(fit.codes_native.shape[1]))
        order_report = {
            "selection_rule": "native residual order; position permutation is not allowed",
            "native": {"permutation": permutation},
            "selected": {"permutation": permutation},
            "worst": None,
            "all_permutations": None,
        }
    raw_codes = apply_position_order(fit.codes_native, permutation)
    raw_collision = collision_summary(raw_codes)
    if raw_collision["collision_groups"]:
        last_native_level = int(permutation[-1])
        resolved_codes, collision_resolution = minimum_cost_unique_last_codes(
            raw_codes, fit.level_costs(last_native_level)
        )
    else:
        resolved_codes = raw_codes.copy()
        collision_resolution = {
            "method": "none",
            "repaired_prefixes": 0,
            "repaired_items": 0,
            "changed_items": 0,
            "max_repaired_prefix_size": 0,
            "assignment_cost_delta": 0.0,
            "post_collision_groups": 0,
        }
    native_resolved = undo_position_order(resolved_codes, permutation)
    raw_reconstruction = fit.reconstruct_native(fit.codes_native)
    resolved_reconstruction = fit.reconstruct_native(native_resolved)
    raw_audit = audit_code_matrix(
        raw_codes,
        codebook_size=codebook_size,
        semantic_vectors=semantic_audit_vectors,
        cf_vectors=cf_vectors,
        sample_indices=sample_indices,
        semantic_neighbors=semantic_neighbors,
        cf_neighbors=cf_neighbors,
        neighbor_ks=neighbor_ks,
    )
    resolved_audit = audit_code_matrix(
        resolved_codes,
        codebook_size=codebook_size,
        semantic_vectors=semantic_audit_vectors,
        cf_vectors=cf_vectors,
        sample_indices=sample_indices,
        semantic_neighbors=semantic_neighbors,
        cf_neighbors=cf_neighbors,
        neighbor_ks=neighbor_ks,
    )
    reconstructed_neighbors = exact_cosine_neighbors(
        faiss, resolved_reconstruction, sample_indices, max(neighbor_ks)
    )

    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "codes.native.npy", fit.codes_native.astype(np.uint16))
    np.save(method_dir / "codes.selected_raw.npy", raw_codes.astype(np.uint16))
    np.save(
        method_dir / "codes.selected_resolved.npy", resolved_codes.astype(np.uint16)
    )
    np.savez_compressed(method_dir / "quantizer_state.npz", **fit.state)
    write_semantic_ids(
        method_dir / "semantic_ids.jsonl",
        skill_ids,
        resolved_codes,
        raw_codes,
        fit.method,
    )
    report = {
        "method": fit.method,
        "family": "parallel_product" if fit.parallel_positions else "residual_additive",
        "metadata": fit.metadata,
        "selected_position_order": permutation,
        "position_order_audit": order_report,
        "raw_code_audit": raw_audit,
        "resolved_code_audit": resolved_audit,
        "collision_resolution": collision_resolution,
        "raw_reconstruction": reconstruction_metrics(
            quantizer_vectors, raw_reconstruction
        ),
        "resolved_reconstruction": reconstruction_metrics(
            quantizer_vectors, resolved_reconstruction
        ),
        "resolved_reconstruction_semantic_neighbor_overlap": neighbor_overlap(
            semantic_neighbors, reconstructed_neighbors, neighbor_ks
        ),
        "semantic_ids_sha256": semantic_ids_sha256(skill_ids, resolved_codes),
        "artifacts": {
            "semantic_ids": str(method_dir / "semantic_ids.jsonl"),
            "codes_native": str(method_dir / "codes.native.npy"),
            "codes_selected_raw": str(method_dir / "codes.selected_raw.npy"),
            "codes_selected_resolved": str(method_dir / "codes.selected_resolved.npy"),
            "quantizer_state": str(method_dir / "quantizer_state.npz"),
        },
    }
    write_json(
        method_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "method": "letter_opq_identifier_probe",
            "quantizer": fit.method,
            "semantic_id_mode": "learned_codes",
            "skills": len(skill_ids),
            "num_emb_list": [codebook_size] * resolved_codes.shape[1],
            "selected_position_order": permutation,
            "semantic_ids_sha256": report["semantic_ids_sha256"],
            "note": (
                "This is an OPQ-validation artifact, not an "
                "official_letter_rqvae_adapter manifest."
            ),
        },
    )
    write_json(method_dir / "audit.json", report)
    return report


def _prefix_vector(report: dict[str, Any], space: str, k: int) -> list[float]:
    result = []
    prefixes = report["resolved_code_audit"]["prefixes"]
    for prefix in prefixes[:-1]:
        result.append(float(prefix[f"{space}_neighbor_prefix_recall"][str(k)]))
    return result


def build_offline_gate(
    reports: dict[str, dict[str, Any]],
    letter_report: dict[str, Any] | None,
    *,
    k: int,
    tolerance: float,
    min_distortion_improvement: float,
    has_cf: bool,
) -> dict[str, Any]:
    if "opq" not in reports or "rq" not in reports:
        return {"available": False, "reason": "both opq and rq methods are required"}
    opq = reports["opq"]
    rq = reports["rq"]
    opq_distortion = opq["resolved_reconstruction"]["variance_normalized_sse"]
    rq_distortion = rq["resolved_reconstruction"]["variance_normalized_sse"]
    improvement = 1.0 - opq_distortion / rq_distortion if rq_distortion else 0.0
    opq_semantic = _prefix_vector(opq, "semantic", k)
    rq_semantic = _prefix_vector(rq, "semantic", k)
    semantic_deltas = [left - right for left, right in zip(opq_semantic, rq_semantic)]
    pure_checks = {
        "distortion_relative_improvement": improvement,
        "distortion_pass": improvement >= min_distortion_improvement,
        "semantic_prefix_deltas_opq_minus_rq": semantic_deltas,
        "semantic_prefix_noninferiority_pass": all(
            value >= -tolerance for value in semantic_deltas
        ),
    }
    if has_cf:
        opq_cf = _prefix_vector(opq, "cf", k)
        rq_cf = _prefix_vector(rq, "cf", k)
        cf_deltas = [left - right for left, right in zip(opq_cf, rq_cf)]
        pure_checks["cf_prefix_deltas_opq_minus_rq"] = cf_deltas
        pure_checks["cf_prefix_noninferiority_pass"] = all(
            value >= -tolerance for value in cf_deltas
        )
    pure_pass_keys = [key for key in pure_checks if key.endswith("_pass")]
    pure_checks["advance_pure_quantizer_candidate"] = all(
        bool(pure_checks[key]) for key in pure_pass_keys
    )

    replacement: dict[str, Any] = {
        "available": letter_report is not None,
        "reason": (
            None
            if letter_report is not None
            else "--letter-semantic-ids was not provided"
        ),
    }
    if letter_report is not None:
        letter_semantic = _prefix_vector(letter_report, "semantic", k)
        semantic_vs_letter = [
            left - right for left, right in zip(opq_semantic, letter_semantic)
        ]
        replacement["semantic_prefix_deltas_opq_minus_letter"] = semantic_vs_letter
        replacement["semantic_prefix_noninferiority_pass"] = all(
            value >= -tolerance for value in semantic_vs_letter
        )
        if has_cf:
            opq_cf = _prefix_vector(opq, "cf", k)
            letter_cf = _prefix_vector(letter_report, "cf", k)
            cf_vs_letter = [left - right for left, right in zip(opq_cf, letter_cf)]
            replacement["cf_prefix_deltas_opq_minus_letter"] = cf_vs_letter
            replacement["cf_prefix_noninferiority_pass"] = all(
                value >= -tolerance for value in cf_vs_letter
            )
        pass_keys = [key for key in replacement if key.endswith("_pass")]
        replacement["advance_to_decoder_ab"] = pure_checks[
            "advance_pure_quantizer_candidate"
        ] and all(bool(replacement[key]) for key in pass_keys)
    return {
        "available": True,
        "neighbor_k": k,
        "absolute_prefix_noninferiority_tolerance": tolerance,
        "minimum_distortion_relative_improvement": min_distortion_improvement,
        "pure_quantizer_opq_vs_rq": pure_checks,
        "system_replacement_opq_vs_letter": replacement,
        "warning": (
            "Passing this offline gate only authorizes a from-scratch decoder A/B; it is not an "
            "end-to-end retrieval result. Standard OPQ does not contain LETTER's CF loss."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument(
        "--semantic-audit-embeddings",
        type=Path,
        help=(
            "optional original semantic vectors for prefix audits when --embeddings contains "
            "frozen LETTER latents"
        ),
    )
    parser.add_argument(
        "--quantizer-space",
        choices=("semantic", "letter_latent", "custom"),
        default="semantic",
    )
    parser.add_argument("--cf-embeddings", type=Path)
    parser.add_argument("--skills-jsonl", type=Path)
    parser.add_argument("--letter-semantic-ids", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default="pq,opq,rq")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--normalization", choices=("none", "l2"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--opq-iters", type=int, default=20)
    parser.add_argument("--opq-pq-iters", type=int, default=4)
    parser.add_argument("--rq-max-beam-size", type=int, default=16)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--neighbor-sample-size", type=int, default=2048)
    parser.add_argument("--neighbor-ks", type=parse_int_list, default=[10, 50])
    parser.add_argument("--order-k", type=int, default=10)
    parser.add_argument("--prefix-noninferiority-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--min-distortion-relative-improvement", type=float, default=0.0
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"unknown --methods values: {unknown}; expected {METHODS}")
    if not methods:
        raise SystemExit("--methods must select at least one quantizer")
    if len(set(methods)) != len(methods):
        raise SystemExit("--methods contains duplicates")
    if args.depth < 1:
        raise SystemExit("--depth must be positive")
    if args.codebook_size != 256:
        raise SystemExit(
            "this LETTER comparison currently requires --codebook-size 256 (8-bit codes)"
        )
    if args.neighbor_sample_size < 1:
        raise SystemExit("--neighbor-sample-size must be positive")
    if any(value < 1 for value in args.neighbor_ks) or args.order_k < 1:
        raise SystemExit("--neighbor-ks and --order-k must be positive")
    if args.kmeans_iters < 1 or args.opq_iters < 1 or args.opq_pq_iters < 1:
        raise SystemExit("all quantizer iteration counts must be positive")
    if args.rq_max_beam_size < 1 or args.faiss_threads < 1:
        raise SystemExit("--rq-max-beam-size and --faiss-threads must be positive")
    if args.prefix_noninferiority_tolerance < 0:
        raise SystemExit("--prefix-noninferiority-tolerance must be non-negative")
    if (
        args.quantizer_space == "letter_latent"
        and args.semantic_audit_embeddings is None
    ):
        raise SystemExit(
            "--quantizer-space letter_latent requires --semantic-audit-embeddings"
        )
    if args.order_k not in args.neighbor_ks:
        args.neighbor_ks = sorted(set([*args.neighbor_ks, args.order_k]))
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise SystemExit(
            f"--output-dir exists but is not a directory: {args.output_dir}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "faiss is required; install the historically pinned faiss-cpu==1.13.2"
        ) from exc
    faiss.omp_set_num_threads(int(args.faiss_threads))

    embeddings, payload_skill_ids, embedding_meta = load_embedding_matrix(
        args.embeddings
    )
    if payload_skill_ids is None:
        if args.skills_jsonl is None:
            raise SystemExit(
                "embedding payload has no skill_ids; provide --skills-jsonl"
            )
        skill_ids = load_skill_ids(args.skills_jsonl)
    else:
        skill_ids = payload_skill_ids
    if len(skill_ids) != len(embeddings) or len(set(skill_ids)) != len(skill_ids):
        raise SystemExit("semantic embedding skill IDs must be unique and row-aligned")
    if len(embeddings) < args.codebook_size:
        raise SystemExit(
            f"quantizer training requires at least {args.codebook_size} embedding rows"
        )
    if args.normalization == "l2":
        embeddings = normalize_rows(embeddings)

    semantic_audit_vectors = embeddings
    semantic_audit_meta = embedding_meta
    if args.semantic_audit_embeddings is not None:
        raw_semantic_audit, audit_skill_ids, semantic_audit_meta = (
            load_embedding_matrix(args.semantic_audit_embeddings)
        )
        semantic_audit_vectors, alignment = align_matrix(
            raw_semantic_audit,
            audit_skill_ids,
            skill_ids,
            args.semantic_audit_embeddings,
        )
        semantic_audit_meta["alignment"] = alignment

    cf_vectors = None
    cf_meta = None
    if args.cf_embeddings is not None:
        raw_cf, cf_skill_ids, cf_meta = load_embedding_matrix(args.cf_embeddings)
        cf_vectors, alignment = align_matrix(
            raw_cf, cf_skill_ids, skill_ids, args.cf_embeddings
        )
        cf_meta["alignment"] = alignment

    rng = np.random.default_rng(args.seed)
    sample_size = min(args.neighbor_sample_size, len(embeddings))
    sample_indices = np.sort(
        rng.choice(len(embeddings), size=sample_size, replace=False)
    )
    max_neighbor_k = min(max(args.neighbor_ks), len(embeddings) - 1)
    args.order_k = min(args.order_k, max_neighbor_k)
    args.neighbor_ks = [min(value, max_neighbor_k) for value in args.neighbor_ks]
    args.neighbor_ks = sorted(set([*args.neighbor_ks, args.order_k]))
    semantic_neighbors = exact_cosine_neighbors(
        faiss, semantic_audit_vectors, sample_indices, max(args.neighbor_ks)
    )
    cf_neighbors = (
        exact_cosine_neighbors(faiss, cf_vectors, sample_indices, max(args.neighbor_ks))
        if cf_vectors is not None
        else None
    )

    letter_report = None
    letter_meta = None
    if args.letter_semantic_ids is not None:
        raw_letter, final_letter, letter_meta = load_letter_codes(
            args.letter_semantic_ids, skill_ids
        )
        if final_letter.shape[1] != args.depth:
            raise SystemExit(
                f"LETTER depth {final_letter.shape[1]} != requested probe depth {args.depth}"
            )
        letter_report = {
            "method": "letter_rqvae_cf",
            "raw_code_audit": audit_code_matrix(
                raw_letter,
                codebook_size=args.codebook_size,
                semantic_vectors=semantic_audit_vectors,
                cf_vectors=cf_vectors,
                sample_indices=sample_indices,
                semantic_neighbors=semantic_neighbors,
                cf_neighbors=cf_neighbors,
                neighbor_ks=args.neighbor_ks,
            ),
            "resolved_code_audit": audit_code_matrix(
                final_letter,
                codebook_size=args.codebook_size,
                semantic_vectors=semantic_audit_vectors,
                cf_vectors=cf_vectors,
                sample_indices=sample_indices,
                semantic_neighbors=semantic_neighbors,
                cf_neighbors=cf_neighbors,
                neighbor_ks=args.neighbor_ks,
            ),
            "semantic_ids_sha256": semantic_ids_sha256(skill_ids, final_letter),
            "source": letter_meta,
        }

    nbits = int(math.log2(args.codebook_size))
    reports: dict[str, dict[str, Any]] = {}
    fits: dict[str, FittedQuantizer] = {}
    for method in methods:
        started = time.time()
        print(f"fitting {method} on {embeddings.shape}", flush=True)
        if method == "pq":
            fit = fit_product_quantizer(
                faiss,
                embeddings,
                depth=args.depth,
                nbits=nbits,
                seed=args.seed,
                kmeans_iters=args.kmeans_iters,
                opq_iters=args.opq_iters,
                opq_pq_iters=args.opq_pq_iters,
                optimized=False,
                verbose=args.verbose,
            )
        elif method == "opq":
            fit = fit_product_quantizer(
                faiss,
                embeddings,
                depth=args.depth,
                nbits=nbits,
                seed=args.seed,
                kmeans_iters=args.kmeans_iters,
                opq_iters=args.opq_iters,
                opq_pq_iters=args.opq_pq_iters,
                optimized=True,
                verbose=args.verbose,
            )
        elif method == "rq":
            fit = fit_residual_quantizer(
                faiss,
                embeddings,
                depth=args.depth,
                nbits=nbits,
                seed=args.seed,
                kmeans_iters=args.kmeans_iters,
                max_beam_size=args.rq_max_beam_size,
                verbose=args.verbose,
            )
        else:
            opq_fit = fits.get("opq")
            if opq_fit is None:
                opq_fit = fit_product_quantizer(
                    faiss,
                    embeddings,
                    depth=args.depth,
                    nbits=nbits,
                    seed=args.seed,
                    kmeans_iters=args.kmeans_iters,
                    opq_iters=args.opq_iters,
                    opq_pq_iters=args.opq_pq_iters,
                    optimized=True,
                    verbose=args.verbose,
                )
            fit = fit_residual_quantizer(
                faiss,
                opq_fit.quant_vectors,
                depth=args.depth,
                nbits=nbits,
                seed=args.seed,
                kmeans_iters=args.kmeans_iters,
                max_beam_size=args.rq_max_beam_size,
                method="opq_rq",
                reverse_transform=opq_fit.reverse_transform,
                extra_state={
                    key: value
                    for key, value in opq_fit.state.items()
                    if key.startswith("rotation_")
                },
                verbose=args.verbose,
            )
        fits[method] = fit
        report = process_fitted_quantizer(
            fit,
            quantizer_vectors=embeddings,
            semantic_audit_vectors=semantic_audit_vectors,
            cf_vectors=cf_vectors,
            skill_ids=skill_ids,
            method_dir=args.output_dir / method,
            codebook_size=args.codebook_size,
            sample_indices=sample_indices,
            semantic_neighbors=semantic_neighbors,
            cf_neighbors=cf_neighbors,
            neighbor_ks=args.neighbor_ks,
            order_k=args.order_k,
            faiss=faiss,
        )
        report["wall_seconds"] = time.time() - started
        reports[method] = report
        print(f"finished {method} in {report['wall_seconds']:.2f}s", flush=True)

    gate = build_offline_gate(
        reports,
        letter_report,
        k=args.order_k,
        tolerance=args.prefix_noninferiority_tolerance,
        min_distortion_improvement=args.min_distortion_relative_improvement,
        has_cf=cf_vectors is not None,
    )
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": "replace LETTER residual quantization with standard OPQ+PQ",
        "scope": "offline identifier audit; no decoder was trained or evaluated",
        "protocol": {
            "depth": args.depth,
            "codebook_size": args.codebook_size,
            "bits_per_identifier": args.depth * nbits,
            "normalization": args.normalization,
            "quantizer_space": args.quantizer_space,
            "seed": args.seed,
            "methods": methods,
            "kmeans_iters": args.kmeans_iters,
            "opq_iters": args.opq_iters,
            "opq_pq_iters": args.opq_pq_iters,
            "rq_max_beam_size": args.rq_max_beam_size,
            "faiss_threads": args.faiss_threads,
            "neighbor_sample_size": sample_size,
            "neighbor_sample_indices_sha256": hashlib.sha256(
                sample_indices.astype("<i8").tobytes()
            ).hexdigest(),
            "neighbor_ks": args.neighbor_ks,
            "order_k": args.order_k,
            "no_core75_tuning": True,
        },
        "environment": {
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "numpy_version": np.__version__,
        },
        "inputs": {
            "quantizer_embeddings": embedding_meta,
            "semantic_audit_embeddings": semantic_audit_meta,
            "cf_embeddings": cf_meta,
            "letter_semantic_ids": letter_meta,
        },
        "letter_baseline": letter_report,
        "methods": reports,
        "offline_gate": gate,
    }
    write_json(args.output_dir / "comparison.json", comparison)
    print(f"saved comparison to {args.output_dir / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
