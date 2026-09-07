#!/usr/bin/env python3
"""Build long semantic-only or CF-aware OPQ DocIDs for SkillGR.

The default path preserves the historical semantic-only M=32/M=64 artifacts.
The optional CF path forms a provenance-checked, weighted concatenation of
row-normalized semantic and collaborative views before standard OPQ+PQ.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from probe_letter_opq import (
    align_matrix,
    collision_summary,
    file_sha256,
    fit_product_quantizer,
    load_embedding_matrix,
    load_skill_ids,
    minimum_cost_unique_last_codes,
    normalize_rows,
    reconstruction_metrics,
    semantic_ids_sha256,
    write_json,
    write_semantic_ids,
)


SCHEMA_VERSION = "letter_opq_docids_v1"
ALLOWED_DEPTHS = (32, 64)


def weighted_semantic_cf_fusion(
    semantic_vectors: np.ndarray,
    cf_vectors: np.ndarray,
    *,
    cf_energy_fraction: float,
    dimension_multiple: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a joint metric whose squared L2 energy has an explicit view mix."""
    if semantic_vectors.ndim != 2 or cf_vectors.ndim != 2:
        raise ValueError("semantic and CF embeddings must both be 2D")
    if len(semantic_vectors) != len(cf_vectors):
        raise ValueError(
            f"semantic/CF row mismatch: {len(semantic_vectors)} != {len(cf_vectors)}"
        )
    if not 0.0 < cf_energy_fraction < 1.0:
        raise ValueError("cf_energy_fraction must be strictly between 0 and 1")
    if dimension_multiple < 1:
        raise ValueError("dimension_multiple must be positive")

    semantic_weight = float(np.sqrt(1.0 - cf_energy_fraction))
    cf_weight = float(np.sqrt(cf_energy_fraction))
    semantic_normalized = normalize_rows(semantic_vectors)
    cf_normalized = normalize_rows(cf_vectors)
    unpadded = np.concatenate(
        [
            semantic_weight * semantic_normalized,
            cf_weight * cf_normalized,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    padding_dim = (-unpadded.shape[1]) % dimension_multiple
    if padding_dim:
        unpadded = np.pad(unpadded, ((0, 0), (0, padding_dim)))
    fused = np.ascontiguousarray(unpadded, dtype=np.float32)
    norms = np.linalg.norm(fused, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-6, rtol=0.0):
        raise RuntimeError(
            "weighted semantic/CF fusion did not preserve unit row norm: "
            f"min={norms.min()} max={norms.max()}"
        )
    return fused, {
        "mode": "weighted_l2_normalized_semantic_cf_concat_v1",
        "semantic_dim": int(semantic_vectors.shape[1]),
        "cf_dim": int(cf_vectors.shape[1]),
        "unpadded_dim": int(semantic_vectors.shape[1] + cf_vectors.shape[1]),
        "padding_dim": int(padding_dim),
        "quantizer_dim": int(fused.shape[1]),
        "semantic_energy_fraction": float(1.0 - cf_energy_fraction),
        "cf_energy_fraction": float(cf_energy_fraction),
        "semantic_amplitude": semantic_weight,
        "cf_amplitude": cf_weight,
        "row_normalization": "l2_per_view_before_weighting",
    }


def position_usage(codes: np.ndarray, codebook_size: int) -> list[dict[str, Any]]:
    result = []
    for position in range(codes.shape[1]):
        counts = np.bincount(codes[:, position], minlength=codebook_size)
        nonzero = counts[counts > 0]
        probabilities = nonzero.astype(np.float64) / float(nonzero.sum())
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        result.append(
            {
                "position": position,
                "used_codes": int(len(nonzero)),
                "entropy_bits": entropy,
                "max_bucket_size": int(nonzero.max()),
                "min_nonempty_bucket_size": int(nonzero.min()),
            }
        )
    return result


def prefix_summary(codes: np.ndarray) -> list[dict[str, int]]:
    """Report trie growth at every native OPQ position without tuning order."""
    result = []
    for length in range(1, codes.shape[1] + 1):
        _, counts = np.unique(codes[:, :length], axis=0, return_counts=True)
        result.append(
            {
                "length": length,
                "unique_prefixes": int(len(counts)),
                "max_bucket_size": int(counts.max()),
            }
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path)
    parser.add_argument("--cf-embeddings", type=Path)
    parser.add_argument("--cf-skills-jsonl", type=Path)
    parser.add_argument("--cf-energy-fraction", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=ALLOWED_DEPTHS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--opq-iters", type=int, default=20)
    parser.add_argument("--opq-pq-iters", type=int, default=4)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.kmeans_iters < 1 or args.opq_iters < 1 or args.opq_pq_iters < 1:
        raise SystemExit("all OPQ iteration counts must be positive")
    if args.faiss_threads < 1:
        raise SystemExit("--faiss-threads must be positive")
    cf_args = (
        args.cf_embeddings,
        args.cf_skills_jsonl,
        args.cf_energy_fraction,
    )
    if any(value is not None for value in cf_args) and not all(
        value is not None for value in cf_args
    ):
        raise SystemExit(
            "--cf-embeddings, --cf-skills-jsonl, and --cf-energy-fraction "
            "must be provided together"
        )
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise SystemExit(
            f"--output-dir exists but is not a directory: {args.output_dir}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output directory: {args.output_dir}"
        )

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("faiss-cpu==1.13.2 is required") from exc

    faiss.omp_set_num_threads(args.faiss_threads)
    embeddings, embedded_skill_ids, embedding_meta = load_embedding_matrix(
        args.embeddings
    )
    if embedded_skill_ids is None:
        if args.skills_jsonl is None:
            raise SystemExit(
                "embedding payload has no skill_ids; provide --skills-jsonl"
            )
        skill_ids = load_skill_ids(args.skills_jsonl)
        alignment = "skills_jsonl_row_order"
    else:
        skill_ids = embedded_skill_ids
        alignment = "embedding_payload_skill_ids"
        if args.skills_jsonl is not None:
            catalog_skill_ids = load_skill_ids(args.skills_jsonl)
            if catalog_skill_ids != skill_ids:
                raise SystemExit(
                    "--skills-jsonl is not row-aligned with embedding skill_ids"
                )
            alignment = "embedding_payload_and_skills_jsonl_exact"
    if len(skill_ids) != len(embeddings) or len(set(skill_ids)) != len(skill_ids):
        raise SystemExit("embedding skill IDs must be unique and row-aligned")
    semantic_embeddings = embeddings
    cf_meta = None
    cf_skill_meta = None
    fusion = {"mode": "semantic_only", "quantizer_dim": int(embeddings.shape[1])}
    if args.cf_embeddings is not None:
        cf_embeddings, embedded_cf_ids, cf_meta = load_embedding_matrix(
            args.cf_embeddings
        )
        cf_catalog_ids = load_skill_ids(args.cf_skills_jsonl)
        if embedded_cf_ids is not None and embedded_cf_ids != cf_catalog_ids:
            raise SystemExit(
                "--cf-skills-jsonl is not row-aligned with CF embedding skill_ids"
            )
        cf_embeddings, cf_alignment = align_matrix(
            cf_embeddings,
            embedded_cf_ids or cf_catalog_ids,
            skill_ids,
            args.cf_embeddings,
        )
        embeddings, fusion = weighted_semantic_cf_fusion(
            semantic_embeddings,
            cf_embeddings,
            cf_energy_fraction=float(args.cf_energy_fraction),
            dimension_multiple=args.depth,
        )
        fusion["cf_alignment"] = cf_alignment
        cf_skill_meta = {
            "path": str(args.cf_skills_jsonl),
            "sha256": file_sha256(args.cf_skills_jsonl),
            "rows": len(cf_catalog_ids),
        }
    if embeddings.shape[1] % args.depth:
        raise SystemExit(
            f"quantizer dim {embeddings.shape[1]} is not divisible by OPQ depth {args.depth}"
        )
    if len(embeddings) < 256:
        raise SystemExit("8-bit OPQ requires at least 256 training vectors")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    fit = fit_product_quantizer(
        faiss,
        embeddings,
        depth=args.depth,
        nbits=8,
        seed=args.seed,
        kmeans_iters=args.kmeans_iters,
        opq_iters=args.opq_iters,
        opq_pq_iters=args.opq_pq_iters,
        optimized=True,
        verbose=args.verbose,
    )
    raw_codes = np.ascontiguousarray(fit.codes_native, dtype=np.int64)
    raw_collisions = collision_summary(raw_codes)
    if raw_collisions["collision_groups"]:
        codes, collision_resolution = minimum_cost_unique_last_codes(
            raw_codes,
            fit.level_costs(args.depth - 1),
        )
    else:
        codes = raw_codes.copy()
        collision_resolution = {
            "method": "none",
            "repaired_prefixes": 0,
            "repaired_items": 0,
            "changed_items": 0,
            "max_repaired_prefix_size": 0,
            "assignment_cost_delta": 0.0,
            "post_collision_groups": 0,
        }
    resolved_collisions = collision_summary(codes)
    if resolved_collisions["collision_groups"]:
        raise RuntimeError(
            f"collision resolution did not produce unique DocIDs: {resolved_collisions}"
        )

    reconstructed = fit.reconstruct_native(codes)
    logical_sha = semantic_ids_sha256(skill_ids, codes)
    np.save(args.output_dir / "codes.native.npy", raw_codes.astype(np.uint8))
    np.save(args.output_dir / "codes.resolved.npy", codes.astype(np.uint8))
    np.savez_compressed(args.output_dir / "quantizer_state.npz", **fit.state)
    write_semantic_ids(
        args.output_dir / "semantic_ids.jsonl",
        skill_ids,
        codes,
        raw_codes,
        "opq",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": "letter_opq_docids",
        "scope": (
            "replace_docid_construction_with_cf_aware_opq"
            if args.cf_embeddings is not None
            else "replace_docid_construction_only"
        ),
        "decoder_pipeline": "letter_sequential_autoregressive_trie",
        "quantizer": "faiss_opq_matrix_plus_product_quantizer",
        "semantic_id_mode": "learned_codes",
        "skills": len(skill_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "fusion": fusion,
        "depth": args.depth,
        "subvector_dim": int(embeddings.shape[1] // args.depth),
        "bits_per_code": 8,
        "bits_per_identifier": 8 * args.depth,
        "codebook_size": 256,
        "num_emb_list": [256] * args.depth,
        "position_order": list(range(args.depth)),
        "position_order_policy": "faiss_native_no_search",
        "collision_policy": "letter_minimum_cost_unique_last_code",
        "semantic_ids_sha256": logical_sha,
        "semantic_ids_file_sha256": file_sha256(args.output_dir / "semantic_ids.jsonl"),
        "inputs": {
            "embeddings": embedding_meta,
            "cf_embeddings": cf_meta,
            "cf_skills_jsonl": cf_skill_meta,
            "skills_jsonl": (
                {
                    "path": str(args.skills_jsonl),
                    "sha256": file_sha256(args.skills_jsonl),
                }
                if args.skills_jsonl is not None
                else None
            ),
            "alignment": alignment,
        },
        "training": {
            "seed": args.seed,
            "kmeans_iters": args.kmeans_iters,
            "opq_iters": args.opq_iters,
            "opq_pq_iters": args.opq_pq_iters,
            "faiss_threads": args.faiss_threads,
            "normalization": (
                "l2_per_view_before_weighted_concat"
                if args.cf_embeddings is not None
                else "none"
            ),
        },
        "integrity": {
            "fixed_length": args.depth,
            "minimum_code": int(codes.min()),
            "maximum_code": int(codes.max()),
            "raw_full_tuple": raw_collisions,
            "resolved_full_tuple": resolved_collisions,
            "collision_resolution": collision_resolution,
        },
        "reconstruction": reconstruction_metrics(embeddings, reconstructed),
        "position_usage": position_usage(codes, 256),
        "prefixes": prefix_summary(codes),
        "artifacts": {
            "semantic_ids": "semantic_ids.jsonl",
            "codes_native": "codes.native.npy",
            "codes_resolved": "codes.resolved.npy",
            "quantizer_state": "quantizer_state.npz",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
        },
        "wall_seconds": time.time() - started,
    }
    write_json(args.output_dir / "opq_docid_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "depth": args.depth,
                "skills": len(skill_ids),
                "semantic_ids_sha256": logical_sha,
                "raw_collision_groups": raw_collisions["collision_groups"],
                "changed_items": collision_resolution["changed_items"],
                "resolved_collision_groups": resolved_collisions["collision_groups"],
                "wall_seconds": manifest["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
