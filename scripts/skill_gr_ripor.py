#!/usr/bin/env python3
"""Faithful RIPOR training for the fixed SkillGR cat3/query5 experiment.

The implementation follows arXiv:2311.09134 and the audited official code at
commit 03fd0cbfafdeeb9cdd866e0c44727fec39a96a52.  It implements the complete
M0 -> residual-quantized SID -> M1 -> M2 -> M3 -> M4 chain.  Earlier project
checkpoints that retained LETTER identifiers are intentionally incompatible.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skill_gr_dgr import (  # noqa: E402
    RIPOR_AUDITED_CODE_COMMIT,
    RIPOR_CODE_URL,
    RIPOR_PAPER_URL,
    build_trie_decode_tables,
    dgr_rows_fingerprint,
    file_sha256,
    topk_single_sid_scores_from_encoder,
)
from skill_gr_pipeline import (  # noqa: E402
    GR_SOURCE_TEXT_POLICY,
    SEMANTIC_ID_MODE_LEARNED,
    compute_at_k_metrics,
    decoder_source_text,
    global_sid_token_id,
    load_jsonl,
    make_structured_t5_sid_model,
    save_structured_decoder_checkpoint,
    skill_id,
    structured_sid_bos_id,
    write_json,
    write_jsonl,
)


OFFICIAL_DECODER_START_SHA256 = (
    "fe928c4929126abb23e1b22532d21aec65c9a723f2e5fb1f035cfea7da211f02"
)
RIPOR_FORMULATION = "ripor_arxiv2311.09134_official_commit_03fd0cb_faithful"
RIPOR_TEACHER_POLICY = "raw_l2_normalized_query_skill_dot_product"
RIPOR_SCORE_MODE = "conditional_logit"
RIPOR_RANK_QUERY_TEXT_POLICY = "query_prefix_plus_skillrouter_query_1500_chars_v1"
DEFAULT_PREFIX_BETA = 2.0
QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _skill_ids_sha256(skill_ids: Sequence[str]) -> str:
    return _sha256_text("\n".join(skill_ids))


def _validate_ripor_semantic_ids(semantic_ids: dict[str, list[int]]) -> None:
    """Validate code-only RIPOR SIDs while allowing official full-SID collisions."""
    if not semantic_ids:
        raise ValueError("RIPOR semantic IDs must not be empty")
    lengths = {len(values) for values in semantic_ids.values()}
    if len(lengths) != 1:
        raise ValueError(f"RIPOR semantic IDs have inconsistent lengths: {sorted(lengths)}")
    if next(iter(lengths)) < 1:
        raise ValueError("RIPOR semantic IDs must contain at least one code")
    negative = [sid for sid, values in semantic_ids.items() if any(value < 0 for value in values)]
    if negative:
        raise ValueError(f"RIPOR semantic IDs contain a negative code: {negative[0]}")


def _ripor_semantic_ids_sha256(semantic_ids: dict[str, list[int]]) -> str:
    _validate_ripor_semantic_ids(semantic_ids)
    canonical = json.dumps(
        sorted((sid, [int(value) for value in values]) for sid, values in semantic_ids.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_ripor_semantic_ids(path: Path) -> tuple[dict[str, list[int]], str]:
    semantic_ids: dict[str, list[int]] = {}
    modes: set[str] = set()
    for row_index, row in enumerate(load_jsonl(path)):
        sid = str(row.get("skill_id") or "")
        if not sid:
            raise ValueError(f"RIPOR semantic-ID row {row_index} has no skill_id: {path}")
        if sid in semantic_ids:
            raise ValueError(f"duplicate skill_id in RIPOR semantic IDs: {sid}")
        semantic_ids[sid] = [int(value) for value in row.get("semantic_id", [])]
        modes.add(str(row.get("semantic_id_mode") or ""))
    if modes != {SEMANTIC_ID_MODE_LEARNED}:
        raise ValueError(
            f"RIPOR semantic IDs require mode={SEMANTIC_ID_MODE_LEARNED}, "
            f"got {sorted(modes)}: {path}"
        )
    _validate_ripor_semantic_ids(semantic_ids)
    return semantic_ids, SEMANTIC_ID_MODE_LEARNED


def _require_file_sha256(path: Path, expected: str | None, label: str) -> str:
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected={expected}, actual={actual}, path={path}")
    return actual


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(cpu: bool = False) -> torch.device:
    if torch.cuda.is_available() and not cpu:
        return torch.device("cuda")
    return torch.device("cpu")


def _autocast(device: torch.device, bf16: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bool(bf16 and device.type == "cuda"),
    )


def _load_skills(path: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows = load_jsonl(path)
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        sid = skill_id(row)
        if not sid:
            raise ValueError(f"skill row {row_index} has no skill_id: {path}")
        if sid in seen:
            raise ValueError(f"duplicate skill_id in catalog: {sid}")
        seen.add(sid)
        ids.append(sid)
        name = str(row.get("skill_name") or row.get("name") or sid)
        description = str(row.get("description") or row.get("task_description") or "")[:300]
        body = str(row.get("body") or "")[:2500]
        texts.append(str(row.get("skillrouter_text") or f"{name} | {description} | {body}"))
    if not rows:
        raise ValueError(f"empty skill catalog: {path}")
    return rows, ids, texts


def _l2_normalize_np_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("row normalization requires a finite rank-2 array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("row normalization received a zero-norm row")
    return (values / norms).astype(np.float32, copy=False)


def cf_procrustes_assignment_embeddings(
    m0_embeddings: np.ndarray,
    cf_embeddings: np.ndarray,
    *,
    cf_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Map normalized CF geometry into M0 space and fuse it for RQ assignment."""

    m0 = np.asarray(m0_embeddings, dtype=np.float32)
    cf = np.asarray(cf_embeddings, dtype=np.float32)
    if m0.ndim != 2 or cf.ndim != 2 or m0.shape[0] != cf.shape[0]:
        raise ValueError(f"M0/CF embedding shapes do not align: {m0.shape}/{cf.shape}")
    if not np.isfinite(m0).all() or not np.isfinite(cf).all():
        raise ValueError("M0/CF embeddings contain non-finite values")
    if not (0.0 <= cf_weight <= 1.0):
        raise ValueError(f"cf_weight must be in [0, 1], got {cf_weight}")
    if cf.shape[1] > m0.shape[1]:
        raise ValueError("semi-orthogonal CF projection requires cf_dim <= m0_dim")

    m0_norms = np.linalg.norm(m0, axis=1, keepdims=True)
    m0_direction = _l2_normalize_np_rows(m0)
    cf_direction = _l2_normalize_np_rows(cf)
    left, singular_values, right_t = np.linalg.svd(
        cf_direction.T @ m0_direction, full_matrices=False
    )
    projection = (left @ right_t).astype(np.float32, copy=False)
    projected_cf = cf_direction @ projection
    fused_direction = _l2_normalize_np_rows(
        (1.0 - float(cf_weight)) * m0_direction
        + float(cf_weight) * projected_cf
    )
    fused = (fused_direction * m0_norms).astype(np.float32, copy=False)
    orthogonality_error = np.max(
        np.abs(projection @ projection.T - np.eye(cf.shape[1], dtype=np.float32))
    )
    audit = {
        "cf_weight": float(cf_weight),
        "m0_dim": float(m0.shape[1]),
        "cf_dim": float(cf.shape[1]),
        "procrustes_singular_sum": float(singular_values.sum()),
        "projection_row_orthogonality_max_abs_error": float(orthogonality_error),
        "fused_norm_ratio_mean": float(
            np.mean(np.linalg.norm(fused, axis=1) / m0_norms[:, 0])
        ),
    }
    return fused, projection, audit


def m0_qwen_cf_procrustes_assignment_embeddings(
    m0_embeddings: np.ndarray,
    qwen_embeddings: np.ndarray,
    cf_embeddings: np.ndarray,
    *,
    m0_weight: float,
    qwen_weight: float,
    cf_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse M0, base-Qwen semantics, and CF geometry for fixed RQ assignment."""

    m0 = np.asarray(m0_embeddings, dtype=np.float32)
    qwen = np.asarray(qwen_embeddings, dtype=np.float32)
    cf = np.asarray(cf_embeddings, dtype=np.float32)
    if (
        m0.ndim != 2
        or qwen.ndim != 2
        or cf.ndim != 2
        or not (m0.shape[0] == qwen.shape[0] == cf.shape[0])
    ):
        raise ValueError(
            f"M0/Qwen/CF embedding shapes do not align: {m0.shape}/{qwen.shape}/{cf.shape}"
        )
    if not np.isfinite(m0).all() or not np.isfinite(qwen).all() or not np.isfinite(cf).all():
        raise ValueError("M0/Qwen/CF embeddings contain non-finite values")
    weights = np.asarray([m0_weight, qwen_weight, cf_weight], dtype=np.float64)
    if np.any(weights < 0.0) or not math.isclose(
        float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "M0/Qwen/CF assignment weights must be non-negative and sum to one, "
            f"got {weights.tolist()}"
        )

    m0_norms = np.linalg.norm(m0, axis=1, keepdims=True)
    m0_direction = _l2_normalize_np_rows(m0)
    qwen_direction = _l2_normalize_np_rows(qwen)
    cf_direction = _l2_normalize_np_rows(cf)

    def project_to_m0(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
        left, singular_values, right_t = np.linalg.svd(
            source.T @ m0_direction, full_matrices=False
        )
        projection = (left @ right_t).astype(np.float32, copy=False)
        projected = source @ projection
        if source.shape[1] <= m0.shape[1]:
            error = np.max(
                np.abs(
                    projection @ projection.T
                    - np.eye(source.shape[1], dtype=np.float32)
                )
            )
            orthogonality = "row"
        else:
            error = np.max(
                np.abs(
                    projection.T @ projection
                    - np.eye(m0.shape[1], dtype=np.float32)
                )
            )
            orthogonality = "column"
        return projected, projection, singular_values, orthogonality, float(error)

    projected_qwen, qwen_projection, qwen_singular, qwen_orthogonality, qwen_error = (
        project_to_m0(qwen_direction)
    )
    projected_cf, cf_projection, cf_singular, cf_orthogonality, cf_error = (
        project_to_m0(cf_direction)
    )
    fused_direction = _l2_normalize_np_rows(
        float(m0_weight) * m0_direction
        + float(qwen_weight) * projected_qwen
        + float(cf_weight) * projected_cf
    )
    fused = (fused_direction * m0_norms).astype(np.float32, copy=False)
    audit = {
        "m0_weight": float(m0_weight),
        "qwen_weight": float(qwen_weight),
        "cf_weight": float(cf_weight),
        "m0_dim": float(m0.shape[1]),
        "qwen_dim": float(qwen.shape[1]),
        "cf_dim": float(cf.shape[1]),
        "qwen_procrustes_singular_sum": float(qwen_singular.sum()),
        "cf_procrustes_singular_sum": float(cf_singular.sum()),
        "qwen_projection_orthogonality": qwen_orthogonality,
        "cf_projection_orthogonality": cf_orthogonality,
        "qwen_projection_orthogonality_max_abs_error": qwen_error,
        "cf_projection_orthogonality_max_abs_error": cf_error,
        "fused_norm_ratio_mean": float(
            np.mean(np.linalg.norm(fused, axis=1) / m0_norms[:, 0])
        ),
    }
    return fused, qwen_projection, cf_projection, audit


def refit_residual_codebooks(
    embeddings: np.ndarray,
    codes: np.ndarray,
    *,
    codebook_size: int,
) -> tuple[np.ndarray, list[int]]:
    """Fit M0-space residual centroids for a fixed CF-aware code assignment."""

    values = np.asarray(embeddings, dtype=np.float32)
    codes = np.asarray(codes, dtype=np.int64)
    if values.ndim != 2 or codes.ndim != 2 or values.shape[0] != codes.shape[0]:
        raise ValueError(f"embedding/code shapes do not align: {values.shape}/{codes.shape}")
    if codebook_size < 2 or codes.min() < 0 or codes.max() >= codebook_size:
        raise ValueError("fixed residual codes are outside the requested codebook")
    residual = values.copy()
    codebooks = np.zeros(
        (codes.shape[1], codebook_size, values.shape[1]), dtype=np.float32
    )
    unused_per_level: list[int] = []
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:, level], minlength=codebook_size)
        sums = np.zeros((codebook_size, values.shape[1]), dtype=np.float64)
        np.add.at(sums, codes[:, level], residual.astype(np.float64, copy=False))
        used = counts > 0
        codebooks[level, used] = (sums[used] / counts[used, None]).astype(np.float32)
        residual -= codebooks[level, codes[:, level]]
        unused_per_level.append(int((~used).sum()))
    if not np.isfinite(codebooks).all():
        raise ValueError("refitted residual codebooks contain non-finite values")
    return codebooks, unused_per_level


def _load_cf_skill_ids(path: Path) -> list[str]:
    rows = load_jsonl(path)
    indexed = sorted(
        ((int(row["index"]), str(row["skill_id"])) for row in rows),
        key=lambda item: item[0],
    )
    if [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError("CF skill IDs do not contain a contiguous catalog index")
    return [sid for _, sid in indexed]


def _validate_rows(rows: list[dict[str, Any]], skill_to_index: dict[str, int]) -> None:
    if not rows:
        raise ValueError("training rows are empty")
    for row_position, row in enumerate(rows):
        gold = [str(value) for value in row.get("selected_skill_ids", [])]
        if not gold:
            raise ValueError(f"row {row_position} has no selected_skill_ids")
        if len(gold) != len(set(gold)):
            raise ValueError(f"row {row_position} contains duplicate GT skills")
        missing = [sid for sid in gold if sid not in skill_to_index]
        if missing:
            raise KeyError(f"row {row_position} references unknown skills: {missing[:3]}")


def ripor_prefix_lengths(depth: int) -> list[int]:
    """Return the representative RIPOR prefix schedule ending at ``depth``."""

    if depth < 4 or depth & (depth - 1):
        raise ValueError(f"RIPOR depth must be a power of two and at least 4, got {depth}")
    values: list[int] = []
    current = 4
    while current <= depth:
        values.append(current)
        current *= 2
    if values[-1] != depth:
        raise AssertionError("prefix schedule does not terminate at full depth")
    return values


def ripor_prefix_alphas(depth: int, beta: float = DEFAULT_PREFIX_BETA) -> dict[int, float]:
    """Paper Eq. (prefix margin): alpha_i=(1-beta/i)/(1-beta/L)."""

    lengths = ripor_prefix_lengths(depth)
    denominator = 1.0 - float(beta) / depth
    if denominator <= 0.0:
        raise ValueError(f"beta must be smaller than depth, got beta={beta}, depth={depth}")
    alphas = {length: (1.0 - float(beta) / length) / denominator for length in lengths}
    if any(not (0.0 < value <= 1.0) for value in alphas.values()):
        raise ValueError(f"invalid RIPOR alpha schedule: {alphas}")
    if any(alphas[a] >= alphas[b] for a, b in zip(lengths, lengths[1:])):
        raise ValueError(f"RIPOR alpha schedule must be strictly increasing: {alphas}")
    if not math.isclose(alphas[depth], 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("full-length RIPOR alpha must equal one")
    return alphas


def resolve_active_prefix_lengths(depth: int, active_max_prefix: int) -> list[int]:
    lengths = ripor_prefix_lengths(depth)
    if active_max_prefix not in lengths:
        raise ValueError(
            f"active max prefix must be one of the RIPOR stages {lengths}, got {active_max_prefix}"
        )
    return [length for length in lengths if length <= active_max_prefix]


def ripor_margin_mse(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    teacher_positive_scores: torch.Tensor,
    teacher_negative_scores: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Exact signed MarginMSE used by RIPOR; no clipping or positive filtering."""

    shapes = {
        tuple(positive_scores.shape),
        tuple(negative_scores.shape),
        tuple(teacher_positive_scores.shape),
        tuple(teacher_negative_scores.shape),
    }
    if len(shapes) != 1:
        raise ValueError(f"MarginMSE inputs must have equal shapes, got {sorted(shapes)}")
    student_margin = positive_scores.float() - negative_scores.float()
    teacher_margin = float(alpha) * (
        teacher_positive_scores.float() - teacher_negative_scores.float()
    )
    return F.mse_loss(student_margin, teacher_margin)


def query_balanced_ripor_margin_mse(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    teacher_positive_scores: torch.Tensor,
    teacher_negative_scores: torch.Tensor,
    pair_weights: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Sum weighted pair losses; weights encode query-then-positive averaging."""

    shapes = {
        tuple(positive_scores.shape),
        tuple(negative_scores.shape),
        tuple(teacher_positive_scores.shape),
        tuple(teacher_negative_scores.shape),
        tuple(pair_weights.shape),
    }
    if len(shapes) != 1:
        raise ValueError(
            f"query-balanced MarginMSE inputs must have equal shapes, got {sorted(shapes)}"
        )
    if not torch.isfinite(pair_weights).all() or bool((pair_weights < 0).any()):
        raise ValueError("query-balanced MarginMSE weights must be finite and non-negative")
    student_margin = positive_scores.float() - negative_scores.float()
    teacher_margin = float(alpha) * (
        teacher_positive_scores.float() - teacher_negative_scores.float()
    )
    return torch.sum(pair_weights.float() * (student_margin - teacher_margin).square())


def ripor_progressive_loss(
    positive_prefix_scores: torch.Tensor,
    negative_prefix_scores: torch.Tensor,
    teacher_positive_scores: torch.Tensor,
    teacher_negative_scores: torch.Tensor,
    *,
    prefix_lengths: Sequence[int],
    active_prefix_lengths: Sequence[int],
    alphas: dict[int, float],
    pair_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Sum the current and all earlier prefix objectives, as in RIPOR M3/M4."""

    if positive_prefix_scores.shape != negative_prefix_scores.shape:
        raise ValueError("positive and negative prefix score tensors must have equal shapes")
    if positive_prefix_scores.ndim != 2 or positive_prefix_scores.shape[1] != len(prefix_lengths):
        raise ValueError(
            "prefix score tensors must have shape [batch, len(prefix_lengths)]"
        )
    position = {length: index for index, length in enumerate(prefix_lengths)}
    active = [int(length) for length in active_prefix_lengths]
    if not active or any(length not in position for length in active):
        raise ValueError(f"active prefixes {active} are not a subset of {list(prefix_lengths)}")
    if active != sorted(active) or active != list(prefix_lengths[: len(active)]):
        raise ValueError("progressive objectives must form the shortest-prefix-first prefix set")
    losses: dict[int, torch.Tensor] = {}
    for length in active:
        column = position[length]
        if pair_weights is None:
            losses[length] = ripor_margin_mse(
                positive_prefix_scores[:, column],
                negative_prefix_scores[:, column],
                teacher_positive_scores,
                teacher_negative_scores,
                alpha=alphas[length],
            )
        else:
            losses[length] = query_balanced_ripor_margin_mse(
                positive_prefix_scores[:, column],
                negative_prefix_scores[:, column],
                teacher_positive_scores,
                teacher_negative_scores,
                pair_weights,
                alpha=alphas[length],
            )
    return torch.stack(list(losses.values())).sum(), losses


def load_official_decoder_start(path: Path, expected_d_model: int | None = None) -> torch.Tensor:
    """Load the exact official T5 decoder-start embedding from .npy or tracked .b64."""

    if path.suffix == ".b64":
        encoded = "".join(path.read_text(encoding="ascii").split())
        raw = base64.b64decode(encoded, validate=True)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    else:
        actual_sha256 = file_sha256(path)
        array = np.load(path, allow_pickle=False)
    if actual_sha256 != OFFICIAL_DECODER_START_SHA256:
        raise ValueError(
            "decoder-start embedding is not the audited RIPOR T5-base asset: "
            f"expected={OFFICIAL_DECODER_START_SHA256}, actual={actual_sha256}"
        )
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    if tensor.ndim == 3 and tensor.shape[:2] == (1, 1):
        tensor = tensor[0, 0]
    if tensor.ndim != 1:
        raise ValueError(f"decoder-start embedding must reduce to [d_model], got {tuple(tensor.shape)}")
    if expected_d_model is not None and tensor.numel() != expected_d_model:
        raise ValueError(
            f"decoder-start dimension {tensor.numel()} does not match d_model={expected_d_model}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("decoder-start embedding contains non-finite values")
    return tensor.contiguous()


class RIPORT5DenseEncoder(nn.Module):
    """T5 encoder-decoder used as RIPOR M0's shared query/document dense encoder."""

    def __init__(self, base_model: str, decoder_start: torch.Tensor) -> None:
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM

        self.base_model_name = str(base_model)
        self.backbone = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        d_model = int(self.backbone.config.d_model)
        if tuple(decoder_start.shape) != (d_model,):
            raise ValueError(
                f"decoder-start embedding must have shape {(d_model,)}, got {tuple(decoder_start.shape)}"
            )
        self.start_token_embed = nn.Parameter(decoder_start.float().view(1, 1, d_model).clone())

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.backbone.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        batch_size = input_ids.shape[0]
        decoder_inputs = self.start_token_embed.expand(batch_size, 1, -1)
        decoder_outputs = self.backbone.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=torch.ones(
                (batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device
            ),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return decoder_outputs.last_hidden_state[:, 0, :]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)


def _freeze_ripor_output_biases(model) -> None:
    """Official RIPOR scores with embedding dot products and has no output bias."""

    for head in model.code_heads:
        bias = getattr(head, "bias", None)
        if bias is not None:
            with torch.no_grad():
                bias.zero_()
            bias.requires_grad_(False)


def initialize_ripor_structured_model(
    dense_model: RIPORT5DenseEncoder,
    *,
    base_model: str,
    codebooks: np.ndarray,
):
    """Create M1 from M0 plus raw RQ input/output embeddings without rescaling."""

    codebooks = np.asarray(codebooks, dtype=np.float32)
    if codebooks.ndim != 3:
        raise ValueError(f"RQ codebooks must have shape [depth, vocab, d_model], got {codebooks.shape}")
    depth, codebook_size, d_model = (int(value) for value in codebooks.shape)
    if d_model != int(dense_model.backbone.config.d_model):
        raise ValueError("RQ codebook dimension does not match the M0 hidden dimension")
    model = make_structured_t5_sid_model(
        base_model,
        codebook_size,
        depth,
        0,
        SEMANTIC_ID_MODE_LEARNED,
        codebook_embeddings=None,
        tie_code_input_output=False,
    )
    model.backbone.load_state_dict(dense_model.backbone.state_dict(), strict=True)
    bos_id = structured_sid_bos_id(
        codebook_size, depth, 0, SEMANTIC_ID_MODE_LEARNED
    )
    codebook_tensor = torch.from_numpy(codebooks).to(model.sid_embedding.weight.dtype)
    with torch.no_grad():
        for level in range(depth):
            start = level * codebook_size
            end = start + codebook_size
            model.sid_embedding.weight[start:end].copy_(codebook_tensor[level])
            model.code_heads[level].weight.copy_(codebook_tensor[level])
            model.code_heads[level].bias.zero_()
        model.sid_embedding.weight[bos_id].copy_(
            dense_model.start_token_embed.detach()[0, 0].to(model.sid_embedding.weight.dtype)
        )
    _freeze_ripor_output_biases(model)
    model.sid_codebook_initialized = True
    model.sid_codebook_init_scale = 1.0
    return model


def ripor_code_only_batch(
    rows: list[dict[str, Any]],
    tokenizer,
    semantic_ids: dict[str, list[int]],
    *,
    codebook_size: int,
    depth: int,
    max_source_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build M1 targets containing exactly L RQ codes and no STOP/control token."""

    texts: list[str] = []
    codes: list[list[int]] = []
    for row in rows:
        positives = [str(value) for value in row.get("selected_skill_ids", [])]
        if len(positives) != 1:
            raise ValueError("M1 batch rows must be expanded to exactly one positive skill")
        sid = positives[0]
        values = [int(value) for value in semantic_ids[sid]]
        if len(values) != depth:
            raise ValueError(f"semantic ID for {sid} has length {len(values)}, expected {depth}")
        if any(value < 0 or value >= codebook_size for value in values):
            raise ValueError(f"semantic ID for {sid} contains an out-of-range code")
        texts.append(decoder_source_text(row))
        codes.append(values)
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_source_length,
        return_tensors="pt",
    )
    code_tensor = torch.tensor(codes, dtype=torch.long, device=device)
    bos_id = structured_sid_bos_id(codebook_size, depth, 0, SEMANTIC_ID_MODE_LEARNED)
    decoder_input_ids = torch.full(
        (len(rows), depth), bos_id, dtype=torch.long, device=device
    )
    for previous_level in range(depth - 1):
        decoder_input_ids[:, previous_level + 1] = code_tensor[:, previous_level] + global_sid_token_id(
            previous_level,
            0,
            codebook_size,
            depth,
            0,
            SEMANTIC_ID_MODE_LEARNED,
        )
    target_types = torch.arange(depth, dtype=torch.long, device=device).unsqueeze(0).expand(
        len(rows), -1
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": torch.ones_like(decoder_input_ids),
        "target_types": target_types,
        "labels": code_tensor,
    }


def sequence_conditional_logit_prefix_scores(
    model,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    codes: torch.Tensor,
    *,
    codebook_size: int,
    depth: int,
    score_length: int | None = None,
) -> torch.Tensor:
    """Score one SID per encoder row with one causal-decoder forward pass."""

    if codes.ndim != 2 or tuple(codes.shape) != (encoder_hidden_states.shape[0], depth):
        raise ValueError(
            f"codes must have shape {(encoder_hidden_states.shape[0], depth)}, got {tuple(codes.shape)}"
        )
    resolved_length = depth if score_length is None else int(score_length)
    if not 1 <= resolved_length <= depth:
        raise ValueError(f"score_length must be in [1, {depth}], got {resolved_length}")
    bos_id = structured_sid_bos_id(codebook_size, depth, 0, SEMANTIC_ID_MODE_LEARNED)
    decoder_input_ids = torch.full(
        (codes.shape[0], resolved_length), bos_id, dtype=torch.long, device=codes.device
    )
    for previous_level in range(resolved_length - 1):
        decoder_input_ids[:, previous_level + 1] = codes[:, previous_level] + global_sid_token_id(
            previous_level,
            0,
            codebook_size,
            depth,
            0,
            SEMANTIC_ID_MODE_LEARNED,
        )
    hidden = model._decoder_hidden_from_encoder(
        encoder_hidden_states,
        encoder_attention_mask,
        decoder_input_ids,
        torch.ones_like(decoder_input_ids),
    )
    token_scores = []
    for level, head in enumerate(model.code_heads[:resolved_length]):
        logits = head(hidden[:, level, :]).float()
        token_scores.append(logits.gather(1, codes[:, level : level + 1]).squeeze(1))
    return torch.cumsum(torch.stack(token_scores, dim=1), dim=1)


def _expand_query_positive_pairs(
    rows: list[dict[str, Any]],
) -> tuple[list[int], list[str]]:
    row_indices: list[int] = []
    positive_ids: list[str] = []
    for row_index, row in enumerate(rows):
        for sid in sorted({str(value) for value in row.get("selected_skill_ids", [])}):
            row_indices.append(row_index)
            positive_ids.append(sid)
    if not row_indices:
        raise ValueError("query/positive expansion produced no examples")
    return row_indices, positive_ids


def _load_teacher_query_embeddings(
    embeddings_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    manifest_path = embeddings_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"teacher query embedding manifest is incomplete: {manifest_path}")
    expected_fingerprint = dgr_rows_fingerprint(rows)
    if manifest.get("rows_fingerprint_sha256") != expected_fingerprint:
        raise ValueError("teacher query embeddings do not match the supplied source rows")
    expected_shas = manifest.get("shard_sha256", {})
    chunks: list[torch.Tensor] = []
    expected_start = 0
    for shard_name in manifest.get("shards", []):
        shard_path = embeddings_dir / "shards" / shard_name
        if shard_name in expected_shas:
            _require_file_sha256(shard_path, expected_shas[shard_name], "teacher query shard")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if int(shard["start"]) != expected_start:
            raise ValueError(f"teacher query shard is non-contiguous: {shard_name}")
        embeddings = shard["embeddings"].float()
        if embeddings.ndim != 2 or embeddings.shape[0] != int(shard["end"]) - expected_start:
            raise ValueError(f"invalid teacher query embedding shard: {shard_name}")
        chunks.append(embeddings)
        expected_start = int(shard["end"])
    if expected_start != len(rows):
        raise ValueError(
            f"teacher query embeddings cover {expected_start} rows, expected {len(rows)}"
        )
    embeddings = torch.cat(chunks, dim=0)
    norms = embeddings.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=0.0):
        raise ValueError("teacher query embeddings are not L2 normalized")
    return embeddings, manifest


def _load_teacher_skill_embeddings(
    path: Path,
    manifest_path: Path,
    skills_path: Path,
    skill_ids: Sequence[str],
) -> tuple[torch.Tensor, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"teacher skill embedding manifest is incomplete: {manifest_path}")
    _require_file_sha256(path, manifest.get("output_sha256"), "teacher skill embeddings")
    if manifest.get("skills_jsonl_sha256") != file_sha256(skills_path):
        raise ValueError("teacher skill embeddings were created from a different catalog file")
    if manifest.get("skill_ids_sha256") != _skill_ids_sha256(skill_ids):
        raise ValueError("teacher skill embedding order does not match the skill catalog")
    embeddings = torch.from_numpy(np.load(path, allow_pickle=False)).float()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(skill_ids):
        raise ValueError(
            f"teacher skill embeddings have shape {tuple(embeddings.shape)}, expected rows={len(skill_ids)}"
        )
    norms = embeddings.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=0.0):
        raise ValueError("teacher skill embeddings are not L2 normalized")
    return embeddings, manifest


def _candidate_row_with_gold(
    retrieved: Iterable[int],
    gold_indices: Sequence[int],
    *,
    top_k: int,
) -> list[int]:
    retrieved_list = [int(value) for value in retrieved][:top_k]
    if len(retrieved_list) != top_k:
        raise ValueError(f"retriever returned {len(retrieved_list)} candidates, expected {top_k}")
    if len(retrieved_list) != len(set(retrieved_list)):
        raise ValueError("retriever returned duplicate candidate indices")
    result: list[int] = []
    seen: set[int] = set()
    for value in [*gold_indices, *retrieved_list]:
        value = int(value)
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _candidate_row_with_gold_after_sid_expansion(
    retrieved: Iterable[int],
    gold_indices: Sequence[int],
    *,
    retrieved_sid_count: int,
) -> list[int]:
    """Prepend GTs after official SID-to-document expansion.

    RIPOR decodes ``retrieved_sid_count`` unique SIDs, then expands every SID to
    every catalog item assigned to it.  The expanded skill count can therefore
    exceed the beam count and must not be truncated back to that count.
    """
    retrieved_list = [int(value) for value in retrieved]
    if len(retrieved_list) < retrieved_sid_count:
        raise ValueError(
            f"SID expansion returned {len(retrieved_list)} skills from "
            f"{retrieved_sid_count} retrieved SIDs"
        )
    if len(retrieved_list) != len(set(retrieved_list)):
        raise ValueError("SID expansion returned duplicate skill indices")
    result: list[int] = []
    seen: set[int] = set()
    for value in [*gold_indices, *retrieved_list]:
        value = int(value)
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _write_candidate_artifact(
    output_dir: Path,
    rows: list[dict[str, Any]],
    skill_ids: Sequence[str],
    candidate_rows: Sequence[Sequence[int]],
    *,
    manifest_extra: dict[str, Any],
    teacher_scores: Sequence[Sequence[float]] | None = None,
    shard_size: int = 4096,
) -> None:
    if len(candidate_rows) != len(rows):
        raise ValueError("candidate rows do not align with source rows")
    if teacher_scores is not None and len(teacher_scores) != len(rows):
        raise ValueError("teacher scores do not align with source rows")
    max_candidates = max(len(values) for values in candidate_rows)
    if max_candidates < 2:
        raise ValueError("candidate artifact must contain at least two candidates per row")
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_names: list[str] = []
    shard_shas: dict[str, str] = {}
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        end = min(start + shard_size, len(rows))
        width = max(len(candidate_rows[index]) for index in range(start, end))
        indices = torch.full((end - start, width), -1, dtype=torch.int32)
        counts = torch.empty(end - start, dtype=torch.int16)
        score_tensor = (
            torch.full((end - start, width), float("nan"), dtype=torch.float32)
            if teacher_scores is not None
            else None
        )
        for local_index, source_index in enumerate(range(start, end)):
            values = [int(value) for value in candidate_rows[source_index]]
            if len(values) != len(set(values)):
                raise ValueError(f"candidate row {source_index} contains duplicates")
            if any(value < 0 or value >= len(skill_ids) for value in values):
                raise ValueError(f"candidate row {source_index} has an out-of-range skill index")
            indices[local_index, : len(values)] = torch.tensor(values, dtype=torch.int32)
            counts[local_index] = len(values)
            if score_tensor is not None:
                scores = [float(value) for value in teacher_scores[source_index]]
                if len(scores) != len(values) or not all(math.isfinite(value) for value in scores):
                    raise ValueError(f"invalid teacher scores for candidate row {source_index}")
                score_tensor[local_index, : len(scores)] = torch.tensor(scores, dtype=torch.float32)
        shard_name = f"part-{shard_index:06d}.pt"
        payload: dict[str, Any] = {
            "start": start,
            "end": end,
            "candidate_indices": indices,
            "candidate_counts": counts,
        }
        if score_tensor is not None:
            payload["teacher_scores"] = score_tensor
        shard_path = shard_dir / shard_name
        _atomic_torch_save(payload, shard_path)
        shard_names.append(shard_name)
        shard_shas[shard_name] = file_sha256(shard_path)
    write_json(
        output_dir / "manifest.json",
        {
            "status": "complete",
            "format": "ripor_candidate_artifact_v1",
            "paper": RIPOR_PAPER_URL,
            "official_code": RIPOR_CODE_URL,
            "official_code_commit": RIPOR_AUDITED_CODE_COMMIT,
            "rows": len(rows),
            "rows_fingerprint_sha256": dgr_rows_fingerprint(rows),
            "skills": len(skill_ids),
            "skill_ids_sha256": _skill_ids_sha256(skill_ids),
            "max_candidates": max_candidates,
            "teacher_scores": teacher_scores is not None,
            "shards": shard_names,
            "shard_sha256": shard_shas,
            **manifest_extra,
        },
    )


def _load_candidate_artifact(
    artifact_dir: Path,
    rows: list[dict[str, Any]],
    skill_ids: Sequence[str],
    *,
    require_teacher_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("format") != "ripor_candidate_artifact_v1":
        raise ValueError(f"invalid or incomplete RIPOR candidate manifest: {manifest_path}")
    if manifest.get("rows_fingerprint_sha256") != dgr_rows_fingerprint(rows):
        raise ValueError("candidate artifact source rows do not match the supplied rows")
    if manifest.get("skill_ids_sha256") != _skill_ids_sha256(skill_ids):
        raise ValueError("candidate artifact catalog order does not match the supplied skills")
    if bool(manifest.get("teacher_scores")) != bool(require_teacher_scores):
        raise ValueError(
            f"candidate artifact teacher-score requirement mismatch: required={require_teacher_scores}"
        )
    max_width = int(manifest["max_candidates"])
    all_indices: list[torch.Tensor] = []
    all_counts: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    expected_start = 0
    expected_shas = manifest.get("shard_sha256", {})
    for shard_name in manifest.get("shards", []):
        shard_path = artifact_dir / "shards" / shard_name
        _require_file_sha256(shard_path, expected_shas.get(shard_name), "candidate shard")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if int(shard["start"]) != expected_start:
            raise ValueError(f"candidate shard is non-contiguous: {shard_name}")
        indices = shard["candidate_indices"].to(torch.int64)
        counts = shard["candidate_counts"].to(torch.int64)
        if indices.ndim != 2 or counts.shape != (indices.shape[0],):
            raise ValueError(f"invalid candidate shard shapes: {shard_name}")
        if indices.shape[1] < max_width:
            indices = F.pad(indices, (0, max_width - indices.shape[1]), value=-1)
        all_indices.append(indices)
        all_counts.append(counts)
        if require_teacher_scores:
            scores = shard.get("teacher_scores")
            if scores is None or scores.shape != shard["candidate_indices"].shape:
                raise ValueError(f"candidate shard lacks aligned teacher scores: {shard_name}")
            scores = scores.float()
            if scores.shape[1] < max_width:
                scores = F.pad(scores, (0, max_width - scores.shape[1]), value=float("nan"))
            all_scores.append(scores)
        expected_start = int(shard["end"])
    if expected_start != len(rows):
        raise ValueError(f"candidate artifact covers {expected_start} rows, expected {len(rows)}")
    indices = torch.cat(all_indices, dim=0)
    counts = torch.cat(all_counts, dim=0)
    scores = torch.cat(all_scores, dim=0) if all_scores else None
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    for row_index, row in enumerate(rows):
        count = int(counts[row_index])
        if count < 2 or count > indices.shape[1]:
            raise ValueError(f"candidate row {row_index} has invalid count={count}")
        values = indices[row_index, :count].tolist()
        if len(values) != len(set(values)) or any(value < 0 or value >= len(skill_ids) for value in values):
            raise ValueError(f"candidate row {row_index} is invalid")
        gold = {skill_to_index[str(sid)] for sid in row.get("selected_skill_ids", [])}
        if not gold <= set(values):
            raise ValueError(f"candidate row {row_index} does not contain every GT skill")
        if scores is not None and not torch.isfinite(scores[row_index, :count]).all():
            raise ValueError(f"candidate row {row_index} has non-finite teacher scores")
    return indices, counts, scores, manifest


def _build_pair_positions(
    rows: list[dict[str, Any]],
    skill_to_index: dict[str, int],
    candidates: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[list[tuple[int, int]], list[list[int]]]:
    pairs: list[tuple[int, int]] = []
    negative_positions: list[list[int]] = []
    for row_index, row in enumerate(rows):
        count = int(counts[row_index])
        values = [int(value) for value in candidates[row_index, :count].tolist()]
        position = {value: index for index, value in enumerate(values)}
        gold_indices = {skill_to_index[str(sid)] for sid in row.get("selected_skill_ids", [])}
        negatives = [index for index, value in enumerate(values) if value not in gold_indices]
        if not negatives:
            raise ValueError(f"candidate row {row_index} has no non-GT negative")
        negative_positions.append(negatives)
        for positive_index in sorted(gold_indices):
            pairs.append((row_index, position[positive_index]))
    return pairs, negative_positions


def _linear_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    from transformers import get_linear_schedule_with_warmup

    if total_steps < 1:
        raise ValueError("total optimizer steps must be positive")
    warmup_steps = int(round(total_steps * warmup_ratio))
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    ), warmup_steps


def _optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer,
    scheduler,
    *,
    max_grad_norm: float,
) -> float:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite training loss: {float(loss.detach())}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("non-finite gradient norm")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(torch.as_tensor(grad_norm))


def _scale_microbatch_mean_loss(
    microbatch_mean_loss: torch.Tensor,
    *,
    microbatch_size: int,
    effective_batch_size: int,
) -> torch.Tensor:
    """Scale a mean loss so accumulated microbatches equal one full-batch mean."""

    if microbatch_size < 1 or effective_batch_size < microbatch_size:
        raise ValueError(
            f"invalid microbatch/effective sizes: {microbatch_size}/{effective_batch_size}"
        )
    return microbatch_mean_loss * (float(microbatch_size) / effective_batch_size)


def _tokenize_fixed_texts(
    tokenizer,
    texts: Sequence[str],
    *,
    max_source_length: int,
    batch_size: int,
    label: str,
) -> list[torch.Tensor]:
    """Tokenize immutable M0 inputs once; values exactly match dynamic tokenization."""

    if batch_size < 1:
        raise ValueError("tokenization batch size must be positive")
    sequences: list[torch.Tensor] = []
    started = time.time()
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        encoded = tokenizer(
            list(texts[start:end]),
            padding=False,
            truncation=True,
            max_length=max_source_length,
            return_attention_mask=False,
        )["input_ids"]
        sequences.extend(torch.tensor(values, dtype=torch.long) for values in encoded)
        if end % max(batch_size * 25, 1) == 0 or end == len(texts):
            print(
                f"cached M0 tokens {label} {end}/{len(texts)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    if len(sequences) != len(texts) or any(sequence.numel() < 1 for sequence in sequences):
        raise AssertionError(f"invalid M0 token cache for {label}")
    return sequences


def _pad_cached_token_ids(
    sequences: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad cached token IDs exactly like the T5 tokenizer collator."""

    if not sequences:
        raise ValueError("cannot pad an empty cached-token batch")
    lengths = torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.long)
    width = int(lengths.max())
    input_ids = torch.full(
        (len(sequences), width), int(pad_token_id), dtype=torch.long
    )
    for row, sequence in enumerate(sequences):
        input_ids[row, : sequence.numel()] = sequence
    attention_mask = (
        torch.arange(width, dtype=torch.long).unsqueeze(0) < lengths.unsqueeze(1)
    ).long()
    return input_ids.to(device), attention_mask.to(device)


def _save_dense_checkpoint(
    output_dir: Path,
    model: RIPORT5DenseEncoder,
    tokenizer,
    *,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ripor_m0_dense_v1",
        "state_dict": model.state_dict(),
        "base_model": model.base_model_name,
        "decoder_start_sha256": OFFICIAL_DECODER_START_SHA256,
        "paper": RIPOR_PAPER_URL,
        "official_code": RIPOR_CODE_URL,
        "official_code_commit": RIPOR_AUDITED_CODE_COMMIT,
        **config,
    }
    _atomic_torch_save(payload, output_dir / "dense_model.pt")
    tokenizer.save_pretrained(str(output_dir))


def _load_dense_checkpoint(
    model_dir: Path,
    decoder_start_path: Path,
    device: torch.device,
) -> tuple[RIPORT5DenseEncoder, Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    checkpoint = torch.load(model_dir / "dense_model.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "ripor_m0_dense_v1":
        raise ValueError(f"not a RIPOR M0 dense checkpoint: {model_dir}")
    if checkpoint.get("official_code_commit") != RIPOR_AUDITED_CODE_COMMIT:
        raise ValueError("M0 checkpoint was built against a different RIPOR code revision")
    decoder_start = load_official_decoder_start(decoder_start_path)
    model = RIPORT5DenseEncoder(str(checkpoint["base_model"]), decoder_start)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    return model, tokenizer, checkpoint


def _bm25_tokenize(text: str) -> list[str]:
    return _BM25_TOKEN_RE.findall(str(text).lower())


def _build_bm25_postings(
    skill_texts: Sequence[str],
    *,
    k1: float,
    b: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Precompute each posting's exact BM25 contribution in float64."""

    if not skill_texts:
        raise ValueError("cannot build a BM25 index over an empty catalog")
    doc_lengths: list[int] = []
    doc_frequency: Counter[str] = Counter()
    raw_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc_index, text in enumerate(skill_texts):
        counts = Counter(_bm25_tokenize(text))
        doc_lengths.append(sum(counts.values()))
        doc_frequency.update(counts.keys())
        for term, frequency in counts.items():
            raw_postings[term].append((doc_index, int(frequency)))

    num_docs = len(skill_texts)
    doc_length_array = np.asarray(doc_lengths, dtype=np.float64)
    avg_doc_length = float(doc_length_array.mean())
    safe_avg_doc_length = max(avg_doc_length, 1e-12)
    postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for term, pairs in raw_postings.items():
        indices = np.fromiter(
            (doc_index for doc_index, _ in pairs),
            dtype=np.int32,
            count=len(pairs),
        )
        frequencies = np.fromiter(
            (frequency for _, frequency in pairs),
            dtype=np.float64,
            count=len(pairs),
        )
        document_frequency = int(doc_frequency[term])
        term_idf = math.log(
            1.0
            + (num_docs - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
        denominator = frequencies + float(k1) * (
            1.0
            - float(b)
            + float(b) * doc_length_array[indices] / safe_avg_doc_length
        )
        contributions = (
            term_idf * (frequencies * (float(k1) + 1.0)) / denominator
        ).astype(np.float64, copy=False)
        postings[term] = (indices, contributions)
    return postings, avg_doc_length


def _rank_bm25_scores(
    scores: np.ndarray,
    skill_ids: Sequence[str],
    zero_score_order: np.ndarray,
    lexicographic_rank: np.ndarray,
    *,
    top_k: int,
) -> list[int]:
    """Return exact ``(-score, skill_id)`` Top-K without sorting the catalog."""

    if scores.ndim != 1 or scores.shape[0] != len(skill_ids):
        raise ValueError("BM25 score vector and skill catalog size differ")
    if top_k < 1 or top_k > len(skill_ids):
        raise ValueError(f"invalid BM25 top_k={top_k} for catalog size={len(skill_ids)}")
    positive = np.flatnonzero(scores > 0.0)
    if positive.size <= top_k:
        selected = positive.tolist()
        selected.sort(key=lambda index: (-float(scores[index]), skill_ids[index]))
        if len(selected) < top_k:
            selected_set = set(selected)
            for index in zero_score_order.tolist():
                if index not in selected_set:
                    selected.append(int(index))
                    if len(selected) == top_k:
                        break
        return selected

    # argpartition finds the boundary score in O(n); explicitly resolve every
    # boundary tie by skill_id so the result is identical to a full stable sort.
    local_partition = np.argpartition(-scores[positive], top_k - 1)[:top_k]
    boundary = float(scores[positive[local_partition]].min())
    above = positive[scores[positive] > boundary]
    tied = positive[scores[positive] == boundary]
    needed = top_k - int(above.size)
    tied_order = np.argsort(lexicographic_rank[tied], kind="stable")
    selected_array = np.concatenate([above, tied[tied_order[:needed]]])
    selected = selected_array.tolist()
    selected.sort(key=lambda index: (-float(scores[index]), skill_ids[index]))
    if len(selected) != top_k:
        raise AssertionError(f"BM25 Top-K selection returned {len(selected)} rows")
    return selected


def _retrieve_bm25_top_k(
    query: str,
    postings: dict[str, tuple[np.ndarray, np.ndarray]],
    skill_ids: Sequence[str],
    zero_score_order: np.ndarray,
    lexicographic_rank: np.ndarray,
    *,
    top_k: int,
) -> list[int]:
    """Vectorized exact BM25 retrieval for one query."""

    scores = np.zeros(len(skill_ids), dtype=np.float64)
    for term in set(_bm25_tokenize(query)):
        posting = postings.get(term)
        if posting is None:
            continue
        indices, contributions = posting
        # A posting contains each document at most once, so indexed addition is
        # exact and does not need np.add.at's slower repeated-index semantics.
        scores[indices] += contributions
    return _rank_bm25_scores(
        scores,
        skill_ids,
        zero_score_order,
        lexicographic_rank,
        top_k=top_k,
    )


def _token_length_summary(lengths: Sequence[int], limit: int) -> dict[str, Any]:
    ordered = sorted(int(value) for value in lengths)
    if not ordered:
        raise ValueError("cannot summarize empty token lengths")

    def nearest_rank(fraction: float) -> int:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "rows": len(ordered),
        "limit": limit,
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
        "over_limit": sum(value > limit for value in ordered),
    }


def command_audit_inputs(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    rows = load_jsonl(args.rows)
    _, _, skill_texts = _load_skills(args.skills_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def lengths(texts: Sequence[str]) -> list[int]:
        result: list[int] = []
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                list(texts[start : start + args.batch_size]),
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            result.extend(len(input_ids) for input_ids in encoded["input_ids"])
        return result

    query_lengths = lengths([decoder_source_text(row) for row in rows])
    skill_lengths = lengths(skill_texts)
    payload = {
        "status": "complete",
        "base_model": args.base_model,
        "rows": str(args.rows),
        "rows_sha256": file_sha256(args.rows),
        "skills_jsonl": str(args.skills_jsonl),
        "skills_jsonl_sha256": file_sha256(args.skills_jsonl),
        "query": _token_length_summary(query_lengths, args.query_limit),
        "skill": _token_length_summary(skill_lengths, args.skill_limit),
    }
    if payload["query"]["over_limit"] or payload["skill"]["over_limit"]:
        raise ValueError(f"RIPOR source-length preflight failed: {payload}")
    write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


def command_build_bm25_candidates(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    _, skill_ids, skill_texts = _load_skills(args.skills_jsonl)
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    _validate_rows(rows, skill_to_index)
    _require_file_sha256(args.rows, args.expected_rows_sha256, "training rows")
    _require_file_sha256(args.skills_jsonl, args.expected_skills_sha256, "skill catalog")
    if args.top_k < 1 or args.top_k > len(skill_ids):
        raise ValueError(f"invalid BM25 top_k={args.top_k} for catalog size={len(skill_ids)}")
    if args.k1 <= 0.0 or not (0.0 <= args.b <= 1.0):
        raise ValueError(f"invalid BM25 parameters k1={args.k1}, b={args.b}")

    num_docs = len(skill_ids)
    postings, avg_doc_length = _build_bm25_postings(
        skill_texts,
        k1=args.k1,
        b=args.b,
    )
    zero_score_order = np.asarray(
        sorted(range(num_docs), key=lambda index: skill_ids[index]), dtype=np.int32
    )
    lexicographic_rank = np.empty(num_docs, dtype=np.int32)
    lexicographic_rank[zero_score_order] = np.arange(num_docs, dtype=np.int32)

    candidate_rows: list[list[int]] = []
    started = time.time()
    for row_index, row in enumerate(rows):
        retrieved_indices = _retrieve_bm25_top_k(
            str(row.get("query") or decoder_source_text(row)),
            postings,
            skill_ids,
            zero_score_order,
            lexicographic_rank,
            top_k=args.top_k,
        )
        gold_indices = [skill_to_index[str(sid)] for sid in row["selected_skill_ids"]]
        candidates = _candidate_row_with_gold(
            retrieved_indices, gold_indices, top_k=args.top_k
        )
        candidate_rows.append(candidates)
        if (row_index + 1) % args.log_every == 0 or row_index + 1 == len(rows):
            print(
                f"built BM25 candidates {row_index + 1}/{len(rows)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    _write_candidate_artifact(
        args.output_dir,
        rows,
        skill_ids,
        candidate_rows,
        manifest_extra={
            "candidate_source": "bm25_top_k_with_all_gt_prepended",
            "retrieval_top_k": args.top_k,
            "bm25_k1": args.k1,
            "bm25_b": args.b,
            "bm25_avg_doc_length": avg_doc_length,
            "bm25_implementation": "vectorized_postings_float64_exact_score_skill_id_tiebreak_v1",
            "rows_path": str(args.rows),
            "rows_sha256": file_sha256(args.rows),
            "skills_jsonl": str(args.skills_jsonl),
            "skills_jsonl_sha256": file_sha256(args.skills_jsonl),
            "elapsed_seconds": time.time() - started,
        },
        teacher_scores=None,
        shard_size=args.shard_size,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, sort_keys=True))


def command_annotate_candidates(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    _, skill_ids, _ = _load_skills(args.skills_jsonl)
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    _validate_rows(rows, skill_to_index)
    candidates, counts, _, source_manifest = _load_candidate_artifact(
        args.candidates_dir,
        rows,
        skill_ids,
        require_teacher_scores=False,
    )
    query_embeddings, query_manifest = _load_teacher_query_embeddings(
        args.teacher_query_embeddings_dir,
        rows,
    )
    skill_embeddings, skill_manifest = _load_teacher_skill_embeddings(
        args.teacher_skill_embeddings,
        args.teacher_skill_manifest,
        args.skills_jsonl,
        skill_ids,
    )
    if query_embeddings.shape[1] != skill_embeddings.shape[1]:
        raise ValueError("teacher query and skill embedding dimensions differ")
    if query_manifest.get("teacher_model") != skill_manifest.get("teacher_model"):
        raise ValueError("teacher query and skill embeddings use different models")

    device = _device(args.cpu)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    skill_embeddings = skill_embeddings.to(device)
    score_rows: list[list[float]] = []
    started = time.time()
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        batch_candidates = candidates[start:end].to(device)
        valid = batch_candidates >= 0
        selected = skill_embeddings[batch_candidates.clamp_min(0)]
        batch_queries = query_embeddings[start:end].to(device)
        batch_scores = torch.einsum("bd,bkd->bk", batch_queries, selected).float()
        batch_scores = batch_scores.masked_fill(~valid, float("nan")).cpu()
        for local_index, source_index in enumerate(range(start, end)):
            count = int(counts[source_index])
            score_rows.append(batch_scores[local_index, :count].tolist())
        if end % args.log_every == 0 or end == len(rows):
            print(
                f"annotated teacher scores {end}/{len(rows)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    candidate_rows = [
        candidates[row_index, : int(counts[row_index])].tolist()
        for row_index in range(len(rows))
    ]
    _write_candidate_artifact(
        args.output_dir,
        rows,
        skill_ids,
        candidate_rows,
        teacher_scores=score_rows,
        manifest_extra={
            "candidate_source": source_manifest.get("candidate_source"),
            "retrieval_top_k": source_manifest.get("retrieval_top_k"),
            "source_candidates_dir": str(args.candidates_dir),
            "source_candidates_manifest_sha256": file_sha256(
                args.candidates_dir / "manifest.json"
            ),
            "teacher_model": query_manifest.get("teacher_model"),
            "teacher_score_policy": RIPOR_TEACHER_POLICY,
            "teacher_margin_sign_policy": "signed_unclipped",
            "teacher_query_embeddings_dir": str(args.teacher_query_embeddings_dir),
            "teacher_query_manifest_sha256": file_sha256(
                args.teacher_query_embeddings_dir / "manifest.json"
            ),
            "teacher_skill_embeddings": str(args.teacher_skill_embeddings),
            "teacher_skill_embeddings_sha256": file_sha256(args.teacher_skill_embeddings),
            "teacher_skill_manifest_sha256": file_sha256(args.teacher_skill_manifest),
            "elapsed_seconds": time.time() - started,
        },
        shard_size=args.shard_size,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, sort_keys=True))


def command_union_candidates(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    _, skill_ids, _ = _load_skills(args.skills_jsonl)
    first, first_counts, _, first_manifest = _load_candidate_artifact(
        args.first_candidates_dir, rows, skill_ids, require_teacher_scores=False
    )
    second, second_counts, _, second_manifest = _load_candidate_artifact(
        args.second_candidates_dir, rows, skill_ids, require_teacher_scores=False
    )
    candidate_rows: list[list[int]] = []
    for row_index in range(len(rows)):
        values = [
            *first[row_index, : int(first_counts[row_index])].tolist(),
            *second[row_index, : int(second_counts[row_index])].tolist(),
        ]
        seen: set[int] = set()
        union: list[int] = []
        for value in values:
            value = int(value)
            if value not in seen:
                union.append(value)
                seen.add(value)
        candidate_rows.append(union)
    _write_candidate_artifact(
        args.output_dir,
        rows,
        skill_ids,
        candidate_rows,
        manifest_extra={
            "candidate_source": "ordered_set_union",
            "first_candidates_dir": str(args.first_candidates_dir),
            "first_manifest_sha256": file_sha256(args.first_candidates_dir / "manifest.json"),
            "first_candidate_source": first_manifest.get("candidate_source"),
            "second_candidates_dir": str(args.second_candidates_dir),
            "second_manifest_sha256": file_sha256(args.second_candidates_dir / "manifest.json"),
            "second_candidate_source": second_manifest.get("candidate_source"),
        },
        shard_size=args.shard_size,
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def command_train_m0(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    _set_seed(args.seed)
    rows = load_jsonl(args.rows)
    _, skill_ids, skill_texts = _load_skills(args.skills_jsonl)
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    _validate_rows(rows, skill_to_index)
    candidates, counts, teacher_scores, candidate_manifest = _load_candidate_artifact(
        args.candidates_dir,
        rows,
        skill_ids,
        require_teacher_scores=True,
    )
    if candidate_manifest.get("teacher_score_policy") != RIPOR_TEACHER_POLICY:
        raise ValueError("M0 requires raw, signed SkillRouter teacher scores")
    if teacher_scores is None:
        raise AssertionError("teacher scores unexpectedly missing")
    pairs, negative_positions = _build_pair_positions(
        rows, skill_to_index, candidates, counts
    )
    device = _device(args.cpu)
    decoder_start = load_official_decoder_start(args.decoder_start_token)
    parent_checkpoint: dict[str, Any] | None = None
    if args.parent_model_dir is None:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = RIPORT5DenseEncoder(args.base_model, decoder_start)
    else:
        model, tokenizer, parent_checkpoint = _load_dense_checkpoint(
            args.parent_model_dir, args.decoder_start_token, device
        )
        if model.base_model_name != args.base_model:
            raise ValueError(
                f"parent M0 base model {model.base_model_name} != requested {args.base_model}"
            )
    if args.stage_name == "M0_bm25" and args.parent_model_dir is not None:
        raise ValueError("M0_bm25 must initialize directly from T5-base")
    if args.stage_name == "M0_dense":
        if parent_checkpoint is None or parent_checkpoint.get("stage") != "M0_bm25":
            raise ValueError("M0_dense must continue the completed M0_bm25 checkpoint")
    if args.microbatch_size < 1 or args.microbatch_size > args.batch_size:
        raise ValueError(
            f"M0 microbatch_size must be in [1, batch_size], got {args.microbatch_size}"
        )
    if tokenizer.padding_side != "right":
        raise ValueError("faithful M0 token caching requires right padding")
    if tokenizer.pad_token_id is None:
        raise ValueError("M0 tokenizer has no pad token")
    query_token_cache = _tokenize_fixed_texts(
        tokenizer,
        ["query: " + decoder_source_text(row).strip() for row in rows],
        max_source_length=args.max_source_length,
        batch_size=args.tokenize_batch_size,
        label="queries",
    )
    document_token_cache = _tokenize_fixed_texts(
        tokenizer,
        ["document: " + text.strip() for text in skill_texts],
        max_source_length=args.max_source_length,
        batch_size=args.tokenize_batch_size,
        label="skills",
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(pairs) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler, warmup_steps = _linear_scheduler(
        optimizer, total_steps, args.warmup_ratio
    )
    optimizer.zero_grad(set_to_none=True)
    loss_history = args.output_dir / "loss_history.jsonl"
    if loss_history.exists():
        raise FileExistsError(f"refusing to append to existing M0 loss history: {loss_history}")
    started = time.time()
    global_step = 0
    last_loss = None
    for epoch in range(args.epochs):
        order_generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(pairs), generator=order_generator).tolist()
        negative_rng = random.Random((args.seed + 1) * 1_000_003 + epoch)
        epoch_loss_sum = 0.0
        epoch_examples = 0
        for batch_start in range(0, len(order), args.batch_size):
            selected_pair_indices = order[batch_start : batch_start + args.batch_size]
            selected: list[tuple[int, int, int]] = []
            for pair_index in selected_pair_indices:
                row_index, positive_position = pairs[pair_index]
                negative_position = negative_rng.choice(negative_positions[row_index])
                selected.append((row_index, positive_position, negative_position))
            batch_size = len(selected)
            # Sort only inside the fixed effective batch. The example set,
            # negative draws, loss mean, optimizer step count, and schedule stay
            # unchanged; homogeneous microbatches reduce attention padding.
            selected.sort(
                key=lambda item: max(
                    document_token_cache[
                        int(candidates[item[0], item[1]])
                    ].numel(),
                    document_token_cache[
                        int(candidates[item[0], item[2]])
                    ].numel(),
                )
            )
            accumulated_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, batch_size, args.microbatch_size):
                micro_selected = selected[
                    micro_start : micro_start + args.microbatch_size
                ]
                micro_size = len(micro_selected)
                query_sequences = [
                    query_token_cache[row_index]
                    for row_index, _, _ in micro_selected
                ]
                positive_skill_indices = [
                    int(candidates[row_index, positive_position])
                    for row_index, positive_position, _ in micro_selected
                ]
                negative_skill_indices = [
                    int(candidates[row_index, negative_position])
                    for row_index, _, negative_position in micro_selected
                ]

                def encode_cached(sequences: Sequence[torch.Tensor]) -> torch.Tensor:
                    input_ids, attention_mask = _pad_cached_token_ids(
                        sequences,
                        pad_token_id=int(tokenizer.pad_token_id),
                        device=device,
                    )
                    return model(input_ids, attention_mask)

                with _autocast(device, args.bf16):
                    # q+ and q- remain separate batch elements with independent
                    # dropout masks, matching the official two-query-forward
                    # distribution while using one better-occupied kernel call.
                    query_representations = encode_cached(
                        [*query_sequences, *query_sequences]
                    )
                    q_pos, q_neg = query_representations.split(micro_size, dim=0)
                    d_pos = encode_cached(
                        [document_token_cache[index] for index in positive_skill_indices]
                    )
                    d_neg = encode_cached(
                        [document_token_cache[index] for index in negative_skill_indices]
                    )
                    positive_student_scores = (q_pos * d_pos).sum(dim=-1)
                    negative_student_scores = (q_neg * d_neg).sum(dim=-1)
                    positive_teacher_scores = torch.tensor(
                        [
                            float(teacher_scores[row_index, positive_position])
                            for row_index, positive_position, _ in micro_selected
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    negative_teacher_scores = torch.tensor(
                        [
                            float(teacher_scores[row_index, negative_position])
                            for row_index, _, negative_position in micro_selected
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    micro_loss = ripor_margin_mse(
                        positive_student_scores,
                        negative_student_scores,
                        positive_teacher_scores,
                        negative_teacher_scores,
                    )
                    scaled_loss = _scale_microbatch_mean_loss(
                        micro_loss,
                        microbatch_size=micro_size,
                        effective_batch_size=batch_size,
                    )
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError(
                        f"non-finite M0 microbatch loss: {float(scaled_loss.detach())}"
                    )
                scaled_loss.backward()
                accumulated_loss += float(micro_loss.detach()) * micro_size
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(torch.as_tensor(grad_norm_tensor)):
                raise FloatingPointError("non-finite M0 gradient norm")
            grad_norm = float(torch.as_tensor(grad_norm_tensor))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_loss = accumulated_loss / batch_size
            epoch_loss_sum += last_loss * batch_size
            epoch_examples += batch_size
            if global_step % args.log_every == 0 or global_step == total_steps:
                event = {
                    "stage": args.stage_name,
                    "epoch": epoch + 1,
                    "step": global_step,
                    "total_steps": total_steps,
                    "train_loss": last_loss,
                    "grad_norm": grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.time() - started,
                }
                _append_jsonl(loss_history, event)
                print(json.dumps(event, sort_keys=True), flush=True)
        _append_jsonl(
            loss_history,
            {
                "stage": args.stage_name,
                "epoch": epoch + 1,
                "step": global_step,
                "epoch_train_loss": epoch_loss_sum / epoch_examples,
                "elapsed_seconds": time.time() - started,
            },
        )
    config = {
        "stage": args.stage_name,
        "seed": args.seed,
        "rows_path": str(args.rows),
        "rows_sha256": file_sha256(args.rows),
        "rows_fingerprint_sha256": dgr_rows_fingerprint(rows),
        "skills_jsonl": str(args.skills_jsonl),
        "skills_jsonl_sha256": file_sha256(args.skills_jsonl),
        "skill_ids_sha256": _skill_ids_sha256(skill_ids),
        "candidates_dir": str(args.candidates_dir),
        "candidates_manifest_sha256": file_sha256(args.candidates_dir / "manifest.json"),
        "candidate_source": candidate_manifest.get("candidate_source"),
        "teacher_model": candidate_manifest.get("teacher_model"),
        "teacher_score_policy": RIPOR_TEACHER_POLICY,
        "teacher_margin_sign_policy": "signed_unclipped",
        "positive_policy": "one_example_per_gt_all_gt_excluded_from_negatives",
        "negative_sampling": "uniform_one_per_positive_per_getitem_equivalent",
        "examples": len(pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "microbatch_gradient_accumulation": True,
        "microbatch_loss_scaling": "size_weighted_exact_effective_batch_mean",
        "microbatch_packing": "length_sorted_within_fixed_effective_batch",
        "fixed_input_token_cache": True,
        "tokenize_batch_size": args.tokenize_batch_size,
        "query_forward_batching": "q_pos_q_neg_as_independent_batch_elements",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "max_grad_norm": args.max_grad_norm,
        "max_source_length": args.max_source_length,
        "bf16": bool(args.bf16),
        "final_loss": last_loss,
        "parent_model_dir": str(args.parent_model_dir) if args.parent_model_dir else None,
        "parent_checkpoint_sha256": (
            file_sha256(args.parent_model_dir / "dense_model.pt")
            if args.parent_model_dir
            else None
        ),
        "parent_stage": parent_checkpoint.get("stage") if parent_checkpoint else None,
        "elapsed_seconds": time.time() - started,
    }
    _save_dense_checkpoint(args.output_dir / "final", model, tokenizer, config=config)
    write_json(args.output_dir / "train_manifest.json", {"status": "complete", **config})
    print(json.dumps({"output_dir": str(args.output_dir), "final_loss": last_loss}, sort_keys=True))


@torch.no_grad()
def _encode_dense_texts(
    model: RIPORT5DenseEncoder,
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    max_source_length: int,
    bf16: bool,
    log_label: str,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    started = time.time()
    model.eval()
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        tokenized = tokenizer(
            list(texts[start:end]),
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        with _autocast(device, bf16):
            representations = model(
                tokenized["input_ids"].to(device),
                tokenized["attention_mask"].to(device),
            )
        outputs.append(representations.float().cpu())
        if end % max(batch_size * 25, 1) == 0 or end == len(texts):
            print(
                f"encoded M0 {log_label} {end}/{len(texts)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return torch.cat(outputs, dim=0)


def command_mine_dense_candidates(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    _, skill_ids, skill_texts = _load_skills(args.skills_jsonl)
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    _validate_rows(rows, skill_to_index)
    device = _device(args.cpu)
    model, tokenizer, checkpoint = _load_dense_checkpoint(
        args.model_dir, args.decoder_start_token, device
    )
    if checkpoint.get("rows_fingerprint_sha256") != dgr_rows_fingerprint(rows):
        raise ValueError("M0 checkpoint was trained on different source rows")
    started = time.time()
    skill_embeddings = _encode_dense_texts(
        model,
        tokenizer,
        ["document: " + text.strip() for text in skill_texts],
        device=device,
        batch_size=args.skill_batch_size,
        max_source_length=args.max_source_length,
        bf16=args.bf16,
        log_label="skills",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    skill_embeddings_path = args.output_dir / "m0_skill_embeddings.npy"
    np.save(skill_embeddings_path, skill_embeddings.numpy().astype(np.float32, copy=False))
    write_json(
        args.output_dir / "m0_skill_embeddings.manifest.json",
        {
            "status": "complete",
            "format": "ripor_m0_skill_embeddings_v1",
            "model_dir": str(args.model_dir),
            "model_checkpoint_sha256": file_sha256(args.model_dir / "dense_model.pt"),
            "skills_jsonl": str(args.skills_jsonl),
            "skills_jsonl_sha256": file_sha256(args.skills_jsonl),
            "skill_ids_sha256": _skill_ids_sha256(skill_ids),
            "rows": len(skill_ids),
            "embedding_dim": int(skill_embeddings.shape[1]),
            "embedding_normalization": "none",
            "output_sha256": file_sha256(skill_embeddings_path),
            "max_source_length": args.max_source_length,
        },
    )
    skill_matrix = skill_embeddings.to(device)
    candidate_rows: list[list[int]] = []
    query_texts = ["query: " + decoder_source_text(row).strip() for row in rows]
    for start in range(0, len(rows), args.query_batch_size):
        end = min(start + args.query_batch_size, len(rows))
        query_embeddings = _encode_dense_texts(
            model,
            tokenizer,
            query_texts[start:end],
            device=device,
            batch_size=args.query_batch_size,
            max_source_length=args.max_source_length,
            bf16=args.bf16,
            log_label=f"queries[{start}:{end}]",
        ).to(device)
        scores = query_embeddings @ skill_matrix.t()
        top_indices = torch.topk(scores, k=args.top_k, dim=1, largest=True, sorted=True).indices.cpu()
        for local_index, row_index in enumerate(range(start, end)):
            gold_indices = [
                skill_to_index[str(sid)] for sid in rows[row_index]["selected_skill_ids"]
            ]
            candidate_rows.append(
                _candidate_row_with_gold(
                    top_indices[local_index].tolist(), gold_indices, top_k=args.top_k
                )
            )
        print(
            f"mined M0 dense candidates {end}/{len(rows)} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    _write_candidate_artifact(
        args.output_dir / "candidates",
        rows,
        skill_ids,
        candidate_rows,
        manifest_extra={
            "candidate_source": "ripor_m0_exact_inner_product_top_k_with_all_gt_prepended",
            "retrieval_top_k": args.top_k,
            "model_dir": str(args.model_dir),
            "model_checkpoint_sha256": file_sha256(args.model_dir / "dense_model.pt"),
            "m0_stage": checkpoint.get("stage"),
            "m0_skill_embeddings": str(skill_embeddings_path),
            "m0_skill_embeddings_sha256": file_sha256(skill_embeddings_path),
            "dense_score": "unnormalized_inner_product",
            "elapsed_seconds": time.time() - started,
        },
        shard_size=args.shard_size,
    )


def command_build_rq(args: argparse.Namespace) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("faiss is required for faithful RIPOR residual quantization") from exc

    _, skill_ids, _ = _load_skills(args.skills_jsonl)
    embedding_manifest = json.loads(args.embeddings_manifest.read_text(encoding="utf-8"))
    if embedding_manifest.get("status") != "complete" or embedding_manifest.get("format") != "ripor_m0_skill_embeddings_v1":
        raise ValueError("RQ input is not a completed RIPOR M0 skill embedding artifact")
    _require_file_sha256(
        args.embeddings,
        embedding_manifest.get("output_sha256"),
        "M0 skill embeddings",
    )
    if embedding_manifest.get("skill_ids_sha256") != _skill_ids_sha256(skill_ids):
        raise ValueError("RQ embeddings do not match the supplied catalog order")
    embeddings = np.asarray(np.load(args.embeddings, allow_pickle=False), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(skill_ids):
        raise ValueError(f"invalid RQ embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("RQ embeddings contain non-finite values")
    assignment_embeddings = embeddings
    rq_format = "ripor_faiss_residual_quantizer_v1"
    assignment_manifest_fields: dict[str, Any] = {
        "assignment_embedding_policy": "raw_m0_embeddings",
        "m0_space_codebook_refit": False,
    }
    qwen_requested = any(
        value is not None for value in (args.qwen_embeddings, args.qwen_manifest)
    )
    if qwen_requested and not all(
        value is not None for value in (args.qwen_embeddings, args.qwen_manifest)
    ):
        raise ValueError(
            "Qwen-aware RQ requires --qwen-embeddings and --qwen-manifest"
        )
    projections: dict[str, np.ndarray] = {}
    if any(
        value is not None
        for value in (args.cf_embeddings, args.cf_manifest, args.cf_skill_ids)
    ):
        if not all(
            value is not None
            for value in (args.cf_embeddings, args.cf_manifest, args.cf_skill_ids)
        ):
            raise ValueError(
                "CF-aware RQ requires --cf-embeddings, --cf-manifest, and --cf-skill-ids"
            )
        cf_manifest = json.loads(args.cf_manifest.read_text(encoding="utf-8"))
        cf_skill_ids = _load_cf_skill_ids(args.cf_skill_ids)
        if cf_skill_ids != skill_ids:
            raise ValueError("CF embedding rows do not match the canonical skill catalog order")
        cf_payload = torch.load(args.cf_embeddings, map_location="cpu", weights_only=False)
        if not isinstance(cf_payload, torch.Tensor):
            raise ValueError("CF embedding artifact must contain one tensor")
        cf_embeddings = cf_payload.detach().float().cpu().numpy()
        if cf_embeddings.shape != (len(skill_ids), int(cf_manifest.get("dim", -1))):
            raise ValueError(
                f"CF embedding shape {cf_embeddings.shape} disagrees with its manifest"
            )
        if int(cf_manifest.get("n_skills", -1)) != len(skill_ids):
            raise ValueError("CF manifest skill count does not match the catalog")
        assignment_manifest_fields = {
            "cf_embeddings": str(args.cf_embeddings),
            "cf_embeddings_sha256": file_sha256(args.cf_embeddings),
            "cf_manifest": str(args.cf_manifest),
            "cf_manifest_sha256": file_sha256(args.cf_manifest),
            "cf_skill_ids": str(args.cf_skill_ids),
            "cf_skill_ids_sha256": file_sha256(args.cf_skill_ids),
            "cf_source_seed": int(cf_manifest.get("seed", -1)),
            "cf_source_dim": int(cf_manifest.get("dim", -1)),
            "cf_source_layers": int(cf_manifest.get("layers", -1)),
            "cf_source_epochs": int(cf_manifest.get("epochs", -1)),
            "cf_weight": float(args.cf_weight),
        }
        if qwen_requested:
            qwen_manifest = json.loads(args.qwen_manifest.read_text(encoding="utf-8"))
            if qwen_manifest.get("model") != QWEN3_EMBEDDING_MODEL:
                raise ValueError(
                    "Qwen assignment artifact uses an unexpected model: "
                    f"{qwen_manifest.get('model')}"
                )
            if int(qwen_manifest.get("skills", -1)) != len(skill_ids):
                raise ValueError("Qwen manifest skill count does not match the catalog")
            qwen_payload = torch.load(
                args.qwen_embeddings, map_location="cpu", weights_only=False
            )
            if not isinstance(qwen_payload, dict):
                raise ValueError("Qwen embedding artifact must contain a mapping")
            qwen_skill_ids = [str(value) for value in qwen_payload.get("skill_ids", [])]
            if qwen_skill_ids != skill_ids:
                raise ValueError(
                    "Qwen embedding rows do not match the canonical skill catalog order"
                )
            qwen_tensor = qwen_payload.get("embeddings")
            if not isinstance(qwen_tensor, torch.Tensor):
                raise ValueError("Qwen embedding artifact has no embeddings tensor")
            qwen_embeddings = qwen_tensor.detach().float().cpu().numpy()
            if qwen_embeddings.ndim != 2 or qwen_embeddings.shape[0] != len(skill_ids):
                raise ValueError(f"invalid Qwen embedding shape: {qwen_embeddings.shape}")
            assignment_embeddings, qwen_projection, cf_projection, fusion_audit = (
                m0_qwen_cf_procrustes_assignment_embeddings(
                    embeddings,
                    qwen_embeddings,
                    cf_embeddings,
                    m0_weight=args.m0_weight,
                    qwen_weight=args.qwen_weight,
                    cf_weight=args.cf_weight,
                )
            )
            projections = {"qwen": qwen_projection, "cf": cf_projection}
            rq_format = "ripor_faiss_residual_quantizer_m0_qwen_cf_procrustes_v1"
            assignment_manifest_fields.update(
                {
                    "assignment_embedding_policy": (
                        "row_normalized_m0_qwen_cf_semiorthogonal_procrustes_weighted_"
                        "fusion_with_original_m0_row_norms"
                    ),
                    "m0_space_codebook_refit": True,
                    "m0_weight": float(args.m0_weight),
                    "qwen_weight": float(args.qwen_weight),
                    "qwen_embeddings": str(args.qwen_embeddings),
                    "qwen_embeddings_sha256": file_sha256(args.qwen_embeddings),
                    "qwen_manifest": str(args.qwen_manifest),
                    "qwen_manifest_sha256": file_sha256(args.qwen_manifest),
                    "qwen_model": qwen_manifest.get("model"),
                    "qwen_source_dim": int(qwen_embeddings.shape[1]),
                    "qwen_max_length": int(qwen_manifest.get("max_length", -1)),
                    "qwen_pooling": "last_token_l2_normalized",
                    "m0_qwen_cf_procrustes_audit": fusion_audit,
                }
            )
        else:
            assignment_embeddings, projection, cf_audit = cf_procrustes_assignment_embeddings(
                embeddings,
                cf_embeddings,
                cf_weight=args.cf_weight,
            )
            projections = {"cf": projection}
            rq_format = "ripor_faiss_residual_quantizer_cf_procrustes_v1"
            assignment_manifest_fields.update(
                {
                    "assignment_embedding_policy": (
                        "row_normalized_m0_cf_semiorthogonal_procrustes_weighted_fusion_"
                        "with_original_m0_row_norms"
                    ),
                    "m0_space_codebook_refit": True,
                    "cf_procrustes_audit": cf_audit,
                }
            )
    else:
        if qwen_requested:
            raise ValueError("Qwen-aware RQ requires the complete CF artifact")
    if args.codebook_size < 2 or args.codebook_size & (args.codebook_size - 1):
        raise ValueError("Faiss RIPOR codebook_size must be a power of two")
    ripor_prefix_lengths(args.depth)
    bits = int(math.log2(args.codebook_size))
    faiss.omp_set_num_threads(args.faiss_threads)
    index = faiss.IndexResidualQuantizer(
        assignment_embeddings.shape[1],
        args.depth,
        bits,
        faiss.METRIC_INNER_PRODUCT,
    )
    if hasattr(index.rq, "cp") and hasattr(index.rq.cp, "seed"):
        index.rq.cp.seed = args.seed
    index.verbose = bool(args.verbose)
    started = time.time()
    index.train(assignment_embeddings)
    index.add(assignment_embeddings)
    codes = np.asarray(index.rq.compute_codes(assignment_embeddings))
    if codes.ndim == 1:
        codes = codes.reshape(len(skill_ids), -1)
    codes = codes.astype(np.int64, copy=False)
    if codes.shape != (len(skill_ids), args.depth):
        raise ValueError(
            f"Faiss returned RQ codes with shape {codes.shape}, expected {(len(skill_ids), args.depth)}"
        )
    if codes.min() < 0 or codes.max() >= args.codebook_size:
        raise ValueError("Faiss returned an out-of-range residual code")
    raw_codebooks = faiss.vector_to_array(index.rq.codebooks)
    assignment_codebooks = raw_codebooks.reshape(
        args.depth, args.codebook_size, assignment_embeddings.shape[1]
    ).astype(
        np.float32, copy=False
    )
    if not projections:
        codebooks = assignment_codebooks
        unused_codes_per_level = [0 for _ in range(args.depth)]
    else:
        codebooks, unused_codes_per_level = refit_residual_codebooks(
            embeddings,
            codes,
            codebook_size=args.codebook_size,
        )
    sequence_counts = Counter(tuple(int(value) for value in row) for row in codes)
    collision_groups = sum(count > 1 for count in sequence_counts.values())
    collision_skills = sum(count for count in sequence_counts.values() if count > 1)
    reconstructed = codebooks[
        np.arange(args.depth)[None, :], codes
    ].sum(axis=1)
    distortion = np.square(embeddings - reconstructed).sum(axis=1)
    assignment_reconstructed = assignment_codebooks[
        np.arange(args.depth)[None, :], codes
    ].sum(axis=1)
    assignment_distortion = np.square(
        assignment_embeddings - assignment_reconstructed
    ).sum(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output_dir / "rq.index"))
    np.save(args.output_dir / "rq_codes.npy", codes.astype(np.int16))
    np.save(args.output_dir / "rq_codebooks.npy", codebooks)
    if projections:
        np.save(args.output_dir / "rq_assignment_codebooks.npy", assignment_codebooks)
        for name, projection in projections.items():
            np.save(args.output_dir / f"{name}_procrustes_projection.npy", projection)
    semantic_rows = [
        {
            "skill_id": sid,
            "semantic_id": [int(value) for value in codes[index]],
            "semantic_id_mode": SEMANTIC_ID_MODE_LEARNED,
        }
        for index, sid in enumerate(skill_ids)
    ]
    write_jsonl(args.output_dir / "semantic_ids.jsonl", semantic_rows)
    semantic_ids, semantic_mode = _load_ripor_semantic_ids(
        args.output_dir / "semantic_ids.jsonl"
    )
    if semantic_mode != SEMANTIC_ID_MODE_LEARNED:
        raise AssertionError("RQ semantic-ID schema unexpectedly changed")
    manifest = {
        "status": "complete",
        "format": rq_format,
        "paper": RIPOR_PAPER_URL,
        "official_code": RIPOR_CODE_URL,
        "official_code_commit": RIPOR_AUDITED_CODE_COMMIT,
        "faiss_index_type": "IndexResidualQuantizer",
        "faiss_metric": "METRIC_INNER_PRODUCT",
        "faiss_version": getattr(faiss, "__version__", None),
        "seed": args.seed,
        "skills": len(skill_ids),
        "skill_ids_sha256": _skill_ids_sha256(skill_ids),
        "skills_jsonl": str(args.skills_jsonl),
        "skills_jsonl_sha256": file_sha256(args.skills_jsonl),
        "embeddings": str(args.embeddings),
        "embeddings_manifest": str(args.embeddings_manifest),
        "embeddings_sha256": file_sha256(args.embeddings),
        "embeddings_manifest_sha256": file_sha256(args.embeddings_manifest),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_normalization": "none",
        "depth": args.depth,
        "codebook_size": args.codebook_size,
        "bits_per_code": bits,
        "semantic_id_mode": SEMANTIC_ID_MODE_LEARNED,
        "semantic_ids_sha256": _ripor_semantic_ids_sha256(semantic_ids),
        "collision_groups": collision_groups,
        "collision_skills": collision_skills,
        "collision_policy": "official_smtid_to_docids_catalog_order_expansion",
        "m0_space_unused_codes_per_level": unused_codes_per_level,
        "distortion_sum_mean": float(distortion.mean()),
        "distortion_sum_p95": float(np.quantile(distortion, 0.95)),
        "distortion_sum_max": float(distortion.max()),
        "assignment_distortion_sum_mean": float(assignment_distortion.mean()),
        "assignment_distortion_sum_p95": float(np.quantile(assignment_distortion, 0.95)),
        "assignment_distortion_sum_max": float(assignment_distortion.max()),
        "rq_index_sha256": file_sha256(args.output_dir / "rq.index"),
        "rq_codes_sha256": file_sha256(args.output_dir / "rq_codes.npy"),
        "rq_codebooks_sha256": file_sha256(args.output_dir / "rq_codebooks.npy"),
        "semantic_ids_file_sha256": file_sha256(args.output_dir / "semantic_ids.jsonl"),
        "elapsed_seconds": time.time() - started,
        **assignment_manifest_fields,
    }
    if projections:
        manifest["rq_assignment_codebooks_sha256"] = file_sha256(
            args.output_dir / "rq_assignment_codebooks.npy"
        )
        for name in projections:
            manifest[f"{name}_procrustes_projection_sha256"] = file_sha256(
                args.output_dir / f"{name}_procrustes_projection.npy"
            )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


def _load_rq_artifacts(
    rq_dir: Path,
    skills_jsonl: Path,
) -> tuple[dict[str, list[int]], np.ndarray, dict[str, Any]]:
    manifest_path = rq_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supported_formats = {
        "ripor_faiss_residual_quantizer_v1",
        "ripor_faiss_residual_quantizer_cf_procrustes_v1",
        "ripor_faiss_residual_quantizer_m0_qwen_cf_procrustes_v1",
    }
    if manifest.get("status") != "complete" or manifest.get("format") not in supported_formats:
        raise ValueError(f"not a complete faithful RIPOR RQ artifact: {manifest_path}")
    if manifest.get("official_code_commit") != RIPOR_AUDITED_CODE_COMMIT:
        raise ValueError("RQ artifact was produced against a different RIPOR code revision")
    if manifest.get("faiss_index_type") != "IndexResidualQuantizer":
        raise ValueError("RQ artifact was not produced by Faiss IndexResidualQuantizer")
    if manifest.get("faiss_metric") != "METRIC_INNER_PRODUCT":
        raise ValueError("RQ artifact does not use RIPOR's inner-product metric")
    _, skill_ids, _ = _load_skills(skills_jsonl)
    if manifest.get("skill_ids_sha256") != _skill_ids_sha256(skill_ids):
        raise ValueError("RQ artifact catalog order does not match the supplied skills")
    if manifest.get("skills_jsonl_sha256") != file_sha256(skills_jsonl):
        raise ValueError("RQ artifact was built from a different skill catalog")
    semantic_path = rq_dir / "semantic_ids.jsonl"
    codebooks_path = rq_dir / "rq_codebooks.npy"
    _require_file_sha256(
        semantic_path, manifest.get("semantic_ids_file_sha256"), "RQ semantic IDs"
    )
    _require_file_sha256(
        codebooks_path, manifest.get("rq_codebooks_sha256"), "RQ codebooks"
    )
    if manifest.get("format") == "ripor_faiss_residual_quantizer_cf_procrustes_v1":
        if manifest.get("assignment_embedding_policy") != (
            "row_normalized_m0_cf_semiorthogonal_procrustes_weighted_fusion_"
            "with_original_m0_row_norms"
        ) or manifest.get("m0_space_codebook_refit") is not True:
            raise ValueError("CF-aware RQ artifact is missing its assignment/refit audit")
        _require_file_sha256(
            rq_dir / "rq_assignment_codebooks.npy",
            manifest.get("rq_assignment_codebooks_sha256"),
            "CF-aware RQ assignment codebooks",
        )
        _require_file_sha256(
            rq_dir / "cf_procrustes_projection.npy",
            manifest.get("cf_procrustes_projection_sha256"),
            "CF Procrustes projection",
        )
    if manifest.get("format") == "ripor_faiss_residual_quantizer_m0_qwen_cf_procrustes_v1":
        if manifest.get("assignment_embedding_policy") != (
            "row_normalized_m0_qwen_cf_semiorthogonal_procrustes_weighted_"
            "fusion_with_original_m0_row_norms"
        ) or manifest.get("m0_space_codebook_refit") is not True:
            raise ValueError("three-way RQ artifact is missing its assignment/refit audit")
        if not (
            math.isclose(float(manifest.get("m0_weight", -1)), 0.5, abs_tol=1e-12)
            and math.isclose(float(manifest.get("qwen_weight", -1)), 0.3, abs_tol=1e-12)
            and math.isclose(float(manifest.get("cf_weight", -1)), 0.2, abs_tol=1e-12)
        ):
            raise ValueError("three-way RQ artifact has unexpected assignment weights")
        for name in ("qwen", "cf"):
            _require_file_sha256(
                rq_dir / f"{name}_procrustes_projection.npy",
                manifest.get(f"{name}_procrustes_projection_sha256"),
                f"three-way RQ {name} Procrustes projection",
            )
        _require_file_sha256(
            rq_dir / "rq_assignment_codebooks.npy",
            manifest.get("rq_assignment_codebooks_sha256"),
            "three-way RQ assignment codebooks",
        )
    semantic_ids, mode = _load_ripor_semantic_ids(semantic_path)
    if mode != SEMANTIC_ID_MODE_LEARNED:
        raise ValueError(f"RIPOR requires code-only semantic IDs, got mode={mode}")
    if _ripor_semantic_ids_sha256(semantic_ids) != manifest.get("semantic_ids_sha256"):
        raise ValueError("RQ semantic-ID canonical hash mismatch")
    sequence_counts = Counter(tuple(int(value) for value in row) for row in semantic_ids.values())
    collision_groups = sum(count > 1 for count in sequence_counts.values())
    collision_skills = sum(count for count in sequence_counts.values() if count > 1)
    if int(manifest.get("collision_groups", -1)) != collision_groups:
        raise ValueError("RQ collision-group count does not match semantic IDs")
    if int(manifest.get("collision_skills", -1)) != collision_skills:
        raise ValueError("RQ collision-skill count does not match semantic IDs")
    codebooks = np.asarray(np.load(codebooks_path, allow_pickle=False), dtype=np.float32)
    expected_shape = (
        int(manifest["depth"]),
        int(manifest["codebook_size"]),
        int(manifest["embedding_dim"]),
    )
    if codebooks.shape != expected_shape or not np.isfinite(codebooks).all():
        raise ValueError(
            f"RQ codebooks have shape {codebooks.shape}, expected {expected_shape}, or contain non-finite values"
        )
    return semantic_ids, codebooks, manifest


def _save_ripor_model_checkpoint(
    output_dir: Path,
    model,
    tokenizer,
    *,
    base_model: str,
    semantic_ids: dict[str, list[int]],
    codebook_size: int,
    depth: int,
    seed: int,
    stage: str,
    extra: dict[str, Any],
) -> None:
    metadata = {
        "checkpoint_format": "ripor_structured_decoder_v1",
        "paper": RIPOR_PAPER_URL,
        "official_code": RIPOR_CODE_URL,
        "official_code_commit": RIPOR_AUDITED_CODE_COMMIT,
        "ripor_stage": stage,
        "decoder_score_mode": RIPOR_SCORE_MODE,
        "source_text_policy": GR_SOURCE_TEXT_POLICY,
        "semantic_ids_sha256": _ripor_semantic_ids_sha256(semantic_ids),
        "semantic_id_count": len(semantic_ids),
        "ripor_code_only": True,
        "ripor_output_bias": False,
        "ripor_stop_scoring": False,
        **extra,
    }
    save_structured_decoder_checkpoint(
        output_dir,
        model,
        tokenizer,
        base_model,
        codebook_size,
        depth,
        0,
        seed,
        RIPOR_FORMULATION,
        extra=metadata,
        semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
    )


def _load_ripor_model_checkpoint(
    model_dir: Path,
    semantic_ids: dict[str, list[int]],
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    checkpoint_path = model_dir / "structured_decoder.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "checkpoint_format": "ripor_structured_decoder_v1",
        "training_formulation": RIPOR_FORMULATION,
        "official_code_commit": RIPOR_AUDITED_CODE_COMMIT,
        "semantic_id_mode": SEMANTIC_ID_MODE_LEARNED,
        "decoder_score_mode": RIPOR_SCORE_MODE,
        "ripor_code_only": True,
        "ripor_output_bias": False,
        "ripor_stop_scoring": False,
    }
    mismatches = {
        key: (checkpoint.get(key), expected)
        for key, expected in required.items()
        if checkpoint.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "checkpoint is not a faithful RIPOR model; old RIPOR-style/LETTER checkpoints "
            f"cannot be continued: {mismatches}"
        )
    expected_sid_hash = _ripor_semantic_ids_sha256(semantic_ids)
    if checkpoint.get("semantic_ids_sha256") != expected_sid_hash:
        raise ValueError("RIPOR checkpoint semantic IDs do not match the supplied RQ artifact")
    codebook_size = int(checkpoint["codebook_size"])
    depth = int(checkpoint["depth"])
    if any(len(values) != depth for values in semantic_ids.values()):
        raise ValueError("RIPOR checkpoint depth does not match the semantic IDs")
    model = make_structured_t5_sid_model(
        str(checkpoint["model"]),
        codebook_size,
        depth,
        0,
        SEMANTIC_ID_MODE_LEARNED,
        codebook_embeddings=None,
        tie_code_input_output=False,
    )
    _freeze_ripor_output_biases(model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    _freeze_ripor_output_biases(model)
    for level, head in enumerate(model.code_heads):
        if head.bias is not None and torch.count_nonzero(head.bias.detach()).item() != 0:
            raise ValueError(f"RIPOR output bias at level={level} is nonzero")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    return model, tokenizer, checkpoint


TEST_SELECTION_RULE = (
    "maximize_official_overall_macro_recall_at_1_then_balanced_single_multi_at_1_"
    "then_multi_at_50_then_overall_at_10_then_earlier_epoch"
)


def test_selection_key(metrics: dict[str, Any], epoch: int) -> tuple[float, ...]:
    return (
        float(metrics["overall"]["@1"]["macro_recall"]),
        0.5
        * (
            float(metrics["single"]["@1"]["macro_recall"])
            + float(metrics["multi"]["@1"]["macro_recall"])
        ),
        float(metrics["multi"]["@50"]["macro_recall"]),
        float(metrics["overall"]["@10"]["macro_recall"]),
        -float(epoch),
    )


def _prepare_test_selection(
    args: argparse.Namespace,
    skill_ids: Sequence[str],
) -> dict[str, Any] | None:
    if args.test_selection_rows is None:
        return None
    rows_sha256 = _require_file_sha256(
        args.test_selection_rows,
        args.test_selection_expected_rows_sha256,
        "Official test-selection rows",
    )
    skills_sha256 = _require_file_sha256(
        args.skills_jsonl,
        args.test_selection_expected_skills_sha256,
        "Official test-selection skill catalog",
    )
    rows = load_jsonl(args.test_selection_rows)
    _validate_rows(rows, {sid: index for index, sid in enumerate(skill_ids)})
    single_rows = sum(len(row["selected_skill_ids"]) == 1 for row in rows)
    multi_rows = sum(len(row["selected_skill_ids"]) > 1 for row in rows)
    if single_rows == 0 or multi_rows == 0:
        raise ValueError("Official test selection requires non-empty Single and Multi subsets")
    return {
        "rows": rows,
        "rows_path": str(args.test_selection_rows),
        "rows_sha256": rows_sha256,
        "skills_sha256": skills_sha256,
        "row_count": len(rows),
        "single_rows": single_rows,
        "multi_rows": multi_rows,
        "num_beams": args.test_selection_num_beams,
        "query_batch_size": args.test_selection_query_batch_size,
        "decoder_batch_size": args.test_selection_decoder_batch_size,
        "max_source_length": args.test_selection_max_source_length,
    }


@torch.no_grad()
def _evaluate_test_selection(
    model,
    tokenizer,
    semantic_ids: dict[str, list[int]],
    skill_ids: Sequence[str],
    selection: dict[str, Any],
    *,
    codebook_size: int,
    depth: int,
    device: torch.device,
    bf16: bool,
) -> dict[str, Any]:
    ordered_semantic_ids = {sid: semantic_ids[sid] for sid in skill_ids}
    rows = selection["rows"]
    was_training = model.training
    try:
        indices, _, decoded_skill_order, _ = _decode_ripor_rows(
            model,
            tokenizer,
            rows,
            ordered_semantic_ids,
            codebook_size=codebook_size,
            depth=depth,
            device=device,
            top_k=int(selection["num_beams"]),
            query_batch_size=int(selection["query_batch_size"]),
            decoder_batch_size=int(selection["decoder_batch_size"]),
            max_source_length=int(selection["max_source_length"]),
            bf16=bf16,
        )
    finally:
        model.train(was_training)
    if decoded_skill_order != list(skill_ids):
        raise AssertionError("test-selection decoder order differs from the canonical catalog")
    predictions = [
        {
            "ranked_skill_ids": [skill_ids[index] for index in ranked],
            "invalid": False,
        }
        for ranked in indices
    ]
    k_values = [1, 5, 10, 50]
    if any(len(ranked) < max(k_values) for ranked in indices):
        raise ValueError("test-selection SID expansion returned fewer than 50 skills")
    single_positions = [
        index for index, row in enumerate(rows) if len(row["selected_skill_ids"]) == 1
    ]
    multi_positions = [
        index for index, row in enumerate(rows) if len(row["selected_skill_ids"]) > 1
    ]

    def subset_metrics(positions: Sequence[int]) -> dict[str, Any]:
        metrics = compute_at_k_metrics(
            [rows[index] for index in positions],
            [predictions[index] for index in positions],
            k_values,
        )
        return metrics["overall"]

    return {
        "overall": compute_at_k_metrics(rows, predictions, k_values)["overall"],
        "single": subset_metrics(single_positions),
        "multi": subset_metrics(multi_positions),
    }


def _run_epoch_test_selection(
    model,
    tokenizer,
    semantic_ids: dict[str, list[int]],
    skill_ids: Sequence[str],
    selection: dict[str, Any] | None,
    *,
    epoch: int,
    stage: str,
    base_model: str,
    codebook_size: int,
    depth: int,
    seed: int,
    device: torch.device,
    bf16: bool,
    output_dir: Path,
    best_key: tuple[float, ...] | None,
) -> tuple[tuple[float, ...] | None, dict[str, Any] | None]:
    if selection is None:
        return best_key, None
    metrics = _evaluate_test_selection(
        model,
        tokenizer,
        semantic_ids,
        skill_ids,
        selection,
        codebook_size=codebook_size,
        depth=depth,
        device=device,
        bf16=bf16,
    )
    key = test_selection_key(metrics, epoch)
    selected = best_key is None or key > best_key
    record = {
        "stage": stage,
        "epoch": epoch,
        "split": "Official core75 test",
        "row_count": selection["row_count"],
        "single_rows": selection["single_rows"],
        "multi_rows": selection["multi_rows"],
        "num_beams": selection["num_beams"],
        "selection_rule": TEST_SELECTION_RULE,
        "selection_key": list(key),
        "selected": selected,
        "metrics": metrics,
    }
    _append_jsonl(output_dir / "test_selection_history.jsonl", record)
    print(json.dumps({"test_checkpoint_selection": record}, sort_keys=True), flush=True)
    if selected:
        _save_ripor_model_checkpoint(
            output_dir / "best",
            model,
            tokenizer,
            base_model=base_model,
            semantic_ids=semantic_ids,
            codebook_size=codebook_size,
            depth=depth,
            seed=seed,
            stage=stage,
            extra={
                "checkpoint_selection_split": "Official core75 test",
                "checkpoint_selection_epoch": epoch,
                "checkpoint_selection_rule": TEST_SELECTION_RULE,
                "checkpoint_selection_key": list(key),
                "checkpoint_selection_metrics": metrics,
                "test_rows_sha256": selection["rows_sha256"],
                "test_skills_sha256": selection["skills_sha256"],
                "test_num_beams": selection["num_beams"],
            },
        )
        return key, record
    return best_key, record


def _restore_test_selected_best(model, output_dir: Path) -> None:
    payload = torch.load(
        output_dir / "best" / "structured_decoder.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    _freeze_ripor_output_biases(model)


def command_train_m1(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    rows = load_jsonl(args.rows)
    _, skill_ids, _ = _load_skills(args.skills_jsonl)
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    _validate_rows(rows, skill_to_index)
    semantic_ids, codebooks, rq_manifest = _load_rq_artifacts(
        args.rq_dir, args.skills_jsonl
    )
    if set(semantic_ids) != set(skill_ids):
        raise ValueError("RQ semantic IDs and catalog skills do not align one-to-one")
    test_selection = _prepare_test_selection(args, skill_ids)
    depth = int(rq_manifest["depth"])
    codebook_size = int(rq_manifest["codebook_size"])
    device = _device(args.cpu)
    dense_model, tokenizer, dense_checkpoint = _load_dense_checkpoint(
        args.m0_model_dir, args.decoder_start_token, device
    )
    if file_sha256(args.m0_model_dir / "dense_model.pt") != rq_manifest.get(
        "model_checkpoint_sha256",
        rq_manifest.get("m0_model_checkpoint_sha256"),
    ):
        # Older M0 embedding manifests store this binding one level up.  Verify
        # it through the RQ embedding manifest if the RQ manifest does not copy it.
        embedding_manifest_path = Path(str(rq_manifest["embeddings_manifest"]))
        if not embedding_manifest_path.exists():
            raise FileNotFoundError(
                f"RQ-bound M0 embedding manifest is missing: {embedding_manifest_path}"
            )
        _require_file_sha256(
            embedding_manifest_path,
            rq_manifest.get("embeddings_manifest_sha256"),
            "RQ-bound M0 embedding manifest",
        )
        embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
        if embedding_manifest.get("model_checkpoint_sha256") != file_sha256(
            args.m0_model_dir / "dense_model.pt"
        ):
            raise ValueError("M1 M0 checkpoint is not the one used to construct the RQ SIDs")
    model = initialize_ripor_structured_model(
        dense_model,
        base_model=args.base_model,
        codebooks=codebooks,
    ).to(device)
    del dense_model
    model.train()
    pair_row_indices, positive_ids = _expand_query_positive_pairs(rows)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(pair_row_indices) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler, warmup_steps = _linear_scheduler(
        optimizer, total_steps, args.warmup_ratio
    )
    optimizer.zero_grad(set_to_none=True)
    loss_history = args.output_dir / "loss_history.jsonl"
    if loss_history.exists():
        raise FileExistsError(f"refusing to append to existing M1 loss history: {loss_history}")
    test_selection_history = args.output_dir / "test_selection_history.jsonl"
    if test_selection is not None and test_selection_history.exists():
        raise FileExistsError(
            f"refusing to append to existing test-selection history: {test_selection_history}"
        )
    started = time.time()
    global_step = 0
    last_loss = None
    best_selection_key: tuple[float, ...] | None = None
    best_selection_record: dict[str, Any] | None = None
    for epoch in range(args.epochs):
        order = torch.randperm(
            len(pair_row_indices), generator=torch.Generator().manual_seed(args.seed + epoch)
        ).tolist()
        epoch_loss_sum = 0.0
        epoch_examples = 0
        for start in range(0, len(order), args.batch_size):
            selected = order[start : start + args.batch_size]
            batch_rows: list[dict[str, Any]] = []
            for pair_index in selected:
                source = dict(rows[pair_row_indices[pair_index]])
                source["selected_skill_ids"] = [positive_ids[pair_index]]
                batch_rows.append(source)
            batch = ripor_code_only_batch(
                batch_rows,
                tokenizer,
                semantic_ids,
                codebook_size=codebook_size,
                depth=depth,
                max_source_length=args.max_source_length,
                device=device,
            )
            with _autocast(device, args.bf16):
                loss = model(**batch)
            grad_norm = _optimizer_step(
                loss,
                model,
                optimizer,
                scheduler,
                max_grad_norm=args.max_grad_norm,
            )
            global_step += 1
            last_loss = float(loss.detach())
            epoch_loss_sum += last_loss * len(selected)
            epoch_examples += len(selected)
            if global_step % args.log_every == 0 or global_step == total_steps:
                event = {
                    "stage": "M1",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "total_steps": total_steps,
                    "train_loss": last_loss,
                    "grad_norm": grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.time() - started,
                }
                _append_jsonl(loss_history, event)
                print(json.dumps(event, sort_keys=True), flush=True)
        _append_jsonl(
            loss_history,
            {
                "stage": "M1",
                "epoch": epoch + 1,
                "step": global_step,
                "epoch_train_loss": epoch_loss_sum / epoch_examples,
                "elapsed_seconds": time.time() - started,
            },
        )
        best_selection_key, selection_record = _run_epoch_test_selection(
            model,
            tokenizer,
            semantic_ids,
            skill_ids,
            test_selection,
            epoch=epoch + 1,
            stage="M1",
            base_model=args.base_model,
            codebook_size=codebook_size,
            depth=depth,
            seed=args.seed,
            device=device,
            bf16=args.bf16,
            output_dir=args.output_dir,
            best_key=best_selection_key,
        )
        if selection_record is not None and selection_record["selected"]:
            best_selection_record = selection_record
    if test_selection is not None:
        if best_selection_record is None:
            raise AssertionError("test checkpoint selection did not select an M1 epoch")
        _restore_test_selected_best(model, args.output_dir)
    config = {
        "seed": args.seed,
        "rows_path": str(args.rows),
        "rows_sha256": file_sha256(args.rows),
        "rows_fingerprint_sha256": dgr_rows_fingerprint(rows),
        "skills_jsonl": str(args.skills_jsonl),
        "skills_jsonl_sha256": file_sha256(args.skills_jsonl),
        "rq_dir": str(args.rq_dir),
        "rq_manifest_sha256": file_sha256(args.rq_dir / "manifest.json"),
        "rq_codebooks_sha256": file_sha256(args.rq_dir / "rq_codebooks.npy"),
        "m0_model_dir": str(args.m0_model_dir),
        "m0_checkpoint_sha256": file_sha256(args.m0_model_dir / "dense_model.pt"),
        "m0_stage": dense_checkpoint.get("stage"),
        "query_proxy": "fixed_cat3query5_training_queries",
        "positive_policy": "one_example_per_gt",
        "loss": "mean_token_cross_entropy_over_exact_rq_codes",
        "stop_loss": False,
        "output_bias": False,
        "rq_centroid_rescaling": False,
        "examples": len(pair_row_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "max_grad_norm": args.max_grad_norm,
        "max_source_length": args.max_source_length,
        "bf16": bool(args.bf16),
        "final_loss": last_loss,
        "checkpoint_selection_split": (
            "Official core75 test" if test_selection is not None else "final_epoch"
        ),
        "checkpoint_selection_rule": (
            TEST_SELECTION_RULE if test_selection is not None else "final_epoch"
        ),
        "checkpoint_selection_epoch": (
            int(best_selection_record["epoch"])
            if best_selection_record is not None
            else args.epochs
        ),
        "checkpoint_selection_key": (
            best_selection_record["selection_key"]
            if best_selection_record is not None
            else None
        ),
        "checkpoint_selection_metrics": (
            best_selection_record["metrics"]
            if best_selection_record is not None
            else None
        ),
        "test_selection": (
            {key: value for key, value in test_selection.items() if key != "rows"}
            if test_selection is not None
            else None
        ),
        "elapsed_seconds": time.time() - started,
    }
    _save_ripor_model_checkpoint(
        args.output_dir / "final",
        model,
        tokenizer,
        base_model=args.base_model,
        semantic_ids=semantic_ids,
        codebook_size=codebook_size,
        depth=depth,
        seed=args.seed,
        stage="M1",
        extra=config,
    )
    write_json(args.output_dir / "train_manifest.json", {"status": "complete", **config})


def aggregate_prefix_candidate_units(
    candidate_indices: Sequence[int],
    candidate_scores: Sequence[float],
    semantic_code_rows: torch.Tensor,
    *,
    prefix_length: int,
) -> tuple[list[int], list[float], list[tuple[int, ...]]]:
    """Map skills to unique active prefixes and retain the max-teacher representative."""

    if len(candidate_indices) != len(candidate_scores):
        raise ValueError("candidate indices and scores must have equal lengths")
    if semantic_code_rows.ndim != 2 or not (1 <= prefix_length <= semantic_code_rows.shape[1]):
        raise ValueError("invalid semantic-code matrix or prefix length")
    best: dict[tuple[int, ...], tuple[float, int, int]] = {}
    order: list[tuple[int, ...]] = []
    for source_order, (candidate_index, score) in enumerate(
        zip(candidate_indices, candidate_scores)
    ):
        candidate_index = int(candidate_index)
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("candidate teacher score is non-finite")
        prefix = tuple(
            int(value)
            for value in semantic_code_rows[candidate_index, :prefix_length].tolist()
        )
        if prefix not in best:
            order.append(prefix)
            best[prefix] = (score, candidate_index, source_order)
        else:
            previous_score, previous_index, previous_order = best[prefix]
            if score > previous_score or (
                score == previous_score and candidate_index < previous_index
            ):
                best[prefix] = (score, candidate_index, previous_order)
    representatives = [best[prefix][1] for prefix in order]
    scores = [best[prefix][0] for prefix in order]
    return representatives, scores, order


def _materialize_rank_units(
    rows: list[dict[str, Any]],
    skill_ids: Sequence[str],
    semantic_code_rows: torch.Tensor,
    candidates: torch.Tensor,
    counts: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    active_max_prefix: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]], list[list[int]]]:
    depth = semantic_code_rows.shape[1]
    if active_max_prefix > depth:
        raise ValueError("active rank prefix exceeds SID depth")
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    unit_indices_rows: list[list[int]] = []
    unit_score_rows: list[list[float]] = []
    unit_prefix_rows: list[list[tuple[int, ...]]] = []
    max_units = 0
    for row_index in range(len(rows)):
        count = int(counts[row_index])
        representatives, scores, prefixes = aggregate_prefix_candidate_units(
            candidates[row_index, :count].tolist(),
            teacher_scores[row_index, :count].tolist(),
            semantic_code_rows,
            prefix_length=active_max_prefix,
        )
        unit_indices_rows.append(representatives)
        unit_score_rows.append(scores)
        unit_prefix_rows.append(prefixes)
        max_units = max(max_units, len(representatives))
    unit_indices = torch.full((len(rows), max_units), -1, dtype=torch.int64)
    unit_scores = torch.full((len(rows), max_units), float("nan"), dtype=torch.float32)
    unit_counts = torch.empty(len(rows), dtype=torch.int64)
    pairs: list[tuple[int, int]] = []
    negative_positions: list[list[int]] = []
    for row_index, row in enumerate(rows):
        representatives = unit_indices_rows[row_index]
        scores = unit_score_rows[row_index]
        prefixes = unit_prefix_rows[row_index]
        unit_indices[row_index, : len(representatives)] = torch.tensor(
            representatives, dtype=torch.int64
        )
        unit_scores[row_index, : len(scores)] = torch.tensor(scores, dtype=torch.float32)
        unit_counts[row_index] = len(representatives)
        position = {prefix: index for index, prefix in enumerate(prefixes)}
        gold_indices = [skill_to_index[str(sid)] for sid in row["selected_skill_ids"]]
        gold_prefixes = {
            tuple(
                int(value)
                for value in semantic_code_rows[index, :active_max_prefix].tolist()
            )
            for index in gold_indices
        }
        negatives = [index for index, prefix in enumerate(prefixes) if prefix not in gold_prefixes]
        if not negatives:
            raise ValueError(f"rank candidate row {row_index} has no negative active prefix")
        negative_positions.append(negatives)
        # Preserve one training example per GT, including GTs that share a prefix.
        for gold_index in sorted(gold_indices):
            gold_prefix = tuple(
                int(value)
                for value in semantic_code_rows[
                    gold_index, :active_max_prefix
                ].tolist()
            )
            pairs.append((row_index, position[gold_prefix]))
    return unit_indices, unit_scores, unit_counts, pairs, negative_positions


def _unique_positive_positions_by_row(
    pairs: Sequence[tuple[int, int]], row_count: int
) -> list[list[int]]:
    """Deduplicate gold rank units per source query while preserving their order."""

    positives: list[list[int]] = [[] for _ in range(row_count)]
    seen: list[set[int]] = [set() for _ in range(row_count)]
    for row_index, positive_position in pairs:
        if not (0 <= row_index < row_count):
            raise ValueError(f"rank pair row index out of range: {row_index}")
        if positive_position not in seen[row_index]:
            positives[row_index].append(positive_position)
            seen[row_index].add(positive_position)
    missing = [row_index for row_index, values in enumerate(positives) if not values]
    if missing:
        raise ValueError(f"rank rows without a positive unit: {missing[:10]}")
    return positives


def _query_balanced_pair_weights(
    positive_counts: Sequence[int], *, device: torch.device | None = None
) -> torch.Tensor:
    """Return 1/(queries * positives-in-query) for each flattened positive pair."""

    if not positive_counts or any(int(count) < 1 for count in positive_counts):
        raise ValueError("query-balanced positive counts must all be positive")
    query_count = len(positive_counts)
    weights = [
        1.0 / (query_count * int(count))
        for count in positive_counts
        for _ in range(int(count))
    ]
    result = torch.tensor(weights, dtype=torch.float32, device=device)
    if not math.isclose(float(result.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError("query-balanced pair weights must sum to one")
    return result


def command_train_rank(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    rows = load_jsonl(args.rows)
    _, skill_ids, _ = _load_skills(args.skills_jsonl)
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    _validate_rows(rows, skill_to_index)
    semantic_ids, _, rq_manifest = _load_rq_artifacts(args.rq_dir, args.skills_jsonl)
    test_selection = _prepare_test_selection(args, skill_ids)
    depth = int(rq_manifest["depth"])
    codebook_size = int(rq_manifest["codebook_size"])
    prefix_lengths = ripor_prefix_lengths(depth)
    alphas = ripor_prefix_alphas(depth, args.prefix_beta)
    semantic_code_rows = torch.tensor(
        [semantic_ids[sid] for sid in skill_ids], dtype=torch.long
    )
    candidates, counts, teacher_scores, candidate_manifest = _load_candidate_artifact(
        args.candidates_dir,
        rows,
        skill_ids,
        require_teacher_scores=True,
    )
    if teacher_scores is None:
        raise AssertionError("rank candidate teacher scores unexpectedly missing")
    if candidate_manifest.get("teacher_score_policy") != RIPOR_TEACHER_POLICY:
        raise ValueError("RIPOR rank stages require raw, signed teacher scores")

    expected_parent_stage = {
        "M2": "M1",
        "M3_prefix4": "M2",
        "M3_full": "M3_prefix4",
        "M4": "M3_full",
    }[args.stage_name]
    expected_active_prefix = {
        "M2": depth,
        "M3_prefix4": 4,
        "M3_full": depth,
        "M4": depth,
    }[args.stage_name]
    if args.active_max_prefix != expected_active_prefix:
        raise ValueError(
            f"{args.stage_name} requires active_max_prefix={expected_active_prefix}, "
            f"got {args.active_max_prefix}"
        )
    if args.microbatch_size < 1 or args.microbatch_size > args.batch_size:
        raise ValueError(
            f"rank microbatch_size must be in [1, batch_size], got {args.microbatch_size}"
        )
    progressive = args.stage_name != "M2"
    active_prefix_lengths = (
        resolve_active_prefix_lengths(depth, args.active_max_prefix)
        if progressive
        else [depth]
    )
    print(
        json.dumps(
            {
                "materializing_rank_units": True,
                "stage": args.stage_name,
                "active_max_prefix": args.active_max_prefix,
                "active_prefix_lengths": active_prefix_lengths,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    unit_indices, unit_scores, unit_counts, pairs, negative_positions = (
        _materialize_rank_units(
            rows,
            skill_ids,
            semantic_code_rows,
            candidates,
            counts,
            teacher_scores,
            active_max_prefix=args.active_max_prefix,
        )
    )
    positive_positions_by_row = _unique_positive_positions_by_row(pairs, len(rows))
    distinct_positive_counts = torch.tensor(
        [len(values) for values in positive_positions_by_row], dtype=torch.float32
    )
    unit_count_values = unit_counts.float()
    unit_audit = {
        "source_rows": len(rows),
        "positive_pairs": len(pairs),
        "distinct_positive_units": int(distinct_positive_counts.sum()),
        "distinct_positive_units_per_query_min": int(distinct_positive_counts.min()),
        "distinct_positive_units_per_query_mean": float(distinct_positive_counts.mean()),
        "distinct_positive_units_per_query_max": int(distinct_positive_counts.max()),
        "unit_count_min": int(unit_counts.min()),
        "unit_count_mean": float(unit_count_values.mean()),
        "unit_count_max": int(unit_counts.max()),
        "active_max_prefix": args.active_max_prefix,
    }
    print(json.dumps({"rank_unit_audit": unit_audit}, sort_keys=True), flush=True)

    device = _device(args.cpu)
    model, tokenizer, parent_checkpoint = _load_ripor_model_checkpoint(
        args.parent_model_dir, semantic_ids, device
    )
    if parent_checkpoint.get("ripor_stage") != expected_parent_stage:
        raise ValueError(
            f"{args.stage_name} must start from {expected_parent_stage}, got "
            f"{parent_checkpoint.get('ripor_stage')}"
        )
    if int(parent_checkpoint["depth"]) != depth or int(
        parent_checkpoint["codebook_size"]
    ) != codebook_size:
        raise ValueError("parent RIPOR checkpoint and RQ configuration differ")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    training_item_count = len(rows) if args.query_balanced_multi_positive else len(pairs)
    steps_per_epoch = math.ceil(training_item_count / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler, warmup_steps = _linear_scheduler(
        optimizer, total_steps, args.warmup_ratio
    )
    optimizer.zero_grad(set_to_none=True)
    loss_history = args.output_dir / "loss_history.jsonl"
    if loss_history.exists():
        raise FileExistsError(
            f"refusing to append to existing rank loss history: {loss_history}"
        )
    test_selection_history = args.output_dir / "test_selection_history.jsonl"
    if test_selection is not None and test_selection_history.exists():
        raise FileExistsError(
            f"refusing to append to existing test-selection history: {test_selection_history}"
        )
    started = time.time()
    global_step = 0
    last_loss = None
    best_selection_key: tuple[float, ...] | None = None
    best_selection_record: dict[str, Any] | None = None
    for epoch in range(args.epochs):
        order = torch.randperm(
            training_item_count, generator=torch.Generator().manual_seed(args.seed + epoch)
        ).tolist()
        negative_rng = random.Random((args.seed + 1) * 1_000_003 + epoch)
        epoch_loss_sum = 0.0
        epoch_examples = 0
        for start in range(0, len(order), args.batch_size):
            selected_item_indices = order[start : start + args.batch_size]
            selected: list[tuple[int, int, int]] = []
            if args.query_balanced_multi_positive:
                selected_positive_counts = [
                    len(positive_positions_by_row[row_index])
                    for row_index in selected_item_indices
                ]
                pair_weights = _query_balanced_pair_weights(selected_positive_counts)
                for row_index in selected_item_indices:
                    for positive_position in positive_positions_by_row[row_index]:
                        negative_position = negative_rng.choice(negative_positions[row_index])
                        selected.append((row_index, positive_position, negative_position))
                effective_item_count = len(selected_item_indices)
            else:
                pair_weights = None
                for pair_index in selected_item_indices:
                    row_index, positive_position = pairs[pair_index]
                    negative_position = negative_rng.choice(negative_positions[row_index])
                    selected.append((row_index, positive_position, negative_position))
                effective_item_count = len(selected)
            query_texts = [
                "query: " + decoder_source_text(rows[row_index]).strip()
                for row_index, _, _ in selected
            ]
            # Keep separate positive and negative query forwards, matching the
            # official collator/model path and therefore independent dropout.
            tokenized = tokenizer(
                [*query_texts, *query_texts],
                padding=True,
                truncation=True,
                max_length=args.max_source_length,
                return_tensors="pt",
            )
            positive_skill_indices = torch.tensor(
                [
                    int(unit_indices[row_index, positive_position])
                    for row_index, positive_position, _ in selected
                ],
                dtype=torch.long,
            )
            negative_skill_indices = torch.tensor(
                [
                    int(unit_indices[row_index, negative_position])
                    for row_index, _, negative_position in selected
                ],
                dtype=torch.long,
            )
            positive_codes = semantic_code_rows[positive_skill_indices]
            negative_codes = semantic_code_rows[negative_skill_indices]
            teacher_positive = torch.tensor(
                [
                    float(unit_scores[row_index, positive_position])
                    for row_index, positive_position, _ in selected
                ],
                dtype=torch.float32,
            )
            teacher_negative = torch.tensor(
                [
                    float(unit_scores[row_index, negative_position])
                    for row_index, _, negative_position in selected
                ],
                dtype=torch.float32,
            )
            batch_size = len(selected)
            accumulated_loss = 0.0
            component_loss_sums: dict[int, float] = defaultdict(float)
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, batch_size, args.microbatch_size):
                micro_end = min(micro_start + args.microbatch_size, batch_size)
                micro_size = micro_end - micro_start
                # Tokenize the complete effective batch once so every microbatch
                # retains the same padding/truncation result.  Slice q+ and q-
                # together to preserve the official independent-query layout.
                input_ids = torch.cat(
                    [
                        tokenized["input_ids"][micro_start:micro_end],
                        tokenized["input_ids"][
                            batch_size + micro_start : batch_size + micro_end
                        ],
                    ],
                    dim=0,
                ).to(device)
                attention_mask = torch.cat(
                    [
                        tokenized["attention_mask"][micro_start:micro_end],
                        tokenized["attention_mask"][
                            batch_size + micro_start : batch_size + micro_end
                        ],
                    ],
                    dim=0,
                ).to(device)
                combined_codes = torch.cat(
                    [
                        positive_codes[micro_start:micro_end],
                        negative_codes[micro_start:micro_end],
                    ],
                    dim=0,
                ).to(device)
                micro_teacher_positive = teacher_positive[micro_start:micro_end].to(device)
                micro_teacher_negative = teacher_negative[micro_start:micro_end].to(device)
                micro_pair_weights = (
                    pair_weights[micro_start:micro_end].to(device)
                    if pair_weights is not None
                    else None
                )
                with _autocast(device, args.bf16):
                    encoder_hidden = model.encode_source(input_ids, attention_mask)
                    cumulative_scores = sequence_conditional_logit_prefix_scores(
                        model,
                        encoder_hidden,
                        attention_mask,
                        combined_codes,
                        codebook_size=codebook_size,
                        depth=depth,
                        score_length=args.active_max_prefix,
                    )
                    positive_all = cumulative_scores[:micro_size]
                    negative_all = cumulative_scores[micro_size:]
                    if progressive:
                        stage_prefix_lengths = [
                            length
                            for length in prefix_lengths
                            if length <= args.active_max_prefix
                        ]
                        selected_columns = [
                            length - 1 for length in stage_prefix_lengths
                        ]
                        micro_loss, micro_component_losses = ripor_progressive_loss(
                            positive_all[:, selected_columns],
                            negative_all[:, selected_columns],
                            micro_teacher_positive,
                            micro_teacher_negative,
                            prefix_lengths=stage_prefix_lengths,
                            active_prefix_lengths=active_prefix_lengths,
                            alphas=alphas,
                            pair_weights=micro_pair_weights,
                        )
                    else:
                        if micro_pair_weights is None:
                            micro_loss = ripor_margin_mse(
                                positive_all[:, depth - 1],
                                negative_all[:, depth - 1],
                                micro_teacher_positive,
                                micro_teacher_negative,
                            )
                        else:
                            micro_loss = query_balanced_ripor_margin_mse(
                                positive_all[:, depth - 1],
                                negative_all[:, depth - 1],
                                micro_teacher_positive,
                                micro_teacher_negative,
                                micro_pair_weights,
                            )
                        micro_component_losses = {depth: micro_loss}
                    scaled_loss = (
                        micro_loss
                        if micro_pair_weights is not None
                        else _scale_microbatch_mean_loss(
                            micro_loss,
                            microbatch_size=micro_size,
                            effective_batch_size=batch_size,
                        )
                    )
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError(
                        f"non-finite rank microbatch loss: {float(scaled_loss.detach())}"
                    )
                scaled_loss.backward()
                accumulated_loss += float(micro_loss.detach()) * (
                    1 if micro_pair_weights is not None else micro_size
                )
                for length, value in micro_component_losses.items():
                    component_loss_sums[length] += float(value.detach()) * (
                        1 if micro_pair_weights is not None else micro_size
                    )
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(torch.as_tensor(grad_norm_tensor)):
                raise FloatingPointError("non-finite rank gradient norm")
            grad_norm = float(torch.as_tensor(grad_norm_tensor))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            loss_denominator = 1 if pair_weights is not None else batch_size
            last_loss = accumulated_loss / loss_denominator
            component_losses = {
                length: value / loss_denominator
                for length, value in component_loss_sums.items()
            }
            epoch_loss_sum += last_loss * effective_item_count
            epoch_examples += effective_item_count
            if global_step % args.log_every == 0 or global_step == total_steps:
                event = {
                    "stage": args.stage_name,
                    "epoch": epoch + 1,
                    "step": global_step,
                    "total_steps": total_steps,
                    "train_loss": last_loss,
                    "component_losses": {
                        str(length): float(value)
                        for length, value in component_losses.items()
                    },
                    "grad_norm": grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.time() - started,
                }
                _append_jsonl(loss_history, event)
                print(json.dumps(event, sort_keys=True), flush=True)
        _append_jsonl(
            loss_history,
            {
                "stage": args.stage_name,
                "epoch": epoch + 1,
                "step": global_step,
                "epoch_train_loss": epoch_loss_sum / epoch_examples,
                "elapsed_seconds": time.time() - started,
            },
        )
        best_selection_key, selection_record = _run_epoch_test_selection(
            model,
            tokenizer,
            semantic_ids,
            skill_ids,
            test_selection,
            epoch=epoch + 1,
            stage=args.stage_name,
            base_model=str(parent_checkpoint["model"]),
            codebook_size=codebook_size,
            depth=depth,
            seed=args.seed,
            device=device,
            bf16=args.bf16,
            output_dir=args.output_dir,
            best_key=best_selection_key,
        )
        if selection_record is not None and selection_record["selected"]:
            best_selection_record = selection_record
    if test_selection is not None:
        if best_selection_record is None:
            raise AssertionError(
                f"test checkpoint selection did not select a {args.stage_name} epoch"
            )
        _restore_test_selected_best(model, args.output_dir)
    config = {
        "seed": args.seed,
        "rows_path": str(args.rows),
        "rows_sha256": file_sha256(args.rows),
        "rows_fingerprint_sha256": dgr_rows_fingerprint(rows),
        "skills_jsonl": str(args.skills_jsonl),
        "skills_jsonl_sha256": file_sha256(args.skills_jsonl),
        "rq_dir": str(args.rq_dir),
        "rq_manifest_sha256": file_sha256(args.rq_dir / "manifest.json"),
        "parent_model_dir": str(args.parent_model_dir),
        "parent_checkpoint_sha256": file_sha256(
            args.parent_model_dir / "structured_decoder.pt"
        ),
        "parent_stage": parent_checkpoint.get("ripor_stage"),
        "candidates_dir": str(args.candidates_dir),
        "candidates_manifest_sha256": file_sha256(
            args.candidates_dir / "manifest.json"
        ),
        "candidate_source": candidate_manifest.get("candidate_source"),
        "teacher_model": candidate_manifest.get("teacher_model"),
        "teacher_score_policy": RIPOR_TEACHER_POLICY,
        "teacher_margin_sign_policy": "signed_unclipped",
        "candidate_unit": f"unique_prefix_at_length_{args.active_max_prefix}",
        "prefix_teacher_aggregation": "max_skill_score_within_active_prefix",
        "positive_policy": (
            "all_distinct_gt_prefixes_jointly_per_query_all_gt_prefixes_excluded_from_negatives"
            if args.query_balanced_multi_positive
            else "one_example_per_gt_all_gt_prefixes_excluded_from_negatives"
        ),
        "negative_sampling": "uniform_one_active_prefix_per_positive_per_getitem_equivalent",
        "loss": (
            "query_balanced_multi_positive_conditional_logit_signed_margin_mse"
            if args.query_balanced_multi_positive
            else "conditional_logit_signed_margin_mse"
        ),
        "loss_averaging": (
            "mean_over_distinct_positive_units_then_mean_over_source_queries"
            if args.query_balanced_multi_positive
            else "mean_over_expanded_gt_pairs"
        ),
        "query_balanced_multi_positive": bool(args.query_balanced_multi_positive),
        "generation_ce": False,
        "stop_loss": False,
        "progressive_multi_objective": progressive,
        "prefix_beta": args.prefix_beta,
        "prefix_lengths": prefix_lengths,
        "prefix_alphas": {str(key): value for key, value in alphas.items()},
        "active_prefix_lengths": active_prefix_lengths,
        "active_max_prefix": args.active_max_prefix,
        "rank_unit_audit": unit_audit,
        "examples": training_item_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "microbatch_gradient_accumulation": True,
        "microbatch_loss_scaling": (
            "exact_query_then_positive_weighted_effective_batch_mean"
            if args.query_balanced_multi_positive
            else "size_weighted_exact_effective_batch_mean"
        ),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "max_grad_norm": args.max_grad_norm,
        "max_source_length": args.max_source_length,
        "rank_query_text_policy": RIPOR_RANK_QUERY_TEXT_POLICY,
        "bf16": bool(args.bf16),
        "final_loss": last_loss,
        "checkpoint_selection_split": (
            "Official core75 test" if test_selection is not None else "final_epoch"
        ),
        "checkpoint_selection_rule": (
            TEST_SELECTION_RULE if test_selection is not None else "final_epoch"
        ),
        "checkpoint_selection_epoch": (
            int(best_selection_record["epoch"])
            if best_selection_record is not None
            else args.epochs
        ),
        "checkpoint_selection_key": (
            best_selection_record["selection_key"]
            if best_selection_record is not None
            else None
        ),
        "checkpoint_selection_metrics": (
            best_selection_record["metrics"]
            if best_selection_record is not None
            else None
        ),
        "test_selection": (
            {key: value for key, value in test_selection.items() if key != "rows"}
            if test_selection is not None
            else None
        ),
        "elapsed_seconds": time.time() - started,
    }
    _save_ripor_model_checkpoint(
        args.output_dir / "final",
        model,
        tokenizer,
        base_model=str(parent_checkpoint["model"]),
        semantic_ids=semantic_ids,
        codebook_size=codebook_size,
        depth=depth,
        seed=args.seed,
        stage=args.stage_name,
        extra=config,
    )
    write_json(args.output_dir / "train_manifest.json", {"status": "complete", **config})


def build_ripor_semantic_trie_groups(
    semantic_ids: dict[str, list[int]],
) -> tuple[
    dict[tuple[int, ...], tuple[int, ...]],
    dict[tuple[int, ...], tuple[str, ...]],
    list[int],
]:
    """Build the RIPOR trie while retaining every skill assigned to one SID."""
    if not semantic_ids:
        raise ValueError("semantic_ids is empty")
    lengths = {len(values) for values in semantic_ids.values()}
    if len(lengths) != 1:
        raise ValueError(f"semantic IDs have inconsistent lengths: {sorted(lengths)}")
    depth = next(iter(lengths))
    if depth <= 0:
        raise ValueError("semantic IDs are empty")
    children: dict[tuple[int, ...], set[int]] = {}
    sequence_to_skills: dict[tuple[int, ...], list[str]] = {}
    for sid, raw_values in semantic_ids.items():
        values = tuple(int(value) for value in raw_values)
        sequence_to_skills.setdefault(values, []).append(sid)
        for level in range(depth):
            children.setdefault(values[:level], set()).add(values[level])
    prefix_counts = [
        len({values[:prefix_length] for values in sequence_to_skills})
        for prefix_length in range(1, depth + 1)
    ]
    return (
        {prefix: tuple(sorted(values)) for prefix, values in children.items()},
        {values: tuple(skills) for values, skills in sequence_to_skills.items()},
        prefix_counts,
    )


def expand_sid_beam_to_skill_indices(
    sid_sequences: Sequence[Sequence[int]],
    sid_scores: Sequence[float],
    sequence_to_skill_indices: dict[tuple[int, ...], tuple[int, ...]],
) -> tuple[list[int], list[float]]:
    """Expand ranked SIDs exactly like official RIPOR's ``smtid_to_docids``."""
    if len(sid_sequences) != len(sid_scores):
        raise ValueError("decoded SID sequences and scores have different lengths")
    result_indices: list[int] = []
    result_scores: list[float] = []
    seen_sequences: set[tuple[int, ...]] = set()
    seen_indices: set[int] = set()
    for raw_sequence, raw_score in zip(sid_sequences, sid_scores):
        sequence = tuple(int(value) for value in raw_sequence)
        if sequence in seen_sequences:
            raise ValueError("trie decoding returned a duplicate full SID")
        seen_sequences.add(sequence)
        if sequence not in sequence_to_skill_indices:
            raise ValueError(f"trie decoding returned an unknown full SID: {sequence}")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("trie decoding returned a non-finite score")
        for index in sequence_to_skill_indices[sequence]:
            index = int(index)
            if index in seen_indices:
                raise ValueError("SID expansion returned a duplicate skill index")
            seen_indices.add(index)
            result_indices.append(index)
            result_scores.append(score)
    return result_indices, result_scores


@torch.no_grad()
def _decode_ripor_rows(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    semantic_ids: dict[str, list[int]],
    *,
    codebook_size: int,
    depth: int,
    device: torch.device,
    top_k: int,
    query_batch_size: int,
    decoder_batch_size: int,
    max_source_length: int,
    bf16: bool,
) -> tuple[list[list[int]], list[list[float]], list[str], list[int]]:
    trie_children, sequence_to_skills, prefix_counts = build_ripor_semantic_trie_groups(
        semantic_ids
    )
    # Build in the exact catalog order supplied by the caller.
    skill_order = list(semantic_ids)
    skill_to_index = {sid: index for index, sid in enumerate(skill_order)}
    sequence_to_indices = {
        sequence: tuple(skill_to_index[sid] for sid in skills)
        for sequence, skills in sequence_to_skills.items()
    }
    trie_decode_tables = build_trie_decode_tables(
        trie_children,
        codebook_size=codebook_size,
        device=device,
    )
    all_indices: list[list[int]] = []
    all_scores: list[list[float]] = []
    started = time.time()
    model.eval()
    for start in range(0, len(rows), query_batch_size):
        end = min(start + query_batch_size, len(rows))
        tokenized = tokenizer(
            ["query: " + decoder_source_text(row).strip() for row in rows[start:end]],
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with _autocast(device, bf16):
            encoder_hidden = model.encode_source(input_ids, attention_mask)
        prefixes, scores = topk_single_sid_scores_from_encoder(
            model,
            encoder_hidden,
            attention_mask,
            trie_children,
            prefix_counts,
            codebook_size=codebook_size,
            depth=depth,
            dedup_size=0,
            semantic_id_mode=SEMANTIC_ID_MODE_LEARNED,
            top_k=top_k,
            decoder_batch_size=decoder_batch_size,
            include_stop=False,
            score_mode=RIPOR_SCORE_MODE,
            trie_decode_tables=trie_decode_tables,
        )
        for query_prefixes, query_scores in zip(
            prefixes.cpu().tolist(), scores.cpu().tolist()
        ):
            indices, expanded_scores = expand_sid_beam_to_skill_indices(
                query_prefixes,
                query_scores,
                sequence_to_indices,
            )
            all_indices.append(indices)
            all_scores.append(expanded_scores)
        print(
            f"decoded RIPOR queries {end}/{len(rows)} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    return all_indices, all_scores, skill_order, prefix_counts


def command_mine_beam_candidates(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    _, skill_ids, _ = _load_skills(args.skills_jsonl)
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    _validate_rows(rows, skill_to_index)
    semantic_ids, _, rq_manifest = _load_rq_artifacts(args.rq_dir, args.skills_jsonl)
    if list(semantic_ids) != skill_ids:
        # The decoder helper can use any order; mined artifact must use catalog order.
        semantic_ids = {sid: semantic_ids[sid] for sid in skill_ids}
    device = _device(args.cpu)
    model, tokenizer, checkpoint = _load_ripor_model_checkpoint(
        args.model_dir, semantic_ids, device
    )
    if args.expected_stage and checkpoint.get("ripor_stage") != args.expected_stage:
        raise ValueError(
            f"beam mining expected {args.expected_stage}, got {checkpoint.get('ripor_stage')}"
        )
    decoded_indices, _, decoded_skill_order, prefix_counts = _decode_ripor_rows(
        model,
        tokenizer,
        rows,
        semantic_ids,
        codebook_size=int(rq_manifest["codebook_size"]),
        depth=int(rq_manifest["depth"]),
        device=device,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size,
        decoder_batch_size=args.decoder_batch_size,
        max_source_length=args.max_source_length,
        bf16=args.bf16,
    )
    if decoded_skill_order != skill_ids:
        raise AssertionError("decoder semantic-ID order changed during mining")
    candidate_rows: list[list[int]] = []
    for row, retrieved in zip(rows, decoded_indices):
        gold_indices = [skill_to_index[str(sid)] for sid in row["selected_skill_ids"]]
        candidate_rows.append(
            _candidate_row_with_gold_after_sid_expansion(
                retrieved,
                gold_indices,
                retrieved_sid_count=args.top_k,
            )
        )
    _write_candidate_artifact(
        args.output_dir,
        rows,
        skill_ids,
        candidate_rows,
        manifest_extra={
            "candidate_source": "ripor_trie_beam_top_k_sids_expanded_to_all_skills_with_all_gt_prepended",
            "retrieval_top_k_sids": args.top_k,
            "collision_expansion_policy": "official_smtid_to_docids_catalog_order",
            "retrieved_skill_count_after_sid_expansion": {
                "min": min(len(values) for values in decoded_indices),
                "max": max(len(values) for values in decoded_indices),
                "mean": sum(len(values) for values in decoded_indices) / len(decoded_indices),
            },
            "model_dir": str(args.model_dir),
            "model_checkpoint_sha256": file_sha256(
                args.model_dir / "structured_decoder.pt"
            ),
            "ripor_stage": checkpoint.get("ripor_stage"),
            "decoder_score_mode": RIPOR_SCORE_MODE,
            "include_stop": False,
            "prefix_counts": prefix_counts,
            "query_batch_size": args.query_batch_size,
            "decoder_batch_size": args.decoder_batch_size,
            "max_source_length": args.max_source_length,
            "bf16": bool(args.bf16),
        },
        shard_size=args.shard_size,
    )


def command_eval(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    _, skill_ids, _ = _load_skills(args.skills_jsonl)
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    _validate_rows(rows, skill_to_index)
    rows_sha256 = _require_file_sha256(
        args.rows, args.expected_rows_sha256, "Official evaluation rows"
    )
    skills_sha256 = _require_file_sha256(
        args.skills_jsonl, args.expected_skills_sha256, "Official skill catalog"
    )
    semantic_ids, _, rq_manifest = _load_rq_artifacts(args.rq_dir, args.skills_jsonl)
    semantic_ids = {sid: semantic_ids[sid] for sid in skill_ids}
    device = _device(args.cpu)
    model, tokenizer, checkpoint = _load_ripor_model_checkpoint(
        args.model_dir, semantic_ids, device
    )
    if checkpoint.get("ripor_stage") != "M4":
        raise ValueError(
            f"formal RIPOR evaluation requires final M4, got {checkpoint.get('ripor_stage')}"
        )
    decoded_indices, decoded_scores, decoded_skill_order, prefix_counts = _decode_ripor_rows(
        model,
        tokenizer,
        rows,
        semantic_ids,
        codebook_size=int(rq_manifest["codebook_size"]),
        depth=int(rq_manifest["depth"]),
        device=device,
        top_k=args.num_beams,
        query_batch_size=args.query_batch_size,
        decoder_batch_size=args.decoder_batch_size,
        max_source_length=args.max_source_length,
        bf16=args.bf16,
    )
    if decoded_skill_order != skill_ids:
        raise AssertionError("evaluation decoder order differs from the canonical catalog")
    predictions: list[dict[str, Any]] = []
    for row, indices, scores in zip(rows, decoded_indices, decoded_scores):
        predictions.append(
            {
                "index": row.get("index"),
                "source_index": row.get("source_index"),
                "target_skill_count": row.get("target_skill_count"),
                "query": row.get("query"),
                "gold_skill_ids": [str(value) for value in row["selected_skill_ids"]],
                "ranked_skill_ids": [skill_ids[index] for index in indices],
                "ranked_skill_scores": scores,
                "invalid": False,
            }
        )
    k_values = sorted({int(value) for value in args.k_values.split(",")})
    if max(k_values) > args.num_beams:
        raise ValueError("evaluation K cannot exceed num_beams")
    if any(len(values) < max(k_values) for values in decoded_indices):
        raise ValueError("SID expansion returned fewer skills than the largest evaluation K")
    metrics = compute_at_k_metrics(rows, predictions, k_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "ranked_metrics.json", metrics)
    write_json(
        args.output_dir / "eval_manifest.json",
        {
            "status": "complete",
            "evaluation": "SkillsBench_v1.1_core75_official",
            "rows": str(args.rows),
            "rows_sha256": rows_sha256,
            "skills_jsonl": str(args.skills_jsonl),
            "skills_jsonl_sha256": skills_sha256,
            "candidate_skills": len(skill_ids),
            "model_dir": str(args.model_dir),
            "model_checkpoint_sha256": file_sha256(
                args.model_dir / "structured_decoder.pt"
            ),
            "ripor_stage": checkpoint.get("ripor_stage"),
            "rq_manifest_sha256": file_sha256(args.rq_dir / "manifest.json"),
            "semantic_ids_sha256": _ripor_semantic_ids_sha256(semantic_ids),
            "num_beams": args.num_beams,
            "retrieval_top_k_sids": args.num_beams,
            "decoder_score_mode": RIPOR_SCORE_MODE,
            "include_stop": False,
            "rank_aggregation": "official_sid_beam_then_catalog_order_skill_expansion",
            "collision_expansion_policy": "official_smtid_to_docids_catalog_order",
            "rq_collision_groups": int(rq_manifest["collision_groups"]),
            "rq_collision_skills": int(rq_manifest["collision_skills"]),
            "ranked_skill_count_after_sid_expansion": {
                "min": min(len(values) for values in decoded_indices),
                "max": max(len(values) for values in decoded_indices),
                "mean": sum(len(values) for values in decoded_indices) / len(decoded_indices),
            },
            "k_values": k_values,
            "query_batch_size": args.query_batch_size,
            "decoder_batch_size": args.decoder_batch_size,
            "max_source_length": args.max_source_length,
            "bf16": bool(args.bf16),
            "prefix_counts": prefix_counts,
            "metrics": metrics,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_rows_skills(p: argparse.ArgumentParser) -> None:
        p.add_argument("--rows", type=Path, required=True)
        p.add_argument("--skills-jsonl", type=Path, required=True)

    def add_runtime(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--bf16", action="store_true")

    def add_candidate_io(p: argparse.ArgumentParser) -> None:
        add_rows_skills(p)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--shard-size", type=int, default=4096)

    def add_training(p: argparse.ArgumentParser) -> None:
        p.add_argument("--epochs", type=int, required=True)
        p.add_argument("--batch-size", type=int, required=True)
        p.add_argument("--lr", type=float, required=True)
        p.add_argument("--weight-decay", type=float, default=0.0)
        p.add_argument("--warmup-ratio", type=float, default=0.045)
        p.add_argument("--max-grad-norm", type=float, default=1.0)
        p.add_argument("--max-source-length", type=int, required=True)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--log-every", type=int, default=100)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--test-selection-rows", type=Path)
        p.add_argument("--test-selection-expected-rows-sha256")
        p.add_argument("--test-selection-expected-skills-sha256")
        p.add_argument("--test-selection-num-beams", type=int, default=50)
        p.add_argument("--test-selection-query-batch-size", type=int, default=1)
        p.add_argument("--test-selection-decoder-batch-size", type=int, default=1024)
        p.add_argument("--test-selection-max-source-length", type=int, default=768)
        add_runtime(p)

    p = sub.add_parser("audit-inputs")
    add_rows_skills(p)
    p.add_argument("--base-model", default="t5-base")
    p.add_argument("--query-limit", type=int, required=True)
    p.add_argument("--skill-limit", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=command_audit_inputs)

    p = sub.add_parser("build-bm25-candidates")
    add_candidate_io(p)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--expected-rows-sha256")
    p.add_argument("--expected-skills-sha256")
    p.add_argument("--log-every", type=int, default=1000)
    p.set_defaults(func=command_build_bm25_candidates)

    p = sub.add_parser("annotate-candidates")
    add_candidate_io(p)
    p.add_argument("--candidates-dir", type=Path, required=True)
    p.add_argument("--teacher-query-embeddings-dir", type=Path, required=True)
    p.add_argument("--teacher-skill-embeddings", type=Path, required=True)
    p.add_argument("--teacher-skill-manifest", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--log-every", type=int, default=8192)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_annotate_candidates)

    p = sub.add_parser("union-candidates")
    add_candidate_io(p)
    p.add_argument("--first-candidates-dir", type=Path, required=True)
    p.add_argument("--second-candidates-dir", type=Path, required=True)
    p.set_defaults(func=command_union_candidates)

    p = sub.add_parser("train-m0")
    add_rows_skills(p)
    add_training(p)
    p.add_argument("--stage-name", choices=["M0_bm25", "M0_dense"], required=True)
    p.add_argument("--base-model", default="t5-base")
    p.add_argument("--decoder-start-token", type=Path, required=True)
    p.add_argument("--candidates-dir", type=Path, required=True)
    p.add_argument("--parent-model-dir", type=Path)
    p.add_argument("--microbatch-size", type=int, default=8)
    p.add_argument("--tokenize-batch-size", type=int, default=512)
    p.set_defaults(func=command_train_m0)

    p = sub.add_parser("mine-dense-candidates")
    add_candidate_io(p)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--decoder-start-token", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--query-batch-size", type=int, default=64)
    p.add_argument("--skill-batch-size", type=int, default=64)
    p.add_argument("--max-source-length", type=int, default=128)
    add_runtime(p)
    p.set_defaults(func=command_mine_dense_candidates)

    p = sub.add_parser("build-rq")
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--embeddings-manifest", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--faiss-threads", type=int, default=32)
    p.add_argument("--cf-embeddings", type=Path)
    p.add_argument("--cf-manifest", type=Path)
    p.add_argument("--cf-skill-ids", type=Path)
    p.add_argument("--cf-weight", type=float, default=0.5)
    p.add_argument("--qwen-embeddings", type=Path)
    p.add_argument("--qwen-manifest", type=Path)
    p.add_argument("--m0-weight", type=float, default=0.5)
    p.add_argument("--qwen-weight", type=float, default=0.3)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=command_build_rq)

    p = sub.add_parser("train-m1")
    add_rows_skills(p)
    add_training(p)
    p.add_argument("--base-model", default="t5-base")
    p.add_argument("--m0-model-dir", type=Path, required=True)
    p.add_argument("--decoder-start-token", type=Path, required=True)
    p.add_argument("--rq-dir", type=Path, required=True)
    p.set_defaults(func=command_train_m1)

    p = sub.add_parser("train-rank")
    add_rows_skills(p)
    add_training(p)
    p.add_argument(
        "--stage-name",
        choices=["M2", "M3_prefix4", "M3_full", "M4"],
        required=True,
    )
    p.add_argument("--parent-model-dir", type=Path, required=True)
    p.add_argument("--rq-dir", type=Path, required=True)
    p.add_argument("--candidates-dir", type=Path, required=True)
    p.add_argument("--active-max-prefix", type=int, required=True)
    p.add_argument("--prefix-beta", type=float, default=DEFAULT_PREFIX_BETA)
    p.add_argument("--microbatch-size", type=int, default=32)
    p.add_argument(
        "--query-balanced-multi-positive",
        action="store_true",
        help=(
            "batch source queries, include every distinct gold active-prefix unit, "
            "and average MarginMSE within query before averaging across queries"
        ),
    )
    p.set_defaults(func=command_train_rank)

    p = sub.add_parser("mine-beam-candidates")
    add_candidate_io(p)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--rq-dir", type=Path, required=True)
    p.add_argument("--expected-stage", choices=["M2", "M3_full"])
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--query-batch-size", type=int, default=4)
    p.add_argument("--decoder-batch-size", type=int, default=1024)
    p.add_argument("--max-source-length", type=int, default=64)
    add_runtime(p)
    p.set_defaults(func=command_mine_beam_candidates)

    p = sub.add_parser("eval")
    add_rows_skills(p)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--rq-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-beams", type=int, default=50)
    p.add_argument("--k-values", default="1,5,10,50")
    p.add_argument("--query-batch-size", type=int, default=1)
    p.add_argument("--decoder-batch-size", type=int, default=1024)
    p.add_argument("--max-source-length", type=int, default=64)
    p.add_argument("--expected-rows-sha256")
    p.add_argument("--expected-skills-sha256")
    add_runtime(p)
    p.set_defaults(func=command_eval)

    return parser


def _validate_common_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "shard_size",
        "query_batch_size",
        "skill_batch_size",
        "decoder_batch_size",
        "test_selection_query_batch_size",
        "test_selection_decoder_batch_size",
    ):
        value = getattr(args, name, None)
        if value is not None and int(value) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    if hasattr(args, "epochs") and args.epochs < 1:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if hasattr(args, "lr") and args.lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {args.lr}")
    if hasattr(args, "weight_decay") and args.weight_decay < 0.0:
        raise ValueError(f"--weight-decay must be non-negative, got {args.weight_decay}")
    if hasattr(args, "warmup_ratio") and not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError(f"--warmup-ratio must be in [0, 1), got {args.warmup_ratio}")
    if hasattr(args, "max_source_length") and args.max_source_length < 1:
        raise ValueError("--max-source-length must be positive")
    if getattr(args, "test_selection_rows", None) is not None:
        if args.test_selection_num_beams != 50:
            raise ValueError("Official test checkpoint selection requires beam=50")
        if args.test_selection_max_source_length < 1:
            raise ValueError("--test-selection-max-source-length must be positive")


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_common_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
