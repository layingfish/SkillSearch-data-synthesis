#!/usr/bin/env python3
"""Frozen c1 candidates and cumulative-prefix ranking loss for the legacy decoder."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from analyze_legacy_c1_hard_negative_coverage import file_sha256, first_divergence
except ModuleNotFoundError:
    from scripts.analyze_legacy_c1_hard_negative_coverage import (
        file_sha256,
        first_divergence,
    )


SCOPE_DEPTHS = {
    "disabled": (),
    "c1_only": (2,),
    "sequence_only": (4,),
    "cumulative": (2, 3, 4),
}
TRAIN_MARKER = "_legacy_prefix_rank_train_row"
NEGATIVE_ID_FIELD = "_legacy_prefix_rank_negative_skill_id"
CANDIDATE_KEY_FIELD = "_legacy_prefix_rank_candidate_key"


def scalar_key(value: Any) -> str:
    if value is None or isinstance(value, (str, int, float, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise ValueError(f"candidate index must be a JSON scalar, got {type(value).__name__}")


def query_sha256(row: Mapping[str, Any]) -> str:
    value = row.get("query")
    if value is None and isinstance(row.get("generated"), Mapping):
        value = row["generated"].get("query")
    query = str(value or "")
    if not query:
        raise ValueError(f"index={row.get('index')}: empty query")
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def candidate_key(row: Mapping[str, Any], gold_skill_id: str) -> str:
    return "\x1f".join(
        (scalar_key(row.get("index")), query_sha256(row), str(gold_skill_id))
    )


def active_depths(scope: str, semantic_depth: int) -> tuple[int, ...]:
    if scope not in SCOPE_DEPTHS:
        raise ValueError(f"unknown prefix ranking scope: {scope}")
    depths = SCOPE_DEPTHS[scope]
    if any(depth <= 0 or depth > semantic_depth for depth in depths):
        raise ValueError(
            f"scope={scope} is incompatible with semantic depth={semantic_depth}"
        )
    return depths


@dataclass
class FrozenC1Candidates:
    negative_by_key: dict[str, str | None]
    semantic_ids: dict[str, tuple[int, ...]]
    source_rows: int
    gold_mentions: int
    selected_mentions: int
    candidate_limit: int
    manifest_sha256: str
    summary_sha256: str

    @classmethod
    def load(
        cls,
        *,
        summary_path: Path,
        manifest_path: Path,
        semantic_ids: Mapping[str, Sequence[int]],
        expected_summary_sha256: str,
        expected_manifest_sha256: str,
        expected_rows_sha256: str,
        expected_semantic_ids_sha256: str,
    ) -> "FrozenC1Candidates":
        summary_sha256 = file_sha256(summary_path)
        manifest_sha256 = file_sha256(manifest_path)
        if summary_sha256 != expected_summary_sha256:
            raise ValueError(
                f"candidate summary SHA-256 mismatch: {summary_sha256}"
            )
        if manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                f"candidate manifest SHA-256 mismatch: {manifest_sha256}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("inputs", {}).get("rows_sha256") != expected_rows_sha256:
            raise ValueError("candidate summary source-row SHA-256 mismatch")
        if (
            summary.get("inputs", {}).get("semantic_ids_sha256")
            != expected_semantic_ids_sha256
        ):
            raise ValueError("candidate summary semantic-ID SHA-256 mismatch")
        if (
            summary.get("outputs", {}).get("candidate_manifest_sha256")
            != manifest_sha256
        ):
            raise ValueError("candidate summary does not bind the manifest SHA-256")

        normalized_semantic_ids = {
            str(skill_id): tuple(int(value) for value in values)
            for skill_id, values in semantic_ids.items()
        }
        semantic_depths = {len(values) for values in normalized_semantic_ids.values()}
        if semantic_depths != {int(summary["protocol"]["semantic_depth"])}:
            raise ValueError(
                f"semantic depth mismatch: observed={sorted(semantic_depths)}"
            )

        negative_by_key: dict[str, str | None] = {}
        source_rows = 0
        gold_mentions = 0
        selected_mentions = 0
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                source_rows += 1
                row_gold = [str(value) for value in row.get("gold_skill_ids") or []]
                if not row_gold or len(set(row_gold)) != len(row_gold):
                    raise ValueError(
                        f"manifest line={line_number}: invalid source gold set"
                    )
                payloads = row.get("gold_candidates") or []
                if len(payloads) != len(row_gold):
                    raise ValueError(
                        f"manifest line={line_number}: gold payload count mismatch"
                    )
                payload_by_gold = {
                    str(payload.get("gold_skill_id")): payload for payload in payloads
                }
                if set(payload_by_gold) != set(row_gold):
                    raise ValueError(
                        f"manifest line={line_number}: gold payload IDs mismatch"
                    )
                manifest_row = {
                    "index": row.get("index"),
                    "query": "manifest-query-placeholder",
                }
                index_part = scalar_key(manifest_row["index"])
                query_hash = str(row.get("query_sha256") or "")
                if len(query_hash) != 64:
                    raise ValueError(
                        f"manifest line={line_number}: invalid query SHA-256"
                    )
                for gold_skill_id in row_gold:
                    gold_mentions += 1
                    gold_codes = normalized_semantic_ids.get(gold_skill_id)
                    if gold_codes is None:
                        raise ValueError(
                            f"manifest line={line_number}: missing gold semantic ID"
                        )
                    payload = payload_by_gold[gold_skill_id]
                    stored_gold_codes = tuple(
                        int(value) for value in payload.get("gold_semantic_id") or []
                    )
                    if stored_gold_codes != gold_codes:
                        raise ValueError(
                            f"manifest line={line_number}: gold semantic ID drift"
                        )
                    selected = payload.get("selected_c1_negative")
                    negative_skill_id = None
                    if selected is not None:
                        negative_skill_id = str(selected.get("skill_id") or "")
                        negative_codes = normalized_semantic_ids.get(negative_skill_id)
                        if not negative_skill_id or negative_codes is None:
                            raise ValueError(
                                f"manifest line={line_number}: missing negative semantic ID"
                            )
                        if negative_skill_id in row_gold:
                            raise ValueError(
                                f"manifest line={line_number}: sibling GT selected as negative"
                            )
                        stored_negative_codes = tuple(
                            int(value) for value in selected.get("semantic_id") or []
                        )
                        if stored_negative_codes != negative_codes:
                            raise ValueError(
                                f"manifest line={line_number}: negative semantic ID drift"
                            )
                        if first_divergence(gold_codes, negative_codes) != 1:
                            raise ValueError(
                                f"manifest line={line_number}: negative does not first diverge at c1"
                            )
                        rank = int(selected.get("teacher_rank", 0))
                        if not 1 <= rank <= int(summary["protocol"]["candidate_limit"]):
                            raise ValueError(
                                f"manifest line={line_number}: invalid teacher rank={rank}"
                            )
                        selected_mentions += 1
                    key = "\x1f".join((index_part, query_hash, gold_skill_id))
                    if key in negative_by_key:
                        raise ValueError(
                            f"manifest line={line_number}: duplicate candidate key"
                        )
                    negative_by_key[key] = negative_skill_id

        counts = summary.get("counts", {})
        expected_counts = {
            "rows": source_rows,
            "gold_mentions": gold_mentions,
            "selected_c1_gold_mentions": selected_mentions,
        }
        observed_counts = {key: counts.get(key) for key in expected_counts}
        if observed_counts != expected_counts:
            raise ValueError(
                f"candidate summary count mismatch: expected={expected_counts}, "
                f"observed={observed_counts}"
            )
        return cls(
            negative_by_key=negative_by_key,
            semantic_ids=normalized_semantic_ids,
            source_rows=source_rows,
            gold_mentions=gold_mentions,
            selected_mentions=selected_mentions,
            candidate_limit=int(summary["protocol"]["candidate_limit"]),
            manifest_sha256=manifest_sha256,
            summary_sha256=summary_sha256,
        )


@dataclass
class PrefixRankingRuntime:
    candidates: FrozenC1Candidates
    scope: str
    epochs: int
    expander_calls: int = 0
    query_train_batch_calls: int = 0
    train_presentations: Counter[str] = field(default_factory=Counter)
    selected_pair_presentations: int = 0
    target_layout_verified: bool = False

    @property
    def depths(self) -> tuple[int, ...]:
        semantic_depth = len(next(iter(self.candidates.semantic_ids.values())))
        return active_depths(self.scope, semantic_depth)

    def wrap_expander(self, original_expander: Callable[..., Any]) -> Callable[..., Any]:
        def controlled_expander(*args: Any, **kwargs: Any) -> Any:
            rows = original_expander(*args, **kwargs)
            self.expander_calls += 1
            is_train = self.expander_calls == 1
            if self.expander_calls > 2:
                raise RuntimeError("unexpected third single-SID expansion call")
            observed_keys: set[str] = set()
            for row in rows:
                row[TRAIN_MARKER] = is_train
                if not is_train:
                    continue
                gold_skill_id = str(row.get("positive_skill_id") or "")
                key = candidate_key(row, gold_skill_id)
                if key not in self.candidates.negative_by_key:
                    raise ValueError(
                        f"expanded train row is absent from frozen candidates: key={key}"
                    )
                if key in observed_keys:
                    raise ValueError(f"duplicate expanded train candidate key={key}")
                observed_keys.add(key)
                row[CANDIDATE_KEY_FIELD] = key
                row[NEGATIVE_ID_FIELD] = self.candidates.negative_by_key[key]
            if is_train:
                expected_keys = set(self.candidates.negative_by_key)
                if observed_keys != expected_keys:
                    raise ValueError(
                        "expanded train rows do not close over the frozen candidate manifest"
                    )
            return rows

        return controlled_expander

    def wrap_batch_builder(self, original_builder: Callable[..., Any]) -> Callable[..., Any]:
        def controlled_builder(
            batch_rows: list[dict[str, Any]], *args: Any, **kwargs: Any
        ) -> Any:
            batch = original_builder(batch_rows, *args, **kwargs)
            train_flags = [bool(row.get(TRAIN_MARKER, False)) for row in batch_rows]
            if not any(train_flags):
                return batch
            if not all(train_flags):
                raise ValueError("query-train and non-train rows were mixed in one batch")

            self.query_train_batch_calls += 1
            eligible_positions: list[int] = []
            negative_rows: list[dict[str, Any]] = []
            for position, row in enumerate(batch_rows):
                key = str(row.get(CANDIDATE_KEY_FIELD) or "")
                if key not in self.candidates.negative_by_key:
                    raise ValueError("query-train batch lost its candidate key")
                self.train_presentations[key] += 1
                negative_skill_id = row.get(NEGATIVE_ID_FIELD)
                if negative_skill_id is None:
                    continue
                eligible_positions.append(position)
                negative_row = dict(row)
                negative_row["selected_skill_ids"] = [str(negative_skill_id)]
                negative_row["positive_skill_id"] = str(negative_skill_id)
                negative_rows.append(negative_row)

            if not eligible_positions:
                return batch
            negative_batch = original_builder(negative_rows, *args, **kwargs)
            import torch

            eligible_indices = torch.tensor(
                eligible_positions,
                dtype=torch.long,
                device=batch["input_ids"].device,
            )
            if not self.target_layout_verified:
                expected_types = batch["target_types"].index_select(0, eligible_indices)
                if not torch.equal(negative_batch["target_types"], expected_types):
                    raise ValueError("positive/negative target-type layouts differ")
                self.target_layout_verified = True
            self.selected_pair_presentations += len(eligible_positions)
            batch.update(
                {
                    "prefix_rank_eligible_indices": eligible_indices,
                    "prefix_rank_negative_decoder_input_ids": negative_batch[
                        "decoder_input_ids"
                    ],
                    "prefix_rank_negative_decoder_attention_mask": negative_batch[
                        "decoder_attention_mask"
                    ],
                    "prefix_rank_negative_labels": negative_batch["labels"],
                    "prefix_rank_query_step": self.query_train_batch_calls,
                }
            )
            return batch

        return controlled_builder

    def validate(self) -> dict[str, Any]:
        if self.expander_calls != 2:
            raise ValueError(f"expected two expansion calls, got {self.expander_calls}")
        if set(self.train_presentations) != set(self.candidates.negative_by_key):
            raise ValueError("query training did not present every frozen candidate key")
        presentation_counts = set(self.train_presentations.values())
        if presentation_counts != {self.epochs}:
            raise ValueError(
                f"per-row query presentation mismatch: {sorted(presentation_counts)}"
            )
        expected_pairs = self.candidates.selected_mentions * self.epochs
        if self.selected_pair_presentations != expected_pairs:
            raise ValueError(
                f"eligible pair presentations={self.selected_pair_presentations} "
                f"does not match expected={expected_pairs}"
            )
        return {
            "expander_calls": self.expander_calls,
            "query_train_batch_calls": self.query_train_batch_calls,
            "query_train_row_presentations": sum(self.train_presentations.values()),
            "per_row_presentations": self.epochs,
            "eligible_pair_presentations": self.selected_pair_presentations,
            "active_comparisons": self.selected_pair_presentations * len(self.depths),
            "active_depths": list(self.depths),
            "target_layout_verified": self.target_layout_verified,
        }


def cumulative_prefix_scores(
    model: Any,
    hidden_states: Any,
    labels: Any,
    depths: Sequence[int],
) -> Any:
    import torch
    import torch.nn.functional as F

    normalized_depths = tuple(int(depth) for depth in depths)
    if not normalized_depths:
        raise ValueError("at least one active prefix depth is required")
    max_depth = max(normalized_depths)
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("hidden_states/labels must have rank 3/2")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states/labels sequence shapes differ")
    if max_depth > len(model.code_heads) or max_depth > labels.shape[1]:
        raise ValueError("active prefix depth exceeds the learned-code sequence")

    token_scores = []
    for level in range(max_depth):
        logits = model.code_heads[level](hidden_states[:, level, :])
        log_probs = F.log_softmax(logits.float(), dim=-1)
        level_labels = labels[:, level]
        token_scores.append(log_probs.gather(1, level_labels[:, None]).squeeze(1))
    cumulative = torch.stack(token_scores, dim=1).cumsum(dim=1)
    return torch.stack([cumulative[:, depth - 1] for depth in normalized_depths], dim=1)


def legacy_stepwise_prefix_scores(
    model: Any,
    encoder_hidden_states: Any,
    encoder_attention_mask: Any,
    decoder_input_ids: Any,
    decoder_attention_mask: Any,
    labels: Any,
    depths: Sequence[int],
) -> Any:
    import torch
    import torch.nn.functional as F

    normalized_depths = tuple(int(depth) for depth in depths)
    max_depth = max(normalized_depths)
    token_scores = []
    for level in range(max_depth):
        logits = model.logits_for_next_type_from_encoder(
            encoder_hidden_states,
            encoder_attention_mask,
            decoder_input_ids[:, : level + 1],
            decoder_attention_mask[:, : level + 1],
            level,
        )
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_scores.append(
            log_probs.gather(1, labels[:, level, None]).squeeze(1)
        )
    cumulative = torch.stack(token_scores, dim=1).cumsum(dim=1)
    return torch.stack([cumulative[:, depth - 1] for depth in normalized_depths], dim=1)


def verify_legacy_prefix_score_parity(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    decoder_input_ids: Any,
    decoder_attention_mask: Any,
    labels: Any,
    depths: Sequence[int],
    *,
    rtol: float = 1e-5,
    atol: float = 5e-5,
) -> dict[str, float | bool]:
    import torch

    with torch.no_grad(), torch.autocast(
        device_type=input_ids.device.type,
        enabled=False,
    ):
        encoder_hidden_states = model.encode_source(input_ids, attention_mask)
        full_hidden = model._decoder_hidden_from_encoder(
            encoder_hidden_states,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
        )
        full_scores = cumulative_prefix_scores(model, full_hidden, labels, depths)
        stepwise_scores = legacy_stepwise_prefix_scores(
            model,
            encoder_hidden_states,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            labels,
            depths,
        )
    max_abs_error = float(
        (full_scores - stepwise_scores).abs().max().detach().cpu()
    )
    passed = bool(torch.allclose(full_scores, stepwise_scores, rtol=rtol, atol=atol))
    if not passed:
        raise ValueError(
            "full-sequence prefix scores do not match legacy stepwise decoder: "
            f"max_abs_error={max_abs_error}, rtol={rtol}, atol={atol}"
        )
    return {
        "passed": passed,
        "max_abs_error": max_abs_error,
        "rtol": rtol,
        "atol": atol,
    }


def prefix_ranking_loss(
    positive_scores: Any,
    negative_scores: Any,
    *,
    temperature: float,
) -> Any:
    import torch.nn.functional as F

    if positive_scores.shape != negative_scores.shape or positive_scores.ndim != 2:
        raise ValueError("positive/negative prefix score matrices must align")
    if positive_scores.numel() == 0:
        raise ValueError("prefix ranking loss requires at least one comparison")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    return F.softplus((negative_scores - positive_scores) / temperature).mean()


def attach_prefix_ranking_forward(
    model: Any,
    *,
    weight: float,
    scope: str,
    temperature: float,
    trace: dict[str, Any],
    execute_when_zero: bool = False,
) -> Any:
    import torch

    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"prefix ranking weight must be finite and non-negative, got {weight}")
    depths = active_depths(scope, int(model.depth))
    if weight > 0.0 and not depths:
        raise ValueError("positive prefix ranking weight requires an active scope")
    original_forward = model.forward

    def controlled_forward(*args: Any, **kwargs: Any) -> Any:
        eligible_indices = kwargs.pop("prefix_rank_eligible_indices", None)
        negative_decoder_input_ids = kwargs.pop(
            "prefix_rank_negative_decoder_input_ids", None
        )
        negative_decoder_attention_mask = kwargs.pop(
            "prefix_rank_negative_decoder_attention_mask", None
        )
        negative_labels = kwargs.pop("prefix_rank_negative_labels", None)
        query_step = kwargs.pop("prefix_rank_query_step", None)
        ce_loss = original_forward(*args, **kwargs)
        if eligible_indices is None or (weight == 0.0 and not execute_when_zero):
            return ce_loss
        if not model.training:
            raise RuntimeError("prefix ranking path was activated during evaluation")
        required = (
            negative_decoder_input_ids,
            negative_decoder_attention_mask,
            negative_labels,
        )
        if any(value is None for value in required):
            raise ValueError("prefix ranking batch is incomplete")

        input_ids = kwargs["input_ids"].index_select(0, eligible_indices)
        attention_mask = kwargs["attention_mask"].index_select(0, eligible_indices)
        positive_decoder_input_ids = kwargs["decoder_input_ids"].index_select(
            0, eligible_indices
        )
        positive_decoder_attention_mask = kwargs["decoder_attention_mask"].index_select(
            0, eligible_indices
        )
        positive_labels = kwargs["labels"].index_select(0, eligible_indices)

        fork_devices = []
        if input_ids.device.type == "cuda":
            fork_devices = [
                input_ids.device.index
                if input_ids.device.index is not None
                else torch.cuda.current_device()
            ]
        # The auxiliary decoder uses dropout, but it must not advance the main
        # CPU/CUDA RNG streams that determine future legacy CE dropout masks.
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            if query_step == 1:
                was_training = model.training
                model.eval()
                try:
                    trace["score_parity"] = {
                        "positive": verify_legacy_prefix_score_parity(
                            model,
                            input_ids[:1],
                            attention_mask[:1],
                            positive_decoder_input_ids[:1],
                            positive_decoder_attention_mask[:1],
                            positive_labels[:1],
                            depths,
                        ),
                        "negative": verify_legacy_prefix_score_parity(
                            model,
                            input_ids[:1],
                            attention_mask[:1],
                            negative_decoder_input_ids[:1],
                            negative_decoder_attention_mask[:1],
                            negative_labels[:1],
                            depths,
                        ),
                    }
                finally:
                    model.train(was_training)
            encoder_hidden_states = model.encode_source(input_ids, attention_mask)
            decoder_cpu_rng = torch.random.get_rng_state()
            decoder_cuda_rng = (
                torch.cuda.get_rng_state(fork_devices[0]) if fork_devices else None
            )
            positive_hidden = model._decoder_hidden_from_encoder(
                encoder_hidden_states,
                attention_mask,
                positive_decoder_input_ids,
                positive_decoder_attention_mask,
            )
            torch.random.set_rng_state(decoder_cpu_rng)
            if decoder_cuda_rng is not None:
                torch.cuda.set_rng_state(decoder_cuda_rng, fork_devices[0])
            negative_hidden = model._decoder_hidden_from_encoder(
                encoder_hidden_states,
                attention_mask,
                negative_decoder_input_ids,
                negative_decoder_attention_mask,
            )
            positive_scores = cumulative_prefix_scores(
                model, positive_hidden, positive_labels, depths
            )
            negative_scores = cumulative_prefix_scores(
                model, negative_hidden, negative_labels, depths
            )
            rank_loss = prefix_ranking_loss(
                positive_scores, negative_scores, temperature=temperature
            )
            total_loss = ce_loss + weight * rank_loss
        if query_step in {1, 50, 500, 38_805}:
            trace[str(query_step)] = {
                "ce_loss": float(ce_loss.detach().float().cpu()),
                "prefix_rank_loss": float(rank_loss.detach().float().cpu()),
                "total_loss": float(total_loss.detach().float().cpu()),
                "eligible_pairs": int(eligible_indices.numel()),
                "active_comparisons": int(eligible_indices.numel()) * len(depths),
                "finite": bool(torch.isfinite(total_loss.detach()).cpu()),
            }
        return total_loss

    model.forward = controlled_forward
    return model
