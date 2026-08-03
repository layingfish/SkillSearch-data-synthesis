#!/usr/bin/env python3
"""Reproduce the SkillRouter SR-Emb training recipe on SkillGR data.

This script implements the first-stage SkillRouter encoder pipeline described
in arXiv:2603.22455: Qwen3-Embedding-0.6B bi-encoder fine-tuning with
multi-source hard negatives, false-negative filtering, and in-batch InfoNCE.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skill_gr_pipeline import compute_at_k_metrics, load_jsonl, write_json, write_jsonl  # noqa: E402
from distributed_utils import (
    all_gather_tensor,
    all_gather_tensor_with_grad,
    barrier,
    destroy_distributed,
    initialize_distributed,
    unwrap_model,
    wrap_ddp,
)


QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant skill document "
    "that would help an agent complete the task\nQuery:"
)
TOKEN_RE = re.compile(r"[a-z0-9_./+-]+", re.I)


def skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "")


def category_of(sid: str) -> str:
    return sid.split("/", 1)[0] if "/" in sid else "unknown"


def skill_name(row: dict[str, Any]) -> str:
    return str(row.get("name") or row.get("skill_name") or skill_id(row).rsplit("/", 1)[-1])


def query_text(row: dict[str, Any], max_chars: int = 1500) -> str:
    raw = str(row.get("query") or row.get("query_text") or "")
    if row.get("query_text"):
        text = str(row["query_text"])
        marker = "\nQuery:"
        if marker in text:
            raw = text.split(marker, 1)[1]
    return QUERY_INSTRUCTION + raw[:max_chars]


def skill_text(row: dict[str, Any], desc_max: int = 300, body_max: int = 2500) -> str:
    # Rebuild text instead of reusing cached skillrouter_text so the field caps
    # exactly match the SkillRouter paper's encoder recipe.
    name = skill_name(row)
    desc = str(row.get("description") or row.get("task_description") or "")[:desc_max]
    body = str(row.get("body") or "")[:body_max]
    return f"{name} | {desc} | {body}"


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def char_trigrams(text: str) -> set[str]:
    text = re.sub(r"\s+", " ", str(text).lower()).strip()
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if bool(left_padding):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def load_qwen_encoder(model_name_or_path: str, device: torch.device, bf16: bool, gradient_checkpointing: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if bf16 and device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=dtype).to(device)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return tokenizer, model


@torch.no_grad()
def encode_texts_eval(
    texts: list[str],
    tokenizer,
    model: nn.Module,
    device: torch.device,
    max_length: int,
    batch_size: int,
    label: str,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    started = time.time()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        chunks.append(F.normalize(pooled.float(), p=2, dim=1).detach().cpu())
        done = min(start + len(batch), len(texts))
        if done % max(batch_size * 20, 1) == 0 or done == len(texts):
            print(f"encoded {label} {done}/{len(texts)} elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(chunks, dim=0)


@dataclass
class TokenizedTextChunk:
    indices: tuple[int, ...]
    encoded: dict[str, torch.Tensor]


def text_batch_order(texts: list[str], chunk_size: int, length_sort: bool = True) -> list[int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    order = list(range(len(texts)))
    if length_sort and len(order) > chunk_size:
        order.sort(key=lambda idx: len(texts[idx]), reverse=True)
    return order


def tokenize_text_chunks(
    texts: list[str],
    tokenizer,
    max_length: int,
    chunk_size: int,
    length_sort: bool = True,
) -> list[TokenizedTextChunk]:
    order = text_batch_order(texts, chunk_size, length_sort)
    chunks: list[TokenizedTextChunk] = []
    for start in range(0, len(order), chunk_size):
        indices = tuple(order[start : start + chunk_size])
        batch = [texts[idx] for idx in indices]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        chunks.append(TokenizedTextChunk(indices=indices, encoded=dict(encoded)))
    return chunks


def encode_tokenized_text_chunks(
    chunks: list[TokenizedTextChunk],
    num_texts: int,
    model: nn.Module,
    device: torch.device,
    detach: bool = False,
) -> torch.Tensor:
    encoded_by_index: list[torch.Tensor | None] = [None] * num_texts
    for chunk in chunks:
        encoded = {key: value.to(device) for key, value in chunk.encoded.items()}
        outputs = model(**encoded)
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled.float(), p=2, dim=1)
        for local_idx, original_idx in enumerate(chunk.indices):
            value = pooled[local_idx]
            encoded_by_index[original_idx] = value.detach() if detach else value
    if any(value is None for value in encoded_by_index):
        raise RuntimeError("tokenized chunks did not cover every input text")
    return torch.stack([value for value in encoded_by_index if value is not None], dim=0)


def encode_texts_train(
    texts: list[str],
    tokenizer,
    model: nn.Module,
    device: torch.device,
    max_length: int,
    chunk_size: int,
    length_sort: bool = True,
) -> torch.Tensor:
    chunks = tokenize_text_chunks(texts, tokenizer, max_length, chunk_size, length_sort)
    return encode_tokenized_text_chunks(chunks, len(texts), model, device)


def encode_texts_rep_cache(
    texts: list[str],
    tokenizer,
    model: nn.Module,
    device: torch.device,
    max_length: int,
    chunk_size: int,
    length_sort: bool = True,
) -> torch.Tensor:
    chunks = tokenize_text_chunks(texts, tokenizer, max_length, chunk_size, length_sort)
    with torch.no_grad():
        return encode_tokenized_text_chunks(chunks, len(texts), model, device, detach=True)


def backprop_tokenized_text_chunks(
    chunks: list[TokenizedTextChunk],
    rep_grads: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> None:
    for chunk in chunks:
        encoded = {key: value.to(device) for key, value in chunk.encoded.items()}
        outputs = model(**encoded)
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled.float(), p=2, dim=1)
        grad = torch.stack([rep_grads[idx] for idx in chunk.indices], dim=0).to(device)
        torch.autograd.backward(pooled, grad_tensors=grad)


def backprop_text_reps_from_cache_grads(
    texts: list[str],
    rep_grads: torch.Tensor,
    tokenizer,
    model: nn.Module,
    device: torch.device,
    max_length: int,
    chunk_size: int,
    length_sort: bool = True,
) -> None:
    chunks = tokenize_text_chunks(texts, tokenizer, max_length, chunk_size, length_sort)
    backprop_tokenized_text_chunks(chunks, rep_grads, model, device)


def gradcache_stochastic_settings(model: nn.Module) -> list[str]:
    settings: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and float(module.p) > 0:
            settings.add(f"module:{name or '<root>'}.p={float(module.p)}")
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "to_dict"):
        for key, value in config.to_dict().items():
            if "drop" in key.lower() and isinstance(value, (int, float)) and float(value) > 0:
                settings.add(f"config:{key}={float(value)}")
    return sorted(settings)


def ensure_gradcache_deterministic(model: nn.Module) -> None:
    stochastic = gradcache_stochastic_settings(model)
    if stochastic:
        joined = ", ".join(stochastic)
        raise ValueError(
            "gradcache recomputation requires deterministic forward passes; "
            f"positive dropout settings found: {joined}"
        )


def build_bm25_index(skills: list[dict[str, Any]]) -> tuple[list[Counter[str]], list[int], dict[str, float], dict[str, list[tuple[int, int]]], float]:
    doc_term_freqs: list[Counter[str]] = []
    doc_lengths: list[int] = []
    doc_freq: Counter[str] = Counter()
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc_index, skill in enumerate(skills):
        counts = Counter(tokenize(skill["text"]))
        doc_term_freqs.append(counts)
        doc_lengths.append(sum(counts.values()))
        doc_freq.update(counts.keys())
        for term, freq in counts.items():
            postings[term].append((doc_index, freq))
    num_docs = len(skills)
    avgdl = sum(doc_lengths) / max(1, num_docs)
    idf = {term: math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}
    return doc_term_freqs, doc_lengths, idf, postings, avgdl


def bm25_topk(
    query: str,
    doc_lengths: list[int],
    idf: dict[str, float],
    postings: dict[str, list[tuple[int, int]]],
    avgdl: float,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for term in set(tokenize(query)):
        term_idf = idf.get(term)
        if term_idf is None:
            continue
        for doc_index, freq in postings.get(term, []):
            dl = doc_lengths[doc_index]
            denom = freq + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
            scores[doc_index] += term_idf * (freq * (k1 + 1.0)) / denom
    return [idx for idx, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


@dataclass
class SkillMeta:
    row: dict[str, Any]
    sid: str
    name: str
    category: str
    text: str
    body_trigrams: set[str]


def command_mine(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = load_jsonl(args.rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    raw_skills = load_jsonl(args.skills_jsonl)
    skills_meta: list[SkillMeta] = []
    sid_to_index: dict[str, int] = {}
    for row in raw_skills:
        sid = skill_id(row)
        if not sid:
            continue
        text = skill_text(row, args.desc_max, args.body_max)
        sid_to_index[sid] = len(skills_meta)
        skills_meta.append(
            SkillMeta(
                row=row,
                sid=sid,
                name=skill_name(row).lower(),
                category=category_of(sid),
                text=text,
                body_trigrams=char_trigrams(str(row.get("body") or "")[: args.body_max]),
            )
        )
    skills_for_index = [{"skill_id": item.sid, "text": item.text} for item in skills_meta]
    print(f"rows={len(rows)} skills={len(skills_meta)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer, model = load_qwen_encoder(args.model, device, args.bf16, gradient_checkpointing=False)
    skill_encode_batch_size = args.skill_encode_batch_size or args.encode_batch_size
    query_encode_batch_size = args.query_encode_batch_size or args.encode_batch_size
    skill_embeddings = encode_texts_eval(
        [item.text for item in skills_meta],
        tokenizer,
        model,
        device,
        args.max_length,
        skill_encode_batch_size,
        "skills",
    )
    query_embeddings = encode_texts_eval(
        [query_text(row, args.query_max_chars) for row in rows],
        tokenizer,
        model,
        device,
        args.max_length,
        query_encode_batch_size,
        "queries",
    )
    if args.similarity_device == "auto":
        similarity_device = device
    else:
        similarity_device = torch.device(args.similarity_device)
    if similarity_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--similarity-device cuda requested, but CUDA is unavailable")
    if similarity_device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
    search_skill_embeddings = skill_embeddings.to(similarity_device)
    semantic_top: list[list[int]] = []
    for start in range(0, len(rows), args.sim_chunk_size):
        query_chunk = query_embeddings[start : start + args.sim_chunk_size].to(similarity_device)
        sims = query_chunk @ search_skill_embeddings.T
        _, top_indices = torch.topk(sims, k=min(args.mine_top_k, len(skills_meta)), dim=1)
        semantic_top.extend([[int(v) for v in row] for row in top_indices.cpu().tolist()])
        print(f"semantic topk {min(start + args.sim_chunk_size, len(rows))}/{len(rows)}", flush=True)
    del search_skill_embeddings

    print("building bm25 index", flush=True)
    _, doc_lengths, idf, postings, avgdl = build_bm25_index(skills_for_index)
    bm25_top: list[list[int]] = []
    for i, row in enumerate(rows):
        bm25_top.append(bm25_topk(query_text(row, args.query_max_chars), doc_lengths, idf, postings, avgdl, args.mine_top_k))
        if (i + 1) % 1000 == 0 or i + 1 == len(rows):
            print(f"bm25 topk {i + 1}/{len(rows)}", flush=True)

    by_category: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(skills_meta):
        by_category[item.category].append(idx)
    all_indices = list(range(len(skills_meta)))

    filter_counts = Counter()
    source_counts = Counter()
    pair_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def is_false_negative(
        candidate_idx: int,
        positive_indices: list[int],
        cache: dict[int, str | None],
    ) -> str | None:
        if candidate_idx in cache:
            return cache[candidate_idx]
        cand = skills_meta[candidate_idx]
        for pos_idx in positive_indices:
            pos = skills_meta[pos_idx]
            if cand.name == pos.name:
                cache[candidate_idx] = "same_name"
                return cache[candidate_idx]
            if jaccard(cand.body_trigrams, pos.body_trigrams) > args.trigram_jaccard_threshold:
                cache[candidate_idx] = "body_overlap"
                return cache[candidate_idx]
            sim = float(torch.dot(skill_embeddings[candidate_idx], skill_embeddings[pos_idx]))
            if sim > args.embedding_similarity_threshold:
                cache[candidate_idx] = "embedding_similarity"
                return cache[candidate_idx]
        cache[candidate_idx] = None
        return cache[candidate_idx]

    def take_from(
        source: list[int],
        positive_indices: list[int],
        selected: list[int],
        n: int,
        source_name: str,
        false_negative_cache: dict[int, str | None],
    ) -> None:
        candidates = list(source)
        rng.shuffle(candidates)
        positive_set = set(positive_indices)
        selected_set = set(selected)
        added = 0
        for idx in candidates:
            if len(selected) >= args.negatives_per_pair:
                break
            if added >= n:
                break
            if idx in positive_set or idx in selected_set:
                continue
            reason = is_false_negative(idx, positive_indices, false_negative_cache)
            if reason:
                filter_counts[reason] += 1
                continue
            selected.append(idx)
            selected_set.add(idx)
            added += 1
            source_counts[source_name] += 1

    def take_random(
        positive_indices: list[int],
        selected: list[int],
        n: int,
        source_name: str,
        false_negative_cache: dict[int, str | None],
        excluded_category: str | None,
    ) -> None:
        positive_set = set(positive_indices)
        selected_set = set(selected)
        added = 0
        attempts = 0
        max_attempts = max(100, len(all_indices) * 2)
        while added < n and attempts < max_attempts and len(selected) < args.negatives_per_pair:
            idx = all_indices[rng.randrange(len(all_indices))]
            attempts += 1
            if idx in positive_set or idx in selected_set:
                continue
            if excluded_category is not None and skills_meta[idx].category == excluded_category:
                continue
            reason = is_false_negative(idx, positive_indices, false_negative_cache)
            if reason:
                filter_counts[reason] += 1
                continue
            selected.append(idx)
            selected_set.add(idx)
            added += 1
            source_counts[source_name] += 1

        if added >= n or len(selected) >= args.negatives_per_pair:
            return
        start = rng.randrange(len(all_indices))
        for offset in range(len(all_indices)):
            idx = all_indices[(start + offset) % len(all_indices)]
            if idx in positive_set or idx in selected_set:
                continue
            if excluded_category is not None and skills_meta[idx].category == excluded_category:
                continue
            reason = is_false_negative(idx, positive_indices, false_negative_cache)
            if reason:
                filter_counts[reason] += 1
                continue
            selected.append(idx)
            selected_set.add(idx)
            added += 1
            source_counts[source_name] += 1
            if added >= n or len(selected) >= args.negatives_per_pair:
                break

    with args.output.open("w", encoding="utf-8") as out:
        for row_index, row in enumerate(rows):
            gold_sids = unique_keep_order([str(sid) for sid in row.get("selected_skill_ids", [])])
            positive_indices = [sid_to_index[sid] for sid in gold_sids if sid in sid_to_index]
            if not positive_indices:
                filter_counts["rows_without_positive_in_pool"] += 1
                continue
            false_negative_cache: dict[int, str | None] = {}
            for pos_idx in positive_indices:
                selected: list[int] = []
                semantic_candidates = semantic_top[row_index]
                bm25_candidates = bm25_top[row_index]
                same_cat = [idx for idx in by_category[skills_meta[pos_idx].category] if skills_meta[idx].name != skills_meta[pos_idx].name]
                take_from(
                    semantic_candidates,
                    positive_indices,
                    selected,
                    args.semantic_negatives,
                    "semantic",
                    false_negative_cache,
                )
                take_from(
                    bm25_candidates,
                    positive_indices,
                    selected,
                    args.lexical_negatives,
                    "lexical",
                    false_negative_cache,
                )
                take_from(
                    same_cat,
                    positive_indices,
                    selected,
                    args.taxonomy_negatives,
                    "taxonomy",
                    false_negative_cache,
                )
                take_random(
                    positive_indices,
                    selected,
                    args.random_negatives,
                    "random",
                    false_negative_cache,
                    excluded_category=skills_meta[pos_idx].category,
                )
                if len(selected) < args.negatives_per_pair:
                    fallback = unique_keep_order(semantic_candidates + bm25_candidates + same_cat)
                    take_from(
                        fallback,
                        positive_indices,
                        selected,
                        args.negatives_per_pair - len(selected),
                        "fallback",
                        false_negative_cache,
                    )
                if len(selected) < args.negatives_per_pair:
                    take_random(
                        positive_indices,
                        selected,
                        args.negatives_per_pair - len(selected),
                        "fallback",
                        false_negative_cache,
                        excluded_category=None,
                    )
                if len(selected) < args.negatives_per_pair:
                    filter_counts["pairs_with_short_negatives"] += 1
                selected = selected[: args.negatives_per_pair]
                record = {
                    "row_index": row_index,
                    "source_index": row.get("source_index"),
                    "query": str(row.get("query") or ""),
                    "query_text": query_text(row, args.query_max_chars),
                    "positive_skill_id": skills_meta[pos_idx].sid,
                    "positive_text": skills_meta[pos_idx].text,
                    "gold_skill_ids": gold_sids,
                    "negative_skill_ids": [skills_meta[idx].sid for idx in selected],
                    "negative_texts": [skills_meta[idx].text for idx in selected],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                pair_count += 1
            if (row_index + 1) % 1000 == 0 or row_index + 1 == len(rows):
                print(f"mined rows {row_index + 1}/{len(rows)} pairs={pair_count}", flush=True)

    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "rows": str(args.rows),
            "skills_jsonl": str(args.skills_jsonl),
            "output": str(args.output),
            "pairs": pair_count,
            "skills": len(skills_meta),
            "model": args.model,
            "mine_top_k": args.mine_top_k,
            "negatives_per_pair": args.negatives_per_pair,
            "skill_encode_batch_size": skill_encode_batch_size,
            "query_encode_batch_size": query_encode_batch_size,
            "similarity_device": str(similarity_device),
            "source_counts": dict(source_counts),
            "filter_counts": dict(filter_counts),
            "thresholds": {
                "trigram_jaccard": args.trigram_jaccard_threshold,
                "embedding_similarity": args.embedding_similarity_threshold,
            },
        },
    )
    print(json.dumps({"pairs": pair_count, "source_counts": dict(source_counts), "filter_counts": dict(filter_counts)}, sort_keys=True))


class PairDataset(Dataset):
    def __init__(self, path: Path, max_examples: int | None = None):
        self.rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))
                if max_examples and len(self.rows) >= max_examples:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def command_train(args: argparse.Namespace) -> None:
    dataset = PairDataset(args.pairs, args.max_examples)
    if len(dataset) == 0:
        raise SystemExit("empty training pair dataset")
    context = initialize_distributed(cpu=args.cpu)
    if args.training_mode == "gradcache" and context.enabled:
        raise SystemExit(
            "--training-mode gradcache is intentionally single-process: its manual recomputation path "
            "needs a dedicated cross-rank gradient cache; use --training-mode standard for 8-GPU DDP"
        )
    rng = random.Random(args.seed + context.rank)
    random.seed(args.seed)
    torch.manual_seed(args.seed + context.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + context.rank)
    device = context.device
    if args.training_mode == "gradcache" and args.fp16:
        raise SystemExit("gradcache mode currently supports bf16/fp32 only; use --bf16 instead of --fp16")
    use_gradient_checkpointing = not args.no_gradient_checkpointing and args.training_mode == "standard"
    tokenizer, model = load_qwen_encoder(
        args.model,
        device,
        args.bf16,
        gradient_checkpointing=use_gradient_checkpointing,
    )
    model = wrap_ddp(model, context)
    model.train()
    if args.training_mode == "gradcache":
        ensure_gradcache_deterministic(model)
    wandb_run = None
    if context.is_main and args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            )
        except Exception as exc:
            print(f"wandb disabled: {exc}", flush=True)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Shuffle negatives per epoch/step.  Keep a fixed doc tensor shape for
        # cross-rank all-gather, but mask missing negatives instead of repeating
        # a real document (which would change its InfoNCE denominator weight).
        query_texts = [row["query_text"] for row in batch]
        doc_texts: list[str] = []
        doc_valid: list[bool] = []
        labels: list[int] = []
        for row in batch:
            negs = list(row.get("negative_texts") or [])
            rng.shuffle(negs)
            negs = negs[: args.negatives_per_pair]
            labels.append(len(doc_texts))
            doc_texts.append(row["positive_text"])
            doc_valid.append(True)
            doc_texts.extend(negs)
            doc_valid.extend([True] * len(negs))
            missing = args.negatives_per_pair - len(negs)
            if missing > 0:
                doc_texts.extend([""] * missing)
                doc_valid.extend([False] * missing)
        return {
            "queries": query_texts,
            "docs": doc_texts,
            "labels": torch.tensor(labels, dtype=torch.long),
            "doc_valid_mask": torch.tensor(doc_valid, dtype=torch.bool),
        }

    sampler = None
    if context.enabled:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
    )
    if args.fused_optimizer and device.type != "cuda":
        raise SystemExit("--fused-optimizer requires CUDA")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=args.fused_optimizer,
    )
    updates_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum))
    total_updates = args.max_updates or max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler_enabled = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_dir / "train_manifest.json",
            {
                "pairs": str(args.pairs),
                "examples": len(dataset),
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "negatives_per_pair": args.negatives_per_pair,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "temperature": args.temperature,
                "max_length": args.max_length,
                "bf16": args.bf16,
                "fp16": args.fp16,
                "training_mode": args.training_mode,
                "gradient_checkpointing": use_gradient_checkpointing,
                "query_chunk_size": args.query_chunk_size,
                "doc_chunk_size": args.doc_chunk_size,
                "length_sort": args.length_sort,
                "fused_optimizer": args.fused_optimizer,
                "gradcache_tokenization_reuse": args.training_mode == "gradcache",
                "ddp_world_size": context.world_size,
                "per_device_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * context.world_size,
                "cross_rank_in_batch_negatives": bool(context.enabled),
                "short_negative_padding": "masked_dummy_docs",
            },
        )
    log_path = args.output_dir / "train_loss.jsonl"
    if context.is_main and log_path.exists():
        log_path.unlink()
    barrier(context)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch - 1)
        for step, batch in enumerate(loader, start=1):
            labels = batch["labels"].to(device)
            doc_valid_mask = batch["doc_valid_mask"].to(device)
            autocast_enabled = args.bf16 and device.type == "cuda"
            if args.training_mode == "gradcache":
                query_chunks = tokenize_text_chunks(
                    batch["queries"],
                    tokenizer,
                    args.max_length,
                    args.query_chunk_size,
                    args.length_sort,
                )
                doc_chunks = tokenize_text_chunks(
                    batch["docs"],
                    tokenizer,
                    args.max_length,
                    args.doc_chunk_size,
                    args.length_sort,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    with torch.no_grad():
                        query_emb = encode_tokenized_text_chunks(
                            query_chunks, len(batch["queries"]), model, device, detach=True
                        ).requires_grad_(True)
                        doc_emb = encode_tokenized_text_chunks(
                            doc_chunks, len(batch["docs"]), model, device, detach=True
                        ).requires_grad_(True)
                    global_doc_emb = all_gather_tensor_with_grad(doc_emb, context)
                    global_doc_valid = all_gather_tensor(doc_valid_mask.to(torch.uint8), context).bool()
                    labels = labels + context.rank * doc_emb.shape[0]
                    scores = (query_emb @ global_doc_emb.T) / args.temperature
                    scores = scores.masked_fill(~global_doc_valid.unsqueeze(0), torch.finfo(scores.dtype).min)
                    loss = F.cross_entropy(scores, labels) / args.grad_accum
                loss.backward()
                query_grad = query_emb.grad.detach()
                doc_grad = doc_emb.grad.detach()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    backprop_tokenized_text_chunks(query_chunks, query_grad, model, device)
                    backprop_tokenized_text_chunks(doc_chunks, doc_grad, model, device)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    query_emb = encode_texts_train(
                        batch["queries"],
                        tokenizer,
                        model,
                        device,
                        args.max_length,
                        args.query_chunk_size,
                        args.length_sort,
                    )
                    doc_emb = encode_texts_train(
                        batch["docs"],
                        tokenizer,
                        model,
                        device,
                        args.max_length,
                        args.doc_chunk_size,
                        args.length_sort,
                    )
                    global_doc_emb = all_gather_tensor_with_grad(doc_emb, context)
                    global_doc_valid = all_gather_tensor(doc_valid_mask.to(torch.uint8), context).bool()
                    labels = labels + context.rank * doc_emb.shape[0]
                    scores = (query_emb @ global_doc_emb.T) / args.temperature
                    scores = scores.masked_fill(~global_doc_valid.unsqueeze(0), torch.finfo(scores.dtype).min)
                    loss = F.cross_entropy(scores, labels) / args.grad_accum
                if scaler_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if step % args.grad_accum == 0 or step == len(loader):
                if scaler_enabled:
                    scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                event = {
                    "epoch": epoch,
                    "update": global_step,
                    "loader_step": step,
                    "loss": float(loss.detach().cpu()) * args.grad_accum,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "elapsed": time.time() - started,
                }
                if context.is_main:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")
                    if wandb_run is not None:
                        wandb_run.log({"train/loss": event["loss"], "train/lr": event["lr"], "train/epoch": epoch}, step=global_step)
                if context.is_main and (global_step % args.logging_steps == 0 or global_step == 1):
                    print(
                        f"epoch={epoch} update={global_step}/{total_updates} "
                        f"loss={event['loss']:.6f} lr={event['lr']:.2e} elapsed={event['elapsed']:.1f}s",
                        flush=True,
                    )
                if args.save_steps and global_step % args.save_steps == 0:
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                    if context.is_main:
                        unwrap_model(model).save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)
                        print(f"saved checkpoint to {checkpoint_dir}", flush=True)
                    barrier(context)
                if args.max_updates and global_step >= args.max_updates:
                    break
        if args.max_updates and global_step >= args.max_updates:
            break

    if context.is_main:
        unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    if wandb_run is not None:
        wandb_run.finish()
    if context.is_main:
        print(f"saved model to {args.output_dir}", flush=True)
    barrier(context)
    destroy_distributed(context)


def command_eval(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.rows)
    skills_raw = load_jsonl(args.skills_jsonl)
    skills = [{"skill_id": skill_id(row), "text": skill_text(row, args.desc_max, args.body_max)} for row in skills_raw]
    skills = [row for row in skills if row["skill_id"]]
    k_values = sorted({int(part) for part in args.k_values.split(",") if part.strip()})
    max_rank = max(k_values)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer, model = load_qwen_encoder(args.model, device, args.bf16, gradient_checkpointing=False)
    skill_embeddings = encode_texts_eval(
        [row["text"] for row in skills], tokenizer, model, device, args.max_skill_length, args.batch_size, "skills"
    )
    query_embeddings = encode_texts_eval(
        [query_text(row, args.query_max_chars) for row in rows],
        tokenizer,
        model,
        device,
        args.max_query_length,
        args.batch_size,
        args.split_name,
    )
    scores = query_embeddings @ skill_embeddings.T
    top_scores, top_indices = torch.topk(scores, k=min(max_rank, len(skills)), dim=1)
    predictions = []
    for row_index, row in enumerate(rows):
        ranked = [skills[int(index)]["skill_id"] for index in top_indices[row_index].tolist()]
        predictions.append(
            {
                "index": row.get("index"),
                "source_index": row.get("source_index"),
                "target_skill_count": row.get("target_skill_count"),
                "query": row.get("query"),
                "gold_skill_ids": [str(sid) for sid in row.get("selected_skill_ids", [])],
                "ranked_skill_ids": ranked,
                "ranked_scores": [float(score) for score in top_scores[row_index].tolist()],
                "invalid": False,
            }
        )
    metrics = compute_at_k_metrics(rows, predictions, k_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "ranked_metrics.json", metrics)
    write_json(
        args.output_dir / "eval_manifest.json",
        {
            "baseline": "skillrouter_embedding_repro",
            "model": str(args.model),
            "rows": str(args.rows),
            "skills_jsonl": str(args.skills_jsonl),
            "candidate_skills": len(skills),
            "split": args.split_name,
            "k_values": k_values,
            "pooling": "last_token_pool",
            "normalize": True,
            "query_max_chars": args.query_max_chars,
            "desc_max": args.desc_max,
            "body_max": args.body_max,
            "max_query_length": args.max_query_length,
            "max_skill_length": args.max_skill_length,
        },
    )
    print(json.dumps(metrics["overall"], ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mine")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--seed", type=int, default=20260707)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--query-max-chars", type=int, default=1500)
    p.add_argument("--desc-max", type=int, default=300)
    p.add_argument("--body-max", type=int, default=2500)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--skill-encode-batch-size", type=int)
    p.add_argument("--query-encode-batch-size", type=int)
    p.add_argument("--sim-chunk-size", type=int, default=256)
    p.add_argument("--similarity-device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--mine-top-k", type=int, default=50)
    p.add_argument("--semantic-negatives", type=int, default=4)
    p.add_argument("--lexical-negatives", type=int, default=3)
    p.add_argument("--taxonomy-negatives", type=int, default=2)
    p.add_argument("--random-negatives", type=int, default=1)
    p.add_argument("--negatives-per-pair", type=int, default=10)
    p.add_argument("--trigram-jaccard-threshold", type=float, default=0.6)
    p.add_argument("--embedding-similarity-threshold", type=float, default=0.92)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_mine)

    p = sub.add_parser("train")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--seed", type=int, default=20260707)
    p.add_argument("--max-examples", type=int)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--negatives-per-pair", type=int, default=10)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--query-chunk-size", type=int, default=8)
    p.add_argument("--doc-chunk-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--training-mode", choices=["standard", "gradcache"], default="standard")
    p.add_argument("--length-sort", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fused-optimizer", action="store_true")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-updates", type=int)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-run-name")
    p.add_argument("--wandb-mode", default="online")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_train)

    p = sub.add_parser("eval")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--skills-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split-name", default="test")
    p.add_argument("--k-values", default="1,10,50")
    p.add_argument("--query-max-chars", type=int, default=1500)
    p.add_argument("--desc-max", type=int, default=300)
    p.add_argument("--body-max", type=int, default=2500)
    p.add_argument("--max-query-length", type=int, default=2048)
    p.add_argument("--max-skill-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_eval)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
