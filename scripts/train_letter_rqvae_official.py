#!/usr/bin/env python3
"""Train LETTER's official RQ-VAE tokenizer on SkillGR embeddings.

This script is intentionally a thin adapter.  The tokenizer model, residual
quantizer, diversity loss, Sinkhorn assignment, and trainer are imported from
the official LETTER RQ-VAE code.  Project-specific code only prepares inputs
and converts LETTER indices back to SkillGR semantic_ids.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from skill_gr_pipeline import (  # noqa: E402
    SEMANTIC_ID_MODE_DEDUP,
    SEMANTIC_ID_MODE_LEARNED,
    SEMANTIC_ID_MODES,
    collect_query_skill_interactions,
    deduplicate_code_tuples,
    load_jsonl,
    normalize_semantic_id_mode,
    semantic_id_tokens,
    validate_semantic_ids,
    write_json,
    write_jsonl,
)


DIVERSITY_LOSS_IMPL_OFFICIAL = "official"
DIVERSITY_LOSS_IMPL_VECTORIZED = "vectorized_bitwise_v1"
DIVERSITY_LOSS_IMPLS = (DIVERSITY_LOSS_IMPL_OFFICIAL, DIVERSITY_LOSS_IMPL_VECTORIZED)


def vectorized_letter_diversity_loss_main_entry(self, x, x_q, indices, labels):
    """Match LETTER's diversity loss while avoiding per-item CUDA synchronization."""
    import torch
    import torch.nn.functional as F

    indices_cpu = indices.detach().cpu().tolist()
    indices_cluster = [labels[index] for index in indices_cpu]
    indices_list = {
        target: [index for index, cluster in enumerate(labels) if cluster == target]
        for target in range(10)
    }
    pos_list = [indices_list[cluster] for cluster in indices_cluster]
    pos_sample = []
    for batch_index, positives in enumerate(pos_list):
        sampled = random.choice(positives)
        while sampled == indices_cpu[batch_index]:
            sampled = random.choice(positives)
        pos_sample.append(sampled)

    y_true = torch.tensor(pos_sample, device=x_q.device)
    similarities = torch.matmul(x_q, self.embedding.weight.t())
    self_mask = torch.zeros_like(similarities)
    self_mask.scatter_(1, indices[:, None], 1e12)
    return F.cross_entropy((similarities - self_mask) / 1, y_true)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def train_query_skill_cf_embeddings(
    rows: list[dict[str, Any]],
    skill_ids: list[str],
    dim: int,
    epochs: int,
    batch_size: int,
    negatives: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    interactions, missing = collect_query_skill_interactions(rows, skill_to_index)
    if not interactions:
        raise SystemExit("no query-skill interactions found for CF embedding training")

    query_emb = nn.Embedding(len(rows), dim).to(device)
    skill_emb = nn.Embedding(len(skill_ids), dim).to(device)
    nn.init.normal_(query_emb.weight, mean=0.0, std=0.02)
    nn.init.normal_(skill_emb.weight, mean=0.0, std=0.02)
    opt = torch.optim.AdamW(
        [*query_emb.parameters(), *skill_emb.parameters()],
        lr=lr,
        weight_decay=weight_decay,
    )
    pos_queries = torch.tensor([query_index for query_index, _ in interactions], dtype=torch.long)
    pos_skills = torch.tensor([skill_index for _, skill_index in interactions], dtype=torch.long)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    steps_per_epoch = max(1, math.ceil(len(interactions) / batch_size))
    last_loss = None
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(interactions), generator=rng)
        losses = []
        for start in range(0, len(interactions), batch_size):
            batch_order = order[start : start + batch_size]
            q_idx = pos_queries[batch_order].to(device)
            pos_idx = pos_skills[batch_order].to(device)
            neg_idx = torch.randint(0, len(skill_ids), (len(batch_order), negatives), device=device)
            q_vec = query_emb(q_idx)
            pos_vec = skill_emb(pos_idx)
            neg_vec = skill_emb(neg_idx)
            scale = math.sqrt(dim)
            pos_logits = (q_vec * pos_vec).sum(dim=-1) / scale
            neg_logits = (q_vec.unsqueeze(1) * neg_vec).sum(dim=-1) / scale
            loss = F.softplus(-pos_logits).mean() + F.softplus(neg_logits).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = sum(losses) / max(1, len(losses))
        print(f"cf_epoch={epoch} cf_step={epoch * steps_per_epoch} cf_loss={last_loss:.6f}", flush=True)

    with torch.no_grad():
        cf_embeddings = F.normalize(skill_emb.weight.detach().cpu().float(), p=2, dim=1)
    return cf_embeddings.numpy(), interactions, missing, last_loss


def official_constrained_km(data, n_clusters: int = 10):
    from k_means_constrained import KMeansConstrained
    import torch

    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    clf = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=n_clusters * 6,
        max_iter=10,
        n_init=10,
        n_jobs=10,
        verbose=False,
    )
    clf.fit(x)
    return torch.from_numpy(clf.cluster_centers_), torch.from_numpy(clf.labels_).tolist()


def collision_item_groups(all_indices: list[tuple[int, ...]]) -> list[list[int]]:
    index_to_items: dict[tuple[int, ...], list[int]] = {}
    for item_index, code in enumerate(all_indices):
        index_to_items.setdefault(tuple(code), []).append(item_index)
    return [items for items in index_to_items.values() if len(items) > 1]


def minimum_cost_unique_last_codes(
    all_indices: list[tuple[int, ...]],
    last_code_costs,
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    """Jointly reassign the last code in collided prefixes with minimum total distortion."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not all_indices:
        raise ValueError("cannot resolve collisions for an empty code list")
    depth = len(all_indices[0])
    if depth < 1 or any(len(code) != depth for code in all_indices):
        raise ValueError("all learned code tuples must have one fixed positive length")
    costs = np.asarray(last_code_costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] != len(all_indices):
        raise ValueError(
            f"last_code_costs must have shape ({len(all_indices)}, codebook_size), got {costs.shape}"
        )
    if not np.isfinite(costs).all():
        raise ValueError("last_code_costs contains non-finite values")

    prefix_to_items: dict[tuple[int, ...], list[int]] = {}
    for item_index, code in enumerate(all_indices):
        prefix_to_items.setdefault(tuple(code[:-1]), []).append(item_index)

    resolved = [tuple(int(value) for value in code) for code in all_indices]
    repaired_prefixes = 0
    repaired_items = 0
    changed_items = 0
    assignment_cost = 0.0
    max_repaired_prefix_size = 0
    for prefix in sorted(prefix_to_items):
        item_indices = prefix_to_items[prefix]
        last_counts = Counter(all_indices[item_index][-1] for item_index in item_indices)
        if max(last_counts.values()) <= 1:
            continue
        if len(item_indices) > costs.shape[1]:
            raise ValueError(
                f"prefix {prefix} has {len(item_indices)} items but the last codebook has "
                f"only {costs.shape[1]} entries"
            )
        repaired_prefixes += 1
        repaired_items += len(item_indices)
        max_repaired_prefix_size = max(max_repaired_prefix_size, len(item_indices))
        row_indices, assigned_codes = linear_sum_assignment(costs[item_indices])
        if len(row_indices) != len(item_indices):
            raise RuntimeError(f"incomplete last-code assignment for prefix {prefix}")
        for local_row, assigned_code in zip(row_indices.tolist(), assigned_codes.tolist()):
            item_index = item_indices[local_row]
            old_code = all_indices[item_index][-1]
            resolved[item_index] = (*prefix, int(assigned_code))
            changed_items += int(old_code != assigned_code)
            assignment_cost += float(costs[item_index, assigned_code])

    post_counts = Counter(resolved)
    post_collision_groups = sum(count > 1 for count in post_counts.values())
    if post_collision_groups:
        duplicate = next(code for code, count in post_counts.items() if count > 1)
        raise RuntimeError(f"strict last-code assignment left duplicate semantic id: {duplicate}")
    audit = {
        "repaired_prefixes": repaired_prefixes,
        "repaired_items": repaired_items,
        "changed_items": changed_items,
        "max_repaired_prefix_size": max_repaired_prefix_size,
        "assignment_cost": assignment_cost,
        "post_collision_groups": post_collision_groups,
    }
    return resolved, audit


def compute_last_code_costs(
    model,
    dataset,
    all_indices: list[tuple[int, ...]],
    batch_size: int,
    num_workers: int,
    device,
):
    import torch
    from torch.utils.data import DataLoader

    if not all_indices:
        raise ValueError("cannot compute last-code costs for an empty code list")
    if len(all_indices) != len(dataset):
        raise ValueError(f"code count {len(all_indices)} != dataset size {len(dataset)}")
    model.eval()
    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=getattr(device, "type", "") == "cuda",
    )
    last_codebook = model.rq.vq_layers[-1].embedding.weight
    cost_batches = []
    with torch.no_grad():
        for data, item_indices in loader:
            data = data.to(device)
            item_indices_list = [int(index) for index in item_indices.tolist()]
            residual = model.encoder(data)
            for level, vq_layer in enumerate(model.rq.vq_layers[:-1]):
                level_codes = torch.tensor(
                    [all_indices[item_index][level] for item_index in item_indices_list],
                    dtype=torch.long,
                    device=device,
                )
                residual = residual - vq_layer.embedding(level_codes)
            costs = (
                torch.sum(residual**2, dim=1, keepdim=True)
                + torch.sum(last_codebook**2, dim=1).unsqueeze(0)
                - 2 * torch.matmul(residual, last_codebook.t())
            )
            cost_batches.append(costs.cpu().double())
    return torch.cat(cost_batches, dim=0).numpy()


def generate_letter_indices(model, dataset, batch_size: int, num_workers: int, device, max_collision_passes: int):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model.eval()
    labels: dict[str, list[int]] = {}
    embs = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]
    for idx, emb in enumerate(embs):
        _, label = official_constrained_km(emb)
        labels[str(idx)] = label

    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False, pin_memory=True)
    all_indices: list[tuple[int, ...]] = []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), ncols=100, desc="LETTER indices"):
            data = batch[0].to(device)
            indices = model.get_indices(data, labels, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            all_indices.extend(tuple(int(v) for v in index) for index in indices)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    for _ in range(max_collision_passes):
        groups = collision_item_groups(all_indices)
        if not groups:
            break
        with torch.no_grad():
            for collision_items in groups:
                data = dataset[collision_items][0].to(device)
                indices = model.get_indices(data, labels, use_sk=True)
                indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                for item, index in zip(collision_items, indices):
                    all_indices[item] = tuple(int(v) for v in index)
    return all_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--letter-rqvae-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-emb-list", default="256,256,256")
    parser.add_argument(
        "--semantic-id-mode",
        choices=SEMANTIC_ID_MODES,
        default=SEMANTIC_ID_MODE_DEDUP,
        help="legacy RQ codes plus ordinal dedup, or collision-free learned codes only",
    )
    parser.add_argument("--e-dim", type=int, default=128)
    parser.add_argument("--layers", default="512")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--quant-loss-weight", type=float, default=1.0)
    parser.add_argument("--sk-epsilons", default="")
    parser.add_argument("--sk-iters", type=int, default=50)
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-step", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cf-epochs", type=int, default=50)
    parser.add_argument("--cf-batch-size", type=int, default=4096)
    parser.add_argument("--cf-negatives", type=int, default=16)
    parser.add_argument("--cf-lr", type=float, default=1e-3)
    parser.add_argument("--cf-weight-decay", type=float, default=1e-5)
    parser.add_argument("--cf-embeddings", type=Path, default=None, help="external LETTER-compatible skill CF embeddings")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-collision-passes", type=int, default=20)
    parser.add_argument(
        "--diversity-loss-impl",
        choices=DIVERSITY_LOSS_IMPLS,
        default=DIVERSITY_LOSS_IMPL_OFFICIAL,
        help="official Python loop or a bitwise-equivalent vectorized implementation",
    )
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    import numpy as np
    import torch

    letter_dir = args.letter_rqvae_dir.resolve()
    if not (letter_dir / "models" / "rqvae.py").exists():
        raise SystemExit(f"invalid LETTER RQ-VAE directory: {letter_dir}")
    sys.path.insert(0, str(letter_dir))

    from datasets import EmbDataset
    from models.rqvae import RQVAE
    from models.vq import VectorQuantizer
    from trainer import Trainer

    if args.diversity_loss_impl == DIVERSITY_LOSS_IMPL_VECTORIZED:
        VectorQuantizer.diversity_loss_main_entry = vectorized_letter_diversity_loss_main_entry
    print(f"LETTER diversity loss implementation: {args.diversity_loss_impl}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    semantic_id_mode = normalize_semantic_id_mode(args.semantic_id_mode)
    num_emb_list = parse_int_list(args.num_emb_list)
    layers = parse_int_list(args.layers)
    if semantic_id_mode == SEMANTIC_ID_MODE_LEARNED and len(num_emb_list) != 4:
        raise SystemExit("learned_codes LETTER runs require exactly four entries in --num-emb-list")
    if semantic_id_mode == SEMANTIC_ID_MODE_LEARNED and len(set(num_emb_list)) != 1:
        raise SystemExit("learned_codes currently requires a uniform codebook size at all four levels")
    sk_epsilons = parse_float_list(args.sk_epsilons) if args.sk_epsilons else [0.0] * (len(num_emb_list) - 1) + [0.003]
    if len(sk_epsilons) != len(num_emb_list):
        raise SystemExit("--sk-epsilons length must match --num-emb-list length")

    payload = torch.load(args.embeddings, map_location="cpu")
    embeddings = payload["embeddings"].float().numpy()
    skill_ids = [str(sid) for sid in payload["skill_ids"]]
    rows = load_jsonl(args.rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "skill_embeddings.npy"
    cf_path = args.output_dir / "query_skill_cf_embeddings.pt"
    np.save(data_path, embeddings.astype("float32"))
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    interactions, missing = collect_query_skill_interactions(rows, skill_to_index)
    final_cf_loss = None
    cf_source = "query_skill_internal"
    if args.cf_embeddings is not None:
        import torch.nn.functional as F

        cf_source = str(args.cf_embeddings)
        loaded_cf = torch.load(args.cf_embeddings, map_location="cpu")
        if isinstance(loaded_cf, dict):
            for key in ("embeddings", "cf_embeddings", "item_embeddings"):
                if key in loaded_cf:
                    loaded_cf = loaded_cf[key]
                    break
        cf_tensor = torch.as_tensor(loaded_cf).float().squeeze()
        if cf_tensor.ndim != 2:
            raise SystemExit(f"--cf-embeddings must be a 2D tensor, got shape {tuple(cf_tensor.shape)}")
        if cf_tensor.shape[0] != len(skill_ids):
            raise SystemExit(f"--cf-embeddings row count {cf_tensor.shape[0]} != skills {len(skill_ids)}")
        if cf_tensor.shape[1] != args.e_dim:
            raise SystemExit(f"--cf-embeddings dim {cf_tensor.shape[1]} != --e-dim {args.e_dim}")
        cf_embeddings = F.normalize(cf_tensor, p=2, dim=1).cpu().numpy()
    else:
        cf_embeddings, interactions, missing, final_cf_loss = train_query_skill_cf_embeddings(
            rows=rows,
            skill_ids=skill_ids,
            dim=args.e_dim,
            epochs=args.cf_epochs,
            batch_size=args.cf_batch_size,
            negatives=args.cf_negatives,
            lr=args.cf_lr,
            weight_decay=args.cf_weight_decay,
            seed=args.seed,
            device=device,
        )
    torch.save(torch.from_numpy(cf_embeddings), cf_path)

    official_args = SimpleNamespace(
        lr=args.lr,
        learner="AdamW",
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        eval_step=args.eval_step,
        device=str(device),
        ckpt_dir=str(args.output_dir / "official_checkpoints"),
        data_path=str(data_path),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        kmeans_init=True,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=sk_epsilons,
        sk_iters=args.sk_iters,
        num_emb_list=num_emb_list,
        e_dim=args.e_dim,
        quant_loss_weight=args.quant_loss_weight,
        alpha=args.alpha,
        beta=args.beta,
        n_clusters=10,
        sample_strategy="all",
        cf_emb=str(cf_path),
        layers=layers,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset = EmbDataset(str(data_path))
    model = RQVAE(
        in_dim=dataset.dim,
        num_emb_list=official_args.num_emb_list,
        e_dim=official_args.e_dim,
        layers=official_args.layers,
        dropout_prob=official_args.dropout_prob,
        bn=official_args.bn,
        loss_type=official_args.loss_type,
        quant_loss_weight=official_args.quant_loss_weight,
        kmeans_init=official_args.kmeans_init,
        kmeans_iters=official_args.kmeans_iters,
        sk_epsilons=official_args.sk_epsilons,
        sk_iters=official_args.sk_iters,
        beta=official_args.beta,
        alpha=official_args.alpha,
        n_clusters=official_args.n_clusters,
        sample_strategy=official_args.sample_strategy,
        cf_embedding=cf_embeddings,
    )
    trainer = Trainer(official_args, model)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    best_loss, best_collision_rate = trainer.fit(loader)

    checkpoint_path = Path(trainer.ckpt_dir) / trainer.best_collision_ckpt
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device)
    sinkhorn_code_list = generate_letter_indices(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        max_collision_passes=args.max_collision_passes,
    )
    pre_resolution_counter = Counter(sinkhorn_code_list)
    collision_resolution: dict[str, Any] = {
        "method": "none",
        "repaired_prefixes": 0,
        "repaired_items": 0,
        "changed_items": 0,
        "max_repaired_prefix_size": 0,
        "assignment_cost": 0.0,
        "post_collision_groups": sum(count > 1 for count in pre_resolution_counter.values()),
    }
    resolved_code_list = sinkhorn_code_list
    if semantic_id_mode == SEMANTIC_ID_MODE_LEARNED:
        last_code_costs = compute_last_code_costs(
            model=model,
            dataset=dataset,
            all_indices=sinkhorn_code_list,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        resolved_code_list, assignment_audit = minimum_cost_unique_last_codes(
            sinkhorn_code_list,
            last_code_costs,
        )
        collision_resolution = {"method": "minimum_cost_unique_last_code", **assignment_audit}

    sinkhorn_codes = {sid: list(sinkhorn_code_list[index]) for index, sid in enumerate(skill_ids)}
    resolved_codes = {sid: list(resolved_code_list[index]) for index, sid in enumerate(skill_ids)}
    semantic_ids = (
        deduplicate_code_tuples(resolved_codes)
        if semantic_id_mode == SEMANTIC_ID_MODE_DEDUP
        else resolved_codes
    )
    validate_semantic_ids(semantic_ids, semantic_id_mode)
    code_width = max(3, len(str(max(max(values) for values in semantic_ids.values()))))
    rows_out = []
    for sid in sorted(skill_ids):
        output_row = {
            "skill_id": sid,
            "rq_codes": resolved_codes[sid],
            "semantic_id": semantic_ids[sid],
            "semantic_id_mode": semantic_id_mode,
            "tokens": semantic_id_tokens(semantic_ids[sid], code_width, semantic_id_mode),
        }
        if sinkhorn_codes[sid] != resolved_codes[sid]:
            output_row["pre_assignment_rq_codes"] = sinkhorn_codes[sid]
        rows_out.append(output_row)
    write_jsonl(args.output_dir / "semantic_ids.jsonl", rows_out)
    post_resolution_counter = Counter(tuple(values) for values in resolved_codes.values())
    write_json(
        args.output_dir / "rqvae_manifest.json",
        {
            "method": "official_letter_rqvae_adapter",
            "semantic_id_mode": semantic_id_mode,
            "letter_rqvae_dir": str(letter_dir),
            "skills": len(skill_ids),
            "rows": len(rows),
            "query_skill_interactions": len(interactions),
            "missing_skill_assignments": sum(missing.values()),
            "missing_skill_ids": len(missing),
            "num_emb_list": num_emb_list,
            "e_dim": args.e_dim,
            "layers": layers,
            "alpha": args.alpha,
            "beta": args.beta,
            "quant_loss_weight": args.quant_loss_weight,
            "sk_epsilons": sk_epsilons,
            "sk_iters": args.sk_iters,
            "kmeans_iters": args.kmeans_iters,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "eval_step": args.eval_step,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "diversity_loss_impl": args.diversity_loss_impl,
            "cf_epochs": args.cf_epochs,
            "cf_batch_size": args.cf_batch_size,
            "cf_negatives": args.cf_negatives,
            "cf_lr": args.cf_lr,
            "cf_weight_decay": args.cf_weight_decay,
            "cf_source": cf_source,
            "external_cf_embeddings": str(args.cf_embeddings) if args.cf_embeddings is not None else None,
            "final_cf_embedding_loss": final_cf_loss,
            "best_loss": best_loss,
            "best_collision_rate": best_collision_rate,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
            "collision_groups": sum(count > 1 for count in pre_resolution_counter.values()),
            "collided_items": sum(count for count in pre_resolution_counter.values() if count > 1),
            "max_collision": max(pre_resolution_counter.values()),
            "post_resolution_collision_groups": sum(
                count > 1 for count in post_resolution_counter.values()
            ),
            "post_resolution_collided_items": sum(
                count for count in post_resolution_counter.values() if count > 1
            ),
            "post_resolution_max_collision": max(post_resolution_counter.values()),
            "collision_resolution": collision_resolution,
            "dedup_max": (
                max(row["semantic_id"][-1] for row in rows_out)
                if semantic_id_mode == SEMANTIC_ID_MODE_DEDUP
                else None
            ),
            "level_usage": [
                len({values[level] for values in resolved_codes.values()})
                for level in range(len(num_emb_list))
            ],
            "pre_resolution_level_usage": [
                len({values[level] for values in sinkhorn_codes.values()})
                for level in range(len(num_emb_list))
            ],
            "code_width": code_width,
            "seed": args.seed,
        },
    )
    print(f"saved official LETTER semantic ids to {args.output_dir / 'semantic_ids.jsonl'}")


if __name__ == "__main__":
    main()
