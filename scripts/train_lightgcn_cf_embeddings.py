#!/usr/bin/env python3
"""Train LightGCN skill CF embeddings for LETTER.

This is a thin SkillGR adapter around the PyTorch LightGCN implementation
linked from the official LightGCN repository.  It converts query-skill rows
into an implicit-feedback bipartite graph, trains LightGCN with BPR loss, and
exports the final item embeddings as LETTER-compatible ``cf_emb``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from skill_gr_pipeline import collect_query_skill_interactions, load_jsonl, write_json, write_jsonl  # noqa: E402


class SkillGraphDataset:
    def __init__(
        self,
        n_users: int,
        n_items: int,
        interactions: list[tuple[int, int]],
        device: torch.device,
    ) -> None:
        self._n_users = n_users
        self._m_items = n_items
        self.interactions = interactions
        self.device = device
        self.Graph = self._build_sparse_graph().coalesce().to(device)

    @property
    def n_users(self) -> int:
        return self._n_users

    @property
    def m_items(self) -> int:
        return self._m_items

    def getSparseGraph(self):
        return self.Graph

    def _build_sparse_graph(self) -> torch.Tensor:
        src: list[int] = []
        dst: list[int] = []
        for user, item in self.interactions:
            item_node = self._n_users + item
            src.extend([user, item_node])
            dst.extend([item_node, user])
        if not src:
            raise ValueError("cannot build LightGCN graph without interactions")

        num_nodes = self._n_users + self._m_items
        src_t = torch.tensor(src, dtype=torch.long)
        dst_t = torch.tensor(dst, dtype=torch.long)
        degree = torch.zeros(num_nodes, dtype=torch.float32)
        degree.scatter_add_(0, src_t, torch.ones_like(src_t, dtype=torch.float32))
        degree.clamp_(min=1.0)
        values = torch.rsqrt(degree[src_t]) * torch.rsqrt(degree[dst_t])
        indices = torch.stack([src_t, dst_t], dim=0)
        return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightgcn-code-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device", default="")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-name", default="lightgcn_cf_embeddings.pt")
    return parser.parse_args()


def load_skill_ids(path: Path) -> list[str]:
    payload = torch.load(path, map_location="cpu")
    if "skill_ids" not in payload:
        raise KeyError(f"{path} does not contain skill_ids")
    return [str(sid) for sid in payload["skill_ids"]]


def positives_by_user(interactions: list[tuple[int, int]], n_users: int) -> list[set[int]]:
    positives = [set() for _ in range(n_users)]
    for user, item in interactions:
        positives[user].add(item)
    return positives


def sample_negatives(
    users: torch.Tensor,
    positives: list[set[int]],
    n_items: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    neg_items = rng.integers(0, n_items, size=len(users), endpoint=False)
    users_np = users.cpu().numpy()
    for idx, user in enumerate(users_np):
        # Collisions are rare at this item count; resample only collided draws.
        while int(neg_items[idx]) in positives[int(user)]:
            neg_items[idx] = rng.integers(0, n_items, endpoint=False)
    return torch.from_numpy(neg_items.astype(np.int64))


def import_lightgcn(code_dir: Path, device: torch.device):
    code_dir = code_dir.resolve()
    if not (code_dir / "model.py").exists() or not (code_dir / "world.py").exists():
        raise SystemExit(f"invalid LightGCN-PyTorch code directory: {code_dir}")
    sys.path.insert(0, str(code_dir))
    # LightGCN-PyTorch parses CLI arguments at import time via world.py.
    # Hide this adapter's arguments while importing the external code.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        import world  # type: ignore
        from model import LightGCN  # type: ignore
    finally:
        sys.argv = original_argv

    world.device = device
    return LightGCN


def cooccurrence_diagnostics(
    item_embeddings: torch.Tensor,
    interactions: list[tuple[int, int]],
    seed: int,
    max_pairs: int = 50000,
) -> dict[str, Any]:
    query_to_items: dict[int, list[int]] = {}
    for user, item in interactions:
        query_to_items.setdefault(user, []).append(item)

    co_pairs: list[tuple[int, int]] = []
    for items in query_to_items.values():
        unique_items = sorted(set(items))
        if len(unique_items) < 2:
            continue
        for left_index, left in enumerate(unique_items):
            for right in unique_items[left_index + 1 :]:
                co_pairs.append((left, right))

    if not co_pairs:
        return {"cooccurrence_pairs": 0}

    rng = np.random.default_rng(seed)
    if len(co_pairs) > max_pairs:
        selected = rng.choice(len(co_pairs), size=max_pairs, replace=False)
        co_pairs = [co_pairs[int(i)] for i in selected]

    random_pairs = []
    n_items = item_embeddings.shape[0]
    while len(random_pairs) < len(co_pairs):
        left = int(rng.integers(0, n_items))
        right = int(rng.integers(0, n_items))
        if left != right:
            random_pairs.append((left, right))

    emb = torch.nn.functional.normalize(item_embeddings.float(), p=2, dim=1).cpu()

    def pair_stats(pairs: list[tuple[int, int]]) -> dict[str, float]:
        left = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        right = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        sims = (emb[left] * emb[right]).sum(dim=-1).numpy()
        return {
            "mean": float(np.mean(sims)),
            "p50": float(np.percentile(sims, 50)),
            "p75": float(np.percentile(sims, 75)),
            "p90": float(np.percentile(sims, 90)),
        }

    return {
        "cooccurrence_pairs": int(len(co_pairs)),
        "cooccurrence_cosine": pair_stats(co_pairs),
        "random_cosine": pair_stats(random_pairs),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    LightGCN = import_lightgcn(args.lightgcn_code_dir, device)

    skill_ids = load_skill_ids(args.embeddings)
    skill_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    rows = load_jsonl(args.rows)
    interactions, missing = collect_query_skill_interactions(rows, skill_to_index)
    if not interactions:
        raise SystemExit("no query-skill interactions found for LightGCN training")

    n_users = len(rows)
    n_items = len(skill_ids)
    positives = positives_by_user(interactions, n_users)
    item_degree = Counter(item for _, item in interactions)
    user_degree = Counter(user for user, _ in interactions)

    dataset = SkillGraphDataset(n_users=n_users, n_items=n_items, interactions=interactions, device=device)
    config = {
        "latent_dim_rec": args.dim,
        "lightGCN_n_layers": args.layers,
        "keep_prob": 1.0,
        "A_split": False,
        "dropout": False,
        "pretrain": 0,
    }
    model = LightGCN(config, dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    pos_users = torch.tensor([user for user, _ in interactions], dtype=torch.long)
    pos_items = torch.tensor([item for _, item in interactions], dtype=torch.long)
    steps_per_epoch = max(1, math.ceil(len(interactions) / args.batch_size))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    final_loss = None
    final_bpr = None
    final_reg = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(interactions))
        losses: list[float] = []
        bpr_losses: list[float] = []
        reg_losses: list[float] = []
        for start in range(0, len(interactions), args.batch_size):
            batch_order = order[start : start + args.batch_size]
            users = pos_users[batch_order]
            pos = pos_items[batch_order]
            neg = sample_negatives(users, positives, n_items, rng)

            users_d = users.to(device)
            pos_d = pos.to(device)
            neg_d = neg.to(device)
            bpr_loss, reg_loss = model.bpr_loss(users_d, pos_d, neg_d)
            loss = bpr_loss + args.reg * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            bpr_losses.append(float(bpr_loss.detach().cpu()))
            reg_losses.append(float(reg_loss.detach().cpu()))

        final_loss = float(np.mean(losses))
        final_bpr = float(np.mean(bpr_losses))
        final_reg = float(np.mean(reg_losses))
        row = {
            "epoch": epoch,
            "step": epoch * steps_per_epoch,
            "loss": final_loss,
            "bpr_loss": final_bpr,
            "reg_loss": final_reg,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} step={epoch * steps_per_epoch} "
                f"loss={final_loss:.6f} bpr={final_bpr:.6f} reg={final_reg:.6f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        _, item_embeddings = model.computer()
        item_embeddings = torch.nn.functional.normalize(item_embeddings.detach().cpu().float(), p=2, dim=1)

    cf_path = args.output_dir / args.save_name
    torch.save(item_embeddings, cf_path)
    write_jsonl(args.output_dir / "skill_ids.jsonl", [{"index": i, "skill_id": sid} for i, sid in enumerate(skill_ids)])

    diagnostics = cooccurrence_diagnostics(item_embeddings, interactions, seed=args.seed)
    manifest = {
        "source": "LightGCN-PyTorch linked by official LightGCN README",
        "lightgcn_code_dir": str(args.lightgcn_code_dir),
        "embeddings": str(args.embeddings),
        "rows": str(args.rows),
        "cf_embeddings": str(cf_path),
        "n_queries": n_users,
        "n_skills": n_items,
        "interactions": len(interactions),
        "missing_selected_skills": dict(missing),
        "covered_skills": len(item_degree),
        "zero_degree_skills": n_items - len(item_degree),
        "user_degree": {
            "min": int(min(user_degree.values())) if user_degree else 0,
            "max": int(max(user_degree.values())) if user_degree else 0,
            "mean": float(np.mean(list(user_degree.values()))) if user_degree else 0.0,
        },
        "item_degree": {
            "min_nonzero": int(min(item_degree.values())) if item_degree else 0,
            "max": int(max(item_degree.values())) if item_degree else 0,
            "mean_nonzero": float(np.mean(list(item_degree.values()))) if item_degree else 0.0,
        },
        "dim": args.dim,
        "layers": args.layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "reg": args.reg,
        "seed": args.seed,
        "final_loss": final_loss,
        "final_bpr_loss": final_bpr,
        "final_reg_loss": final_reg,
        "diagnostics": diagnostics,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"saved LightGCN cf embeddings to {cf_path}", flush=True)


if __name__ == "__main__":
    main()
