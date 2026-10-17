"""BPR training for one graph variant, with early stopping on unbiased validation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from esb.metrics import evaluate_ranking
from esb.model import LightGCN, normalized_adjacency
from esb.sampling import aggregate_weights


@dataclass
class Variant:
    name: str
    edges: np.ndarray
    graph_weight: np.ndarray
    loss_weight: np.ndarray
    n_layers: int


@dataclass
class TrainedVariant:
    name: str
    user_emb: np.ndarray
    item_emb: np.ndarray
    test_metrics: dict
    best_epoch: int
    best_val_ndcg: float
    val_history: list[float]
    seconds: float
    n_edges: int
    n_layers: int


def train_variant(
    variant: Variant,
    n_users: int,
    n_items: int,
    ban: np.ndarray,
    val_relevant: np.ndarray,
    test_relevant: np.ndarray,
    buckets: np.ndarray,
    forbidden: np.ndarray,
    cfg,
    device: str,
    seed: int,
) -> TrainedVariant:
    edges, graph_weight = aggregate_weights(variant.edges, variant.graph_weight)
    _, loss_weight = aggregate_weights(variant.edges, variant.loss_weight)
    if len(edges) < 2:
        raise RuntimeError(f"{variant.name} has {len(edges)} edges.")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    adjacency = normalized_adjacency(n_users, n_items, edges, graph_weight)
    model = LightGCN(n_users, n_items, cfg.dim, adjacency).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed + 17)
    loss_weight_t = torch.as_tensor(loss_weight, dtype=torch.float32)
    best_state = None
    best_val = float("-inf")
    best_epoch = 0
    since_improve = 0
    history: list[float] = []
    started = time.perf_counter()
    metric_k = max(cfg.ks)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(len(edges))
        total = 0.0
        batches = 0
        for start in range(0, len(edges), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            users = edges[batch, 0]
            positive = edges[batch, 1]
            negative = sample_negatives(users, forbidden, rng)
            weights = loss_weight_t[batch]
            loss = _bpr(model, users, positive, negative, weights, variant.n_layers, cfg.l2, device)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{variant.name} produced a non-finite loss at epoch {epoch}. "
                    "Lower --lr or --ips_clip."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        val_ndcg = float("nan")
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            user_emb, item_emb = _embed(model, variant.n_layers)
            val_metrics = evaluate_ranking(
                user_emb,
                item_emb,
                ban,
                val_relevant,
                buckets,
                ks=(metric_k,),
                device=device,
                with_buckets=False,
            )
            val_ndcg = val_metrics[f"ndcg@{metric_k}"]
            history.append(val_ndcg)
            if np.isnan(val_ndcg):
                improved = False
            else:
                improved = val_ndcg > best_val
            if improved:
                best_val = val_ndcg
                best_epoch = epoch
                since_improve = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                since_improve += 1
            if since_improve >= cfg.patience and best_state is not None:
                print(
                    f"  {variant.name:18s} epoch {epoch:3d}  loss {total / max(batches, 1):.4f}  "
                    f"val_ndcg@{metric_k} {val_ndcg:.4f}",
                    flush=True,
                )
                print(f"  {variant.name}: early stop at epoch {epoch}", flush=True)
                break
        print(
            f"  {variant.name:18s} epoch {epoch:3d}  loss {total / max(batches, 1):.4f}  "
            f"val_ndcg@{metric_k} {val_ndcg:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    user_emb, item_emb = _embed(model, variant.n_layers)
    test_metrics = evaluate_ranking(
        user_emb,
        item_emb,
        ban,
        test_relevant,
        buckets,
        ks=tuple(cfg.ks),
        device=device,
        with_buckets=True,
    )
    return TrainedVariant(
        name=variant.name,
        user_emb=user_emb,
        item_emb=item_emb,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        best_val_ndcg=None if best_val == float("-inf") else float(best_val),
        val_history=history,
        seconds=time.perf_counter() - started,
        n_edges=int(len(edges)),
        n_layers=variant.n_layers,
    )


def sample_negatives(users: np.ndarray, forbidden: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_items = forbidden.shape[1]
    negative = rng.integers(0, n_items, size=len(users))
    for _ in range(30):
        bad = forbidden[users, negative]
        n_bad = int(bad.sum())
        if n_bad == 0:
            return negative
        negative[bad] = rng.integers(0, n_items, size=n_bad)
    bad = forbidden[users, negative]
    for row in np.flatnonzero(bad):
        allowed = np.flatnonzero(~forbidden[int(users[row])])
        if len(allowed) == 0:
            continue
        negative[row] = allowed[int(rng.integers(0, len(allowed)))]
    return negative


def _bpr(model, users, positive, negative, weights, n_layers, l2, device) -> torch.Tensor:
    users_t = torch.as_tensor(users, dtype=torch.long, device=device)
    positive_t = torch.as_tensor(positive, dtype=torch.long, device=device)
    negative_t = torch.as_tensor(negative, dtype=torch.long, device=device)
    weights_t = weights.to(device)
    user_emb, item_emb = model.encode(n_layers)
    pos_score = (user_emb[users_t] * item_emb[positive_t]).sum(dim=-1)
    neg_score = (user_emb[users_t] * item_emb[negative_t]).sum(dim=-1)
    weighted = weights_t / weights_t.sum().clamp_min(1e-8)
    ranking = -(weighted * nn.functional.logsigmoid(pos_score - neg_score)).sum()
    ego = model.embedding.weight
    regularizer = (
        ego[users_t].pow(2).mean()
        + ego[positive_t + model.n_users].pow(2).mean()
        + ego[negative_t + model.n_users].pow(2).mean()
    )
    return ranking + l2 * regularizer


def _embed(model: LightGCN, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        users, items = model.encode(n_layers)
    return users.detach().cpu().numpy(), items.detach().cpu().numpy()
