"""Ranking, robustness, and neighborhood metrics.

Leakage is the overlap of a model's item neighbors with the exposure
co-occurrence neighborhood, minus their overlap with the preference
co-occurrence neighborhood. Positive leakage means the embedding recovered
what was shown together rather than what was preferred together.
"""

from __future__ import annotations

import numpy as np
import torch


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA. Invariant to orthogonal transforms and isotropic scaling."""
    x_t = torch.as_tensor(x, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    x_t = x_t - x_t.mean(dim=0, keepdim=True)
    y_t = y_t - y_t.mean(dim=0, keepdim=True)
    xty = x_t.T @ y_t
    xtx = x_t.T @ x_t
    yty = y_t.T @ y_t
    hsic_xy = torch.sum(xty * xty)
    hsic_xx = torch.sum(xtx * xtx)
    hsic_yy = torch.sum(yty * yty)
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if float(denom) < 1e-12:
        return float("nan")
    return float(hsic_xy / denom)


def embedding_knn(embeddings: np.ndarray, k: int, block: int = 2048) -> np.ndarray:
    vectors = torch.as_tensor(embeddings, dtype=torch.float32)
    norms = vectors.norm(dim=1, keepdim=True).clamp_min(1e-8)
    vectors = vectors / norms
    n_items, _ = vectors.shape
    k_eff = min(k, max(n_items - 1, 1))
    if n_items <= 1:
        return np.full((n_items, 1), -1, dtype=np.int64)
    neighbors = np.empty((n_items, k_eff), dtype=np.int64)
    for start in range(0, n_items, block):
        stop = min(n_items, start + block)
        similarity = vectors[start:stop] @ vectors.T
        rows = torch.arange(stop - start)
        similarity[rows, torch.arange(start, stop)] = -1e9
        neighbors[start:stop] = torch.topk(similarity, k_eff, dim=1).indices.cpu().numpy()
    return neighbors


def interaction_knn(
    edges: np.ndarray,
    n_users: int,
    n_items: int,
    k: int,
    min_degree: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Item–item cosine neighbors from a binary user–item graph.

    The returned array always has `k` columns. Missing neighbors are -1.
    """
    from scipy import sparse

    neighbors = np.full((n_items, max(k, 1)), -1, dtype=np.int64)
    valid = np.zeros(n_items, dtype=bool)
    if n_items <= 1 or len(edges) == 0 or k < 1:
        return neighbors, valid
    matrix = sparse.csr_matrix(
        (np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(n_users, n_items),
    )
    matrix.data[:] = 1.0
    degree = np.asarray(matrix.sum(axis=0)).ravel()
    valid = degree >= min_degree
    n_valid = int(valid.sum())
    k_eff = min(k, n_valid - 1)
    if k_eff < 1:
        return neighbors, np.zeros(n_items, dtype=bool)
    norms = np.sqrt(np.maximum(degree, 0)).astype(np.float32)
    gram = (matrix.T @ matrix).toarray().astype(np.float32)
    np.fill_diagonal(gram, 0.0)
    denom = np.maximum(norms[:, None] * norms[None, :], 1e-8)
    # Zero co-occurrence is not a neighbor. Filling the rest of the top-k with
    # arbitrary items would make the Jaccard and popularity checks noise.
    cosine = np.where(gram > 0, gram / denom, -np.inf)
    blocked = ~valid
    cosine[blocked, :] = -np.inf
    cosine[:, blocked] = -np.inf
    valid = valid & np.isfinite(cosine).any(axis=1)
    n_valid = int(valid.sum())
    k_eff = min(k, n_valid - 1)
    if k_eff < 1:
        return neighbors, np.zeros(n_items, dtype=bool)
    usable = np.flatnonzero(valid)
    part = np.argpartition(-cosine[usable], kth=k_eff - 1, axis=1)[:, :k_eff]
    chosen = np.take_along_axis(cosine[usable], part, axis=1)
    part = part.astype(np.int64)
    part[~np.isfinite(chosen)] = -1
    neighbors[usable, :k_eff] = part
    return neighbors, valid


def mean_jaccard(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    scores = []
    for index in np.where(valid)[0]:
        a = {int(v) for v in left[index].tolist() if int(v) >= 0}
        b = {int(v) for v in right[index].tolist() if int(v) >= 0}
        if not a or not b:
            continue
        scores.append(len(a & b) / len(a | b))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    if float(np.std(left)) < 1e-8 or float(np.std(right)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def knn_popularity_corr(neighbors: np.ndarray, popularity: np.ndarray, valid: np.ndarray) -> float:
    if int(valid.sum()) < 3:
        return float("nan")
    rows = neighbors[valid]
    mask = rows >= 0
    counts = mask.sum(axis=1)
    keep = counts > 0
    if int(keep.sum()) < 3:
        return float("nan")
    safe_rows = np.where(mask, rows, 0)
    neighbor_pop = (popularity[safe_rows] * mask).sum(axis=1)[keep] / counts[keep]
    return safe_corr(popularity[valid][keep], neighbor_pop)


def neighborhood_report(
    embedding_neighbors: np.ndarray,
    preference_neighbors: np.ndarray,
    exposure_neighbors: np.ndarray,
    valid: np.ndarray,
    popularity: np.ndarray,
) -> dict:
    prefer = mean_jaccard(embedding_neighbors, preference_neighbors, valid)
    expose = mean_jaccard(embedding_neighbors, exposure_neighbors, valid)
    return {
        "jaccard_preference": prefer,
        "jaccard_exposure": expose,
        "leakage": expose - prefer,
        "knn_popularity_corr": knn_popularity_corr(embedding_neighbors, popularity, valid),
        "n_items": int(valid.sum()),
    }


def _idcg_table(max_k: int) -> np.ndarray:
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))
    prefix = np.cumsum(discounts)
    table = np.zeros(max_k + 1, dtype=np.float64)
    table[1:] = prefix
    return table


def evaluate_ranking(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    ban: np.ndarray,
    relevant: np.ndarray,
    buckets: np.ndarray,
    ks: tuple[int, ...] = (10, 20),
    device: str = "cpu",
    batch_size: int = 512,
    with_buckets: bool = True,
) -> dict:
    """Full-catalog ranking. `ban` and `relevant` are boolean [n_users, n_items]."""
    users = np.flatnonzero(relevant.any(axis=1))
    max_k = max(ks)
    max_k = min(max_k, item_emb.shape[0])
    idcg = _idcg_table(max_k)
    bucket_names = {0: "tail", 1: "mid", 2: "head"}
    totals = {f"{name}@{k}": [] for k in ks for name in ("recall", "ndcg")}
    for bucket_name in bucket_names.values():
        for metric in ("recall", "ndcg"):
            totals[f"{bucket_name}_{metric}@{max(ks)}"] = []
    if len(users) == 0 or max_k < 1:
        return _empty_ranking(ks)

    user_t = torch.as_tensor(user_emb, dtype=torch.float32, device=device)
    item_t = torch.as_tensor(item_emb, dtype=torch.float32, device=device)
    ban_t = torch.as_tensor(ban, dtype=torch.bool, device=device)
    rel_t = torch.as_tensor(relevant, dtype=torch.bool, device=device)
    bucket_t = torch.as_tensor(buckets, dtype=torch.int8, device=device)

    with torch.inference_mode():
        for start in range(0, len(users), batch_size):
            batch = users[start : start + batch_size]
            batch_t = torch.as_tensor(batch, dtype=torch.long, device=device)
            scores = user_t[batch_t] @ item_t.T
            scores = scores.masked_fill(ban_t[batch_t], -1e9)
            topk = torch.topk(scores, max_k, dim=1).indices
            rel_batch = rel_t[batch_t]
            n_rel = rel_batch.sum(dim=1)
            hit = rel_batch.gather(1, topk).float()
            _accumulate(totals, hit, n_rel, ks, idcg, prefix="")
            if not with_buckets:
                continue
            for bucket_id, bucket_name in bucket_names.items():
                in_bucket = bucket_t == bucket_id
                rel_bucket = rel_batch & in_bucket.unsqueeze(0)
                n_bucket = rel_bucket.sum(dim=1)
                hit_bucket = rel_bucket.gather(1, topk).float()
                _accumulate(
                    totals,
                    hit_bucket,
                    n_bucket,
                    (max(ks),),
                    idcg,
                    prefix=f"{bucket_name}_",
                )
    result = {key: float(np.mean(values)) if values else float("nan") for key, values in totals.items()}
    result["n_eval_users"] = int(len(users))
    gap_head = result.get(f"head_ndcg@{max(ks)}", float("nan"))
    gap_tail = result.get(f"tail_ndcg@{max(ks)}", float("nan"))
    result[f"head_tail_gap@{max(ks)}"] = gap_head - gap_tail
    return result


def _accumulate(totals, hit, n_rel, ks, idcg, prefix: str) -> None:
    keep = n_rel > 0
    if int(keep.sum()) == 0:
        return
    hit = hit[keep]
    n_rel = n_rel[keep]
    for k in ks:
        k_hit = hit[:, :k]
        discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=hit.device, dtype=torch.float32))
        dcg = (k_hit * discounts).sum(dim=1)
        n_at_k = torch.minimum(n_rel, torch.tensor(k, device=hit.device))
        ideal = torch.as_tensor(idcg, dtype=torch.float32, device=hit.device)[n_at_k.long()]
        ndcg = dcg / ideal.clamp_min(1e-8)
        recall = k_hit.sum(dim=1) / n_rel.clamp_min(1).float()
        totals[f"{prefix}recall@{k}"].extend(recall.cpu().tolist())
        totals[f"{prefix}ndcg@{k}"].extend(ndcg.cpu().tolist())


def _empty_ranking(ks: tuple[int, ...]) -> dict:
    max_k = max(ks)
    result = {f"{name}@{k}": float("nan") for k in ks for name in ("recall", "ndcg")}
    for bucket in ("tail", "mid", "head"):
        result[f"{bucket}_recall@{max_k}"] = float("nan")
        result[f"{bucket}_ndcg@{max_k}"] = float("nan")
    result[f"head_tail_gap@{max_k}"] = float("nan")
    result["n_eval_users"] = 0
    return result


def pairs_to_mask(n_users: int, n_items: int, edges: np.ndarray) -> np.ndarray:
    mask = np.zeros((n_users, n_items), dtype=bool)
    if len(edges):
        mask[edges[:, 0], edges[:, 1]] = True
    return mask
