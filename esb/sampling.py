"""Edge splits, exposure policies, and deconfounded graph variants.

The simulated policy keeps a preference edge for item i with probability

    p_i = clip((pop_i / max pop) ** alpha, min_prop, 1).

`raw` is the graph of edges that survive that coin flip.
`uniform_matched` draws the same number of edges from the full preference
graph, so a gap between it and `raw` is the popularity skew rather than the
smaller sample. `resampled` keeps exposed edges with probability proportional
to 1/p_i, which changes the edge set instead of only reweighting it.
"""

from __future__ import annotations

import numpy as np


def item_propensity(popularity: np.ndarray, alpha: float, min_prop: float) -> np.ndarray:
    pop = np.asarray(popularity, dtype=np.float64)
    if pop.size == 0 or float(pop.max()) <= 0:
        return np.ones(pop.shape, dtype=np.float64)
    scaled = (pop / pop.max()) ** alpha
    return np.clip(scaled, min_prop, 1.0)


def uniform_propensity(n_items: int, keep_prob: float) -> np.ndarray:
    keep = float(np.clip(keep_prob, 1e-3, 1.0))
    return np.full(n_items, keep, dtype=np.float64)


def edge_keys(edges: np.ndarray, n_items: int) -> np.ndarray:
    edges = np.asarray(edges)
    if len(edges) == 0:
        return np.zeros(0, dtype=np.int64)
    return edges[:, 0].astype(np.int64) * np.int64(n_items) + edges[:, 1].astype(np.int64)


def unique_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(edges, axis=0)


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(parts, axis=0)


def k_core(edges: np.ndarray, min_user: int, min_item: int, rounds: int = 10) -> np.ndarray:
    edges = unique_edges(edges)
    if len(edges) == 0 or (min_user <= 1 and min_item <= 1):
        return edges
    for _ in range(rounds):
        users, user_counts = np.unique(edges[:, 0], return_counts=True)
        items, item_counts = np.unique(edges[:, 1], return_counts=True)
        keep_users = users[user_counts >= min_user]
        keep_items = items[item_counts >= min_item]
        mask = np.isin(edges[:, 0], keep_users) & np.isin(edges[:, 1], keep_items)
        nxt = edges[mask]
        if len(nxt) == len(edges):
            return nxt
        edges = nxt
    return edges


def remap_pairs(arrays: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    """Map raw ids appearing in any array onto contiguous ranges."""
    usable = [np.asarray(a, dtype=np.int64) for a in arrays if len(a)]
    if not usable:
        return [np.zeros((0, 2), dtype=np.int64) for _ in arrays], 0, 0
    users = np.unique(np.concatenate([a[:, 0] for a in usable]))
    items = np.unique(np.concatenate([a[:, 1] for a in usable]))
    mapped = []
    for array in arrays:
        array = np.asarray(array, dtype=np.int64)
        if len(array) == 0:
            mapped.append(np.zeros((0, 2), dtype=np.int64))
            continue
        user_idx = np.searchsorted(users, array[:, 0])
        item_idx = np.searchsorted(items, array[:, 1])
        if np.any(users[user_idx] != array[:, 0]) or np.any(items[item_idx] != array[:, 1]):
            raise RuntimeError("Found an id that was not included in the remap.")
        mapped.append(np.column_stack([user_idx, item_idx]).astype(np.int64))
    return mapped, int(len(users)), int(len(items))


def split_by_user(
    edges: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
    min_train: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-user split. Users with fewer than `min_train` edges stay in train."""
    edges = unique_edges(edges)
    empty = np.zeros((0, 2), dtype=np.int64)
    if len(edges) == 0:
        return empty, empty, empty
    order = np.argsort(edges[:, 0], kind="mergesort")
    edges = edges[order]
    _, starts, counts = np.unique(edges[:, 0], return_index=True, return_counts=True)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for start, count in zip(starts, counts):
        block = edges[start : start + count]
        block = block[rng.permutation(int(count))]
        if int(count) <= min_train:
            train_parts.append(block)
            continue
        n_test = max(1, int(round(count * test_ratio)))
        n_val = int(round(count * val_ratio))
        max_holdout = int(count) - min_train
        if n_test > max_holdout:
            n_test = max_holdout
            n_val = 0
        else:
            n_val = min(n_val, max_holdout - n_test)
            if n_val == 0 and max_holdout - n_test >= 1 and int(count) >= min_train + 2:
                n_val = 1
                if n_test > max_holdout - n_val:
                    n_test = max_holdout - n_val
        test_parts.append(block[:n_test])
        if n_val:
            val_parts.append(block[n_test : n_test + n_val])
        train_parts.append(block[n_test + n_val :])
    return _concat(train_parts), _concat(val_parts), _concat(test_parts)


def ensure_min_per_user(
    source: np.ndarray,
    kept: np.ndarray,
    rng: np.random.Generator,
    n_items: int,
    min_keep: int = 1,
) -> tuple[np.ndarray, int]:
    """Add source edges back until every source user has `min_keep` kept edges."""
    source = unique_edges(source)
    kept = unique_edges(kept)
    if len(source) == 0 or min_keep <= 0:
        return kept, 0
    source_ids = edge_keys(source, n_items)
    kept_ids = edge_keys(kept, n_items)
    kept_set = set(kept_ids.tolist())
    by_user: dict[int, list[int]] = {}
    for edge_id, user in zip(source_ids.tolist(), source[:, 0].tolist()):
        by_user.setdefault(int(user), []).append(int(edge_id))
    kept_count: dict[int, int] = {}
    for user in kept[:, 0].tolist() if len(kept) else []:
        kept_count[int(user)] = kept_count.get(int(user), 0) + 1
    forced: list[int] = []
    for user, ids in by_user.items():
        have = kept_count.get(user, 0)
        if have >= min_keep:
            continue
        missing = [edge_id for edge_id in ids if edge_id not in kept_set]
        rng.shuffle(missing)
        need = min_keep - have
        forced.extend(missing[:need])
    if not forced:
        return kept, 0
    extra = np.column_stack(
        (
            np.array(forced, dtype=np.int64) // np.int64(n_items),
            np.array(forced, dtype=np.int64) % np.int64(n_items),
        )
    )
    return unique_edges(np.concatenate([kept, extra], axis=0)), int(len(forced))


def apply_exposure(
    edges: np.ndarray,
    propensity: np.ndarray,
    rng: np.random.Generator,
    n_items: int,
    min_keep: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep each edge with probability propensity[item]. Returns edges, edge propensities, n_forced."""
    edges = unique_edges(edges)
    if len(edges) == 0:
        return edges, np.zeros(0, dtype=np.float64), 0
    edge_p = np.asarray(propensity, dtype=np.float64)[edges[:, 1]]
    if np.any(edge_p <= 0) or np.any(edge_p > 1):
        raise ValueError("Exposure propensities must lie in (0, 1].")
    keep = rng.random(len(edges)) < edge_p
    kept = edges[keep]
    kept, n_forced = ensure_min_per_user(edges, kept, rng, n_items, min_keep=min_keep)
    kept_p = np.asarray(propensity, dtype=np.float64)[kept[:, 1]]
    return kept, kept_p, n_forced


def uniform_subsample(
    edges: np.ndarray,
    target: int,
    rng: np.random.Generator,
    n_items: int,
    min_keep: int = 1,
) -> np.ndarray:
    edges = unique_edges(edges)
    if target <= 0 or len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if target >= len(edges):
        return edges
    chosen = edges[rng.choice(len(edges), size=target, replace=False)]
    chosen, _ = ensure_min_per_user(edges, chosen, rng, n_items, min_keep=min_keep)
    return chosen


def inverse_propensity_keep_probs(propensity: np.ndarray, target_mean: float) -> np.ndarray:
    """Keep probabilities proportional to 1/p, capped at 1, with the requested mean."""
    prop = np.clip(np.asarray(propensity, dtype=np.float64), 1e-8, None)
    inv = 1.0 / prop
    target = float(np.clip(target_mean, 1e-3, 1.0))
    # The scale has to exceed max(1/p) when the target keep rate is below 1.
    lo, hi = 1e-8, max(float(inv.max()) / target, 1e-8)
    for _ in range(60):
        scale = (lo + hi) / 2.0
        mean = float(np.minimum(1.0, inv / scale).mean())
        if mean > target:
            lo = scale
        else:
            hi = scale
    return np.minimum(1.0, inv / hi)


def resample_by_inverse_propensity(
    edges: np.ndarray,
    edge_propensity: np.ndarray,
    rng: np.random.Generator,
    n_items: int,
    target_mean: float = 0.5,
    min_keep: int = 1,
) -> np.ndarray:
    edges = unique_edges(edges)
    if len(edges) == 0:
        return edges
    keep_p = inverse_propensity_keep_probs(edge_propensity, target_mean)
    kept = edges[rng.random(len(edges)) < keep_p]
    kept, _ = ensure_min_per_user(edges, kept, rng, n_items, min_keep=min_keep)
    return kept


def ips_weights(propensity: np.ndarray, clip: float) -> np.ndarray:
    prop = np.clip(np.asarray(propensity, dtype=np.float64), 1e-8, None)
    weights = np.minimum(1.0 / prop, clip)
    return weights.astype(np.float32)


def assign_ips(propensity: np.ndarray, mode: str, clip: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (graph_weight, loss_weight) for one of graph / loss / both."""
    weights = ips_weights(propensity, clip)
    ones = np.ones_like(weights)
    if mode == "graph":
        return weights, ones
    if mode == "loss":
        return ones, weights
    if mode == "both":
        return weights, weights
    raise ValueError(f"Unknown ips mode: {mode}")


def head_mask(counts: np.ndarray, frac: float = 0.2) -> np.ndarray:
    counts = np.asarray(counts)
    mask = np.zeros(len(counts), dtype=bool)
    if len(counts) == 0:
        return mask
    n_head = max(1, int(round(len(counts) * frac)))
    order = np.argsort(-counts, kind="mergesort")
    mask[order[:n_head]] = True
    return mask


def edge_share_on_items(edges: np.ndarray, item_mask: np.ndarray) -> float:
    if len(edges) == 0:
        return float("nan")
    return float(item_mask[edges[:, 1]].mean())


def exposure_buckets(counts: np.ndarray, present: np.ndarray | None = None) -> np.ndarray:
    """Terciles of exposure among items that occur in training.

    0 = tail, 1 = mid, 2 = head, and -1 = absent from the training split.
    Absent items are not dumped into the tail, because that would make the
    robustness slice a set of items the model never saw.
    """
    counts = np.asarray(counts)
    buckets = np.full(len(counts), -1, dtype=np.int8)
    if present is None:
        observed = np.flatnonzero(counts > 0)
    else:
        observed = np.flatnonzero(np.asarray(present, dtype=bool))
    if len(observed) == 0:
        return buckets
    order = observed[np.argsort(counts[observed], kind="mergesort")]
    first = len(order) // 3
    second = 2 * len(order) // 3
    buckets[order[:first]] = 0
    buckets[order[first:second]] = 1
    buckets[order[second:]] = 2
    return buckets


def drop_pairs(edges: np.ndarray, banned: np.ndarray, n_items: int) -> np.ndarray:
    """Remove any edge that also appears in `banned`."""
    edges = unique_edges(edges)
    banned = unique_edges(banned)
    if len(edges) == 0 or len(banned) == 0:
        return edges
    ban_keys = np.sort(edge_keys(banned, n_items))
    keys = edge_keys(edges, n_items)
    pos = np.searchsorted(ban_keys, keys)
    pos = np.clip(pos, 0, len(ban_keys) - 1)
    return edges[ban_keys[pos] != keys]


def item_counts(edges: np.ndarray, n_items: int) -> np.ndarray:
    if len(edges) == 0:
        return np.zeros(n_items, dtype=np.float64)
    return np.bincount(edges[:, 1], minlength=n_items).astype(np.float64)


def aggregate_weights(edges: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(edges, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    weights = weights[order]
    if len(edges) == 1:
        return edges, weights.astype(np.float32)
    change = np.any(np.diff(edges, axis=0), axis=1)
    starts = np.concatenate(([0], np.flatnonzero(change) + 1))
    summed = np.add.reduceat(weights, starts)
    counts = np.diff(np.concatenate((starts, [len(edges)])))
    return edges[starts], (summed / counts).astype(np.float32)
