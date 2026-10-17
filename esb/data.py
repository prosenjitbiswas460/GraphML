"""Load KuaiRec, KuaiRand-Pure, and MovieLens-1M into one split format."""

from __future__ import annotations

import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from esb.sampling import (
    apply_exposure,
    drop_pairs,
    edge_keys,
    edge_share_on_items,
    head_mask,
    item_counts,
    item_propensity,
    k_core,
    remap_pairs,
    resample_by_inverse_propensity,
    split_by_user,
    uniform_propensity,
    uniform_subsample,
    unique_edges,
)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
KUAIREC_URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
KUAIRAND_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"

MAX_USERS = 100_000
MAX_ITEMS = 20_000


@dataclass
class PreparedSplit:
    dataset: str
    n_users: int
    n_items: int
    raw_edges: np.ndarray
    raw_propensity: np.ndarray
    uniform_edges: np.ndarray | None
    resampled_edges: np.ndarray
    counterfactual_edges: np.ndarray
    val_edges: np.ndarray
    test_edges: np.ndarray
    policy_popularity: np.ndarray
    meta: dict = field(default_factory=dict)
    shift_edges: np.ndarray | None = None


def prepare(cfg, seed: int) -> PreparedSplit:
    rng = np.random.default_rng(seed)
    dataset = cfg.dataset
    if dataset == "synthetic":
        align = float(np.clip(getattr(cfg, "exposure_align", 0.0), 0.0, 1.0))
        edges, popularity, exposure_info = make_synthetic(rng, align=align)
        split = prepare_simulated(edges, rng, cfg, popularity, "synthetic", "known_policy")
        split.meta.update(exposure_info)
        return split
    if dataset == "movielens":
        ensure_movielens(Path(cfg.data_dir), cfg.download)
        edges = load_movielens(Path(cfg.data_dir), cfg.rating_threshold)
        return prepare_simulated(edges, rng, cfg, None, "movielens", "known_policy")
    if dataset == "kuairec":
        ensure_kuairec(Path(cfg.data_dir), cfg.download)
        edges = load_kuairec_positives(Path(cfg.data_dir), cfg.watch_threshold, which="small")
        return prepare_simulated(edges, rng, cfg, None, "kuairec", "known_policy")
    if dataset == "kuairec_natural":
        ensure_kuairec(Path(cfg.data_dir), cfg.download)
        return prepare_kuairec_natural(cfg, rng)
    if dataset == "kuairand":
        ensure_kuairand(Path(cfg.data_dir), cfg.download)
        return prepare_kuairand(cfg, rng)
    raise ValueError(f"Unknown dataset: {dataset}")


def make_synthetic(
    rng: np.random.Generator,
    n_users: int = 400,
    n_items: int = 160,
    n_pos: int = 12,
    align: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Two preference communities and an exposure head of tunable alignment.

    `align` is the fraction of the exposure head drawn from preference
    community 0. Zero keeps the head independent of preference. One makes
    the head a subset of that community, which is ordinary popularity bias.
    """
    user_group = rng.integers(0, 2, size=n_users)
    item_group = rng.integers(0, 2, size=n_items)
    score = (user_group[:, None] == item_group[None, :]).astype(np.float64)
    score += rng.normal(scale=0.25, size=score.shape)
    chosen = np.argpartition(-score, kth=n_pos - 1, axis=1)[:, :n_pos]
    edges = np.column_stack(
        (np.repeat(np.arange(n_users), n_pos), chosen.reshape(-1))
    ).astype(np.int64)
    head_size = max(1, n_items // 5)
    align = float(np.clip(align, 0.0, 1.0))
    community = np.flatnonzero(item_group == 0)
    if len(community) == 0:
        align = 0.0
    if align <= 0:
        head = rng.choice(n_items, size=head_size, replace=False)
    else:
        n_aligned = min(len(community), int(round(align * head_size)))
        if align >= 1:
            n_aligned = min(len(community), head_size)
            head_size = n_aligned
        aligned = rng.choice(community, size=n_aligned, replace=False)
        n_rest = head_size - n_aligned
        if n_rest:
            pool = np.setdiff1d(np.arange(n_items), aligned)
            rest = rng.choice(pool, size=n_rest, replace=False)
            head = np.concatenate([aligned, rest])
        else:
            head = aligned
    exposure = np.ones(n_items, dtype=np.float64)
    exposure[head] = 100.0
    info = {
        "exposure_align": align,
        "aligned_community": 0,
        "head_size": int(len(head)),
        "head_in_aligned_community": float(np.mean(item_group[head] == 0)),
    }
    return unique_edges(edges), exposure, info


def prepare_simulated(
    edges: np.ndarray,
    rng: np.random.Generator,
    cfg,
    item_popularity: np.ndarray | None,
    dataset: str,
    propensity_source: str,
) -> PreparedSplit:
    edges = unique_edges(np.asarray(edges, dtype=np.int64))
    if len(edges) == 0:
        raise RuntimeError(f"{dataset} has no positive edges.")
    raw_items = np.unique(edges[:, 1])
    (edges,), _, _ = remap_pairs([edges])
    if item_popularity is None:
        aligned_popularity = None
    else:
        aligned_popularity = np.asarray(item_popularity, dtype=np.float64)[raw_items]
    train, val, test = split_by_user(
        edges, cfg.val_ratio, cfg.test_ratio, rng, min_train=cfg.min_train
    )
    n_test_before = len(test)
    # The core is fit on the training split only. Test edges do not keep a node alive.
    train = k_core(train, cfg.min_user, cfg.min_item)
    if len(train) == 0:
        raise RuntimeError(f"{dataset} has no training edges after the core filter.")
    train_users = np.unique(train[:, 0])
    train_items = np.unique(train[:, 1])
    val = _filter_ids(val, train_users, train_items)
    test = _filter_ids(test, train_users, train_items)
    named, n_users, n_items, surviving_items = _remap_many(
        {"train": train, "val": val, "test": test}
    )
    _check_size(dataset, n_users, n_items)
    train, val, test = named["train"], named["val"], named["test"]
    if aligned_popularity is None:
        popularity = item_counts(train, n_items)
    else:
        popularity = aligned_popularity[surviving_items]
    if cfg.policy == "popularity":
        propensity = item_propensity(popularity, cfg.alpha, cfg.min_prop)
    elif cfg.policy == "uniform":
        propensity = uniform_propensity(n_items, cfg.keep_prob)
        propensity_source = "uniform_policy"
    else:
        raise ValueError(f"Unknown policy: {cfg.policy}")
    raw, raw_p, n_forced = apply_exposure(train, propensity, rng, n_items, min_keep=0)
    if len(raw) == 0:
        raise RuntimeError(
            f"{dataset} exposure mask removed every training edge. Raise --min-prop."
        )
    if cfg.propensity == "estimated":
        estimated = item_propensity(item_counts(raw, n_items), cfg.alpha, cfg.min_prop)
        raw_p = estimated[raw[:, 1]]
        propensity_source = "estimated_from_observed_edges"
    elif cfg.propensity not in {"auto", "known"}:
        raise ValueError(f"Unknown propensity mode: {cfg.propensity}")
    uniform = _uniform_control(train, raw, rng, n_items)
    resampled = resample_by_inverse_propensity(
        raw, raw_p, rng, n_items, target_mean=cfg.resample_keep, min_keep=0
    )
    shift, shift_policy = _policy_shift(test, propensity, cfg, rng, n_items)
    _assert_heldout_disjoint(n_items, [raw, uniform, resampled, train], val, test)
    if not _is_subset(shift, test, n_items):
        raise RuntimeError("The shift test is not a subset of the held-out preferences.")
    meta = _graph_meta(
        dataset,
        raw,
        uniform,
        resampled,
        train,
        val,
        test,
        popularity,
        propensity_source,
        {
            "n_forced_keeps": n_forced,
            "policy": cfg.policy,
            "alpha": cfg.alpha,
            "propensity_min": float(np.min(propensity)),
            "propensity_max": float(np.max(propensity)),
            "propensity_mean": float(np.mean(propensity)),
            "core_fit_on": "training_split",
            "n_test_dropped_with_cold_nodes": int(n_test_before - len(test)),
            "test_labels_in_training_loss": False,
            "shift_policy": shift_policy,
            "n_shift_edges": int(len(shift)),
        },
    )
    return PreparedSplit(
        dataset=dataset,
        n_users=n_users,
        n_items=n_items,
        raw_edges=raw,
        raw_propensity=raw_p.astype(np.float64),
        uniform_edges=uniform,
        resampled_edges=resampled,
        counterfactual_edges=train,
        val_edges=val,
        test_edges=test,
        policy_popularity=popularity.astype(np.float64),
        meta=meta,
        shift_edges=shift,
    )


def prepare_kuairand(cfg, rng: np.random.Generator) -> PreparedSplit:
    root = Path(cfg.data_dir)
    early = _read_kuairand_log(_find(root, "log_standard_4_08_to_4_21_pure.csv"))
    late = _read_kuairand_log(_find(root, "log_standard_4_22_to_5_08_pure.csv"))
    random_log = _read_kuairand_log(_find(root, "log_random_4_22_to_5_08_pure.csv"))
    impression_counts = early.groupby("video_id").size()
    standard_pos = _feedback_edges(pd.concat([early, late], ignore_index=True), cfg.feedback)
    random_pos = _feedback_edges(random_log, cfg.feedback)
    del early, late, random_log
    standard_pos = unique_edges(standard_pos)
    cf_train, val, test = split_random_feedback(random_pos, rng, cfg.val_ratio, cfg.test_ratio)
    span = _item_span([standard_pos, random_pos])
    raw = drop_pairs(standard_pos, _concat_edges([val, test]), span)
    n_overlap_removed = int(len(standard_pos) - len(raw))
    n_test_before = len(test)
    raw, cf_train, val, test = _restrict_to_training_nodes(raw, cf_train, val, test)
    if len(raw) == 0 or len(test) == 0:
        raise RuntimeError(
            "KuaiRand produced an empty raw graph or an empty unbiased test. "
            f"Try another --feedback (current: {cfg.feedback})."
        )
    named, n_users, n_items, old_items = _remap_many(
        {"raw": raw, "cf": cf_train, "val": val, "test": test}
    )
    _check_size("kuairand", n_users, n_items)
    raw, cf_train, val, test = named["raw"], named["cf"], named["val"], named["test"]
    impression_counts.index = impression_counts.index.astype(np.int64)
    popularity = impression_counts.reindex(old_items).fillna(0).to_numpy(dtype=np.float64)
    if len(popularity) != n_items:
        raise RuntimeError("KuaiRand impression counts do not match the remapped items.")
    propensity = item_propensity(popularity, cfg.alpha, cfg.min_prop)
    raw_p = propensity[raw[:, 1]]
    resampled = resample_by_inverse_propensity(
        raw, raw_p, rng, n_items, target_mean=cfg.resample_keep, min_keep=0
    )
    _assert_heldout_disjoint(n_items, [raw, resampled, cf_train], val, test)
    meta = _graph_meta(
        "kuairand",
        raw,
        None,
        resampled,
        cf_train,
        val,
        test,
        popularity,
        "estimated_from_preperiod_impressions",
        {
            "feedback": cfg.feedback,
            "policy": "logged_standard_vs_random",
            "n_overlap_removed": n_overlap_removed,
            "core_fit_on": "training_edges",
            "n_test_dropped_with_cold_nodes": int(n_test_before - len(test)),
            "test_labels_in_training_loss": False,
        },
    )
    return PreparedSplit(
        dataset="kuairand",
        n_users=n_users,
        n_items=n_items,
        raw_edges=raw,
        raw_propensity=raw_p.astype(np.float64),
        uniform_edges=None,
        resampled_edges=resampled,
        counterfactual_edges=cf_train,
        val_edges=val,
        test_edges=test,
        policy_popularity=popularity,
        meta=meta,
    )


def prepare_kuairec_natural(cfg, rng: np.random.Generator) -> PreparedSplit:
    root = Path(cfg.data_dir)
    small = _read_matrix(_find(root, "small_matrix.csv"))
    big = _read_matrix(_find(root, "big_matrix.csv"))
    small_pos = _watch_edges(small, cfg.watch_threshold)
    big_pos = _watch_edges(big, cfg.watch_threshold)
    block_users = small["user_id"].to_numpy()
    block_items = small["video_id"].to_numpy()
    impression_counts = big.groupby("video_id").size()
    del small, big
    small_pos = _filter_ids(small_pos, block_users, block_items)
    big_pos = _filter_ids(big_pos, block_users, block_items)
    if len(small_pos) == 0:
        raise RuntimeError(
            f"No small-matrix pairs have watch_ratio > {cfg.watch_threshold}. "
            "Lower --watch_threshold."
        )
    train, val, test = split_by_user(
        small_pos, cfg.val_ratio, cfg.test_ratio, rng, min_train=cfg.min_train
    )
    n_test_before = len(test)
    train = k_core(train, cfg.min_user, cfg.min_item)
    if len(train) == 0:
        raise RuntimeError("The small-matrix training split is empty after the core filter.")
    train_users = np.unique(train[:, 0])
    train_items = np.unique(train[:, 1])
    val = _filter_ids(val, train_users, train_items)
    test = _filter_ids(test, train_users, train_items)
    big_pos = _filter_ids(big_pos, train_users, train_items)
    raw = drop_pairs(big_pos, _concat_edges([val, test]), _item_span([train, val, test, big_pos]))
    if len(raw) < 1000 or len(test) == 0:
        raise RuntimeError(
            "The big matrix shares too few positive edges with the fully observed "
            "small-matrix users and videos. Run --dataset kuairec, which applies a "
            "known exposure policy to the small matrix instead."
        )
    named, n_users, n_items, old_items = _remap_many(
        {"train": train, "val": val, "test": test, "raw": raw}
    )
    _check_size("kuairec_natural", n_users, n_items)
    train, val, test, raw = named["train"], named["val"], named["test"], named["raw"]
    impression_counts.index = impression_counts.index.astype(np.int64)
    popularity = impression_counts.reindex(old_items).fillna(0).to_numpy(dtype=np.float64)
    if len(popularity) != n_items:
        raise RuntimeError("Big-matrix impression counts do not match the remapped items.")
    propensity = item_propensity(popularity, cfg.alpha, cfg.min_prop)
    raw_p = propensity[raw[:, 1]]
    resampled = resample_by_inverse_propensity(
        raw, raw_p, rng, n_items, target_mean=cfg.resample_keep, min_keep=0
    )
    uniform = _uniform_control(train, raw, rng, n_items)
    _assert_heldout_disjoint(n_items, [raw, uniform, resampled, train], val, test)
    meta = _graph_meta(
        "kuairec_natural",
        raw,
        uniform,
        resampled,
        train,
        val,
        test,
        popularity,
        "estimated_from_big_matrix_impressions",
        {
            "policy": "logged_big_matrix",
            "watch_threshold": cfg.watch_threshold,
            "core_fit_on": "training_split",
            "n_test_dropped_with_cold_nodes": int(n_test_before - len(test)),
            "test_labels_in_training_loss": False,
        },
    )
    return PreparedSplit(
        dataset="kuairec_natural",
        n_users=n_users,
        n_items=n_items,
        raw_edges=raw,
        raw_propensity=raw_p.astype(np.float64),
        uniform_edges=uniform,
        resampled_edges=resampled,
        counterfactual_edges=train,
        val_edges=val,
        test_edges=test,
        policy_popularity=popularity.astype(np.float64),
        meta=meta,
    )


def split_random_feedback(edges, rng, val_ratio, test_ratio):
    """Hold out unbiased KuaiRand labels first. Training keeps what remains."""
    edges = unique_edges(edges)
    if len(edges) == 0:
        empty = np.zeros((0, 2), dtype=np.int64)
        return empty, empty, empty
    order = np.argsort(edges[:, 0], kind="mergesort")
    edges = edges[order]
    _, starts, counts = np.unique(edges[:, 0], return_index=True, return_counts=True)
    train_parts, val_parts, test_parts = [], [], []
    for start, count in zip(starts, counts):
        block = edges[start : start + int(count)]
        block = block[rng.permutation(int(count))]
        count = int(count)
        if count == 1:
            test_parts.append(block)
            continue
        if count == 2:
            train_parts.append(block[:1])
            test_parts.append(block[1:])
            continue
        if count < 5:
            test_parts.append(block[:1])
            val_parts.append(block[1:2])
            train_parts.append(block[2:])
            continue
        n_test = max(1, int(round(count * test_ratio)))
        n_val = max(1, int(round(count * val_ratio)))
        n_test = min(n_test, count - 2)
        n_val = min(n_val, count - n_test - 1)
        test_parts.append(block[:n_test])
        val_parts.append(block[n_test : n_test + n_val])
        train_parts.append(block[n_test + n_val :])
    return (
        _concat_edges(train_parts),
        _concat_edges(val_parts),
        _concat_edges(test_parts),
    )


def _uniform_control(preference, raw, rng, n_items):
    if len(raw) == 0 or len(preference) <= len(raw) * 1.02:
        return None
    return uniform_subsample(preference, len(raw), rng, n_items, min_keep=0)


def _policy_shift(test, propensity, cfg, rng, n_items):
    """Held-out preferences under a second exposure policy.

    The coin flips use the test edges only, after the training graph is fixed.
    A popularity training policy is shifted to a uniform draw. A uniform
    training policy is shifted to the popularity mask.
    """
    empty = np.zeros((0, 2), dtype=np.int64)
    if len(test) == 0:
        return empty, "none"
    if cfg.policy == "uniform":
        kept, _, _ = apply_exposure(test, propensity, rng, n_items, min_keep=0)
        return kept, "popularity_on_heldout_preferences"
    keep = float(np.clip(getattr(cfg, "shift_keep", 0.5), 0.05, 1.0))
    target = min(len(test), max(1, int(round(len(test) * keep))))
    kept = uniform_subsample(test, target, rng, n_items, min_keep=0)
    return kept, "uniform_on_heldout_preferences"


def _is_subset(small: np.ndarray, big: np.ndarray, n_items: int) -> bool:
    if len(small) == 0:
        return True
    if len(big) == 0:
        return False
    universe = np.sort(edge_keys(big, n_items))
    keys = edge_keys(small, n_items)
    pos = np.clip(np.searchsorted(universe, keys), 0, len(universe) - 1)
    return bool(np.all(universe[pos] == keys))


def _item_span(arrays: list[np.ndarray]) -> int:
    max_item = -1
    for array in arrays:
        if len(array):
            max_item = max(max_item, int(np.max(array[:, 1])))
    return max_item + 1


def _remap_many(named: dict[str, np.ndarray]):
    names = list(named)
    arrays = [np.asarray(named[name], dtype=np.int64) for name in names]
    nonempty = [array for array in arrays if len(array)]
    if not nonempty:
        raise RuntimeError("No edges remained after the training-split filter.")
    old_items = np.unique(np.concatenate([array[:, 1] for array in nonempty]))
    mapped, n_users, n_items = remap_pairs(arrays)
    if len(old_items) != n_items:
        raise RuntimeError("Item ids could not be aligned after filtering.")
    return dict(zip(names, mapped)), n_users, n_items, old_items


def _restrict_to_training_nodes(raw, counterfactual, val, test):
    """Drop held-out pairs whose user or item never appears in a training graph."""
    pieces = [edges for edges in (raw, counterfactual) if len(edges)]
    if not pieces:
        raise RuntimeError("Both training graphs are empty.")
    users = np.unique(np.concatenate([edges[:, 0] for edges in pieces]))
    items = np.unique(np.concatenate([edges[:, 1] for edges in pieces]))
    return (
        _filter_ids(raw, users, items),
        _filter_ids(counterfactual, users, items),
        _filter_ids(val, users, items),
        _filter_ids(test, users, items),
    )


def _overlaps(left: np.ndarray, right: np.ndarray, n_items: int) -> bool:
    if len(left) == 0 or len(right) == 0:
        return False
    banned = np.sort(edge_keys(right, n_items))
    keys = edge_keys(left, n_items)
    pos = np.clip(np.searchsorted(banned, keys), 0, len(banned) - 1)
    return bool(np.any(banned[pos] == keys))


def _assert_heldout_disjoint(n_items: int, training: list, val: np.ndarray, test: np.ndarray) -> None:
    if _overlaps(val, test, n_items):
        raise RuntimeError("Validation and test pairs overlap.")
    for edges in training:
        if edges is None:
            continue
        if _overlaps(edges, val, n_items) or _overlaps(edges, test, n_items):
            raise RuntimeError("A validation or test pair is also present in a training graph.")


def _graph_meta(dataset, raw, uniform, resampled, cf, val, test, popularity, source, extra):
    heads = head_mask(popularity, 0.2)
    meta = {
        "dataset": dataset,
        "n_raw_edges": int(len(raw)),
        "n_uniform_matched_edges": None if uniform is None else int(len(uniform)),
        "n_resampled_edges": int(len(resampled)),
        "n_counterfactual_edges": int(len(cf)),
        "n_val_edges": int(len(val)),
        "n_test_edges": int(len(test)),
        "keep_rate": float(len(raw) / len(cf)) if len(cf) else float("nan"),
        "propensity_source": source,
        "head_edge_share": {
            "raw": edge_share_on_items(raw, heads),
            "resampled": edge_share_on_items(resampled, heads),
            "counterfactual": edge_share_on_items(cf, heads),
            "test": edge_share_on_items(test, heads),
        },
    }
    if uniform is not None:
        meta["head_edge_share"]["uniform_matched"] = edge_share_on_items(uniform, heads)
    meta.update(extra)
    return meta


def _check_size(dataset: str, n_users: int, n_items: int) -> None:
    if n_users > MAX_USERS or n_items > MAX_ITEMS:
        raise RuntimeError(
            f"{dataset} has {n_users} users and {n_items} items. This experiment "
            "is built for the fully observed KuaiRec block, KuaiRand-Pure, and "
            "MovieLens-1M. Refusing the full KuaiRand-27K item catalog."
        )
    if n_users < 2 or n_items < 2:
        raise RuntimeError(f"{dataset} collapsed to {n_users} users and {n_items} items.")


def load_movielens(root: Path, rating_threshold: float) -> np.ndarray:
    path = _find(root, "ratings.dat")
    frame = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    kept = frame.loc[frame["rating"] >= rating_threshold, ["user_id", "item_id"]]
    print(f"MovieLens positives (rating >= {rating_threshold}): {len(kept):,} from {path}", flush=True)
    return unique_edges(kept.to_numpy(dtype=np.int64))


def load_kuairec_positives(root: Path, threshold: float, which: str) -> np.ndarray:
    name = "small_matrix.csv" if which == "small" else "big_matrix.csv"
    frame = _read_matrix(_find(root, name))
    edges = _watch_edges(frame, threshold)
    print(
        f"KuaiRec {which} positives (watch_ratio > {threshold}): {len(edges):,}",
        flush=True,
    )
    return edges


def _watch_edges(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    kept = frame.loc[frame["watch_ratio"] > threshold, ["user_id", "video_id"]]
    return unique_edges(kept.to_numpy(dtype=np.int64))


def _read_matrix(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    lookup = {c.strip().lower(): c for c in header.columns}
    wanted = [
        name
        for name in ("user_id", "video_id", "watch_ratio", "play_duration", "video_duration")
        if name in lookup
    ]
    if "user_id" not in lookup or "video_id" not in lookup:
        raise RuntimeError(f"{path} is missing user_id or video_id.")
    frame = pd.read_csv(path, usecols=[lookup[name] for name in wanted])
    frame.columns = [c.strip().lower() for c in frame.columns]
    if "watch_ratio" not in frame.columns:
        frame["watch_ratio"] = frame["play_duration"] / frame["video_duration"].replace(0, np.nan)
    frame["user_id"] = frame["user_id"].astype(np.int64)
    frame["video_id"] = frame["video_id"].astype(np.int64)
    print(f"Loaded {len(frame):,} rows from {path}", flush=True)
    return frame


def _read_kuairand_log(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    lookup = {c.strip().lower(): c for c in header.columns}
    wanted = [
        name
        for name in ("user_id", "video_id", "is_click", "long_view", "is_like", "is_hate")
        if name in lookup
    ]
    frame = pd.read_csv(path, usecols=[lookup[name] for name in wanted])
    frame.columns = [c.strip().lower() for c in frame.columns]
    frame["user_id"] = frame["user_id"].astype(np.int64)
    frame["video_id"] = frame["video_id"].astype(np.int64)
    print(f"Loaded {len(frame):,} KuaiRand rows from {path}", flush=True)
    return frame


def _feedback_edges(frame: pd.DataFrame, feedback: str) -> np.ndarray:
    if feedback not in frame.columns:
        raise RuntimeError(
            f"KuaiRand log has no '{feedback}' column. Available: {list(frame.columns)}"
        )
    mask = frame[feedback].to_numpy() == 1
    if "is_hate" in frame.columns:
        mask = mask & (frame["is_hate"].to_numpy() == 0)
    kept = frame.loc[mask, ["user_id", "video_id"]]
    return unique_edges(kept.to_numpy(dtype=np.int64))


def _filter_ids(edges: np.ndarray, users: np.ndarray, items: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return edges
    mask = np.isin(edges[:, 0], users) & np.isin(edges[:, 1], items)
    return unique_edges(edges[mask])


def _concat_edges(parts: list[np.ndarray]) -> np.ndarray:
    parts = [np.asarray(p, dtype=np.int64) for p in parts if len(p)]
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return unique_edges(np.concatenate(parts, axis=0))


def _find(root: Path, filename: str) -> Path:
    matches = sorted(p for p in root.rglob(filename) if p.is_file())
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {root}. "
            "Pass --download or set --data_dir to the extracted dataset."
        )
    if len(matches) > 1:
        print(f"Using {matches[0]} ({len(matches)} copies of {filename})", flush=True)
    return matches[0]


def ensure_movielens(root: Path, download: bool) -> None:
    if _exists(root, "ratings.dat"):
        return
    if not download:
        raise FileNotFoundError(
            "MovieLens-1M is not in --data_dir. Re-run with --download or fetch\n"
            f"  wget {MOVIELENS_URL}"
        )
    _download_and_extract(MOVIELENS_URL, root, root / "downloads" / "ml-1m.zip")


def ensure_kuairec(root: Path, download: bool) -> None:
    if _exists(root, "small_matrix.csv"):
        return
    if not download:
        raise FileNotFoundError(
            "KuaiRec is not in --data_dir. Re-run with --download or fetch\n"
            f"  wget {KUAIREC_URL}"
        )
    _download_and_extract(KUAIREC_URL, root, root / "downloads" / "KuaiRec.zip")


def ensure_kuairand(root: Path, download: bool) -> None:
    if _exists(root, "log_random_4_22_to_5_08_pure.csv"):
        return
    if not download:
        raise FileNotFoundError(
            "KuaiRand-Pure is not in --data_dir. Re-run with --download or fetch\n"
            f"  wget {KUAIRAND_URL}"
        )
    _download_and_extract(KUAIRAND_URL, root, root / "downloads" / "KuaiRand-Pure.tar.gz")


def _exists(root: Path, filename: str) -> bool:
    return root.exists() and any(root.rglob(filename))


def _download_and_extract(url: str, root: Path, archive: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or archive.stat().st_size == 0:
        print(f"Downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "esb-research/1.0"})
        with urllib.request.urlopen(request, timeout=300) as response, archive.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    print(f"Extracting {archive}", flush=True)
    name = archive.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(root)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as bundle:
            try:
                bundle.extractall(root, filter="data")
            except TypeError:
                bundle.extractall(root)
    else:
        raise RuntimeError(f"Do not know how to extract {archive}")
