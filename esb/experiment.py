"""Train raw, deconfounded, and counterfactual graphs and compare them."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from esb.data import prepare
from scipy.stats import spearmanr

from esb.metrics import (
    embedding_knn,
    evaluate_ranking,
    interaction_knn,
    knn_popularity_corr,
    linear_cka,
    mean_jaccard,
    neighborhood_report,
    pairs_to_mask,
)
from esb.sampling import assign_ips, exposure_buckets, ips_weights, item_counts, unique_edges
from esb.theory import propagation_probe
from esb.train import Variant, train_variant

PROTOCOLS = {
    "synthetic": (
        "Synthetic preferences come from two communities. --exposure-align is the "
        "fraction of the exposure head drawn from one preference community. "
        "The core filter uses the training split only. A second test keeps a "
        "uniform slice of the held-out preferences and is never used for training."
    ),
    "movielens": (
        "Ratings of 4 and above are the preference graph. The core filter and the "
        "exposure policy use the training split only. Held-out items are eligible "
        "negatives. The reported test metrics come from the validation checkpoint."
    ),
    "kuairec": (
        "The fully observed KuaiRec small matrix is binarized with watch_ratio > threshold. "
        "A known exposure policy then hides training edges. The core is fit on the training "
        "split, and the counterfactual graph is those same preferences without the mask."
    ),
    "kuairec_natural": (
        "The raw graph is the big-matrix log on users and videos that remain after the "
        "small-matrix training split is cored. The counterfactual graph is that training "
        "split. Propensities are item frequencies in the big matrix, not the logging policy."
    ),
    "kuairand": (
        "The raw graph is standard-policy positive feedback. The counterfactual graph and "
        "the test labels are random-exposure positives. Nodes that appear only in the "
        "held-out random pairs are dropped. Propensities are item impression counts from "
        "8–21 April, before the random-exposure window."
    ),
}


def run_one(cfg, seed: int, device: str) -> dict:
    print(f"\n=== {cfg.dataset}  seed {seed}  device {device} ===", flush=True)
    split = prepare(cfg, seed)
    _print_data(split)
    anchors = _reference_structure(split, cfg)
    if anchors is not None:
        print(
            "Data neighborhoods: "
            f"jaccard(exposure, preference)={anchors['reference_jaccard']:.3f}  "
            f"exposure popularity corr={anchors['exposure_knn_popularity_corr']:.3f}  "
            f"preference popularity corr={anchors['preference_knn_popularity_corr']:.3f}",
            flush=True,
        )
    _warn_manipulation(split)
    probe = propagation_probe(split, anchors, cfg.ips_clip)
    result = {
        "dataset": split.dataset,
        "seed": seed,
        "device": device,
        "protocol": PROTOCOLS.get(split.dataset, ""),
        "config": _public_config(cfg),
        "data": split.meta,
        "structure_anchors": anchors,
        "propagation_probe": probe,
        "variants": {},
    }
    _print_probe(probe)
    if cfg.dry_run:
        _drop_private(anchors)
        print("Dry run: graphs are built, models were not trained.", flush=True)
        _write_result(cfg, seed, result)
        return result

    variants = _variants(split, cfg)
    result["data"]["ips_clip_fraction"] = _clip_fraction(split, cfg)
    ban_edges = unique_edges(np.concatenate([variant.edges for variant in variants], axis=0))
    ban = pairs_to_mask(split.n_users, split.n_items, ban_edges)
    val_relevant = pairs_to_mask(split.n_users, split.n_items, split.val_edges)
    test_relevant = pairs_to_mask(split.n_users, split.n_items, split.test_edges)
    leaked = int((ban & val_relevant).sum() + (ban & test_relevant).sum())
    if leaked:
        raise RuntimeError(f"{leaked} validation or test pairs also appear in a training graph.")
    # Negatives are everything outside the training split. Holding val and test
    # positives out of the negative pool would use those labels during training.
    forbidden = ban
    present = np.zeros(split.n_items, dtype=bool)
    for edges in (split.raw_edges, split.counterfactual_edges):
        if len(edges):
            present[edges[:, 1]] = True
    buckets = exposure_buckets(split.policy_popularity, present)
    result["data"]["negatives"] = "outside_the_training_split"
    result["data"]["candidate_mask"] = "union_of_training_edges"
    if int(val_relevant.sum()) == 0:
        print("No validation edges; training will run for every epoch.", flush=True)

    trained = []
    for variant in variants:
        print(f"\nTraining {variant.name}  edges={len(variant.edges):,}  layers={variant.n_layers}", flush=True)
        trained.append(
            train_variant(
                variant,
                split.n_users,
                split.n_items,
                ban,
                val_relevant,
                test_relevant,
                buckets,
                forbidden,
                cfg,
                device,
                seed,
            )
        )
        result["variants"][variant.name] = _variant_record(trained[-1])
        _write_result(cfg, seed, result)

    _add_structure(result, trained, split, cfg, anchors)
    _add_shift(result, trained, split, ban, buckets, cfg, device)
    result["contrasts"] = _contrasts(result["variants"], max(cfg.ks))
    result["selector"] = _selector(result["variants"], max(cfg.ks))
    _print_table(result, max(cfg.ks))
    _print_contrasts(result["contrasts"])
    _print_selector(result["selector"], split)
    _write_result(cfg, seed, result)
    return result


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this PyTorch build cannot see a GPU.")
    return name


def _variants(split, cfg) -> list[Variant]:
    raw_ones = np.ones(len(split.raw_edges), dtype=np.float32)
    variants = [Variant("raw", split.raw_edges, raw_ones, raw_ones.copy(), cfg.layers)]
    if split.uniform_edges is not None:
        uniform_ones = np.ones(len(split.uniform_edges), dtype=np.float32)
        variants.append(
            Variant("uniform_matched", split.uniform_edges, uniform_ones, uniform_ones.copy(), cfg.layers)
        )
    # Weight the edges once. Putting 1/p in both the adjacency and the loss
    # double-counts the same correction.
    graph_weight, loss_weight = assign_ips(split.raw_propensity, "graph", cfg.ips_clip)
    variants.append(Variant("ips_graph", split.raw_edges, graph_weight, loss_weight, cfg.layers))
    graph_weight, loss_weight = assign_ips(split.raw_propensity, "loss", cfg.ips_clip)
    variants.append(Variant("ips_loss", split.raw_edges, graph_weight, loss_weight, cfg.layers))
    if cfg.include_resampled and len(split.resampled_edges) > 1:
        resampled_ones = np.ones(len(split.resampled_edges), dtype=np.float32)
        variants.append(
            Variant("resampled", split.resampled_edges, resampled_ones, resampled_ones.copy(), cfg.layers)
        )
    cf_ones = np.ones(len(split.counterfactual_edges), dtype=np.float32)
    variants.append(
        Variant("counterfactual", split.counterfactual_edges, cf_ones, cf_ones.copy(), cfg.layers)
    )
    if cfg.include_mf:
        variants.append(Variant("mf_raw", split.raw_edges, raw_ones, raw_ones.copy(), 0))
    return variants


def _clip_fraction(split, cfg) -> float:
    weights = ips_weights(split.raw_propensity, cfg.ips_clip)
    if len(weights) == 0:
        return float("nan")
    return float(np.mean(np.isclose(weights, cfg.ips_clip)))


def _reference_structure(split, cfg):
    if len(split.counterfactual_edges) == 0 or len(split.raw_edges) == 0:
        return None
    pref_degree = item_counts(split.counterfactual_edges, split.n_items)
    expo_degree = item_counts(split.raw_edges, split.n_items)
    n_pref = int((pref_degree >= cfg.min_degree).sum())
    n_expo = int((expo_degree >= 1).sum())
    k_eff = min(cfg.neigh_k, n_pref - 1, n_expo - 1)
    if k_eff < 2:
        print("Too few items with neighbors to compare neighborhoods.", flush=True)
        return None
    preference, pref_valid = interaction_knn(
        split.counterfactual_edges, split.n_users, split.n_items, k_eff, cfg.min_degree
    )
    exposure, expo_valid = interaction_knn(
        split.raw_edges, split.n_users, split.n_items, k_eff, 1
    )
    valid = pref_valid & expo_valid
    popularity = np.asarray(split.policy_popularity, dtype=np.float64)
    return {
        "k": int(k_eff),
        "n_items": int(valid.sum()),
        "reference_jaccard": mean_jaccard(preference, exposure, valid),
        "preference_knn_popularity_corr": knn_popularity_corr(preference, popularity, valid),
        "exposure_knn_popularity_corr": knn_popularity_corr(exposure, popularity, valid),
        "_preference": preference,
        "_exposure": exposure,
        "_valid": valid,
    }


def _add_structure(result, trained, split, cfg, anchors) -> None:
    if anchors is None:
        for record in result["variants"].values():
            record["structure"] = None
        return
    preference = anchors.pop("_preference")
    exposure = anchors.pop("_exposure")
    valid = anchors.pop("_valid")
    popularity = np.asarray(split.policy_popularity, dtype=np.float64)
    by_name = {model.name: model for model in trained}
    reference = by_name.get("counterfactual")
    for model in trained:
        neighbors = embedding_knn(model.item_emb, anchors["k"])
        structure = neighborhood_report(neighbors, preference, exposure, valid, popularity)
        if reference is None:
            structure["item_cka_to_counterfactual"] = float("nan")
            structure["user_cka_to_counterfactual"] = float("nan")
        else:
            structure["item_cka_to_counterfactual"] = linear_cka(model.item_emb, reference.item_emb)
            structure["user_cka_to_counterfactual"] = linear_cka(model.user_emb, reference.user_emb)
        result["variants"][model.name]["structure"] = structure


def _variant_record(model) -> dict:
    return {
        "best_epoch": model.best_epoch,
        "best_val_ndcg": model.best_val_ndcg,
        "seconds": model.seconds,
        "n_edges": model.n_edges,
        "n_layers": model.n_layers,
        "val_history": model.val_history,
        "test": model.test_metrics,
        "structure": None,
    }


def _contrasts(variants: dict, k: int) -> dict:
    def value(name, *path):
        node = variants.get(name)
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        if isinstance(node, float) and math.isnan(node):
            return None
        return node

    def minus(left, right):
        if left is None or right is None:
            return None
        return left - right

    raw_leak = value("raw", "structure", "leakage")
    raw_tail = value("raw", "test", f"tail_ndcg@{k}")
    raw_gap = value("raw", "test", f"head_tail_gap@{k}")
    return {
        "leakage_raw_minus_uniform_matched": minus(raw_leak, value("uniform_matched", "structure", "leakage")),
        "leakage_raw_minus_ips_graph": minus(raw_leak, value("ips_graph", "structure", "leakage")),
        "leakage_raw_minus_ips_loss": minus(raw_leak, value("ips_loss", "structure", "leakage")),
        "leakage_raw_minus_resampled": minus(raw_leak, value("resampled", "structure", "leakage")),
        "leakage_raw_minus_counterfactual": minus(
            raw_leak, value("counterfactual", "structure", "leakage")
        ),
        "tail_ndcg_uniform_matched_minus_raw": minus(
            value("uniform_matched", "test", f"tail_ndcg@{k}"), raw_tail
        ),
        "tail_ndcg_ips_graph_minus_raw": minus(value("ips_graph", "test", f"tail_ndcg@{k}"), raw_tail),
        "tail_ndcg_ips_loss_minus_raw": minus(value("ips_loss", "test", f"tail_ndcg@{k}"), raw_tail),
        "tail_ndcg_resampled_minus_raw": minus(value("resampled", "test", f"tail_ndcg@{k}"), raw_tail),
        "tail_ndcg_counterfactual_minus_raw": minus(
            value("counterfactual", "test", f"tail_ndcg@{k}"), raw_tail
        ),
        "gap_raw_minus_counterfactual": minus(
            raw_gap, value("counterfactual", "test", f"head_tail_gap@{k}")
        ),
    }


def _add_shift(result, trained, split, ban, buckets, cfg, device) -> None:
    """Score the checkpoint on a second exposure policy. Training never sees it."""
    shift = split.shift_edges
    if shift is None or len(shift) == 0:
        for record in result["variants"].values():
            record["shift"] = None
        return
    relevant = pairs_to_mask(split.n_users, split.n_items, shift)
    leaked = int((ban & relevant).sum())
    if leaked:
        raise RuntimeError(f"{leaked} shift-test pairs also appear in a training graph.")
    by_name = {model.name: model for model in trained}
    for name, record in result["variants"].items():
        model = by_name[name]
        record["shift"] = evaluate_ranking(
            model.user_emb,
            model.item_emb,
            ban,
            relevant,
            buckets,
            ks=tuple(cfg.ks),
            device=device,
            with_buckets=True,
        )


def _selector(variants: dict, k: int) -> dict:
    """Ask whether validation NDCG or leakage ranks the shifted test.

    The counterfactual graph and the resampled graph have different edge
    counts, so they are left out of this ranking. The remaining models are
    the raw graph, the density-matched graph, and the two single IPS corrections.
    """
    comparable = [name for name in ("raw", "uniform_matched", "ips_graph", "ips_loss") if name in variants]
    names = comparable or list(variants)
    val = [_finite(variants[name].get("best_val_ndcg")) for name in names]
    leakage = [_finite(_dig(variants[name], ("structure", "leakage"))) for name in names]
    shift = [_finite(_dig(variants[name], ("shift", f"ndcg@{k}"))) for name in names]
    test = [_finite(_dig(variants[name], ("test", f"ndcg@{k}"))) for name in names]
    val_shift = _spearman(val, shift)
    leak_shift = _spearman(leakage, shift)
    better = None
    if val_shift is not None and leak_shift is not None:
        # Higher validation NDCG should track shifted NDCG if validation is the
        # right selector. Lower leakage should track higher shifted NDCG.
        better = bool(leak_shift < 0 and (-leak_shift) > val_shift)
    return {
        "spearman_val_vs_shift": val_shift,
        "spearman_leakage_vs_shift": leak_shift,
        "spearman_val_vs_test": _spearman(val, test),
        "leakage_ranks_shift_better_than_val": better,
    }


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _spearman(left, right):
    pairs = [
        (a, b)
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    if len(pairs) < 4:
        return None
    xs = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    ys = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if float(xs.std()) < 1e-12 or float(ys.std()) < 1e-12:
        return None
    ranked = spearmanr(xs, ys)
    coef = float(getattr(ranked, "statistic", ranked.correlation))
    if math.isnan(coef):
        return None
    return coef


def _print_probe(probe) -> None:
    if not probe:
        return
    parts = []
    for layer, gap in probe["gaps"].items():
        parts.append(
            f"L{layer} uniform {_fmt(gap['raw_minus_uniform_matched'])} "
            f"ips {_fmt(gap['raw_minus_ips_graph'])}"
        )
    print("Propagation gap, raw minus the other graph, before training: " + " | ".join(parts), flush=True)


def _print_selector(selector, split) -> None:
    if not selector or not split.meta.get("n_shift_edges"):
        return
    policy = split.meta.get("shift_policy", "none")
    n_shift = split.meta.get("n_shift_edges", 0)
    print(
        f"Shift test ({policy}, {n_shift} edges). "
        f"Spearman val→shift {_fmt(selector['spearman_val_vs_shift'])}  "
        f"leakage→shift {_fmt(selector['spearman_leakage_vs_shift'])}. "
        "A useful leakage score is negative: lower leakage, higher shifted NDCG.",
        flush=True,
    )


def _print_data(split) -> None:
    meta = split.meta
    shares = meta["head_edge_share"]
    share_text = "  ".join(f"{name}={value:.3f}" for name, value in shares.items())
    print(
        f"Users {split.n_users:,}  items {split.n_items:,}  "
        f"raw {meta['n_raw_edges']:,}  counterfactual {meta['n_counterfactual_edges']:,}  "
        f"resampled {meta['n_resampled_edges']:,}  "
        f"val {meta['n_val_edges']:,}  test {meta['n_test_edges']:,}",
        flush=True,
    )
    print(f"Head-item edge share (top 20% by exposure score): {share_text}", flush=True)
    print(f"Propensity: {meta['propensity_source']}", flush=True)
    if meta.get("exposure_align") is not None:
        print(
            f"Exposure alignment {meta['exposure_align']:.2f}  "
            f"head in community 0 = {meta.get('head_in_aligned_community', float('nan')):.2f}",
            flush=True,
        )
    if meta.get("n_shift_edges") is not None:
        print(
            f"Shift test: {meta['shift_policy']}  edges {meta['n_shift_edges']:,}",
            flush=True,
        )


def _warn_manipulation(split) -> None:
    shares = split.meta["head_edge_share"]
    policy = split.meta.get("policy")
    if policy != "popularity":
        return
    raw_share = shares["raw"]
    cf_share = shares["counterfactual"]
    if raw_share <= cf_share + 0.02:
        print(
            "Warning: the exposure mask did not concentrate edges on the head. "
            "Check --alpha and --min_prop.",
            flush=True,
        )


def _print_table(result: dict, k: int) -> None:
    columns = [
        ("ndcg", ("test", f"ndcg@{k}")),
        ("recall", ("test", f"recall@{k}")),
        ("tail", ("test", f"tail_ndcg@{k}")),
        ("gap", ("test", f"head_tail_gap@{k}")),
        ("leakage", ("structure", "leakage")),
        ("jac_pref", ("structure", "jaccard_preference")),
        ("jac_expo", ("structure", "jaccard_exposure")),
        ("item_cka", ("structure", "item_cka_to_counterfactual")),
        ("user_cka", ("structure", "user_cka_to_counterfactual")),
    ]
    header = f"{'variant':18s}" + "".join(f"{name:>10s}" for name, _ in columns)
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)
    for name, record in result["variants"].items():
        cells = []
        for _, path in columns:
            cells.append(f"{_fmt(_dig(record, path)):>10s}")
        print(f"{name:18s}" + "".join(cells), flush=True)
    print(
        "\nLeakage contrasts are raw minus the other model. "
        "Tail-NDCG contrasts are the other model minus raw.",
        flush=True,
    )


def _print_contrasts(contrasts: dict) -> None:
    for name, value in contrasts.items():
        print(f"  {name}: {_fmt(value)}", flush=True)


def _dig(record, path):
    node = record
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.4f}"


def _public_config(cfg) -> dict:
    skip = {"dry_run"}
    exported = {}
    for key, value in vars(cfg).items():
        if key in skip:
            continue
        if isinstance(value, (list, tuple)):
            exported[key] = list(value)
        else:
            exported[key] = value
    return exported


def _write_result(cfg, seed: int, result: dict) -> None:
    directory = Path(cfg.results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result_stem(cfg, seed)}.json"
    path.write_text(json.dumps(_jsonify(result), indent=2) + "\n")
    print(f"Wrote {path}", flush=True)


def result_stem(cfg, seed: int) -> str:
    if cfg.dataset == "kuairand":
        middle = cfg.feedback
    elif cfg.dataset == "kuairec_natural":
        middle = f"watch{cfg.watch_threshold:g}"
    else:
        middle = cfg.policy
    align = float(getattr(cfg, "exposure_align", 0.0))
    return f"{cfg.dataset}_{middle}_a{cfg.alpha:g}_r{align:g}_L{cfg.layers}_seed{seed}"


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _drop_private(anchors) -> None:
    if not isinstance(anchors, dict):
        return
    for key in ("_preference", "_exposure", "_valid"):
        anchors.pop(key, None)


def summarize_runs(runs: list[dict]) -> dict:
    """Mean and sample standard deviation of the comparison metrics."""
    if len(runs) == 1:
        return runs[0]
    if not runs[0].get("variants"):
        return {
            "dataset": runs[0]["dataset"],
            "seeds": [run["seed"] for run in runs],
            "n_runs": len(runs),
            "variants": {},
            "contrasts": {},
        }
    k = max(runs[0]["config"]["ks"])
    names = []
    for run in runs:
        for name in run["variants"]:
            if name not in names:
                names.append(name)
    summary = {
        "dataset": runs[0]["dataset"],
        "seeds": [run["seed"] for run in runs],
        "n_runs": len(runs),
        "variants": {},
        "contrasts": {},
    }
    for name in names:
        summary["variants"][name] = {}
        for path in (
            ("test", f"ndcg@{k}"),
            ("test", f"recall@{k}"),
            ("test", f"tail_ndcg@{k}"),
            ("test", f"head_ndcg@{k}"),
            ("test", f"head_tail_gap@{k}"),
            ("structure", "leakage"),
            ("structure", "jaccard_preference"),
            ("structure", "jaccard_exposure"),
            ("structure", "knn_popularity_corr"),
            ("structure", "item_cka_to_counterfactual"),
            ("structure", "user_cka_to_counterfactual"),
            ("shift", f"ndcg@{k}"),
            ("shift", f"tail_ndcg@{k}"),
        ):
            values = [_dig(run["variants"].get(name, {}), path) for run in runs]
            summary["variants"][name]["/".join(path)] = _mean_std(values)
    contrast_names = runs[0].get("contrasts", {})
    for name in contrast_names:
        values = [run.get("contrasts", {}).get(name) for run in runs]
        summary["contrasts"][name] = _mean_std(values)
    return summary


def _mean_std(values) -> dict:
    clean = []
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        clean.append(float(value))
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    array = np.asarray(clean, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {"mean": float(array.mean()), "std": std, "n": int(len(array))}
