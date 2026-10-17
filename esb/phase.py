"""Phase diagram over exposure skew, alignment, and propagation depth."""

from __future__ import annotations

import copy

import numpy as np

from esb.experiment import run_one

DECISION_MODELS = ("raw", "uniform_matched", "ips_graph", "ips_loss", "resampled", "counterfactual")


def run_phase(cfg, device: str, alphas, aligns, layer_grid, seeds) -> dict:
    rows = []
    for alpha in alphas:
        for align in aligns:
            for layers in layer_grid:
                for seed in seeds:
                    cell = copy.copy(cfg)
                    cell.alpha = float(alpha)
                    cell.exposure_align = float(align)
                    cell.layers = int(layers)
                    print(
                        f"\n##### phase alpha={cell.alpha:g} align={cell.exposure_align:g} "
                        f"layers={cell.layers} seed={seed} #####",
                        flush=True,
                    )
                    rows.append(flatten_run(run_one(cell, int(seed), device)))
    return summarize_phase(rows)


def flatten_run(result: dict) -> dict:
    if not result.get("variants"):
        k = None
    else:
        k = max(result["config"]["ks"])
    variants = result.get("variants") or {}

    def metric(name, *path):
        node = variants.get(name)
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        if isinstance(node, float) and np.isnan(node):
            return None
        return node

    present = [name for name in DECISION_MODELS if name in variants]
    leakage = {name: metric(name, "structure", "leakage") for name in present}
    ndcg = {name: metric(name, "test", f"ndcg@{k}") for name in present} if k else {}
    shift = {name: metric(name, "shift", f"ndcg@{k}") for name in present} if k else {}
    val = {name: metric(name, "best_val_ndcg") for name in present}
    probe = result.get("propagation_probe") or {}
    return {
        "dataset": result["dataset"],
        "alpha": result["config"]["alpha"],
        "exposure_align": result["config"].get("exposure_align", 0.0),
        "layers": result["config"]["layers"],
        "seed": result["seed"],
        "head_in_aligned_community": result["data"].get("head_in_aligned_community"),
        "head_edge_share": result["data"].get("head_edge_share"),
        "leakage": leakage,
        "ndcg": ndcg,
        "shift_ndcg": shift,
        "val_ndcg": val,
        "probe_gaps": probe.get("gaps"),
        "selector": result.get("selector"),
    }


def summarize_phase(rows: list[dict]) -> dict:
    grouped: dict[tuple, list] = {}
    for row in rows:
        key = (row["alpha"], row["exposure_align"], row["layers"])
        grouped.setdefault(key, []).append(row)
    cells = [_average_cell(key, group) for key, group in sorted(grouped.items())]
    return {
        "reading": (
            "A neighborhood has flipped when raw leakage is positive and larger than "
            "the density-matched graph. The support gap is raw minus uniform_matched. "
            "The weight gap is raw minus ips_graph. Extra layers should widen the "
            "support gap by more than the weight gap. Leakage ranks the shifted test "
            "when its Spearman with shifted NDCG is negative and stronger than "
            "validation NDCG. That correlation uses raw, uniform_matched, ips_graph, "
            "and ips_loss, which share an edge count."
        ),
        "n_runs": len(rows),
        "cells": cells,
        "frontier": _frontier(cells),
        "propagation": _propagation(cells),
        "trained_depth": _trained_depth(cells),
        "selector": _selector_summary(rows),
    }


def _average_cell(key, group: list[dict]) -> dict:
    alpha, align, layers = key
    leakage = _mean_dict([row["leakage"] for row in group])
    ndcg = _mean_dict([row["ndcg"] for row in group])
    shift = _mean_dict([row["shift_ndcg"] for row in group])
    probe_gaps = _mean_probe([row.get("probe_gaps") for row in group])
    raw = leakage.get("raw")
    uniform = leakage.get("uniform_matched")
    flipped = None
    if raw is not None and uniform is not None:
        flipped = bool(raw > 0 and raw > uniform)
    elif raw is not None:
        flipped = bool(raw > 0)
    return {
        "alpha": alpha,
        "exposure_align": align,
        "layers": layers,
        "n_seeds": len(group),
        "leakage": leakage,
        "ndcg": ndcg,
        "shift_ndcg": shift,
        "probe_gaps": probe_gaps,
        "neighborhood_flipped": flipped,
        "head_edge_share_raw": _mean_nested(group, ("head_edge_share", "raw")),
        "head_edge_share_uniform": _mean_nested(group, ("head_edge_share", "uniform_matched")),
    }


def _frontier(cells: list[dict]) -> list[dict]:
    buckets: dict[tuple, list] = {}
    for cell in cells:
        buckets.setdefault((cell["exposure_align"], cell["layers"]), []).append(cell)
    found = []
    for (align, layers), group in sorted(buckets.items()):
        ordered = sorted(group, key=lambda cell: cell["alpha"])
        crossing = None
        for cell in ordered:
            if cell["neighborhood_flipped"]:
                crossing = cell["alpha"]
                break
        found.append(
            {
                "exposure_align": align,
                "layers": layers,
                "alpha_when_neighborhood_flips": crossing,
                "alphas": [cell["alpha"] for cell in ordered],
                "flipped": [cell["neighborhood_flipped"] for cell in ordered],
            }
        )
    return found


def _propagation(cells: list[dict]) -> list[dict]:
    reports = []
    for cell in cells:
        gaps = cell.get("probe_gaps") or {}
        support_1 = _gap(gaps, "1", "raw_minus_uniform_matched")
        support_3 = _gap(gaps, "3", "raw_minus_uniform_matched")
        ips_1 = _gap(gaps, "1", "raw_minus_ips_graph")
        ips_3 = _gap(gaps, "3", "raw_minus_ips_graph")
        support_grew = _grew(support_1, support_3)
        ips_grew = _grew(ips_1, ips_3)
        ips_grew_less = None
        if support_grew is not None and ips_grew is not None and support_1 is not None and ips_1 is not None:
            support_change = support_3 - support_1
            ips_change = ips_3 - ips_1
            ips_grew_less = bool(ips_change < support_change)
        reports.append(
            {
                "alpha": cell["alpha"],
                "exposure_align": cell["exposure_align"],
                "trained_layers": cell["layers"],
                "support_gap_L1": support_1,
                "support_gap_L3": support_3,
                "ips_gap_L1": ips_1,
                "ips_gap_L3": ips_3,
                "support_gap_grew": support_grew,
                "ips_gap_grew_less_than_support": ips_grew_less,
            }
        )
    return reports


def _trained_depth(cells: list[dict]) -> list[dict]:
    by_setting: dict[tuple, dict] = {}
    for cell in cells:
        by_setting.setdefault((cell["alpha"], cell["exposure_align"]), {})[cell["layers"]] = cell
    reports = []
    for (alpha, align), layers in sorted(by_setting.items()):
        if 1 not in layers or 3 not in layers:
            continue
        support_1 = _model_gap(layers[1], "uniform_matched")
        support_3 = _model_gap(layers[3], "uniform_matched")
        ips_1 = _model_gap(layers[1], "ips_graph")
        ips_3 = _model_gap(layers[3], "ips_graph")
        reports.append(
            {
                "alpha": alpha,
                "exposure_align": align,
                "support_gap_L1": support_1,
                "support_gap_L3": support_3,
                "ips_gap_L1": ips_1,
                "ips_gap_L3": ips_3,
                "support_gap_grew": _grew(support_1, support_3),
                "ips_gap_grew_less_than_support": _grew_less(support_1, support_3, ips_1, ips_3),
            }
        )
    return reports


def _selector_summary(rows: list[dict]) -> dict:
    val_shift = []
    leak_shift = []
    better = []
    for row in rows:
        selector = row.get("selector") or {}
        if selector.get("spearman_val_vs_shift") is not None:
            val_shift.append(selector["spearman_val_vs_shift"])
        if selector.get("spearman_leakage_vs_shift") is not None:
            leak_shift.append(selector["spearman_leakage_vs_shift"])
        flag = selector.get("leakage_ranks_shift_better_than_val")
        if flag is not None:
            better.append(bool(flag))
    return {
        "mean_spearman_val_vs_shift": _mean(val_shift),
        "mean_spearman_leakage_vs_shift": _mean(leak_shift),
        "n_cells_leakage_ranks_shift": int(sum(better)),
        "n_cells_compared": len(better),
    }


def _model_gap(cell, other: str):
    raw = cell["leakage"].get("raw")
    alt = cell["leakage"].get(other)
    if raw is None or alt is None:
        return None
    return raw - alt


def _gap(gaps, layer, name):
    layer_gaps = gaps.get(layer) or gaps.get(int(layer)) or {}
    value = layer_gaps.get(name)
    if value is None:
        return None
    return float(value)


def _grew(early, late):
    if early is None or late is None:
        return None
    return bool(late > early)


def _grew_less(support_1, support_3, ips_1, ips_3):
    if None in (support_1, support_3, ips_1, ips_3):
        return None
    return bool((ips_3 - ips_1) < (support_3 - support_1))


def _mean(values) -> float | None:
    clean = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            clean.append(number)
    if not clean:
        return None
    return float(np.mean(clean))


def _mean_dict(rows: list[dict]) -> dict:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return {key: _mean([row.get(key) for row in rows]) for key in keys}


def _mean_probe(probes: list) -> dict:
    layers = []
    names = []
    usable = [probe for probe in probes if isinstance(probe, dict)]
    for probe in usable:
        for layer, gaps in probe.items():
            if layer not in layers:
                layers.append(layer)
            for name in gaps:
                if name not in names:
                    names.append(name)
    averaged = {}
    for layer in layers:
        averaged[layer] = {
            name: _mean([(probe.get(layer) or {}).get(name) for probe in usable])
            for name in names
        }
    return averaged


def _mean_nested(group, path):
    values = []
    for row in group:
        node = row
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        values.append(node)
    return _mean(values)
