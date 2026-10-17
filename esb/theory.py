"""Leakage of the LightGCN adjacency polynomial, before any ranking loss.

An inverse-propensity weight rescales edges that exist. An edge the exposure
policy removed is absent from every power of the adjacency, so extra layers
still propagate the exposed support. The probe uses one shared initial
embedding and only changes the adjacency.
"""

from __future__ import annotations

import numpy as np
import torch

from esb.metrics import embedding_knn, neighborhood_report
from esb.model import normalized_adjacency
from esb.sampling import assign_ips


PROBE_LAYERS = (1, 2, 3)
PROBE_DIM = 32


def propagation_probe(split, anchors, ips_clip: float = 10.0) -> dict | None:
    if anchors is None:
        return None
    preference = anchors["_preference"]
    exposure = anchors["_exposure"]
    valid = anchors["_valid"]
    popularity = np.asarray(split.policy_popularity, dtype=np.float64)
    k = int(anchors["k"])
    graphs = _graphs(split, ips_clip)
    ego = torch.randn(split.n_users + split.n_items, PROBE_DIM, generator=torch.Generator().manual_seed(20260317))
    leakage = {}
    for name, edges, weights in graphs:
        adjacency = normalized_adjacency(split.n_users, split.n_items, edges, weights)
        leakage[name] = {}
        for n_layers in PROBE_LAYERS:
            items = _propagate(adjacency, ego, split.n_users, n_layers)
            neighbors = embedding_knn(items, k)
            report = neighborhood_report(neighbors, preference, exposure, valid, popularity)
            leakage[name][str(n_layers)] = report["leakage"]
    gaps = {}
    for n_layers in PROBE_LAYERS:
        key = str(n_layers)
        raw = leakage["raw"][key]
        ips = leakage["ips_graph"][key]
        uniform = leakage.get("uniform_matched", {}).get(key)
        gaps[key] = {
            "raw_minus_uniform_matched": None if uniform is None else raw - uniform,
            "raw_minus_ips_graph": raw - ips,
        }
    return {"leakage": leakage, "gaps": gaps}


def _graphs(split, ips_clip: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
    raw_ones = np.ones(len(split.raw_edges), dtype=np.float32)
    graphs = [("raw", split.raw_edges, raw_ones)]
    if split.uniform_edges is not None and len(split.uniform_edges):
        graphs.append(
            (
                "uniform_matched",
                split.uniform_edges,
                np.ones(len(split.uniform_edges), dtype=np.float32),
            )
        )
    graph_weight, _ = assign_ips(split.raw_propensity, "graph", ips_clip)
    graphs.append(("ips_graph", split.raw_edges, graph_weight))
    graphs.append(
        (
            "counterfactual",
            split.counterfactual_edges,
            np.ones(len(split.counterfactual_edges), dtype=np.float32),
        )
    )
    return graphs


def _propagate(adjacency: torch.Tensor, ego: torch.Tensor, n_users: int, n_layers: int) -> np.ndarray:
    hidden = ego
    accumulated = ego
    for _ in range(n_layers):
        hidden = torch.sparse.mm(adjacency, hidden)
        accumulated = accumulated + hidden
    averaged = accumulated / (n_layers + 1)
    return averaged[n_users:].detach().cpu().numpy()
