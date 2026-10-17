"""LightGCN with a fixed weighted bipartite adjacency."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def normalized_adjacency(
    n_users: int,
    n_items: int,
    edges: np.ndarray,
    weights: np.ndarray,
) -> torch.Tensor:
    """Symmetric normalization of the undirected user–item graph. No self-loops."""
    if len(edges) == 0:
        size = n_users + n_items
        index = torch.zeros((2, 0), dtype=torch.long)
        value = torch.zeros(0, dtype=torch.float32)
        return torch.sparse_coo_tensor(index, value, (size, size)).coalesce()
    users = torch.as_tensor(edges[:, 0], dtype=torch.long)
    items = torch.as_tensor(edges[:, 1], dtype=torch.long) + n_users
    base = torch.as_tensor(weights, dtype=torch.float32)
    src = torch.cat([users, items])
    dst = torch.cat([items, users])
    doubled = torch.cat([base, base])
    size = n_users + n_items
    degree = torch.zeros(size, dtype=torch.float32)
    degree.scatter_add_(0, src, doubled)
    inv_sqrt = torch.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = degree[positive].pow(-0.5)
    norm = inv_sqrt[src] * doubled * inv_sqrt[dst]
    index = torch.stack([src, dst], dim=0)
    return torch.sparse_coo_tensor(index, norm, (size, size)).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, adjacency: torch.Tensor):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.register_buffer("adj_index", adjacency.indices())
        self.register_buffer("adj_value", adjacency.values())
        self.n_nodes = int(adjacency.size(0))
        self._adj: torch.Tensor | None = None

    def adjacency(self) -> torch.Tensor:
        weight = self.embedding.weight
        if self._adj is None or self._adj.device != weight.device:
            self._adj = torch.sparse_coo_tensor(
                self.adj_index,
                self.adj_value,
                (self.n_nodes, self.n_nodes),
            ).coalesce()
        return self._adj

    def encode(self, n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
        ego = self.embedding.weight
        if n_layers <= 0:
            return ego[: self.n_users], ego[self.n_users :]
        adjacency = self.adjacency()
        hidden = ego
        accumulated = ego
        for _ in range(n_layers):
            hidden = torch.sparse.mm(adjacency, hidden)
            accumulated = accumulated + hidden
        averaged = accumulated / (n_layers + 1)
        return averaged[: self.n_users], averaged[self.n_users :]
