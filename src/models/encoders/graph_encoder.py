from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ManualGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.self_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.nei_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        normalized_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_features)
        for self_layer, nei_layer, norm in zip(self.self_layers, self.nei_layers, self.norms):
            agg = normalized_adj @ h
            update = self_layer(h) + nei_layer(agg)
            h = norm(torch.relu(update) + h)

        g = self.global_proj(h.mean(dim=1))
        return h, g
