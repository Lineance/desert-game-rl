from __future__ import annotations

import torch
import torch.nn as nn


class GateFusion(nn.Module):
    def __init__(self, graph_dim: int = 64, resource_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(graph_dim * 2 + resource_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )
        self.map_branch = nn.Sequential(
            nn.Linear(graph_dim * 2, output_dim),
            nn.ReLU(),
        )
        self.res_branch = nn.Sequential(
            nn.Linear(resource_dim, output_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self, current_node: torch.Tensor, global_graph: torch.Tensor, resource: torch.Tensor
    ) -> torch.Tensor:
        gate_input = torch.cat([current_node, global_graph, resource], dim=-1)
        z = self.gate(gate_input)
        map_repr = self.map_branch(torch.cat([current_node, global_graph], dim=-1))
        res_repr = self.res_branch(resource)
        fused = z * map_repr + (1.0 - z) * res_repr
        return self.norm(fused)
