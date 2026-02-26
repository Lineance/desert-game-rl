from __future__ import annotations

import torch
import torch.nn as nn


class ResourceEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
