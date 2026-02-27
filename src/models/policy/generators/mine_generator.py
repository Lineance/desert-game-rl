from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class MineGenerator(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)

    def _distribution(self, state: torch.Tensor):
        feat = self.net(state)
        alpha = torch.nn.functional.softplus(self.alpha_head(feat)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(feat)) + 1.0
        return torch.distributions.Beta(alpha, beta)

    def sample(self, state: torch.Tensor, can_mine: bool) -> Dict[str, torch.Tensor]:
        if not can_mine:
            zero = torch.zeros(state.shape[0], device=state.device)
            return {"mine_intensity": zero, "log_prob": zero, "entropy": zero}
        dist = self._distribution(state)
        intensity = dist.sample().squeeze(-1)
        return {
            "mine_intensity": intensity,
            "log_prob": dist.log_prob(intensity.unsqueeze(-1)).squeeze(-1),
            "entropy": dist.entropy().squeeze(-1),
        }

    def evaluate(
        self, state: torch.Tensor, mine_flag: torch.Tensor, can_mine: bool
    ) -> Dict[str, torch.Tensor]:
        if not can_mine:
            zero = torch.zeros(state.shape[0], device=state.device)
            return {"log_prob": zero, "entropy": zero}
        dist = self._distribution(state)
        eps = 1e-4
        intensity = torch.clamp(mine_flag.float(), eps, 1.0 - eps).unsqueeze(-1)
        return {
            "log_prob": dist.log_prob(intensity).squeeze(-1),
            "entropy": dist.entropy().squeeze(-1),
        }
