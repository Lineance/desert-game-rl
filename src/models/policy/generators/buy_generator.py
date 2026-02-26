from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class BuyGenerator(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

    def _distribution(self, state: torch.Tensor, max_buy_water: int, max_buy_food: int):
        feat = self.net(state)
        ratio_mean = torch.sigmoid(self.mean_head(feat))
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(ratio_mean)
        ratio_dist = torch.distributions.Normal(ratio_mean, std)
        return ratio_dist, ratio_mean

    def sample(
        self, state: torch.Tensor, max_buy_water: int, max_buy_food: int
    ) -> Dict[str, torch.Tensor]:
        ratio_dist, _ = self._distribution(state, max_buy_water, max_buy_food)
        ratio = torch.clamp(ratio_dist.sample(), 0.0, 1.0)
        max_tensor = torch.tensor(
            [max_buy_water, max_buy_food], device=state.device, dtype=state.dtype
        )
        amount = torch.round(ratio * max_tensor).long()
        log_prob = ratio_dist.log_prob(ratio).sum(-1)
        entropy = ratio_dist.entropy().sum(-1)
        return {
            "buy_water": amount[:, 0],
            "buy_food": amount[:, 1],
            "log_prob": log_prob,
            "entropy": entropy,
        }

    def evaluate(
        self,
        state: torch.Tensor,
        buy_water: torch.Tensor,
        buy_food: torch.Tensor,
        max_buy_water: int,
        max_buy_food: int,
    ) -> Dict[str, torch.Tensor]:
        ratio_dist, _ = self._distribution(state, max_buy_water, max_buy_food)
        max_w = max(float(max_buy_water), 1.0)
        max_f = max(float(max_buy_food), 1.0)
        ratio = torch.stack([buy_water.float() / max_w, buy_food.float() / max_f], dim=-1)
        ratio = torch.clamp(ratio, 0.0, 1.0)
        return {
            "log_prob": ratio_dist.log_prob(ratio).sum(-1),
            "entropy": ratio_dist.entropy().sum(-1),
        }
