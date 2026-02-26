from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    returns: List[float] = []
    advantages: List[float] = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    return returns, advantages


def clip_gradients(module: nn.Module, max_norm: float) -> float:
    return float(nn.utils.clip_grad_norm_(module.parameters(), max_norm).item())


def dynamic_alpha(
    water_ratio: torch.Tensor, food_ratio: torch.Tensor, dist_to_end: torch.Tensor
) -> torch.Tensor:
    alpha = torch.full_like(water_ratio, 0.5)
    survival_priority = (water_ratio < 0.2) | (food_ratio < 0.2)
    near_end = dist_to_end < 0.2
    alpha = torch.where(survival_priority, torch.full_like(alpha, 0.9), alpha)
    alpha = torch.where(~survival_priority & near_end, torch.full_like(alpha, 0.1), alpha)
    return alpha
