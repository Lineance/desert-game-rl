from __future__ import annotations

from typing import Dict

import torch


class InitialBuyGenerator:
    def __call__(
        self, max_buy_water: int, max_buy_food: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        water = int(max_buy_water * 0.8)
        food = int(max_buy_food * 0.8)
        water_tensor = torch.tensor([water], dtype=torch.long, device=device)
        food_tensor = torch.tensor([food], dtype=torch.long, device=device)
        zero = torch.zeros(1, device=device)
        return {
            "buy_water": water_tensor,
            "buy_food": food_tensor,
            "log_prob": zero,
            "entropy": zero,
        }
