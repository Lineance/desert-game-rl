from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn

from src.models.policy.generators.buy_generator import BuyGenerator
from src.models.policy.generators.initial_buy import InitialBuyGenerator
from src.models.policy.generators.mine_generator import MineGenerator
from src.utils.mask_utils import infer_location_type


@dataclass
class PolicyContext:
    state: torch.Tensor
    node_embeddings: torch.Tensor
    day_norm: torch.Tensor
    target_position: int
    valid_actions: Dict


class LocationActionSelector(nn.Module):
    def __init__(
        self,
        state_dim: int,
        start: int,
        end: int,
        mines: Sequence[int],
        villages: Sequence[int],
    ):
        super().__init__()
        self.start = int(start)
        self.end = int(end)
        self.mines = set(int(m) for m in mines)
        self.villages = set(int(v) for v in villages)
        self.buy_generator = BuyGenerator(state_dim)
        self.mine_generator = MineGenerator(state_dim)
        self.initial_buy = InitialBuyGenerator()

    def sample(self, ctx: PolicyContext) -> Dict[str, torch.Tensor]:
        location_type = infer_location_type(
            ctx.target_position,
            self.start,
            self.end,
            tuple(self.mines),
            tuple(self.villages),
        )
        valid = ctx.valid_actions
        can_buy = bool(valid.get("can_buy", False))
        can_mine = bool(valid.get("can_mine", False))
        max_buy_water = int(valid.get("max_buy_water", 0))
        max_buy_food = int(valid.get("max_buy_food", 0))
        is_day0 = bool(float(ctx.day_norm.item()) < 1e-6)

        zero = torch.zeros(ctx.state.shape[0], device=ctx.state.device)
        out = {
            "mine": torch.zeros(ctx.state.shape[0], dtype=torch.long, device=ctx.state.device),
            "mine_intensity": zero,
            "buy_water": torch.zeros(ctx.state.shape[0], dtype=torch.long, device=ctx.state.device),
            "buy_food": torch.zeros(ctx.state.shape[0], dtype=torch.long, device=ctx.state.device),
            "log_prob": zero,
            "entropy": zero,
        }

        if location_type == "village" and can_buy:
            buy = self.buy_generator.sample(ctx.state, max_buy_water, max_buy_food)
            out.update({"buy_water": buy["buy_water"], "buy_food": buy["buy_food"]})
            out["log_prob"] = out["log_prob"] + buy["log_prob"]
            out["entropy"] = out["entropy"] + buy["entropy"]
            return out

        if location_type == "mine" and can_mine:
            mine = self.mine_generator.sample(ctx.state, can_mine=True)
            mine_flag = (mine["mine_intensity"] > 0.5).long()
            out.update({"mine": mine_flag, "mine_intensity": mine["mine_intensity"]})
            out["log_prob"] = out["log_prob"] + mine["log_prob"]
            out["entropy"] = out["entropy"] + mine["entropy"]
            return out

        if location_type == "start" and is_day0 and can_buy:
            init_buy = self.initial_buy(max_buy_water, max_buy_food, ctx.state.device)
            out.update({"buy_water": init_buy["buy_water"], "buy_food": init_buy["buy_food"]})
            out["log_prob"] = out["log_prob"] + init_buy["log_prob"]
            out["entropy"] = out["entropy"] + init_buy["entropy"]
            return out

        return out

    def evaluate(
        self, ctx: PolicyContext, actions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        location_type = infer_location_type(
            ctx.target_position,
            self.start,
            self.end,
            tuple(self.mines),
            tuple(self.villages),
        )
        valid = ctx.valid_actions
        can_buy = bool(valid.get("can_buy", False))
        can_mine = bool(valid.get("can_mine", False))
        max_buy_water = int(valid.get("max_buy_water", 0))
        max_buy_food = int(valid.get("max_buy_food", 0))
        is_day0 = bool(float(ctx.day_norm.item()) < 1e-6)

        zero = torch.zeros(ctx.state.shape[0], device=ctx.state.device)
        log_prob = zero
        entropy = zero

        if location_type == "village" and can_buy:
            buy_eval = self.buy_generator.evaluate(
                ctx.state,
                actions["buy_water"],
                actions["buy_food"],
                max_buy_water,
                max_buy_food,
            )
            log_prob = log_prob + buy_eval["log_prob"]
            entropy = entropy + buy_eval["entropy"]
        elif location_type == "mine" and can_mine:
            mine_eval = self.mine_generator.evaluate(ctx.state, actions["mine"], can_mine=True)
            log_prob = log_prob + mine_eval["log_prob"]
            entropy = entropy + mine_eval["entropy"]
        elif location_type == "start" and is_day0 and can_buy:
            pass

        return {"log_prob": log_prob, "entropy": entropy}
