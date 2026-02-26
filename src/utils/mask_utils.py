from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch


def masked_log_softmax(logits: torch.Tensor, allowed_indices: Sequence[int]) -> torch.Tensor:
    masked = torch.full_like(logits, float("-inf"))
    if len(allowed_indices) == 0:
        masked[..., 0] = 0.0
    else:
        idx_tensor = torch.tensor(list(allowed_indices), device=logits.device, dtype=torch.long)
        masked[..., idx_tensor] = 0.0
    return torch.log_softmax(logits + masked, dim=-1)


def apply_move_mask(logits: torch.Tensor, valid_moves: Iterable[int]) -> torch.Tensor:
    masked = torch.full_like(logits, float("-inf"))
    valid_moves = list(valid_moves)
    if not valid_moves:
        masked[..., 0] = 0.0
        return logits + masked
    idx = torch.tensor(valid_moves, device=logits.device, dtype=torch.long)
    masked[..., idx] = 0.0
    return logits + masked


def infer_location_type(
    position: int, start: int, end: int, mines: Sequence[int], villages: Sequence[int]
) -> str:
    if position == start:
        return "start"
    if position == end:
        return "end"
    if position in set(mines):
        return "mine"
    if position in set(villages):
        return "village"
    return "other"


def normalize_valid_actions(valid_actions: Dict | None) -> Dict:
    if valid_actions is None:
        return {
            "valid_moves": [0],
            "can_mine": False,
            "can_buy": False,
            "max_buy_water": 0,
            "max_buy_food": 0,
        }
    return {
        "valid_moves": list(valid_actions.get("valid_moves", [0])),
        "can_mine": bool(valid_actions.get("can_mine", False)),
        "can_buy": bool(valid_actions.get("can_buy", False)),
        "max_buy_water": int(valid_actions.get("max_buy_water", 0)),
        "max_buy_food": int(valid_actions.get("max_buy_food", 0)),
    }
