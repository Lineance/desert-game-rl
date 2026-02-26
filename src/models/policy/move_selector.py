from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from src.utils.mask_utils import apply_move_mask


class MoveSelector(nn.Module):
    def __init__(self, state_dim: int, node_dim: int):
        super().__init__()
        self.query = nn.Linear(state_dim, node_dim)
        self.key = nn.Linear(node_dim, node_dim)

    def forward(
        self,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        valid_actions: Dict,
    ) -> Dict[str, torch.Tensor]:
        q = self.query(state)
        keys = self.key(node_embeddings)
        logits = torch.matmul(keys, q.unsqueeze(-1)).squeeze(-1) / (keys.shape[-1] ** 0.5)
        logits = apply_move_mask(logits, valid_actions.get("valid_moves", []))
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}

    def sample(
        self, state: torch.Tensor, node_embeddings: torch.Tensor, valid_actions: Dict
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(state, node_embeddings, valid_actions)
        dist = torch.distributions.Categorical(out["probs"])
        move = dist.sample()
        return {"move": move, "log_prob": dist.log_prob(move), "entropy": dist.entropy()}
