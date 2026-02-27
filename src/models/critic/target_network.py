from __future__ import annotations

from copy import deepcopy

import torch.nn as nn


class TargetNetwork:
    def __init__(self, online: nn.Module):
        self.online = online
        self.target = deepcopy(online)
        self.target.eval()

    def soft_update(self, tau: float = 0.995) -> None:
        try:
            online_device = next(self.online.parameters()).device
        except StopIteration:
            return

        try:
            target_device = next(self.target.parameters()).device
        except StopIteration:
            return

        if target_device != online_device:
            self.target.to(online_device)

        for t_param, o_param in zip(self.target.parameters(), self.online.parameters()):
            t_param.data.copy_(tau * t_param.data + (1.0 - tau) * o_param.data)
