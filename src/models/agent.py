"""
模块化智能体入口
"""

from __future__ import annotations

from os import PathLike
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.env.config import RLConfig
from src.models.belief import WeatherBeliefModel, extract_belief_features
from src.models.critic import FundCritic, SurvivalCritic, TargetNetwork
from src.models.encoders import GateFusion, ManualGNN, ResourceEncoder
from src.models.policy import LocationActionSelector, MoveSelector, PolicyContext
from src.utils.graph_utils import build_graph_static_data
from src.utils.mask_utils import normalize_valid_actions


class ActorNetwork(nn.Module):
    """分层策略网络。"""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_locations: int,
        start: int = 0,
        end: int = 0,
        mines: Optional[list[int]] = None,
        villages: Optional[list[int]] = None,
    ):
        super().__init__()
        self.num_locations = num_locations
        self.move_selector = MoveSelector(state_dim=state_dim, node_dim=hidden_dim)
        self.location_selector = LocationActionSelector(
            state_dim=state_dim,
            start=start,
            end=end,
            mines=mines or [],
            villages=villages or [],
        )

    def forward(
        self,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        day_norm: torch.Tensor,
        valid_actions: Optional[Dict],
    ) -> Dict[str, torch.Tensor]:
        valid = normalize_valid_actions(valid_actions)
        move_out = self.move_selector(state, node_embeddings, valid)
        can_mine = bool(valid.get("can_mine", False))
        mine_probs = torch.zeros((state.shape[0], 2), device=state.device, dtype=state.dtype)
        if can_mine:
            mine_probs[:, 0] = 0.5
            mine_probs[:, 1] = 0.5
        else:
            mine_probs[:, 0] = 1.0
            mine_probs[:, 1] = 0.0
        return {
            "move_probs": move_out["probs"],
            "move_logits": move_out["logits"],
            "day_norm": day_norm,
            "mine_probs": mine_probs,
            "buy_water_mean": torch.zeros(
                (state.shape[0], 1), device=state.device, dtype=state.dtype
            ),
            "buy_water_std": torch.ones(
                (state.shape[0], 1), device=state.device, dtype=state.dtype
            ),
            "buy_food_mean": torch.zeros(
                (state.shape[0], 1), device=state.device, dtype=state.dtype
            ),
            "buy_food_std": torch.ones((state.shape[0], 1), device=state.device, dtype=state.dtype),
        }

    def sample_action(
        self,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        day_norm: torch.Tensor,
        valid_actions: Optional[Dict],
    ):
        valid = normalize_valid_actions(valid_actions)
        move_sample = self.move_selector.sample(state, node_embeddings, valid)
        target_position = int(move_sample["move"].item())
        ctx = PolicyContext(
            state=state,
            node_embeddings=node_embeddings,
            day_norm=day_norm,
            target_position=target_position,
            valid_actions=valid,
        )
        loc_sample = self.location_selector.sample(ctx)

        action = {
            "move": target_position,
            "mine": bool(loc_sample["mine"].item() > 0),
            "mine_intensity": float(loc_sample["mine_intensity"].item()),
            "buy_water": int(loc_sample["buy_water"].item()),
            "buy_food": int(loc_sample["buy_food"].item()),
        }
        log_prob = move_sample["log_prob"] + loc_sample["log_prob"]
        return action, log_prob

    def evaluate_actions(
        self,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        day_norm: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        valid_actions: Optional[Dict],
    ):
        valid = normalize_valid_actions(valid_actions)
        move_out = self.move_selector(state, node_embeddings, valid)
        move_dist = torch.distributions.Categorical(move_out["probs"])
        move_log_prob = move_dist.log_prob(actions["move"])
        move_entropy = move_dist.entropy()

        target_position = int(actions["move"].item())
        ctx = PolicyContext(
            state=state,
            node_embeddings=node_embeddings,
            day_norm=day_norm,
            target_position=target_position,
            valid_actions=valid,
        )
        loc_eval = self.location_selector.evaluate(ctx, actions)
        log_probs = move_log_prob + loc_eval["log_prob"]
        entropy = move_entropy + loc_eval["entropy"]
        return log_probs, entropy


class Agent(nn.Module):
    def __init__(self, obs_dim: int, num_locations: int, config: Optional[RLConfig] = None):
        super().__init__()
        if config is None:
            config = RLConfig()

        self.config = config
        self.obs_dim = obs_dim
        self.num_locations = num_locations

        dummy_belief_model = WeatherBeliefModel()
        dummy_belief_model.update(0)
        dummy_features = extract_belief_features(dummy_belief_model)
        self.belief_dim = len(dummy_features)

        self.resource_encoder = ResourceEncoder(input_dim=6, hidden_dim=64, output_dim=64)

        self.graph_encoder = ManualGNN(
            num_nodes=num_locations,
            node_feat_dim=8,
            hidden_dim=64,
            num_layers=3,
        )
        self.fusion = GateFusion(graph_dim=64, resource_dim=64, output_dim=128)

        self.survival_critic = SurvivalCritic(state_dim=128, hidden_dim=config.HIDDEN_DIM)
        self.fund_critic = FundCritic(state_dim=128, hidden_dim=config.HIDDEN_DIM)
        self.survival_target = TargetNetwork(self.survival_critic)
        self.fund_target = TargetNetwork(self.fund_critic)

        self.actor = ActorNetwork(
            state_dim=128,
            hidden_dim=64,
            num_locations=num_locations,
        )

        self.register_buffer("graph_node_features", torch.zeros(1, num_locations, 8))
        self.register_buffer("graph_norm_adj", torch.eye(num_locations).unsqueeze(0))

    def set_graph_structure(self, env) -> None:
        graph = build_graph_static_data(
            num_nodes=env.config.NUM_NODES,
            edges_1_based=env.config.EDGES,
            end_node=env.config.END,
            mines=env.config.MINES,
        )

        node_feat = np.zeros((env.config.NUM_NODES, 8), dtype=np.float32)
        for idx in range(env.config.NUM_NODES):
            node_feat[idx, 0] = 1.0 if idx == env.config.START else 0.0
            node_feat[idx, 1] = 1.0 if idx == env.config.END else 0.0
            node_feat[idx, 2] = 1.0 if idx in env.config.MINES else 0.0
            node_feat[idx, 3] = 1.0 if idx in env.config.VILLAGES else 0.0
            node_feat[idx, 4] = float(idx) / max(1.0, float(env.config.NUM_NODES - 1))
            node_feat[idx, 5] = graph.dist_to_end[idx] / max(1.0, float(env.config.NUM_NODES))
            node_feat[idx, 6] = graph.dist_to_mine[idx] / max(1.0, float(env.config.NUM_NODES))
            node_feat[idx, 7] = float(len(graph.neighbors[idx])) / max(
                1.0, float(env.config.NUM_NODES)
            )

        self.graph_node_features = torch.from_numpy(node_feat).unsqueeze(0)
        self.graph_norm_adj = torch.from_numpy(graph.normalized_adjacency).unsqueeze(0)

        self.actor = ActorNetwork(
            state_dim=128,
            hidden_dim=64,
            num_locations=env.config.NUM_NODES,
            start=env.config.START,
            end=env.config.END,
            mines=list(env.config.MINES),
            villages=list(env.config.VILLAGES),
        ).to(next(self.parameters()).device)

    @staticmethod
    def _extract_resource_features(obs: torch.Tensor) -> torch.Tensor:
        day = obs[:, 0:1]
        water = obs[:, 2:3]
        food = obs[:, 3:4]
        money = obs[:, 4:5]
        dist_to_end = obs[:, 5:6]
        remaining_day = 1.0 - day
        return torch.cat([water, food, money, day, remaining_day, dist_to_end], dim=-1)

    def _encode_graph(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        node_feat = self.graph_node_features.expand(batch_size, -1, -1)
        norm_adj = self.graph_norm_adj.expand(batch_size, -1, -1)
        return self.graph_encoder(node_feat, norm_adj)

    def encode_observation_full(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        node_embeddings, global_graph = self._encode_graph(batch_size)
        resource_features = self._extract_resource_features(obs)
        resource_embed = self.resource_encoder(resource_features)

        positions = torch.clamp(
            torch.round(obs[:, 1] * (self.num_locations - 1)).long(),
            min=0,
            max=self.num_locations - 1,
        )
        batch_idx = torch.arange(batch_size, device=obs.device)
        current_node = node_embeddings[batch_idx, positions]
        state = self.fusion(current_node, global_graph, resource_embed)
        day_norm = obs[:, 0:1]
        return state, node_embeddings, day_norm

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        state, _, _ = self.encode_observation_full(obs)
        return state

    def forward(self, obs: torch.Tensor, valid_actions: Optional[Dict] = None):
        state, node_embeddings, day_norm = self.encode_observation_full(obs)
        action_dist = self.actor(state, node_embeddings, day_norm, valid_actions)
        survival_value = self.survival_critic(state)
        fund_value = self.fund_critic(state)
        value = 0.5 * (survival_value + fund_value)
        return {
            "action_dist": action_dist,
            "value": value,
            "survival_value": survival_value,
            "fund_value": fund_value,
            "state": state,
            "node_embeddings": node_embeddings,
            "day_norm": day_norm,
        }

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[Dict] = None,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ) -> Tuple[Dict, float]:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(next(self.parameters()).device)
            output = self.forward(obs_tensor, valid_actions)
            state = output["state"]
            node_embeddings = output["node_embeddings"]
            day_norm = output["day_norm"]

            if deterministic:
                move_probs = output["action_dist"]["move_probs"]
                move = int(torch.argmax(move_probs, dim=-1).item())
                ctx = PolicyContext(
                    state=state,
                    node_embeddings=node_embeddings,
                    day_norm=day_norm,
                    target_position=move,
                    valid_actions=normalize_valid_actions(valid_actions),
                )
                loc_sample = self.actor.location_selector.sample(ctx)
                action = {
                    "move": move,
                    "mine": bool(loc_sample["mine"].item() > 0),
                    "mine_intensity": float(loc_sample["mine_intensity"].item()),
                    "buy_water": int(loc_sample["buy_water"].item()),
                    "buy_food": int(loc_sample["buy_food"].item()),
                }
            else:
                action, _ = self.actor.sample_action(
                    state, node_embeddings, day_norm, valid_actions
                )

            if epsilon > 0.0 and np.random.random() < epsilon:
                valid = normalize_valid_actions(valid_actions)
                action["move"] = int(np.random.choice(valid["valid_moves"]))
                is_day0 = bool(len(obs) > 0 and float(obs[0]) < 1e-6)
                if valid["can_buy"] and is_day0:
                    max_buy_water = int(valid.get("max_buy_water", 0))
                    max_buy_food = int(valid.get("max_buy_food", 0))
                    action["buy_water"] = (
                        int(np.random.randint(1, max_buy_water + 1)) if max_buy_water > 0 else 0
                    )
                    action["buy_food"] = (
                        int(np.random.randint(1, max_buy_food + 1)) if max_buy_food > 0 else 0
                    )
                if not valid.get("can_mine", False):
                    action["mine"] = False
                    action["mine_intensity"] = 0.0

            value = output["value"].item()
        return action, value

    def evaluate_action_log_prob(
        self,
        obs_tensor: torch.Tensor,
        action: Dict[str, torch.Tensor],
        valid_actions: Optional[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state, node_embeddings, day_norm = self.encode_observation_full(obs_tensor)
        return self.actor.evaluate_actions(state, node_embeddings, day_norm, action, valid_actions)

    def set_stage_trainability(self, stage_name: str, freeze_fund_critic: bool) -> None:
        _ = stage_name
        for param in self.fund_critic.parameters():
            param.requires_grad = not freeze_fund_critic

    def get_trainability_snapshot(self) -> Dict[str, bool]:
        fund_flags = [bool(p.requires_grad) for p in self.fund_critic.parameters()]
        return {
            "fund_critic_trainable": bool(all(fund_flags)) if fund_flags else False,
            "fund_critic_frozen": bool(not any(fund_flags)) if fund_flags else True,
        }

    def update_targets(self, tau: float = 0.995) -> None:
        self.survival_target.soft_update(tau=tau)
        self.fund_target.soft_update(tau=tau)

    def reset_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
        pass

    def save(self, path: Union[str, PathLike[str]]):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "num_locations": self.num_locations,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        agent = cls(
            obs_dim=checkpoint["obs_dim"],
            num_locations=checkpoint["num_locations"],
            config=checkpoint["config"],
        )
        agent.load_state_dict(checkpoint["state_dict"])
        agent.to(device)
        return agent


def create_agent(env, config: Optional[RLConfig] = None, device: str = "cpu"):
    obs_dim = 19
    agent = Agent(obs_dim=obs_dim, num_locations=env.config.NUM_NODES, config=config)
    agent.set_graph_structure(env)
    agent.to(device)
    return agent
