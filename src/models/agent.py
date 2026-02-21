"""
智能体
"""

from os import PathLike
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.config import RLConfig
from src.models.belief import extract_belief_features


class ActorNetwork(nn.Module):
    """Actor - 使用全局节点ID"""

    def __init__(self, state_dim: int, hidden_dim: int, num_locations: int):
        super().__init__()
        self.num_locations = num_locations

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.move_head = nn.Linear(hidden_dim, num_locations)
        self.mine_head = nn.Linear(hidden_dim, 2)
        self.buy_water_mean = nn.Linear(hidden_dim, 1)
        self.buy_water_logstd = nn.Parameter(torch.zeros(1))
        self.buy_food_mean = nn.Linear(hidden_dim, 1)
        self.buy_food_logstd = nn.Parameter(torch.zeros(1))

    def forward(self, state: torch.Tensor, valid_actions: Optional[Dict] = None):
        features = self.net(state)

        # 移动概率 - 全局输出+动态掩码
        move_logits = self.move_head(features)
        if valid_actions is not None and "valid_moves" in valid_actions:
            valid_moves = valid_actions["valid_moves"]
            # 创建全-inf掩码，然后对有效位置设为0
            mask = torch.full_like(move_logits, float("-inf"))
            for vm in valid_moves:
                if isinstance(vm, int) and vm < self.num_locations:
                    mask[:, vm] = 0
            move_logits = move_logits + mask
        # Softmax会在有效位置上归一化
        move_probs = F.softmax(move_logits, dim=-1)

        # 挖矿概率
        mine_logits = self.mine_head(features)
        if valid_actions is not None and "can_mine" in valid_actions:
            can_mine = bool(valid_actions["can_mine"])
            mine_mask = torch.tensor([[True, can_mine]], device=features.device, dtype=torch.bool)
            mine_logits = mine_logits.masked_fill(~mine_mask, float("-inf"))
        mine_probs = F.softmax(mine_logits, dim=-1)

        # 购买分布
        max_buy_water = 100
        max_buy_food = 100
        can_buy = True
        if valid_actions is not None:
            can_buy = bool(valid_actions.get("can_buy", True))
            max_buy_water = int(valid_actions.get("max_buy_water", 100))
            max_buy_food = int(valid_actions.get("max_buy_food", 100))

        buy_water_mean = torch.clamp(self.buy_water_mean(features), 0, max_buy_water)
        buy_water_std = torch.exp(self.buy_water_logstd)
        buy_food_mean = torch.clamp(self.buy_food_mean(features), 0, max_buy_food)
        buy_food_std = torch.exp(self.buy_food_logstd)

        if not can_buy:
            buy_water_mean = torch.zeros_like(buy_water_mean)
            buy_food_mean = torch.zeros_like(buy_food_mean)

        return {
            "move_probs": move_probs,
            "mine_probs": mine_probs,
            "buy_water_mean": buy_water_mean,
            "buy_water_std": buy_water_std,
            "buy_food_mean": buy_food_mean,
            "buy_food_std": buy_food_std,
        }

    def sample_action(self, state: torch.Tensor, valid_actions: Optional[Dict] = None):
        """采样动作"""
        dist = self(state, valid_actions)

        move_dist = torch.distributions.Categorical(dist["move_probs"])
        move_global_id = move_dist.sample()
        move_log_prob = move_dist.log_prob(move_global_id)

        mine_dist = torch.distributions.Categorical(dist["mine_probs"])
        mine_action = mine_dist.sample()
        mine_log_prob = mine_dist.log_prob(mine_action)

        can_buy = bool(valid_actions.get("can_buy", True)) if valid_actions is not None else True
        max_buy_water = (
            int(valid_actions.get("max_buy_water", 100)) if valid_actions is not None else 100
        )
        max_buy_food = (
            int(valid_actions.get("max_buy_food", 100)) if valid_actions is not None else 100
        )

        if can_buy:
            water_dist = torch.distributions.Normal(dist["buy_water_mean"], dist["buy_water_std"])
            buy_water = torch.clamp(water_dist.sample(), 0, max_buy_water)
            water_log_prob = water_dist.log_prob(buy_water).sum(-1)

            food_dist = torch.distributions.Normal(dist["buy_food_mean"], dist["buy_food_std"])
            buy_food = torch.clamp(food_dist.sample(), 0, max_buy_food)
            food_log_prob = food_dist.log_prob(buy_food).sum(-1)
        else:
            buy_water = torch.zeros_like(dist["buy_water_mean"])
            buy_food = torch.zeros_like(dist["buy_food_mean"])
            water_log_prob = torch.zeros_like(move_log_prob)
            food_log_prob = torch.zeros_like(move_log_prob)

        action = {
            "move": move_global_id.item(),
            "mine": mine_action.item() == 1,
            "buy_water": int(buy_water.item()),
            "buy_food": int(buy_food.item()),
        }

        log_prob = move_log_prob + mine_log_prob + water_log_prob + food_log_prob
        return action, log_prob

    def evaluate_actions(
        self,
        state: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        valid_actions: Optional[Dict] = None,
    ):
        dist = self.forward(state, valid_actions)

        move_dist = torch.distributions.Categorical(dist["move_probs"])
        move_log_prob = move_dist.log_prob(actions["move"])
        move_entropy = move_dist.entropy()

        mine_dist = torch.distributions.Categorical(dist["mine_probs"])
        mine_log_prob = mine_dist.log_prob(actions["mine"].long())
        mine_entropy = mine_dist.entropy()

        can_buy = bool(valid_actions.get("can_buy", True)) if valid_actions is not None else True
        if can_buy:
            water_dist = torch.distributions.Normal(dist["buy_water_mean"], dist["buy_water_std"])
            water_log_prob = water_dist.log_prob(actions["buy_water"].float()).sum(-1)
            water_entropy = 0.5 * torch.log(2 * np.pi * np.e * dist["buy_water_std"].pow(2)).sum(-1)

            food_dist = torch.distributions.Normal(dist["buy_food_mean"], dist["buy_food_std"])
            food_log_prob = food_dist.log_prob(actions["buy_food"].float()).sum(-1)
            food_entropy = 0.5 * torch.log(2 * np.pi * np.e * dist["buy_food_std"].pow(2)).sum(-1)
        else:
            water_log_prob = torch.zeros_like(move_log_prob)
            food_log_prob = torch.zeros_like(move_log_prob)
            water_entropy = torch.zeros_like(move_entropy)
            food_entropy = torch.zeros_like(move_entropy)

        log_probs = move_log_prob + mine_log_prob + water_log_prob + food_log_prob
        entropy = move_entropy + mine_entropy + water_entropy + food_entropy

        return log_probs, entropy


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class HybridRNNAgent(nn.Module):
    """简化版智能体 - MLP + Belief"""

    def __init__(self, obs_dim: int, num_locations: int, config: Optional[RLConfig] = None):
        super().__init__()
        if config is None:
            config = RLConfig()

        self.config = config
        self.obs_dim = obs_dim
        self.num_locations = num_locations

        # 信念编码器（动态获取维度）
        # 信念特征: probs(3) + confidence(1) + entropy(1) + mode(1) = 6
        from src.models.belief import WeatherBeliefModel

        dummy_belief_model = WeatherBeliefModel()
        dummy_belief_model.update(0)  # 更新一次生成信念
        dummy_features = extract_belief_features(dummy_belief_model)
        self.belief_dim = len(dummy_features)
        self.belief_encoder = nn.Sequential(
            nn.Linear(self.belief_dim, 64),
            nn.ReLU(),
        )

        # 状态编码器（其余部分）
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim - self.belief_dim, 128),
            nn.ReLU(),
        )

        self.combined_input_dim = 192
        self.lstm_hidden_dim = int(getattr(config, "LSTM_HIDDEN_DIM", config.HIDDEN_DIM))
        self.batch_first = bool(getattr(config, "BATCH_FIRST", True))

        self.temporal_encoder = nn.LSTM(
            input_size=self.combined_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=int(config.LSTM_LAYERS),
            batch_first=self.batch_first,
        )

        self.actor = ActorNetwork(self.lstm_hidden_dim, config.HIDDEN_DIM, num_locations)
        self.critic = CriticNetwork(self.lstm_hidden_dim, config.HIDDEN_DIM)

        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _encode_parts(self, obs_2d: torch.Tensor) -> torch.Tensor:
        state_part = obs_2d[:, : -self.belief_dim]
        belief_part = obs_2d[:, -self.belief_dim :]
        state_encoded = self.state_encoder(state_part)
        belief_encoded = self.belief_encoder(belief_part)
        return torch.cat([state_encoded, belief_encoded], dim=-1)

    def _to_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)
        elif obs.dim() != 3:
            raise ValueError(f"obs维度不支持: {tuple(obs.shape)}")
        return obs

    def hidden_from_numpy(
        self,
        hidden_state: Optional[Tuple[np.ndarray, np.ndarray]],
        device: Optional[torch.device] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if hidden_state is None:
            return None
        if device is None:
            device = next(self.parameters()).device
        h, c = hidden_state
        return (
            torch.from_numpy(np.asarray(h)).float().to(device),
            torch.from_numpy(np.asarray(c)).float().to(device),
        )

    def get_hidden_state_numpy(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.hidden_state is None:
            return None
        h, c = self.hidden_state
        return h.detach().cpu().numpy().copy(), c.detach().cpu().numpy().copy()

    def encode_sequence(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        obs_seq = self._to_sequence(obs)
        batch_size, seq_len, feat_dim = obs_seq.shape
        flat_obs = obs_seq.reshape(batch_size * seq_len, feat_dim)
        encoded_flat = self._encode_parts(flat_obs)
        encoded_seq = encoded_flat.reshape(batch_size, seq_len, -1)

        if hidden_state is None:
            hidden_state = self.hidden_state

        outputs, next_hidden = self.temporal_encoder(encoded_seq, hidden_state)
        return outputs, next_hidden

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """兼容接口：返回最后时间步的时序特征"""
        outputs, _ = self.encode_sequence(obs, hidden_state=None)
        return outputs[:, -1, :]

    def forward_step(
        self,
        obs: torch.Tensor,
        valid_actions: Optional[Dict] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        update_internal_hidden: bool = False,
    ):
        outputs, next_hidden = self.encode_sequence(obs, hidden_state=hidden_state)
        step_features = outputs[:, -1, :]
        action_dist = self.actor(step_features, valid_actions)
        value = self.critic(step_features)
        if update_internal_hidden:
            self.hidden_state = (next_hidden[0].detach(), next_hidden[1].detach())
        return {"action_dist": action_dist, "value": value, "next_hidden": next_hidden}

    def forward(self, obs: torch.Tensor, valid_actions: Optional[Dict] = None):
        return self.forward_step(obs, valid_actions, update_internal_hidden=False)

    def evaluate_actions_sequence(
        self,
        obs_seq: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        valid_actions_seq: Optional[list] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs, next_hidden = self.encode_sequence(obs_seq, hidden_state=hidden_state)
        values = self.critic(outputs).squeeze(-1)

        if outputs.size(0) != 1:
            raise ValueError("当前evaluate_actions_sequence仅支持batch_size=1")

        log_probs = []
        entropies = []
        seq_len = outputs.size(1)

        for t in range(seq_len):
            feat_t = outputs[:, t, :]
            action_t = {
                "move": actions["move"][t : t + 1],
                "mine": actions["mine"][t : t + 1],
                "buy_water": actions["buy_water"][t : t + 1],
                "buy_food": actions["buy_food"][t : t + 1],
            }
            valid_t = None
            if valid_actions_seq is not None and t < len(valid_actions_seq):
                valid_t = valid_actions_seq[t]
            lp_t, ent_t = self.actor.evaluate_actions(feat_t, action_t, valid_t)
            log_probs.append(lp_t.squeeze(0))
            entropies.append(ent_t.squeeze(0))

        return (
            torch.stack(log_probs),
            torch.stack(entropies),
            values.squeeze(0),
            next_hidden,
        )

    @staticmethod
    def _day0_purchase_floor(valid_actions: Optional[Dict]) -> Tuple[int, int]:
        if valid_actions is None:
            return 0, 0
        if not bool(valid_actions.get("can_buy", False)):
            return 0, 0
        max_buy_water = int(valid_actions.get("max_buy_water", 0))
        max_buy_food = int(valid_actions.get("max_buy_food", 0))
        floor_water = min(90, max_buy_water)
        floor_food = min(90, max_buy_food)
        return max(0, floor_water), max(0, floor_food)

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[Dict] = None,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ) -> Tuple[Dict, float]:
        import numpy as np

        is_day0 = bool(len(obs) > 0 and float(obs[0]) < 1e-6)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(next(self.parameters()).device)
            output = self.forward_step(obs_tensor, valid_actions, update_internal_hidden=True)

            if not deterministic and np.random.random() < epsilon:
                if valid_actions is not None and "valid_moves" in valid_actions:
                    valid_moves = valid_actions["valid_moves"]
                    move_global_id = int(np.random.choice(valid_moves))
                else:
                    move_global_id = 0

                buy_water = 0
                buy_food = 0
                can_buy = (
                    bool(valid_actions.get("can_buy", False))
                    if valid_actions is not None
                    else False
                )
                if can_buy and is_day0:
                    max_buy_water = int(valid_actions.get("max_buy_water", 0))
                    max_buy_food = int(valid_actions.get("max_buy_food", 0))
                    min_buy_water, min_buy_food = self._day0_purchase_floor(valid_actions)
                    if max_buy_water > 0:
                        buy_water = int(np.random.randint(min_buy_water, max_buy_water + 1))
                    if max_buy_food > 0:
                        buy_food = int(np.random.randint(min_buy_food, max_buy_food + 1))

                mine_action = False
                if valid_actions is not None and bool(valid_actions.get("can_mine", False)):
                    mine_action = bool(np.random.random() < 0.3)

                action = {
                    "move": move_global_id,
                    "mine": mine_action,
                    "buy_water": buy_water,
                    "buy_food": buy_food,
                }
            elif deterministic:
                dist = output["action_dist"]
                move_global_id = dist["move_probs"].argmax(dim=-1).item()
                buy_water = int(dist["buy_water_mean"].item())
                buy_food = int(dist["buy_food_mean"].item())
                if is_day0:
                    floor_water, floor_food = self._day0_purchase_floor(valid_actions)
                    buy_water = max(buy_water, floor_water)
                    buy_food = max(buy_food, floor_food)
                action = {
                    "move": move_global_id,
                    "mine": dist["mine_probs"][:, 1].item() > 0.5,
                    "buy_water": buy_water,
                    "buy_food": buy_food,
                }
            else:
                dist = output["action_dist"]

                move_dist = torch.distributions.Categorical(dist["move_probs"])
                move_global_id = move_dist.sample().item()

                mine_dist = torch.distributions.Categorical(dist["mine_probs"])
                mine_action = mine_dist.sample().item() == 1

                max_buy_water = (
                    int(valid_actions.get("max_buy_water", 100))
                    if valid_actions is not None
                    else 100
                )
                max_buy_food = (
                    int(valid_actions.get("max_buy_food", 100))
                    if valid_actions is not None
                    else 100
                )
                can_buy = (
                    bool(valid_actions.get("can_buy", True)) if valid_actions is not None else True
                )

                if can_buy:
                    water_dist = torch.distributions.Normal(
                        dist["buy_water_mean"], dist["buy_water_std"]
                    )
                    buy_water = int(torch.clamp(water_dist.sample(), 0, max_buy_water).item())

                    food_dist = torch.distributions.Normal(
                        dist["buy_food_mean"], dist["buy_food_std"]
                    )
                    buy_food = int(torch.clamp(food_dist.sample(), 0, max_buy_food).item())
                else:
                    buy_water = 0
                    buy_food = 0

                action = {
                    "move": move_global_id,
                    "mine": mine_action,
                    "buy_water": buy_water,
                    "buy_food": buy_food,
                }

                # 第0天设置采购下限，避免低资源早死局部最优
                can_buy = (
                    bool(valid_actions.get("can_buy", False))
                    if valid_actions is not None
                    else False
                )
                if can_buy and is_day0:
                    floor_water, floor_food = self._day0_purchase_floor(valid_actions)
                    action["buy_water"] = max(int(action["buy_water"]), floor_water)
                    action["buy_food"] = max(int(action["buy_food"]), floor_food)

            value = output["value"].item()

        return action, value

    def reset_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """重置LSTM隐藏状态"""
        if device is None:
            device = next(self.parameters()).device
        layers = int(self.config.LSTM_LAYERS)
        hidden_dim = int(self.lstm_hidden_dim)
        h0 = torch.zeros(layers, batch_size, hidden_dim, device=device)
        c0 = torch.zeros(layers, batch_size, hidden_dim, device=device)
        self.hidden_state = (h0, c0)

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
    """创建智能体"""
    obs_dim = 19  # 与environment一致: 6状态+3天气+4地点+6信念
    agent = HybridRNNAgent(obs_dim=obs_dim, num_locations=env.config.NUM_NODES, config=config)
    agent.to(device)
    return agent
