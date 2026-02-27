"""
PPO (Proximal Policy Optimization) 算法实现
支持RNN序列训练和GAE优势估计
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.env.config import RLConfig
from src.models.agent import Agent
from src.utils.training_utils import clip_gradients, compute_gae, dynamic_alpha


@dataclass
class Transition:
    """存储单步转移"""

    obs: np.ndarray
    action: Dict[str, Any]
    reward: float
    next_obs: np.ndarray
    done: bool
    value: float
    log_prob: float
    hidden_state: Optional[Tuple[np.ndarray, np.ndarray]]


class RolloutBuffer:
    """
    回放缓冲区，存储一个回合的数据
    支持序列数据（用于RNN训练）
    """

    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[Dict] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.hidden_states: List[Optional[Tuple]] = []
        self.valid_actions_list: List[Optional[Dict]] = []  # 存储valid_actions
        self.episode_reached: bool = False
        self.episode_final_money_ratio: float = 0.0

    def add(
        self,
        obs,
        action,
        reward,
        value,
        log_prob,
        done,
        hidden=None,
        valid_actions=None,
    ):
        """添加一个时间步的数据"""
        self.observations.append(np.array(obs, dtype=np.float32, copy=True))
        self.actions.append(deepcopy(action))
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.hidden_states.append(hidden)
        self.valid_actions_list.append(valid_actions)  # 存储valid_actions

    def clear(self):
        """清空缓冲区"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.hidden_states.clear()
        self.valid_actions_list.clear()
        self.episode_reached = False
        self.episode_final_money_ratio = 0.0

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Tuple[List, List]:
        """
        使用GAE计算回报和优势

        Args:
            last_value: 最后状态的价值估计
            gamma: 折扣因子
            gae_lambda: GAE参数

        Returns:
            returns: 折扣回报
            advantages: 优势估计
        """
        return compute_gae(
            rewards=self.rewards,
            values=self.values,
            dones=self.dones,
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def get_batch(self) -> Dict:
        """获取批量数据"""
        return {
            "observations": np.array(self.observations),
            "actions": self.actions,
            "rewards": np.array(self.rewards),
            "values": np.array(self.values),
            "log_probs": np.array(self.log_probs),
            "dones": np.array(self.dones),
        }


class PPOTrainer:
    """PPO训练器"""

    def __init__(
        self,
        agent: Agent,
        config: Optional[RLConfig] = None,
        device: str = "cpu",
    ):
        """
        Args:
            agent: 智能体
            config: RL配置
            device: 计算设备
        """
        self.agent = agent
        self.config = config or RLConfig()
        self.device = torch.device(device)

        # 优化器
        self.optimizer = optim.Adam(
            [
                {"params": agent.actor.parameters(), "lr": self.config.LR_ACTOR},
                {
                    "params": list(agent.survival_critic.parameters())
                    + list(agent.fund_critic.parameters()),
                    "lr": self.config.LR_CRITIC,
                },
                {
                    "params": list(agent.resource_encoder.parameters())
                    + list(agent.graph_encoder.parameters())
                    + list(agent.fusion.parameters()),
                    "lr": self.config.LR_ACTOR,
                },
            ]
        )

        # 训练统计
        self.train_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

    def update(
        self,
        rollout_buffer: RolloutBuffer,
        last_value: float,
        alpha_override: Optional[float] = None,
        stage_name: str = "stage3",
    ) -> Dict[str, float]:
        """
        使用收集的数据更新策略

        Args:
            rollout_buffer: 回放缓冲区
            last_value: 最后状态价值

        Returns:
            训练统计
        """
        # 计算回报和优势
        returns, advantages = rollout_buffer.compute_returns_and_advantages(
            last_value, gamma=self.config.GAMMA, gae_lambda=self.config.GAE_LAMBDA
        )

        # 转换为张量
        obs_array = np.asarray(rollout_buffer.observations, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs_array).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(rollout_buffer.log_probs).to(self.device)
        values_tensor = torch.FloatTensor(rollout_buffer.values).to(self.device)  # 用于价值裁剪

        # 归一化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # 准备动作张量
        actions = {
            "move": torch.LongTensor([a["move"] for a in rollout_buffer.actions]).to(self.device),
            "mine": torch.FloatTensor(
                [float(bool(a.get("mine", False))) for a in rollout_buffer.actions]
            ).to(self.device),
            "mine_intensity": torch.FloatTensor(
                [
                    float(a.get("mine_intensity", 1.0 if a.get("mine", False) else 0.0))
                    for a in rollout_buffer.actions
                ]
            ).to(self.device),
            "buy_water": torch.FloatTensor([a["buy_water"] for a in rollout_buffer.actions]).to(
                self.device
            ),
            "buy_food": torch.FloatTensor([a["buy_food"] for a in rollout_buffer.actions]).to(
                self.device
            ),
        }

        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_survival_value_loss = 0
        total_fund_value_loss = 0
        total_alpha_mean = 0
        total_entropy = 0
        total_kl = 0
        total_clip_fraction = 0
        effective_updates = 0
        target_kl_hit = False
        max_ratio_seen = 1.0
        min_ratio_seen = 1.0

        num_updates = self.config.EPOCHS_PER_UPDATE

        for _epoch in range(num_updates):
            # 使用存储的valid_actions，并添加价值裁剪

            # 逐个处理（保证valid_actions正确性）
            batch_new_log_probs = []
            batch_entropy = []
            batch_new_survival_values = []
            batch_new_fund_values = []
            batch_new_values = []

            for i in range(len(obs_tensor)):
                obs_i = obs_tensor[i : i + 1]
                action_i = {k: v[i : i + 1] for k, v in actions.items()}
                valid_i = rollout_buffer.valid_actions_list[i]

                # 编码观测
                state_i, node_embeddings_i, day_norm_i = self.agent.encode_observation_full(obs_i)

                # 评估动作（传入valid_actions）
                new_log_prob_i, entropy_i = self.agent.actor.evaluate_actions(
                    state_i,
                    node_embeddings_i,
                    day_norm_i,
                    action_i,
                    valid_i,
                )
                new_survival = self.agent.survival_critic(state_i).squeeze(-1)
                new_fund = self.agent.fund_critic(state_i).squeeze(-1)
                new_value_i = 0.5 * (new_survival + new_fund)

                batch_new_log_probs.append(new_log_prob_i)
                batch_entropy.append(entropy_i)
                batch_new_survival_values.append(new_survival)
                batch_new_fund_values.append(new_fund)
                batch_new_values.append(new_value_i)

            new_log_probs = torch.cat(batch_new_log_probs)
            entropy = torch.cat(batch_entropy)
            new_survival_values = torch.cat(batch_new_survival_values)
            new_fund_values = torch.cat(batch_new_fund_values)
            new_values = torch.cat(batch_new_values)
            old_values = values_tensor  # 用于价值裁剪

            # 策略损失（PPO裁剪）
            raw_log_ratio = new_log_probs - old_log_probs_tensor
            if not torch.isfinite(raw_log_ratio).all():
                raise ValueError("检测到非有限log_ratio，请检查hidden state与valid_actions一致性")
            clipped_log_ratio = torch.clamp(raw_log_ratio, min=-20.0, max=20.0)
            ratio = torch.exp(clipped_log_ratio)
            if not torch.isfinite(ratio).all():
                raise ValueError("检测到非有限ratio，请检查log_prob数值范围")
            max_ratio_seen = max(max_ratio_seen, float(ratio.max().item()))
            min_ratio_seen = min(min_ratio_seen, float(ratio.min().item()))

            surr1 = ratio * advantages_tensor
            surr2 = (
                torch.clamp(ratio, 1 - self.config.CLIP_EPS, 1 + self.config.CLIP_EPS)
                * advantages_tensor
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # 动态权重 alpha（基于观测中的资源比例与终点距离）
            water_ratio = torch.clamp(obs_tensor[:, 2], 0.0, 1.0)
            food_ratio = torch.clamp(obs_tensor[:, 3], 0.0, 1.0)
            dist_to_end_ratio = torch.clamp(obs_tensor[:, 5], 0.0, 1.0)
            if alpha_override is not None:
                alpha = torch.full_like(water_ratio, float(alpha_override))
            else:
                alpha = dynamic_alpha(water_ratio, food_ratio, dist_to_end_ratio)

            # Survival Critic 损失（回合成败监督）
            survival_target = torch.full_like(
                new_survival_values,
                1.0 if rollout_buffer.episode_reached else 0.0,
            )
            survival_value_loss_element = nn.functional.binary_cross_entropy(
                new_survival_values,
                survival_target,
                reduction="none",
            )

            # Fund Critic 损失（最终资金监督，按初始资金归一化）
            fund_target = torch.full_like(
                new_fund_values,
                float(rollout_buffer.episode_final_money_ratio),
            )
            fund_value_loss_element = (new_fund_values - fund_target).pow(2)

            weighted_value_loss_element = (
                alpha * survival_value_loss_element + (1.0 - alpha) * fund_value_loss_element
            )
            value_loss = weighted_value_loss_element.mean()
            survival_value_loss = survival_value_loss_element.mean()
            fund_value_loss = fund_value_loss_element.mean()

            # 熵奖励
            entropy_loss = -entropy.mean()

            # 总损失
            loss = policy_loss + 0.5 * value_loss + self.config.ENTROPY_COEF * entropy_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            clip_gradients(self.agent, self.config.MAX_GRAD_NORM)

            self.optimizer.step()
            self.agent.update_targets(tau=0.995)

            # 统计
            with torch.no_grad():
                approx_kl = ((ratio - 1) - clipped_log_ratio).mean().item()
                clip_fraction = ((ratio - 1).abs() > self.config.CLIP_EPS).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_survival_value_loss += survival_value_loss.item()
            total_fund_value_loss += fund_value_loss.item()
            total_alpha_mean += alpha.mean().item()
            total_entropy += entropy.mean().item()
            total_kl += approx_kl
            total_clip_fraction += clip_fraction
            effective_updates += 1

            if approx_kl > self.config.TARGET_KL:
                target_kl_hit = True
                break

        # 返回平均统计
        n = max(1, effective_updates)

        y_pred = values_tensor.detach().cpu().numpy()
        y_true = returns_tensor.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))

        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "survival_value_loss": total_survival_value_loss / n,
            "fund_value_loss": total_fund_value_loss / n,
            "alpha_mean": total_alpha_mean / n,
            "alpha_source": 0.0 if alpha_override is None else 1.0,
            "entropy": total_entropy / n,
            "approx_kl": total_kl / n,
            "clip_fraction": total_clip_fraction / n,
            "effective_updates": float(effective_updates),
            "target_kl_hit": float(1.0 if target_kl_hit else 0.0),
            "max_ratio": max_ratio_seen,
            "min_ratio": min_ratio_seen,
            "explained_variance": explained_var,
        }

    def collect_rollout(
        self, env, max_steps: int = 1000, epsilon: float = 0.0
    ) -> Tuple[RolloutBuffer, float, Dict]:
        """
        收集一个回合的数据

        Args:
            env: 环境
            max_steps: 最大步数
            epsilon: ε-贪婪探索率

        Returns:
            rollout_buffer: 回放缓冲区
            episode_return: 回合回报
            episode_info: 回合信息
        """
        buffer = RolloutBuffer()

        obs, info = env.reset()
        self.agent.reset_hidden(device=self.device)

        episode_return = 0
        step = 0
        move_count = 0
        stay_count = 0
        mine_count = 0
        buy_count = 0
        total_buy_water = 0
        total_buy_food = 0

        while step < max_steps:
            # 获取有效动作
            valid_actions = env.get_valid_actions()

            # 选择动作（带ε-贪婪探索）
            action, value = self.agent.select_action(obs, valid_actions, epsilon=epsilon)

            can_mine = bool(valid_actions.get("can_mine", False))
            if can_mine and epsilon > 0.0 and np.random.random() < min(0.6, epsilon + 0.2):
                action["move"] = int(env.state.position)
                action["mine"] = True
                action["mine_intensity"] = 1.0
                action["buy_water"] = 0
                action["buy_food"] = 0

            # 执行动作
            next_obs, reward, done, truncated, info = env.step(action)
            done_flag = done or truncated

            last_action = info.get("last_action") or action
            move_from = last_action.get("move_from")
            move_to = last_action.get("move_to")
            if move_from is not None and move_to is not None:
                if move_from == move_to:
                    stay_count += 1
                else:
                    move_count += 1
            elif action.get("move") == info.get("position"):
                stay_count += 1
            else:
                move_count += 1

            if bool(last_action.get("mine", False)):
                mine_count += 1

            bought_w = int(last_action.get("buy_water", 0))
            bought_f = int(last_action.get("buy_food", 0))
            total_buy_water += bought_w
            total_buy_food += bought_f
            if bought_w > 0 or bought_f > 0:
                buy_count += 1

            # 直接使用全局节点ID，不要转换为局部索引
            # 因为evaluate_actions期望的是全局ID来索引全局概率分布
            action_idx = action  # 直接使用，不转换
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                state_i, node_embeddings_i, day_norm_i = self.agent.encode_observation_full(
                    obs_tensor
                )
                new_log_probs, _ = self.agent.actor.evaluate_actions(
                    state_i,
                    node_embeddings_i,
                    day_norm_i,
                    {
                        "move": torch.LongTensor([action_idx["move"]]).to(self.device),
                        "mine": torch.FloatTensor([float(bool(action_idx.get("mine", False)))]).to(
                            self.device
                        ),
                        "mine_intensity": torch.FloatTensor(
                            [
                                float(
                                    action_idx.get(
                                        "mine_intensity",
                                        1.0 if action_idx.get("mine", False) else 0.0,
                                    )
                                )
                            ]
                        ).to(self.device),
                        "buy_water": torch.FloatTensor([action_idx["buy_water"]]).to(self.device),
                        "buy_food": torch.FloatTensor([action_idx["buy_food"]]).to(self.device),
                    },
                    valid_actions,
                )
                log_prob = new_log_probs.item()
                if not np.isfinite(log_prob):
                    raise ValueError(
                        "collect_rollout得到非有限log_prob，请检查动作掩码与hidden state"
                    )

            # 存储（使用索引版本，并传入valid_actions用于后续训练）
            buffer.add(
                obs,
                action_idx,
                reward,
                value,
                log_prob,
                done_flag,
                valid_actions=valid_actions,
            )

            episode_return += reward
            obs = next_obs
            step += 1

            if done_flag:
                break

        # 获取最后状态价值（用于GAE）
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            state_i, _, _ = self.agent.encode_observation_full(obs_tensor)
            last_value = 0.5 * (
                self.agent.survival_critic(state_i).item() + self.agent.fund_critic(state_i).item()
            )

        episode_info = {
            "return": episode_return,
            "length": step,
            "reached": info.get("reached", False),
            "final_money": info.get("money", 0),
            "final_day": info.get("day", 0),
            "move_count": move_count,
            "stay_count": stay_count,
            "mine_count": mine_count,
            "buy_count": buy_count,
            "total_buy_water": total_buy_water,
            "total_buy_food": total_buy_food,
        }

        buffer.episode_reached = bool(episode_info["reached"])
        buffer.episode_final_money_ratio = float(episode_info["final_money"]) / float(
            max(1.0, float(env.config.INIT_MONEY))
        )

        return buffer, last_value, episode_info
