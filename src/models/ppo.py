"""
PPO (Proximal Policy Optimization) 算法实现
支持RNN序列训练和GAE优势估计
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.env.config import RLConfig
from src.models.agent import HybridRNNAgent


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
        self.valid_actions_list: List[
            Optional[Dict]
        ] = []  # BUG修复1: 存储valid_actions

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
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.hidden_states.append(hidden)
        self.valid_actions_list.append(valid_actions)  # BUG修复1: 存储valid_actions

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
        returns = []
        advantages = []

        gae = 0
        next_value = last_value

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            # TD误差
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )

            # GAE
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)

            # 回报 = 优势 + 价值
            returns.insert(0, gae + self.values[t])

        return returns, advantages

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
        agent: HybridRNNAgent,
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
                {"params": agent.critic.parameters(), "lr": self.config.LR_CRITIC},
                {
                    "params": agent.state_encoder.parameters(),
                    "lr": self.config.LR_ACTOR,
                },
                {
                    "params": agent.belief_encoder.parameters(),
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
        self, rollout_buffer: RolloutBuffer, last_value: float
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
        old_log_probs_tensor = torch.FloatTensor(rollout_buffer.log_probs).to(
            self.device
        )
        values_tensor = torch.FloatTensor(rollout_buffer.values).to(
            self.device
        )  # BUG修复6: 用于价值裁剪

        # 归一化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # 准备动作张量
        actions = {
            "move": torch.LongTensor([a["move"] for a in rollout_buffer.actions]).to(
                self.device
            ),
            "mine": torch.FloatTensor([a["mine"] for a in rollout_buffer.actions]).to(
                self.device
            ),
            "buy_water": torch.FloatTensor(
                [a["buy_water"] for a in rollout_buffer.actions]
            ).to(self.device),
            "buy_food": torch.FloatTensor(
                [a["buy_food"] for a in rollout_buffer.actions]
            ).to(self.device),
        }

        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_fraction = 0
        effective_updates = 0
        target_kl_hit = False
        max_ratio_seen = 1.0
        min_ratio_seen = 1.0

        num_updates = self.config.EPOCHS_PER_UPDATE

        for epoch in range(num_updates):
            # BUG修复1 & 6: 使用存储的valid_actions，并添加价值裁剪

            # 逐个处理（保证valid_actions正确性）
            batch_new_log_probs = []
            batch_entropy = []
            batch_new_values = []

            for i in range(len(obs_tensor)):
                obs_i = obs_tensor[i : i + 1]
                action_i = {k: v[i : i + 1] for k, v in actions.items()}
                valid_i = rollout_buffer.valid_actions_list[i]

                # 编码观测
                state_i = self.agent.encode_observation(obs_i)

                # 评估动作（传入valid_actions）
                new_log_prob_i, entropy_i = self.agent.actor.evaluate_actions(
                    state_i, action_i, valid_i
                )
                new_value_i = self.agent.critic(state_i).squeeze(-1)

                batch_new_log_probs.append(new_log_prob_i)
                batch_entropy.append(entropy_i)
                batch_new_values.append(new_value_i)

            new_log_probs = torch.cat(batch_new_log_probs)
            entropy = torch.cat(batch_entropy)
            new_values = torch.cat(batch_new_values)
            old_values = values_tensor  # 用于价值裁剪

            # 策略损失（PPO裁剪）
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            max_ratio_seen = max(max_ratio_seen, float(ratio.max().item()))
            min_ratio_seen = min(min_ratio_seen, float(ratio.min().item()))

            surr1 = ratio * advantages_tensor
            surr2 = (
                torch.clamp(ratio, 1 - self.config.CLIP_EPS, 1 + self.config.CLIP_EPS)
                * advantages_tensor
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # BUG修复6: 价值裁剪
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values, -self.config.CLIP_EPS, self.config.CLIP_EPS
            )
            value_loss1 = (new_values - returns_tensor).pow(2)
            value_loss2 = (value_pred_clipped - returns_tensor).pow(2)
            value_loss = torch.max(value_loss1, value_loss2).mean()

            # 熵奖励
            entropy_loss = -entropy.mean()

            # 总损失
            loss = (
                policy_loss + 0.5 * value_loss + self.config.ENTROPY_COEF * entropy_loss
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.MAX_GRAD_NORM)

            self.optimizer.step()

            # 统计
            with torch.no_grad():
                approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                clip_fraction = (
                    ((ratio - 1).abs() > self.config.CLIP_EPS).float().mean().item()
                )

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
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
            action, value = self.agent.select_action(
                obs, valid_actions, epsilon=epsilon
            )

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

            # BUG修复: 直接使用全局节点ID，不要转换为局部索引
            # 因为evaluate_actions期望的是全局ID来索引全局概率分布
            action_idx = action  # 直接使用，不转换
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                states = self.agent.encode_observation(obs_tensor)
                new_log_probs, _ = self.agent.actor.evaluate_actions(
                    states,
                    {
                        "move": torch.LongTensor([action_idx["move"]]).to(self.device),
                        "mine": torch.FloatTensor([action_idx["mine"]]).to(self.device),
                        "buy_water": torch.FloatTensor([action_idx["buy_water"]]).to(
                            self.device
                        ),
                        "buy_food": torch.FloatTensor([action_idx["buy_food"]]).to(
                            self.device
                        ),
                    },
                    valid_actions,
                )
                log_prob = new_log_probs.item()

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
            last_value = self.agent.critic(
                self.agent.encode_observation(obs_tensor)
            ).item()

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

        return buffer, last_value, episode_info


class CurriculumScheduler:
    """课程学习调度器"""

    def __init__(self, config: RLConfig):
        self.stages = config.CURRICULUM_STAGES
        self.current_stage = 0
        self.episode_count = 0

    def get_stage(self) -> Dict:
        """获取当前阶段的配置"""
        return self.stages[min(self.current_stage, len(self.stages) - 1)]

    def update(self, performance: float):
        """根据性能更新阶段"""
        self.episode_count += 1

        # 简单策略：每N个回合升级
        episodes_per_stage = 500
        new_stage = min(self.episode_count // episodes_per_stage, len(self.stages) - 1)

        if new_stage > self.current_stage:
            print(f"课程学习升级：阶段 {self.current_stage} -> {new_stage}")
            self.current_stage = new_stage

    def apply_to_env(self, env):
        """将当前阶段配置应用到环境"""
        stage = self.get_stage()
        # 根据阶段调整环境参数
        if "weather_known" in stage and stage["weather_known"]:
            # 阶段1：全知天气
            pass
        if "mode" in stage:
            env.weather_mode = stage["mode"]
            env.weather_mode = stage["mode"]
