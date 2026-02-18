#!/usr/bin/env python

import argparse
from collections import deque
from typing import Optional

import torch

from src.env.config import CHECKPOINTS_DIR, RLConfig
from src.env.environment import make_env
from src.models.agent import create_agent
from src.models.ppo import PPOTrainer


def train(level: int = 3, num_episodes: int = 2000, device: Optional[str] = None, log_interval: int = 100) -> None:
    """训练入口。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = CHECKPOINTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    config = RLConfig()

    env = make_env(level=level, seed=42)
    agent = create_agent(env, config, device=device)
    trainer = PPOTrainer(agent, config, device=device)

    best_return = float("-inf")
    stats = None
    rolling_window = 100
    recent_returns = deque(maxlen=rolling_window)
    recent_success = deque(maxlen=rolling_window)
    recent_lengths = deque(maxlen=rolling_window)
    recent_final_money = deque(maxlen=rolling_window)
    recent_mine = deque(maxlen=rolling_window)
    recent_buy = deque(maxlen=rolling_window)
    recent_move = deque(maxlen=rolling_window)
    recent_stay = deque(maxlen=rolling_window)
    recent_buy_water = deque(maxlen=rolling_window)
    recent_buy_food = deque(maxlen=rolling_window)

    print("=" * 60)
    print("训练开始")
    print("=" * 60)

    max_steps = env.config.NUM_DAYS + 2

    for episode in range(num_episodes):
        epsilon = max(0.01, 0.3 * (1 - episode / num_episodes))

        rollout_buffer, last_value, episode_info = trainer.collect_rollout(
            env,
            max_steps=max_steps,
            epsilon=epsilon,
        )

        if rollout_buffer.rewards:
            stats = trainer.update(rollout_buffer, last_value)

        recent_returns.append(float(episode_info["return"]))
        recent_success.append(1.0 if episode_info["reached"] else 0.0)
        recent_lengths.append(float(episode_info["length"]))
        recent_final_money.append(float(episode_info.get("final_money", 0.0)))
        recent_mine.append(float(episode_info.get("mine_count", 0)))
        recent_buy.append(float(episode_info.get("buy_count", 0)))
        recent_move.append(float(episode_info.get("move_count", 0)))
        recent_stay.append(float(episode_info.get("stay_count", 0)))
        recent_buy_water.append(float(episode_info.get("total_buy_water", 0)))
        recent_buy_food.append(float(episode_info.get("total_buy_food", 0)))

        if episode_info["return"] > best_return:
            best_return = episode_info["return"]
            agent.save(str(save_dir / f"level{level}_best.pt"))

        if episode % log_interval == 0:
            window_n = max(1, len(recent_success))
            rolling_success_rate = sum(recent_success) / window_n
            rolling_return = sum(recent_returns) / window_n
            rolling_length = sum(recent_lengths) / window_n
            rolling_money = sum(recent_final_money) / window_n
            rolling_mine = sum(recent_mine) / window_n
            rolling_buy = sum(recent_buy) / window_n
            rolling_move = sum(recent_move) / window_n
            rolling_stay = sum(recent_stay) / window_n
            rolling_buy_w = sum(recent_buy_water) / window_n
            rolling_buy_f = sum(recent_buy_food) / window_n

            print(f"\nEpisode {episode}")
            print(f"  最近{window_n}局成功率: {rolling_success_rate * 100:.1f}%")
            print(f"  最近{window_n}局: 均回报={rolling_return:.1f}, 均天数={rolling_length:.1f}, 均资金={rolling_money:.1f}")
            print(
                f"  最近{window_n}局行为: 均移动={rolling_move:.1f}, 均停留={rolling_stay:.1f}, "
                f"均挖矿={rolling_mine:.1f}, 均购买={rolling_buy:.1f}, "
                f"均购水={rolling_buy_w:.1f}, 均购食={rolling_buy_f:.1f}"
            )
            print(f"  最佳回报: {best_return:.1f}")
            print(
                f"  本回合: 奖励={episode_info['return']:.1f}, "
                f"到达={episode_info['reached']}, 步数={episode_info['length']}, 天数={episode_info['final_day']}, "
                f"资金={episode_info['final_money']:.1f}"
            )
            if stats is not None:
                print(
                    "  损失: "
                    f"P={stats['policy_loss']:.4f}, "
                    f"V={stats['value_loss']:.4f}, "
                    f"H={stats['entropy']:.4f}, "
                    f"KL={stats['approx_kl']:.4f}, "
                    f"ClipFrac={stats['clip_fraction']:.3f}, "
                    f"ExplVar={stats['explained_variance']:.3f}"
                )
                print(
                    "  更新诊断: "
                    f"effective_updates={int(stats['effective_updates'])}/{config.EPOCHS_PER_UPDATE}, "
                    f"target_kl_hit={bool(stats['target_kl_hit'])}, "
                    f"ratio=[{stats['min_ratio']:.3f}, {stats['max_ratio']:.3f}], "
                    f"epsilon={epsilon:.3f}"
                )

        if episode % 500 == 0 and episode > 0:
            agent.save(str(save_dir / f"level{level}_episode{episode}.pt"))

    agent.save(str(save_dir / f"level{level}_final.pt"))
    print(f"\n训练完成！最佳奖励: {best_return:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    train(level=args.level, num_episodes=args.episodes, device=args.device, log_interval=args.log_interval)
