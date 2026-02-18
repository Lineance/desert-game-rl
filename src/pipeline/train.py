#!/usr/bin/env python
"""训练脚本：仅使用 ppo.py 中的 PPOTrainer。"""

import argparse
from typing import Optional

import torch
from src.env.config import CHECKPOINTS_DIR, RLConfig
from src.env.environment import make_env
from src.models.agent import create_agent
from src.models.ppo import PPOTrainer


def train(level: int = 3, num_episodes: int = 2000, device: Optional[str] = None) -> None:
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
    success_count = 0
    stats = None

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

        if episode_info["reached"]:
            success_count += 1

        if episode_info["return"] > best_return:
            best_return = episode_info["return"]
            agent.save(str(save_dir / f"level{level}_best.pt"))

        if episode % 100 == 0:
            recent_success = success_count / min(100, episode + 1)
            print(f"\nEpisode {episode}")
            print(f"  最近成功率: {recent_success * 100:.1f}%")
            print(f"  最佳回报: {best_return:.1f}")
            print(
                f"  本回合: 奖励={episode_info['return']:.1f}, "
                f"到达={episode_info['reached']}, 天数={episode_info['final_day']}"
            )
            if stats is not None:
                print(
                    "  损失: "
                    f"P={stats['policy_loss']:.4f}, "
                    f"V={stats['value_loss']:.4f}, "
                    f"H={stats['entropy']:.4f}, "
                    f"KL={stats['approx_kl']:.4f}"
                )
            success_count = 0

        if episode % 500 == 0 and episode > 0:
            agent.save(str(save_dir / f"level{level}_episode{episode}.pt"))

    agent.save(str(save_dir / f"level{level}_final.pt"))
    print(f"\n训练完成！最佳奖励: {best_return:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(level=args.level, num_episodes=args.episodes, device=args.device)
