"""Oracle warmup utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.models.agent import HybridRNNAgent
from src.models.ppo import PPOTrainer
from src.pipeline.oracle import solve_theoretical_plan


def _sanitize_oracle_action(action: Dict[str, Any], env, valid_actions: Dict[str, Any]) -> Dict:
    move = int(action.get("move", env.state.position))
    valid_moves = valid_actions.get("valid_moves", []) if valid_actions else []
    if valid_moves and move not in valid_moves:
        move = int(env.state.position)

    can_mine = bool(valid_actions.get("can_mine", False)) if valid_actions else False
    mine = bool(action.get("mine", False)) and can_mine

    if valid_actions and bool(valid_actions.get("can_buy", False)):
        max_w = int(valid_actions.get("max_buy_water", 0))
        max_f = int(valid_actions.get("max_buy_food", 0))
        buy_water = max(0, min(int(action.get("buy_water", 0)), max_w))
        buy_food = max(0, min(int(action.get("buy_food", 0)), max_f))
    else:
        buy_water = 0
        buy_food = 0

    return {
        "move": move,
        "mine": mine,
        "buy_water": buy_water,
        "buy_food": buy_food,
    }


def warmup_with_oracle(
    agent: HybridRNNAgent,
    trainer: PPOTrainer,
    env,
    level: int,
    episodes: int,
    time_limit: int,
    device: str,
    value_weight: float = 0.1,
    log_interval: int = 10,
    metrics: Optional[List[Dict[str, float]]] = None,
) -> None:
    if episodes <= 0:
        return
    if level not in (3, 35, 4):
        print("Oracle预热仅支持level 3/35/4，已跳过。")
        return

    agent.train()
    total_loss = 0.0
    total_steps = 0
    solved = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        if env.state is None:
            continue

        weather_seq = list(env.state.weather_future)
        oracle = solve_theoretical_plan(level, weather_seq, time_limit=time_limit)
        plan = oracle.get("plan", [])
        if not plan:
            if ep % log_interval == 0:
                print(f"Oracle预热: episode={ep}, plan为空，跳过")
            continue

        solved += 1
        episode_steps = 0
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []

        for oracle_action in plan:
            valid_actions = env.get_valid_actions()
            action = _sanitize_oracle_action(oracle_action, env, valid_actions)

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            state = agent.encode_observation(obs_tensor)
            action_tensor = {
                "move": torch.LongTensor([action["move"]]).to(device),
                "mine": torch.FloatTensor([action["mine"]]).to(device),
                "buy_water": torch.FloatTensor([action["buy_water"]]).to(device),
                "buy_food": torch.FloatTensor([action["buy_food"]]).to(device),
            }
            log_prob, _ = agent.actor.evaluate_actions(state, action_tensor, valid_actions)
            value = agent.critic(state).squeeze(-1)
            log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))

            next_obs, reward, done, truncated, _ = env.step(action)
            rewards.append(float(reward))
            obs = next_obs
            episode_steps += 1
            if done or truncated:
                break

        if episode_steps == 0 or not log_probs or not values:
            continue

        returns: List[float] = []
        running_return = 0.0
        for r in reversed(rewards):
            running_return = r + trainer.config.GAMMA * running_return
            returns.insert(0, running_return)

        policy_loss = -torch.stack(log_probs).mean()
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        values_tensor = torch.stack(values)
        value_loss = F.mse_loss(values_tensor, returns_tensor)
        episode_loss = policy_loss + value_weight * value_loss

        trainer.optimizer.zero_grad()
        episode_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), trainer.config.MAX_GRAD_NORM)
        trainer.optimizer.step()

        total_loss += float(episode_loss.item())
        total_steps += episode_steps

        if metrics is not None:
            metrics.append(
                {
                    "episode": float(ep),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "episode_loss": float(episode_loss.item()),
                    "steps": float(episode_steps),
                    "first_value": float(values_tensor[0].item())
                    if len(values_tensor) > 0
                    else float("nan"),
                    "first_return": float(returns_tensor[0].item())
                    if len(returns_tensor) > 0
                    else float("nan"),
                }
            )

        if ep % log_interval == 0:
            avg_loss = total_loss / max(1, solved)
            avg_steps = total_steps / max(1, solved)
            print(
                f"Oracle预热进度: episode={ep}, solved={solved}, "
                f"avg_steps={avg_steps:.1f}, avg_loss={avg_loss:.4f}"
            )

    avg_loss = total_loss / max(1, solved)
    avg_steps = total_steps / max(1, solved)
    print(
        f"Oracle预热完成: episodes={episodes}, solved={solved}, "
        f"avg_steps={avg_steps:.1f}, avg_loss={avg_loss:.4f}"
    )
