"""Oracle warmup utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.env.environment import make_env
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


def _rollout_student_policy(
    agent: HybridRNNAgent,
    level: int,
    weather_mode: str,
    seed: int,
) -> Dict[str, float]:
    eval_env = make_env(level=level, weather_mode=weather_mode, seed=None)
    obs, info = eval_env.reset(seed=seed)

    steps = 0
    mine_days = 0
    buy_water = 0
    buy_food = 0
    done = False
    truncated = False

    while not done and not truncated and steps < 200:
        valid_actions = eval_env.get_valid_actions()
        action, _ = agent.select_action(obs, valid_actions, deterministic=True)
        obs, _, done, truncated, info = eval_env.step(action)
        last_action = info.get("last_action")
        if isinstance(last_action, dict):
            if bool(last_action.get("mine", False)):
                mine_days += 1
            buy_water += int(last_action.get("buy_water", 0))
            buy_food += int(last_action.get("buy_food", 0))
        steps += 1

    return {
        "steps": float(steps),
        "mine_days": float(mine_days),
        "buy_water": float(buy_water),
        "buy_food": float(buy_food),
        "success": float(bool(info.get("reached", False))),
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
    start_episode: int = 0,
    metrics: Optional[List[Dict[str, float]]] = None,
    on_episode_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    if episodes <= 0:
        return {
            "episodes": 0.0,
            "solved": 0.0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_mine_per_episode": 0.0,
            "avg_buy_water": 0.0,
            "avg_buy_food": 0.0,
            "avg_value_loss": 0.0,
            "final_value_loss": 0.0,
            "value_loss_trend": 0.0,
            "match_rate": 0.0,
        }
    if level not in (3, 35, 4):
        print("Oracle预热仅支持level 3/35/4，已跳过。")
        return {
            "episodes": float(episodes),
            "solved": 0.0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_mine_per_episode": 0.0,
            "avg_buy_water": 0.0,
            "avg_buy_food": 0.0,
            "avg_value_loss": 0.0,
            "final_value_loss": 0.0,
            "value_loss_trend": 0.0,
            "match_rate": 0.0,
        }

    agent.train()
    total_loss = 0.0
    total_steps = 0
    solved = 0
    success = 0
    total_mine_days = 0
    total_buy_water = 0
    total_buy_food = 0
    matched_actions = 0
    total_actions = 0
    value_losses: List[float] = []
    student_total_steps = 0
    student_total_mine_days = 0
    student_total_buy_water = 0
    student_total_buy_food = 0
    student_success = 0
    student_eval_episodes = 0

    weather_mode = str(getattr(env, "weather_mode", "balanced"))

    for ep in range(start_episode, start_episode + episodes):
        should_eval_student = (log_interval > 0) and (ep % log_interval == 0)
        episode_record: Dict[str, float] = {
            "episode": float(ep),
            "solved": 0.0,
            "success": 0.0,
            "steps": 0.0,
            "mine_days": 0.0,
            "buy_water": 0.0,
            "buy_food": 0.0,
            "match_rate": 0.0,
            "student_steps": float("nan"),
            "student_mine_days": float("nan"),
            "student_buy_water": float("nan"),
            "student_buy_food": float("nan"),
            "student_success": float("nan"),
        }
        obs, _ = env.reset(seed=ep)
        if env.state is None:
            if on_episode_end is not None:
                on_episode_end(ep, dict(episode_record))
            continue

        if should_eval_student:
            try:
                student_eval = _rollout_student_policy(agent, level, weather_mode, ep)
                student_total_steps += int(student_eval["steps"])
                student_total_mine_days += int(student_eval["mine_days"])
                student_total_buy_water += int(student_eval["buy_water"])
                student_total_buy_food += int(student_eval["buy_food"])
                student_success += int(student_eval["success"])
                student_eval_episodes += 1
                episode_record["student_steps"] = float(student_eval["steps"])
                episode_record["student_mine_days"] = float(student_eval["mine_days"])
                episode_record["student_buy_water"] = float(student_eval["buy_water"])
                episode_record["student_buy_food"] = float(student_eval["buy_food"])
                episode_record["student_success"] = float(student_eval["success"])
            except Exception:
                episode_record["student_steps"] = float("nan")
                episode_record["student_mine_days"] = float("nan")
                episode_record["student_buy_water"] = float("nan")
                episode_record["student_buy_food"] = float("nan")
                episode_record["student_success"] = float("nan")

        weather_seq = list(env.state.weather_future)
        oracle = solve_theoretical_plan(level, weather_seq, time_limit=time_limit)
        plan = oracle.get("plan", [])
        if not plan:
            if ep % log_interval == 0:
                print(f"Oracle预热: episode={ep}, plan为空，跳过")
            if on_episode_end is not None:
                on_episode_end(ep, dict(episode_record))
            continue

        solved += 1
        episode_record["solved"] = 1.0
        episode_steps = 0
        episode_mine_days = 0
        episode_buy_water = 0
        episode_buy_food = 0
        episode_matched = 0
        episode_total_actions = 0
        episode_success = 0
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []

        for oracle_action in plan:
            valid_actions = env.get_valid_actions()
            action = _sanitize_oracle_action(oracle_action, env, valid_actions)

            pred_action, _ = agent.select_action(obs, valid_actions, deterministic=True)
            move_match = int(pred_action.get("move", env.state.position)) == int(action["move"])
            if action["buy_water"] == 0:
                water_match = int(pred_action.get("buy_water", 0)) == 0
            else:
                water_match = (
                    abs(int(pred_action.get("buy_water", 0)) - int(action["buy_water"]))
                    / max(1.0, float(action["buy_water"]))
                    <= 0.1
                )
            if action["buy_food"] == 0:
                food_match = int(pred_action.get("buy_food", 0)) == 0
            else:
                food_match = (
                    abs(int(pred_action.get("buy_food", 0)) - int(action["buy_food"]))
                    / max(1.0, float(action["buy_food"]))
                    <= 0.1
                )
            if move_match and water_match and food_match:
                matched_actions += 1
                episode_matched += 1
            total_actions += 1
            episode_total_actions += 1

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
            last_action = env.state.last_action if env.state is not None else None
            if isinstance(last_action, dict):
                if bool(last_action.get("mine", False)):
                    total_mine_days += 1
                    episode_mine_days += 1
                total_buy_water += int(last_action.get("buy_water", 0))
                total_buy_food += int(last_action.get("buy_food", 0))
                episode_buy_water += int(last_action.get("buy_water", 0))
                episode_buy_food += int(last_action.get("buy_food", 0))
            rewards.append(float(reward))
            obs = next_obs
            episode_steps += 1
            if done or truncated:
                break

        if episode_steps == 0 or not log_probs or not values:
            if on_episode_end is not None:
                episode_record["steps"] = float(episode_steps)
                on_episode_end(ep, dict(episode_record))
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
        value_losses.append(float(value_loss.item()))

        trainer.optimizer.zero_grad()
        episode_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), trainer.config.MAX_GRAD_NORM)
        trainer.optimizer.step()

        total_loss += float(episode_loss.item())
        total_steps += episode_steps

        if metrics is not None:
            if env.state is not None and bool(env.state.reached):
                episode_success = 1
            episode_record["steps"] = float(episode_steps)
            episode_record["mine_days"] = float(episode_mine_days)
            episode_record["buy_water"] = float(episode_buy_water)
            episode_record["buy_food"] = float(episode_buy_food)
            episode_record["success"] = float(episode_success)
            episode_record["match_rate"] = float(episode_matched / max(1, episode_total_actions))
            episode_record["value_loss"] = float(value_loss.item())
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
                    "mine_days": float(episode_mine_days),
                    "buy_water": float(episode_buy_water),
                    "buy_food": float(episode_buy_food),
                    "success": float(episode_success),
                    "match_rate": float(episode_matched / max(1, episode_total_actions)),
                    "student_steps": float(episode_record.get("student_steps", 0.0)),
                    "student_mine_days": float(episode_record.get("student_mine_days", 0.0)),
                    "student_buy_water": float(episode_record.get("student_buy_water", 0.0)),
                    "student_buy_food": float(episode_record.get("student_buy_food", 0.0)),
                    "student_success": float(episode_record.get("student_success", 0.0)),
                }
            )
        else:
            if env.state is not None and bool(env.state.reached):
                episode_success = 1
            episode_record["steps"] = float(episode_steps)
            episode_record["mine_days"] = float(episode_mine_days)
            episode_record["buy_water"] = float(episode_buy_water)
            episode_record["buy_food"] = float(episode_buy_food)
            episode_record["success"] = float(episode_success)
            episode_record["match_rate"] = float(episode_matched / max(1, episode_total_actions))
            episode_record["value_loss"] = float(value_loss.item())

        if log_interval > 0 and ep % log_interval == 0:
            avg_loss = total_loss / max(1, solved)
            avg_steps = total_steps / max(1, solved)
            avg_mine = total_mine_days / max(1, solved)
            match_rate = matched_actions / max(1, total_actions)
            student_avg_steps = student_total_steps / max(1, student_eval_episodes)
            student_avg_mine = student_total_mine_days / max(1, student_eval_episodes)
            student_success_rate = student_success / max(1, student_eval_episodes)
            print(
                f"Oracle预热进度: episode={ep}, solved={solved}, "
                f"avg_steps={avg_steps:.1f}, avg_mine={avg_mine:.2f}, "
                f"match_rate={match_rate:.2%}, avg_loss={avg_loss:.4f}; "
                f"student(avg_steps={student_avg_steps:.1f}, avg_mine={student_avg_mine:.2f}, "
                f"success={student_success_rate:.2%}, eval_points={student_eval_episodes})"
            )

        if env.state is not None and bool(env.state.reached):
            success += 1

        if on_episode_end is not None:
            on_episode_end(ep, dict(episode_record))

    avg_loss = total_loss / max(1, solved)
    avg_steps = total_steps / max(1, solved)
    avg_mine = total_mine_days / max(1, solved)
    avg_buy_water = total_buy_water / max(1, solved)
    avg_buy_food = total_buy_food / max(1, solved)
    success_rate = success / max(1, solved)
    match_rate = matched_actions / max(1, total_actions)
    avg_value_loss = sum(value_losses) / max(1, len(value_losses))
    final_value_loss = value_losses[-1] if value_losses else 0.0
    value_loss_trend = value_losses[-1] - value_losses[0] if len(value_losses) >= 2 else 0.0
    student_avg_steps = student_total_steps / max(1, student_eval_episodes)
    student_avg_mine = student_total_mine_days / max(1, student_eval_episodes)
    student_avg_buy_water = student_total_buy_water / max(1, student_eval_episodes)
    student_avg_buy_food = student_total_buy_food / max(1, student_eval_episodes)
    student_success_rate = student_success / max(1, student_eval_episodes)
    print(
        f"Oracle预热完成: episodes={episodes}, solved={solved}, "
        f"success_rate={success_rate:.2%}, avg_steps={avg_steps:.1f}, "
        f"avg_mine={avg_mine:.2f}, avg_buy_w={avg_buy_water:.1f}, "
        f"avg_buy_f={avg_buy_food:.1f}, match_rate={match_rate:.2%}, avg_loss={avg_loss:.4f}; "
        f"student(success_rate={student_success_rate:.2%}, avg_steps={student_avg_steps:.1f}, "
        f"avg_mine={student_avg_mine:.2f}, avg_buy_w={student_avg_buy_water:.1f}, "
        f"avg_buy_f={student_avg_buy_food:.1f})"
    )

    return {
        "episodes": float(episodes),
        "solved": float(solved),
        "success_rate": float(success_rate),
        "avg_steps": float(avg_steps),
        "avg_mine_per_episode": float(avg_mine),
        "avg_buy_water": float(avg_buy_water),
        "avg_buy_food": float(avg_buy_food),
        "avg_value_loss": float(avg_value_loss),
        "final_value_loss": float(final_value_loss),
        "value_loss_trend": float(value_loss_trend),
        "match_rate": float(match_rate),
        "student_success_rate": float(student_success_rate),
        "student_avg_steps": float(student_avg_steps),
        "student_avg_mine_per_episode": float(student_avg_mine),
        "student_avg_buy_water": float(student_avg_buy_water),
        "student_avg_buy_food": float(student_avg_buy_food),
    }
