import math
from typing import List

import pytest
import torch

from src.env.environment import make_env
from src.models.agent import create_agent
from src.models.belief import WeatherBeliefModel
from src.models.ppo import PPOTrainer
from src.pipeline.pretrain import _sanitize_oracle_action, warmup_with_oracle
from src.utils.oracle import solve_theoretical_plan


def _fixed_sequence(num_days: int) -> List[int]:
    base = [0, 0, 1, 1, 2]
    seq = []
    while len(seq) < num_days:
        seq.extend(base)
    return seq[:num_days]


def _set_fixed_weather(env, seq: List[int]) -> None:
    env.state.weather_future = list(seq)
    env.state.weather_today = int(seq[0])
    env.belief_model = WeatherBeliefModel(
        transition_prior=env.config.WEATHER_TRANSITION.copy(),
        learning_rate=0.1,
    )
    env.belief_model.update(env.state.weather_today)


def _collect_oracle_rollout(env, seq: List[int]) -> List[float]:
    oracle = solve_theoretical_plan(3, seq, time_limit=10)
    plan = oracle.get("plan", [])
    if not plan:
        return []

    rewards: List[float] = []
    obs, _ = env.reset(seed=123)
    _set_fixed_weather(env, seq)

    for oracle_action in plan:
        valid_actions = env.get_valid_actions()
        action = _sanitize_oracle_action(oracle_action, env, valid_actions)
        _, reward, done, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if done or truncated:
            break

    return rewards


@pytest.mark.slow
def test_warmup_critic_mc_smoke(monkeypatch):
    env = make_env(level=3, weather_mode="no_sandstorm", seed=0)
    seq = _fixed_sequence(env.config.NUM_DAYS)
    monkeypatch.setattr(env, "_generate_weather_sequence", lambda: list(seq))

    agent = create_agent(env, config=None, device="cpu")
    trainer = PPOTrainer(agent, config=None, device="cpu")

    metrics: List[dict] = []
    warmup_with_oracle(
        agent,
        trainer,
        env,
        level=3,
        episodes=100,
        time_limit=30,
        device="cpu",
        log_interval=1000,
        metrics=metrics,
    )

    assert metrics, "Warmup should produce metrics entries"
    first_loss = metrics[0]["value_loss"]
    last_loss = metrics[-1]["value_loss"]
    assert last_loss < first_loss * 0.5

    rewards = _collect_oracle_rollout(env, seq)
    assert rewards, "Oracle rollout should produce rewards"

    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + trainer.config.GAMMA * running
        returns.insert(0, running)

    obs, _ = env.reset(seed=456)
    _set_fixed_weather(env, seq)
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        value = agent.critic(agent.encode_observation(obs_tensor)).item()

    target = returns[0]
    denom = max(1.0, abs(target))
    assert math.isfinite(value)
    assert abs(value - target) / denom < 0.5


@pytest.mark.random
@pytest.mark.slow
def test_warmup_behavior_cloning_match_rate(monkeypatch):
    env = make_env(level=3, weather_mode="no_sandstorm", seed=0)
    seq = _fixed_sequence(env.config.NUM_DAYS)
    monkeypatch.setattr(env, "_generate_weather_sequence", lambda: list(seq))

    agent = create_agent(env, config=None, device="cpu")
    trainer = PPOTrainer(agent, config=None, device="cpu")

    warmup_with_oracle(
        agent,
        trainer,
        env,
        level=3,
        episodes=100,
        time_limit=10,
        device="cpu",
        log_interval=1000,
    )

    sequences: List[List[int]] = []
    for seed in range(5):
        temp_env = make_env(level=3, weather_mode="no_sandstorm", seed=seed)
        temp_env.reset(seed=seed)
        sequences.append(list(temp_env.state.weather_future))

    matched = 0
    total = 0
    for seq in sequences:
        eval_env = make_env(level=3, weather_mode="no_sandstorm", seed=0)
        eval_env.reset(seed=0)
        _set_fixed_weather(eval_env, seq)

        oracle = solve_theoretical_plan(3, seq, time_limit=10)
        plan = oracle.get("plan", [])
        if not plan:
            continue

        obs, _ = eval_env.reset(seed=0)
        _set_fixed_weather(eval_env, seq)

        for oracle_action in plan:
            valid_actions = eval_env.get_valid_actions()
            agent_action, _ = agent.select_action(obs, valid_actions, deterministic=True)
            oracle_clean = _sanitize_oracle_action(oracle_action, eval_env, valid_actions)

            move_match = agent_action["move"] == oracle_clean["move"]
            if oracle_clean["buy_water"] == 0:
                water_match = agent_action["buy_water"] == 0
            else:
                water_match = (
                    abs(agent_action["buy_water"] - oracle_clean["buy_water"])
                    / oracle_clean["buy_water"]
                    <= 0.1
                )
            if oracle_clean["buy_food"] == 0:
                food_match = agent_action["buy_food"] == 0
            else:
                food_match = (
                    abs(agent_action["buy_food"] - oracle_clean["buy_food"])
                    / oracle_clean["buy_food"]
                    <= 0.1
                )

            if move_match and water_match and food_match:
                matched += 1
            total += 1

            obs, _, done, truncated, _ = eval_env.step(oracle_clean)
            if done or truncated:
                break

    assert total > 0
    match_rate = matched / total
    assert match_rate > 0.85


def test_warmup_skips_non_optimal_oracle_solution(monkeypatch):
    env = make_env(level=3, weather_mode="no_sandstorm", seed=0)
    agent = create_agent(env, config=None, device="cpu")
    trainer = PPOTrainer(agent, config=None, device="cpu")

    monkeypatch.setattr(
        "solve_theoretical_plan",
        lambda *args, **kwargs: {
            "status": "Not Solved",
            "reached": True,
            "reach_day": 4,
            "plan": [
                {"move": 0, "mine": False, "buy_water": 0, "buy_food": 0},
                {"move": 0, "mine": False, "buy_water": 0, "buy_food": 0},
            ],
        },
    )

    summary = warmup_with_oracle(
        agent,
        trainer,
        env,
        level=3,
        episodes=3,
        time_limit=1,
        device="cpu",
        log_interval=1000,
    )

    assert summary["episodes"] == 3.0
    assert summary["solved"] == 0.0
    assert summary["avg_steps"] == 0.0
    assert summary["success_rate"] == 0.0
