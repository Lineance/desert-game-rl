from typing import Any, Dict

from src.env.environment import make_env
from src.models.agent import create_agent
from src.models.ppo import PPOTrainer
from src.pipeline.warmup import warmup_with_oracle


def _fake_plan(_: int, __, time_limit: int = 0) -> Dict[str, Any]:
    return {
        "status": "Optimal",
        "reached": False,
        "reach_day": 1,
        "plan": [{"move": 0, "mine": False, "buy_water": 0, "buy_food": 0}],
    }


def _fake_plan_with_config(config_cls, weather_seq, time_limit: int = 0):
    return {
        "status": "Optimal",
        "reached": False,
        "reach_day": 1,
        "plan": [
            {
                "move": int(config_cls.START),
                "mine": False,
                "buy_water": 0,
                "buy_food": 0,
            }
        ],
    }


def test_warmup_filter_counts(monkeypatch):
    env = make_env(level=3, weather_mode="no_sandstorm", seed=0)
    agent = create_agent(env, config=None, device="cpu")
    trainer = PPOTrainer(agent, config=None, device="cpu")

    import src.pipeline.warmup as warmup_module

    monkeypatch.setattr(warmup_module, "solve_theoretical_plan", _fake_plan)

    summary = warmup_with_oracle(
        agent,
        trainer,
        env,
        level=3,
        episodes=2,
        time_limit=1,
        device="cpu",
        log_interval=100,
        trajectory_filter=lambda payload: False,
    )

    assert summary["accepted_episodes"] == 0.0
    assert summary["filtered_episodes"] == 2.0


def test_warmup_uses_oracle_config_path(monkeypatch):
    env = make_env(level=3, weather_mode="no_sandstorm", seed=0)
    agent = create_agent(env, config=None, device="cpu")
    trainer = PPOTrainer(agent, config=None, device="cpu")

    import src.pipeline.warmup as warmup_module

    called = {"with_config": 0}

    def _wrapped_with_config(config_cls, weather_seq, time_limit=0):
        called["with_config"] += 1
        return _fake_plan_with_config(config_cls, weather_seq, time_limit=time_limit)

    monkeypatch.setattr(warmup_module, "solve_theoretical_plan_with_config", _wrapped_with_config)

    summary = warmup_with_oracle(
        agent,
        trainer,
        env,
        level=3,
        episodes=1,
        time_limit=1,
        device="cpu",
        log_interval=100,
        oracle_config=env.config,
    )

    assert called["with_config"] == 1
    assert summary["episodes"] == 1.0
