import time
from typing import Any, Dict

from src.env.environment import make_env
from src.models.agent import create_agent
from src.models.ppo import PPOTrainer
from src.pipeline.warmup import warmup_with_oracle


def _fake_plan(_: int, __, time_limit: int = 0) -> Dict[str, Any]:
    return {
        "status": "Optimal",
        "reached": True,
        "reach_day": 1,
        "plan": [{"move": 0, "mine": False, "buy_water": 0, "buy_food": 0}],
    }


def _fake_plan_with_config(config_cls, weather_seq, time_limit: int = 0):
    return {
        "status": "Optimal",
        "reached": True,
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


def test_warmup_oracle_solve_interval_reuses_last_plan(monkeypatch):
    env = make_env(level=3, weather_mode="no_sandstorm", seed=0)
    agent = create_agent(env, config=None, device="cpu")
    trainer = PPOTrainer(agent, config=None, device="cpu")

    import src.pipeline.warmup as warmup_module

    called = {"solve": 0}

    def _wrapped(level, weather_seq, time_limit=0):
        called["solve"] += 1
        return _fake_plan(level, weather_seq, time_limit=time_limit)

    monkeypatch.setattr(warmup_module, "solve_theoretical_plan", _wrapped)

    summary = warmup_with_oracle(
        agent,
        trainer,
        env,
        level=3,
        episodes=5,
        time_limit=1,
        device="cpu",
        log_interval=100,
        oracle_solve_interval=3,
    )

    assert called["solve"] == 2
    assert summary["oracle_solve_calls"] == 2.0
    assert summary["oracle_reuse_calls"] == 3.0


def test_warmup_custom_oracle_failure_fallback(monkeypatch):
    env = make_env(level=35, weather_mode="train_easy", seed=0)
    agent = create_agent(env, config=None, device="cpu")
    trainer = PPOTrainer(agent, config=None, device="cpu")

    import src.pipeline.warmup as warmup_module

    called = {"fallback": 0}

    def _raise_custom(config_cls, weather_seq, time_limit=0):
        raise RuntimeError("custom solver failed")

    def _fallback(level, weather_seq, time_limit=0):
        called["fallback"] += 1
        return _fake_plan(level, weather_seq, time_limit=time_limit)

    monkeypatch.setattr(warmup_module, "solve_theoretical_plan_with_config", _raise_custom)
    monkeypatch.setattr(warmup_module, "solve_theoretical_plan", _fallback)

    summary = warmup_with_oracle(
        agent,
        trainer,
        env,
        level=35,
        episodes=2,
        time_limit=1,
        device="cpu",
        log_interval=100,
        oracle_config=env.config,
    )

    assert called["fallback"] >= 1
    assert summary["oracle_fallback_calls"] >= 1.0


def test_warmup_discards_unreached_oracle_solution(monkeypatch):
    env = make_env(level=3, weather_mode="no_sandstorm", seed=0)
    agent = create_agent(env, config=None, device="cpu")
    trainer = PPOTrainer(agent, config=None, device="cpu")

    import src.pipeline.warmup as warmup_module

    def _unreached_plan(level, weather_seq, time_limit=0):
        return {
            "status": "Infeasible",
            "reached": False,
            "reach_day": 0,
            "plan": [{"move": 0, "mine": False, "buy_water": 0, "buy_food": 0}],
        }

    monkeypatch.setattr(warmup_module, "solve_theoretical_plan", _unreached_plan)

    summary = warmup_with_oracle(
        agent,
        trainer,
        env,
        level=3,
        episodes=1,
        time_limit=1,
        device="cpu",
        log_interval=100,
        oracle_solve_interval=1,
    )

    assert summary["solved"] == 0.0
    assert summary["oracle_discarded_unreached"] == 1.0


def test_warmup_oracle_delay_reproduction_and_timing(monkeypatch):
    env = make_env(level=3, weather_mode="no_sandstorm", seed=0)
    agent = create_agent(env, config=None, device="cpu")
    trainer = PPOTrainer(agent, config=None, device="cpu")

    import src.pipeline.warmup as warmup_module

    def _slow_fake_plan(level, weather_seq, time_limit=0):
        time.sleep(0.08)
        return _fake_plan(level, weather_seq, time_limit=time_limit)

    monkeypatch.setattr(warmup_module, "solve_theoretical_plan", _slow_fake_plan)

    t0 = time.perf_counter()
    summary_no_reuse = warmup_with_oracle(
        agent,
        trainer,
        env,
        level=3,
        episodes=6,
        time_limit=1,
        device="cpu",
        log_interval=100,
        oracle_solve_interval=1,
    )
    elapsed_no_reuse = time.perf_counter() - t0

    t1 = time.perf_counter()
    summary_reuse = warmup_with_oracle(
        agent,
        trainer,
        env,
        level=3,
        episodes=6,
        time_limit=1,
        device="cpu",
        log_interval=100,
        oracle_solve_interval=3,
    )
    elapsed_reuse = time.perf_counter() - t1

    assert summary_no_reuse["oracle_solve_calls"] == 6.0
    assert summary_reuse["oracle_solve_calls"] == 2.0
    assert elapsed_reuse < elapsed_no_reuse * 0.8
