from typing import Any

import numpy as np

import src.pipeline.train as train


class _FakeEvalEnv:
    def __init__(self):
        self.config = type("Cfg", (), {"NUM_DAYS": 10})

    def reset(self, seed=None):
        obs = np.zeros(20, dtype=np.float32)
        return obs, {}

    def get_valid_actions(self):
        return {"valid_moves": [0, 1], "can_mine": False, "can_buy": False}


class _FakeAgent:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def reset_hidden(self, device=None):
        return None

    def select_action(self, obs, valid_actions=None, deterministic=False, epsilon=0.0):
        move = 1 if float(obs[8]) > 0.5 else 0
        return {"move": move, "mine": False, "buy_water": 0, "buy_food": 0}, 0.0


def test_resolve_weather_split_modes_level35_prefers_train_val_test():
    train_mode, val_mode, test_mode = train._resolve_weather_split_modes(35, "train_medium")
    assert train_mode == "train"
    assert val_mode == "val"
    assert test_mode == "test"


def test_weather_sensitivity_probe_detects_change(monkeypatch):
    monkeypatch.setattr(train, "make_env", lambda **kwargs: _FakeEvalEnv())

    fake_agent: Any = _FakeAgent()
    sensitive = train._weather_sensitivity_test(
        fake_agent,
        level=35,
        weather_mode="val",
        device="cpu",
    )

    assert sensitive is True
