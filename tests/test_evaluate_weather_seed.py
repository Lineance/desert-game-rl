import torch

from src.env.environment import make_env
from src.pipeline.evaluate import run_episode


class _StayAndBuyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def reset_hidden(self, batch_size: int = 1, device=None):
        return None

    def select_action(self, obs, valid_actions, deterministic: bool = True):
        can_buy = bool(valid_actions.get("can_buy", False))
        max_w = int(valid_actions.get("max_buy_water", 0))
        max_f = int(valid_actions.get("max_buy_food", 0))

        buy_water = min(40, max_w) if can_buy else 0
        buy_food = min(40, max_f) if can_buy else 0

        return {
            "move": int(valid_actions["valid_moves"][-1]),
            "mine": False,
            "buy_water": buy_water,
            "buy_food": buy_food,
        }, 0.0


def test_weather_sequence_same_seed_is_reproducible():
    env = make_env(level=3, weather_mode="sunny_bias", seed=None)

    env.reset(seed=123)
    seq1 = tuple(env.state.weather_future)

    env.reset(seed=123)
    seq2 = tuple(env.state.weather_future)

    assert seq1 == seq2


def test_weather_sequence_changes_across_seeds():
    env = make_env(level=3, weather_mode="sunny_bias", seed=None)

    sequences = set()
    for seed in range(10):
        env.reset(seed=seed)
        sequences.add(tuple(env.state.weather_future))

    assert len(sequences) > 1


def test_run_episode_is_reproducible_given_same_seed():
    env = make_env(level=3, weather_mode="sunny_bias", seed=None)
    agent = _StayAndBuyAgent()

    r1 = run_episode(agent, env, seed=7, deterministic=True, verbose=False)
    r2 = run_episode(agent, env, seed=7, deterministic=True, verbose=False)

    rec1 = [(x["day"], x["position"], x["weather"], x["water"], x["food"]) for x in r1["records"]]
    rec2 = [(x["day"], x["position"], x["weather"], x["water"], x["food"]) for x in r2["records"]]

    assert rec1 == rec2
    assert r1["final_money"] == r2["final_money"]
    assert r1["length"] == r2["length"]
    assert r1["reached"] == r2["reached"]
