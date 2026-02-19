import math

import numpy as np
import pytest

from src.env.config import BASE_CONSUMPTION
from src.env.environment import make_env

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st


def _sample_action(valid_actions, rng):
    move = int(rng.choice(valid_actions["valid_moves"]))
    mine = bool(valid_actions.get("can_mine", False) and rng.rand() < 0.5)

    if valid_actions.get("can_buy", False):
        max_w = int(valid_actions.get("max_buy_water", 0))
        max_f = int(valid_actions.get("max_buy_food", 0))
        buy_water = int(rng.randint(0, max_w + 1)) if max_w > 0 else 0
        buy_food = int(rng.randint(0, max_f + 1)) if max_f > 0 else 0
    else:
        buy_water = 0
        buy_food = 0

    return {
        "move": move,
        "mine": mine,
        "buy_water": buy_water,
        "buy_food": buy_food,
    }


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000), steps=st.integers(min_value=3, max_value=12))
def test_fuzz_invariants_no_negative_resources(seed, steps):
    env = make_env(level=3, seed=seed)
    env.reset(seed=seed)

    rng = np.random.RandomState(seed)

    # Day0 action (purchase allowed)
    action0 = _sample_action(env.get_valid_actions(), rng)
    _, _, done0, _, info0 = env.step(action0)

    assert done0 is False

    last0 = info0.get("last_action") or {}
    buy0_w = int(last0.get("buy_water", 0))
    buy0_f = int(last0.get("buy_food", 0))

    assert last0.get("mine", False) is False
    assert last0.get("move_to") == env.config.START

    expected_money0 = env.config.INIT_MONEY - buy0_w * env.config.WATER_PRICE_BASE - buy0_f * env.config.FOOD_PRICE_BASE
    assert env.state.money == expected_money0
    assert env.state.water == buy0_w
    assert env.state.food == buy0_f

    for _ in range(steps):
        prev_day = env.state.day
        prev_pos = env.state.position
        prev_weather = env.state.weather_today
        prev_water = env.state.water
        prev_food = env.state.food
        prev_money = env.state.money
        prev_reached = env.state.reached
        prev_weight = prev_water * env.config.WATER_WEIGHT + prev_food * env.config.FOOD_WEIGHT
        valid_actions = env.get_valid_actions()
        action = _sample_action(valid_actions, rng)
        _, reward, done, _, info = env.step(action)

        assert env.state.water >= 0
        assert env.state.food >= 0
        assert math.isfinite(env.state.money)

        weight = env.state.water * env.config.WATER_WEIGHT + env.state.food * env.config.FOOD_WEIGHT
        assert weight <= env.config.WEIGHT_LIMIT + 1e-6

        assert env.state.reached is False or env.state.position == env.config.END
        assert env.state.reached >= prev_reached

        if prev_weather == 2:  # Weather.SANDSTORM
            assert env.state.position == prev_pos

        last_action = info.get("last_action") or {}
        move_from = last_action.get("move_from")
        move_to = last_action.get("move_to")
        mining = bool(last_action.get("mine", False))
        buy_w = int(last_action.get("buy_water", 0))
        buy_f = int(last_action.get("buy_food", 0))

        if move_from is not None and move_to is not None:
            assert move_to == move_from or move_to in env.neighbors[move_from]
            assert move_to in valid_actions.get("valid_moves", [move_to])

        if mining:
            assert move_from == move_to
            assert env.state.position in env.config.MINES
        else:
            if valid_actions.get("can_mine") is False:
                assert mining is False

        action_day = info.get("action_day")
        if action_day is not None:
            assert action_day == prev_day
            if buy_w > 0 or buy_f > 0:
                if action_day == 0:
                    assert env.state.position == env.config.START
                else:
                    assert env.state.position in env.config.VILLAGES
            if valid_actions.get("can_buy") is False:
                assert buy_w == 0
                assert buy_f == 0
            max_buy_w = int(valid_actions.get("max_buy_water", 0))
            max_buy_f = int(valid_actions.get("max_buy_food", 0))
            assert buy_w <= max_buy_w
            assert buy_f <= max_buy_f

        if action_day is not None and action_day >= 1 and not info.get("reached", False) and not done:
            assert prev_weight <= env.config.WEIGHT_LIMIT + 1e-6
            base_w, base_f = BASE_CONSUMPTION[prev_weather]
            moved = move_from is not None and move_to is not None and move_from != move_to
            factor = 3 if mining else (2 if moved else 1)
            assert prev_water >= base_w * factor
            assert prev_food >= base_f * factor
            expected_w = prev_water - base_w * factor + buy_w
            expected_f = prev_food - base_f * factor + buy_f

            assert env.state.water == expected_w
            assert env.state.food == expected_f

            if buy_w > 0 or buy_f > 0:
                price_mul = 2
                expected_cost = buy_w * env.config.WATER_PRICE_BASE * price_mul + buy_f * env.config.FOOD_PRICE_BASE * price_mul
            else:
                expected_cost = 0

            expected_money = prev_money + (env.config.MINE_INCOME if mining else 0) - expected_cost
            assert math.isfinite(expected_money)
            assert env.state.money == expected_money

            assert env.state.day == prev_day + 1

        if done:
            snapshot = (env.state.position, env.state.water, env.state.food, env.state.money)
            _, reward2, done2, _, info2 = env.step(action)
            assert done2 is True
            assert info2.get("reached", False) == info.get("reached", False)
            assert reward2 == 0.0
            assert snapshot == (env.state.position, env.state.water, env.state.food, env.state.money)
            break
