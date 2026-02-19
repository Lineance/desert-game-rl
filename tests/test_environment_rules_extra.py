import numpy as np
import pytest

from src.env.config import Weather
from src.env.environment import make_env


def test_day0_purchase_only_once():
    env = make_env(level=3, seed=1)
    env.reset(seed=1)

    env.step({"move": env.state.position, "mine": False, "buy_water": 10, "buy_food": 10})

    assert env.state.has_purchased_at_start is True
    assert env.state.day == 1
    assert env.state.position == env.config.START

    va = env.get_valid_actions()
    assert va["can_buy"] is False


def test_day0_purchase_has_no_consumption():
    env = make_env(level=3, seed=2)
    env.reset(seed=2)

    env.step({"move": env.state.position, "mine": False, "buy_water": 15, "buy_food": 12})

    assert env.state.water == 15
    assert env.state.food == 12


def test_sandstorm_mining_allowed_when_staying_on_mine():
    env = make_env(level=3, seed=3)
    env.reset(seed=3)

    env.step({"move": env.state.position, "mine": False, "buy_water": 60, "buy_food": 60})

    mine_node = env.config.MINES[0]
    env.state.day = 1
    env.state.position = mine_node
    env.state.water = 100
    env.state.food = 100
    env.state.money = 1000
    env.state.weather_today = Weather.SANDSTORM
    env.state.path_history = [mine_node, mine_node]

    _, _, done, _, info = env.step({"move": mine_node, "mine": True, "buy_water": 0, "buy_food": 0})

    assert done is False
    assert info["last_action"]["mine"] is True
    assert env.state.money == 1200


def test_valid_actions_respects_weight_limit_on_day0():
    env = make_env(level=3, seed=4)
    env.reset(seed=4)

    env.state.water = env.config.WEIGHT_LIMIT // env.config.WATER_WEIGHT
    env.state.food = 0

    va = env.get_valid_actions()
    assert va["can_buy"] is True
    assert va["max_buy_water"] == 0
    assert va["max_buy_food"] == 0


def test_step_terminates_when_resources_insufficient():
    env = make_env(level=3, seed=5)
    env.reset(seed=5)

    env.step({"move": env.state.position, "mine": False, "buy_water": 10, "buy_food": 10})

    env.state.day = 1
    env.state.position = env.config.START
    env.state.water = 0
    env.state.food = 0
    env.state.money = 1000
    env.state.weather_today = Weather.SUNNY
    env.state.path_history = [env.config.START, env.config.START]

    _, _, done, _, info = env.step({"move": env.state.position, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is True
    assert env.state.terminated is True
    assert info["last_action"]["move_to"] == env.config.START


def test_non_adjacent_move_is_blocked():
    env = make_env(level=3, seed=6)
    env.reset(seed=6)

    env.step({"move": env.state.position, "mine": False, "buy_water": 30, "buy_food": 30})

    env.state.day = 1
    env.state.weather_today = Weather.SUNNY
    prev_pos = env.state.position
    invalid_target = next(i for i in range(env.config.NUM_NODES) if i not in env.neighbors[prev_pos] and i != prev_pos)

    _, _, done, _, info = env.step({"move": invalid_target, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is False
    assert env.state.position == prev_pos
    assert info["last_action"]["move_to"] == prev_pos


def test_reach_end_refund_matches_remaining_resources():
    env = make_env(level=3, seed=7)
    env.reset(seed=7)

    end_node = env.config.END
    prev_node = env.neighbors[end_node][0]

    env.state.day = 1
    env.state.position = prev_node
    env.state.water = 20
    env.state.food = 20
    env.state.money = 5000
    env.state.weather_today = Weather.SUNNY
    env.state.path_history = [prev_node, prev_node]

    _, _, done, _, info = env.step({"move": end_node, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is True
    assert info["reached"] is True
    assert env.state.water == 0
    assert env.state.food == 0

    remaining_water = 20 - 6
    remaining_food = 20 - 8
    expected_refund = remaining_water * env.config.WATER_PRICE_BASE * 0.5 + remaining_food * env.config.FOOD_PRICE_BASE * 0.5
    assert abs(env.state.money - (5000 + expected_refund)) < 1e-6


def test_timeout_terminates_when_not_reached():
    env = make_env(level=3, seed=8)
    env.reset(seed=8)

    env.step({"move": env.state.position, "mine": False, "buy_water": 30, "buy_food": 30})

    env.state.day = env.config.NUM_DAYS
    env.state.position = env.config.START
    env.state.water = 100
    env.state.food = 100
    env.state.money = 5000
    env.state.weather_today = Weather.SUNNY
    env.state.path_history = [env.config.START, env.config.START]

    _, _, done, _, info = env.step({"move": env.state.position, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is True
    assert info["reached"] is False
    assert env.state.terminated is True


def test_sandstorm_move_attempt_forces_stay_and_penalty():
    env = make_env(level=3, seed=9)
    env.reset(seed=9)

    env.step({"move": env.state.position, "mine": False, "buy_water": 50, "buy_food": 50})

    env.state.day = 1
    env.state.weather_today = Weather.SANDSTORM
    prev_pos = env.state.position
    neighbor = env.neighbors[prev_pos][0]
    env.state.path_history = [prev_pos, prev_pos]

    _, reward, done, _, info = env.step({"move": neighbor, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is False
    assert env.state.position == prev_pos
    assert info["last_action"]["move_to"] == prev_pos
    assert reward <= -50.0


def test_weather_sequence_same_seed_is_deterministic():
    env = make_env(level=3, seed=None)

    env.reset(seed=42)
    seq1 = tuple(env.state.weather_future)

    env.reset(seed=42)
    seq2 = tuple(env.state.weather_future)

    assert seq1 == seq2


def test_observation_does_not_include_future_weather():
    env = make_env(level=3, seed=10)
    obs, _ = env.reset(seed=10)

    env.state.weather_today = Weather.SUNNY
    if len(env.state.weather_future) > 1:
        env.state.weather_future[1] = Weather.SANDSTORM

    obs = env._get_observation()
    weather_onehot = obs[6:9]

    assert weather_onehot[Weather.SUNNY] == 1.0
    assert weather_onehot[Weather.HOT] == 0.0
    assert weather_onehot[Weather.SANDSTORM] == 0.0


def test_hot_weather_consumption_on_stay():
    env = make_env(level=3, seed=11)
    env.reset(seed=11)

    env.step({"move": env.state.position, "mine": False, "buy_water": 40, "buy_food": 40})

    env.state.day = 1
    env.state.position = env.config.START
    env.state.water = 50
    env.state.food = 50
    env.state.weather_today = Weather.HOT
    env.state.path_history = [env.config.START, env.config.START]

    _, _, done, _, _ = env.step({"move": env.state.position, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is False
    assert env.state.water == 41
    assert env.state.food == 41


def test_hot_weather_consumption_on_move():
    env = make_env(level=3, seed=14)
    env.reset(seed=14)

    env.step({"move": env.state.position, "mine": False, "buy_water": 50, "buy_food": 50})

    env.state.day = 1
    env.state.weather_today = Weather.HOT
    env.state.water = 60
    env.state.food = 60
    env.state.path_history = [env.config.START, env.config.START]

    prev_pos = env.state.position
    next_pos = env.neighbors[prev_pos][0]

    _, _, done, _, _ = env.step({"move": next_pos, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is False
    assert env.state.position == next_pos
    assert env.state.water == 42
    assert env.state.food == 42


def test_hot_weather_consumption_on_mine():
    env = make_env(level=3, seed=15)
    env.reset(seed=15)

    env.step({"move": env.state.position, "mine": False, "buy_water": 80, "buy_food": 80})

    mine_node = env.config.MINES[0]
    env.state.day = 1
    env.state.position = mine_node
    env.state.weather_today = Weather.HOT
    env.state.water = 100
    env.state.food = 100
    env.state.money = 1000
    env.state.path_history = [mine_node, mine_node]

    _, _, done, _, info = env.step({"move": mine_node, "mine": True, "buy_water": 0, "buy_food": 0})

    assert done is False
    assert info["last_action"]["mine"] is True
    assert env.state.water == 73
    assert env.state.food == 73
    assert env.state.money == 1000 + env.config.MINE_INCOME


def test_sandstorm_mining_consumption_on_mine():
    env = make_env(level=3, seed=16)
    env.reset(seed=16)

    env.step({"move": env.state.position, "mine": False, "buy_water": 80, "buy_food": 80})

    mine_node = env.config.MINES[0]
    env.state.day = 1
    env.state.position = mine_node
    env.state.weather_today = Weather.SANDSTORM
    env.state.water = 100
    env.state.food = 100
    env.state.money = 1000
    env.state.path_history = [mine_node, mine_node]

    _, _, done, _, info = env.step({"move": mine_node, "mine": True, "buy_water": 0, "buy_food": 0})

    assert done is False
    assert info["last_action"]["mine"] is True
    assert env.state.water == 70
    assert env.state.food == 70
    assert env.state.money == 1000 + env.config.MINE_INCOME


def test_village_purchase_after_arrival(monkeypatch):
    env = make_env(level=3, seed=12)
    env.reset(seed=12)

    env.step({"move": env.state.position, "mine": False, "buy_water": 30, "buy_food": 30})

    village_node = env.neighbors[env.config.START][0]
    monkeypatch.setattr(env.config, "VILLAGES", [village_node], raising=False)

    env.state.day = 1
    env.state.position = village_node
    env.state.weather_today = Weather.SUNNY
    env.state.path_history = [env.config.START, village_node]

    prev_money = env.state.money
    prev_water = env.state.water
    prev_food = env.state.food

    _, _, done, _, info = env.step({"move": village_node, "mine": False, "buy_water": 2, "buy_food": 3})

    assert done is False
    assert info["last_action"]["buy_water"] == 2
    assert info["last_action"]["buy_food"] == 3
    assert env.state.water == prev_water - 3 + 2
    assert env.state.food == prev_food - 4 + 3
    assert env.state.money == prev_money - (2 * env.config.WATER_PRICE_BASE * 2 + 3 * env.config.FOOD_PRICE_BASE * 2)


def test_observation_and_info_fields_from_reset_and_step():
    env = make_env(level=3, seed=13)
    obs, info = env.reset(seed=13)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (19,)
    assert set(info.keys()) >= {"day", "position", "water", "food", "money", "weather_today", "belief", "reached", "last_action"}

    env.step({"move": env.state.position, "mine": False, "buy_water": 10, "buy_food": 10})
    obs2, _, _, _, info2 = env.step({"move": env.state.position, "mine": False, "buy_water": 0, "buy_food": 0})

    assert isinstance(obs2, np.ndarray)
    assert obs2.shape == (19,)
    assert set(info2.keys()) >= {"day", "position", "water", "food", "money", "weather_today", "belief", "reached", "last_action"}


def test_observation_consistency_with_state_features():
    env = make_env(level=3, seed=17)
    obs, _ = env.reset(seed=17)

    state = env.state
    expected = np.array(
        [
            state.day / env.config.NUM_DAYS,
            state.position / env.config.NUM_NODES,
            state.water / 200.0,
            state.food / 200.0,
            state.money / env.config.INIT_MONEY,
            env.dist_to_end[state.position] / env.config.NUM_NODES,
        ],
        dtype=np.float32,
    )

    assert np.allclose(obs[:6], expected)
    weather_onehot = obs[6:9]
    assert weather_onehot[state.weather_today] == 1.0


def test_weather_sequence_values_and_length():
    env = make_env(level=3, seed=18)
    env.reset(seed=18)

    seq = env.state.weather_future
    assert len(seq) == env.config.NUM_DAYS
    assert all(w in (Weather.SUNNY, Weather.HOT, Weather.SANDSTORM) for w in seq)


def test_weather_revealed_daily_matches_sequence():
    env = make_env(level=3, seed=20)
    env.reset(seed=20)

    sequence = list(env.state.weather_future)

    obs0, _, _, _, info0 = env.step({"move": env.state.position, "mine": False, "buy_water": 40, "buy_food": 40})
    assert info0["action_day"] == 0
    assert info0["weather_today"] == sequence[0]
    assert obs0[6:9][sequence[0]] == 1.0

    obs1, _, _, _, info1 = env.step({"move": env.state.position, "mine": False, "buy_water": 0, "buy_food": 0})
    assert info1["action_day"] == 1
    assert info1["weather_today"] == sequence[0]
    assert obs1[6:9][sequence[0]] == 1.0

    obs2, _, _, _, info2 = env.step({"move": env.state.position, "mine": False, "buy_water": 0, "buy_food": 0})
    assert info2["action_day"] == 2
    assert info2["weather_today"] == sequence[1]
    assert obs2[6:9][sequence[1]] == 1.0


def test_weather_sequence_deterministic_across_steps_with_seed():
    env1 = make_env(level=3, seed=42)
    env2 = make_env(level=3, seed=42)

    env1.reset(seed=42)
    env2.reset(seed=42)

    env1.step({"move": env1.state.position, "mine": False, "buy_water": 40, "buy_food": 40})
    env2.step({"move": env2.state.position, "mine": False, "buy_water": 40, "buy_food": 40})

    weather1 = [env1.state.weather_today]
    weather2 = [env2.state.weather_today]

    for _ in range(5):
        env1.step({"move": env1.state.position, "mine": False, "buy_water": 0, "buy_food": 0})
        env2.step({"move": env2.state.position, "mine": False, "buy_water": 0, "buy_food": 0})
        weather1.append(env1.state.weather_today)
        weather2.append(env2.state.weather_today)

    assert weather1 == weather2


def test_sunny_and_sandstorm_consumption_on_stay():
    env = make_env(level=3, seed=21)
    env.reset(seed=21)

    env.step({"move": env.state.position, "mine": False, "buy_water": 60, "buy_food": 60})

    env.state.day = 1
    env.state.position = env.config.START
    env.state.water = 50
    env.state.food = 50
    env.state.weather_today = Weather.SUNNY
    env.state.path_history = [env.config.START, env.config.START]

    env.step({"move": env.state.position, "mine": False, "buy_water": 0, "buy_food": 0})

    assert env.state.water == 47
    assert env.state.food == 46

    env.state.weather_today = Weather.SANDSTORM
    env.state.path_history = [env.config.START, env.config.START]

    env.step({"move": env.state.position, "mine": False, "buy_water": 0, "buy_food": 0})

    assert env.state.water == 37
    assert env.state.food == 36


def test_village_purchase_on_arrival_from_move(monkeypatch):
    env = make_env(level=3, seed=22)
    env.reset(seed=22)

    env.step({"move": env.state.position, "mine": False, "buy_water": 50, "buy_food": 50})

    village_node = env.neighbors[env.config.START][0]
    monkeypatch.setattr(env.config, "VILLAGES", [village_node], raising=False)

    env.state.day = 1
    env.state.position = env.config.START
    env.state.weather_today = Weather.SUNNY
    env.state.water = 50
    env.state.food = 50
    env.state.money = 5000
    env.state.path_history = [env.config.START, env.config.START]

    _, _, done, _, info = env.step({"move": village_node, "mine": False, "buy_water": 2, "buy_food": 3})

    assert done is False
    assert info["last_action"]["move_to"] == village_node
    assert env.state.water == 46
    assert env.state.food == 45
    assert env.state.money == 5000 - (2 * env.config.WATER_PRICE_BASE * 2 + 3 * env.config.FOOD_PRICE_BASE * 2)


def test_final_day_sandstorm_blocks_reaching_end():
    env = make_env(level=3, seed=23)
    env.reset(seed=23)

    env.step({"move": env.state.position, "mine": False, "buy_water": 60, "buy_food": 60})

    end_node = env.config.END
    prev_node = env.neighbors[end_node][0]

    env.state.day = env.config.NUM_DAYS
    env.state.position = prev_node
    env.state.weather_today = Weather.SANDSTORM
    env.state.water = 100
    env.state.food = 100
    env.state.money = 1000
    env.state.path_history = [prev_node, prev_node]

    _, _, done, _, info = env.step({"move": end_node, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is True
    assert info["reached"] is False
    assert env.state.position == prev_node


def test_arrival_on_deadline_day_is_allowed():
    env = make_env(level=3, seed=24)
    env.reset(seed=24)

    env.step({"move": env.state.position, "mine": False, "buy_water": 60, "buy_food": 60})

    end_node = env.config.END
    prev_node = env.neighbors[end_node][0]

    env.state.day = env.config.NUM_DAYS
    env.state.position = prev_node
    env.state.weather_today = Weather.SUNNY
    env.state.water = 100
    env.state.food = 100
    env.state.money = 1000
    env.state.path_history = [prev_node, prev_node]

    _, _, done, _, info = env.step({"move": end_node, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is True
    assert info["reached"] is True


def test_reset_seed_supports_stochastic_evaluation_interface():
    returns = []

    for seed in range(5):
        env = make_env(level=3, seed=seed)
        env.reset(seed=seed)
        env.step({"move": env.state.position, "mine": False, "buy_water": 40, "buy_food": 40})

        total_reward = 0.0
        for _ in range(3):
            _, reward, done, _, _ = env.step({"move": env.state.position, "mine": False, "buy_water": 0, "buy_food": 0})
            total_reward += float(reward)
            if done:
                break

        returns.append(total_reward)

    assert all(not np.isnan(r) for r in returns)
