import numpy as np
from src.env.config import Weather
from src.env.environment import make_env


def test_day0_can_buy_and_dynamic_limits():
    env = make_env(level=3, seed=0)
    _, _ = env.reset(seed=0)
    va = env.get_valid_actions()

    assert va["can_buy"] is True
    assert va["max_buy_water"] > 0
    assert va["max_buy_food"] > 0
    assert va["max_buy_water"] <= 200
    assert va["max_buy_food"] <= 200


def test_sandstorm_forces_stay():
    env = make_env(level=3, seed=0)
    _, _ = env.reset(seed=0)

    # 第0天先购买，避免资源不足立即终止
    env.step({"move": env.state.position, "mine": False, "buy_water": 40, "buy_food": 40})

    # 强制到第1天沙暴，并尝试移动
    env.state.weather_today = Weather.SANDSTORM
    prev_pos = env.state.position
    neighbor = env.neighbors[prev_pos][0]

    _, _, done, _, info = env.step({"move": neighbor, "mine": False, "buy_water": 0, "buy_food": 0})

    assert env.state.position == prev_pos
    assert info["last_action"]["move_from"] == prev_pos
    assert info["last_action"]["move_to"] == prev_pos
    assert done is False


def test_arrive_day_cannot_mine():
    env = make_env(level=3, seed=0)
    _, _ = env.reset(seed=0)

    mine_node = env.config.MINES[0]
    from_node = env.neighbors[mine_node][0]
    env.state.day = 1
    env.state.position = from_node
    env.state.water = 100
    env.state.food = 100
    env.state.money = 1000
    env.state.weather_today = Weather.SUNNY
    env.state.path_history = [from_node, from_node]

    prev_money = env.state.money
    _, _, _, _, info = env.step({"move": mine_node, "mine": True, "buy_water": 0, "buy_food": 0})

    # 到达当天挖矿会被忽略，不应获得矿山收益
    assert env.state.position == mine_node
    assert info["last_action"]["mine"] is False
    assert env.state.money == prev_money


def test_reach_end_zeroes_resources_after_refund():
    env = make_env(level=3, seed=0)
    _, _ = env.reset(seed=0)

    end_node = env.config.END
    prev_node = env.neighbors[end_node][0]

    env.state.day = 1
    env.state.position = prev_node
    env.state.water = 200
    env.state.food = 200
    env.state.money = 5000
    env.state.weather_today = Weather.SUNNY
    env.state.path_history = [prev_node, prev_node]

    _, _, done, _, info = env.step({"move": end_node, "mine": False, "buy_water": 0, "buy_food": 0})

    assert done is True
    assert info["reached"] is True
    assert env.state.water == 0
    assert env.state.food == 0
