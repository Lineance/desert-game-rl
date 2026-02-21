import pytest

from src.env.config import Level3Config, Weather
from src.pipeline.oracle import (
    solve_theoretical_optimal,
    solve_theoretical_plan_with_config,
)


def test_oracle_rejects_wrong_weather_length():
    with pytest.raises(ValueError):
        solve_theoretical_optimal(3, [Weather.SUNNY] * (Level3Config.NUM_DAYS - 1), time_limit=1)


def test_oracle_returns_result_dict():
    weather_seq = [Weather.SUNNY] * Level3Config.NUM_DAYS
    result = solve_theoretical_optimal(3, weather_seq, time_limit=2)

    assert "status" in result
    assert "objective" in result
    assert "reached" in result
    assert "final_money" in result
    assert "final_water" in result
    assert "final_food" in result
    assert "reach_day" in result
    assert "length" in result
    assert "return" in result


def test_oracle_plan_with_config_returns_plan_dict():
    weather_seq = [Weather.SUNNY] * Level3Config.NUM_DAYS
    result = solve_theoretical_plan_with_config(Level3Config, weather_seq, time_limit=2)

    assert "status" in result
    assert "reached" in result
    assert "reach_day" in result
    assert "plan" in result
    assert isinstance(result["plan"], list)
