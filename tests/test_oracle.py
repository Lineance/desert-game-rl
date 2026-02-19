import pytest
from src.env.config import Level3Config, Weather
from src.pipeline.oracle_optimal import solve_theoretical_optimal


def test_oracle_rejects_wrong_weather_length():
    with pytest.raises(ValueError):
        solve_theoretical_optimal(
            3, [Weather.SUNNY] * (Level3Config.NUM_DAYS - 1), time_limit=1
        )


def test_oracle_returns_result_dict():
    weather_seq = [Weather.SUNNY] * Level3Config.NUM_DAYS
    result = solve_theoretical_optimal(3, weather_seq, time_limit=2)

    assert "status" in result
    assert "objective" in result
    assert "reached" in result
