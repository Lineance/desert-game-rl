import pulp
import pytest

from src.env.config import Level3Config, Weather
from src.utils.oracle import solve_theoretical_optimal, solve_theoretical_plan


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
    assert "is_proven_optimal" in result

    assert result["status"] == "Optimal"
    assert result["is_proven_optimal"] is True
    assert result["reached"] is True
    assert result["length"] == result["reach_day"]
    assert result["length"] <= Level3Config.NUM_DAYS
    assert result["objective"] == pytest.approx(
        result["final_money"]
        + 0.5 * Level3Config.WATER_PRICE_BASE * result["final_water"]
        + 0.5 * Level3Config.FOOD_PRICE_BASE * result["final_food"]
    )
    assert result["return"] == pytest.approx(result["objective"] - Level3Config.INIT_MONEY)


def test_oracle_plan_length_matches_reach_day_for_optimal():
    weather_seq = [Weather.SUNNY] * Level3Config.NUM_DAYS
    plan_result = solve_theoretical_plan(3, weather_seq, time_limit=2)

    assert plan_result["status"] == "Optimal"
    assert plan_result["is_proven_optimal"] is True
    assert plan_result["reached"] is True
    assert len(plan_result["plan"]) == plan_result["reach_day"] + 1


def test_oracle_non_optimal_has_no_upper_bound_or_plan(monkeypatch):
    original_solve = pulp.LpProblem.solve

    def _fake_solve(self, solver=None, **kwargs):
        self.status = pulp.LpStatusNotSolved
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)
    try:
        weather_seq = [Weather.SUNNY] * Level3Config.NUM_DAYS
        result = solve_theoretical_optimal(3, weather_seq, time_limit=1)
        plan_result = solve_theoretical_plan(3, weather_seq, time_limit=1)
    finally:
        monkeypatch.setattr(pulp.LpProblem, "solve", original_solve)

    assert result["status"] == "Not Solved"
    assert result["is_proven_optimal"] is False
    assert result["objective"] != result["objective"]

    assert plan_result["status"] == "Not Solved"
    assert plan_result["is_proven_optimal"] is False
    assert plan_result["plan"] == []
