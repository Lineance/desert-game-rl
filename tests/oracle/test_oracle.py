import pulp
import pytest

from src.env.config import Level3Config, Weather
from src.pipeline.benchmark import _build_oracle_detailed, _summarize_oracle_detailed
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


def test_oracle_invalid_level_raises():
    with pytest.raises(ValueError):
        solve_theoretical_optimal(99, [Weather.SUNNY] * Level3Config.NUM_DAYS, time_limit=1)


def test_evaluate_oracle_upper_bound_with_monkeypatch(monkeypatch):
    import src.pipeline.benchmark as benchmark

    def fake_oracle(level, weather_seq, time_limit=30):
        return {"status": "Optimal", "objective": 1234.0, "reached": True}

    monkeypatch.setattr(benchmark, "solve_theoretical_optimal", fake_oracle)

    detailed = _build_oracle_detailed(
        level=3,
        runs=2,
        weather_modes=["no_sandstorm"],
        oracle_time_limit=1,
    )
    summary = _summarize_oracle_detailed(detailed, level=3)

    assert summary["no_sandstorm"]["oracle_solved_rate"] == 1.0
    assert summary["no_sandstorm"]["oracle_avg_objective"] == 1234.0
    assert "overall" in summary


def test_build_oracle_detailed_with_empty_oracle(monkeypatch):
    import src.pipeline.benchmark as benchmark

    def fake_oracle(level, weather_seq, time_limit=30):
        return {}

    monkeypatch.setattr(benchmark, "solve_theoretical_optimal", fake_oracle)

    detailed = _build_oracle_detailed(
        level=3,
        runs=1,
        weather_modes=["no_sandstorm"],
        oracle_time_limit=1,
    )

    record = detailed["no_sandstorm"][0]
    assert record["status"] == "Unknown"
    assert record["objective"] == 0.0


def test_build_oracle_detailed_oracle_exception(monkeypatch):
    import src.pipeline.benchmark as benchmark

    def fake_oracle(level, weather_seq, time_limit=30):
        raise RuntimeError("oracle failed")

    monkeypatch.setattr(benchmark, "solve_theoretical_optimal", fake_oracle)

    with pytest.raises(RuntimeError):
        _build_oracle_detailed(
            level=3,
            runs=1,
            weather_modes=["no_sandstorm"],
            oracle_time_limit=1,
        )
