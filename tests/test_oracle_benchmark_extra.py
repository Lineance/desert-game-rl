import pytest

from src.env.config import Level3Config, Weather
from src.pipeline.benchmark import _build_oracle_detailed, _summarize_oracle_detailed
from src.utils.oracle import solve_theoretical_optimal


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
