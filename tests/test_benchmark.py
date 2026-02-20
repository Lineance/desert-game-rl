from src.pipeline.benchmark import (
    _summarize,
    compare_model_vs_oracle,
    evaluate_random_baseline,
)


def test_summarize_basic():
    results = [
        {"reached": True, "final_money": 100.0, "return": 10.0, "length": 5},
        {"reached": False, "final_money": 50.0, "return": -10.0, "length": 1},
    ]
    stats = _summarize(results, level=3)
    assert stats["runs"] == 2
    assert 0 <= stats["success_rate"] <= 1
    assert "early_fail_rate" in stats


def test_random_baseline_runs():
    summary = evaluate_random_baseline(level=3, runs=2, weather_modes=["no_sandstorm"])
    assert "no_sandstorm" in summary
    assert "overall" in summary


def test_compare_model_vs_oracle_with_monkeypatch():
    model_detailed = {
        "no_sandstorm": [
            {"seed": 0, "final_money": 500.0},
            {"seed": 1, "final_money": 800.0},
        ]
    }
    oracle_detailed = {
        "no_sandstorm": [
            {"seed": 0, "objective": 1000.0, "status": "Optimal"},
            {"seed": 1, "objective": 1000.0, "status": "Optimal"},
        ]
    }
    comp = compare_model_vs_oracle(model_detailed, oracle_detailed)

    assert "no_sandstorm" in comp
    assert "overall" in comp
    assert 0 < comp["overall"]["avg_money_to_oracle_ratio"] < 1
