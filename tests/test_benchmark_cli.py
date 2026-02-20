import json
import sys

import pytest

import src.pipeline.benchmark as benchmark


def test_benchmark_main_writes_output_json(monkeypatch, tmp_path):
    def fake_eval_detailed(**kwargs):
        return {
            "no_sandstorm": [
                {"seed": 0, "final_money": 100.0},
                {"seed": 1, "final_money": 200.0},
            ]
        }

    def fake_summarize_detailed(**kwargs):
        return {
            "overall": {"success_rate": 0.5},
            "no_sandstorm": {"success_rate": 0.5},
        }

    monkeypatch.setattr(benchmark, "evaluate_model", fake_eval_detailed)
    monkeypatch.setattr(benchmark, "_summarize_detailed", fake_summarize_detailed)
    monkeypatch.setattr(benchmark, "_print_summary", lambda title, summary: None)

    out_path = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "dummy.pt",
            "--level",
            "3",
            "--runs",
            "2",
            "--weather-modes",
            "no_sandstorm",
            "--output-json",
            str(out_path),
        ],
    )

    benchmark.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["level"] == 3
    assert payload["runs_per_mode"] == 2
    assert payload["weather_modes"] == ["no_sandstorm"]
    assert "model" in payload


def test_benchmark_main_min_success_rate_failure(monkeypatch):
    def fake_eval_detailed(**kwargs):
        return {"no_sandstorm": [{"seed": 0, "final_money": 0.0}]}

    def fake_summarize_detailed(**kwargs):
        return {"overall": {"success_rate": 0.1}}

    monkeypatch.setattr(benchmark, "evaluate_model", fake_eval_detailed)
    monkeypatch.setattr(benchmark, "_summarize_detailed", fake_summarize_detailed)
    monkeypatch.setattr(benchmark, "_print_summary", lambda title, summary: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "dummy.pt",
            "--level",
            "3",
            "--runs",
            "1",
            "--weather-modes",
            "no_sandstorm",
            "--min-success-rate",
            "0.3",
        ],
    )

    with pytest.raises(SystemExit):
        benchmark.main()


def test_benchmark_main_invalid_weather_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "dummy.pt",
            "--level",
            "3",
            "--runs",
            "1",
            "--weather-modes",
            "bad_mode",
        ],
    )

    with pytest.raises(SystemExit):
        benchmark.main()
