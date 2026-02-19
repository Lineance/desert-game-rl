from pathlib import Path

import src.pipeline.evaluate as evaluate


class _FakeAgent:
    def eval(self):
        return None


def test_evaluate_multiple_runs_uses_incremental_seeds(monkeypatch):
    called_seeds = []

    def fake_load(path, device="cpu"):
        return _FakeAgent()

    def fake_run_episode(agent, env, seed=None, deterministic=True, verbose=False):
        called_seeds.append(seed)
        return {
            "reached": (seed % 2 == 0),
            "final_money": float(100 + seed),
            "length": 5,
            "return": float(seed),
            "path": [1, 2, 3],
        }

    monkeypatch.setattr(evaluate.HybridRNNAgent, "load", staticmethod(fake_load))
    monkeypatch.setattr(evaluate, "run_episode", fake_run_episode)

    results = evaluate.evaluate_multiple_runs("dummy.pt", level=3, num_runs=4, device="cpu")

    assert len(results) == 4
    assert called_seeds == [0, 1, 2, 3]


def test_generate_result_excel_selects_best_reached(monkeypatch, tmp_path):
    def fake_load(path, device="cpu"):
        return _FakeAgent()

    def fake_run_episode(agent, env, seed=None, deterministic=True, verbose=False):
        return {
            "records": [],
            "reached": seed in {1, 4},
            "final_money": 1000.0 + float(seed),
            "final_water": 0,
            "final_food": 0,
            "length": 5,
            "return": float(seed),
            "path": [1, 2, 3],
        }

    captured = {}

    def fake_export(result, filepath, level):
        captured["result"] = result
        captured["filepath"] = filepath
        captured["level"] = level

    monkeypatch.setattr(evaluate.HybridRNNAgent, "load", staticmethod(fake_load))
    monkeypatch.setattr(evaluate, "run_episode", fake_run_episode)
    monkeypatch.setattr(evaluate, "export_to_xlsx", fake_export)
    monkeypatch.setattr(evaluate, "analyze_strategy", lambda result, env: None)

    result = evaluate.generate_result_excel(
        "dummy.pt", level=3, output_dir=str(tmp_path), device="cpu"
    )

    assert result["reached"] is True
    assert result["final_money"] == 1004.0
    assert captured["level"] == 3
    assert Path(captured["filepath"]).name == "Result_第三关.xlsx"


def test_generate_result_excel_fallback_when_no_reached(monkeypatch, tmp_path):
    def fake_load(path, device="cpu"):
        return _FakeAgent()

    def fake_run_episode(agent, env, seed=None, deterministic=True, verbose=False):
        return {
            "records": [],
            "reached": False,
            "final_money": 900.0,
            "final_water": 0,
            "final_food": 0,
            "length": 5,
            "return": float(seed),
            "path": [1, 2, 3],
        }

    monkeypatch.setattr(evaluate.HybridRNNAgent, "load", staticmethod(fake_load))
    monkeypatch.setattr(evaluate, "run_episode", fake_run_episode)
    monkeypatch.setattr(evaluate, "export_to_xlsx", lambda result, filepath, level: None)
    monkeypatch.setattr(evaluate, "analyze_strategy", lambda result, env: None)

    result = evaluate.generate_result_excel(
        "dummy.pt", level=3, output_dir=str(tmp_path), device="cpu"
    )

    assert result["reached"] is False
    assert result["return"] == 19.0
