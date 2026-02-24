import sys

import src.pipeline.evaluate as evaluate
import src.pipeline.rollout as rollout


class _FakeAgent:
    def eval(self):
        return None


def test_evaluate_main_runs_single_episode(monkeypatch, tmp_path):
    captured = {}

    def fake_load(path, device="cpu"):
        return _FakeAgent()

    def fake_run_episode(agent, env, seed=None, deterministic=True, verbose=False):
        return {
            "records": [],
            "return": 0.0,
            "reached": True,
            "final_money": 100.0,
            "final_water": 0,
            "final_food": 0,
            "length": 1,
            "path": [1],
        }

    def fake_export(result, filepath, level):
        captured["filepath"] = filepath
        captured["level"] = level

    monkeypatch.setattr(rollout.Agent, "load", staticmethod(fake_load))
    monkeypatch.setattr(rollout, "run_episode", fake_run_episode)
    monkeypatch.setattr(rollout, "_export_to_xlsx", fake_export)
    monkeypatch.setattr(rollout, "_analyze_strategy", lambda result, env: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "dummy.pt",
            "--level",
            "3",
            "--output",
            str(tmp_path),
            "--seed",
            "7",
        ],
    )

    rollout.rollout_main()

    assert captured["level"] == 3
    assert str(tmp_path) in captured["filepath"]
    assert captured["filepath"].endswith("level3_result.xlsx")


def test_evaluate_main_runs_multi_episode(monkeypatch):
    called = {}

    def fake_evaluate_multiple_runs(agent_path, level, num_runs, device):
        called["agent_path"] = agent_path
        called["level"] = level
        called["num_runs"] = num_runs
        called["device"] = device

    monkeypatch.setattr(evaluate, "evaluate_multiple_runs", fake_evaluate_multiple_runs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "dummy.pt",
            "--level",
            "4",
            "--runs",
            "12",
            "--device",
            "cpu",
        ],
    )

    evaluate.evaluate_main()

    assert called == {
        "agent_path": "dummy.pt",
        "level": 4,
        "num_runs": 12,
        "device": "cpu",
    }
