import src.pipeline.evaluate as evaluate
import src.pipeline.rollout as rollout


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

    monkeypatch.setattr(evaluate.Agent, "load", staticmethod(fake_load))
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

    monkeypatch.setattr(rollout.Agent, "load", staticmethod(fake_load))
    monkeypatch.setattr(rollout, "run_episode", fake_run_episode)

    result = rollout.rollout_majority_voting("dummy.pt", level=3, episodes=5, device="cpu")

    assert result["reached"] is True
    assert result["final_money"] == 1004.0


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

    monkeypatch.setattr(rollout.Agent, "load", staticmethod(fake_load))
    monkeypatch.setattr(rollout, "run_episode", fake_run_episode)

    result = rollout.rollout_majority_voting("dummy.pt", level=3, device="cpu")

    assert result["reached"] is False
    assert result["return"] == 19.0
