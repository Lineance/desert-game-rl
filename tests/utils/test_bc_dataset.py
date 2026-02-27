import json

import src.utils.bc_dataset as bc_dataset


class _DummyState:
    def __init__(self, weather_future):
        self.weather_future = weather_future


class _DummyEnv:
    def __init__(self):
        self.state = None

    def reset(self, seed=None):
        day0 = int(seed or 0)
        weather = [day0 % 3, (day0 + 1) % 3, (day0 + 2) % 3]
        self.state = _DummyState(weather)
        return None, {}


def test_generate_and_load_bc_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(bc_dataset, "_make_env", lambda **kwargs: _DummyEnv())

    def _fake_plan(level, weather_seq, time_limit=20):
        _ = level, time_limit
        return {
            "status": "Optimal",
            "reached": True,
            "plan": [{"move": 0, "mine": False, "buy_water": 0, "buy_food": 0}],
        }

    monkeypatch.setattr(bc_dataset, "_solve_theoretical_plan", _fake_plan)

    output_path = tmp_path / "bc_level3.json"
    summary = bc_dataset.generate_behavior_cloning_dataset(
        level=3,
        episodes=3,
        seed_start=0,
        output_path=str(output_path),
        include_unsolved=False,
    )

    assert output_path.exists()
    assert summary["episodes"] == 3
    assert summary["records"] == 3
    assert summary["solved"] == 3

    index = bc_dataset.load_behavior_cloning_dataset_index(output_path)
    assert len(index) == 3
    assert 0 in index
    assert isinstance(index[0]["plan"], list)


def test_generate_bc_dataset_filters_unsolved(tmp_path, monkeypatch):
    monkeypatch.setattr(bc_dataset, "_make_env", lambda **kwargs: _DummyEnv())

    def _fake_plan(level, weather_seq, time_limit=20):
        _ = level, time_limit
        if weather_seq[0] == 0:
            return {"status": "Optimal", "reached": True, "plan": [{"move": 0}]}
        return {"status": "Infeasible", "reached": False, "plan": []}

    monkeypatch.setattr(bc_dataset, "_solve_theoretical_plan", _fake_plan)

    output_path = tmp_path / "bc_level3_filtered.json"
    summary = bc_dataset.generate_behavior_cloning_dataset(
        level=3,
        episodes=4,
        output_path=str(output_path),
        include_unsolved=False,
    )

    with output_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert summary["episodes"] == 4
    assert summary["records"] < summary["episodes"]
    assert all(item.get("status") == "Optimal" for item in payload["records"])


def test_generate_bc_dataset_parallel_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(bc_dataset, "_make_env", lambda **kwargs: _DummyEnv())

    called = {"batch": 0}
    received = {"progress_every": None, "progress_prefix": None}

    def _fake_batch(
        level,
        weather_seqs,
        time_limit=20,
        return_plan=False,
        max_workers=None,
        config_cls=None,
        base_consumption=None,
        solver_config=None,
        solver_options=None,
        threads=None,
        progress_every=0,
        progress_prefix="Oracle batch",
    ):
        _ = (
            level,
            time_limit,
            return_plan,
            max_workers,
            config_cls,
            base_consumption,
            solver_config,
            solver_options,
            threads,
        )
        called["batch"] += 1
        received["progress_every"] = progress_every
        received["progress_prefix"] = progress_prefix
        return [
            {
                "status": "Optimal",
                "reached": True,
                "plan": [{"move": 0, "mine": False, "buy_water": 0, "buy_food": 0}],
            }
            for _ in weather_seqs
        ]

    monkeypatch.setattr(bc_dataset, "_solve_theoretical_batch", _fake_batch)

    output_path = tmp_path / "bc_parallel.json"
    summary = bc_dataset.generate_behavior_cloning_dataset(
        level=3,
        episodes=3,
        output_path=str(output_path),
        oracle_parallel_workers=2,
        oracle_solver_threads=1,
        show_progress=True,
        progress_interval=2,
    )

    assert called["batch"] == 1
    assert received["progress_every"] == 2
    assert received["progress_prefix"] == "BC Oracle"
    assert summary["records"] == 3
    assert output_path.exists()
