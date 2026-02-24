import src.pipeline.benchmark as benchmark


class _FakeAgent:
    def eval(self):
        return None


def test_get_weather_modes_level3_and_level4():
    modes3 = benchmark._get_weather_modes(3)
    modes4 = benchmark._get_weather_modes(4)

    assert "no_sandstorm" in modes3
    assert len(modes4) >= 1


def test_evaluate_model_detailed_adds_seed_and_mode(monkeypatch):
    def fake_load(path, device="cpu"):
        return _FakeAgent(), "cpu"

    def fake_run_episodes(
        agent,
        env,
        runs,
        deterministic,
        *,
        verbose=False,
        include_seed=False,
        progress_every=None,
        progress_fn=None,
    ):
        results = []
        for seed in range(runs):
            result = {
                "reached": True,
                "final_money": 1000.0 + seed,
                "return": 1.0,
                "length": 3,
            }
            if include_seed:
                result["seed"] = seed
            results.append(result)
        return results

    monkeypatch.setattr(benchmark, "load_agent_for_eval", fake_load)
    monkeypatch.setattr(benchmark, "run_episodes", fake_run_episodes)

    out = benchmark.evaluate_model(
        agent_path="dummy.pt",
        level=3,
        runs=3,
        device="cpu",
        weather_modes=["no_sandstorm"],
        deterministic=True,
    )

    assert "no_sandstorm" in out
    assert [r["seed"] for r in out["no_sandstorm"]] == [0, 1, 2]
    assert all(r["weather_mode"] == "no_sandstorm" for r in out["no_sandstorm"])


def test_print_summary_and_oracle_summary_smoke(capsys):
    summary = {
        "no_sandstorm": {
            "success_rate": 0.5,
            "avg_final_money": 9000.0,
            "avg_return": 1.0,
            "avg_length": 10.0,
            "early_fail_rate": 0.1,
        },
        "overall": {
            "success_rate": 0.5,
            "avg_final_money": 9000.0,
            "avg_return": 1.0,
            "avg_length": 10.0,
            "early_fail_rate": 0.1,
        },
    }
    benchmark._print_summary("模型策略评测", summary)

    oracle_summary = {
        "no_sandstorm": {"oracle_solved_rate": 1.0, "oracle_avg_objective": 1234.0},
        "overall": {"oracle_solved_rate": 1.0, "oracle_avg_objective": 1234.0},
    }
    benchmark._print_oracle_summary("Oracle上界评测", oracle_summary)

    comp_summary = {
        "no_sandstorm": {
            "oracle_solved_rate": 1.0,
            "avg_money_to_oracle_ratio": 0.8,
            "avg_oracle_gap": 200.0,
        },
        "overall": {
            "avg_money_to_oracle_ratio": 0.8,
            "avg_oracle_gap": 200.0,
        },
    }
    benchmark._print_oracle_summary("模型 vs Oracle 对比", comp_summary)

    out = capsys.readouterr().out
    assert "模型策略评测" in out
    assert "Oracle上界评测" in out
    assert "模型 vs Oracle 对比" in out
