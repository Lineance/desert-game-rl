from pathlib import Path

from src.pipeline import pretrain as pretrain_mod


class _DummyAgent:
    def __init__(self):
        self.config = object()
        self.saved_paths = []

    def save(self, path: str) -> None:
        self.saved_paths.append(path)


class _DummyTrainer:
    def __init__(self, agent, config, device="cpu"):
        self.agent = agent
        self.config = config
        self.device = device


def test_pretrain_save_interval(monkeypatch, tmp_path):
    dummy_agent = _DummyAgent()

    monkeypatch.setattr(pretrain_mod, "make_env", lambda **kwargs: object())
    monkeypatch.setattr(pretrain_mod, "create_agent", lambda env, config, device: dummy_agent)
    monkeypatch.setattr(pretrain_mod, "PPOTrainer", _DummyTrainer)

    def _fake_warmup(*args, **kwargs):
        callback = kwargs.get("on_episode_end")
        for ep in range(5):
            if callback is not None:
                callback(ep, {"episode": float(ep)})
        return {
            "success_rate": 1.0,
            "avg_steps": 7.0,
            "avg_mine_per_episode": 3.0,
            "avg_buy_water": 130.0,
            "avg_buy_food": 110.0,
            "avg_value_loss": 100.0,
            "final_value_loss": 90.0,
            "match_rate": 0.5,
        }

    monkeypatch.setattr(pretrain_mod, "behavior_cloning_with_oracle", _fake_warmup)

    checkpoints_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(pretrain_mod, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(pretrain_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(pretrain_mod, "LOGS_DIR", logs_dir)

    output_path = tmp_path / "final_pretrained.pt"
    pretrain_mod.pretrain_behavior_cloning(
        level=3,
        warmup_episodes=5,
        device="cpu",
        oracle_time_limit=1,
        output_path=str(output_path),
        save_interval=2,
    )

    saved_names = [Path(p).name for p in dummy_agent.saved_paths]
    assert "level3_pretrain_episode2.pt" in saved_names
    assert "level3_pretrain_episode4.pt" in saved_names
    assert output_path.name in saved_names


def test_pretrain_save_interval_disabled(monkeypatch, tmp_path):
    dummy_agent = _DummyAgent()

    monkeypatch.setattr(pretrain_mod, "make_env", lambda **kwargs: object())
    monkeypatch.setattr(pretrain_mod, "create_agent", lambda env, config, device: dummy_agent)
    monkeypatch.setattr(pretrain_mod, "PPOTrainer", _DummyTrainer)

    monkeypatch.setattr(
        pretrain_mod,
        "behavior_cloning_with_oracle",
        lambda *args, **kwargs: {
            "success_rate": 1.0,
            "avg_steps": 7.0,
            "avg_mine_per_episode": 3.0,
            "avg_buy_water": 130.0,
            "avg_buy_food": 110.0,
            "avg_value_loss": 100.0,
            "final_value_loss": 90.0,
            "match_rate": 0.5,
        },
    )

    checkpoints_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(pretrain_mod, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(pretrain_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(pretrain_mod, "LOGS_DIR", logs_dir)

    output_path = tmp_path / "final_pretrained.pt"
    pretrain_mod.pretrain_behavior_cloning(
        level=3,
        warmup_episodes=1,
        device="cpu",
        oracle_time_limit=1,
        output_path=str(output_path),
        save_interval=0,
    )

    saved_names = [Path(p).name for p in dummy_agent.saved_paths]
    assert saved_names == [output_path.name]
