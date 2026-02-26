import torch

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


def test_resolve_resume_path_latest(monkeypatch, tmp_path):
    monkeypatch.setattr(pretrain_mod, "CHECKPOINTS_DIR", tmp_path)
    resolved = pretrain_mod._resolve_resume_path("latest", level=35)
    assert resolved == tmp_path / "level35_pretrain_latest.pt"


def test_infer_resume_episode_from_filename_and_checkpoint(tmp_path):
    by_name = pretrain_mod._infer_resume_episode(tmp_path / "level35_pretrain_episode500.pt", None)
    assert by_name == 500

    by_checkpoint = pretrain_mod._infer_resume_episode(
        tmp_path / "some_checkpoint.pt",
        {"episode": 123},
    )
    assert by_checkpoint == 124


def test_pretrain_resume_passes_start_episode(monkeypatch, tmp_path):
    dummy_agent = _DummyAgent()

    monkeypatch.setattr(pretrain_mod, "make_env", lambda **kwargs: object())
    monkeypatch.setattr(pretrain_mod, "create_agent", lambda env, config, device: dummy_agent)
    monkeypatch.setattr(pretrain_mod, "PPOTrainer", _DummyTrainer)
    monkeypatch.setattr(
        pretrain_mod,
        "_load_pretrain_agent_from_checkpoint",
        lambda path, device: dummy_agent,
    )

    captured = {"start_episode": None}

    def _fake_warmup(*args, **kwargs):
        captured["start_episode"] = kwargs.get("start_episode")
        return {
            "success_rate": 1.0,
            "avg_steps": 7.0,
            "avg_mine_per_episode": 3.0,
            "avg_buy_water": 130.0,
            "avg_buy_food": 110.0,
            "avg_value_loss": 100.0,
            "final_value_loss": 90.0,
            "match_rate": 0.5,
            "value_loss_trend": -10.0,
        }

    monkeypatch.setattr(pretrain_mod, "behavior_cloning_with_oracle", _fake_warmup)

    checkpoints_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(pretrain_mod, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(pretrain_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(pretrain_mod, "LOGS_DIR", logs_dir)

    resume_path = tmp_path / "resume.pt"
    torch.save({"episode": 4}, resume_path)

    pretrain_mod.pretrain_behavior_cloning(
        level=3,
        warmup_episodes=2,
        device="cpu",
        oracle_time_limit=1,
        output_path=str(tmp_path / "final_pretrained.pt"),
        save_interval=0,
        resume=str(resume_path),
    )

    assert captured["start_episode"] == 5
