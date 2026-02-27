import random

import numpy as np
import torch

import src.pipeline.train as train


def test_train_auto_stages_can_switch_to_stage3_and_persist_context(tmp_path, monkeypatch):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    checkpoints_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"

    monkeypatch.setattr(train, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(train, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train, "LOGS_DIR", logs_dir)

    train.train(
        level=3,
        num_episodes=2,
        device="cpu",
        log_interval=50,
        resume=None,
        checkpoint_interval=0,
        auto_stages=True,
        stage2_min_episodes=1,
        stage2_success_threshold=0.0,
        stage2_entropy_threshold=10.0,
        stage2_alpha=0.9,
        stage3_alpha_start=0.9,
        stage3_alpha_end=0.3,
        stage3_alpha_decay_episodes=10,
    )

    latest = torch.load(checkpoints_dir / "level3_latest.pt", map_location="cpu", weights_only=False)
    stage_context = latest.get("stage_context")

    assert isinstance(stage_context, dict)
    assert stage_context.get("stage_name") in {"stage2", "stage3", "default"}
    assert "stage2_episodes_done" in stage_context
    assert "stage3_episodes_done" in stage_context
