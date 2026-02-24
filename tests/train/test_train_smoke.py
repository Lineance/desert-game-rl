import random

import numpy as np
import torch

import src.pipeline.train as train


def test_train_smoke_writes_metrics_and_checkpoint(tmp_path, monkeypatch):
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
        num_episodes=1,
        device="cpu",
        log_interval=50,
        resume=None,
        checkpoint_interval=0,
    )

    metrics_path = results_dir / "level3_train_metrics.csv"
    checkpoint_path = checkpoints_dir / "level3_latest.pt"
    final_path = checkpoints_dir / "level3_final.pt"

    assert metrics_path.exists()
    assert checkpoint_path.exists()
    assert final_path.exists()
