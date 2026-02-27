import math

from src.env.config import RLConfig
from src.env.environment import make_env
from src.models.agent import create_agent
from src.models.ppo import PPOTrainer


def test_ppo_update_alpha_override_applied():
    env = make_env(level=3, seed=0)
    agent = create_agent(env, RLConfig(), device="cpu")
    trainer = PPOTrainer(agent, RLConfig(), device="cpu")

    buffer, last_value, _ = trainer.collect_rollout(env, max_steps=5, epsilon=0.0)
    if not buffer.rewards:
        return

    stats = trainer.update(buffer, last_value, alpha_override=0.77, stage_name="stage2")

    assert "alpha_mean" in stats
    assert "alpha_source" in stats
    assert math.isfinite(stats["alpha_mean"])
    assert abs(stats["alpha_mean"] - 0.77) < 1e-5
    assert stats["alpha_source"] == 1.0
