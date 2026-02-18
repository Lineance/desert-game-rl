from src.env.config import RLConfig
from src.env.environment import make_env
from src.models.agent import create_agent
from src.models.ppo import PPOTrainer


def test_collect_rollout_and_update_smoke():
    env = make_env(level=3, seed=0)
    agent = create_agent(env, RLConfig(), device="cpu")
    trainer = PPOTrainer(agent, RLConfig(), device="cpu")

    buffer, last_value, info = trainer.collect_rollout(env, max_steps=5, epsilon=0.2)

    assert isinstance(buffer.rewards, list)
    assert isinstance(last_value, float)
    assert "reached" in info

    if buffer.rewards:
        stats = trainer.update(buffer, last_value)
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats
        assert "approx_kl" in stats
