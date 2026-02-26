import numpy as np
import torch

from src.env.config import RLConfig
from src.env.environment import make_env
from src.models.agent import ActorNetwork
from src.models.ppo import PPOTrainer, RolloutBuffer
from src.utils.training_utils import dynamic_alpha


def test_rolloutbuffer_compute_returns_and_advantages_simple():
    buffer = RolloutBuffer()
    for _ in range(3):
        buffer.add(obs=np.zeros(1), action={}, reward=1.0, value=0.0, log_prob=0.0, done=False)

    returns, advantages = buffer.compute_returns_and_advantages(
        last_value=0.0, gamma=1.0, gae_lambda=1.0
    )

    assert returns == [3.0, 2.0, 1.0]
    assert advantages == [3.0, 2.0, 1.0]


def test_actor_masking_respects_valid_moves_and_mine():
    torch.manual_seed(0)
    actor = ActorNetwork(state_dim=8, hidden_dim=16, num_locations=5)
    state = torch.zeros((1, 8))
    node_embeddings = torch.zeros((1, 5, 16))
    day_norm = torch.zeros((1, 1))

    valid_actions = {
        "valid_moves": [1, 3],
        "can_mine": False,
        "can_buy": False,
        "max_buy_water": 0,
        "max_buy_food": 0,
    }

    dist = actor(state, node_embeddings, day_norm, valid_actions)
    move_probs = dist["move_probs"].detach().cpu().numpy().ravel()

    assert move_probs[0] == 0.0
    assert move_probs[2] == 0.0
    assert move_probs[4] == 0.0
    assert move_probs[1] > 0.0
    assert move_probs[3] > 0.0

    mine_probs = dist["mine_probs"].detach().cpu().numpy().ravel()
    assert mine_probs[1] == 0.0
    assert mine_probs[0] == 1.0


def test_actor_sample_action_respects_no_buy():
    torch.manual_seed(0)
    actor = ActorNetwork(state_dim=8, hidden_dim=16, num_locations=3)
    state = torch.zeros((1, 8))
    node_embeddings = torch.zeros((1, 3, 16))
    day_norm = torch.zeros((1, 1))

    valid_actions = {
        "valid_moves": [0, 1],
        "can_mine": True,
        "can_buy": False,
        "max_buy_water": 0,
        "max_buy_food": 0,
    }

    action, _ = actor.sample_action(state, node_embeddings, day_norm, valid_actions)

    assert action["buy_water"] == 0
    assert action["buy_food"] == 0


def test_dynamic_alpha_threshold_rules():
    water = torch.tensor([0.1, 0.8, 0.4])
    food = torch.tensor([0.6, 0.8, 0.4])
    dist = torch.tensor([0.5, 0.1, 0.6])

    alpha = dynamic_alpha(water, food, dist)

    assert torch.allclose(alpha, torch.tensor([0.9, 0.1, 0.5]))


def test_ppo_update_reports_dynamic_alpha_stats():
    env = make_env(level=3, seed=0)
    from src.models.agent import create_agent

    agent = create_agent(env, RLConfig(), device="cpu")
    trainer = PPOTrainer(agent, RLConfig(), device="cpu")
    buffer, last_value, _ = trainer.collect_rollout(env, max_steps=5, epsilon=0.0)

    if buffer.rewards:
        stats = trainer.update(buffer, last_value)
        assert "alpha_mean" in stats
        assert "survival_value_loss" in stats
        assert "fund_value_loss" in stats
        assert 0.0 <= stats["alpha_mean"] <= 1.0
