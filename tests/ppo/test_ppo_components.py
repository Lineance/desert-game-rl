import numpy as np
import torch

from src.models.agent import ActorNetwork
from src.models.ppo import RolloutBuffer


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

    valid_actions = {
        "valid_moves": [1, 3],
        "can_mine": False,
        "can_buy": False,
        "max_buy_water": 0,
        "max_buy_food": 0,
    }

    dist = actor(state, valid_actions)
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

    valid_actions = {
        "valid_moves": [0, 1],
        "can_mine": True,
        "can_buy": False,
        "max_buy_water": 0,
        "max_buy_food": 0,
    }

    action, _ = actor.sample_action(state, valid_actions)

    assert action["buy_water"] == 0
    assert action["buy_food"] == 0
