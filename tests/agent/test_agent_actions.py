import torch

from src.env.environment import make_env
from src.models.agent import create_agent


def test_epsilon_exploration_buys_on_day0_when_can_buy():
    env = make_env(level=3, seed=0)
    obs, _ = env.reset(seed=0)
    agent = create_agent(env, device="cpu")

    va = env.get_valid_actions()
    action, _ = agent.select_action(obs, va, deterministic=False, epsilon=1.0)

    assert va["can_buy"] is True
    assert action["buy_water"] > 0 or action["buy_food"] > 0


def test_buy_logprob_masked_when_cannot_buy():
    env = make_env(level=3, seed=0)
    obs, _ = env.reset(seed=0)
    agent = create_agent(env, device="cpu")

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    state = agent.encode_observation(obs_tensor)

    valid_no_buy = {
        "valid_moves": [env.state.position],
        "can_mine": False,
        "can_buy": False,
        "max_buy_water": 0,
        "max_buy_food": 0,
    }

    actions_a = {
        "move": torch.tensor([env.state.position], dtype=torch.long),
        "mine": torch.tensor([0.0]),
        "buy_water": torch.tensor([0.0]),
        "buy_food": torch.tensor([0.0]),
    }
    actions_b = {
        "move": torch.tensor([env.state.position], dtype=torch.long),
        "mine": torch.tensor([0.0]),
        "buy_water": torch.tensor([99.0]),
        "buy_food": torch.tensor([99.0]),
    }

    logp_a, _ = agent.actor.evaluate_actions(state, actions_a, valid_no_buy)
    logp_b, _ = agent.actor.evaluate_actions(state, actions_b, valid_no_buy)

    assert torch.allclose(logp_a, logp_b, atol=1e-6)
