import numpy as np
import torch
import torch.nn.functional as F

from src.env.environment import make_env
from src.models.agent import create_agent
from src.models.ppo import PPOTrainer


def _clone_obs_with_weather(obs: np.ndarray, weather_idx: int) -> np.ndarray:
    cloned = np.array(obs, dtype=np.float32, copy=True)
    cloned[6:9] = 0.0
    cloned[6 + weather_idx] = 1.0
    return cloned


def _run_sequence_values(agent, obs_seq, valid_seq):
    device = next(agent.parameters()).device
    agent.reset_hidden(batch_size=1, device=device)
    values = []
    for obs, valid_actions in zip(obs_seq, valid_seq):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        out = agent.forward_step(obs_tensor, valid_actions, update_internal_hidden=True)
        values.append(out["value"].detach().cpu().clone())
    return values


def test_lstm_state_reset():
    env = make_env(level=3, seed=0)
    obs, _ = env.reset(seed=0)
    agent = create_agent(env, device="cpu")

    valid_actions = env.get_valid_actions()
    agent.reset_hidden(batch_size=1, device=torch.device("cpu"))

    for _ in range(5):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        _ = agent.forward_step(obs_tensor, valid_actions, update_internal_hidden=True)

    h_end, c_end = agent.hidden_state
    h_end = h_end.clone()
    c_end = c_end.clone()

    agent.reset_hidden(batch_size=1, device=torch.device("cpu"))
    h_new, c_new = agent.hidden_state

    assert torch.allclose(h_new, torch.zeros_like(h_new))
    assert torch.allclose(c_new, torch.zeros_like(c_new))
    assert not torch.allclose(h_new, h_end)
    assert not torch.allclose(c_new, c_end)


def test_temporal_dependency():
    torch.manual_seed(7)
    np.random.seed(7)

    env = make_env(level=3, seed=1)
    obs, _ = env.reset(seed=1)
    agent = create_agent(env, device="cpu")
    agent.eval()

    obs_common = np.array(obs, dtype=np.float32)
    obs_sunny = _clone_obs_with_weather(obs_common, weather_idx=0)
    obs_storm = _clone_obs_with_weather(obs_common, weather_idx=2)

    # 放大首日天气差异，避免随机初始化下价值头“压平”差异导致误报
    obs_sunny[6:9] = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    obs_storm[6:9] = np.array([0.0, 0.0, 3.0], dtype=np.float32)

    seq_a = [obs_sunny] + [obs_common] * 8
    seq_b = [obs_storm] + [obs_common] * 8

    obs_seq_a = torch.tensor(np.stack(seq_a, axis=0), dtype=torch.float32).unsqueeze(0)
    obs_seq_b = torch.tensor(np.stack(seq_b, axis=0), dtype=torch.float32).unsqueeze(0)

    agent.reset_hidden(batch_size=1, device=torch.device("cpu"))
    feat_a, hid_a = agent.encode_sequence(obs_seq_a)
    agent.reset_hidden(batch_size=1, device=torch.device("cpu"))
    feat_b, hid_b = agent.encode_sequence(obs_seq_b)

    last_feat_a = feat_a[:, -1, :]
    last_feat_b = feat_b[:, -1, :]

    # 关键断言：历史首步不同应影响末步时序特征/hidden（记忆生效）
    assert not torch.allclose(last_feat_a, last_feat_b, atol=1e-6)
    assert not torch.allclose(hid_a[0], hid_b[0], atol=1e-6)

    valid_seq = [
        {
            "valid_moves": list(range(env.config.NUM_NODES)),
            "can_mine": False,
            "can_buy": False,
            "max_buy_water": 0,
            "max_buy_food": 0,
        }
        for _ in range(9)
    ]

    vals_a = _run_sequence_values(agent, seq_a, valid_seq)
    vals_b = _run_sequence_values(agent, seq_b, valid_seq)

    assert not torch.allclose(vals_a[-1], vals_b[-1], atol=1e-6)


def test_gradient_flow_through_time():
    env = make_env(level=3, seed=2)
    agent = create_agent(env, device="cpu")
    agent.train()

    obs_dim = agent.obs_dim
    obs_seq = torch.randn(1, 10, obs_dim, requires_grad=True)

    agent.reset_hidden(batch_size=1, device=torch.device("cpu"))
    features, _ = agent.encode_sequence(obs_seq)
    values = agent.critic(features)
    loss = values.sum()
    loss.backward()

    assert obs_seq.grad is not None
    assert obs_seq.grad.abs().sum().item() > 0


def test_can_overfit_single_episode():
    torch.manual_seed(0)
    np.random.seed(0)

    env = make_env(level=3, seed=3)
    obs, _ = env.reset(seed=3)
    agent = create_agent(env, device="cpu")
    agent.train()

    seq_len = 10
    obs_seq_np = np.stack([np.array(obs, dtype=np.float32) for _ in range(seq_len)], axis=0)
    obs_seq = torch.tensor(obs_seq_np, dtype=torch.float32).unsqueeze(0)

    valid_actions_seq = [
        {
            "valid_moves": list(range(env.config.NUM_NODES)),
            "can_mine": True,
            "can_buy": False,
            "max_buy_water": 0,
            "max_buy_food": 0,
        }
        for _ in range(seq_len)
    ]

    target_actions = {
        "move": torch.zeros(seq_len, dtype=torch.long),
        "mine": torch.zeros(seq_len, dtype=torch.float32),
        "buy_water": torch.zeros(seq_len, dtype=torch.float32),
        "buy_food": torch.zeros(seq_len, dtype=torch.float32),
    }

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)

    with torch.no_grad():
        init_logp, _, init_values, _ = agent.evaluate_actions_sequence(
            obs_seq, target_actions, valid_actions_seq=valid_actions_seq, hidden_state=None
        )
        init_loss = float(
            (
                -init_logp.mean() + 0.1 * F.mse_loss(init_values, torch.zeros_like(init_values))
            ).item()
        )

    final_loss = init_loss
    for _ in range(120):
        optimizer.zero_grad()
        logp, _, values, _ = agent.evaluate_actions_sequence(
            obs_seq, target_actions, valid_actions_seq=valid_actions_seq, hidden_state=None
        )
        loss = -logp.mean() + 0.1 * F.mse_loss(values, torch.zeros_like(values))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = float(loss.item())

    assert final_loss < init_loss * 0.2
    assert final_loss < 0.1


def test_hidden_state_reset_on_done():
    """验证 done 后 hidden state 应重置为全零且与上一段计算图断开。"""
    env = make_env(level=3, seed=11)
    obs, _ = env.reset(seed=11)
    agent = create_agent(env, device="cpu")

    valid_actions = env.get_valid_actions()
    agent.reset_hidden(batch_size=1, device=torch.device("cpu"))

    prev_h = None
    prev_c = None
    for _ in range(5):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        _ = agent.forward_step(obs_t, valid_actions, update_internal_hidden=True)
        prev_h, prev_c = agent.hidden_state

        # 模拟 episode 终止：应立即重置 hidden
        agent.reset_hidden(batch_size=1, device=torch.device("cpu"))

    h, c = agent.hidden_state
    assert torch.allclose(h, torch.zeros_like(h))
    assert torch.allclose(c, torch.zeros_like(c))
    assert h.grad_fn is None
    assert c.grad_fn is None

    # 重置后的状态不应沿用上一段末状态
    assert prev_h is not None and prev_c is not None
    assert not torch.allclose(h, prev_h)
    assert not torch.allclose(c, prev_c)


def test_rollout_buffer_with_lstm_hidden_states():
    """验证 collect_rollout 存储的 hidden_state 形状正确，且可直接用于 PPO update。"""
    env = make_env(level=3, seed=12)
    agent = create_agent(env, device="cpu")
    trainer = PPOTrainer(agent=agent, device="cpu")

    rollout_buffer, last_value, _ = trainer.collect_rollout(env, max_steps=20, epsilon=0.0)

    assert len(rollout_buffer.observations) > 0
    assert len(rollout_buffer.hidden_states) == len(rollout_buffer.observations)

    expected_shape = (int(agent.config.LSTM_LAYERS), 1, int(agent.lstm_hidden_dim))
    for hidden_state in rollout_buffer.hidden_states:
        assert hidden_state is not None
        h_np, c_np = hidden_state
        assert tuple(h_np.shape) == expected_shape
        assert tuple(c_np.shape) == expected_shape
        assert np.isfinite(h_np).all()
        assert np.isfinite(c_np).all()

    segments = rollout_buffer.build_episode_segments()
    assert len(segments) >= 1
    first_hidden = segments[0]["hidden_state"]
    assert first_hidden is not None
    assert np.allclose(first_hidden[0], np.zeros_like(first_hidden[0]))
    assert np.allclose(first_hidden[1], np.zeros_like(first_hidden[1]))

    stats = trainer.update(rollout_buffer, last_value)
    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])
