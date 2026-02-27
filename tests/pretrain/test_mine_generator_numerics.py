import torch

from src.models.policy.generators.mine_generator import MineGenerator


def test_mine_generator_evaluate_binary_flags_is_finite():
    torch.manual_seed(0)
    generator = MineGenerator(state_dim=128)
    state = torch.randn(1, 128)

    out_zero = generator.evaluate(state, torch.tensor([0.0]), can_mine=True)
    out_one = generator.evaluate(state, torch.tensor([1.0]), can_mine=True)

    assert torch.isfinite(out_zero["log_prob"]).all()
    assert torch.isfinite(out_one["log_prob"]).all()
    assert torch.isfinite(out_zero["entropy"]).all()
    assert torch.isfinite(out_one["entropy"]).all()
