from importlib import util
from pathlib import Path


def _load_module(path: Path):
    spec = util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Failed to load module: {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rules_audit_mapping_exists():
    base_dir = Path(__file__).resolve().parent

    rules = {
        "test_environment_rules_extra.py": [
            "test_day0_purchase_only_once",
            "test_timeout_terminates_when_not_reached",
            "test_valid_actions_respects_weight_limit_on_day0",
            "test_step_terminates_when_resources_insufficient",
            "test_weather_sequence_same_seed_is_deterministic",
            "test_observation_does_not_include_future_weather",
            "test_weather_revealed_daily_matches_sequence",
            "test_weather_sequence_deterministic_across_steps_with_seed",
            "test_observation_consistency_with_state_features",
            "test_weather_sequence_values_and_length",
            "test_sandstorm_move_attempt_forces_stay_and_penalty",
            "test_non_adjacent_move_is_blocked",
            "test_final_day_sandstorm_blocks_reaching_end",
            "test_arrival_on_deadline_day_is_allowed",
            "test_insufficient_resources_fail_even_if_buy_attempted",
            "test_reached_end_state_is_frozen",
            "test_failed_state_is_frozen",
            "test_hot_weather_consumption_on_stay",
            "test_hot_weather_consumption_on_move",
            "test_sunny_and_sandstorm_consumption_on_stay",
            "test_reach_end_refund_matches_remaining_resources",
            "test_hot_weather_consumption_on_mine",
            "test_sandstorm_mining_allowed_when_staying_on_mine",
            "test_sandstorm_mining_consumption_on_mine",
            "test_village_purchase_after_arrival",
            "test_village_purchase_on_arrival_from_move",
            "test_reset_seed_supports_stochastic_evaluation_interface",
            "test_village_purchase_blocked_when_overweight",
            "test_village_purchase_blocked_when_money_insufficient",
        ],
        "test_validator_negative_more.py": [
            "test_validator_flags_resource_mismatch",
            "test_validator_flags_non_adjacent_move",
            "test_validator_flags_end_refund_mismatch",
            "test_validator_flags_mine_on_arrival",
            "test_validator_flags_wrong_village_price",
            "test_validator_flags_sandstorm_mining_consumption",
        ],
    }

    for filename, expected_tests in rules.items():
        module_path = base_dir / filename
        assert module_path.exists()
        module = _load_module(module_path)
        for name in expected_tests:
            assert hasattr(module, name), f"Missing rule test: {filename}:{name}"
