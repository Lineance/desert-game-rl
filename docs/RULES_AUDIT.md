# Rules Audit (Task2 - Level 3/4)

This document maps problem rules to concrete automated tests. It is intended to verify that the game logic in third/fourth levels follows the problem statement.

## Scope

- Rule coverage focuses on environment behavior and validator correctness.
- Training quality and model performance are out of scope.

## Rule-to-Test Mapping

- Rule 1 (day 0 start, timeline, deadline):
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_day0_purchase_only_once`
    - `test_timeout_terminates_when_not_reached`
    - `test_failed_state_is_frozen`
  - [tests/test_validator_negative_more.py](tests/test_validator_negative_more.py)
    - `test_validator_flags_end_refund_mismatch` (end-state accounting)

- Rule 2 (resource limits, failure when exhausted):
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_valid_actions_respects_weight_limit_on_day0`
    - `test_step_terminates_when_resources_insufficient`
    - `test_insufficient_resources_fail_even_if_buy_attempted`
    - `test_village_purchase_blocked_when_overweight`
  - [tests/test_validator_negative_more.py](tests/test_validator_negative_more.py)
    - `test_validator_flags_resource_mismatch`

- Rule 3 (weather is global, today known):
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_weather_sequence_same_seed_is_deterministic`
    - `test_observation_does_not_include_future_weather`
    - `test_weather_revealed_daily_matches_sequence`
    - `test_weather_sequence_deterministic_across_steps_with_seed`
    - `test_observation_consistency_with_state_features`
    - `test_weather_sequence_values_and_length`

- Rule 4 (movement rules, sandstorm must stay):
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_sandstorm_move_attempt_forces_stay_and_penalty`
    - `test_non_adjacent_move_is_blocked`
    - `test_final_day_sandstorm_blocks_reaching_end`
    - `test_arrival_on_deadline_day_is_allowed`
  - [tests/test_validator_negative_more.py](tests/test_validator_negative_more.py)
    - `test_validator_flags_non_adjacent_move`

- Rule 5 (consumption factors: stay=1x, move=2x):
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_hot_weather_consumption_on_stay`
    - `test_hot_weather_consumption_on_move`
    - `test_sunny_and_sandstorm_consumption_on_stay`

- Rule 6 (day 0 purchase at base price, refund half price at end):
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_day0_purchase_only_once`
    - `test_reach_end_refund_matches_remaining_resources`
    - `test_reached_end_state_is_frozen`
  - [tests/test_validator_negative_more.py](tests/test_validator_negative_more.py)
    - `test_validator_flags_end_refund_mismatch`

- Rule 7 (mining rules, 3x consumption, no mining on arrival, sandstorm allowed):
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_sandstorm_mining_allowed_when_staying_on_mine`
    - `test_hot_weather_consumption_on_mine`
    - `test_sandstorm_mining_consumption_on_mine`
  - [tests/test_validator_negative_more.py](tests/test_validator_negative_more.py)
    - `test_validator_flags_mine_on_arrival`
    - `test_validator_flags_sandstorm_mining_consumption`

- Rule 8 (village purchase at 2x price):
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_village_purchase_after_arrival`
    - `test_village_purchase_on_arrival_from_move`
    - `test_village_purchase_blocked_when_money_insufficient`
  - [tests/test_validator_negative_more.py](tests/test_validator_negative_more.py)
    - `test_validator_flags_wrong_village_price`

## Partial Observability (Question 2)

- Daily reveal and seed reproducibility (no future leakage):
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_weather_revealed_daily_matches_sequence`
    - `test_weather_sequence_deterministic_across_steps_with_seed`
    - `test_observation_does_not_include_future_weather`

- Stochastic evaluation interface:
  - [tests/test_environment_rules_extra.py](tests/test_environment_rules_extra.py)
    - `test_reset_seed_supports_stochastic_evaluation_interface`

## Audit Test Module

- [tests/test_rules_audit.py](tests/test_rules_audit.py)
  - Validates that each referenced test function exists.
