import numpy as np

from src.env.config import BASE_CONSUMPTION, Weather
from src.models.belief import (
    AdaptiveBeliefModel,
    WeatherBeliefModel,
    extract_belief_features,
)


def test_belief_update_sets_prediction_row():
    model = WeatherBeliefModel(use_dirichlet_prior=False)
    model.update(Weather.SUNNY)

    probs = model.current_belief.probs
    expected = model.transition_matrix[Weather.SUNNY]

    assert np.allclose(probs, expected)
    assert abs(probs.sum() - 1.0) < 1e-6


def test_belief_vector_and_features_shapes_and_ranges():
    model = WeatherBeliefModel(use_dirichlet_prior=False)
    model.update(Weather.HOT)

    vec = model.get_belief_vector()
    feats = extract_belief_features(model)

    assert vec.shape == (4,)
    assert feats.shape == (6,)
    assert 0.0 <= feats[4] <= 1.0  # normalized entropy


def test_expected_consumption_matches_weighted_average():
    model = WeatherBeliefModel(use_dirichlet_prior=False)
    model.current_belief.probs = np.array([0.2, 0.3, 0.5])

    exp_w, exp_f = model.expected_consumption(Weather.SUNNY, BASE_CONSUMPTION)
    expected_w = (
        0.2 * BASE_CONSUMPTION[0][0]
        + 0.3 * BASE_CONSUMPTION[1][0]
        + 0.5 * BASE_CONSUMPTION[2][0]
    )
    expected_f = (
        0.2 * BASE_CONSUMPTION[0][1]
        + 0.3 * BASE_CONSUMPTION[1][1]
        + 0.5 * BASE_CONSUMPTION[2][1]
    )

    assert abs(exp_w - expected_w) < 1e-6
    assert abs(exp_f - expected_f) < 1e-6


def test_risk_measure_branches():
    model = WeatherBeliefModel(use_dirichlet_prior=False)

    model.current_belief.probs = np.array([0.2, 0.2, 0.6])
    assert model.get_risk_measure("worst_case") == Weather.SANDSTORM

    model.current_belief.probs = np.array([0.2, 0.6, 0.2])
    assert model.get_risk_measure("worst_case") == Weather.HOT

    model.current_belief.probs = np.array([0.7, 0.2, 0.1])
    assert model.get_risk_measure("worst_case") == Weather.SUNNY

    model.current_belief.probs = np.array([0.1, 0.2, 0.7])
    assert model.get_risk_measure("expected") == Weather.SANDSTORM


def test_adaptive_belief_model_detects_change():
    model = AdaptiveBeliefModel(window_size=4, adaptation_threshold=0.01)

    # drive recent transitions to create a noticeable diff
    seq = [Weather.SUNNY, Weather.HOT, Weather.SUNNY, Weather.HOT, Weather.SUNNY]
    for w in seq:
        model.update(w)

    assert model.learning_rate >= 0.05
    for w in seq:
        model.update(w)

    assert model.learning_rate >= 0.05
