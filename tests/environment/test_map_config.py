import numpy as np
import pytest

from src.env.config import (
    Level3Config,
    Level4Config,
    Level35Config,
    Weather,
    compute_shortest_distances,
    get_adjacency_matrix,
)


def _validate_map_config(config_cls):
    num_nodes = int(config_cls.NUM_NODES)
    assert num_nodes > 1

    start = int(config_cls.START)
    end = int(config_cls.END)
    assert 0 <= start < num_nodes
    assert 0 <= end < num_nodes

    edges = list(config_cls.EDGES)
    assert edges, "EDGES cannot be empty"

    seen = set()
    used_nodes = {start, end}
    used_nodes.update(int(n) for n in config_cls.MINES)
    used_nodes.update(int(n) for n in config_cls.VILLAGES)

    for u, v in edges:
        assert 1 <= int(u) <= num_nodes
        assert 1 <= int(v) <= num_nodes
        assert int(u) != int(v)
        edge = tuple(sorted((int(u), int(v))))
        assert edge not in seen
        seen.add(edge)
        used_nodes.add(int(u) - 1)
        used_nodes.add(int(v) - 1)

    conn = get_adjacency_matrix(num_nodes, edges)
    distances = compute_shortest_distances(conn, start)

    assert distances[end] >= 0
    for node in sorted(used_nodes):
        assert 0 <= node < num_nodes
        assert distances[node] >= 0

    # Tightened validation: all nodes are reachable from start.
    assert all(dist >= 0 for dist in distances)


def _validate_weather_config(config_cls):
    modes = config_cls.WEATHER_MODES
    assert isinstance(modes, dict)
    assert modes, "WEATHER_MODES cannot be empty"

    for key, probs in modes.items():
        assert len(probs) == 3, f"WEATHER_MODES[{key}] must have length 3"
        assert all(p >= 0 for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-6

    transition = config_cls.WEATHER_TRANSITION
    assert transition.shape == (3, 3)
    for row in transition:
        assert all(p >= 0 for p in row)
        assert abs(float(sum(row)) - 1.0) < 1e-6


def _assert_oracle_feasible(config_cls, time_limit=60, samples_per_mode=2):
    pytest.importorskip("pulp")
    from src.utils.oracle import solve_theoretical_optimal_with_config

    def _sample_weather_sequence(mode, rng):
        probs = config_cls.WEATHER_MODES[mode]
        transition = config_cls.WEATHER_TRANSITION
        num_days = int(config_cls.NUM_DAYS)
        sequence = []
        current = int(rng.choice(3, p=probs))
        for _ in range(num_days):
            sequence.append(current)
            current = int(rng.choice(3, p=transition[current]))
        return sequence

    rng = np.random.RandomState(0)

    sequences = [[Weather.SUNNY] * int(config_cls.NUM_DAYS)]
    for mode in sorted(config_cls.WEATHER_MODES.keys()):
        for _ in range(samples_per_mode):
            sequences.append(_sample_weather_sequence(mode, rng))

    for weather_seq in sequences:
        result = solve_theoretical_optimal_with_config(
            config_cls,
            weather_seq,
            time_limit=time_limit,
        )
        assert result["status"] != "Infeasible"
        assert result["reached"] is True


@pytest.mark.parametrize(
    "config_cls",
    [Level3Config, Level35Config, Level4Config],
)
@pytest.mark.slow
def test_map_configs_are_valid(config_cls):
    _validate_map_config(config_cls)
    _validate_weather_config(config_cls)
    _assert_oracle_feasible(config_cls)
