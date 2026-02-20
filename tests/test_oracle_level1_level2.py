from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import pulp
import pytest

from src.env.config import Weather
from src.pipeline.oracle import solve_theoretical_optimal_with_config

REFERENCE_TIME_LIMIT = 300
ORACLE_TIME_LIMIT = 300
HIGH_PRECISION_OPTIONS = [
    "--mip_rel_gap",
    "0.00001",
    "--mip_abs_gap",
    "0.00001",
    "--primal_feasibility_tolerance",
    "1e-9",
    "--dual_feasibility_tolerance",
    "1e-9",
    "--random_seed",
    "42",
]


def _load_tool_config(name: str):
    base = Path(__file__).resolve().parents[1] / "tools"
    path = base / f"{name}.py"
    loader = SourceFileLoader(name, str(path))
    spec = spec_from_loader(name, loader)
    if spec is None:
        raise RuntimeError(f"Unable to load {path}")
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


def _build_config(module):
    if hasattr(module, "MINES"):
        mines = list(module.MINES)
    else:
        mines = [module.MINE]

    if hasattr(module, "VILLAGES"):
        villages = list(module.VILLAGES)
    else:
        villages = [module.VILLAGE]

    class ToolLevelConfig:
        NUM_NODES = module.NUM_NODES
        NUM_DAYS = module.NUM_DAYS
        START = module.START
        END = module.END
        MINES = mines
        VILLAGES = villages
        INIT_MONEY = module.INIT_MONEY
        WEIGHT_LIMIT = module.WEIGHT_LIMIT
        MINE_INCOME = module.MINE_INCOME
        WATER_WEIGHT = module.WATER_WEIGHT
        WATER_PRICE_BASE = module.WATER_PRICE_BASE
        FOOD_WEIGHT = module.FOOD_WEIGHT
        FOOD_PRICE_BASE = module.FOOD_PRICE_BASE
        EDGES = list(module.EDGES)

    return ToolLevelConfig


def _build_consumption(module):
    return {
        Weather.SUNNY: module.BASE_CONS[1],
        Weather.HOT: module.BASE_CONS[2],
        Weather.SANDSTORM: module.BASE_CONS[3],
    }


def _build_weather_seq(module):
    return [w - 1 for w in module.WEATHER]


def _run_oracle(module):
    config_cls = _build_config(module)
    base_consumption = _build_consumption(module)
    weather_seq = _build_weather_seq(module)
    return solve_theoretical_optimal_with_config(
        config_cls,
        weather_seq,
        base_consumption=base_consumption,
        time_limit=ORACLE_TIME_LIMIT,
        solver_options=HIGH_PRECISION_OPTIONS,
    )


def _solve_reference(module):
    prob, _, _ = module.build_model()

    try:
        solver = pulp.HiGHS(
            msg=False,
            timeLimit=REFERENCE_TIME_LIMIT,
            options=list(HIGH_PRECISION_OPTIONS),
        )
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=REFERENCE_TIME_LIMIT)

    prob.solve(solver)

    if prob.status != pulp.LpStatusOptimal:
        status_name = pulp.LpStatus.get(prob.status, "Unknown")
        pytest.skip(f"Reference solver not optimal: {status_name}")

    return float(pulp.value(prob.objective))


@pytest.mark.slow
def test_oracle_level1_map():
    module = _load_tool_config("level1-solved")
    expected = _solve_reference(module)
    result = _run_oracle(module)

    assert "status" in result
    assert "objective" in result
    assert "reached" in result

    assert result["objective"] == pytest.approx(expected, abs=1e-3)


@pytest.mark.slow
def test_oracle_level2_map():
    module = _load_tool_config("level2-solved")
    expected = _solve_reference(module)
    result = _run_oracle(module)

    assert "status" in result
    assert "objective" in result
    assert "reached" in result

    assert result["objective"] == pytest.approx(expected, abs=1e-3)
