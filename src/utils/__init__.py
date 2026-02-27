from .oracle import OracleSolverConfig as OracleSolverConfig
from .oracle import solve_theoretical_batch as solve_theoretical_batch
from .oracle import solve_theoretical_optimal as solve_theoretical_optimal
from .validator import RLResultValidator as RLResultValidator
from .validator import _build_validation_config as _build_validation_config

__all__ = [
    "generate_behavior_cloning_dataset",
    "load_behavior_cloning_dataset_index",
    "solve_theoretical_optimal",
    "solve_theoretical_batch",
    "OracleSolverConfig",
    "RLResultValidator",
    "_build_validation_config",
]


def __getattr__(name: str):
    if name == "generate_behavior_cloning_dataset":
        from .bc_dataset import generate_behavior_cloning_dataset

        return generate_behavior_cloning_dataset
    if name == "load_behavior_cloning_dataset_index":
        from .bc_dataset import load_behavior_cloning_dataset_index

        return load_behavior_cloning_dataset_index
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
