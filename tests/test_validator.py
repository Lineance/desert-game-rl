import pandas as pd
import pytest

from src.utils.validator import RLResultValidator, _build_validation_config


def test_build_validation_config_level3():
    cfg = _build_validation_config(3)
    assert cfg.start >= 1
    assert cfg.end >= 1
    assert cfg.name == "第三关"


def test_validator_rejects_missing_columns():
    cfg = _build_validation_config(3)
    df = pd.DataFrame({"day": [0], "loc": [cfg.start]})
    validator = RLResultValidator(df, cfg)

    with pytest.raises(ValueError):
        validator.validate()
