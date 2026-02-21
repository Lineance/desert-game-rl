from src.env.environment import _build_runtime_config, make_env


def test_build_runtime_config_overrides_uppercase_only():
    config = _build_runtime_config(
        base_config=type("Base", (), {"INIT_MONEY": 100, "START": 0}),
        config_override={"INIT_MONEY": 200, "start": 3, "CONFIG_NAME": "RuntimeCfg"},
    )
    assert config.__name__ == "RuntimeCfg"
    assert config.INIT_MONEY == 200
    assert config.START == 0
    assert not hasattr(config, "start")


def test_make_env_accepts_config_override():
    env = make_env(
        level=35,
        weather_mode="train_easy",
        seed=42,
        config_override={
            "CONFIG_NAME": "StageEnv",
            "INIT_MONEY": 3000,
            "START": 10,
            "VILLAGES": [],
            "EDGES": [
                (1, 2),
                (2, 3),
                (3, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 5),
                (5, 14),
                (14, 15),
                (15, 16),
                (16, 17),
            ],
        },
    )

    assert env.config.__name__ == "StageEnv"
    assert env.config.INIT_MONEY == 3000
    assert env.config.START == 10
    assert env.config.VILLAGES == []
