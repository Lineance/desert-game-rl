import builtins
import sys

import pandas as pd

from src.pipeline.evaluate import (
    export_to_csv,
    export_to_xlsx,
    format_action,
    run_episode,
)


def test_format_action_with_actual_move_and_purchase():
    action = {
        "move_from": 0,
        "move_to": 3,
        "mine": False,
        "buy_water": 20,
        "buy_food": 10,
    }
    text = format_action(action, position=4)
    assert "移动 1->4" in text
    assert "购买(水20食10)" in text


def test_export_to_csv_writes_expected_columns(tmp_path):
    result = {
        "records": [
            {
                "day": 0,
                "position": 1,
                "money": 9550.0,
                "water": 30,
                "food": 30,
                "weather": "晴朗",
                "action": {
                    "move_from": 0,
                    "move_to": 0,
                    "mine": False,
                    "buy_water": 30,
                    "buy_food": 30,
                },
            },
            {
                "day": 1,
                "position": 4,
                "money": 9550.0,
                "water": 24,
                "food": 22,
                "weather": "晴朗",
                "action": {
                    "move_from": 0,
                    "move_to": 3,
                    "mine": False,
                    "buy_water": 0,
                    "buy_food": 0,
                },
            },
        ]
    }

    out = tmp_path / "result.csv"
    export_to_csv(result, str(out), level=3)

    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "日期",
        "区域",
        "剩余资金(元)",
        "剩余水量(箱)",
        "剩余食物量(箱)",
        "天气",
        "操作",
    ]
    assert len(df) == 2


def test_format_action_with_move_key_stay():
    action = {
        "move": 2,
        "mine": False,
        "buy_water": 0,
        "buy_food": 0,
    }
    text = format_action(action, position=3)
    assert text == "停留"


def test_format_action_defaults_to_stay_and_mine():
    action = {
        "mine": True,
        "buy_water": 0,
        "buy_food": 0,
    }
    text = format_action(action, position=1)
    assert text == "停留+挖矿"


def test_export_to_xlsx_falls_back_to_csv(tmp_path, monkeypatch):
    result = {
        "records": [
            {
                "day": 0,
                "position": 1,
                "money": 9550.0,
                "water": 30,
                "food": 30,
                "weather": "晴朗",
                "action": {
                    "move_from": 0,
                    "move_to": 0,
                    "mine": False,
                    "buy_water": 30,
                    "buy_food": 30,
                },
            }
        ]
    }

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openpyxl":
            raise ImportError("openpyxl missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    out = tmp_path / "result.xlsx"
    export_to_xlsx(result, str(out), level=3)

    assert (tmp_path / "result.csv").exists()


def test_export_to_xlsx_writes_file_with_stub(tmp_path, monkeypatch):
    class _Worksheet:
        def __init__(self):
            self.title = ""

        def append(self, _row):
            return None

    class _Workbook:
        def __init__(self):
            self.active = _Worksheet()

        def create_sheet(self, _name):
            return _Worksheet()

        def save(self, filepath):
            path = str(filepath)
            with open(path, "wb") as f:
                f.write(b"stub")

    stub_module = type(sys)("openpyxl")
    stub_module.Workbook = _Workbook
    monkeypatch.setitem(sys.modules, "openpyxl", stub_module)

    result = {
        "records": [
            {
                "day": 0,
                "position": 1,
                "money": 9550.0,
                "water": 30,
                "food": 30,
                "weather": "晴朗",
                "action": {
                    "move_from": 0,
                    "move_to": 0,
                    "mine": False,
                    "buy_water": 30,
                    "buy_food": 30,
                },
            }
        ],
        "final_money": 100.0,
        "final_water": 1,
        "final_food": 2,
        "reached": False,
        "length": 1,
    }

    out = tmp_path / "result.xlsx"
    export_to_xlsx(result, str(out), level=3)

    assert out.exists()


def test_run_episode_verbose_prints(capsys):
    import numpy as np
    import torch

    class _FakeAgent:
        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

        def reset_hidden(self, device=None):
            return None

        def select_action(self, obs, valid_actions, deterministic=True):
            return {"move": 0, "mine": False, "buy_water": 0, "buy_food": 0}, 0.5

    class _FakeEnv:
        def reset(self, seed=None):
            return np.zeros(1), {"day": 0}

        def get_valid_actions(self):
            return {"valid_moves": [0], "can_mine": False, "can_buy": False}

        def step(self, action):
            info = {
                "action_day": 1,
                "day": 1,
                "position": 0,
                "money": 100.0,
                "water": 1,
                "food": 2,
                "weather_today": 0,
                "belief": np.zeros(2),
                "last_action": {"move": 0, "mine": False, "buy_water": 0, "buy_food": 0},
            }
            return np.zeros(1), 1.0, True, False, info

    result = run_episode(_FakeAgent(), _FakeEnv(), seed=0, deterministic=True, verbose=True)
    output = capsys.readouterr().out

    assert "Day 1" in output
    assert result["length"] == 1
