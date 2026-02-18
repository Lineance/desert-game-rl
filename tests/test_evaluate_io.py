import pandas as pd
from src.pipeline.evaluate import export_to_csv, format_action


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
                "action": {"move_from": 0, "move_to": 0, "mine": False, "buy_water": 30, "buy_food": 30},
            },
            {
                "day": 1,
                "position": 4,
                "money": 9550.0,
                "water": 24,
                "food": 22,
                "weather": "晴朗",
                "action": {"move_from": 0, "move_to": 3, "mine": False, "buy_water": 0, "buy_food": 0},
            },
        ]
    }

    out = tmp_path / "result.csv"
    export_to_csv(result, str(out), level=3)

    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == ["日期", "区域", "剩余资金(元)", "剩余水量(箱)", "剩余食物量(箱)", "天气", "操作"]
    assert len(df) == 2
