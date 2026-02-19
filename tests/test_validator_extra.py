import pandas as pd
import pytest

from src.pipeline.validator import (
    RLResultValidator,
    build_validation_config,
    read_result_file,
)


def _minimal_df(cfg):
    return pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": cfg.start,
                "剩余资金(元)": 10000,
                "剩余水量(箱)": 0,
                "剩余食物量(箱)": 0,
                "天气": "晴朗",
                "操作": "停留",
            }
        ]
    )


def test_parse_action_supports_move_mine_buy():
    parsed = RLResultValidator._parse_action("移动 1->4+挖矿+购买(水20食10)")
    assert parsed.kind == "mine"
    assert parsed.buy_water == 20
    assert parsed.buy_food == 10


def test_normalize_columns_supports_english_aliases():
    cfg = build_validation_config(3)
    df = pd.DataFrame(
        [
            {
                "day": 0,
                "loc": cfg.start,
                "money": 10000,
                "water": 0,
                "food": 0,
                "weather": "晴朗",
                "action": "停留",
            }
        ]
    )

    normalized = RLResultValidator._normalize_columns(df)
    assert list(normalized.columns) == ["day", "loc", "money", "water", "food", "weather", "action"]


def test_read_result_file_csv_and_missing(tmp_path):
    cfg = build_validation_config(3)
    csv_path = tmp_path / "result.csv"
    _minimal_df(cfg).to_csv(csv_path, index=False, encoding="utf-8-sig")

    loaded = read_result_file(str(csv_path))
    assert len(loaded) == 1

    with pytest.raises(FileNotFoundError):
        read_result_file(str(tmp_path / "not_exists.xlsx"))
    assert len(loaded) == 1

    with pytest.raises(FileNotFoundError):
        read_result_file(str(tmp_path / "not_exists.xlsx"))
