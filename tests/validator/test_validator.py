import pandas as pd
import pytest

from src.utils.validator import (
    RLResultValidator,
    _build_validation_config,
    _read_result_file,
)


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


def test_validator_accepts_consistent_small_path_level3():
    cfg = _build_validation_config(3)

    # 路径: 1 -> 4 -> 6 -> 13(终点)
    # 晴朗移动消耗: 水6 食8
    # day0 买 30/30: money=10000-150-300=9550
    # day1 后: 24/22
    # day2 后: 18/14
    # day3 到终点前: 12/6，终点全部退回得60 => money=9610, 水食为0
    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": 1,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": 4,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 24,
                "剩余食物量(箱)": 22,
                "天气": "晴朗",
                "操作": "移动 1->4",
            },
            {
                "日期": 2,
                "区域": 6,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 18,
                "剩余食物量(箱)": 14,
                "天气": "晴朗",
                "操作": "移动 4->6",
            },
            {
                "日期": 3,
                "区域": 13,
                "剩余资金(元)": 9610,
                "剩余水量(箱)": 0,
                "剩余食物量(箱)": 0,
                "天气": "晴朗",
                "操作": "移动 6->13",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()
    assert ok is True
    assert validator.errors == []


def test_validator_flags_mine_on_arrival():
    cfg = _build_validation_config(3)
    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": 1,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": 2,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 24,
                "剩余食物量(箱)": 22,
                "天气": "晴朗",
                "操作": "移动 1->2",
            },
            {
                "日期": 2,
                "区域": 3,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 18,
                "剩余食物量(箱)": 14,
                "天气": "晴朗",
                "操作": "移动 2->3",
            },
            {
                "日期": 3,
                "区域": 9,
                "剩余资金(元)": 9750,
                "剩余水量(箱)": 9,
                "剩余食物量(箱)": 2,
                "天气": "晴朗",
                "操作": "移动 3->9+挖矿",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()

    assert ok is False
    assert any("到达矿山当天不能挖矿" in msg for msg in validator.errors)


def test_validator_flags_resource_mismatch():
    cfg = _build_validation_config(3)
    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": 1,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": 4,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 29,
                "剩余食物量(箱)": 29,
                "天气": "晴朗",
                "操作": "移动 1->4",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()

    assert ok is False
    assert any("水量不匹配" in msg or "食物不匹配" in msg for msg in validator.errors)


def test_validator_flags_wrong_village_price():
    cfg = _build_validation_config(3)
    cfg.villages.add(cfg.start)

    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": cfg.start,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": cfg.start,
                "剩余资金(元)": 9540,
                "剩余水量(箱)": 28,
                "剩余食物量(箱)": 27,
                "天气": "晴朗",
                "操作": "购买(水1食1)",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()

    assert ok is False
    assert any("资金不匹配" in msg for msg in validator.errors)


def test_validator_flags_non_adjacent_move():
    cfg = _build_validation_config(3)
    start = cfg.start
    non_neighbor = next(
        n for n in range(1, cfg.end + 1) if n not in cfg.adjacency[start] and n != start
    )
    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": start,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": non_neighbor,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 24,
                "剩余食物量(箱)": 22,
                "天气": "晴朗",
                "操作": f"移动 {start}->{non_neighbor}",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()

    assert ok is False
    assert any("移动到非相邻节点" in msg for msg in validator.errors)


def test_validator_flags_end_refund_mismatch():
    cfg = _build_validation_config(3)
    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": 1,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": 4,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 24,
                "剩余食物量(箱)": 22,
                "天气": "晴朗",
                "操作": "移动 1->4",
            },
            {
                "日期": 2,
                "区域": 6,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 18,
                "剩余食物量(箱)": 14,
                "天气": "晴朗",
                "操作": "移动 4->6",
            },
            {
                "日期": 3,
                "区域": 13,
                "剩余资金(元)": 9999,
                "剩余水量(箱)": 0,
                "剩余食物量(箱)": 0,
                "天气": "晴朗",
                "操作": "移动 6->13",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()

    assert ok is False
    assert any("终点资金不匹配" in msg for msg in validator.errors)


def test_validator_flags_sandstorm_mining_consumption():
    cfg = _build_validation_config(3)
    mine = next(iter(cfg.mines))
    start = cfg.start

    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": start,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": mine,
                "剩余资金(元)": 9750,
                "剩余水量(箱)": 20,
                "剩余食物量(箱)": 20,
                "天气": "沙暴",
                "操作": "挖矿",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()

    assert ok is False
    assert any("水量不匹配" in msg or "食物不匹配" in msg for msg in validator.errors)


def test_validator_flags_sandstorm_move():
    cfg = _build_validation_config(3)
    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": cfg.start,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": cfg.start + 1,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 10,
                "剩余食物量(箱)": 10,
                "天气": "沙暴",
                "操作": f"移动 {cfg.start}->{cfg.start + 1}",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()

    assert ok is False
    assert any("沙暴日不能移动" in msg for msg in validator.errors)


def test_validator_flags_non_village_purchase():
    cfg = _build_validation_config(3)
    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": cfg.start,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": cfg.start,
                "剩余资金(元)": 9520,
                "剩余水量(箱)": 28,
                "剩余食物量(箱)": 27,
                "天气": "晴朗",
                "操作": "购买(水1食1)",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()

    assert ok is False
    assert any("非村庄购买" in msg for msg in validator.errors)


def test_validator_flags_leaving_end_after_reached():
    cfg = _build_validation_config(3)
    df = pd.DataFrame(
        [
            {
                "日期": 0,
                "区域": 1,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 30,
                "剩余食物量(箱)": 30,
                "天气": "晴朗",
                "操作": "购买(水30食30)",
            },
            {
                "日期": 1,
                "区域": 4,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 24,
                "剩余食物量(箱)": 22,
                "天气": "晴朗",
                "操作": "移动 1->4",
            },
            {
                "日期": 2,
                "区域": 6,
                "剩余资金(元)": 9550,
                "剩余水量(箱)": 18,
                "剩余食物量(箱)": 14,
                "天气": "晴朗",
                "操作": "移动 4->6",
            },
            {
                "日期": 3,
                "区域": 13,
                "剩余资金(元)": 9610,
                "剩余水量(箱)": 0,
                "剩余食物量(箱)": 0,
                "天气": "晴朗",
                "操作": "移动 6->13",
            },
            {
                "日期": 4,
                "区域": 12,
                "剩余资金(元)": 9610,
                "剩余水量(箱)": 0,
                "剩余食物量(箱)": 0,
                "天气": "晴朗",
                "操作": "移动 13->12",
            },
        ]
    )

    validator = RLResultValidator(df, cfg)
    ok = validator.validate()

    assert ok is False
    assert any("已到达终点后离开终点" in msg for msg in validator.errors)


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
    cfg = _build_validation_config(3)
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
    assert list(normalized.columns) == [
        "day",
        "loc",
        "money",
        "water",
        "food",
        "weather",
        "action",
    ]


def test_read_result_file_csv_and_missing(tmp_path):
    cfg = _build_validation_config(3)
    csv_path = tmp_path / "result.csv"
    _minimal_df(cfg).to_csv(csv_path, index=False, encoding="utf-8-sig")

    loaded = _read_result_file(str(csv_path))
    assert len(loaded) == 1

    with pytest.raises(FileNotFoundError):
        _read_result_file(str(tmp_path / "not_exists.xlsx"))
    assert len(loaded) == 1

    with pytest.raises(FileNotFoundError):
        _read_result_file(str(tmp_path / "not_exists.xlsx"))
