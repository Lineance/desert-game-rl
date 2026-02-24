import pandas as pd

from src.utils.validator import RLResultValidator, _build_validation_config


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
