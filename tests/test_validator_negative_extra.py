import pandas as pd

from src.utils.validator import RLResultValidator, _build_validation_config


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
