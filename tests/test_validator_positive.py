import pandas as pd

from src.pipeline.validator import RLResultValidator, build_validation_config


def test_validator_accepts_consistent_small_path_level3():
    cfg = build_validation_config(3)

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
