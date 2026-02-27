"""task2 结果验证器 (适配第三/第四问配置) 。

用途：验证 evaluate.py 导出的 CSV/XLSX 是否满足环境规则与资源守恒。
默认输入列：
- 日期 / day
- 区域 / loc
- 剩余资金(元) / money
- 剩余水量(箱) / water
- 剩余食物量(箱) / food
- 天气 / weather
- 操作 / action
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from src.env.config import BASE_CONSUMPTION, Level3Config, Level4Config, Weather

WEATHER_NAME_TO_ID = {
    "晴朗": Weather.SUNNY,
    "高温": Weather.HOT,
    "沙暴": Weather.SANDSTORM,
}


@dataclass
class ValidationConfig:
    name: str
    start: int
    end: int
    mines: Set[int]
    villages: Set[int]
    adjacency: Dict[int, Set[int]]
    water_weight: int
    food_weight: int
    weight_limit: int
    water_price_base: int
    food_price_base: int
    mine_income: int


@dataclass
class ParsedAction:
    kind: str  # stay/move/mine
    mine_intensity: float
    buy_water: int
    buy_food: int


class RLResultValidator:
    def __init__(self, df: pd.DataFrame, cfg: ValidationConfig):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def error(self, day: int, message: str) -> None:
        self.errors.append(f"第{day}天: {message}")

    def warn(self, day: int, message: str) -> None:
        self.warnings.append(f"第{day}天: {message}")

    def _check_weight(self, day: int, water: float, food: float) -> None:
        weight = water * self.cfg.water_weight + food * self.cfg.food_weight
        if weight > self.cfg.weight_limit + 1e-6:
            self.error(day, f"负重超标: {weight:.1f}kg > {self.cfg.weight_limit}kg")

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "日期": "day",
            "day": "day",
            "区域": "loc",
            "loc": "loc",
            "剩余资金(元)": "money",
            "money": "money",
            "剩余水量(箱)": "water",
            "water": "water",
            "剩余食物量(箱)": "food",
            "food": "food",
            "天气": "weather",
            "weather": "weather",
            "操作": "action",
            "action": "action",
        }

        new_cols: Dict[str, str] = {}
        for col in df.columns:
            key = str(col).strip()
            if key in mapping:
                new_cols[col] = mapping[key]

        df = df.rename(columns=new_cols)

        required = ["day", "loc", "money", "water", "food", "weather", "action"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

        for c in ["day", "loc", "money", "water", "food"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[["day", "loc", "money", "water", "food"]].isna().any().any():
            raise ValueError("数值列存在空值或非法值")

        df["weather"] = df["weather"].astype(str).str.strip()
        df["action"] = df["action"].astype(str).str.strip()
        return df

    @staticmethod
    def _parse_action(action: str) -> ParsedAction:
        parts = [p.strip() for p in action.split("+") if p.strip()]
        kind = "stay"
        mining = False
        mine_intensity = 0.0
        buy_water = 0
        buy_food = 0

        for part in parts:
            if part.startswith("移动"):
                kind = "move"
            elif part == "停留" or part == "出发准备":
                if kind != "move":
                    kind = "stay"
            elif part.startswith("挖矿"):
                mining = True
                m_intensity = re.search(r"挖矿\(强度([0-9]+(?:\.[0-9]+)?)\)", part)
                if m_intensity:
                    mine_intensity = float(m_intensity.group(1))
                else:
                    mine_intensity = 1.0
            elif part.startswith("购买"):
                m = re.search(r"购买\(水(\d+)食(\d+)\)", part)
                if m:
                    buy_water = int(m.group(1))
                    buy_food = int(m.group(2))

        if mining:
            kind = "mine"
            mine_intensity = max(0.0, min(1.0, mine_intensity))
        return ParsedAction(
            kind=kind,
            mine_intensity=mine_intensity,
            buy_water=buy_water,
            buy_food=buy_food,
        )

    def _weather_to_id(self, day: int, weather: str) -> Optional[int]:
        if weather in WEATHER_NAME_TO_ID:
            return WEATHER_NAME_TO_ID[weather]
        self.error(day, f"未知天气: {weather}")
        return None

    def validate(self) -> bool:
        df = self._normalize_columns(self.df)

        # 初始行（第0天）
        row0 = df.iloc[0]
        day0 = int(row0["day"])
        if day0 != 0:
            self.error(day0, "首行必须是第0天")

        loc = int(row0["loc"])
        water = float(row0["water"])
        food = float(row0["food"])

        if loc != self.cfg.start:
            self.error(day0, f"第0天必须在起点{self.cfg.start}，当前{loc}")
        self._check_weight(day0, water, food)

        reached_end = loc == self.cfg.end

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            prev_day = int(prev["day"])
            day = int(curr["day"])
            if day != prev_day + 1:
                self.error(day, f"日期不连续: {prev_day}->{day}")
                continue

            prev_loc = int(prev["loc"])
            curr_loc = int(curr["loc"])

            prev_money = float(prev["money"])
            prev_water = float(prev["water"])
            prev_food = float(prev["food"])

            curr_money = float(curr["money"])
            curr_water = float(curr["water"])
            curr_food = float(curr["food"])

            weather_id = self._weather_to_id(day, curr["weather"])
            if weather_id is None:
                continue

            action = self._parse_action(curr["action"])

            # 已到达终点后：位置应保持终点，不应再获得资源
            if reached_end:
                if curr_loc != self.cfg.end:
                    self.error(day, f"已到达终点后离开终点: {curr_loc}")
                if curr_water > prev_water + 1e-6 or curr_food > prev_food + 1e-6:
                    self.error(day, "终点后资源异常增加")
                if curr_money < prev_money - 1e-6 and (
                    abs(curr_water - prev_water) < 1e-6 and abs(curr_food - prev_food) < 1e-6
                ):
                    self.warn(day, "终点后资金减少")
                continue

            # 行动与位置一致性
            if action.kind == "move":
                if curr_loc == prev_loc:
                    self.error(day, "标记为移动但位置未变化")
                if curr_loc not in self.cfg.adjacency[prev_loc]:
                    self.error(day, f"移动到非相邻节点: {prev_loc}->{curr_loc}")
            else:
                if curr_loc != prev_loc:
                    self.error(day, f"标记为{action.kind}但位置变化: {prev_loc}->{curr_loc}")

            # 沙暴不能移动
            if weather_id == Weather.SANDSTORM and action.kind == "move":
                self.error(day, "沙暴日不能移动")

            # 挖矿限制
            if action.kind == "mine":
                if curr_loc not in self.cfg.mines:
                    self.error(day, "挖矿但不在矿山")
                if prev_loc != curr_loc:
                    self.error(day, "到达矿山当天不能挖矿")

            # 购买限制
            if action.buy_water > 0 or action.buy_food > 0:
                if day == 0:
                    if curr_loc != self.cfg.start:
                        self.error(day, "第0天购买只能在起点")
                else:
                    if curr_loc not in self.cfg.villages:
                        self.error(day, "非村庄购买")

            # 资源消耗
            base_w, base_f = BASE_CONSUMPTION[weather_id]
            factor = (
                (1.0 + 2.0 * action.mine_intensity)
                if action.kind == "mine"
                else (2 if action.kind == "move" else 1)
            )
            cons_w = base_w * factor
            cons_f = base_f * factor

            if prev_water + 1e-6 < cons_w or prev_food + 1e-6 < cons_f:
                self.error(day, f"消耗前资源不足: 需(水{cons_w},食{cons_f})")

            arrive_water = prev_water - cons_w
            arrive_food = prev_food - cons_f
            arrive_money = prev_money + (
                self.cfg.mine_income * action.mine_intensity if action.kind == "mine" else 0
            )

            # 默认：仅由行动文字中的购买量驱动
            buy_w = action.buy_water
            buy_f = action.buy_food

            expected_water_after_buy = arrive_water + buy_w
            expected_food_after_buy = arrive_food + buy_f

            purchase_cost = 0.0
            if buy_w > 0 or buy_f > 0:
                price_mul = 1 if day == 0 else 2
                purchase_cost = (
                    buy_w * self.cfg.water_price_base * price_mul
                    + buy_f * self.cfg.food_price_base * price_mul
                )

            expected_money_after_buy = arrive_money - purchase_cost

            if day == 0:
                # 第0天不消耗，只看起点购买
                expected_water_after_buy = prev_water + buy_w
                expected_food_after_buy = prev_food + buy_f
                expected_money_after_buy = prev_money - (
                    buy_w * self.cfg.water_price_base + buy_f * self.cfg.food_price_base
                )

            # 终点处理：允许两种等价口径
            if curr_loc == self.cfg.end:
                reached_end = True

                # 允许A：不清零，仅减少部分并按半价退款
                back_w = expected_water_after_buy - curr_water
                back_f = expected_food_after_buy - curr_food
                if back_w < -1e-6 or back_f < -1e-6:
                    self.error(day, "终点后资源增加（不能购买）")
                    back_w = max(back_w, 0)
                    back_f = max(back_f, 0)

                expected_money_end = (
                    expected_money_after_buy
                    + (back_w * self.cfg.water_price_base + back_f * self.cfg.food_price_base) * 0.5
                )

                if abs(curr_money - expected_money_end) > 1.0:
                    self.error(
                        day,
                        f"终点资金不匹配: 实际{curr_money:.2f}, 理论{expected_money_end:.2f}",
                    )

                # 若当前实现为清零退款，下面两条自然成立
                # 若不是清零退款，依然允许（通过 back_w/back_f 推断）
            else:
                if abs(curr_water - expected_water_after_buy) > 0.1:
                    self.error(
                        day,
                        f"水量不匹配: 实际{curr_water:.2f}, 理论{expected_water_after_buy:.2f}",
                    )
                if abs(curr_food - expected_food_after_buy) > 0.1:
                    self.error(
                        day,
                        f"食物不匹配: 实际{curr_food:.2f}, 理论{expected_food_after_buy:.2f}",
                    )
                if abs(curr_money - expected_money_after_buy) > 1.0:
                    self.error(
                        day,
                        f"资金不匹配: 实际{curr_money:.2f}, 理论{expected_money_after_buy:.2f}",
                    )

            self._check_weight(day, curr_water, curr_food)

        if not reached_end:
            last_day = int(df.iloc[-1]["day"])
            self.error(last_day, f"最终未到达终点{self.cfg.end}")

        return len(self.errors) == 0


def _build_adjacency_from_config(config_cls) -> Dict[int, Set[int]]:
    # 输出与结果表一致：1-based 节点编号
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for u, v in config_cls.EDGES:
        adjacency[u].add(v)
        adjacency[v].add(u)

    for n in range(1, config_cls.NUM_NODES + 1):
        adjacency[n].add(n)

    return adjacency


def _build_validation_config(level: int) -> ValidationConfig:
    if level == 3:
        cfg = Level3Config
        name = "第三关"
    elif level == 4:
        cfg = Level4Config
        name = "第四关"
    else:
        raise ValueError(f"仅支持 level=3/4，当前: {level}")

    return ValidationConfig(
        name=name,
        start=cfg.START + 1,
        end=cfg.END + 1,
        mines={m + 1 for m in cfg.MINES},
        villages={v + 1 for v in cfg.VILLAGES},
        adjacency=_build_adjacency_from_config(cfg),
        water_weight=cfg.WATER_WEIGHT,
        food_weight=cfg.FOOD_WEIGHT,
        weight_limit=cfg.WEIGHT_LIMIT,
        water_price_base=cfg.WATER_PRICE_BASE,
        food_price_base=cfg.FOOD_PRICE_BASE,
        mine_income=cfg.MINE_INCOME,
    )


def _read_result_file(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path, encoding="utf-8-sig")
    return pd.read_excel(file_path)


def validator_main() -> None:
    parser = argparse.ArgumentParser(description="task2 结果验证器")
    parser.add_argument("file", type=str, help="待验证结果文件（CSV/XLSX）")
    parser.add_argument("--level", type=int, choices=[3, 4], default=3, help="关卡编号")
    args = parser.parse_args()

    config = _build_validation_config(args.level)
    df = _read_result_file(args.file)
    validator = RLResultValidator(df, config)
    ok = validator.validate()

    print(f"\n关卡: {config.name}")
    print("=" * 70)

    if ok:
        print("✅ 验证通过")
    else:
        print(f"❌ 验证失败，错误数: {len(validator.errors)}")
        for msg in validator.errors[:30]:
            print(f"- {msg}")
        if len(validator.errors) > 30:
            print(f"... 还有 {len(validator.errors) - 30} 条错误")

    if validator.warnings:
        print(f"\n⚠️ 警告数: {len(validator.warnings)}")
        for msg in validator.warnings[:10]:
            print(f"- {msg}")


if __name__ == "__main__":
    validator_main()
