"""
沙漠穿越环境 (POMDP)
问题2：仅知当天天气的部分可观测决策
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.env.config import (
    BASE_CONSUMPTION,
    Level3Config,
    Level4Config,
    Level35Config,
    Weather,
    compute_shortest_distances,
    get_adjacency_matrix,
    get_neighbors,
)
from src.models.belief import WeatherBeliefModel, extract_belief_features


def _build_runtime_config(base_config: Any, config_override: Optional[Mapping[str, Any]]) -> Any:
    if not config_override:
        return base_config

    override_dict = dict(config_override)
    safe_name = override_dict.pop("CONFIG_NAME", f"{base_config.__name__}Runtime")

    attributes: Dict[str, Any] = {}
    for key, value in base_config.__dict__.items():
        if key.startswith("__"):
            continue
        attributes[key] = value

    for key, value in override_dict.items():
        if not isinstance(key, str) or not key.isupper():
            continue
        attributes[key] = value

    return type(str(safe_name), (), attributes)


def _normalize_probs(probs: List[float]) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        arr = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        arr = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    else:
        arr = arr / total
    return arr


@dataclass
class State:
    """环境状态"""

    day: int  # 当前天数（从0开始）
    position: int  # 当前位置 (0-based)
    water: int  # 剩余水
    food: int  # 剩余食物
    money: float  # 剩余资金
    weather_today: int  # 今天天气
    weather_future: List[int]  # 未来天气
    reached: bool  # 是否已到达终点
    terminated: bool  # 是否结束
    has_purchased_at_start: bool  # 是否已在起点购买过（防止重复购买）
    path_history: List[int]
    action_history: List[str]
    last_action: Optional[Dict[str, Any]]


class DesertCrossingEnv:
    """沙漠穿越环境 - 修复版"""

    def __init__(self, config=None, weather_mode="balanced", seed=None):
        if config is None:
            config = Level3Config

        self.config = config
        self.weather_mode = weather_mode
        self.rng = np.random.RandomState(seed)

        # 地图
        self.conn = get_adjacency_matrix(config.NUM_NODES, config.EDGES)
        self.neighbors = [get_neighbors(self.conn, i) for i in range(config.NUM_NODES)]
        self.dist_to_end = compute_shortest_distances(self.conn, config.END)

        self.state: Optional[State] = None
        self.belief_model: Optional[WeatherBeliefModel] = None
        self.max_neighbors = max(len(n) for n in self.neighbors) + 1

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # 生成天气序列（第1天到第NUM_DAYS天）
        weather_sequence = self._generate_weather_sequence()

        # BUG修复: 第0天在起点，初始资源为0，资金完整
        # 购买必须通过第0天的显式动作完成
        init_water, init_food = 0, 0
        init_money = self.config.INIT_MONEY

        # 第0天开始（第0天使用第1天天气用于规划）
        self.state = State(
            day=0,  # BUG修复: 从第0天开始
            position=self.config.START,
            water=init_water,
            food=init_food,
            money=init_money,
            weather_today=weather_sequence[0],  # 第0天显示第1天天气
            weather_future=weather_sequence,  # 保留完整天气序列
            reached=False,
            terminated=False,
            has_purchased_at_start=False,  # 初始化购买标志
            path_history=[self.config.START],
            action_history=["start"],
            last_action=None,
        )

        # 初始化信念模型
        self.belief_model = WeatherBeliefModel(
            transition_prior=self.config.WEATHER_TRANSITION.copy(), learning_rate=0.1
        )
        self.belief_model.update(self.state.weather_today)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _generate_weather_sequence(self) -> List[int]:
        """生成天气序列"""
        if self.weather_mode in self.config.WEATHER_MODES:
            probs = self.config.WEATHER_MODES[self.weather_mode]
        else:
            probs = [0.5, 0.5, 0.0]  # 第三关默认无沙暴

        init_probs = _normalize_probs(list(probs))
        transition = np.asarray(self.config.WEATHER_TRANSITION, dtype=np.float64)

        sequence = []
        current = int(self.rng.choice(3, p=init_probs))

        for _ in range(self.config.NUM_DAYS):
            sequence.append(current)
            next_probs = _normalize_probs(list(transition[current]))
            current = int(self.rng.choice(3, p=next_probs))

        return sequence

    def _compute_initial_resources(self) -> Tuple[int, int]:
        """计算初始资源"""
        min_days = self.dist_to_end[self.config.START]
        worst_cons = BASE_CONSUMPTION[Weather.HOT]

        # 预留缓冲
        buffer_days = min_days // 3
        total_days = min(min_days + buffer_days, self.config.NUM_DAYS)

        needed_water = worst_cons[0] * 2 * total_days
        needed_food = worst_cons[1] * 2 * total_days

        max_water = self.config.WEIGHT_LIMIT // self.config.WATER_WEIGHT
        max_food = self.config.WEIGHT_LIMIT // self.config.FOOD_WEIGHT

        water = min(needed_water, max_water // 2)
        food = min(needed_food, max_food // 2)

        # 调整重量
        total_weight = water * self.config.WATER_WEIGHT + food * self.config.FOOD_WEIGHT

        if total_weight > self.config.WEIGHT_LIMIT:
            ratio = self.config.WEIGHT_LIMIT / total_weight * 0.95
            water = int(water * ratio)
            food = int(food * ratio)

        return water, food

    def _compute_purchase_cost(self, water: int, food: int, at_start: bool) -> float:
        """计算购买成本"""
        if at_start:
            return water * self.config.WATER_PRICE_BASE + food * self.config.FOOD_PRICE_BASE
        else:
            return water * self.config.WATER_PRICE_BASE * 2 + food * self.config.FOOD_PRICE_BASE * 2

    def _get_observation(self) -> np.ndarray:
        """构建观测向量"""
        s = self.state

        state_features = np.array(
            [
                s.day / self.config.NUM_DAYS,
                s.position / self.config.NUM_NODES,
                s.water / 200.0,
                s.food / 200.0,
                s.money / self.config.INIT_MONEY,
                self.dist_to_end[s.position] / self.config.NUM_NODES,
            ]
        )

        weather_onehot = np.zeros(3)
        weather_onehot[s.weather_today] = 1.0

        location_type = np.zeros(4)
        if s.position == self.config.START:
            location_type[0] = 1.0
        elif s.position == self.config.END:
            location_type[1] = 1.0
        elif s.position in self.config.MINES:
            location_type[2] = 1.0
        elif s.position in self.config.VILLAGES:
            location_type[3] = 1.0

        belief_features = extract_belief_features(self.belief_model)

        obs = np.concatenate([state_features, weather_onehot, location_type, belief_features])

        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            "day": self.state.day,
            "position": self.state.position,
            "water": self.state.water,
            "food": self.state.food,
            "money": self.state.money,
            "weather_today": self.state.weather_today,
            "belief": self.belief_model.current_belief.probs,
            "reached": self.state.reached,
            "last_action": self.state.last_action,
        }

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作 - 修复版

        正确顺序：
        1. 移动（检查邻居、沙暴限制）
        2. 计算并扣除资源消耗
        3. 挖矿/购买操作
        4. 更新天数和天气
        5. 检查终止条件
        """
        s = self.state

        if s.reached or s.terminated:
            return self._get_observation(), 0.0, True, False, self._get_info()

        action_day = s.day
        action_weather = s.weather_today
        prev_position = s.position
        prev_dist_to_end = self.dist_to_end[prev_position]

        # 解析动作
        move_target = action.get("move", s.position)
        do_mine = action.get("mine", False)
        buy_water = action.get("buy_water", 0)
        buy_food = action.get("buy_food", 0)

        reward = 0.0
        action_name = "停留"
        mining = False
        executed_buy_water = 0
        executed_buy_food = 0

        # ========== 1. 第0天特殊处理 ==========
        if s.day == 0:
            # 第0天只能停留和购买，不消耗资源
            is_moving = False

            # 第0天购买（起点，仅一次）
            if buy_water > 0 or buy_food > 0:
                if s.day == 0 and s.position == self.config.START and not s.has_purchased_at_start:
                    cost = self._compute_purchase_cost(buy_water, buy_food, True)
                    if cost <= s.money:
                        new_weight = (
                            buy_water * self.config.WATER_WEIGHT
                            + buy_food * self.config.FOOD_WEIGHT
                        )
                        if new_weight <= self.config.WEIGHT_LIMIT:
                            s.water += buy_water
                            s.food += buy_food
                            s.money -= cost
                            executed_buy_water = int(buy_water)
                            executed_buy_food = int(buy_food)
                            s.has_purchased_at_start = True  # 设置标志位防止重复购买
                            action_name = "购买"

            # 更新记录
            s.path_history.append(s.position)
            s.action_history.append(action_name)
            s.last_action = {
                "move_from": prev_position,
                "move_to": s.position,
                "move": s.position,
                "mine": False,
                "buy_water": executed_buy_water,
                "buy_food": executed_buy_food,
                "name": action_name,
            }

            # 第0天结束，进入第1天
            s.day = 1
            # 第1天使用天气序列第0项（第1天天气）
            if s.weather_future and (s.day - 1) < len(s.weather_future):
                s.weather_today = s.weather_future[s.day - 1]
            self.belief_model.update(s.weather_today)

            info = self._get_info()
            info["action_day"] = action_day
            return self._get_observation(), reward, False, False, info

        # ========== 2. 移动处理（第1天及以后，沙暴检查前置）==========
        is_moving = move_target != s.position

        if is_moving:
            # BUG修复：先检查沙暴，再移动
            if s.weather_today == Weather.SANDSTORM:
                reward -= 50.0
                is_moving = False  # 强制停留，不移动
            elif move_target in self.neighbors[s.position]:
                s.position = move_target
                action_name = "移动"
            else:
                reward -= 10.0
                is_moving = False

        # ========== 2. 确定行动类型 ==========
        # 挖矿检查：必须在矿山且到达当天不能挖
        just_arrived = is_moving  # 如果移动了，就是刚到达
        can_mine = s.position in self.config.MINES and not just_arrived

        if do_mine and can_mine:
            mining = True
            action_name = "挖矿"
        elif do_mine and not can_mine:
            # 非法挖矿，忽略
            do_mine = False

        # ========== 3. 计算资源消耗 ==========
        base_w, base_f = BASE_CONSUMPTION[s.weather_today]

        # 消耗因子：行走2倍，挖矿3倍，停留1倍
        if mining:
            factor = 3
        elif is_moving:
            factor = 2
        else:
            factor = 1

        water_cons = base_w * factor
        food_cons = base_f * factor

        # 检查资源
        if s.water < water_cons or s.food < food_cons:
            s.terminated = True
            reward -= 50.0
            s.last_action = {
                "move_from": prev_position,
                "move_to": s.position,
                "move": s.position,
                "mine": mining,
                "buy_water": 0,
                "buy_food": 0,
                "name": action_name,
            }
            info = self._get_info()
            info["action_day"] = action_day
            return self._get_observation(), reward, True, False, info

        # 扣除资源
        s.water -= water_cons
        s.food -= food_cons

        # 挖矿收益
        if mining:
            s.money += self.config.MINE_INCOME
            reward += max(1.0, self.config.MINE_INCOME / 50.0)
            if "挖矿" not in s.action_history:
                reward += 3.0

        # ========== 4. 购买处理（第1天及以后，仅村庄）==========
        # 第0天的购买已在前面处理，起点不能重复购买
        if s.day >= 1 and (buy_water > 0 or buy_food > 0):
            at_village = s.position in self.config.VILLAGES

            if at_village:
                cost = self._compute_purchase_cost(buy_water, buy_food, False)
                if cost <= s.money:
                    new_weight = (s.water + buy_water) * self.config.WATER_WEIGHT + (
                        s.food + buy_food
                    ) * self.config.FOOD_WEIGHT
                    if new_weight <= self.config.WEIGHT_LIMIT:
                        s.water += buy_water
                        s.food += buy_food
                        s.money -= cost
                        executed_buy_water = int(buy_water)
                        executed_buy_food = int(buy_food)
                        action_name += "+购买" if action_name != "停留" else "购买"

        # ========== 5. 更新记录 ==========
        s.path_history.append(s.position)
        s.action_history.append(action_name)
        s.last_action = {
            "move_from": prev_position,
            "move_to": s.position,
            "move": s.position,
            "mine": mining,
            "buy_water": executed_buy_water,
            "buy_food": executed_buy_food,
            "name": action_name,
        }

        # ========== 6. 检查终止条件 ==========
        terminated = False

        # 到达终点
        if s.position == self.config.END:
            s.reached = True
            terminated = True
            # 退回剩余资源
            refund = (
                s.water * self.config.WATER_PRICE_BASE * 0.5
                + s.food * self.config.FOOD_PRICE_BASE * 0.5
            )
            s.money += refund
            s.water = 0
            s.food = 0
            # 终点奖励：保持“到达”基线，但显著放大资金效率信号
            money_ratio = s.money / self.config.INIT_MONEY
            reward += 20.0 + (money_ratio - 1.0) * 100.0

        # ========== 7. 更新天数和天气 ==========
        if not terminated:
            s.day += 1

            # 检查超时（第NUM_DAYS天结束后必须到达）
            if s.day > self.config.NUM_DAYS:
                s.terminated = True
                terminated = True
                if not s.reached:
                    reward -= 50.0  # 未到达终点惩罚
                # 如果已到达，前面已处理，不再惩罚
            else:
                # 更新天气
                if s.weather_future and (s.day - 1) < len(s.weather_future):
                    s.weather_today = s.weather_future[s.day - 1]
                self.belief_model.update(s.weather_today)

                # 奖励设计（修复探索崩溃：强制前进）
                if len(s.path_history) > 1:
                    prev_dist = prev_dist_to_end
                    curr_dist = self.dist_to_end[s.position]
                    dist_improvement = prev_dist - curr_dist  # 正数表示靠近

                    if is_moving and s.position in self.config.MINES:
                        reward += 2.0

                    if dist_improvement > 0:
                        reward += 8.0 * dist_improvement
                    elif dist_improvement < 0:
                        reward -= 4.0 * abs(dist_improvement)

                    if dist_improvement == 0 and not mining and action_weather != Weather.SANDSTORM:
                        reward -= 2.0

                # 生存成本（轻度）
                reward -= 1.0

                # 时间压力（轻度）
                reward -= (s.day / self.config.NUM_DAYS) * 4.0

        info = self._get_info()
        info["action_day"] = action_day
        return self._get_observation(), reward, terminated, False, info

    def get_valid_actions(self) -> Dict[str, Any]:
        """获取有效动作"""
        s = self.state

        valid_moves = []
        if s.weather_today != Weather.SANDSTORM:
            valid_moves = list(self.neighbors[s.position])
        valid_moves.append(s.position)

        # 挖矿：必须在矿山且不是刚到达（上一天与当前在同一节点）
        can_mine = (
            s.position in self.config.MINES
            and len(s.path_history) >= 2
            and s.path_history[-1] == s.path_history[-2]
        )

        can_buy_now = (
            s.day == 0 and s.position == self.config.START and not s.has_purchased_at_start
        ) or (s.position in self.config.VILLAGES)
        can_buy_after_move = s.day >= 1 and any(m in self.config.VILLAGES for m in valid_moves)
        can_buy = can_buy_now or can_buy_after_move

        max_buy_water = 0
        max_buy_food = 0
        if can_buy:
            if s.day == 0 and s.position == self.config.START and not s.has_purchased_at_start:
                est_water_after_cons = s.water
                est_food_after_cons = s.food
                est_money_after_cons = s.money
                water_price = self.config.WATER_PRICE_BASE
                food_price = self.config.FOOD_PRICE_BASE
            else:
                base_w, base_f = BASE_CONSUMPTION[s.weather_today]

                can_mine_now = (
                    s.position in self.config.MINES
                    and len(s.path_history) >= 2
                    and s.path_history[-1] == s.path_history[-2]
                )

                possible_factors = [1]
                if len(valid_moves) > 1:
                    possible_factors.append(2)
                if can_mine_now:
                    possible_factors.append(3)

                min_factor = min(possible_factors)
                est_water_after_cons = max(0, s.water - base_w * min_factor)
                est_food_after_cons = max(0, s.food - base_f * min_factor)
                est_money_after_cons = s.money + (self.config.MINE_INCOME if can_mine_now else 0)
                water_price = self.config.WATER_PRICE_BASE * 2
                food_price = self.config.FOOD_PRICE_BASE * 2

            current_weight = (
                est_water_after_cons * self.config.WATER_WEIGHT
                + est_food_after_cons * self.config.FOOD_WEIGHT
            )
            remaining_weight = max(0, self.config.WEIGHT_LIMIT - current_weight)

            max_by_weight_water = remaining_weight // self.config.WATER_WEIGHT
            max_by_weight_food = remaining_weight // self.config.FOOD_WEIGHT
            max_by_money_water = int(est_money_after_cons // water_price) if water_price > 0 else 0
            max_by_money_food = int(est_money_after_cons // food_price) if food_price > 0 else 0

            max_buy_water = int(max(0, min(200, max_by_weight_water, max_by_money_water)))
            max_buy_food = int(max(0, min(200, max_by_weight_food, max_by_money_food)))

        return {
            "valid_moves": valid_moves,
            "can_mine": can_mine,
            "can_buy": can_buy,
            "max_buy_water": max_buy_water,
            "max_buy_food": max_buy_food,
        }


def make_env(level=3, config_override: Optional[Mapping[str, Any]] = None, **kwargs):
    """创建环境"""
    if level == 3:
        base_config = Level3Config
    elif level == 35:
        base_config = Level35Config
    elif level == 4:
        base_config = Level4Config
    else:
        raise ValueError(f"Unknown level: {level}")

    config = _build_runtime_config(base_config, config_override)
    return DesertCrossingEnv(config=config, **kwargs)
