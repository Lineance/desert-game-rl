"""已知完整天气序列时的理论最优（MILP Oracle）。

用于给 benchmark 提供上界参考：
- 输入：关卡配置 + 天气序列（长度=NUM_DAYS，编码0/1/2）
- 输出：最优目标值（最终资金+剩余资源折现）与求解状态
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pulp

from src.env.config import (
    BASE_CONSUMPTION,
    Level3Config,
    Level4Config,
    Level35Config,
    Weather,
    get_adjacency_matrix,
)


def solve_theoretical_optimal(
    level: int,
    weather_seq: Sequence[int],
    time_limit: int = 60,
) -> Dict[str, float]:
    """求解已知天气下的理论最优。"""
    config_cls = _pick_config(level)
    prob, v = _build_model(config_cls, weather_seq)

    try:
        solver = pulp.HiGHS(
            msg=False,
            timeLimit=time_limit,
            options=["--mip_rel_gap", "0.0001", "--mip_abs_gap", "0.1"],
        )
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)

    prob.solve(solver)
    result = _solve_problem(prob, v, config_cls, want_plan=False)
    objective = result["objective"]

    return {
        "status": result["status"],
        "is_proven_optimal": result["is_proven_optimal"],
        "objective": objective,
        "reached": result["reached"],
        "final_money": result["final_money"],
        "final_water": result["final_water"],
        "final_food": result["final_food"],
        "reach_day": result["reach_day"],
        "length": result["reach_day"],
        "return": objective - config_cls.INIT_MONEY,
    }


def solve_theoretical_plan(
    level: int,
    weather_seq: Sequence[int],
    time_limit: int = 60,
) -> Dict[str, Any]:
    """求解已知天气下的理论最优并返回动作方案。"""
    config_cls = _pick_config(level)
    prob, v = _build_model(config_cls, weather_seq)

    try:
        solver = pulp.HiGHS(
            msg=False,
            timeLimit=time_limit,
            options=["--mip_rel_gap", "0.0001", "--mip_abs_gap", "0.1"],
        )
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)

    prob.solve(solver)
    result = _solve_problem(prob, v, config_cls, want_plan=True)

    return {
        "status": result["status"],
        "is_proven_optimal": result["is_proven_optimal"],
        "reached": result["reached"],
        "reach_day": result["reach_day"],
        "plan": result["plan"],
    }


def solve_theoretical_optimal_with_config(
    config_cls,
    weather_seq: Sequence[int],
    base_consumption: Mapping[int, Tuple[int, int]] | None = None,
    time_limit: int = 60,
    solver_options: Sequence[str] | None = None,
) -> Dict[str, float]:
    """求解自定义配置的理论最优（用于扩展关卡测试）。"""
    prob, v = _build_model(config_cls, weather_seq, base_consumption=base_consumption)

    if solver_options is None:
        solver_options = ["--mip_rel_gap", "0.0001", "--mip_abs_gap", "0.1"]

    try:
        solver = pulp.HiGHS(
            msg=False,
            timeLimit=time_limit,
            options=list(solver_options),
        )
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)

    prob.solve(solver)
    result = _solve_problem(prob, v, config_cls, want_plan=False)
    objective = result["objective"]

    return {
        "status": result["status"],
        "is_proven_optimal": result["is_proven_optimal"],
        "objective": objective,
        "reached": result["reached"],
        "final_money": result["final_money"],
        "final_water": result["final_water"],
        "final_food": result["final_food"],
        "reach_day": result["reach_day"],
        "length": result["reach_day"],
        "return": objective - config_cls.INIT_MONEY,
    }


def _pick_config(level: int):
    if level == 3:
        return Level3Config
    if level == 35:
        return Level35Config
    if level == 4:
        return Level4Config
    raise ValueError(f"Unsupported level: {level}")


def _is_proven_optimal(prob: pulp.LpProblem) -> bool:
    return prob.status == pulp.LpStatusOptimal


def _solve_problem(
    prob: pulp.LpProblem,
    v: Dict[str, Dict],
    config_cls,
    want_plan: bool,
) -> Dict[str, Any]:
    status_name = pulp.LpStatus.get(prob.status, "Unknown")
    proven_optimal = _is_proven_optimal(prob)
    objective = float(pulp.value(prob.objective)) if proven_optimal else float("nan")

    def _safe_value(var) -> float:
        value = pulp.value(var)
        return float(value) if value is not None else float("nan")

    reached = False
    if proven_optimal and "reached" in v:
        r_last = pulp.value(v["reached"][config_cls.NUM_DAYS])
        reached = bool(r_last is not None and r_last > 0.5)

    final_money = _safe_value(v["money"][config_cls.NUM_DAYS]) if proven_optimal else float("nan")
    final_water = _safe_value(v["water"][config_cls.NUM_DAYS]) if proven_optimal else float("nan")
    final_food = _safe_value(v["food"][config_cls.NUM_DAYS]) if proven_optimal else float("nan")

    reach_day = config_cls.NUM_DAYS
    if proven_optimal:
        for day in range(config_cls.NUM_DAYS + 1):
            r_day = pulp.value(v["reached"][day])
            if r_day is not None and r_day > 0.5:
                reach_day = day
                break

    plan: List[Dict[str, Any]] = []
    if want_plan and proven_optimal:
        plan, reach_day = _extract_action_plan(v, config_cls)

    return {
        "status": status_name,
        "is_proven_optimal": proven_optimal,
        "objective": objective,
        "reached": reached,
        "final_money": final_money,
        "final_water": final_water,
        "final_food": final_food,
        "reach_day": int(reach_day),
        "plan": plan,
    }


def _build_model(
    config_cls,
    weather_seq: Sequence[int],
    base_consumption: Mapping[int, Tuple[int, int]] | None = None,
) -> Tuple[pulp.LpProblem, Dict[str, Dict]]:
    if len(weather_seq) != config_cls.NUM_DAYS:
        raise ValueError(
            f"weather_seq 长度错误: {len(weather_seq)} != NUM_DAYS({config_cls.NUM_DAYS})"
        )

    if base_consumption is None:
        base_consumption = BASE_CONSUMPTION

    num_nodes = config_cls.NUM_NODES
    num_days = config_cls.NUM_DAYS
    start = config_cls.START
    end = config_cls.END
    mines = list(config_cls.MINES)
    villages = list(config_cls.VILLAGES)

    conn = get_adjacency_matrix(num_nodes, config_cls.EDGES)

    days = range(num_days + 1)
    act_days = range(1, num_days + 1)

    prob = pulp.LpProblem("Desert_Oracle", pulp.LpMaximize)

    v: Dict[str, Dict] = {}
    v["loc"] = pulp.LpVariable.dicts("loc", (range(num_nodes), days), cat="Binary")
    v["stay"] = pulp.LpVariable.dicts("stay", (range(num_nodes), act_days), cat="Binary")
    v["move"] = pulp.LpVariable.dicts(
        "move", (range(num_nodes), range(num_nodes), act_days), cat="Binary"
    )

    v["mine"] = pulp.LpVariable.dicts("mine", act_days, cat="Binary")
    v["eff_w_cons"] = pulp.LpVariable.dicts("ew", act_days, lowBound=0, cat="Continuous")
    v["eff_f_cons"] = pulp.LpVariable.dicts("ef", act_days, lowBound=0, cat="Continuous")

    v["water"] = pulp.LpVariable.dicts("w", days, lowBound=0, cat="Integer")
    v["food"] = pulp.LpVariable.dicts("f", days, lowBound=0, cat="Integer")
    v["money"] = pulp.LpVariable.dicts("m", days, lowBound=0, cat="Continuous")

    v["buy_w"] = pulp.LpVariable.dicts("bw", days, lowBound=0, cat="Integer")
    v["buy_f"] = pulp.LpVariable.dicts("bf", days, lowBound=0, cat="Integer")

    v["reached"] = pulp.LpVariable.dicts("r", days, cat="Binary")

    # 初始
    prob += v["loc"][start][0] == 1
    for i in range(num_nodes):
        if i != start:
            prob += v["loc"][i][0] == 0

    prob += v["water"][0] == v["buy_w"][0]
    prob += v["food"][0] == v["buy_f"][0]
    prob += (
        v["money"][0]
        == config_cls.INIT_MONEY
        - config_cls.WATER_PRICE_BASE * v["buy_w"][0]
        - config_cls.FOOD_PRICE_BASE * v["buy_f"][0]
    )
    prob += (
        config_cls.WATER_WEIGHT * v["water"][0] + config_cls.FOOD_WEIGHT * v["food"][0]
        <= config_cls.WEIGHT_LIMIT
    )
    prob += v["reached"][0] == 0

    # t>=1 时起点不可买，仅村庄可买
    for t in act_days:
        w_type = weather_seq[t - 1]
        base_w, base_f = base_consumption[w_type]

        # 位置唯一
        prob += pulp.lpSum(v["loc"][i][t] for i in range(num_nodes)) == 1

        # 流平衡
        for j in range(num_nodes):
            flow_in = pulp.lpSum(v["move"][i][j][t] for i in range(num_nodes) if conn[i, j] == 1)
            prob += v["loc"][j][t] == v["stay"][j][t] + flow_in

        for i in range(num_nodes):
            flow_out = pulp.lpSum(v["move"][i][j][t] for j in range(num_nodes) if conn[i, j] == 1)
            prob += v["loc"][i][t - 1] == v["stay"][i][t] + flow_out

        total_stay = pulp.lpSum(v["stay"][i][t] for i in range(num_nodes))
        total_move = pulp.lpSum(
            v["move"][i][j][t]
            for i in range(num_nodes)
            for j in range(num_nodes)
            if conn[i, j] == 1
        )
        prob += total_stay + total_move == 1

        # 挖矿：在矿山停留才可挖
        if mines:
            prob += v["mine"][t] <= pulp.lpSum(v["stay"][m][t] for m in mines)
        else:
            prob += v["mine"][t] == 0

        # 沙暴禁移
        if w_type == Weather.SANDSTORM:
            prob += total_move == 0

        # 购买限制
        m_buy = 1000
        if villages:
            loc_v = pulp.lpSum(v["loc"][vv][t] for vv in villages)
            prob += v["buy_w"][t] <= m_buy * loc_v
            prob += v["buy_f"][t] <= m_buy * loc_v
        else:
            prob += v["buy_w"][t] == 0
            prob += v["buy_f"][t] == 0

        purchase_cost = (
            2 * config_cls.WATER_PRICE_BASE * v["buy_w"][t]
            + 2 * config_cls.FOOD_PRICE_BASE * v["buy_f"][t]
        )
        prob += purchase_cost <= v["money"][t - 1]

        prev_r = v["reached"][t - 1]

        cons_factor = total_stay + 2 * total_move + 2 * v["mine"][t]
        water_cons = base_w * cons_factor
        food_cons = base_f * cons_factor

        m_eff = 100
        prob += v["eff_w_cons"][t] <= water_cons
        prob += v["eff_w_cons"][t] <= m_eff * (1 - prev_r)
        prob += v["eff_w_cons"][t] >= water_cons - m_eff * prev_r

        prob += v["eff_f_cons"][t] <= food_cons
        prob += v["eff_f_cons"][t] <= m_eff * (1 - prev_r)
        prob += v["eff_f_cons"][t] >= food_cons - m_eff * prev_r

        prob += v["water"][t - 1] >= v["eff_w_cons"][t]
        prob += v["food"][t - 1] >= v["eff_f_cons"][t]

        prob += v["water"][t] == v["water"][t - 1] - v["eff_w_cons"][t] + v["buy_w"][t]
        prob += v["food"][t] == v["food"][t - 1] - v["eff_f_cons"][t] + v["buy_f"][t]

        prob += (
            config_cls.WATER_WEIGHT * v["water"][t - 1] + config_cls.FOOD_WEIGHT * v["food"][t - 1]
            <= config_cls.WEIGHT_LIMIT
        )
        prob += (
            config_cls.WATER_WEIGHT * v["water"][t] + config_cls.FOOD_WEIGHT * v["food"][t]
            <= config_cls.WEIGHT_LIMIT
        )

        prob += (
            v["money"][t]
            == v["money"][t - 1] + config_cls.MINE_INCOME * v["mine"][t] - purchase_cost
        )

        # 到达与冻结
        prob += v["reached"][t] >= v["loc"][end][t]
        prob += v["reached"][t] >= v["reached"][t - 1]
        prob += v["reached"][t] <= v["loc"][end][t] + v["reached"][t - 1]

        big_m = 50000
        prob += v["loc"][end][t] >= prev_r
        for i in range(num_nodes):
            if i != end:
                prob += v["loc"][i][t] <= 1 - prev_r

        for key in ["water", "food", "money"]:
            var = v[key]
            prob += var[t] - var[t - 1] <= big_m * (1 - prev_r)
            prob += var[t - 1] - var[t] <= big_m * (1 - prev_r)

        prob += v["mine"][t] <= 1 - prev_r
        prob += v["buy_w"][t] <= big_m * (1 - prev_r)
        prob += v["buy_f"][t] <= big_m * (1 - prev_r)

    prob += v["reached"][num_days] == 1

    # 目标：最终资金 + 剩余物资折现
    prob += (
        v["money"][num_days]
        + 0.5 * config_cls.WATER_PRICE_BASE * v["water"][num_days]
        + 0.5 * config_cls.FOOD_PRICE_BASE * v["food"][num_days]
    )

    return prob, v


def _extract_action_plan(v: Dict[str, Dict], config_cls) -> Tuple[List[Dict[str, Any]], int]:
    num_days = config_cls.NUM_DAYS
    num_nodes = config_cls.NUM_NODES
    start = config_cls.START

    def _safe_int(value) -> int:
        if value is None:
            return 0
        return int(round(float(value)))

    actions: List[Dict[str, Any]] = []

    buy_w0 = _safe_int(pulp.value(v["buy_w"][0]))
    buy_f0 = _safe_int(pulp.value(v["buy_f"][0]))
    actions.append(
        {
            "move": int(start),
            "mine": False,
            "buy_water": buy_w0,
            "buy_food": buy_f0,
        }
    )

    reach_day = num_days
    for t in range(1, num_days + 1):
        loc_t = 0
        best_loc_val = -1.0
        for i in range(num_nodes):
            val = pulp.value(v["loc"][i][t])
            if val is not None and val > best_loc_val:
                best_loc_val = float(val)
                loc_t = int(i)

        move_target = loc_t
        best_move_val = -1.0
        for i in range(num_nodes):
            for j in range(num_nodes):
                val = pulp.value(v["move"][i][j][t])
                if val is not None and val > best_move_val:
                    best_move_val = float(val)
                    move_target = int(j)
        if best_move_val < 0.5:
            move_target = loc_t

        mine_val = pulp.value(v["mine"][t])
        buy_w = _safe_int(pulp.value(v["buy_w"][t]))
        buy_f = _safe_int(pulp.value(v["buy_f"][t]))

        actions.append(
            {
                "move": move_target,
                "mine": bool(mine_val is not None and mine_val > 0.5),
                "buy_water": buy_w,
                "buy_food": buy_f,
            }
        )

        r_day = pulp.value(v["reached"][t])
        if r_day is not None and r_day > 0.5:
            reach_day = t
            break

    return actions, int(reach_day)
