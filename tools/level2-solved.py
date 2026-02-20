import pulp

# ============================================
# 常量定义：第二关游戏参数与地图数据
# ============================================

# 节点索引 (逻辑中使用 0-based，但在 EDGES 列表中使用 1-based)
START = 0  # 节点 1
END = 63  # 节点 64
# 支持任意数量：使用列表存储矿山和村庄的 0-based 索引
MINES = [29, 54]  # 对应地图节点 30, 55
VILLAGES = [38, 61]  # 对应地图节点 39, 62

NUM_NODES = 64
NUM_DAYS = 30

# 资源与资金参数
WEIGHT_LIMIT = 1200
INIT_MONEY = 10000
MINE_INCOME = 1000

# 资源属性：重量，基准价格
WATER_WEIGHT, WATER_PRICE_BASE = 3, 5
FOOD_WEIGHT, FOOD_PRICE_BASE = 2, 10
# 村庄价格为基准的2倍
WATER_PRICE_VILLAGE = 2 * WATER_PRICE_BASE
FOOD_PRICE_VILLAGE = 2 * FOOD_PRICE_BASE

# 天气序列 (1=晴朗, 2=高温, 3=沙暴)，索引对应第1-30天
WEATHER = [2, 2, 1, 3, 1, 2, 3, 1, 2, 2, 3, 2, 1, 2, 2, 2, 3, 3, 2, 2, 1, 1, 2, 1, 3, 2, 1, 1, 2, 2]

# 基础消耗量 (水, 食物) 按天气类型
BASE_CONS = {
    1: (5, 7),  # 晴朗
    2: (8, 6),  # 高温
    3: (10, 10),  # 沙暴
}

# 地图边 (1-based，仅用于生成邻接矩阵)
# 根据题目附件第二关地图整理

EDGES = [
    (1, 2),
    (1, 9),
    (2, 3),
    (2, 10),
    (3, 4),
    (3, 11),
    (4, 5),
    (4, 12),
    (5, 6),
    (5, 13),
    (6, 7),
    (6, 14),
    (7, 8),
    (7, 15),
    (8, 16),
    (9, 10),
    (9, 17),
    (9, 18),
    (10, 11),
    (10, 18),
    (10, 19),
    (11, 12),
    (11, 19),
    (11, 20),
    (12, 13),
    (12, 20),
    (12, 21),
    (13, 14),
    (13, 21),
    (13, 22),
    (14, 15),
    (14, 22),
    (14, 23),
    (15, 16),
    (15, 23),
    (15, 24),
    (16, 24),
    (17, 18),
    (17, 25),
    (18, 19),
    (18, 26),
    (19, 20),
    (19, 27),
    (20, 21),
    (20, 28),
    (21, 22),
    (21, 29),
    (22, 23),
    (22, 30),
    (23, 24),
    (23, 31),
    (24, 31),
    (24, 32),
    (25, 26),
    (25, 33),
    (25, 34),
    (26, 27),
    (26, 34),
    (26, 35),
    (27, 28),
    (27, 35),
    (27, 36),
    (28, 29),
    (28, 36),
    (28, 37),
    (29, 30),
    (29, 37),
    (29, 38),
    (30, 31),
    (30, 38),
    (30, 39),
    (31, 32),
    (31, 39),
    (31, 40),
    (32, 40),
    (33, 34),
    (33, 41),
    (34, 35),
    (34, 42),
    (35, 36),
    (35, 43),
    (36, 37),
    (36, 44),
    (37, 38),
    (37, 45),
    (38, 39),
    (38, 46),
    (39, 40),
    (39, 47),
    (40, 47),
    (40, 48),
    (41, 42),
    (41, 49),
    (41, 50),
    (42, 43),
    (42, 50),
    (42, 51),
    (43, 44),
    (43, 51),
    (43, 52),
    (44, 45),
    (44, 52),
    (44, 53),
    (45, 46),
    (45, 53),
    (45, 54),
    (46, 47),
    (46, 54),
    (46, 55),
    (47, 48),
    (47, 55),
    (47, 56),
    (48, 56),
    (49, 50),
    (49, 57),
    (50, 51),
    (50, 58),
    (51, 52),
    (51, 59),
    (52, 53),
    (52, 60),
    (53, 54),
    (53, 61),
    (54, 55),
    (54, 62),
    (55, 56),
    (55, 63),
    (56, 63),
    (56, 64),
]


# ============================================
# 辅助函数
# ============================================


def get_adjacency(n, edges):
    """生成0-based邻接矩阵"""
    conn = [[0] * n for _ in range(n)]
    for u, v in edges:
        u0, v0 = u - 1, v - 1
        if 0 <= u0 < n and 0 <= v0 < n:
            conn[u0][v0] = conn[v0][u0] = 1
    return conn


# ============================================
# 模型构建函数
# ============================================


def build_model():
    """
    构建沙漠穿越问题的MIP模型 (支持多个矿山和村庄)
    返回: (problem, variables_dict, data_dict)
    """
    # 初始化求解问题
    prob = pulp.LpProblem("Desert_Crossing_Level2", pulp.LpMaximize)

    # 邻接关系
    conn = get_adjacency(NUM_NODES, EDGES)

    # 索引集合
    days = range(NUM_DAYS + 1)  # 0..30
    act_days = range(1, NUM_DAYS + 1)  # 1..30

    # 存储所有变量的字典
    v = {}

    # --- 决策变量定义 ---

    # 位置状态: loc[i,t] = 1 表示第t天位于节点i
    v["loc"] = pulp.LpVariable.dicts("loc", (range(NUM_NODES), days), cat="Binary")

    # 停留决策: stay[i,t] = 1 表示第t天在节点i停留
    v["stay"] = pulp.LpVariable.dicts("stay", (range(NUM_NODES), act_days), cat="Binary")

    # 移动决策: move[i,j,t] = 1 表示第t天从i移动到j
    v["move"] = pulp.LpVariable.dicts(
        "move", (range(NUM_NODES), range(NUM_NODES), act_days), cat="Binary"
    )

    # 挖矿决策
    v["mine"] = pulp.LpVariable.dicts("mine", act_days, cat="Binary")

    # 有效消耗（用于到达后不消耗的线性化）
    v["eff_w_cons"] = pulp.LpVariable.dicts("ew", act_days, lowBound=0, cat="Continuous")
    v["eff_f_cons"] = pulp.LpVariable.dicts("ef", act_days, lowBound=0, cat="Continuous")

    # 资源量：规则(2) 最小计量单位均为箱，故水、食物为整数；资金(元)连续
    v["water"] = pulp.LpVariable.dicts("w", days, lowBound=0, cat="Integer")
    v["food"] = pulp.LpVariable.dicts("f", days, lowBound=0, cat="Integer")
    v["money"] = pulp.LpVariable.dicts("m", days, lowBound=0, cat="Continuous")

    # 购买量（整数）
    v["buy_w"] = pulp.LpVariable.dicts("bw", days, lowBound=0, cat="Integer")
    v["buy_f"] = pulp.LpVariable.dicts("bf", days, lowBound=0, cat="Integer")

    # 到达标记
    v["reached"] = pulp.LpVariable.dicts("r", days, cat="Binary")

    # ==========================================
    # 约束构建（使用 addConstraint 避免缩进问题）
    # ==========================================

    # --- 初始条件 (第0天) ---
    prob.addConstraint(v["loc"][START][0] == 1, name="start_at_origin")

    for i in range(NUM_NODES):
        if i != START:
            prob.addConstraint(v["loc"][i][0] == 0, name=f"start_not_at_{i}")

    # 第0天购买（基准价格）与资源初始化
    prob.addConstraint(v["water"][0] == v["buy_w"][0], name="init_water")
    prob.addConstraint(v["food"][0] == v["buy_f"][0], name="init_food")
    prob.addConstraint(
        v["money"][0]
        == INIT_MONEY - WATER_PRICE_BASE * v["buy_w"][0] - FOOD_PRICE_BASE * v["buy_f"][0],
        name="init_money",
    )

    # 第0天负重限制（购买后）
    prob.addConstraint(
        WATER_WEIGHT * v["water"][0] + FOOD_WEIGHT * v["food"][0] <= WEIGHT_LIMIT,
        name="init_weight",
    )

    prob.addConstraint(v["reached"][0] == 0, name="init_not_reached")

    # --- 每日约束 (t = 1..30) ---
    for t in act_days:
        w_type = WEATHER[t - 1]
        base_w_cons, base_f_cons = BASE_CONS[w_type]

        # 1. 位置唯一性
        prob.addConstraint(
            pulp.lpSum([v["loc"][i][t] for i in range(NUM_NODES)]) == 1, name=f"unique_loc_day_{t}"
        )

        # 2. 流平衡约束
        for j in range(NUM_NODES):
            flow_in = pulp.lpSum([v["move"][i][j][t] for i in range(NUM_NODES) if conn[i][j]])
            prob.addConstraint(
                v["loc"][j][t] == v["stay"][j][t] + flow_in, name=f"flow_in_{j}_day_{t}"
            )

        for i in range(NUM_NODES):
            flow_out = pulp.lpSum([v["move"][i][j][t] for j in range(NUM_NODES) if conn[i][j]])
            prob.addConstraint(
                v["loc"][i][t - 1] == v["stay"][i][t] + flow_out, name=f"flow_out_{i}_day_{t}"
            )

        # 3. 动作互斥（停留或移动）
        total_stay = pulp.lpSum([v["stay"][i][t] for i in range(NUM_NODES)])
        total_move = pulp.lpSum(
            [v["move"][i][j][t] for i in range(NUM_NODES) for j in range(NUM_NODES) if conn[i][j]]
        )
        prob.addConstraint(total_stay + total_move == 1, name=f"action_mutex_day_{t}")

        # 4. 挖矿约束（支持任意多个矿井）
        # mine[t] <= sum(stay[m][t] for m in MINES)
        prob.addConstraint(
            v["mine"][t] <= pulp.lpSum([v["stay"][m][t] for m in MINES]),
            name=f"mine_at_any_mine_day_{t}",
        )

        # 5. 沙暴日限制
        if w_type == 3:
            prob.addConstraint(total_move == 0, name=f"sandstorm_no_move_day_{t}")

        # 6. 村庄购买约束（支持任意多个村庄）
        # buy[t] <= M_buy * sum(loc[v][t] for v in VILLAGES)
        M_buy = 500
        loc_at_villages = pulp.lpSum([v["loc"][village][t] for village in VILLAGES])
        prob.addConstraint(v["buy_w"][t] <= M_buy * loc_at_villages, name=f"village_buy_w_day_{t}")
        prob.addConstraint(v["buy_f"][t] <= M_buy * loc_at_villages, name=f"village_buy_f_day_{t}")

        # 购买资金限制（用昨天的钱）
        purchase_cost = WATER_PRICE_VILLAGE * v["buy_w"][t] + FOOD_PRICE_VILLAGE * v["buy_f"][t]
        prob.addConstraint(purchase_cost <= v["money"][t - 1], name=f"money_for_buy_day_{t}")

        # prev_r：昨日是否已到达
        prev_r = v["reached"][t - 1]

        # 7. 资源消耗计算
        cons_factor = total_stay + 2 * total_move + 2 * v["mine"][t]
        water_cons = base_w_cons * cons_factor
        food_cons = base_f_cons * cons_factor

        M_eff = 50
        prob.addConstraint(v["eff_w_cons"][t] <= water_cons, name=f"eff_w_ub_{t}")
        prob.addConstraint(v["eff_w_cons"][t] <= M_eff * (1 - prev_r), name=f"eff_w_reached_{t}")
        prob.addConstraint(v["eff_w_cons"][t] >= water_cons - M_eff * prev_r, name=f"eff_w_lb_{t}")
        prob.addConstraint(v["eff_f_cons"][t] <= food_cons, name=f"eff_f_ub_{t}")
        prob.addConstraint(v["eff_f_cons"][t] <= M_eff * (1 - prev_r), name=f"eff_f_reached_{t}")
        prob.addConstraint(v["eff_f_cons"][t] >= food_cons - M_eff * prev_r, name=f"eff_f_lb_{t}")

        prob.addConstraint(
            v["water"][t - 1] >= v["eff_w_cons"][t], name=f"water_sufficient_before_cons_day_{t}"
        )
        prob.addConstraint(
            v["food"][t - 1] >= v["eff_f_cons"][t], name=f"food_sufficient_before_cons_day_{t}"
        )

        prob.addConstraint(
            v["water"][t] == v["water"][t - 1] - v["eff_w_cons"][t] + v["buy_w"][t],
            name=f"water_balance_day_{t}",
        )
        prob.addConstraint(
            v["food"][t] == v["food"][t - 1] - v["eff_f_cons"][t] + v["buy_f"][t],
            name=f"food_balance_day_{t}",
        )

        prob.addConstraint(v["water"][t] >= 0, name=f"water_nonneg_day_{t}")
        prob.addConstraint(v["food"][t] >= 0, name=f"food_nonneg_day_{t}")

        prob.addConstraint(
            WATER_WEIGHT * v["water"][t - 1] + FOOD_WEIGHT * v["food"][t - 1] <= WEIGHT_LIMIT,
            name=f"weight_start_day_{t}",
        )
        prob.addConstraint(
            WATER_WEIGHT * v["water"][t] + FOOD_WEIGHT * v["food"][t] <= WEIGHT_LIMIT,
            name=f"weight_end_day_{t}",
        )

        # 8. 资金平衡
        prob.addConstraint(
            v["money"][t] == v["money"][t - 1] + MINE_INCOME * v["mine"][t] - purchase_cost,
            name=f"money_balance_day_{t}",
        )

        # 9. 到达逻辑与冻结机制
        prob.addConstraint(v["reached"][t] >= v["loc"][END][t], name=f"reach_if_at_end_day_{t}")
        prob.addConstraint(v["reached"][t] >= v["reached"][t - 1], name=f"reach_monotone_day_{t}")
        prob.addConstraint(
            v["reached"][t] <= v["loc"][END][t] + v["reached"][t - 1], name=f"reach_exact_day_{t}"
        )

        bigM = 20000
        prob.addConstraint(v["loc"][END][t] >= prev_r, name=f"freeze_at_end_day_{t}")
        for i in range(NUM_NODES):
            if i != END:
                prob.addConstraint(v["loc"][i][t] <= 1 - prev_r, name=f"freeze_not_at_{i}_day_{t}")

        for var_key in ["water", "food", "money"]:
            var = v[var_key]
            prob.addConstraint(
                var[t] - var[t - 1] <= bigM * (1 - prev_r), name=f"freeze_{var_key}_up_day_{t}"
            )
            prob.addConstraint(
                var[t - 1] - var[t] <= bigM * (1 - prev_r), name=f"freeze_{var_key}_down_day_{t}"
            )

        prob.addConstraint(v["mine"][t] <= 1 - prev_r, name=f"freeze_mine_day_{t}")
        prob.addConstraint(v["buy_w"][t] <= bigM * (1 - prev_r), name=f"freeze_buy_w_day_{t}")
        prob.addConstraint(v["buy_f"][t] <= bigM * (1 - prev_r), name=f"freeze_buy_f_day_{t}")

    prob.addConstraint(v["reached"][NUM_DAYS] == 1, name="must_reach_end")

    final_money = v["money"][NUM_DAYS]
    final_water_val = 0.5 * WATER_PRICE_BASE * v["water"][NUM_DAYS]
    final_food_val = 0.5 * FOOD_PRICE_BASE * v["food"][NUM_DAYS]
    prob += final_money + final_water_val + final_food_val, "total_profit"

    data = {"conn": conn, "days": days, "act_days": act_days}

    return prob, v, data
