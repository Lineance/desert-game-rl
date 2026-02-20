import pulp

# ============================================
# 常量定义：游戏参数与地图数据
# ============================================

# 节点索引 (0-based)
START, MINE, VILLAGE, END = 0, 11, 14, 26
NUM_NODES = 27
NUM_DAYS = 30

# 资源与资金参数
WEIGHT_LIMIT = 1200
INIT_MONEY = 10000
MINE_INCOME = 1000

# 资源属性：重量(kg)，基准价格(元)
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
EDGES = [
    (1, 2),
    (1, 25),
    (2, 3),
    (3, 4),
    (3, 25),
    (4, 5),
    (4, 24),
    (4, 25),
    (5, 6),
    (5, 24),
    (6, 7),
    (6, 23),
    (6, 24),
    (7, 8),
    (7, 22),
    (8, 9),
    (8, 22),
    (9, 10),
    (9, 15),
    (9, 17),
    (9, 16),
    (9, 21),
    (9, 22),
    (10, 11),
    (10, 13),
    (10, 15),
    (11, 12),
    (11, 13),
    (12, 13),
    (12, 14),
    (13, 14),
    (13, 15),
    (14, 15),
    (14, 16),
    (15, 16),
    (16, 17),
    (16, 18),
    (17, 18),
    (17, 21),
    (18, 19),
    (18, 20),
    (19, 20),
    (20, 21),
    (21, 22),
    (21, 23),
    (21, 27),
    (22, 23),
    (23, 24),
    (23, 26),
    (24, 25),
    (24, 26),
    (25, 26),
    (26, 27),
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
    构建沙漠穿越问题的MIP模型
    返回: (problem, variables_dict, data_dict)
    """
    # 初始化求解问题
    prob = pulp.LpProblem("Desert_Crossing", pulp.LpMaximize)

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

        # 4. 挖矿约束（到达当天不能挖矿由 mine<=stay[MINE] 隐含保证，无需额外约束）
        prob.addConstraint(v["mine"][t] <= v["stay"][MINE][t], name=f"mine_at_mine_day_{t}")

        # 5. 沙暴日限制
        if w_type == 3:
            prob.addConstraint(total_move == 0, name=f"sandstorm_no_move_day_{t}")

        # 6. 村庄购买约束（仅村庄可购买，负重在下方每日约束中检查）
        M_buy = 500
        prob.addConstraint(
            v["buy_w"][t] <= M_buy * v["loc"][VILLAGE][t], name=f"village_buy_w_day_{t}"
        )
        prob.addConstraint(
            v["buy_f"][t] <= M_buy * v["loc"][VILLAGE][t], name=f"village_buy_f_day_{t}"
        )

        # 购买资金限制（用昨天的钱）
        purchase_cost = WATER_PRICE_VILLAGE * v["buy_w"][t] + FOOD_PRICE_VILLAGE * v["buy_f"][t]
        prob.addConstraint(purchase_cost <= v["money"][t - 1], name=f"money_for_buy_day_{t}")

        # prev_r：昨日是否已到达（规则1：到达后游戏结束，不再消耗资源）
        prev_r = v["reached"][t - 1]

        # 7. 资源消耗计算
        # 系数：停留1倍，移动2倍，挖矿3倍；到达后游戏结束，消耗为0
        cons_factor = total_stay + 2 * total_move + 2 * v["mine"][t]
        water_cons = base_w_cons * cons_factor
        food_cons = base_f_cons * cons_factor

        # 有效消耗线性化：eff = cons when prev_r=0, eff=0 when prev_r=1
        M_eff = 50
        prob.addConstraint(v["eff_w_cons"][t] <= water_cons, name=f"eff_w_ub_{t}")
        prob.addConstraint(v["eff_w_cons"][t] <= M_eff * (1 - prev_r), name=f"eff_w_reached_{t}")
        prob.addConstraint(v["eff_w_cons"][t] >= water_cons - M_eff * prev_r, name=f"eff_w_lb_{t}")
        prob.addConstraint(v["eff_f_cons"][t] <= food_cons, name=f"eff_f_ub_{t}")
        prob.addConstraint(v["eff_f_cons"][t] <= M_eff * (1 - prev_r), name=f"eff_f_reached_{t}")
        prob.addConstraint(v["eff_f_cons"][t] >= food_cons - M_eff * prev_r, name=f"eff_f_lb_{t}")

        # 规则(2)：先消耗后购买，消耗前必须有足够资源（不能消耗不存在的物资）
        prob.addConstraint(
            v["water"][t - 1] >= v["eff_w_cons"][t], name=f"water_sufficient_before_cons_day_{t}"
        )
        prob.addConstraint(
            v["food"][t - 1] >= v["eff_f_cons"][t], name=f"food_sufficient_before_cons_day_{t}"
        )

        # 资源平衡（到达前：先消耗后购买；到达后：状态冻结，不消耗）
        prob.addConstraint(
            v["water"][t] == v["water"][t - 1] - v["eff_w_cons"][t] + v["buy_w"][t],
            name=f"water_balance_day_{t}",
        )
        prob.addConstraint(
            v["food"][t] == v["food"][t - 1] - v["eff_f_cons"][t] + v["buy_f"][t],
            name=f"food_balance_day_{t}",
        )

        # 非负约束（安全冗余）
        prob.addConstraint(v["water"][t] >= 0, name=f"water_nonneg_day_{t}")
        prob.addConstraint(v["food"][t] >= 0, name=f"food_nonneg_day_{t}")

        # 每日负重限制（规则2：先消耗后购买，检查每日开始与结束时的负重）
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

        # 冻结机制：一旦到达，必须停留在终点，状态冻结（prev_r 已在上方定义）
        bigM = 20000

        prob.addConstraint(v["loc"][END][t] >= prev_r, name=f"freeze_at_end_day_{t}")
        for i in range(NUM_NODES):
            if i != END:
                prob.addConstraint(v["loc"][i][t] <= 1 - prev_r, name=f"freeze_not_at_{i}_day_{t}")

        # 资源冻结
        for var_key in ["water", "food", "money"]:
            var = v[var_key]
            prob.addConstraint(
                var[t] - var[t - 1] <= bigM * (1 - prev_r), name=f"freeze_{var_key}_up_day_{t}"
            )
            prob.addConstraint(
                var[t - 1] - var[t] <= bigM * (1 - prev_r), name=f"freeze_{var_key}_down_day_{t}"
            )

        # 行动冻结
        prob.addConstraint(v["mine"][t] <= 1 - prev_r, name=f"freeze_mine_day_{t}")
        prob.addConstraint(v["buy_w"][t] <= bigM * (1 - prev_r), name=f"freeze_buy_w_day_{t}")
        prob.addConstraint(v["buy_f"][t] <= bigM * (1 - prev_r), name=f"freeze_buy_f_day_{t}")

    # --- 终止条件 ---
    # 规则(1)：必须在截止日期或之前到达；可提前到达，到达后游戏结束，状态冻结
    prob.addConstraint(v["reached"][NUM_DAYS] == 1, name="must_reach_end")

    # --- 目标函数 ---
    # 最大化：最终现金 + 剩余资源退回价值（基准价的一半）
    final_money = v["money"][NUM_DAYS]
    final_water_val = 0.5 * WATER_PRICE_BASE * v["water"][NUM_DAYS]
    final_food_val = 0.5 * FOOD_PRICE_BASE * v["food"][NUM_DAYS]
    prob += final_money + final_water_val + final_food_val, "total_profit"

    # 打包数据字典供后续使用
    data = {"conn": conn, "days": days, "act_days": act_days}

    return prob, v, data


# ============================================
# 求解与结果提取函数
# ============================================


def solve_and_report(prob, v, data, time_limit=300):
    """
    执行求解并输出详细报告
    """
    print("=" * 60)
    print("开始求解...")
    print(f"问题规模: {NUM_NODES} 节点, {NUM_DAYS} 天")

    # 配置 HiGHS 求解器（高精度）
    try:
        solver = pulp.HiGHS(
            msg=True,
            timeLimit=time_limit,
            threads=20,
            options=[
                "--mip_rel_gap",
                "0.00001",  # 强制全局最优
                "--mip_abs_gap",
                "0.00001",  # 绝对间隙为0
                "--primal_feasibility_tolerance",
                "1e-9",
                "--dual_feasibility_tolerance",
                "1e-9",
                "--random_seed",
                "42",
            ],
        )
        print("使用求解器: HiGHS (高精度模式)")
    except Exception as e:
        print(f"HiGHS 初始化失败 ({e})，回退到 CBC")
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit, threads=4)

    # 求解
    prob.solve(solver)

    # 状态报告
    print("\n" + "=" * 60)
    status = pulp.LpStatus[prob.status]
    print(f"求解状态: {status}")

    if prob.status != pulp.LpStatusOptimal:
        print("警告: 未找到最优解")
        if prob.status == pulp.LpStatusInfeasible:
            print("问题无可行解，请检查约束")
            return
        print("尝试输出当前找到的解（如有）...")

    # 提取结果
    # 1. 路径提取
    path = []
    for t in data["days"]:
        node = None
        for i in range(NUM_NODES):
            if pulp.value(v["loc"][i][t]) > 0.5:
                node = i
                break
        path.append(node + 1 if node is not None else -1)  # 转回1-based

    # 2. 购买记录
    buys = []
    for t in data["days"]:
        bw = pulp.value(v["buy_w"][t])
        bf = pulp.value(v["buy_f"][t])
        if bw > 0.1 or bf > 0.1:
            loc_name = "起点" if t == 0 else ("村庄" if path[t] - 1 == VILLAGE else "其他")
            buys.append((t, loc_name, int(round(bw)), int(round(bf))))

    # 3. 挖矿记录
    mines = [t for t in data["act_days"] if pulp.value(v["mine"][t]) > 0.5]

    # 4. 最终数值
    final_obj = pulp.value(prob.objective)
    final_money = pulp.value(v["money"][NUM_DAYS])
    final_water = pulp.value(v["water"][NUM_DAYS])
    final_food = pulp.value(v["food"][NUM_DAYS])

    # 找到实际到达天数
    actual_arrival = None
    for t in data["days"]:
        if pulp.value(v["reached"][t]) > 0.5:
            actual_arrival = t
            break

    # 输出报告
    print(f"\n最优目标值: {final_obj:.2f}")
    print(f"实际到达终点天数: 第 {actual_arrival} 天" if actual_arrival else "未记录到达")
    print(f"第{NUM_DAYS}天状态:")
    print(f"  - 现金: {final_money:.2f}")
    print(f"  - 剩余水: {final_water:.2f} 箱 (价值 {0.5 * WATER_PRICE_BASE * final_water:.2f})")
    print(f"  - 剩余食物: {final_food:.2f} 箱 (价值 {0.5 * FOOD_PRICE_BASE * final_food:.2f})")

    print(f"\n路径 ({len(path)} 个时间步):")
    path_str = " -> ".join(map(str, path))
    print(path_str)

    print("\n购买记录 [(天, 地点, 水, 食物)]:")
    for t, loc, w, f in buys:
        print(f"  第{t:2d}天 [{loc}]: 水 {w} 箱, 食物 {f} 箱")

    print(f"\n挖矿记录: 共 {len(mines)} 天")
    print(f"  列表: {mines}")

    # 逐日详细日志
    print("\n逐日详细日志:")
    print("-" * 70)
    print(f"{'天':>3} | {'节点':>4} | {'水':>6} | {'食物':>6} | {'资金':>8} | {'动作':<15}")
    print("-" * 70)

    for t in data["days"]:
        node = path[t]
        w_val = pulp.value(v["water"][t])
        f_val = pulp.value(v["food"][t])
        m_val = pulp.value(v["money"][t])

        # 判断动作
        if t == 0:
            action = "出发准备"
        else:
            actions = []
            if pulp.value(v["mine"][t]) > 0.5:
                actions.append("挖矿")

            # 检查是否移动
            prev_node = path[t - 1]
            if node != prev_node:
                actions.append(f"移动 {prev_node}->{node}")
            else:
                actions.append("停留")

            # 检查购买
            if any(b[0] == t for b in buys):
                actions.append("购买")

            action = "+".join(actions)

        print(f"{t:3d} | {node:4d} | {w_val:6.1f} | {f_val:6.1f} | {m_val:8.1f} | {action}")
