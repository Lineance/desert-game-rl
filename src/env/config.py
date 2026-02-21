"""
问题2（第三关、第四关）配置文件
仅知当天天气的部分可观测决策问题
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np


# ==================== 天气类型 ====================
class Weather:
    SUNNY = 0  # 晴朗
    HOT = 1  # 高温
    SANDSTORM = 2  # 沙暴

    NAMES = ["晴朗", "高温", "沙暴"]


# ==================== 基础消耗 (水, 食物) ====================
# 基础消耗量：停留一天
BASE_CONSUMPTION = {
    Weather.SUNNY: (3, 4),  # 晴朗
    Weather.HOT: (9, 9),  # 高温
    Weather.SANDSTORM: (10, 10),  # 沙暴
}


# ==================== 第三关配置 ====================
class Level3Config:
    """第三关参数 - 需手动填写地图数据"""

    # 基础参数（第三关：简化版，10天内无沙暴）
    NUM_NODES = 13  # 节点数（根据实际地图修改）
    NUM_DAYS = 10  # 截止日期：第10天
    INIT_MONEY = 10000  # 初始资金：10000元
    WEIGHT_LIMIT = 1200  # 负重上限：1200千克
    MINE_INCOME = 200  # 挖矿基础收益：200元

    # 资源参数
    WATER_WEIGHT = 3
    WATER_PRICE_BASE = 5
    FOOD_WEIGHT = 2
    FOOD_PRICE_BASE = 10

    # 地点索引 (0-based)
    START = 0
    END = 12
    MINES: List[int] = [8]  # 矿山节点列表，需手动填写
    VILLAGES: List[int] = []  # 村庄节点列表，需手动填写

    # 地图边 (1-based，用于生成邻接矩阵)
    EDGES: List[Tuple[int, int]] = [
        (1, 5),
        (1, 2),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 8),
        (3, 9),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 6),
        (6, 7),
        (6, 12),
        (6, 13),
        (7, 11),
        (7, 12),
        (8, 9),
        (9, 10),
        (9, 11),
        (10, 11),
        (10, 13),
        (11, 12),
        (11, 13),
        (12, 13),
    ]

    # 天气模式（用于训练和模拟）
    # 第三关特殊规则：10天内不会出现沙暴天气
    # 因此沙暴概率设为0，只在晴朗和高温之间转移
    WEATHER_MODES = {
        "no_sandstorm": [0.5, 0.5, 0.0],  # 无沙暴模式：只有晴朗和高温
        "sunny_bias": [0.7, 0.3, 0.0],  # 晴天偏多
        "hot_bias": [0.3, 0.7, 0.0],  # 高温偏多
    }

    # 天气转移矩阵（第三关：10天内无沙暴）
    # 从任何状态转移到沙暴的概率为0
    # WEATHER_TRANSITION[i][j] = P(明天=j | 今天=i)
    WEATHER_TRANSITION = np.array(
        [
            [0.5, 0.5, 0.0],  # 今天晴朗 → 明天50%晴朗, 50%高温, 0%沙暴
            [0.5, 0.5, 0.0],  # 今天高温 → 明天50%晴朗, 50%高温, 0%沙暴
            [0.5, 0.5, 0.0],  # 今天沙暴（理论上不会出现，但保持矩阵完整）
        ]
    )


# ==================== 第3.5关配置 ====================


class Level35Config:
    """Level 3.5 - 课程学习中间态"""

    # 时间：足够挖矿回本，但不至于太长
    NUM_DAYS = 20  # Level 3(10) < 25 < Level 4(50)
    NUM_NODES = 17  # 中等地图（Level 3:13, Level 4:25+）

    # 经济参数
    INIT_MONEY = 10000
    WEIGHT_LIMIT = 1200
    MINE_INCOME = 600  # 介于 Level 3(200) 和 Level 4(1000) 之间

    # 资源（与 Level 4 一致，便于迁移）
    WATER_WEIGHT = 3
    WATER_PRICE_BASE = 5
    FOOD_WEIGHT = 2
    FOOD_PRICE_BASE = 10

    # 关键地点（设计成"去矿山有诱惑但有风险"）
    START = 0
    END = 16
    MINES = [11]  # 偏离主路径 3-4 步，挖矿 3-5 天才能回本
    VILLAGES = [7]  # 位于去矿山的分叉口附近（战略要地）

    # 地图拓扑（关键：主路径短但贫瘠，支路有矿山但需要绕行）
    EDGES = [
        # 主路径（去终点较快，但无补给）
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        # 矿山支路（绕远但有矿）
        (3, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),  # 到矿山
        (11, 12),
        (12, 5),  # 从矿山回主路径
        # 村庄在岔路口
        (7, 13),  # 村庄连接点
        # 终点前段
        (5, 14),
        (14, 15),
        (15, 16),
        (16, 17),  # 17 是终点
        # 一些捷径/环路增加路径选择
        (2, 13),
        (13, 14),
    ]

    # 天气：有沙暴但可预测（让 RL 学习信念更新）
    WEATHER_MODES = {
        "train": [0.4, 0.4, 0.2],  # 训练分布（20%沙暴）
        "val": [0.35, 0.35, 0.3],  # 验证分布（30%沙暴）
        "test": [0.33, 0.33, 0.34],  # 测试分布（近似均匀，最难）
        "train_easy": [0.5, 0.4, 0.1],  # 10% 沙暴，先让 RL 学会基本生存
        "train_medium": [0.4, 0.4, 0.2],  # 兼容别名：等价于train
        "eval": [0.35, 0.35, 0.3],  # 兼容别名：等价于val
    }

    # 关键：天气转移有模式（RL 可以学习预测，MPC 保守策略会错过机会）
    # 比如：沙暴后大概率晴天（适合冒险赶路）
    WEATHER_TRANSITION = np.array(
        [
            [0.5, 0.3, 0.2],  # 晴天 → 50%晴, 30%高温, 20%沙暴
            [0.3, 0.4, 0.3],  # 高温 → 30%晴, 40%高温, 30%沙暴
            [0.6, 0.3, 0.1],  # 沙暴 → 60%晴（恢复期）, 30%高温, 10%沙暴
        ]
    )


# ==================== 第四关配置 ====================
class Level4Config:
    """第四关参数"""

    NUM_NODES = 25  # 节点数（通常比第三关大）
    NUM_DAYS = 30  # 截止日期更长
    INIT_MONEY = 10000
    WEIGHT_LIMIT = 1200
    MINE_INCOME = 1000

    # 资源参数
    WATER_WEIGHT = 3
    WATER_PRICE_BASE = 5
    FOOD_WEIGHT = 2
    FOOD_PRICE_BASE = 10

    # 地点索引
    START = 0
    END = 24
    MINES: List[int] = [17]
    VILLAGES: List[int] = [13]

    # 地图边
    EDGES: List[Tuple[int, int]] = [
        (1, 2),
        (1, 6),
        (2, 3),
        (2, 7),
        (3, 4),
        (3, 8),
        (4, 5),
        (4, 9),
        (5, 10),
        (6, 7),
        (6, 11),
        (7, 8),
        (7, 12),
        (8, 9),
        (8, 13),
        (9, 10),
        (9, 14),
        (10, 15),
        (11, 12),
        (11, 16),
        (12, 13),
        (12, 17),
        (13, 14),
        (13, 18),
        (14, 15),
        (14, 19),
        (15, 20),
        (16, 17),
        (16, 21),
        (17, 18),
        (17, 22),
        (18, 19),
        (18, 23),
        (19, 20),
        (19, 24),
        (20, 25),
    ]

    # 天气模式
    WEATHER_MODES = {
        "balanced": [0.35, 0.35, 0.30],
        "unpredictable": [0.33, 0.33, 0.34],  # 更难预测
    }

    # 更随机的天气转移
    WEATHER_TRANSITION = np.array(
        [
            [0.4, 0.35, 0.25],
            [0.35, 0.4, 0.25],
            [0.35, 0.35, 0.3],
        ]
    )


# ==================== RL训练配置 ====================
class RLConfig:
    """强化学习超参数"""

    # 网络结构
    HIDDEN_DIM = 256  # RNN隐藏层维度
    BELIEF_DIM = 8  # 信念向量维度
    LSTM_LAYERS = 2  # LSTM层数

    # PPO参数
    GAMMA = 0.99  # 折扣因子
    GAE_LAMBDA = 0.95  # GAE参数
    CLIP_EPS = 0.2  # PPO裁剪参数（收紧，抑制策略突变）
    LR_ACTOR = 5e-5  # Actor学习率（再降一档，降低策略突变）
    LR_CRITIC = 2e-4  # Critic学习率（降低以稳定价值拟合）
    TARGET_KL = 0.01  # KL早停阈值（收紧）

    # 训练参数
    NUM_EPISODES = 5000  # 总训练回合数
    BATCH_SIZE = 64  # 批次大小
    EPOCHS_PER_UPDATE = 4  # 每次更新迭代次数（进一步抑制过拟合与发散）
    MAX_GRAD_NORM = 0.5  # 梯度裁剪

    # 探索参数（修复探索崩溃）
    ENTROPY_COEF = 0.03  # 熵系数（降低无效随机探索）
    MIN_ENTROPY = 0.5  # 新增：熵下限，防止归零

    # 课程学习（第三关简单，直接训练）
    USE_CURRICULUM = False  # 禁用课程学习，直接训练
    CURRICULUM_STAGES = [
        {"days": 10, "weather_known": False, "mode": "no_sandstorm"},
    ]

    # 评估参数
    EVAL_INTERVAL = 100  # 评估间隔
    EVAL_EPISODES = 50  # 评估回合数


# ==================== 辅助函数 ====================
def get_adjacency_matrix(num_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """生成0-based邻接矩阵"""
    conn = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    for u, v in edges:
        u0, v0 = u - 1, v - 1
        if 0 <= u0 < num_nodes and 0 <= v0 < num_nodes:
            conn[u0, v0] = conn[v0, u0] = 1
    return conn


def get_neighbors(conn: np.ndarray, node: int) -> List[int]:
    """获取节点的邻居列表"""
    return [i for i in range(conn.shape[0]) if conn[node, i] == 1]


def compute_shortest_distances(conn: np.ndarray, start: int) -> np.ndarray:
    """计算从起点到所有节点的最短距离（BFS）"""
    num_nodes = conn.shape[0]
    distances = np.full(num_nodes, -1, dtype=np.int32)
    distances[start] = 0
    queue = [start]

    while queue:
        current = queue.pop(0)
        for neighbor in range(num_nodes):
            if conn[current, neighbor] == 1 and distances[neighbor] == -1:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


# 默认配置
DEFAULT_CONFIG = Level3Config


# ==================== 产物目录配置 ====================
ARTIFACTS_ROOT = Path("./artifacts")
CHECKPOINTS_DIR = ARTIFACTS_ROOT / "checkpoints"
LOGS_DIR = ARTIFACTS_ROOT / "logs"
RESULTS_DIR = ARTIFACTS_ROOT / "results"
