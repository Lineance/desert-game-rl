"""
贝叶斯信念模块
基于历史观测显式建模天气转移概率
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.env.config import Weather


@dataclass
class BeliefState:
    """信念状态"""

    probs: np.ndarray  # [P(晴朗), P(高温), P(沙暴)]
    confidence: float  # 置信度（基于观测历史长度）

    def __post_init__(self):
        self.probs = np.array(self.probs)
        self.probs = self.probs / self.probs.sum()  # 归一化


class WeatherBeliefModel:
    """
    天气信念模型
    使用贝叶斯更新或最大似然估计学习天气转移概率
    """

    def __init__(
        self,
        transition_prior: Optional[np.ndarray] = None,
        learning_rate: float = 0.1,
        use_dirichlet_prior: bool = True,
    ):
        """
        Args:
            transition_prior: 先验转移矩阵 [3x3]
            learning_rate: 信念更新学习率
            use_dirichlet_prior: 是否使用Dirichlet先验
        """
        # 转移计数矩阵 (用于最大似然估计)
        self.transition_counts = np.ones((3, 3))  # Laplace平滑

        # 当前转移矩阵估计
        if transition_prior is not None:
            self.transition_matrix = transition_prior.copy()
        else:
            # 默认：温和的季节性
            self.transition_matrix = np.array(
                [
                    [0.5, 0.3, 0.2],
                    [0.3, 0.4, 0.3],
                    [0.3, 0.3, 0.4],
                ]
            )

        self.learning_rate = learning_rate
        self.use_dirichlet_prior = use_dirichlet_prior

        # 历史观测
        self.weather_history: List[int] = []

        # 当前信念（明天天气的分布）
        self.current_belief = BeliefState(probs=np.ones(3) / 3, confidence=0.0)

    def reset(self):
        """重置信念状态"""
        self.transition_counts = np.ones((3, 3))
        self.weather_history = []
        self.current_belief = BeliefState(probs=np.ones(3) / 3, confidence=0.0)

    def update(self, weather_today: int):
        """
        根据今天观测到的天气更新信念

        Args:
            weather_today: 今天的天气 (0=晴朗, 1=高温, 2=沙暴)
        """
        self.weather_history.append(weather_today)

        # 更新转移计数
        if len(self.weather_history) >= 2:
            prev_weather = self.weather_history[-2]
            self.transition_counts[prev_weather, weather_today] += 1

        # 更新转移矩阵估计
        if self.use_dirichlet_prior and len(self.weather_history) >= 5:
            # 使用Dirichlet后验均值
            # P(transition) ∝ Dirichlet(counts)
            # E[P(j|i)] = (count[i,j] + alpha) / (sum(count[i,:]) + 3*alpha)
            row_sums = self.transition_counts.sum(axis=1, keepdims=True)
            new_transition = self.transition_counts / row_sums

            # 指数移动平均平滑
            self.transition_matrix = (
                1 - self.learning_rate
            ) * self.transition_matrix + self.learning_rate * new_transition

        # 预测明天天气
        self._predict_next_weather(weather_today)

        # 更新置信度
        self.current_belief.confidence = min(
            1.0,
            len(self.weather_history) / 10.0,  # 10天观测后达到满置信
        )

    def _predict_next_weather(self, weather_today: int):
        """基于转移矩阵预测明天天气"""
        probs = self.transition_matrix[weather_today]
        self.current_belief = BeliefState(probs=probs, confidence=self.current_belief.confidence)

    def get_belief_vector(self) -> np.ndarray:
        """
        获取当前信念向量（用于网络输入）
        Returns: [p_sunny, p_hot, p_sandstorm, confidence]
        """
        return np.concatenate([self.current_belief.probs, [self.current_belief.confidence]])

    def predict_weather_sequence(self, current_weather: int, horizon: int) -> np.ndarray:
        """
        预测未来horizon天的天气分布

        Args:
            current_weather: 当前天气
            horizon: 预测天数

        Returns:
            [horizon, 3] 概率分布
        """
        predictions = np.zeros((horizon, 3))

        # 初始分布
        probs = np.zeros(3)
        probs[current_weather] = 1.0

        for t in range(horizon):
            # 向前传播
            probs = probs @ self.transition_matrix
            predictions[t] = probs

        return predictions

    def expected_consumption(self, current_weather: int, base_consumption: dict) -> tuple:
        """
        计算预期资源消耗（考虑明天天气的不确定性）

        Args:
            current_weather: 当前天气
            base_consumption: 基础消耗字典

        Returns:
            (预期水消耗, 预期食物消耗)
        """
        from src.env.config import BASE_CONSUMPTION

        exp_water = 0.0
        exp_food = 0.0

        for w in range(3):
            prob = self.current_belief.probs[w]
            w_cons, f_cons = BASE_CONSUMPTION[w]
            exp_water += prob * w_cons
            exp_food += prob * f_cons

        return exp_water, exp_food

    def get_risk_measure(self, risk_type: str = "worst_case") -> int:
        """
        获取风险度量（用于风险厌恶决策）

        Args:
            risk_type: 'worst_case', 'cvar', 'expected'

        Returns:
            最可能的恶劣天气类型
        """
        if risk_type == "worst_case":
            # 返回概率最高的恶劣天气
            if self.current_belief.probs[Weather.SANDSTORM] > 0.3:
                return Weather.SANDSTORM
            elif self.current_belief.probs[Weather.HOT] > 0.4:
                return Weather.HOT
            else:
                return Weather.SUNNY
        elif risk_type == "cvar":
            # 条件风险价值：考虑尾部风险
            sorted_probs = np.sort(self.current_belief.probs)[::-1]
            cumsum = np.cumsum(sorted_probs)
            # 找到95%置信区间内的最坏情况
            idx = np.searchsorted(cumsum, 0.95)
            return np.argsort(self.current_belief.probs)[-1]
        else:
            return np.argmax(self.current_belief.probs)


class AdaptiveBeliefModel(WeatherBeliefModel):
    """
    自适应信念模型
    能够检测天气变化模式的变化并快速适应
    """

    def __init__(self, window_size: int = 10, adaptation_threshold: float = 0.3, **kwargs):
        super().__init__(**kwargs)

        self.window_size = window_size
        self.adaptation_threshold = adaptation_threshold

        # 滑动窗口
        self.recent_transitions = []

        # 模式检测
        self.detected_mode = "unknown"
        self.mode_history = []

    def update(self, weather_today: int):
        """更新并检测模式变化"""
        super().update(weather_today)

        # 维护滑动窗口
        if len(self.weather_history) >= 2:
            transition = (self.weather_history[-2], weather_today)
            self.recent_transitions.append(transition)
            if len(self.recent_transitions) > self.window_size:
                self.recent_transitions.pop(0)

        # 检测模式变化
        if len(self.recent_transitions) >= self.window_size:
            self._detect_mode_change()

    def _detect_mode_change(self):
        """检测天气变化模式是否发生突变"""
        # 计算窗口内的转移频率
        window_counts = np.ones((3, 3))
        for prev, curr in self.recent_transitions:
            window_counts[prev, curr] += 1

        window_matrix = window_counts / window_counts.sum(axis=1, keepdims=True)

        # 与历史估计比较
        diff = np.abs(window_matrix - self.transition_matrix).mean()

        if diff > self.adaptation_threshold:
            # 检测到模式变化，加快学习率
            self.learning_rate = min(0.5, self.learning_rate * 1.5)
            self.mode_history.append(("change_detected", len(self.weather_history)))
        else:
            # 逐渐恢复正常学习率
            self.learning_rate = max(0.05, self.learning_rate * 0.95)


# 信念特征提取器（用于神经网络）
def extract_belief_features(belief_model: WeatherBeliefModel) -> np.ndarray:
    """
    提取信念特征向量

    Returns:
        [p_sunny, p_hot, p_sandstorm, confidence, entropy, mode]
    """
    belief = belief_model.current_belief

    # 信息熵
    entropy = -np.sum(belief.probs * np.log(belief.probs + 1e-8))
    max_entropy = np.log(3)
    normalized_entropy = entropy / max_entropy

    # 最可能模式
    mode = np.argmax(belief.probs)

    return np.array(
        [
            *belief.probs,
            belief.confidence,
            normalized_entropy,
            mode / 2.0,  # 归一化到[0,1]
        ]
    )
