"""
问题2：穿越沙漠 - 强化学习求解
第三关、第四关（部分可观测决策）

混合方案：RNN历史编码 + 贝叶斯信念 + PPO训练
"""

from .env.config import Level3Config, Level4Config, RLConfig
from .env.environment import DesertCrossingEnv, make_env
from .models.agent import HybridRNNAgent, create_agent
from .models.belief import WeatherBeliefModel, extract_belief_features
from .models.ppo import PPOTrainer, RolloutBuffer
from .pipeline.validator import RLResultValidator, build_validation_config

__all__ = [
    'Level3Config',
    'Level4Config', 
    'RLConfig',
    'DesertCrossingEnv',
    'make_env',
    'WeatherBeliefModel',
    'extract_belief_features',
    'HybridRNNAgent',
    'create_agent',
    'PPOTrainer',
    'RolloutBuffer',
    'RLResultValidator',
    'build_validation_config',
]
