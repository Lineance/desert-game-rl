# 穿越沙漠问题2：当前RL架构说明（第三问/第四问）

本文档描述 `task2` 目录下**当前实际代码**对应的强化学习架构、数据流与训练监控口径。

---

## 1. 总体目标与约束

- 任务类型：部分可观测决策（仅已知当天天气）
- 训练目标：学习在未知未来天气下可泛化的策略
- 核心约束：
  - 沙暴日不可移动
  - 到达矿山当天不能挖矿
  - 第0天仅可在起点购买且仅一次
  - 村庄可购买，需满足资金与负重限制
  - 到达终点后游戏结束

---

## 2. 当前目录与模块分层

当前工程采用“入口脚本 / 业务分层”结构：

- `scripts/`：命令行入口（`train.py` / `evaluate.py` / `benchmark.py` / `validator.py`）
- `src/env/`：环境与配置
  - `config.py`：关卡参数、训练超参数、产物目录常量
  - `environment.py`：环境动力学、合法动作集合、奖励与终止
- `src/models/`：模型与训练器
  - `belief.py`：天气信念模型与信念特征提取
  - `agent.py`：`HybridRNNAgent`（LSTM + Belief 时序编码）
  - `ppo.py`：唯一 PPO 实现（采样、GAE、更新）
- `src/pipeline/`：流程编排
  - `train.py`：训练循环与日志输出
  - `evaluate.py`：评估与结果导出
  - `benchmark.py`：基准评测与 Oracle 对比
  - `validator.py`：结果一致性验证

> 说明：历史版本中的“多套PPO并存”问题已移除，当前仅保留 `src/models/ppo.py` 一套实现。

---

## 3. 观测、动作与状态表示

### 3.1 观测向量（19维）

环境观测由以下部分拼接：

- 基础状态（6维）
  - 天数进度
  - 当前位置
  - 水量
  - 食物量
  - 资金比例
  - 到终点距离
- 当天天气 one-hot（3维）
- 地点类型 one-hot（4维：起点/终点/矿山/村庄）
- 信念特征（6维）
  - 明天天气概率分布（3维）
  - 置信度
  - 归一化熵
  - 模式索引归一化

合计：`6 + 3 + 4 + 6 = 19`。

### 3.2 动作结构

策略动作字典：

- `move`：目标节点（全局节点ID）
- `mine`：是否挖矿
- `buy_water`：购买水量
- `buy_food`：购买食物量

### 3.3 合法动作掩码

环境通过 `get_valid_actions()` 提供：

- `valid_moves`
- `can_mine`
- `can_buy`
- `max_buy_water`
- `max_buy_food`

Actor 在前向、采样与评估时都使用该掩码：

- 移动分布对非法移动置 `-inf`
- 挖矿分布对非法挖矿置 `-inf`
- 不可购买时将购买动作固定为0并将其 `log_prob/entropy` 置零，避免无效梯度

---

## 4. 策略网络与价值网络（`src/models/agent.py`）

### 4.1 编码器

- `state_encoder`：编码非信念部分
- `belief_encoder`：编码信念特征
- 拼接后形成联合状态表示（192维）
- `temporal_encoder (LSTM)`：对联合状态序列建模并输出时序特征

### 4.2 Actor

多头输出：

- `move_head`：离散 `Categorical`
- `mine_head`：二元 `Categorical`
- `buy_water_mean/std`：连续 `Normal`
- `buy_food_mean/std`：连续 `Normal`

### 4.3 Critic

- 基于 LSTM 时序特征输出状态价值 `V(s)`

---

## 5. PPO训练流程（`src/models/ppo.py`）

### 5.1 采样（`collect_rollout`）

1. 环境重置
2. 读取合法动作掩码
3. `agent.select_action()` 采样动作
4. 环境步进
5. 使用**动作前 hidden state + 同一掩码**重算动作 `log_prob` 并存入 buffer
6. 存储 `obs/action/reward/value/log_prob/done/hidden_state/valid_actions`

同时采集回合行为统计：

- `move_count/stay_count`
- `mine_count`
- `buy_count`
- `total_buy_water/total_buy_food`

> 关键一致性：old_log_prob 与更新阶段使用同样的动作可行域定义。

### 5.2 优势估计

- 使用 GAE（`gamma`, `gae_lambda`）
- 使用终止标记抑制终止后 bootstrap

### 5.3 更新（稳定化版本）

当前更新包含：

- 按完整 episode 切段做序列前向与完整 BPTT（非 TBPTT）
- PPO clip policy loss
- value clip loss
- entropy bonus
- 梯度裁剪
- `TARGET_KL` 早停（若 `approx_kl` 超阈值则提前结束本轮 update）

并输出诊断统计：

- `effective_updates`（实际执行 epoch 数）
- `target_kl_hit`（是否触发 KL 早停）
- `max_ratio/min_ratio`（策略更新幅度）
- `explained_variance`（价值函数解释度）

---

## 6. 训练超参数（当前默认）

定义于 `src/env/config.py` 的 `RLConfig`：

- `CLIP_EPS = 0.2`
- `LR_ACTOR = 5e-5`
- `LR_CRITIC = 2e-4`
- `EPOCHS_PER_UPDATE = 4`
- `TARGET_KL = 0.01`
- `ENTROPY_COEF = 0.03`
- `LSTM_LAYERS = 2`
- `LSTM_HIDDEN_DIM = 256`

这些参数用于抑制策略突变、降低价值发散风险，并提升训练可诊断性。

---

## 7. 环境动力学与奖励（`src/env/environment.py`）

环境按“移动/停留/挖矿/购买”规则执行资源变化，并处理：

- 沙暴强制停留
- 到达当天不可挖矿
- 购买资金与负重检查
- 超时失败
- 到达终点后退款并终止

环境 `info` 提供：

- `day/position/water/food/money/weather_today/reached`
- `last_action`（环境实际执行动作）
- `action_day`（该步动作对应日期）

该设计用于保证评估导出与真实状态演化一致。

---

## 8. 训练与评估入口

### 8.1 训练（推荐入口：`scripts/train.py`）

- 创建环境与智能体
- 使用 `PPOTrainer` 采样与更新
- 支持 `--log-interval` 日志频率控制
- 固定周期打印：
  - 最近窗口成功率/均回报/均步长/均资金
  - 行为统计（移动/停留/挖矿/购买/购水/购食）
  - 损失与更新诊断（KL、ClipFrac、ExplVar、target_kl_hit）
- 保存 best / checkpoint / final 模型

### 8.2 评估（推荐入口：`scripts/evaluate.py`）

- 运行单轮或多轮评估
- 记录并导出**当日动作执行后**状态
- 优先使用 `last_action` 导出真实动作文本

### 8.3 基准与校验

- `scripts/benchmark.py`：模型评测、随机基线、Oracle对比
- `scripts/validator.py`：导出结果规则一致性验证

---

## 9. 与早期文档差异

当前版本关键变化：

- 目录已从平铺改为 `env/models/pipeline` 分层 + `scripts` 入口
- `belief` 已归入模型层（`src/models/belief.py`）
- PPO 增加 `TARGET_KL` 早停与更多训练诊断指标
- 训练日志从“单回合信息”扩展为“滚动窗口 + 更新诊断”

---

## 10. 后续可选增强

- 教师策略蒸馏（用已知天气最优策略预训练 Actor/Critic）
- 课程学习恢复与系统化调度
- 风险敏感目标（如 CVaR）
- 增加自动化回归基准（训练后自动跑 benchmark + validator）
