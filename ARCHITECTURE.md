# 穿越沙漠问题2：当前RL架构说明（第三问/第四问）

本文档描述 `task2` 目录下**当前实际代码**对应的强化学习架构与数据流。

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

## 2. 当前模块结构

- `environment.py`：环境动力学、合法动作集合、奖励与终止
- `belief.py`：天气信念模型与信念特征提取
- `agent.py`：`HybridRNNAgent`（当前为 MLP + Belief 编码）
- `ppo.py`：唯一 PPO 实现（采样、GAE、更新）
- `train.py`：训练入口，仅调用 `ppo.py` 的 `PPOTrainer`
- `evaluate.py`：评估与结果导出（记录执行后状态）

> 说明：历史版本中的“多套PPO训练器并存”问题已移除，当前只保留 `ppo.py` 一套实现。

---

## 3. 观测、动作与状态表示

## 3.1 观测向量（19维）

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

## 3.2 动作结构

策略动作字典：

- `move`：目标节点（全局节点ID）
- `mine`：是否挖矿
- `buy_water`：购买水量
- `buy_food`：购买食物量

## 3.3 合法动作掩码

环境通过 `get_valid_actions()` 提供：

- `valid_moves`
- `can_mine`
- `can_buy`
- `max_buy_water`
- `max_buy_food`

Actor 在前向、采样与评估时都使用该掩码：

- 移动分布对非法移动置 `-inf`
- 挖矿分布对非法挖矿置 `-inf`
- 不可购买时将购买动作固定为0并将其 log_prob/entropy 置零，避免无效梯度

---

## 4. 策略网络与价值网络（agent.py）

## 4.1 编码器

- `state_encoder`：编码非信念部分
- `belief_encoder`：编码信念特征
- 拼接后形成联合状态表示（192维）

## 4.2 Actor

多头输出：

- `move_head`：离散 `Categorical`
- `mine_head`：二元 `Categorical`
- `buy_water_mean/std`：连续 `Normal`
- `buy_food_mean/std`：连续 `Normal`

## 4.3 Critic

- MLP 输出状态价值 `V(s)`

---

## 5. PPO训练流程（ppo.py）

## 5.1 采样

`collect_rollout()` 执行：

1. 环境重置
2. 读取合法动作掩码
3. `agent.select_action()` 采样动作
4. 环境步进
5. 使用**同一掩码**重算动作 log_prob 并存入 buffer
6. 存储 `obs/action/reward/value/log_prob/done/valid_actions`

> 关键一致性：old_log_prob 与更新阶段使用同样的动作可行域定义。

## 5.2 优势估计

- 使用 GAE（`gamma`, `gae_lambda`）
- 使用终止标记抑制终止后 bootstrap

## 5.3 更新

- PPO clip policy loss
- value clip loss
- entropy bonus
- 梯度裁剪

---

## 6. 环境动力学与奖励（environment.py）

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

## 7. 训练与评估入口

## 7.1 训练（train.py）

- 创建环境与智能体
- 仅使用 `ppo.PPOTrainer`
- 固定周期打印回报、成功率、损失统计
- 保存 best / checkpoint / final 模型

## 7.2 评估（evaluate.py）

- 运行单轮或多轮评估
- 记录并导出**当日动作执行后**状态
- 优先使用 `last_action` 导出真实动作文本

---

## 8. 与历史文档差异

本版本与早期文档相比有以下关键变化：

- 当前主体不是 LSTM 时序网络，而是 MLP + Belief 编码
- 只保留一套 PPO 实现（`ppo.py`）
- 评估导出已按“执行后状态”口径记录
- 动作掩码已贯通采样与训练，减少策略-环境脱节

---

## 9. 后续可选增强

- 教师策略蒸馏（用已知天气最优策略预训练 Actor/Critic）
- 课程学习恢复与系统化调度
- 风险敏感目标（如 CVaR）
- 将环境规则验证封装为 `task2` 内部校验模块（训练后自动回归）
