# 穿越沙漠问题2（第三问/第四问）RL求解

当前版本使用PPO 实现训练部分可观测策略：

- 仅已知当天天气
- 通过信念模型建模天气不确定性
- 使用动作掩码保证策略与环境规则一致

训练与调参细节见 [docs/训练与调参.md](docs/%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%B0%83%E5%8F%82.md)。

已完成 pipeline 组件验证（2026-02-27）：

- train / pretrain / evaluate / benchmark / validator 相关测试全通过（54/54）
- 训练日志已接入多目标动态加权指标：`alpha_mean`、`survival_value_loss`、`fund_value_loss`

---

## 项目结构

```text
task2/
├── scripts/
│   ├── train.py       # 训练入口脚本
│   ├── pretrain.py    # 仅Oracle蒸馏预热入口脚本
│   ├── evaluate.py    # 评估入口脚本
│   ├── benchmark.py   # 基准评测入口脚本
│   └── validator.py   # 结果校验入口脚本
├── src/
│   ├── env/
│   │   ├── config.py      # 关卡参数与RL超参数
│   │   └── environment.py # POMDP环境与规则执行
│   ├── models/
│   │   ├── belief.py      # 天气信念模型与特征提取
│   │   ├── agent.py       # 主入口（组装编码器/策略/双价值）
│   │   ├── ppo.py         # PPO训练实现（采样+更新）
│   │   ├── encoders/      # ResourceEncoder/ManualGNN/GateFusion
│   │   ├── policy/        # MoveSelector/LocationActionSelector/Generators
│   │   └── critic/        # SurvivalCritic/FundCritic/TargetNetwork
│   ├── pipeline/
│   │   ├── train.py       # 训练核心逻辑
│   │   ├── pretrain.py    # 评估与导出核心逻辑
│   │   ├── rollout.py     # 单次运行求解
│   │   ├── evaluate.py    # 多次运行详细分析
│   │   └── benchmark.py   # 评价基准
│   └── utils/
│       ├── graph_utils.py    # 图静态特征与最短路
│       ├── mask_utils.py     # 动作掩码与地点类型辅助
│       ├── bc_dataset.py     # Behavior Cloning数据集生成/加载
│       └── training_utils.py # GAE/梯度裁剪/动态alpha
│
└── README.md
```

---

## 当前架构（与实现一致）

- 观测：19维
  - 状态6维 + 天气one-hot 3维 + 地点类型4维 + 信念特征6维
- 模型：模块化分层架构（`src/models`）
  - 编码层：`ResourceEncoder` + `ManualGNN` + `GateFusion`
  - 策略层：`MoveSelector`（阶段1）+ `LocationActionSelector`（阶段2）+ 生成器
  - 价值层：`SurvivalCritic` + `FundCritic` + `TargetNetwork`
- PPO：在 `ppo.py` 实现
  - 使用 `valid_actions` 约束动作合法性
  - 包含 GAE、PPO clip、value clip、entropy、梯度裁剪、KL早停
  - 已接入动态加权：`TotalValueLoss = alpha * SurvivalLoss + (1-alpha) * FundLoss`

> 详细设计见 `docs/模型架构.md`。

---

## 环境规则实现要点

- 第0天：只允许起点购买（且只允许一次）
- 沙暴日：强制停留
- 挖矿：必须在矿山且不能在到达当天挖
- 村庄购买：受资金与负重限制
- 到达终点：退款并终止

`environment.py` 会返回 `valid_actions`，并在 `info` 中返回 `last_action`（实际执行动作）用于评估导出对齐。

---

## 环境准备（uv）

```bash
# 安装并同步依赖（推荐）
uv sync

# 含开发依赖（pytest 等）
uv sync --group dev
```

> 项目已在 `pyproject.toml` 中配置：默认清华源 + `torch` 使用 `cu128` 源。

---

## 训练

### 仅预训练并保存

```bash
# 仅进行Oracle蒸馏预热并保存
uv run python scripts/pretrain.py --level 3 --warmup-episodes 300
```

默认输出：`artifacts/checkpoints/level{n}_pretrained.pt`

### 离线生成并复用 BC 数据（推荐）

当你希望减少重复求解 Oracle（尤其 warmup 多次重跑）时，可以先离线生成数据集，再在 pretrain/train 中复用。

```bash
# 1) 先离线生成BC数据集（一次生成，多次复用）
uv run python -m src.utils.bc_dataset \
  --level 3 \
  --episodes 500 \
  --oracle-time-limit 20 \
  --seed-start 0 \
  --output artifacts/results/level3_bc_dataset.json

# 2) 预训练复用该数据集（命中即直接取plan，未命中自动回退在线Oracle）
uv run python scripts/pretrain.py \
  --level 3 \
  --warmup-episodes 300 \
  --oracle-dataset-path artifacts/results/level3_bc_dataset.json

# 3) train warmup 同样支持复用
uv run python scripts/train.py \
  --level 3 \
  --episodes 3000 \
  --oracle-warmup-episodes 300 \
  --oracle-dataset-path artifacts/results/level3_bc_dataset.json
```

### 命令

```bash
# 第三问
uv run python scripts/train.py --level 3 --episodes 2000

# 第四问
uv run python scripts/train.py --level 4 --episodes 5000

# 指定设备
uv run python scripts/train.py --level 3 --episodes 2000 --device cuda

# 启用Stage2/Stage3自动编排（Stage1仍手动分离）
uv run python scripts/train.py --level 3 --episodes 3000 --auto-stages
```

### 参数

- `--level`：关卡编号（3 或 4）
- `--episodes`：训练回合数
- `--device`：`cuda` 或 `cpu`（默认自动选择）

### 输出

模型保存在 `artifacts/checkpoints/`：

- `level{n}_best.pt`
- `level{n}_episode{k}.pt`
- `level{n}_final.pt`

---

## 评估与导出

### 单轮评估

```bash
uv run python scripts/evaluate.py ./artifacts/checkpoints/level3_best.pt --level 3 --verbose
```

### 多轮统计

```bash
uv run python scripts/evaluate.py ./artifacts/checkpoints/level3_best.pt --level 3 --runs 100
```

### 导出结果

```bash
uv run python scripts/evaluate.py ./artifacts/checkpoints/level3_best.pt --level 3 --output ./artifacts/results
```

导出表记录的是**当日动作执行后**状态，符合题目注2口径；动作列优先使用环境实际执行动作（而非策略意图动作）。

## Benchmark（模型有效性评测）

当训练耗时较长时，建议固定用基准脚本评估模型质量趋势，而不是只看训练日志。

```bash
# 基础评测（每个天气模式100轮）
uv run python scripts/benchmark.py ./artifacts/checkpoints/level3_best.pt --level 3 --runs 100

# 带随机策略基线对照
uv run python scripts/benchmark.py ./artifacts/checkpoints/level3_best.pt --level 3 --runs 100 --with-random-baseline

# 指定天气模式并保存JSON
uv run python scripts/benchmark.py ./artifacts/checkpoints/level3_best.pt --level 3 \
  --weather-modes no_sandstorm,sunny_bias,hot_bias \
  --output-json ./artifacts/results/benchmark_level3.json

# 设定最低成功率门槛（可用于自动回归）
uv run python scripts/benchmark.py ./artifacts/checkpoints/level3_best.pt --level 3 --runs 100 --min-success-rate 0.3

# 加入数学规划Oracle上界（建议runs先设小一些）
uv run python scripts/benchmark.py ./artifacts/checkpoints/level3_best.pt --level 3 --runs 20 --with-oracle --oracle-time-limit 30
```

输出指标：

- 成功率（到达终点比例）
- 平均最终资金
- 平均回报
- 平均步长
- 早死率（未到达且<=2步结束）

使用 `--with-oracle` 时还会输出：

- Oracle 求解率（MILP在时限内达到最优的比例）
- Oracle 平均目标值（理论上界）
- 模型资金 / Oracle 目标值比（越接近1越好）
- 模型与Oracle资金差距（Oracle Gap）

### 结果验证（task2内置）

可直接使用 `scripts/validator.py` 对导出结果做规则一致性检查：

```bash
# 验证第三关结果
uv run python scripts/validator.py ./artifacts/results/level3_result.xlsx --level 3

# 验证第四关结果
uv run python scripts/validator.py ./artifacts/results/level4_result.xlsx --level 4
```

验证器会检查：

- 日期连续性与起终点约束
- 移动连通性、沙暴日移动限制
- 挖矿/购买位置合法性
- 资源消耗、资金收支与负重约束
- 终点退回资金逻辑一致性

---

## Python API 示例

```python
from task2 import make_env, create_agent, RLConfig

env = make_env(level=3, seed=42)
agent = create_agent(env, RLConfig(), device="cpu")

obs, info = env.reset()
done = False

while not done:
    valid_actions = env.get_valid_actions()
    action, value = agent.select_action(obs, valid_actions)
    obs, reward, done, truncated, info = env.step(action)

print(info["money"], info["reached"])
```

---

## 依赖

```bash
# 推荐
uv sync

# 若需要开发依赖
uv sync --group dev

# 仅在不使用 uv 时
pip install torch numpy openpyxl pandas pulp highspy
```

---

## 备注

- 第三/四问是未知未来天气的泛化问题，训练不依赖“已知全天气”的验证器。
- 若需要规则回归测试，可在 `task2` 内增加轻量验证模块用于离线轨迹自检。
