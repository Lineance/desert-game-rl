# ========== 代码质量 ==========

# 格式化代码
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# 检查（不修改，用于 CI）
fmt-check:
    uv run ruff format --check .
    uv run ruff check .

# ========== 测试 ==========

# 运行测试
test:
    uv run pytest -x

# 覆盖率报告
cov arg:
    uv run pytest --cov=src --cov-report=term -m "{{arg}}"

# ========== 训练与评估 ==========

# 训练
train3:
    uv run python scripts/train.py --level 3 --episodes 2000

train4:
    uv run python scripts/train.py --level 4 --episodes 5000

board:
    uv run tensorboard --logdir artifacts/logs/tensorboard --port 6006

# 评估（需指定模型路径）
eval model level="3" seed="42":
    uv run python scripts/evaluate.py {{model}} --level {{level}} --seed {{seed}} --verbose

# 基准测试
bench model level="3":
    uv run python scripts/benchmark.py {{model}} --level {{level}} --runs 100

# 验证结果文件
validate file level="3":
    uv run python scripts/validator.py {{file}} --level {{level}}

# ========== 工具 ==========

# 同步依赖
sync:
    uv sync --group dev