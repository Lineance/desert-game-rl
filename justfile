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

# 第四关加速训练（推荐，复用离线BC数据集）
train4-fast:
    $env:OMP_NUM_THREADS="1"
    $env:MKL_NUM_THREADS="1"
    $env:OPENBLAS_NUM_THREADS="1"
    uv run python scripts/train.py --level 4 --device cuda --episodes 5000 --log-interval 200 --eval-interval 400 --eval-episodes 10 --path-entropy-episodes 20 --checkpoint-interval 500 --oracle-dataset-path artifacts/results/level4_bc_dataset.json --oracle-warmup-episodes 300 --oracle-parallel-workers 10 --oracle-solver-threads 1 --oracle-warmup-batch-episodes 4 --auto-stages

# 第四关加速训练（可调版本）
train4-tuned episodes="5000" warmup="300" workers="10" warmup_batch="4":
    $env:OMP_NUM_THREADS="1"
    $env:MKL_NUM_THREADS="1"
    $env:OPENBLAS_NUM_THREADS="1"
    uv run python scripts/train.py --level 4 --device cuda --episodes {{episodes}} --log-interval 200 --eval-interval 400 --eval-episodes 10 --path-entropy-episodes 20 --checkpoint-interval 500 --oracle-dataset-path artifacts/results/level4_bc_dataset.json --oracle-warmup-episodes {{warmup}} --oracle-parallel-workers {{workers}} --oracle-solver-threads 1 --oracle-warmup-batch-episodes {{warmup_batch}} --auto-stages

board:
    uv run tensorboard --logdir artifacts/logs/tensorboard --port 6006

# 评估（需指定模型路径）
eval model level="3" seed="42":
    uv run python scripts/evaluate.py {{model}} --level {{level}} --seed {{seed}} --verbose

# 基准测试
bench model level="4":
    uv run python scripts/benchmark.py {{model}} --level {{level}} --runs 30 --with-oracle

# 验证结果文件
validate file level="3":
    uv run python scripts/validator.py {{file}} --level {{level}}

# ========== 工具 ==========

# 同步依赖
sync:
    uv sync --group dev