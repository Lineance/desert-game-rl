"""多轮评估统计入口。"""

import argparse
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.env.environment import make_env
from src.models.agent import Agent
from src.pipeline.rollout import run_episode


def load_agent_for_eval(agent_path: str, device: Optional[str]) -> Tuple[Agent, str]:
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent.load(agent_path, device=resolved)
    agent.eval()
    return agent, resolved


def run_episodes(
    agent: Agent,
    env,
    runs: int,
    deterministic: bool,
    *,
    verbose: bool = False,
    include_seed: bool = False,
    progress_every: Optional[int] = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> List[Dict]:
    results: List[Dict] = []

    for seed in range(runs):
        result = run_episode(agent, env, seed=seed, deterministic=deterministic, verbose=verbose)
        if include_seed:
            result["seed"] = seed
        results.append(result)

        if progress_every and progress_fn and (seed + 1) % progress_every == 0:
            progress_fn(seed + 1, runs)

    return results


def evaluate_multiple_runs(agent_path: str, level: int, num_runs: int = 100, device: str = None):
    """
    多次运行评估，统计性能
    """
    # 加载智能体
    agent, device = load_agent_for_eval(agent_path, device)

    # 创建环境
    env = make_env(level=level, seed=None)

    print(f"\n多轮评估: {num_runs} 次")
    print(f"模型: {agent_path}")
    print(f"关卡: {level}")

    def _progress(done: int, total: int) -> None:
        print(f"  已完成 {done}/{total}")

    results = run_episodes(
        agent,
        env,
        num_runs,
        deterministic=True,
        progress_every=20,
        progress_fn=_progress,
    )

    # 统计
    metrics = _extract_basic_metrics(results)
    reached = metrics["reached"]
    final_moneys = metrics["final_moneys"]
    lengths = metrics["lengths"]
    returns = metrics["returns"]

    print("\n统计结果:")
    print(f"  到达率: {np.mean(reached) * 100:.1f}% ({sum(reached)}/{num_runs})")
    print(f"  平均最终资金: {np.mean(final_moneys):.2f} ± {np.std(final_moneys):.2f}")
    print(f"  平均回合长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  平均回报: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

    # 找出最佳和最差
    best_idx = np.argmax(final_moneys)

    print(f"\n最佳表现 (Run {best_idx}):")
    print(f"  最终资金: {results[best_idx]['final_money']:.2f}")
    print(f"  路径: {' -> '.join(map(str, results[best_idx]['path'][:20]))}...")

    return results


def evaluate_main():
    parser = argparse.ArgumentParser(description="评估沙漠穿越智能体（多轮统计）")
    parser.add_argument("agent_path", type=str, help="智能体模型路径")
    parser.add_argument("--level", type=int, default=3, choices=[3, 4, 35], help="关卡")
    parser.add_argument("--runs", type=int, default=100, help="运行次数（需>1）")
    parser.add_argument("--device", type=str, default=None, help="计算设备（cuda/cpu）")

    args = parser.parse_args()

    if args.runs <= 1:
        raise SystemExit("请使用 scripts/rollout.py 进行单轮评估与导出，或将 --runs 设为 >1")

    evaluate_multiple_runs(args.agent_path, args.level, args.runs, args.device)


def _extract_basic_metrics(results: List[Dict]) -> Dict[str, List[float]]:
    return {
        "reached": [1 if r.get("reached") else 0 for r in results],
        "final_moneys": [float(r.get("final_money", 0.0)) for r in results],
        "lengths": [int(r.get("length", 0)) for r in results],
        "returns": [float(r.get("return", 0.0)) for r in results],
    }


if __name__ == "__main__":
    evaluate_main()
