"""模型基准评测脚本（第三/第四问）。

功能：
1) 在多个天气模式 + 多随机种子下评估训练模型
2) 提供随机策略基线
3) 输出关键指标并可保存 JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import numpy as np
import torch
from src.env.config import Level3Config, Level4Config
from src.env.environment import make_env
from src.models.agent import HybridRNNAgent
from src.pipeline.evaluate import run_episode
from src.pipeline.oracle_optimal import solve_theoretical_optimal


def _get_weather_modes(level: int) -> List[str]:
    cfg = Level3Config if level == 3 else Level4Config
    return list(cfg.WEATHER_MODES.keys())


def _summarize(results: List[Dict]) -> Dict[str, float]:
    if not results:
        return {
            "runs": 0,
            "success_rate": 0.0,
            "avg_final_money": 0.0,
            "avg_return": 0.0,
            "avg_length": 0.0,
            "early_fail_rate": 0.0,
        }

    reached = [1 if r["reached"] else 0 for r in results]
    final_moneys = [float(r["final_money"]) for r in results]
    returns = [float(r["return"]) for r in results]
    lengths = [int(r["length"]) for r in results]

    early_fail = [1 if (not r["reached"] and r["length"] <= 2) else 0 for r in results]

    return {
        "runs": len(results),
        "success_rate": float(mean(reached)),
        "avg_final_money": float(mean(final_moneys)),
        "avg_return": float(mean(returns)),
        "avg_length": float(mean(lengths)),
        "early_fail_rate": float(mean(early_fail)),
    }


def _random_policy_episode(env, seed: Optional[int] = None) -> Dict:
    obs, info = env.reset(seed=seed)
    done = False
    steps = 0
    episode_return = 0.0

    while not done and steps < 200:
        va = env.get_valid_actions()

        move = int(np.random.choice(va["valid_moves"]))
        mine = bool(va.get("can_mine", False) and np.random.rand() < 0.5)

        if va.get("can_buy", False):
            max_w = int(va.get("max_buy_water", 0))
            max_f = int(va.get("max_buy_food", 0))
            buy_water = int(np.random.randint(0, max_w + 1)) if max_w > 0 else 0
            buy_food = int(np.random.randint(0, max_f + 1)) if max_f > 0 else 0
        else:
            buy_water, buy_food = 0, 0

        action = {
            "move": move,
            "mine": mine,
            "buy_water": buy_water,
            "buy_food": buy_food,
        }

        obs, reward, done, truncated, info = env.step(action)
        episode_return += float(reward)
        steps += 1
        if truncated:
            break

    return {
        "return": episode_return,
        "reached": bool(info.get("reached", False)),
        "final_money": float(info.get("money", 0.0)),
        "length": int(steps),
    }


def evaluate_model(
    agent_path: str,
    level: int,
    runs: int,
    device: str,
    weather_modes: List[str],
    deterministic: bool,
) -> Dict[str, Dict[str, float]]:
    agent = HybridRNNAgent.load(agent_path, device=device)
    agent.eval()

    summary: Dict[str, Dict[str, float]] = {}
    all_results: List[Dict] = []

    for mode in weather_modes:
        env = make_env(level=level, weather_mode=mode, seed=None)
        mode_results = []
        for seed in range(runs):
            result = run_episode(agent, env, seed=seed, deterministic=deterministic, verbose=False)
            mode_results.append(result)
            all_results.append(result)

        summary[mode] = _summarize(mode_results)

    summary["overall"] = _summarize(all_results)
    return summary


def evaluate_model_detailed(
    agent_path: str,
    level: int,
    runs: int,
    device: str,
    weather_modes: List[str],
    deterministic: bool,
) -> Dict[str, List[Dict]]:
    agent = HybridRNNAgent.load(agent_path, device=device)
    agent.eval()

    detailed: Dict[str, List[Dict]] = {}
    for mode in weather_modes:
        env = make_env(level=level, weather_mode=mode, seed=None)
        mode_results: List[Dict] = []
        for seed in range(runs):
            result = run_episode(agent, env, seed=seed, deterministic=deterministic, verbose=False)
            result["seed"] = seed
            result["weather_mode"] = mode
            mode_results.append(result)
        detailed[mode] = mode_results

    return detailed


def evaluate_random_baseline(level: int, runs: int, weather_modes: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    all_results: List[Dict] = []

    for mode in weather_modes:
        env = make_env(level=level, weather_mode=mode, seed=None)
        mode_results = []
        for seed in range(runs):
            result = _random_policy_episode(env, seed=seed)
            mode_results.append(result)
            all_results.append(result)
        summary[mode] = _summarize(mode_results)

    summary["overall"] = _summarize(all_results)
    return summary


def _print_summary(title: str, summary: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    print("=" * 78)
    print(f"{'模式':<18}{'成功率':>10}{'均资金':>12}{'均回报':>12}{'均步长':>10}{'早死率':>10}")
    for mode, stats in summary.items():
        print(
            f"{mode:<18}"
            f"{stats['success_rate'] * 100:>9.1f}%"
            f"{stats['avg_final_money']:>12.2f}"
            f"{stats['avg_return']:>12.2f}"
            f"{stats['avg_length']:>10.1f}"
            f"{stats['early_fail_rate'] * 100:>9.1f}%"
        )


def evaluate_oracle_upper_bound(
    level: int,
    runs: int,
    weather_modes: List[str],
    oracle_time_limit: int,
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    overall_values: List[float] = []
    overall_solved = 0

    for mode in weather_modes:
        env = make_env(level=level, weather_mode=mode, seed=None)
        values: List[float] = []
        solved = 0

        for seed in range(runs):
            env.reset(seed=seed)
            if env.state is None:
                continue
            weather_seq = list(env.state.weather_future)
            oracle = solve_theoretical_optimal(level, weather_seq, time_limit=oracle_time_limit)
            if np.isfinite(oracle["objective"]):
                values.append(float(oracle["objective"]))
                overall_values.append(float(oracle["objective"]))
            if oracle["status"] == "Optimal":
                solved += 1
                overall_solved += 1

        avg_obj = float(mean(values)) if values else 0.0
        summary[mode] = {
            "runs": runs,
            "oracle_solved_rate": solved / runs if runs > 0 else 0.0,
            "oracle_avg_objective": avg_obj,
        }

    summary["overall"] = {
        "runs": runs * len(weather_modes),
        "oracle_solved_rate": overall_solved / (runs * len(weather_modes)) if weather_modes else 0.0,
        "oracle_avg_objective": float(mean(overall_values)) if overall_values else 0.0,
    }
    return summary


def compare_model_vs_oracle(
    model_detailed: Dict[str, List[Dict]],
    level: int,
    oracle_time_limit: int,
) -> Dict[str, Dict[str, float]]:
    comp: Dict[str, Dict[str, float]] = {}
    all_ratios: List[float] = []
    all_gaps: List[float] = []

    for mode, records in model_detailed.items():
        env = make_env(level=level, weather_mode=mode, seed=None)
        ratios: List[float] = []
        gaps: List[float] = []
        solved = 0

        for r in records:
            seed = int(r["seed"])
            env.reset(seed=seed)
            if env.state is None:
                continue
            weather_seq = list(env.state.weather_future)
            oracle = solve_theoretical_optimal(level, weather_seq, time_limit=oracle_time_limit)
            oracle_obj = float(oracle["objective"])
            model_money = float(r["final_money"])

            if np.isfinite(oracle_obj) and oracle_obj > 1e-6:
                ratio = model_money / oracle_obj
                gap = oracle_obj - model_money
                ratios.append(ratio)
                gaps.append(gap)
                all_ratios.append(ratio)
                all_gaps.append(gap)

            if oracle["status"] == "Optimal":
                solved += 1

        comp[mode] = {
            "runs": len(records),
            "oracle_solved_rate": solved / len(records) if records else 0.0,
            "avg_money_to_oracle_ratio": float(mean(ratios)) if ratios else 0.0,
            "avg_oracle_gap": float(mean(gaps)) if gaps else 0.0,
        }

    comp["overall"] = {
        "avg_money_to_oracle_ratio": float(mean(all_ratios)) if all_ratios else 0.0,
        "avg_oracle_gap": float(mean(all_gaps)) if all_gaps else 0.0,
    }
    return comp


def _print_oracle_summary(title: str, summary: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    print("=" * 78)
    if "oracle_avg_objective" in next(iter(summary.values()), {}):
        print(f"{'模式':<18}{'求解率':>10}{'Oracle均目标':>16}")
        for mode, stats in summary.items():
            print(
                f"{mode:<18}"
                f"{stats.get('oracle_solved_rate', 0.0) * 100:>9.1f}%"
                f"{stats.get('oracle_avg_objective', 0.0):>16.2f}"
            )
    else:
        print(f"{'模式':<18}{'求解率':>10}{'资金/Oracle':>14}{'OracleGap':>12}")
        for mode, stats in summary.items():
            if mode == "overall":
                print(
                    f"{mode:<18}"
                    f"{'-':>10}"
                    f"{stats.get('avg_money_to_oracle_ratio', 0.0):>13.3f}"
                    f"{stats.get('avg_oracle_gap', 0.0):>12.2f}"
                )
            else:
                print(
                    f"{mode:<18}"
                    f"{stats.get('oracle_solved_rate', 0.0) * 100:>9.1f}%"
                    f"{stats.get('avg_money_to_oracle_ratio', 0.0):>13.3f}"
                    f"{stats.get('avg_oracle_gap', 0.0):>12.2f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="task2 模型基准评测")
    parser.add_argument("agent_path", type=str, help="模型路径，如 artifacts/checkpoints/level3_best.pt")
    parser.add_argument("--level", type=int, choices=[3, 4], default=3)
    parser.add_argument("--runs", type=int, default=100, help="每个天气模式的评测轮数")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu")
    parser.add_argument("--weather-modes", type=str, default=None, help="逗号分隔，如 no_sandstorm,sunny_bias")
    parser.add_argument("--stochastic", action="store_true", help="使用随机采样策略评估（默认贪心）")
    parser.add_argument("--with-random-baseline", action="store_true", help="输出随机策略基线")
    parser.add_argument("--with-oracle", action="store_true", help="计算数学规划Oracle上界并对比")
    parser.add_argument("--oracle-time-limit", type=int, default=30, help="Oracle单次求解时限（秒）")
    parser.add_argument("--output-json", type=str, default=None, help="保存结果到 JSON")
    parser.add_argument("--min-success-rate", type=float, default=None, help="若 overall 成功率低于该阈值则返回非0")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.weather_modes:
        weather_modes = [m.strip() for m in args.weather_modes.split(",") if m.strip()]
    else:
        weather_modes = _get_weather_modes(args.level)

    model_summary = evaluate_model(
        agent_path=args.agent_path,
        level=args.level,
        runs=args.runs,
        device=device,
        weather_modes=weather_modes,
        deterministic=(not args.stochastic),
    )
    model_detailed = evaluate_model_detailed(
        agent_path=args.agent_path,
        level=args.level,
        runs=args.runs,
        device=device,
        weather_modes=weather_modes,
        deterministic=(not args.stochastic),
    )

    _print_summary("模型策略评测", model_summary)

    random_summary = None
    if args.with_random_baseline:
        random_summary = evaluate_random_baseline(args.level, args.runs, weather_modes)
        _print_summary("随机基线评测", random_summary)

    oracle_summary = None
    oracle_compare = None
    if args.with_oracle:
        oracle_summary = evaluate_oracle_upper_bound(
            level=args.level,
            runs=args.runs,
            weather_modes=weather_modes,
            oracle_time_limit=args.oracle_time_limit,
        )
        _print_oracle_summary("Oracle上界评测", oracle_summary)

        oracle_compare = compare_model_vs_oracle(
            model_detailed=model_detailed,
            level=args.level,
            oracle_time_limit=args.oracle_time_limit,
        )
        _print_oracle_summary("模型 vs Oracle 对比", oracle_compare)

    if args.output_json:
        payload = {
            "level": args.level,
            "runs_per_mode": args.runs,
            "device": device,
            "weather_modes": weather_modes,
            "deterministic": not args.stochastic,
            "model": model_summary,
            "random_baseline": random_summary,
            "oracle": oracle_summary,
            "model_vs_oracle": oracle_compare,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {out_path}")

    if args.min_success_rate is not None:
        overall_sr = model_summary["overall"]["success_rate"]
        if overall_sr < args.min_success_rate:
            raise SystemExit(
                f"Benchmark failed: overall success_rate={overall_sr:.4f} < {args.min_success_rate:.4f}"
            )


if __name__ == "__main__":
    main()
