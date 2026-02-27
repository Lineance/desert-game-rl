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
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.env.config import Level3Config, Level4Config, Level35Config
from src.env.environment import make_env
from src.pipeline.evaluate import load_agent_for_eval, run_episodes
from src.utils.oracle import solve_theoretical_batch, solve_theoretical_optimal


def evaluate_model(
    agent_path: str,
    level: int,
    runs: int,
    device: str,
    weather_modes: List[str],
    deterministic: bool,
) -> Dict[str, List[Dict]]:

    agent, _ = load_agent_for_eval(agent_path, device)

    detailed: Dict[str, List[Dict]] = {}

    for mode in weather_modes:
        env = make_env(level=level, weather_mode=mode, seed=None)

        mode_results = run_episodes(
            agent,
            env,
            runs,
            deterministic=deterministic,
            include_seed=True,
        )
        for result in mode_results:
            result["weather_mode"] = mode

        detailed[mode] = mode_results

    return detailed


def evaluate_random_baseline(
    level: int, runs: int, weather_modes: List[str]
) -> Dict[str, Dict[str, float]]:

    summary: Dict[str, Dict[str, float]] = {}

    all_results: List[Dict] = []

    for mode in weather_modes:
        env = make_env(level=level, weather_mode=mode, seed=None)

        mode_results = []
        for seed in range(runs):
            result = _random_policy_episode(env, seed=seed)
            mode_results.append(result)
            all_results.append(result)

        summary[mode] = _summarize(mode_results, level)

    summary["overall"] = _summarize(all_results, level)

    return summary


def compare_model_vs_oracle(
    model_detailed: Dict[str, List[Dict]],
    oracle_detailed: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:

    comp: Dict[str, Dict[str, float]] = {}

    all_ratios: List[float] = []

    all_gaps: List[float] = []

    for mode, records in model_detailed.items():
        oracle_records = {int(r["seed"]): r for r in oracle_detailed.get(mode, [])}

        ratios: List[float] = []

        gaps: List[float] = []

        solved = 0

        for r in records:
            seed = int(r.get("seed", -1))
            oracle = oracle_records.get(seed)

            if oracle is None:
                continue

            oracle_obj = float(oracle.get("objective", 0.0))

            model_money = float(r.get("final_money", 0.0))

            if np.isfinite(oracle_obj) and oracle_obj > 1e-6:
                ratio = model_money / oracle_obj

                gap = oracle_obj - model_money
                ratios.append(ratio)
                gaps.append(gap)
                all_ratios.append(ratio)
                all_gaps.append(gap)

            if oracle.get("status") == "Optimal":
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


def benchmark_main() -> None:

    parser = argparse.ArgumentParser(description="task2 模型基准评测")
    parser.add_argument(
        "agent_path", type=str, help="模型路径，如 artifacts/checkpoints/level3_best.pt"
    )

    parser.add_argument("--level", type=int, choices=[3, 4, 35], default=3)

    parser.add_argument("--runs", type=int, default=100, help="每个天气模式的评测轮数")

    parser.add_argument("--device", type=str, default=None, help="cuda/cpu")
    parser.add_argument(
        "--weather-modes",
        type=str,
        default=None,
        help="逗号分隔，如 no_sandstorm,sunny_bias",
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="使用随机采样策略评估（默认贪心）"
    )

    parser.add_argument("--with-random-baseline", action="store_true", help="输出随机策略基线")

    parser.add_argument("--with-oracle", action="store_true", help="计算数学规划Oracle上界并对比")
    parser.add_argument(
        "--oracle-time-limit", type=int, default=30, help="Oracle单次求解时限（秒）"
    )
    parser.add_argument(
        "--oracle-parallel-workers",
        type=int,
        default=1,
        help="Oracle并行求解worker数（<=1表示串行）",
    )
    parser.add_argument(
        "--oracle-solver-threads",
        type=int,
        default=None,
        help="单个Oracle求解器线程数（传给HiGHS/CBC）",
    )

    parser.add_argument("--output-json", type=str, default=None, help="保存结果到 JSON")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=None,
        help="若 overall 成功率低于该阈值则返回非0",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.weather_modes:
        weather_modes = [m.strip() for m in args.weather_modes.split(",") if m.strip()]
    else:
        weather_modes = _get_weather_modes(args.level)

    try:
        _validate_weather_modes(args.level, weather_modes)

    except ValueError as error:
        raise SystemExit(str(error)) from error

    model_detailed = evaluate_model(
        agent_path=args.agent_path,
        level=args.level,
        runs=args.runs,
        device=device,
        weather_modes=weather_modes,
        deterministic=(not args.stochastic),
    )

    model_summary = _summarize_detailed(detailed=model_detailed, level=args.level)

    _print_summary("模型策略评测", model_summary)

    random_summary = None

    random_detailed = None

    if args.with_random_baseline:
        random_summary = evaluate_random_baseline(args.level, args.runs, weather_modes)

        _print_summary("随机基线评测", random_summary)

        random_detailed = {}

        for mode in weather_modes:
            env = make_env(level=args.level, weather_mode=mode, seed=None)

            random_detailed[mode] = []
            for seed in range(args.runs):
                record = _random_policy_episode(env, seed=seed)

                record["seed"] = seed

                record["weather_mode"] = mode

                random_detailed[mode].append(record)

    oracle_summary = None

    oracle_compare = None

    oracle_detailed = None

    if args.with_oracle:
        oracle_detailed = _build_oracle_detailed(
            level=args.level,
            runs=args.runs,
            weather_modes=weather_modes,
            oracle_time_limit=args.oracle_time_limit,
            oracle_parallel_workers=args.oracle_parallel_workers,
            oracle_solver_threads=args.oracle_solver_threads,
        )

        oracle_summary = _summarize_oracle_detailed(oracle_detailed, level=args.level)

        _print_oracle_summary("Oracle上界评测", oracle_summary)

        oracle_compare = compare_model_vs_oracle(
            model_detailed=model_detailed,
            oracle_detailed=oracle_detailed,
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
            "model_detailed": model_detailed,
            "random_baseline": random_summary,
            "random_baseline_detailed": random_detailed,
            "oracle": oracle_summary,
            "oracle_detailed": oracle_detailed,
            "model_vs_oracle": oracle_compare,
        }

        out_path = _resolve_output_json_path(args.output_json)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n结果已保存: {out_path}")

    if args.min_success_rate is not None:
        overall_sr = model_summary["overall"]["success_rate"]

        if overall_sr < args.min_success_rate:
            raise SystemExit(
                f"Benchmark failed: overall success_rate={overall_sr:.4f} < {args.min_success_rate:.4f}"
            )


def _get_weather_modes(level: int) -> List[str]:

    if level == 3:
        cfg = Level3Config
    elif level == 4:
        cfg = Level4Config
    else:
        cfg = Level35Config

    return list(cfg.WEATHER_MODES.keys())


def _get_init_money(level: int) -> float:
    cfg = Level3Config if level == 3 else Level4Config
    return float(cfg.INIT_MONEY)


def _validate_weather_modes(level: int, weather_modes: List[str]) -> None:

    valid_modes = set(_get_weather_modes(level))

    invalid = [mode for mode in weather_modes if mode not in valid_modes]
    if invalid:
        raise ValueError(
            f"Invalid weather modes for level {level}: {invalid}. "
            f"Valid modes: {sorted(valid_modes)}"
        )


def _summarize(results: List[Dict], level: int) -> Dict[str, float]:
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

    init_money = _get_init_money(level)
    returns = [float(r["final_money"]) - init_money for r in results]

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


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:

    return {
        "reached": bool(result.get("reached", False)),
        "final_money": float(result.get("final_money", 0.0)),
        "return": float(result.get("return", 0.0)),
        "length": int(result.get("length", 0)),
    }


def _summarize_detailed(
    detailed: Dict[str, List[Dict]],
    level: int,
) -> Dict[str, Dict[str, float]]:

    summary: Dict[str, Dict[str, float]] = {}

    all_results: List[Dict[str, Any]] = []

    for mode, records in detailed.items():
        normalized = [_normalize_result(r) for r in records]

        summary[mode] = _summarize(normalized, level)

        all_results.extend(normalized)

    summary["overall"] = _summarize(all_results, level)

    return summary


def _random_policy_episode(env, seed: Optional[int] = None) -> Dict:
    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)

    done = False

    steps = 0

    episode_return = 0.0

    while not done and steps < 200:
        va = env.get_valid_actions()

        move = int(rng.choice(va["valid_moves"]))

        mine = bool(va.get("can_mine", False) and rng.random() < 0.5)

        if va.get("can_buy", False):
            max_w = int(va.get("max_buy_water", 0))

            max_f = int(va.get("max_buy_food", 0))

            buy_water = int(rng.integers(0, max_w + 1)) if max_w > 0 else 0

            buy_food = int(rng.integers(0, max_f + 1)) if max_f > 0 else 0
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


def _print_summary(title: str, summary: Dict[str, Dict[str, float]]) -> None:

    print(f"\n{title}")

    print("=" * 78)

    print(f"{'模式':<18}{'成功率':>10}{'均资金':>12}{'均净收益':>12}{'均步长':>10}{'早死率':>10}")

    for mode, stats in summary.items():
        print(
            f"{mode:<18}"
            f"{stats['success_rate'] * 100:>9.1f}%"
            f"{stats['avg_final_money']:>12.2f}"
            f"{stats['avg_return']:>12.2f}"
            f"{stats['avg_length']:>10.1f}"
            f"{stats['early_fail_rate'] * 100:>9.1f}%"
        )


def _build_oracle_detailed(
    level: int,
    runs: int,
    weather_modes: List[str],
    oracle_time_limit: int,
    oracle_parallel_workers: int = 1,
    oracle_solver_threads: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:

    detailed: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in weather_modes}

    for mode in weather_modes:
        env = make_env(level=level, weather_mode=mode, seed=None)
        seeds: List[int] = []
        weather_seqs: List[List[int]] = []
        for seed in range(runs):
            env.reset(seed=seed)

            if env.state is None:
                continue

            weather_seq = list(env.state.weather_future)
            seeds.append(seed)
            weather_seqs.append(weather_seq)

        if oracle_parallel_workers is not None and oracle_parallel_workers > 1:
            oracle_results = solve_theoretical_batch(
                level=level,
                weather_seqs=weather_seqs,
                time_limit=oracle_time_limit,
                max_workers=oracle_parallel_workers,
                threads=oracle_solver_threads,
            )
        else:
            if oracle_solver_threads is None:
                oracle_results = [
                    solve_theoretical_optimal(
                        level,
                        weather_seq,
                        time_limit=oracle_time_limit,
                    )
                    for weather_seq in weather_seqs
                ]
            else:
                oracle_results = [
                    solve_theoretical_optimal(
                        level,
                        weather_seq,
                        time_limit=oracle_time_limit,
                        threads=oracle_solver_threads,
                    )
                    for weather_seq in weather_seqs
                ]

        for seed, oracle in zip(seeds, oracle_results):
            detailed[mode].append(
                {
                    "seed": seed,
                    "weather_mode": mode,
                    "status": oracle.get("status", "Unknown"),
                    "objective": float(oracle.get("objective", 0.0)),
                    "reached": bool(oracle.get("reached", False)),
                    "final_money": float(oracle.get("final_money", 0.0)),
                    "final_water": float(oracle.get("final_water", 0.0)),
                    "final_food": float(oracle.get("final_food", 0.0)),
                    "reach_day": int(oracle.get("reach_day", 0)),
                    "length": int(oracle.get("length", 0)),
                    "return": float(oracle.get("return", oracle.get("objective", 0.0))),
                }
            )

    return detailed


def _summarize_oracle_detailed(
    detailed: Dict[str, List[Dict[str, Any]]],
    level: int,
) -> Dict[str, Dict[str, float]]:

    summary: Dict[str, Dict[str, float]] = {}

    overall_values: List[float] = []

    overall_solved = 0

    overall_runs = 0

    overall_results: List[Dict[str, Any]] = []

    for mode, records in detailed.items():
        values: List[float] = []

        solved = 0

        normalized = [_normalize_result(r) for r in records]

        base_summary = _summarize(normalized, level)

        overall_results.extend(normalized)
        for r in records:
            obj = float(r.get("objective", 0.0))

            if np.isfinite(obj):
                values.append(obj)

                overall_values.append(obj)

            if r.get("status") == "Optimal":
                solved += 1

                overall_solved += 1
        runs = len(records)

        overall_runs += runs

        avg_obj = float(mean(values)) if values else 0.0

        summary[mode] = {
            **base_summary,
            "oracle_solved_rate": solved / runs if runs > 0 else 0.0,
            "oracle_avg_objective": avg_obj,
        }

    overall_base = _summarize(overall_results, level)

    summary["overall"] = {
        **overall_base,
        "oracle_solved_rate": overall_solved / overall_runs if overall_runs > 0 else 0.0,
        "oracle_avg_objective": float(mean(overall_values)) if overall_values else 0.0,
    }

    return summary


def _print_oracle_summary(title: str, summary: Dict[str, Dict[str, float]]) -> None:

    print(f"\n{title}")

    print("=" * 78)

    if "oracle_avg_objective" in next(iter(summary.values()), {}):
        print(f"{'模式':<18}{'求解率':>10}{'均资金':>12}{'均净收益':>12}{'均步长':>10}")

        for mode, stats in summary.items():
            print(
                f"{mode:<18}"
                f"{stats.get('oracle_solved_rate', 0.0) * 100:>9.1f}%"
                f"{stats.get('avg_final_money', 0.0):>16.2f}"
                f"{stats.get('avg_return', 0.0):>12.2f}"
                f"{stats.get('avg_length', 0.0):>12.2f}"
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


def _resolve_output_json_path(output_arg: str) -> Path:

    path = Path(output_arg)

    if path.exists() and path.is_dir():
        return path / "benchmark.json"

    if path.suffix == "" and not path.exists():
        path.mkdir(parents=True, exist_ok=True)

        return path / "benchmark.json"

    return path


if __name__ == "__main__":
    benchmark_main()
    benchmark_main()
