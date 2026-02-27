from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _make_env(*args, **kwargs):
    from src.env.environment import make_env

    return make_env(*args, **kwargs)


def _solve_theoretical_plan(*args, **kwargs):
    from src.utils.oracle import solve_theoretical_plan

    return solve_theoretical_plan(*args, **kwargs)


def _solve_theoretical_batch(*args, **kwargs):
    from src.utils.oracle import solve_theoretical_batch

    return solve_theoretical_batch(*args, **kwargs)


def _default_output_path(level: int) -> Path:
    from src.env.config import RESULTS_DIR

    return RESULTS_DIR / f"level{level}_bc_dataset.json"


def generate_behavior_cloning_dataset(
    level: int,
    episodes: int,
    weather_mode: Optional[str] = None,
    time_limit: int = 20,
    seed_start: int = 0,
    output_path: Optional[str] = None,
    include_unsolved: bool = False,
    oracle_parallel_workers: int = 1,
    oracle_solver_threads: Optional[int] = None,
    show_progress: bool = False,
    progress_interval: int = 20,
) -> Dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    env_kwargs: Dict[str, Any] = {"level": level, "seed": None}
    if weather_mode is not None:
        env_kwargs["weather_mode"] = weather_mode
    env = _make_env(**env_kwargs)
    progress_interval = max(1, int(progress_interval))

    if show_progress:
        print(
            "BC数据集生成开始: "
            f"level={level}, episodes={episodes}, weather_mode={weather_mode}, "
            f"oracle_time_limit={time_limit}, parallel_workers={oracle_parallel_workers}, "
            f"solver_threads={oracle_solver_threads}"
        )

    records = []
    solved = 0
    episodes_list: List[int] = []
    weather_seqs: List[List[int]] = []

    for offset in range(episodes):
        episode = int(seed_start + offset)
        env.reset(seed=episode)
        weather_seq = list(env.state.weather_future) if env.state is not None else []

        episodes_list.append(episode)
        weather_seqs.append(weather_seq)
        if show_progress and ((offset + 1) % progress_interval == 0 or (offset + 1) == episodes):
            print(f"天气序列准备进度: {offset + 1}/{episodes}")

    if oracle_parallel_workers > 1:
        if show_progress:
            print(
                f"开始并行Oracle求解: tasks={len(weather_seqs)}, "
                f"workers={oracle_parallel_workers}, solver_threads={oracle_solver_threads}"
            )
        results = _solve_theoretical_batch(
            level=level,
            weather_seqs=weather_seqs,
            time_limit=time_limit,
            return_plan=True,
            max_workers=oracle_parallel_workers,
            threads=oracle_solver_threads,
            progress_every=(progress_interval if show_progress else 0),
            progress_prefix="BC Oracle",
        )
    else:
        if show_progress:
            print(f"开始串行Oracle求解: tasks={len(weather_seqs)}")
        if oracle_solver_threads is None:
            results = []
            for idx, weather_seq in enumerate(weather_seqs, start=1):
                results.append(
                    _solve_theoretical_plan(
                        level,
                        weather_seq,
                        time_limit=time_limit,
                    )
                )
                if show_progress and (idx % progress_interval == 0 or idx == len(weather_seqs)):
                    print(f"BC Oracle: {idx}/{len(weather_seqs)} completed")
        else:
            results = []
            for idx, weather_seq in enumerate(weather_seqs, start=1):
                results.append(
                    _solve_theoretical_plan(
                        level,
                        weather_seq,
                        time_limit=time_limit,
                        threads=oracle_solver_threads,
                    )
                )
                if show_progress and (idx % progress_interval == 0 or idx == len(weather_seqs)):
                    print(f"BC Oracle: {idx}/{len(weather_seqs)} completed")

    for episode, weather_seq, result in zip(episodes_list, weather_seqs, results):
        status = str(result.get("status", "Unknown"))
        reached = bool(result.get("reached", False))
        plan = result.get("plan", [])
        is_solved = status.lower() == "optimal" and reached and bool(plan)
        if is_solved:
            solved += 1

        if include_unsolved or is_solved:
            records.append(
                {
                    "episode": episode,
                    "seed": episode,
                    "weather_seq": weather_seq,
                    "status": status,
                    "reached": reached,
                    "plan": plan,
                }
            )

    if output_path is None:
        output_file = _default_output_path(level)
    else:
        output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "level": int(level),
            "episodes": int(episodes),
            "seed_start": int(seed_start),
            "weather_mode": weather_mode,
            "time_limit": int(time_limit),
            "include_unsolved": bool(include_unsolved),
            "oracle_parallel_workers": int(oracle_parallel_workers),
            "oracle_solver_threads": oracle_solver_threads,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "records": records,
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if show_progress:
        print(f"写入完成: {output_file}")

    return {
        "level": int(level),
        "episodes": int(episodes),
        "solved": int(solved),
        "records": int(len(records)),
        "saved_path": str(output_file),
    }


def load_behavior_cloning_dataset_index(path: str | Path) -> Dict[int, Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"BC数据集不存在: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Invalid BC dataset format")

    index: Dict[int, Dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        if "episode" not in item:
            continue
        index[int(item["episode"])] = item
    return index


def bc_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="生成可复用的Behavior Cloning数据集")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--weather-mode", type=str, default=None)
    parser.add_argument("--oracle-time-limit", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--include-unsolved", action="store_true")
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
        help="单个Oracle求解器线程数",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭过程进度输出，仅输出最终结果",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=20,
        help="进度输出间隔（默认20）",
    )
    args = parser.parse_args()

    summary = generate_behavior_cloning_dataset(
        level=args.level,
        episodes=args.episodes,
        weather_mode=args.weather_mode,
        time_limit=args.oracle_time_limit,
        seed_start=args.seed_start,
        output_path=args.output,
        include_unsolved=args.include_unsolved,
        oracle_parallel_workers=args.oracle_parallel_workers,
        oracle_solver_threads=args.oracle_solver_threads,
        show_progress=not args.quiet,
        progress_interval=args.progress_interval,
    )
    print(
        "BC数据集生成完成: "
        f"level={summary['level']}, episodes={summary['episodes']}, "
        f"solved={summary['solved']}, records={summary['records']}, "
        f"saved={summary['saved_path']}"
    )


if __name__ == "__main__":
    bc_dataset_main()
