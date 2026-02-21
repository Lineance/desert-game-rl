from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.env.config import CHECKPOINTS_DIR, RESULTS_DIR, Level4Config, RLConfig
from src.env.environment import make_env
from src.models.agent import HybridRNNAgent, create_agent
from src.models.ppo import PPOTrainer
from src.pipeline.warmup import warmup_with_oracle


@dataclass(frozen=True)
class ExitCriteria:
    success_rate_min: float = 0.0
    avg_mine_days_min: float = 0.0
    avg_steps_min: float = 0.0
    reached_mine_rate_min: float = 0.0
    mine_days_after_reach_min: float = 0.0
    avg_net_profit_min: float = -float("inf")
    avg_mine_days_range: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class StageSpec:
    stage_id: int
    name: str
    init_money: int
    num_days: int
    start_node_one_based: int
    villages_one_based: Sequence[int]
    removed_edges_one_based: Sequence[Tuple[int, int]]
    weather_mode: str
    warmup_episodes: int
    warmup_oracle_interval: int
    max_rl_episodes: int
    min_rl_episodes: int
    eval_interval: int
    eval_window: int
    exit_criteria: ExitCriteria
    fallback_note: str


@dataclass
class PipelineConfig:
    level: int = 4
    device: Optional[str] = None
    seed: int = 42
    log_interval: int = 20
    checkpoint_interval: int = 100
    oracle_time_limit: int = 5
    oracle_warmup_time_limit: int = 5
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay_ratio: float = 0.70
    stage2_min_warmup_accepted: int = 50
    stage2_relaxed_extra_episodes: int = 120
    stage1_start_node_one_based: int = 18
    output_prefix: str = "level4_pipeline"


FULL_MAP_EDGES_ONE_BASED: List[Tuple[int, int]] = list(Level4Config.EDGES)

# Stage1（矿山生存）：限制矿区到终点侧的直连走廊，强调先挖矿再规划离开。
STAGE1_REMOVED_EDGES_ONE_BASED: List[Tuple[int, int]] = [
    # 右三角 block 3 4 5 9 10 15
    (4, 9),
    (9, 10),
    (10, 15),
    # 下三角 block 11 16 17 21 22 23
    (16, 21),
    (12, 17),
    (18, 17),
    (22, 23),
    (11, 16),
    (18, 23),
    (23, 24),
    # OPEN LATTER
    (13, 12),
    (13, 8),
    (6, 11),
    (11, 12),
    (2, 3),
    (3, 8),
]

# Stage2（寻矿路径）：削弱主路径直行，强化从起点向矿区分支
STAGE2_REMOVED_EDGES_ONE_BASED: List[Tuple[int, int]] = [
    # 右三角 block 3 4 5 9 10 15
    (4, 9),
    (9, 10),
    (10, 15),
    # 下三角 block 11 16 17 21 22 23
    (16, 21),
    (12, 17),
    (18, 17),
    (22, 23),
    (11, 16),
    (18, 23),
    (23, 24),
    # OPEN LATTER
    (6, 11),
    (11, 12),
    (2, 3),
    (3, 8),
]


def build_stage_specs(cfg: PipelineConfig) -> List[StageSpec]:
    stage1_start = 13

    return [
        StageSpec(
            stage_id=1,
            name="mine_survival",
            init_money=5000,
            num_days=30,
            start_node_one_based=stage1_start,
            villages_one_based=[14],
            removed_edges_one_based=STAGE1_REMOVED_EDGES_ONE_BASED,
            weather_mode="balanced",
            warmup_episodes=350,
            warmup_oracle_interval=4,
            max_rl_episodes=700,
            min_rl_episodes=120,
            eval_interval=20,
            eval_window=80,
            exit_criteria=ExitCriteria(
                success_rate_min=0.70,
                avg_mine_days_min=4.0,
                avg_steps_min=8.0,
            ),
            fallback_note=(
                "Stage1 未达标：建议将 INIT_MONEY 降到 2000，"
                "并在 warmup 增加 30-50 条停留挖矿示范。"
            ),
        ),
        StageSpec(
            stage_id=2,
            name="path_to_mine",
            init_money=5000,
            num_days=30,
            start_node_one_based=1,
            villages_one_based=[],
            removed_edges_one_based=STAGE2_REMOVED_EDGES_ONE_BASED,
            weather_mode="balanced",
            warmup_episodes=260,
            warmup_oracle_interval=5,
            max_rl_episodes=900,
            min_rl_episodes=150,
            eval_interval=20,
            eval_window=100,
            exit_criteria=ExitCriteria(
                success_rate_min=0.60,
                reached_mine_rate_min=0.80,
                mine_days_after_reach_min=3.0,
            ),
            fallback_note=(
                "Stage2 未达标：建议先简化地图（缩短起点到矿山距离），"
                "待到矿率>60%后再恢复完整绕路地图。"
            ),
        ),
        StageSpec(
            stage_id=3,
            name="global_optimization",
            init_money=10000,
            num_days=30,
            start_node_one_based=1,
            villages_one_based=[14],
            removed_edges_one_based=[],
            weather_mode="balanced",
            warmup_episodes=60,
            warmup_oracle_interval=3,
            max_rl_episodes=1200,
            min_rl_episodes=200,
            eval_interval=25,
            eval_window=120,
            exit_criteria=ExitCriteria(
                success_rate_min=0.75,
                avg_net_profit_min=0.0,
                avg_mine_days_range=(1.5, 3.0),
            ),
            fallback_note=(
                "Stage3 回到保守直达：建议回退 Stage2 再训练，或增加轻量挖矿 shaping reward。"
            ),
        ),
    ]


def _to_zero_based(node_one_based: int) -> int:
    return int(node_one_based) - 1


def _norm_edge(edge: Tuple[int, int]) -> Tuple[int, int]:
    u, v = int(edge[0]), int(edge[1])
    return (u, v) if u <= v else (v, u)


def _build_edges(removed_edges_one_based: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    removed = {_norm_edge(edge) for edge in removed_edges_one_based}
    return [edge for edge in FULL_MAP_EDGES_ONE_BASED if _norm_edge(edge) not in removed]


def _stage_override(stage: StageSpec) -> Dict[str, Any]:
    return {
        "CONFIG_NAME": f"Level4PipelineStage{stage.stage_id}",
        "NUM_NODES": Level4Config.NUM_NODES,
        "NUM_DAYS": stage.num_days,
        "INIT_MONEY": stage.init_money,
        "WEIGHT_LIMIT": Level4Config.WEIGHT_LIMIT,
        "MINE_INCOME": Level4Config.MINE_INCOME,
        "WATER_WEIGHT": Level4Config.WATER_WEIGHT,
        "WATER_PRICE_BASE": Level4Config.WATER_PRICE_BASE,
        "FOOD_WEIGHT": Level4Config.FOOD_WEIGHT,
        "FOOD_PRICE_BASE": Level4Config.FOOD_PRICE_BASE,
        "START": _to_zero_based(stage.start_node_one_based),
        "END": Level4Config.END,
        "MINES": list(Level4Config.MINES),
        "VILLAGES": [_to_zero_based(v) for v in stage.villages_one_based],
        "EDGES": _build_edges(stage.removed_edges_one_based),
        "WEATHER_MODES": dict(Level4Config.WEATHER_MODES),
        "WEATHER_TRANSITION": Level4Config.WEATHER_TRANSITION,
    }


def _compute_epsilon(episode: int, max_episodes: int, cfg: PipelineConfig) -> float:
    decay_span = max(1, int(max_episodes * cfg.epsilon_decay_ratio))
    if episode >= decay_span:
        return cfg.epsilon_end
    ratio = episode / decay_span
    return cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start)


def _stage_episode_epsilon(
    episode: int,
    stage_id: int,
    max_episodes: int,
    cfg: PipelineConfig,
) -> float:
    if episode == 0 and stage_id > 1:
        return cfg.epsilon_start
    return _compute_epsilon(episode, max_episodes, cfg)


def _node3_to_mine_choice(path_history: Sequence[int]) -> Optional[float]:
    node3 = _to_zero_based(3)
    mine_branch = _to_zero_based(8)
    for idx in range(len(path_history) - 1):
        if path_history[idx] == node3:
            return 1.0 if path_history[idx + 1] == mine_branch else 0.0
    return None


def _stage_metrics(window_records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    n = max(1, len(window_records))
    success_rate = sum(r["success"] for r in window_records) / n
    avg_steps = sum(r["steps"] for r in window_records) / n
    avg_mine_days = sum(r["mine_days"] for r in window_records) / n
    avg_net_profit = sum(r["net_profit"] for r in window_records) / n
    reached_mine_rate = sum(r["reached_mine"] for r in window_records) / n

    reached_records = [r for r in window_records if r["reached_mine"] > 0.5]
    reached_n = max(1, len(reached_records))
    mine_days_after_reach = sum(r["mine_days"] for r in reached_records) / reached_n

    node3_decisions = [
        r["node3_choice_to_mine"]
        for r in window_records
        if not math.isnan(r["node3_choice_to_mine"])
    ]
    node3_prob = sum(node3_decisions) / len(node3_decisions) if node3_decisions else float("nan")

    return {
        "success_rate": float(success_rate),
        "avg_steps": float(avg_steps),
        "avg_mine_days": float(avg_mine_days),
        "avg_net_profit": float(avg_net_profit),
        "reached_mine_rate": float(reached_mine_rate),
        "mine_days_after_reach": float(mine_days_after_reach),
        "node3_choice_prob": float(node3_prob),
    }


def _pass_criteria(stage: StageSpec, metrics: Dict[str, float]) -> bool:
    c = stage.exit_criteria
    if metrics["success_rate"] < c.success_rate_min:
        return False
    if metrics["avg_mine_days"] < c.avg_mine_days_min:
        return False
    if metrics["avg_steps"] < c.avg_steps_min:
        return False
    if metrics["reached_mine_rate"] < c.reached_mine_rate_min:
        return False
    if metrics["mine_days_after_reach"] < c.mine_days_after_reach_min:
        return False
    if metrics["avg_net_profit"] < c.avg_net_profit_min:
        return False
    if c.avg_mine_days_range is not None:
        low, high = c.avg_mine_days_range
        if not (low <= metrics["avg_mine_days"] <= high):
            return False
    return True


def _warn_level35_trap(records: Sequence[Dict[str, float]], shortest_len: int) -> bool:
    if len(records) < 50:
        return False
    first = list(records[:50])
    avg_steps = sum(r["steps"] for r in first) / len(first)
    avg_mine_days = sum(r["mine_days"] for r in first) / len(first)
    return avg_steps < (shortest_len + 2) and avg_mine_days < 0.5


def _stage2_warmup_filter(payload: Dict[str, Any]) -> bool:
    mine_node = int(Level4Config.MINES[0])
    path = payload.get("path_history", [])
    if not isinstance(path, list):
        return False
    return mine_node in path


def _should_relax_stage2_warmup(
    stage: StageSpec,
    warmup_summary: Dict[str, float],
    min_accepted: int,
) -> bool:
    if stage.stage_id != 2:
        return False
    accepted = int(warmup_summary.get("accepted_episodes", 0.0))
    return accepted < max(0, int(min_accepted))


def _merge_warmup_summaries(primary: Dict[str, float], extra: Dict[str, float]) -> Dict[str, float]:
    merged = dict(primary)

    for key in ["episodes", "accepted_episodes", "filtered_episodes", "solved"]:
        merged[key] = float(primary.get(key, 0.0)) + float(extra.get(key, 0.0))

    total_accepted = max(1.0, float(merged.get("accepted_episodes", 0.0)))
    p_acc = float(primary.get("accepted_episodes", 0.0))
    e_acc = float(extra.get("accepted_episodes", 0.0))

    for key in [
        "avg_steps",
        "avg_mine_per_episode",
        "avg_buy_water",
        "avg_buy_food",
        "avg_value_loss",
        "match_rate",
    ]:
        weighted = p_acc * float(primary.get(key, 0.0)) + e_acc * float(extra.get(key, 0.0))
        merged[key] = weighted / total_accepted

    total_solved = max(1.0, float(merged.get("solved", 0.0)))
    p_sol = float(primary.get("solved", 0.0))
    e_sol = float(extra.get("solved", 0.0))
    merged["success_rate"] = (
        p_sol * float(primary.get("success_rate", 0.0))
        + e_sol * float(extra.get("success_rate", 0.0))
    ) / total_solved

    merged["final_value_loss"] = float(
        extra.get("final_value_loss", primary.get("final_value_loss", 0.0))
    )
    merged["value_loss_trend"] = float(
        extra.get("value_loss_trend", primary.get("value_loss_trend", 0.0))
    )
    return merged


def _run_stage_warmup(
    stage: StageSpec,
    agent: HybridRNNAgent,
    trainer: PPOTrainer,
    env,
    cfg: PipelineConfig,
) -> Dict[str, float]:
    if stage.warmup_episodes <= 0:
        return {"episodes": 0.0, "accepted_episodes": 0.0, "filtered_episodes": 0.0}

    filter_fn = _stage2_warmup_filter if stage.stage_id == 2 else None
    summary = warmup_with_oracle(
        agent=agent,
        trainer=trainer,
        env=env,
        level=cfg.level,
        episodes=stage.warmup_episodes,
        time_limit=cfg.oracle_warmup_time_limit,
        device=cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        log_interval=max(10, stage.eval_interval),
        trajectory_filter=filter_fn,
        oracle_config=env.config,
        oracle_solve_interval=stage.warmup_oracle_interval,
    )

    if _should_relax_stage2_warmup(stage, summary, cfg.stage2_min_warmup_accepted):
        accepted = int(summary.get("accepted_episodes", 0.0))
        missing = max(0, int(cfg.stage2_min_warmup_accepted) - accepted)
        relaxed_extra = max(int(cfg.stage2_relaxed_extra_episodes), missing)
        print(
            "[warmup放宽] Stage2 有效轨迹不足: "
            f"accepted={accepted} < {cfg.stage2_min_warmup_accepted}, "
            f"追加无过滤 warmup {relaxed_extra} episodes"
        )
        relaxed = warmup_with_oracle(
            agent=agent,
            trainer=trainer,
            env=env,
            level=cfg.level,
            episodes=relaxed_extra,
            time_limit=cfg.oracle_warmup_time_limit,
            device=cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            log_interval=max(10, stage.eval_interval),
            trajectory_filter=None,
            oracle_config=env.config,
            oracle_solve_interval=max(1, stage.warmup_oracle_interval),
        )
        summary = _merge_warmup_summaries(summary, relaxed)

    return summary


def run_level4_pipeline(cfg: PipelineConfig) -> None:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stage_specs = build_stage_specs(cfg)
    metrics_csv = RESULTS_DIR / f"{cfg.output_prefix}_metrics.csv"

    with open(metrics_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "stage_id",
                "stage_name",
                "episode",
                "success_rate",
                "avg_steps",
                "avg_mine_days",
                "avg_net_profit",
                "reached_mine_rate",
                "mine_days_after_reach",
                "node3_choice_prob",
                "trap_risk",
            ],
        )
        writer.writeheader()

        agent: Optional[HybridRNNAgent] = None
        trainer: Optional[PPOTrainer] = None

        for stage in stage_specs:
            override = _stage_override(stage)
            env = make_env(
                level=cfg.level,
                weather_mode=stage.weather_mode,
                seed=cfg.seed,
                config_override=override,
            )

            if agent is None:
                rl_cfg = RLConfig()
                agent = create_agent(env, rl_cfg, device=device)
                trainer = PPOTrainer(agent, rl_cfg, device=device)
            assert trainer is not None

            print("=" * 70)
            print(f"Stage {stage.stage_id}: {stage.name}")
            print(
                f"init_money={stage.init_money}, start_node={stage.start_node_one_based}, "
                f"villages={list(stage.villages_one_based)}, weather_mode={stage.weather_mode}"
            )

            warmup_summary = _run_stage_warmup(stage, agent, trainer, env, cfg)
            print(
                f"warmup: episodes={int(warmup_summary.get('episodes', 0))}, "
                f"accepted={int(warmup_summary.get('accepted_episodes', 0))}, "
                f"filtered={int(warmup_summary.get('filtered_episodes', 0))}, "
                f"avg_mine={warmup_summary.get('avg_mine_per_episode', 0.0):.2f}"
            )

            stage_records: List[Dict[str, float]] = []
            shortest_len = (
                int(env.dist_to_end[env.config.START]) if env.dist_to_end is not None else 0
            )
            passed = False
            best_success_rate = float("-inf")

            for ep in range(stage.max_rl_episodes):
                epsilon = _stage_episode_epsilon(ep, stage.stage_id, stage.max_rl_episodes, cfg)
                rollout, last_value, ep_info = trainer.collect_rollout(
                    env,
                    max_steps=env.config.NUM_DAYS + 2,
                    epsilon=epsilon,
                )
                if rollout.rewards:
                    trainer.update(rollout, last_value)

                path_history = list(env.state.path_history) if env.state is not None else []
                mine_node = int(Level4Config.MINES[0])
                reached_mine = 1.0 if mine_node in path_history else 0.0
                node3_choice = _node3_to_mine_choice(path_history)

                stage_records.append(
                    {
                        "success": 1.0 if ep_info.get("reached", False) else 0.0,
                        "steps": float(ep_info.get("length", 0)),
                        "mine_days": float(ep_info.get("mine_count", 0)),
                        "net_profit": float(ep_info.get("final_money", 0.0))
                        - float(env.config.INIT_MONEY),
                        "reached_mine": reached_mine,
                        "node3_choice_to_mine": float(node3_choice)
                        if node3_choice is not None
                        else float("nan"),
                    }
                )

                if ep % cfg.log_interval == 0:
                    print(
                        f"stage={stage.stage_id} ep={ep} "
                        f"success={ep_info.get('reached', False)} steps={ep_info.get('length', 0)} "
                        f"mine={ep_info.get('mine_count', 0)} money={ep_info.get('final_money', 0.0):.1f}"
                    )

                if ep + 1 < stage.min_rl_episodes:
                    continue
                if (ep + 1) % stage.eval_interval != 0:
                    continue

                window = stage_records[-stage.eval_window :]
                metrics = _stage_metrics(window)
                trap_risk = _warn_level35_trap(stage_records, shortest_len)

                writer.writerow(
                    {
                        "stage_id": stage.stage_id,
                        "stage_name": stage.name,
                        "episode": ep + 1,
                        "success_rate": metrics["success_rate"],
                        "avg_steps": metrics["avg_steps"],
                        "avg_mine_days": metrics["avg_mine_days"],
                        "avg_net_profit": metrics["avg_net_profit"],
                        "reached_mine_rate": metrics["reached_mine_rate"],
                        "mine_days_after_reach": metrics["mine_days_after_reach"],
                        "node3_choice_prob": metrics["node3_choice_prob"],
                        "trap_risk": 1.0 if trap_risk else 0.0,
                    }
                )
                fp.flush()

                print(
                    f"stage={stage.stage_id} eval@{ep + 1}: "
                    f"success={metrics['success_rate']:.2%}, mine={metrics['avg_mine_days']:.2f}, "
                    f"steps={metrics['avg_steps']:.2f}, mine_reach={metrics['reached_mine_rate']:.2%}, "
                    f"net={metrics['avg_net_profit']:.1f}, node3->mine={metrics['node3_choice_prob']:.2f}"
                )

                if metrics["success_rate"] > best_success_rate:
                    best_success_rate = metrics["success_rate"]
                    stage_best_ckpt = (
                        CHECKPOINTS_DIR / f"{cfg.output_prefix}_stage{stage.stage_id}_best.pt"
                    )
                    agent.save(str(stage_best_ckpt))

                if trap_risk:
                    print(
                        "[风险] 检测到 Level35 陷阱信号：步长过短且挖矿率过低。"
                        "建议降低资金或删除更多捷径边后重启本阶段。"
                    )

                if _pass_criteria(stage, metrics):
                    passed = True
                    stage_ckpt = CHECKPOINTS_DIR / f"{cfg.output_prefix}_stage{stage.stage_id}.pt"
                    agent.save(str(stage_ckpt))
                    print(f"Stage {stage.stage_id} 达标，已保存: {stage_ckpt}")
                    break

            if not passed:
                print(f"Stage {stage.stage_id} 未达标。{stage.fallback_note}")
                final_path = (
                    CHECKPOINTS_DIR / f"{cfg.output_prefix}_failed_stage{stage.stage_id}.pt"
                )
                agent.save(str(final_path))
                print(f"已保存失败现场模型: {final_path}")
                return

        final_ckpt = CHECKPOINTS_DIR / f"{cfg.output_prefix}_final.pt"
        assert agent is not None
        agent.save(str(final_ckpt))
        print("=" * 70)
        print(f"三阶段训练完成，最终模型: {final_ckpt}")
        print(f"阶段指标CSV: {metrics_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Level4 三阶段资源约束训练管线")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle-time-limit", type=int, default=30)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--stage1-start-node",
        type=int,
        default=18,
        choices=[17, 18],
        help="Stage1 起点（节点编号，1-based）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="level4_pipeline",
        help="输出文件名前缀",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = PipelineConfig(
        device=args.device,
        seed=int(args.seed),
        log_interval=int(args.log_interval),
        checkpoint_interval=int(args.checkpoint_interval),
        oracle_time_limit=int(args.oracle_time_limit),
        stage1_start_node_one_based=int(args.stage1_start_node),
        output_prefix=str(args.output_prefix).strip() or "level4_pipeline",
    )
    run_level4_pipeline(cfg)
