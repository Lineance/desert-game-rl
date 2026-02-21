import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import torch

from src.env.config import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    Level3Config,
    Level4Config,
    Level35Config,
    RLConfig,
)
from src.env.environment import make_env
from src.models.agent import HybridRNNAgent, create_agent
from src.models.ppo import PPOTrainer
from src.pipeline.warmup import warmup_with_oracle


def _resolve_weather_mode(level: int, requested_mode: Optional[str]) -> str:
    if level == 3:
        modes = Level3Config.WEATHER_MODES
        preferred_default = "no_sandstorm"
    elif level == 35:
        modes = Level35Config.WEATHER_MODES
        preferred_default = "train_medium"
    elif level == 4:
        modes = Level4Config.WEATHER_MODES
        preferred_default = "balanced"
    else:
        raise ValueError(f"Unknown level: {level}")

    if requested_mode is not None:
        mode = requested_mode.strip()
        if mode in modes:
            return mode
        fallback = preferred_default if preferred_default in modes else next(iter(modes.keys()))
        print(f"未识别weather_mode='{requested_mode}'，已回退到'{fallback}'")
        return fallback

    if preferred_default in modes:
        return preferred_default
    return next(iter(modes.keys()))


def _init_pretrain_loggers(
    level: int,
) -> tuple[TextIO, csv.DictWriter, Optional[Any], Path, Optional[Path]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"level{level}_pretrain_metrics.csv"
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "episode",
            "policy_loss",
            "value_loss",
            "episode_loss",
            "steps",
            "first_value",
            "first_return",
            "mine_days",
            "buy_water",
            "buy_food",
            "success",
            "match_rate",
            "student_steps",
            "student_mine_days",
            "student_buy_water",
            "student_buy_food",
            "student_success",
        ],
    )
    csv_writer.writeheader()

    tb_writer = None
    tb_dir = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_dir = LOGS_DIR / "tensorboard" / f"pretrain_level{level}_{timestamp}"
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
    except Exception as tensorboard_error:
        print(f"TensorBoard不可用，已跳过TB日志：{tensorboard_error}")

    return csv_file, csv_writer, tb_writer, csv_path, tb_dir


def _resolve_resume_path(resume: Optional[str], level: int) -> Optional[Path]:
    if resume is None:
        return None
    resume = resume.strip()
    if not resume:
        return None
    if resume.lower() == "latest":
        return CHECKPOINTS_DIR / f"level{level}_pretrain_latest.pt"
    return Path(resume)


def _load_pretrain_agent_from_checkpoint(path: Path, device: str) -> HybridRNNAgent:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format")

    if all(k in checkpoint for k in ("state_dict", "obs_dim", "num_locations", "config")):
        agent = HybridRNNAgent(
            obs_dim=int(checkpoint["obs_dim"]),
            num_locations=int(checkpoint["num_locations"]),
            config=checkpoint["config"],
        )
        agent.load_state_dict(checkpoint["state_dict"])
        agent.to(device)
        return agent

    if all(k in checkpoint for k in ("agent_state_dict", "obs_dim", "num_locations", "config")):
        agent = HybridRNNAgent(
            obs_dim=int(checkpoint["obs_dim"]),
            num_locations=int(checkpoint["num_locations"]),
            config=checkpoint["config"],
        )
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.to(device)
        return agent

    raise ValueError("Unsupported checkpoint format for pretrain resume")


def _infer_resume_episode(path: Path, checkpoint: Optional[Dict[str, Any]] = None) -> int:
    if isinstance(checkpoint, dict):
        episode = checkpoint.get("episode")
        if isinstance(episode, int) and episode >= 0:
            return episode + 1

    match = re.search(r"_episode(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return 0


def _save_pretrain_latest_checkpoint(path: Path, agent: HybridRNNAgent, episode: int) -> None:
    torch.save(
        {
            "state_dict": agent.state_dict(),
            "config": agent.config,
            "obs_dim": agent.obs_dim,
            "num_locations": agent.num_locations,
            "episode": int(episode),
        },
        path,
    )


def _write_pretrain_metrics(
    csv_writer: csv.DictWriter,
    tb_writer: Optional[Any],
    warmup_metrics: List[Dict[str, float]],
    summary: Dict[str, float],
) -> None:
    for item in warmup_metrics:
        row = {
            "episode": int(item.get("episode", 0)),
            "policy_loss": float(item.get("policy_loss", float("nan"))),
            "value_loss": float(item.get("value_loss", float("nan"))),
            "episode_loss": float(item.get("episode_loss", float("nan"))),
            "steps": float(item.get("steps", 0.0)),
            "first_value": float(item.get("first_value", float("nan"))),
            "first_return": float(item.get("first_return", float("nan"))),
            "mine_days": float(item.get("mine_days", 0.0)),
            "buy_water": float(item.get("buy_water", 0.0)),
            "buy_food": float(item.get("buy_food", 0.0)),
            "success": float(item.get("success", 0.0)),
            "match_rate": float(item.get("match_rate", 0.0)),
            "student_steps": float(item.get("student_steps", 0.0)),
            "student_mine_days": float(item.get("student_mine_days", 0.0)),
            "student_buy_water": float(item.get("student_buy_water", 0.0)),
            "student_buy_food": float(item.get("student_buy_food", 0.0)),
            "student_success": float(item.get("student_success", 0.0)),
        }
        csv_writer.writerow(row)

        if tb_writer is not None:
            episode = row["episode"]
            tb_writer.add_scalar("pretrain/policy_loss", row["policy_loss"], episode)
            tb_writer.add_scalar("pretrain/value_loss", row["value_loss"], episode)
            tb_writer.add_scalar("pretrain/episode_loss", row["episode_loss"], episode)
            tb_writer.add_scalar("pretrain/steps", row["steps"], episode)
            tb_writer.add_scalar("pretrain/mine_days", row["mine_days"], episode)
            tb_writer.add_scalar("pretrain/buy_water", row["buy_water"], episode)
            tb_writer.add_scalar("pretrain/buy_food", row["buy_food"], episode)
            tb_writer.add_scalar("pretrain/success", row["success"], episode)
            tb_writer.add_scalar("pretrain/match_rate", row["match_rate"], episode)
            tb_writer.add_scalar("pretrain/student_steps", row["student_steps"], episode)
            tb_writer.add_scalar("pretrain/student_mine_days", row["student_mine_days"], episode)
            tb_writer.add_scalar("pretrain/student_buy_water", row["student_buy_water"], episode)
            tb_writer.add_scalar("pretrain/student_buy_food", row["student_buy_food"], episode)
            tb_writer.add_scalar("pretrain/student_success", row["student_success"], episode)

    if tb_writer is not None:
        tb_writer.add_scalar("pretrain_summary/success_rate", float(summary["success_rate"]), 0)
        tb_writer.add_scalar("pretrain_summary/avg_steps", float(summary["avg_steps"]), 0)
        tb_writer.add_scalar(
            "pretrain_summary/avg_mine_per_episode",
            float(summary["avg_mine_per_episode"]),
            0,
        )
        tb_writer.add_scalar("pretrain_summary/avg_buy_water", float(summary["avg_buy_water"]), 0)
        tb_writer.add_scalar("pretrain_summary/avg_buy_food", float(summary["avg_buy_food"]), 0)
        tb_writer.add_scalar("pretrain_summary/avg_value_loss", float(summary["avg_value_loss"]), 0)
        tb_writer.add_scalar(
            "pretrain_summary/final_value_loss",
            float(summary["final_value_loss"]),
            0,
        )
        tb_writer.add_scalar("pretrain_summary/match_rate", float(summary["match_rate"]), 0)
        tb_writer.add_scalar(
            "pretrain_summary/student_success_rate",
            float(summary.get("student_success_rate", 0.0)),
            0,
        )
        tb_writer.add_scalar(
            "pretrain_summary/student_avg_steps",
            float(summary.get("student_avg_steps", 0.0)),
            0,
        )
        tb_writer.add_scalar(
            "pretrain_summary/student_avg_mine_per_episode",
            float(summary.get("student_avg_mine_per_episode", 0.0)),
            0,
        )
        tb_writer.add_scalar(
            "pretrain_summary/student_avg_buy_water",
            float(summary.get("student_avg_buy_water", 0.0)),
            0,
        )
        tb_writer.add_scalar(
            "pretrain_summary/student_avg_buy_food",
            float(summary.get("student_avg_buy_food", 0.0)),
            0,
        )


def evaluate_pretrain_metrics(summary: Dict[str, float]) -> Dict[str, object]:
    avg_mine = float(summary.get("avg_mine_per_episode", 0.0))
    avg_steps = float(summary.get("avg_steps", 0.0))
    avg_buy_water = float(summary.get("avg_buy_water", 0.0))
    avg_buy_food = float(summary.get("avg_buy_food", 0.0))
    success_rate = float(summary.get("success_rate", 0.0))
    avg_value_loss = float(summary.get("avg_value_loss", 0.0))
    final_value_loss = float(summary.get("final_value_loss", 0.0))
    value_loss_trend = float(summary.get("value_loss_trend", 0.0))
    match_rate = float(summary.get("match_rate", 0.0))

    checks: List[Dict[str, str]] = []

    mine_status = "PASS"
    if avg_mine < 1.0:
        mine_status = "FAIL"
        mine_note = "avg_mine_per_episode < 1.0：Warmup 失败（未形成挖矿行为）"
    elif avg_mine > 5.0:
        mine_status = "WARN"
        mine_note = "avg_mine_per_episode > 5.0：可能过度挖矿（Warmup 可接受）"
    else:
        mine_note = "矿山访问率达标"
    checks.append({"name": "矿山访问率", "status": mine_status, "note": mine_note})

    step_status = "PASS"
    if avg_steps <= 4.5:
        step_status = "FAIL"
        step_note = "avg_steps 仍接近 4.0：灾难（未学会绕路去矿山）"
    elif avg_steps > 10.0:
        step_status = "WARN"
        step_note = "avg_steps > 10：绕路偏多，后续RL需优化"
    elif avg_steps < 6.0:
        step_status = "WARN"
        step_note = "avg_steps 偏低，建议继续Warmup"
    else:
        step_note = "路径长度达标"
    checks.append({"name": "步长分布", "status": step_status, "note": step_note})

    buy_status = "PASS"
    if avg_buy_water <= 90.0 or avg_buy_food <= 90.0:
        buy_status = "FAIL"
        buy_note = "购买量仍接近 90/90：不合格（重装示范不足）"
    elif avg_buy_water <= 120.0 or avg_buy_food <= 100.0:
        buy_status = "WARN"
        buy_note = "购买量高于基线但未达到建议阈值"
    else:
        buy_note = "资源购买量达标"
    checks.append({"name": "资源购买量", "status": buy_status, "note": buy_note})

    base_status = "PASS"
    base_notes = []
    if success_rate < 1.0:
        base_status = "FAIL"
        base_notes.append("success_rate 必须为 100%")
    if avg_value_loss >= 200.0:
        base_status = "FAIL"
        base_notes.append("critic_loss(均值) 需 < 200")
    if match_rate <= 0.4:
        base_status = "FAIL"
        base_notes.append("match_rate 需 > 40%")
    base_note = "；".join(base_notes) if base_notes else "成功率/价值收敛/匹配率达标"
    checks.append({"name": "成功率与价值收敛", "status": base_status, "note": base_note})

    redline = (
        avg_steps <= 4.5
        and success_rate >= 0.99
        and avg_mine <= 0.1
        and (value_loss_trend > 0.0 or final_value_loss > 600.0)
        and match_rate < 0.2
    )

    overall_pass = all(c["status"] == "PASS" for c in checks)
    return {
        "checks": checks,
        "overall_pass": overall_pass,
        "redline": redline,
    }


def pretrain_only(
    level: int = 3,
    warmup_episodes: int = 200,
    device: Optional[str] = None,
    weather_mode: Optional[str] = None,
    oracle_time_limit: int = 20,
    output_path: Optional[str] = None,
    save_interval: int = 0,
    resume: Optional[str] = None,
) -> None:
    """仅执行Oracle蒸馏预热并保存模型。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if warmup_episodes <= 0:
        raise ValueError("warmup_episodes必须大于0")
    if save_interval < 0:
        raise ValueError("save_interval不能小于0")

    selected_mode = _resolve_weather_mode(level, weather_mode)
    env = make_env(level=level, seed=42, weather_mode=selected_mode)
    resume_path = _resolve_resume_path(resume, level)
    resume_start_episode = 0

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"未找到断点文件: {resume_path}")
        try:
            resume_agent = _load_pretrain_agent_from_checkpoint(resume_path, device)
        except Exception:
            resume_agent = HybridRNNAgent.load(str(resume_path), device=device)
        agent = resume_agent

        try:
            try:
                resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            except TypeError:
                resume_checkpoint = torch.load(resume_path, map_location=device)
        except Exception:
            resume_checkpoint = None
        resume_start_episode = _infer_resume_episode(resume_path, resume_checkpoint)
    else:
        agent = create_agent(env, RLConfig(), device=device)

    trainer = PPOTrainer(agent, agent.config, device=device)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("仅预训练模式: Oracle蒸馏预热")
    print(
        f"level={level}, weather_mode={selected_mode}, "
        f"warmup_episodes={warmup_episodes}, device={device}"
    )
    if resume_path is not None:
        print(f"resume_from={resume_path}, start_episode={resume_start_episode}")
    print("=" * 60)

    csv_file, csv_writer, tb_writer, csv_path, tb_dir = _init_pretrain_loggers(level)
    warmup_metrics: List[Dict[str, float]] = []

    def _save_interval_checkpoint(ep_idx: int, _: Dict[str, float]) -> None:
        if save_interval <= 0:
            return
        episode_number = ep_idx + 1
        if episode_number % save_interval != 0:
            return

        interval_path = CHECKPOINTS_DIR / f"level{level}_pretrain_episode{episode_number}.pt"
        try:
            agent.save(str(interval_path))
            latest_path = CHECKPOINTS_DIR / f"level{level}_pretrain_latest.pt"
            try:
                _save_pretrain_latest_checkpoint(latest_path, agent, episode_number - 1)
            except Exception as latest_save_error:
                print(
                    f"预训练latest断点更新失败: episode={episode_number}, error={latest_save_error}"
                )
            print(f"预训练阶段模型已保存: episode={episode_number}, path={interval_path}")
        except Exception as save_error:
            print(f"预训练阶段模型保存失败: episode={episode_number}, error={save_error}")

    try:
        summary = warmup_with_oracle(
            agent,
            trainer,
            env,
            level,
            warmup_episodes,
            oracle_time_limit,
            device,
            start_episode=resume_start_episode,
            metrics=warmup_metrics,
            on_episode_end=_save_interval_checkpoint,
        )

        _write_pretrain_metrics(csv_writer, tb_writer, warmup_metrics, summary)
    finally:
        csv_file.flush()
        csv_file.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    print(f"预训练episode指标已保存: {csv_path}")
    if tb_dir is not None:
        print(f"TensorBoard日志目录: {tb_dir}")

    report = evaluate_pretrain_metrics(summary)
    print("=" * 60)
    print("预训练指标检测（4项阈值）")
    for item in report["checks"]:
        print(f"[{item['status']}] {item['name']}: {item['note']}")

    if report["redline"]:
        print("[REDLINE] avg_steps≈4 且 solved=100% 但无挖矿、价值损失恶化、匹配率过低。")

    if report["overall_pass"]:
        print("预训练指标检测结果: PASS")
    else:
        print("预训练指标检测结果: FAIL/WARN（建议继续Warmup并检查Oracle计划质量）")
    print("=" * 60)

    save_dir = CHECKPOINTS_DIR

    save_path = Path(output_path) if output_path else save_dir / f"level{level}_pretrained.pt"
    agent.save(str(save_path))
    print(f"预训练模型已保存: {save_path}")

    latest_path = CHECKPOINTS_DIR / f"level{level}_pretrain_latest.pt"
    final_episode = resume_start_episode + warmup_episodes - 1
    try:
        _save_pretrain_latest_checkpoint(latest_path, agent, final_episode)
        print(f"预训练latest断点已更新: {latest_path}")
    except Exception as latest_save_error:
        print(f"预训练latest断点更新失败: {latest_save_error}")


def pretrain_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--warmup-episodes", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--weather-mode",
        type=str,
        default=None,
        help="天气模式（如 no_sandstorm/sunny_bias/hot_bias/train_easy/train_medium/eval）",
    )
    parser.add_argument(
        "--oracle-time-limit",
        type=int,
        default=20,
        help="Oracle求解时间上限（秒）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="预训练模型输出路径（默认 artifacts/checkpoints/level{n}_pretrained.pt）",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="按episode间隔保存预训练阶段模型（0表示关闭）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从已有模型断点继续预训练（可传文件路径或latest）",
    )
    args = parser.parse_args()

    pretrain_only(
        level=args.level,
        warmup_episodes=args.warmup_episodes,
        device=args.device,
        weather_mode=args.weather_mode,
        oracle_time_limit=args.oracle_time_limit,
        output_path=args.output,
        save_interval=args.save_interval,
        resume=args.resume,
    )
