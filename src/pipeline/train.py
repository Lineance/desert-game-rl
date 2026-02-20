import argparse
import csv
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.env.config import CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR, RLConfig
from src.env.environment import make_env
from src.models.agent import HybridRNNAgent, create_agent
from src.models.ppo import PPOTrainer
from src.pipeline.oracle import solve_theoretical_plan


def _load_checkpoint(path: Path, device: str) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format")
    return checkpoint


def _resolve_resume_path(resume: Optional[str], level: int) -> Optional[Path]:
    if resume is None:
        return None
    resume = resume.strip()
    if not resume:
        return None
    if resume.lower() == "latest":
        return CHECKPOINTS_DIR / f"level{level}_latest.pt"
    return Path(resume)


def _build_agent_from_checkpoint(
    checkpoint: Dict[str, Any],
    fallback_config: RLConfig,
    device: str,
) -> Tuple[HybridRNNAgent, RLConfig]:
    config = checkpoint.get("config", fallback_config)
    obs_dim = int(checkpoint["obs_dim"])
    num_locations = int(checkpoint["num_locations"])
    agent = HybridRNNAgent(obs_dim=obs_dim, num_locations=num_locations, config=config)
    agent.load_state_dict(checkpoint["state_dict"])
    agent.to(device)
    return agent, config


def _save_training_state(
    path: Path,
    agent: HybridRNNAgent,
    trainer: PPOTrainer,
    episode: int,
    best_return: float,
    stats: Optional[Dict[str, float]],
    epsilon: float,
) -> None:
    scheduler = getattr(trainer, "scheduler", None) or getattr(trainer, "lr_scheduler", None)
    scheduler_state = scheduler.state_dict() if scheduler is not None else None

    cuda_rng_state: Optional[List[torch.Tensor]] = None
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state_all()

    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "lr_scheduler_state_dict": scheduler_state,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": cuda_rng_state,
            },
            "episode": int(episode),
            "best_return": float(best_return),
            "epsilon": float(epsilon),
            "stats": stats,
            "config": agent.config,
            "obs_dim": agent.obs_dim,
            "num_locations": agent.num_locations,
        },
        path,
    )


def _restore_rng_state(rng_state: Optional[Dict[str, Any]]) -> None:
    if not isinstance(rng_state, dict):
        return

    python_state = rng_state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = rng_state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    cuda_states = rng_state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(cuda_states, list):
        device_count = torch.cuda.device_count()
        for device_idx in range(min(device_count, len(cuda_states))):
            torch.cuda.set_rng_state(cuda_states[device_idx], device=device_idx)


def _compute_epsilon(
    episode: int, eps_start: float, eps_end: float, eps_decay_episodes: int
) -> float:
    progress = min(1.0, episode / max(1, eps_decay_episodes))
    return eps_start + (eps_end - eps_start) * progress


def _safe_save_latest(
    save_dir: Path,
    level: int,
    agent: HybridRNNAgent,
    trainer: PPOTrainer,
    episode: int,
    best_return: float,
    stats: Optional[Dict[str, float]],
    epsilon: float,
    reason: str,
) -> None:
    try:
        _save_training_state(
            save_dir / f"level{level}_latest.pt",
            agent,
            trainer,
            episode,
            best_return,
            stats,
            epsilon,
        )
        print(f"已自动保存latest断点({reason}): episode={episode}, epsilon={epsilon:.3f}")
    except Exception as save_error:
        print(f"自动保存latest失败({reason}): {save_error}")


def _init_structured_loggers(
    level: int,
) -> Tuple[TextIO, csv.DictWriter, Optional[Any], Path, Optional[Path]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"level{level}_train_metrics.csv"
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "episode",
            "return",
            "success",
            "length",
            "final_money",
            "net_profit",
            "epsilon",
            "rolling_return",
            "rolling_net_profit",
            "rolling_success_rate",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
        ],
    )
    csv_writer.writeheader()

    tb_writer = None
    tb_dir = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_dir = LOGS_DIR / "tensorboard" / f"level{level}_{timestamp}"
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
    except Exception as tensorboard_error:
        print(f"TensorBoard不可用，已跳过TB日志：{tensorboard_error}")

    return csv_file, csv_writer, tb_writer, csv_path, tb_dir


def _sanitize_oracle_action(action: Dict[str, Any], env, valid_actions: Dict[str, Any]) -> Dict:
    move = int(action.get("move", env.state.position))
    valid_moves = valid_actions.get("valid_moves", []) if valid_actions else []
    if valid_moves and move not in valid_moves:
        move = int(env.state.position)

    can_mine = bool(valid_actions.get("can_mine", False)) if valid_actions else False
    mine = bool(action.get("mine", False)) and can_mine

    if valid_actions and bool(valid_actions.get("can_buy", False)):
        max_w = int(valid_actions.get("max_buy_water", 0))
        max_f = int(valid_actions.get("max_buy_food", 0))
        buy_water = max(0, min(int(action.get("buy_water", 0)), max_w))
        buy_food = max(0, min(int(action.get("buy_food", 0)), max_f))
    else:
        buy_water = 0
        buy_food = 0

    return {
        "move": move,
        "mine": mine,
        "buy_water": buy_water,
        "buy_food": buy_food,
    }


def _oracle_warmup(
    agent: HybridRNNAgent,
    trainer: PPOTrainer,
    env,
    level: int,
    episodes: int,
    time_limit: int,
    device: str,
) -> None:
    if episodes <= 0:
        return
    if level not in (3, 35, 4):
        print("Oracle预热仅支持level 3/35/4，已跳过。")
        return

    agent.train()
    total_loss = 0.0
    total_steps = 0
    solved = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        if env.state is None:
            continue

        weather_seq = list(env.state.weather_future)
        oracle = solve_theoretical_plan(level, weather_seq, time_limit=time_limit)
        plan = oracle.get("plan", [])
        if not plan:
            if ep % 10 == 0:
                print(f"Oracle预热: episode={ep}, plan为空，跳过")
            continue

        solved += 1
        episode_loss = None
        episode_steps = 0

        for oracle_action in plan:
            valid_actions = env.get_valid_actions()
            action = _sanitize_oracle_action(oracle_action, env, valid_actions)

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            state = agent.encode_observation(obs_tensor)
            action_tensor = {
                "move": torch.LongTensor([action["move"]]).to(device),
                "mine": torch.FloatTensor([action["mine"]]).to(device),
                "buy_water": torch.FloatTensor([action["buy_water"]]).to(device),
                "buy_food": torch.FloatTensor([action["buy_food"]]).to(device),
            }
            log_prob, _ = agent.actor.evaluate_actions(state, action_tensor, valid_actions)
            step_loss = -log_prob.mean()
            episode_loss = step_loss if episode_loss is None else episode_loss + step_loss

            next_obs, _, done, truncated, _ = env.step(action)
            obs = next_obs
            episode_steps += 1
            if done or truncated:
                break

        if episode_loss is None or episode_steps == 0:
            continue

        trainer.optimizer.zero_grad()
        episode_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), trainer.config.MAX_GRAD_NORM)
        trainer.optimizer.step()

        total_loss += float(episode_loss.item())
        total_steps += episode_steps

        if ep % 10 == 0:
            avg_loss = total_loss / max(1, solved)
            avg_steps = total_steps / max(1, solved)
            print(
                f"Oracle预热进度: episode={ep}, solved={solved}, "
                f"avg_steps={avg_steps:.1f}, avg_loss={avg_loss:.4f}"
            )

    avg_loss = total_loss / max(1, solved)
    avg_steps = total_steps / max(1, solved)
    print(
        f"Oracle预热完成: episodes={episodes}, solved={solved}, "
        f"avg_steps={avg_steps:.1f}, avg_loss={avg_loss:.4f}"
    )


def _log_structured_metrics(
    csv_writer: csv.DictWriter,
    tb_writer: Optional[Any],
    episode: int,
    episode_info: Dict[str, Any],
    epsilon: float,
    rolling_return: float,
    rolling_net_profit: float,
    rolling_success_rate: float,
    stats: Optional[Dict[str, float]],
) -> None:
    net_profit = float(episode_info.get("net_profit", 0.0))
    row = {
        "episode": int(episode),
        "return": float(episode_info.get("return", 0.0)),
        "success": 1.0 if bool(episode_info.get("reached", False)) else 0.0,
        "length": float(episode_info.get("length", 0.0)),
        "final_money": float(episode_info.get("final_money", 0.0)),
        "net_profit": net_profit,
        "epsilon": float(epsilon),
        "rolling_return": float(rolling_return),
        "rolling_net_profit": float(rolling_net_profit),
        "rolling_success_rate": float(rolling_success_rate),
        "policy_loss": float(stats["policy_loss"]) if stats is not None else float("nan"),
        "value_loss": float(stats["value_loss"]) if stats is not None else float("nan"),
        "entropy": float(stats["entropy"]) if stats is not None else float("nan"),
        "approx_kl": float(stats["approx_kl"]) if stats is not None else float("nan"),
        "clip_fraction": float(stats["clip_fraction"]) if stats is not None else float("nan"),
    }
    csv_writer.writerow(row)

    if tb_writer is not None:
        tb_writer.add_scalar("train/return", row["return"], episode)
        tb_writer.add_scalar("train/net_profit", row["net_profit"], episode)
        tb_writer.add_scalar("train/success_rate", row["rolling_success_rate"], episode)
        tb_writer.add_scalar("train/epsilon", row["epsilon"], episode)
        tb_writer.add_scalar("train/rolling_net_profit", row["rolling_net_profit"], episode)
        if stats is not None:
            tb_writer.add_scalar("train/policy_loss", row["policy_loss"], episode)
            tb_writer.add_scalar("train/value_loss", row["value_loss"], episode)
            tb_writer.add_scalar("train/entropy", row["entropy"], episode)
            tb_writer.add_scalar("train/approx_kl", row["approx_kl"], episode)
            tb_writer.add_scalar("train/clip_fraction", row["clip_fraction"], episode)


def train(
    level: int = 3,
    num_episodes: int = 2000,
    device: Optional[str] = None,
    log_interval: int = 100,
    resume: Optional[str] = None,
    checkpoint_interval: int = 100,
    weather_mode: Optional[str] = None,
    oracle_warmup_episodes: int = 0,
    oracle_time_limit: int = 20,
) -> None:
    """训练入口。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = CHECKPOINTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    config = RLConfig()
    env = make_env(level=level, seed=42, weather_mode=weather_mode or "balanced")

    resume_path = _resolve_resume_path(resume, level)
    checkpoint = None
    if resume_path is not None and resume_path.exists():
        checkpoint = _load_checkpoint(resume_path, device)
        if "config" in checkpoint:
            config = checkpoint["config"]
        print(f"加载断点: {resume_path}")
    elif resume_path is not None:
        print(f"未找到断点文件: {resume_path}，将从头训练。")

    if checkpoint is not None and "state_dict" in checkpoint:
        agent, config = _build_agent_from_checkpoint(checkpoint, config, device)
    else:
        agent = create_agent(env, config, device=device)

    trainer = PPOTrainer(agent, config, device=device)

    best_return = float("-inf")
    stats: Optional[Dict[str, float]] = None
    start_episode = 0
    resume_epsilon: Optional[float] = None

    if checkpoint is not None and "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = getattr(trainer, "scheduler", None) or getattr(trainer, "lr_scheduler", None)
        if scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        _restore_rng_state(checkpoint.get("rng_state"))
        start_episode = int(checkpoint.get("episode", -1)) + 1
        best_return = float(checkpoint.get("best_return", best_return))
        resume_epsilon = (
            float(checkpoint.get("epsilon")) if checkpoint.get("epsilon") is not None else None
        )
        stats = checkpoint.get("stats")

    if checkpoint is not None and "state_dict" in checkpoint and start_episode == 0:
        best_return = float(checkpoint.get("best_return", best_return))
        stats = checkpoint.get("stats")

    if oracle_warmup_episodes > 0 and start_episode == 0:
        print("开始Oracle蒸馏预热...")
        _oracle_warmup(
            agent,
            trainer,
            env,
            level,
            oracle_warmup_episodes,
            oracle_time_limit,
            device,
        )
    elif oracle_warmup_episodes > 0 and start_episode > 0:
        print("已从断点恢复训练，跳过Oracle蒸馏预热。")
    rolling_window = 100
    recent_returns = deque(maxlen=rolling_window)
    recent_success = deque(maxlen=rolling_window)
    recent_lengths = deque(maxlen=rolling_window)
    recent_final_money = deque(maxlen=rolling_window)
    recent_net_profit = deque(maxlen=rolling_window)
    recent_mine = deque(maxlen=rolling_window)
    recent_buy = deque(maxlen=rolling_window)
    recent_move = deque(maxlen=rolling_window)
    recent_stay = deque(maxlen=rolling_window)
    recent_buy_water = deque(maxlen=rolling_window)
    recent_buy_food = deque(maxlen=rolling_window)

    print("=" * 60)
    print("训练开始")
    available_modes = ", ".join(sorted(env.config.WEATHER_MODES.keys()))
    selected_mode = weather_mode or "balanced"
    print(f"天气模式: {selected_mode} (可选: {available_modes})")
    print("=" * 60)

    csv_file, csv_writer, tb_writer, csv_path, tb_dir = _init_structured_loggers(level)
    print(f"CSV日志输出: {csv_path}")
    if tb_dir is not None:
        print(f"TensorBoard日志输出: {tb_dir}")

    max_steps = env.config.NUM_DAYS + 2
    eps_start = 0.30
    eps_end = 0.05
    eps_decay_episodes = max(1, int(num_episodes * 0.7))

    if start_episode > 0:
        derived_epsilon = _compute_epsilon(start_episode, eps_start, eps_end, eps_decay_episodes)
        if resume_epsilon is not None:
            print(
                f"恢复断点进度: start_episode={start_episode}, "
                f"saved_epsilon={resume_epsilon:.3f}, next_epsilon={derived_epsilon:.3f}"
            )
        else:
            print(
                f"恢复断点进度: start_episode={start_episode}, next_epsilon={derived_epsilon:.3f}"
            )

    if start_episode >= num_episodes:
        print(f"断点续训起始回合 {start_episode} >= 总回合 {num_episodes}，无需继续训练。")
        return

    last_completed_episode = start_episode - 1
    last_epsilon = _compute_epsilon(last_completed_episode, eps_start, eps_end, eps_decay_episodes)

    try:
        for episode in range(start_episode, num_episodes):
            epsilon = _compute_epsilon(episode, eps_start, eps_end, eps_decay_episodes)

            rollout_buffer, last_value, episode_info = trainer.collect_rollout(
                env,
                max_steps=max_steps,
                epsilon=epsilon,
            )

            if rollout_buffer.rewards:
                stats = trainer.update(rollout_buffer, last_value)

            final_money = float(episode_info.get("final_money", 0.0))
            episode_info["net_profit"] = final_money - float(env.config.INIT_MONEY)

            recent_returns.append(float(episode_info["return"]))
            recent_success.append(1.0 if episode_info["reached"] else 0.0)
            recent_lengths.append(float(episode_info["length"]))
            recent_final_money.append(float(episode_info.get("final_money", 0.0)))
            recent_net_profit.append(float(episode_info.get("net_profit", 0.0)))
            recent_mine.append(float(episode_info.get("mine_count", 0)))
            recent_buy.append(float(episode_info.get("buy_count", 0)))
            recent_move.append(float(episode_info.get("move_count", 0)))
            recent_stay.append(float(episode_info.get("stay_count", 0)))
            recent_buy_water.append(float(episode_info.get("total_buy_water", 0)))
            recent_buy_food.append(float(episode_info.get("total_buy_food", 0)))

            window_n = max(1, len(recent_success))
            rolling_success_rate = sum(recent_success) / window_n
            rolling_return = sum(recent_returns) / window_n
            rolling_net_profit = sum(recent_net_profit) / window_n

            _log_structured_metrics(
                csv_writer,
                tb_writer,
                episode,
                episode_info,
                epsilon,
                rolling_return,
                rolling_net_profit,
                rolling_success_rate,
                stats,
            )
            csv_file.flush()

            if episode_info["return"] > best_return:
                best_return = episode_info["return"]
                agent.save(str(save_dir / f"level{level}_best.pt"))

            if episode % log_interval == 0:
                rolling_length = sum(recent_lengths) / window_n
                rolling_money = sum(recent_final_money) / window_n
                rolling_mine = sum(recent_mine) / window_n
                rolling_buy = sum(recent_buy) / window_n
                rolling_move = sum(recent_move) / window_n
                rolling_stay = sum(recent_stay) / window_n
                rolling_buy_w = sum(recent_buy_water) / window_n
                rolling_buy_f = sum(recent_buy_food) / window_n

                print(f"\nEpisode {episode}")
                print(f"  最近{window_n}局成功率: {rolling_success_rate * 100:.1f}%")
                print(
                    f"  最近{window_n}局: 均回报={rolling_return:.1f}, 均天数={rolling_length:.1f}, "
                    f"均资金={rolling_money:.1f}, 均净收益={rolling_net_profit:.1f}"
                )
                print(
                    f"  最近{window_n}局行为: 均移动={rolling_move:.1f}, 均停留={rolling_stay:.1f}, "
                    f"均挖矿={rolling_mine:.1f}, 均购买={rolling_buy:.1f}, "
                    f"均购水={rolling_buy_w:.1f}, 均购食={rolling_buy_f:.1f}"
                )
                print(f"  最佳回报: {best_return:.1f}")
                print(
                    f"  本回合: 奖励={episode_info['return']:.1f}, "
                    f"到达={episode_info['reached']}, 步数={episode_info['length']}, 天数={episode_info['final_day']}, "
                    f"资金={episode_info['final_money']:.1f}, 净收益={episode_info.get('net_profit', 0.0):.1f}"
                )
                if stats is not None:
                    print(
                        "  损失: "
                        f"P={stats['policy_loss']:.4f}, "
                        f"V={stats['value_loss']:.4f}, "
                        f"H={stats['entropy']:.4f}, "
                        f"KL={stats['approx_kl']:.4f}, "
                        f"ClipFrac={stats['clip_fraction']:.3f}, "
                        f"ExplVar={stats['explained_variance']:.3f}"
                    )
                    print(
                        "  更新诊断: "
                        f"effective_updates={int(stats['effective_updates'])}/{config.EPOCHS_PER_UPDATE}, "
                        f"target_kl_hit={bool(stats['target_kl_hit'])}, "
                        f"ratio=[{stats['min_ratio']:.3f}, {stats['max_ratio']:.3f}], "
                        f"epsilon={epsilon:.3f}"
                    )

            if checkpoint_interval > 0 and episode % checkpoint_interval == 0:
                _save_training_state(
                    save_dir / f"level{level}_latest.pt",
                    agent,
                    trainer,
                    episode,
                    best_return,
                    stats,
                    epsilon,
                )

            if episode % 500 == 0 and episode > 0:
                agent.save(str(save_dir / f"level{level}_episode{episode}.pt"))

            last_completed_episode = episode
            last_epsilon = epsilon

    except KeyboardInterrupt:
        print("\n训练被手动中断，正在自动保存latest断点...")
        _safe_save_latest(
            save_dir,
            level,
            agent,
            trainer,
            last_completed_episode,
            best_return,
            stats,
            last_epsilon,
            reason="keyboard_interrupt",
        )
        if tb_writer is not None:
            tb_writer.close()
        csv_file.close()
        return
    except Exception:
        print("\n训练异常中断，正在自动保存latest断点...")
        _safe_save_latest(
            save_dir,
            level,
            agent,
            trainer,
            last_completed_episode,
            best_return,
            stats,
            last_epsilon,
            reason="exception",
        )
        if tb_writer is not None:
            tb_writer.close()
        csv_file.close()
        raise

    agent.save(str(save_dir / f"level{level}_final.pt"))
    _save_training_state(
        save_dir / f"level{level}_latest.pt",
        agent,
        trainer,
        num_episodes - 1,
        best_return,
        stats,
        _compute_epsilon(num_episodes - 1, eps_start, eps_end, eps_decay_episodes),
    )
    print(f"\n训练完成！最佳奖励: {best_return:.1f}")
    if tb_writer is not None:
        tb_writer.close()
    csv_file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="断点续训模型路径，或使用 'latest' 选择最近保存",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--weather-mode",
        type=str,
        default=None,
        help="天气模式（如 no_sandstorm/sunny_bias/hot_bias/train_easy/train_medium/eval）",
    )
    parser.add_argument(
        "--oracle-warmup-episodes",
        type=int,
        default=0,
        help="Oracle蒸馏预热回合数（仅level 3/4，默认关闭）",
    )
    parser.add_argument(
        "--oracle-time-limit",
        type=int,
        default=20,
        help="Oracle求解时间上限（秒）",
    )
    args = parser.parse_args()

    train(
        level=args.level,
        num_episodes=args.episodes,
        device=args.device,
        log_interval=args.log_interval,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        weather_mode=args.weather_mode,
        oracle_warmup_episodes=args.oracle_warmup_episodes,
        oracle_time_limit=args.oracle_time_limit,
    )
