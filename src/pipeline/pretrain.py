import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

import numpy as np
import torch
import torch.nn.functional as F

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
from src.models.agent import Agent, create_agent
from src.models.ppo import PPOTrainer
from src.utils.bc_dataset import load_behavior_cloning_dataset_index
from src.utils.oracle import solve_theoretical_batch, solve_theoretical_plan


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


def behavior_cloning_with_oracle(
    agent: Agent,
    trainer: PPOTrainer,
    env,
    level: int,
    episodes: int,
    time_limit: int,
    device: str,
    value_weight: float = 0.1,
    log_interval: int = 10,
    start_episode: int = 0,
    metrics: Optional[List[Dict[str, float]]] = None,
    on_episode_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    oracle_dataset_path: Optional[str] = None,
    oracle_parallel_workers: int = 1,
    oracle_solver_threads: Optional[int] = None,
    warmup_batch_episodes: int = 1,
) -> Dict[str, float]:
    if episodes <= 0:
        return {
            "episodes": 0.0,
            "solved": 0.0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_mine_per_episode": 0.0,
            "avg_buy_water": 0.0,
            "avg_buy_food": 0.0,
            "avg_value_loss": 0.0,
            "final_value_loss": 0.0,
            "value_loss_trend": 0.0,
            "match_rate": 0.0,
        }

    warmup_batch_episodes = max(1, int(warmup_batch_episodes))

    agent.train()
    total_loss = 0.0
    total_steps = 0
    solved = 0
    success = 0
    total_mine_days = 0
    total_buy_water = 0
    total_buy_food = 0
    matched_actions = 0
    total_actions = 0
    value_losses: List[float] = []
    student_total_steps = 0
    student_total_mine_days = 0
    student_total_buy_water = 0
    student_total_buy_food = 0
    student_success = 0
    student_eval_episodes = 0
    dataset_index = (
        load_behavior_cloning_dataset_index(oracle_dataset_path)
        if oracle_dataset_path is not None
        else None
    )
    weather_mode = str(getattr(env, "weather_mode", "balanced"))
    prefetched_oracle_results: Dict[int, Dict[str, Any]] = {}
    pending_losses: List[torch.Tensor] = []
    pending_count = 0
    optimizer_steps = 0

    def _flush_pending_losses() -> None:
        nonlocal pending_count, optimizer_steps
        if not pending_losses:
            return
        batch_loss = torch.stack(pending_losses).mean()
        if not torch.isfinite(batch_loss):
            pending_losses.clear()
            pending_count = 0
            return
        trainer.optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), trainer.config.MAX_GRAD_NORM)
        trainer.optimizer.step()
        pending_losses.clear()
        pending_count = 0
        optimizer_steps += 1

    if dataset_index is None and oracle_parallel_workers > 1:
        prefetch_env = make_env(level=level, weather_mode=weather_mode, seed=None)
        weather_seqs: List[List[int]] = []
        episodes_to_solve: List[int] = []
        for ep in range(start_episode, start_episode + episodes):
            prefetch_env.reset(seed=ep)
            if prefetch_env.state is None:
                continue
            weather_seqs.append(list(prefetch_env.state.weather_future))
            episodes_to_solve.append(ep)

        if weather_seqs:
            prefetch_results = solve_theoretical_batch(
                level=level,
                weather_seqs=weather_seqs,
                time_limit=time_limit,
                return_plan=True,
                max_workers=oracle_parallel_workers,
                threads=oracle_solver_threads,
            )
            prefetched_oracle_results = {
                episode: result
                for episode, result in zip(episodes_to_solve, prefetch_results, strict=True)
            }

    for ep in range(start_episode, start_episode + episodes):
        should_eval_student = (log_interval > 0) and (ep % log_interval == 0)
        episode_record: Dict[str, float] = {
            "episode": float(ep),
            "solved": 0.0,
            "success": 0.0,
            "steps": 0.0,
            "mine_days": 0.0,
            "buy_water": 0.0,
            "buy_food": 0.0,
            "match_rate": 0.0,
            "student_steps": float("nan"),
            "student_mine_days": float("nan"),
            "student_buy_water": float("nan"),
            "student_buy_food": float("nan"),
            "student_success": float("nan"),
        }
        obs, _ = env.reset(seed=ep)
        if env.state is None:
            if on_episode_end is not None:
                on_episode_end(ep, dict(episode_record))
            continue

        if should_eval_student:
            try:
                student_eval = _rollout_student_policy(agent, level, weather_mode, ep)
                student_total_steps += int(student_eval["steps"])
                student_total_mine_days += int(student_eval["mine_days"])
                student_total_buy_water += int(student_eval["buy_water"])
                student_total_buy_food += int(student_eval["buy_food"])
                student_success += int(student_eval["success"])
                student_eval_episodes += 1
                episode_record["student_steps"] = float(student_eval["steps"])
                episode_record["student_mine_days"] = float(student_eval["mine_days"])
                episode_record["student_buy_water"] = float(student_eval["buy_water"])
                episode_record["student_buy_food"] = float(student_eval["buy_food"])
                episode_record["student_success"] = float(student_eval["success"])
            except Exception:
                episode_record["student_steps"] = float("nan")
                episode_record["student_mine_days"] = float("nan")
                episode_record["student_buy_water"] = float("nan")
                episode_record["student_buy_food"] = float("nan")
                episode_record["student_success"] = float("nan")

        weather_seq = list(env.state.weather_future)
        oracle_status = "Unknown"
        oracle_reached = False
        plan = []

        used_dataset = False
        if dataset_index is not None:
            dataset_record = dataset_index.get(ep)
            if isinstance(dataset_record, dict):
                dataset_weather = dataset_record.get("weather_seq", [])
                if list(dataset_weather) == weather_seq:
                    oracle_status = str(dataset_record.get("status", "Unknown"))
                    oracle_reached = bool(dataset_record.get("reached", False))
                    plan = dataset_record.get("plan", [])
                    used_dataset = True

        if not used_dataset:
            if prefetched_oracle_results:
                oracle = prefetched_oracle_results.get(ep, {})
            else:
                if oracle_solver_threads is None:
                    oracle = solve_theoretical_plan(
                        level,
                        weather_seq,
                        time_limit=time_limit,
                    )
                else:
                    oracle = solve_theoretical_plan(
                        level,
                        weather_seq,
                        time_limit=time_limit,
                        threads=oracle_solver_threads,
                    )
            oracle_status = str(oracle.get("status", "Unknown"))
            oracle_reached = bool(oracle.get("reached", False))
            plan = oracle.get("plan", [])

        oracle_exact_solved = oracle_status.lower() == "optimal" and oracle_reached and bool(plan)
        if not oracle_exact_solved:
            if ep % log_interval == 0:
                print(
                    "Oracle预热: "
                    f"episode={ep}, status={oracle_status}, reached={oracle_reached}, "
                    f"plan_len={len(plan)}, source={'dataset' if used_dataset else 'oracle'}, 非精确解，跳过"
                )
            if on_episode_end is not None:
                on_episode_end(ep, dict(episode_record))
            continue

        solved += 1
        episode_record["solved"] = 1.0
        episode_steps = 0
        episode_mine_days = 0
        episode_buy_water = 0
        episode_buy_food = 0
        episode_matched = 0
        episode_total_actions = 0
        episode_success = 0
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []

        for oracle_action in plan:
            valid_actions = env.get_valid_actions()
            action = _sanitize_oracle_action(oracle_action, env, valid_actions)

            pred_action, _ = agent.select_action(obs, valid_actions, deterministic=True)
            move_match = int(pred_action.get("move", env.state.position)) == int(action["move"])
            if action["buy_water"] == 0:
                water_match = int(pred_action.get("buy_water", 0)) == 0
            else:
                water_match = (
                    abs(int(pred_action.get("buy_water", 0)) - int(action["buy_water"]))
                    / max(1.0, float(action["buy_water"]))
                    <= 0.1
                )
            if action["buy_food"] == 0:
                food_match = int(pred_action.get("buy_food", 0)) == 0
            else:
                food_match = (
                    abs(int(pred_action.get("buy_food", 0)) - int(action["buy_food"]))
                    / max(1.0, float(action["buy_food"]))
                    <= 0.1
                )
            if move_match and water_match and food_match:
                matched_actions += 1
                episode_matched += 1
            total_actions += 1
            episode_total_actions += 1

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            state, node_embeddings, day_norm = agent.encode_observation_full(obs_tensor)
            action_tensor = {
                "move": torch.LongTensor([action["move"]]).to(device),
                "mine": torch.FloatTensor([action["mine"]]).to(device),
                "mine_intensity": torch.FloatTensor([action.get("mine_intensity", 0.0)]).to(device),
                "buy_water": torch.FloatTensor([action["buy_water"]]).to(device),
                "buy_food": torch.FloatTensor([action["buy_food"]]).to(device),
            }
            log_prob, _ = agent.actor.evaluate_actions(
                state,
                node_embeddings,
                day_norm,
                action_tensor,
                valid_actions,
            )
            value = 0.5 * (agent.survival_critic(state) + agent.fund_critic(state))
            value = value.squeeze(-1)
            if not torch.isfinite(log_prob).all():
                log_prob = torch.zeros_like(log_prob)
            if not torch.isfinite(value).all():
                value = torch.zeros_like(value)
            log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))

            next_obs, reward, done, truncated, _ = env.step(action)
            last_action = env.state.last_action if env.state is not None else None
            if isinstance(last_action, dict):
                if bool(last_action.get("mine", False)):
                    total_mine_days += 1
                    episode_mine_days += 1
                total_buy_water += int(last_action.get("buy_water", 0))
                total_buy_food += int(last_action.get("buy_food", 0))
                episode_buy_water += int(last_action.get("buy_water", 0))
                episode_buy_food += int(last_action.get("buy_food", 0))
            rewards.append(float(reward))
            obs = next_obs
            episode_steps += 1
            if done or truncated:
                break

        if episode_steps == 0 or not log_probs or not values:
            if on_episode_end is not None:
                episode_record["steps"] = float(episode_steps)
                on_episode_end(ep, dict(episode_record))
            continue

        returns: List[float] = []
        running_return = 0.0
        for r in reversed(rewards):
            running_return = r + trainer.config.GAMMA * running_return
            returns.insert(0, running_return)

        policy_loss = -torch.stack(log_probs).mean()
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        values_tensor = torch.stack(values)
        value_loss = F.mse_loss(values_tensor, returns_tensor)
        episode_loss = policy_loss + value_weight * value_loss
        if not torch.isfinite(episode_loss):
            if ep % log_interval == 0:
                print(
                    "Oracle预热: "
                    f"episode={ep}, 非有限loss(policy={float(policy_loss.item()):.4f}, "
                    f"value={float(value_loss.item()):.4f})，跳过本次参数更新"
                )
            if on_episode_end is not None:
                episode_record["steps"] = float(episode_steps)
                on_episode_end(ep, dict(episode_record))
            continue
        value_losses.append(float(value_loss.item()))

        pending_losses.append(episode_loss)
        pending_count += 1
        if pending_count >= warmup_batch_episodes:
            _flush_pending_losses()

        total_loss += float(episode_loss.item())
        total_steps += episode_steps

        if metrics is not None:
            if env.state is not None and bool(env.state.reached):
                episode_success = 1
            episode_record["steps"] = float(episode_steps)
            episode_record["mine_days"] = float(episode_mine_days)
            episode_record["buy_water"] = float(episode_buy_water)
            episode_record["buy_food"] = float(episode_buy_food)
            episode_record["success"] = float(episode_success)
            episode_record["match_rate"] = float(episode_matched / max(1, episode_total_actions))
            episode_record["value_loss"] = float(value_loss.item())
            metrics.append(
                {
                    "episode": float(ep),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "episode_loss": float(episode_loss.item()),
                    "steps": float(episode_steps),
                    "first_value": float(values_tensor[0].item())
                    if len(values_tensor) > 0
                    else float("nan"),
                    "first_return": float(returns_tensor[0].item())
                    if len(returns_tensor) > 0
                    else float("nan"),
                    "mine_days": float(episode_mine_days),
                    "buy_water": float(episode_buy_water),
                    "buy_food": float(episode_buy_food),
                    "success": float(episode_success),
                    "match_rate": float(episode_matched / max(1, episode_total_actions)),
                    "student_steps": float(episode_record.get("student_steps", 0.0)),
                    "student_mine_days": float(episode_record.get("student_mine_days", 0.0)),
                    "student_buy_water": float(episode_record.get("student_buy_water", 0.0)),
                    "student_buy_food": float(episode_record.get("student_buy_food", 0.0)),
                    "student_success": float(episode_record.get("student_success", 0.0)),
                }
            )
        else:
            if env.state is not None and bool(env.state.reached):
                episode_success = 1
            episode_record["steps"] = float(episode_steps)
            episode_record["mine_days"] = float(episode_mine_days)
            episode_record["buy_water"] = float(episode_buy_water)
            episode_record["buy_food"] = float(episode_buy_food)
            episode_record["success"] = float(episode_success)
            episode_record["match_rate"] = float(episode_matched / max(1, episode_total_actions))
            episode_record["value_loss"] = float(value_loss.item())

        if log_interval > 0 and ep % log_interval == 0:
            avg_loss = total_loss / max(1, solved)
            avg_steps = total_steps / max(1, solved)
            avg_mine = total_mine_days / max(1, solved)
            match_rate = matched_actions / max(1, total_actions)
            student_avg_steps = student_total_steps / max(1, student_eval_episodes)
            student_avg_mine = student_total_mine_days / max(1, student_eval_episodes)
            student_success_rate = student_success / max(1, student_eval_episodes)
            print(
                f"Oracle预热进度: episode={ep}, solved={solved}, "
                f"avg_steps={avg_steps:.1f}, avg_mine={avg_mine:.2f}, "
                f"match_rate={match_rate:.2%}, avg_loss={avg_loss:.4f}; "
                f"student(avg_steps={student_avg_steps:.1f}, avg_mine={student_avg_mine:.2f}, "
                f"success={student_success_rate:.2%}, eval_points={student_eval_episodes})"
            )

        if env.state is not None and bool(env.state.reached):
            success += 1

        if on_episode_end is not None:
            on_episode_end(ep, dict(episode_record))

    _flush_pending_losses()

    avg_loss = total_loss / max(1, solved)
    avg_steps = total_steps / max(1, solved)
    avg_mine = total_mine_days / max(1, solved)
    avg_buy_water = total_buy_water / max(1, solved)
    avg_buy_food = total_buy_food / max(1, solved)
    success_rate = success / max(1, episodes)
    success_rate_given_solved = success / max(1, solved)
    match_rate = matched_actions / max(1, total_actions)
    avg_value_loss = sum(value_losses) / max(1, len(value_losses))
    final_value_loss = value_losses[-1] if value_losses else 0.0
    value_loss_trend = value_losses[-1] - value_losses[0] if len(value_losses) >= 2 else 0.0
    student_avg_steps = student_total_steps / max(1, student_eval_episodes)
    student_avg_mine = student_total_mine_days / max(1, student_eval_episodes)
    student_avg_buy_water = student_total_buy_water / max(1, student_eval_episodes)
    student_avg_buy_food = student_total_buy_food / max(1, student_eval_episodes)
    student_success_rate = student_success / max(1, student_eval_episodes)
    print(
        f"Oracle预热完成: episodes={episodes}, solved={solved}, "
        f"success_rate={success_rate:.2%}, avg_steps={avg_steps:.1f}, "
        f"avg_mine={avg_mine:.2f}, avg_buy_w={avg_buy_water:.1f}, "
        f"avg_buy_f={avg_buy_food:.1f}, match_rate={match_rate:.2%}, avg_loss={avg_loss:.4f}; "
        f"optimizer_steps={optimizer_steps}, warmup_batch={warmup_batch_episodes}; "
        f"student(success_rate={student_success_rate:.2%}, avg_steps={student_avg_steps:.1f}, "
        f"avg_mine={student_avg_mine:.2f}, avg_buy_w={student_avg_buy_water:.1f}, "
        f"avg_buy_f={student_avg_buy_food:.1f})"
    )

    return {
        "episodes": float(episodes),
        "solved": float(solved),
        "success_rate": float(success_rate),
        "avg_steps": float(avg_steps),
        "avg_mine_per_episode": float(avg_mine),
        "avg_buy_water": float(avg_buy_water),
        "avg_buy_food": float(avg_buy_food),
        "avg_value_loss": float(avg_value_loss),
        "final_value_loss": float(final_value_loss),
        "value_loss_trend": float(value_loss_trend),
        "match_rate": float(match_rate),
        "success_rate_given_solved": float(success_rate_given_solved),
        "student_success_rate": float(student_success_rate),
        "student_avg_steps": float(student_avg_steps),
        "student_avg_mine_per_episode": float(student_avg_mine),
        "student_avg_buy_water": float(student_avg_buy_water),
        "student_avg_buy_food": float(student_avg_buy_food),
        "optimizer_steps": float(optimizer_steps),
        "warmup_batch_episodes": float(warmup_batch_episodes),
    }


def pretrain_behavior_cloning(
    level: int = 3,
    warmup_episodes: int = 200,
    device: Optional[str] = None,
    weather_mode: Optional[str] = None,
    oracle_time_limit: int = 20,
    output_path: Optional[str] = None,
    save_interval: int = 0,
    resume: Optional[str] = None,
    oracle_dataset_path: Optional[str] = None,
    oracle_parallel_workers: int = 1,
    oracle_solver_threads: Optional[int] = None,
    warmup_batch_episodes: int = 1,
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
            resume_agent = Agent.load(str(resume_path), device=device)
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
        f"warmup_episodes={warmup_episodes}, device={device}, "
        f"warmup_batch_episodes={warmup_batch_episodes}"
    )
    if resume_path is not None:
        print(f"resume_from={resume_path}, start_episode={resume_start_episode}")
    if oracle_dataset_path:
        print(f"oracle_dataset={oracle_dataset_path}")
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
        summary = behavior_cloning_with_oracle(
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
            oracle_dataset_path=oracle_dataset_path,
            oracle_parallel_workers=oracle_parallel_workers,
            oracle_solver_threads=oracle_solver_threads,
            warmup_batch_episodes=warmup_batch_episodes,
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
    report_checks = report.get("checks", [])
    print("=" * 60)
    print("预训练指标检测（4项阈值）")
    if isinstance(report_checks, list):
        for item in report_checks:
            if isinstance(item, dict):
                print(
                    f"[{item.get('status', 'UNKNOWN')}] {item.get('name', '-')}: {item.get('note', '-')}"
                )

    if bool(report.get("redline", False)):
        print("[REDLINE] avg_steps≈4 且 solved=100% 但无挖矿、价值损失恶化、匹配率过低。")

    if bool(report.get("overall_pass", False)):
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
    parser.add_argument(
        "--oracle-dataset-path",
        type=str,
        default=None,
        help="可复用BC数据集文件路径（命中则优先使用，未命中回退在线Oracle）",
    )
    parser.add_argument(
        "--oracle-parallel-workers",
        type=int,
        default=1,
        help="在线Oracle并行求解worker数（<=1表示串行）",
    )
    parser.add_argument(
        "--oracle-solver-threads",
        type=int,
        default=None,
        help="单个Oracle求解器线程数",
    )
    parser.add_argument(
        "--warmup-batch-episodes",
        type=int,
        default=1,
        help="预热阶段按多少个episode累计一次反向传播（>1可提升GPU利用率）",
    )
    args = parser.parse_args()

    pretrain_behavior_cloning(
        level=args.level,
        warmup_episodes=args.warmup_episodes,
        device=args.device,
        weather_mode=args.weather_mode,
        oracle_time_limit=args.oracle_time_limit,
        output_path=args.output,
        save_interval=args.save_interval,
        resume=args.resume,
        oracle_dataset_path=args.oracle_dataset_path,
        oracle_parallel_workers=args.oracle_parallel_workers,
        oracle_solver_threads=args.oracle_solver_threads,
        warmup_batch_episodes=args.warmup_batch_episodes,
    )


def _sanitize_oracle_action(action: Dict[str, Any], env, valid_actions: Dict[str, Any]) -> Dict:
    move = int(action.get("move", env.state.position))
    valid_moves = valid_actions.get("valid_moves", []) if valid_actions else []
    if valid_moves and move not in valid_moves:
        move = int(env.state.position)

    can_mine = bool(valid_actions.get("can_mine", False)) if valid_actions else False
    raw_mine_intensity = action.get("mine_intensity", None)
    if raw_mine_intensity is None:
        raw_mine_intensity = 1.0 if bool(action.get("mine", False)) else 0.0
    mine_intensity = float(np.clip(raw_mine_intensity, 0.0, 1.0)) if can_mine else 0.0
    mine = mine_intensity > 0.0

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
        "mine_intensity": mine_intensity,
        "buy_water": buy_water,
        "buy_food": buy_food,
    }


def _rollout_student_policy(
    agent: Agent,
    level: int,
    weather_mode: str,
    seed: int,
) -> Dict[str, float]:
    eval_env = make_env(level=level, weather_mode=weather_mode, seed=None)
    obs, info = eval_env.reset(seed=seed)

    steps = 0
    mine_days = 0
    buy_water = 0
    buy_food = 0
    done = False
    truncated = False

    while not done and not truncated and steps < 200:
        valid_actions = eval_env.get_valid_actions()
        action, _ = agent.select_action(obs, valid_actions, deterministic=True)
        obs, _, done, truncated, info = eval_env.step(action)
        last_action = info.get("last_action")
        if isinstance(last_action, dict):
            if bool(last_action.get("mine", False)):
                mine_days += 1
            buy_water += int(last_action.get("buy_water", 0))
            buy_food += int(last_action.get("buy_food", 0))
        steps += 1

    return {
        "steps": float(steps),
        "mine_days": float(mine_days),
        "buy_water": float(buy_water),
        "buy_food": float(buy_food),
        "success": float(bool(info.get("reached", False))),
    }


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


def _load_pretrain_agent_from_checkpoint(path: Path, device: str) -> Agent:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format")

    if all(k in checkpoint for k in ("state_dict", "obs_dim", "num_locations", "config")):
        agent = Agent(
            obs_dim=int(checkpoint["obs_dim"]),
            num_locations=int(checkpoint["num_locations"]),
            config=checkpoint["config"],
        )
        agent.load_state_dict(checkpoint["state_dict"])
        agent.to(device)
        return agent

    if all(k in checkpoint for k in ("agent_state_dict", "obs_dim", "num_locations", "config")):
        agent = Agent(
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


def _save_pretrain_latest_checkpoint(path: Path, agent: Agent, episode: int) -> None:
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
