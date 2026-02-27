"""单回合执行与结果导出工具。"""

import argparse
from pathlib import Path
from typing import Dict

import torch

from src.env.config import RESULTS_DIR, Weather
from src.env.environment import make_env
from src.models.agent import Agent


def run_episode(
    agent: Agent,
    env,
    seed: int = None,
    deterministic: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    运行一个完整的回合

    Args:
        agent: 智能体
        env: 环境
        seed: 随机种子
        deterministic: 是否使用确定性策略
        verbose: 是否输出详细日志

    Returns:
        episode_data: 包含完整路径和决策的记录
    """
    obs, info = env.reset(seed=seed)
    agent.reset_hidden(device=next(agent.parameters()).device)

    # 记录数据（按题目注2：记录当日行动执行后的状态）
    records = []
    episode_return = 0
    done = False
    step = 0

    while not done and step < 200:
        # 选择动作
        valid_actions = env.get_valid_actions()
        with torch.no_grad():
            action, value = agent.select_action(obs, valid_actions, deterministic=deterministic)

        # 执行动作
        next_obs, reward, done, truncated, next_info = env.step(action)
        done = done or truncated

        executed_action = next_info.get("last_action") or action
        record = {
            "day": int(next_info.get("action_day", next_info["day"])),
            "position": next_info["position"] + 1,  # 转回1-based
            "money": next_info["money"],
            "water": next_info["water"],
            "food": next_info["food"],
            "weather": Weather.NAMES[next_info["weather_today"]],
            "belief": next_info["belief"].tolist()
            if hasattr(next_info["belief"], "tolist")
            else next_info["belief"],
            "action": executed_action,
            "policy_action": action,
            "value": value,
            "reward": reward,
        }
        records.append(record)

        episode_return += reward
        obs = next_obs
        info = next_info
        step += 1

        if verbose:
            print(
                f"Day {record['day']}: Pos={record['position']}, "
                f"Weather={record['weather']}, Action={action}, "
                f"Money={record['money']:.0f}"
            )

    # 最终结果
    result = {
        "records": records,
        "return": episode_return,
        "reached": info.get("reached", False),
        "final_money": info.get("money", 0),
        "final_water": info.get("water", 0),
        "final_food": info.get("food", 0),
        "length": step,
        "path": [r["position"] for r in records],
    }

    return result


def rollout_main():
    parser = argparse.ArgumentParser(description="沙漠穿越智能执行")
    parser.add_argument("agent_path", type=str, help="智能体模型路径")
    parser.add_argument("--level", type=int, default=3, choices=[3, 4, 35], help="关卡")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR), help="输出目录")
    parser.add_argument("--device", type=str, default=None, help="计算设备(cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--majority-voting", type=bool, default=False, help="多次求解取最大值")
    parser.add_argument("--episodes", type=int, default=50, help="运行次数")

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.majoriry_voting:
        result = rollout_majority_voting(args.agent_path, args.level, args.episodes, device)
    else:
        agent = Agent.load(args.agent_path, device=device)
        agent.eval()
        env = make_env(level=args.level, seed=None)
        result = run_episode(agent, env, seed=args.seed, deterministic=True, verbose=args.verbose)

    _analyze_strategy(result, env)

    _export_to_xlsx(result, f"{args.output}/level{args.level}_result.xlsx", args.level)


def rollout_majority_voting(
    agent_path: str,
    level: int,
    episodes: int = 50,
    device: str = "cpu",
):
    # 加载智能体
    agent = Agent.load(agent_path, device=device)
    agent.eval()

    # 创建环境
    env = make_env(level=level, seed=None)

    # 运行多次，选择最佳
    print("\n寻找最佳策略...")
    best_result = None
    best_money = float("-inf")

    for seed in range(episodes):
        result = run_episode(agent, env, seed=seed, deterministic=True)
        if result["reached"] and result["final_money"] > best_money:
            best_money = result["final_money"]
            best_result = result

    if best_result is None:
        print("警告: 未找到成功到达终点的策略")
        # 选择回报最高的
        best_result = max(
            [run_episode(agent, env, seed=i, deterministic=True) for i in range(20)],
            key=lambda x: x["return"],
        )

    return best_result


def _export_to_csv(result: Dict, filepath: str, level: int):
    """
    导出结果为CSV格式（类似Result.xlsx）

    格式: [日期, 区域, 剩余资金, 剩余水量, 剩余食物量, 操作]
    """
    import csv

    filepath_path = Path(filepath)
    filepath_path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 表头
        writer.writerow(
            [
                "日期",
                "区域",
                "剩余资金(元)",
                "剩余水量(箱)",
                "剩余食物量(箱)",
                "天气",
                "操作",
            ]
        )

        # 数据
        for record in result["records"]:
            action_str = _format_action(record["action"], record["position"])
            writer.writerow(
                [
                    record["day"],
                    record["position"],
                    round(record["money"], 2),
                    record["water"],
                    record["food"],
                    record["weather"],
                    action_str,
                ]
            )

    print(f"结果已导出: {filepath}")


def _export_to_xlsx(result: Dict, filepath: str, level: int):
    """
    导出结果为Excel格式
    """
    try:
        from openpyxl import Workbook
    except ImportError:
        print("警告: 未安装openpyxl，使用CSV格式导出")
        _export_to_csv(result, filepath.replace(".xlsx", ".csv"), level)
        return

    filepath_path = Path(filepath)
    filepath_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "第三关" if level == 3 else "第四关"

    # 表头
    headers = ["日期", "区域", "剩余资金(元)", "剩余水量(箱)", "剩余食物量(箱)", "操作"]
    ws.append(headers)

    # 数据
    for record in result["records"]:
        action_str = _format_action(record["action"], record["position"])
        ws.append(
            [
                record["day"],
                record["position"],
                round(record["money"], 2),
                record["water"],
                record["food"],
                action_str,
            ]
        )

    # 汇总信息
    ws2 = wb.create_sheet("汇总")
    ws2.append(["项目", "值"])
    ws2.append(["最终资金", result["final_money"]])
    ws2.append(["最终水量", result["final_water"]])
    ws2.append(["最终食物", result["final_food"]])
    ws2.append(["到达终点", "是" if result["reached"] else "否"])
    ws2.append(["回合长度", result["length"]])

    wb.save(filepath)
    print(f"结果已导出: {filepath}")


def _format_action(action: Dict, position: int) -> str:
    """格式化动作为字符串"""
    parts = []

    move_from = action.get("move_from")
    move_to = action.get("move_to")
    if move_from is not None and move_to is not None:
        if move_from == move_to:
            parts.append("停留")
        else:
            parts.append(f"移动 {move_from + 1}->{move_to + 1}")
    elif "move" in action:
        move_target = action.get("move", position - 1)
        if move_target == position - 1:
            parts.append("停留")
        else:
            parts.append(f"移动 {position}->{move_target + 1}")
    else:
        parts.append("停留")

    # 挖矿
    if action.get("mine", False):
        if "mine_intensity" in action:
            intensity = float(action.get("mine_intensity", 1.0))
            parts.append(f"挖矿(强度{intensity:.2f})")
        else:
            parts.append("挖矿")

    # 购买
    buy_water = action.get("buy_water", 0)
    buy_food = action.get("buy_food", 0)
    if buy_water > 0 or buy_food > 0:
        parts.append(f"购买(水{buy_water}食{buy_food})")

    return "+".join(parts)


def _analyze_strategy(result: Dict, env):
    """分析策略特征"""
    records = result["records"]

    print("\n策略分析:")
    print(f"  总天数: {len(records)}")
    print(f"  是否到达终点: {'是' if result['reached'] else '否'}")
    print(f"  最终资金: {result['final_money']:.2f}")

    # 统计挖矿天数
    mine_days = [r["day"] for r in records if r["action"].get("mine", False)]
    print(f"  挖矿天数: {len(mine_days)} 天")
    if mine_days:
        print(f"    挖矿日期: {mine_days}")

    # 统计购买
    buy_records = []
    for r in records:
        buy_water = r["action"].get("buy_water", 0)
        buy_food = r["action"].get("buy_food", 0)
        if buy_water > 0 or buy_food > 0:
            buy_records.append((r["day"], buy_water, buy_food))
    print(f"  购买次数: {len(buy_records)}")
    for day, w, f in buy_records:
        print(f"    第{day}天: 水{w}, 食物{f}")

    # 路径分析
    unique_positions = len(set(r["position"] for r in records))
    print(f"  经过区域数: {unique_positions}")

    # 天气应对
    weather_stats = {"晴朗": 0, "高温": 0, "沙暴": 0}
    for r in records:
        weather_stats[r["weather"]] += 1
    print(f"  遇到天气: {weather_stats}")


if __name__ == "__main__":
    rollout_main()
