"""
评估与结果导出
将训练好的策略应用于第三关、第四关，导出Result.xlsx格式结果
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.env.config import RESULTS_DIR, Weather
from src.env.environment import make_env
from src.models.agent import HybridRNNAgent


def run_episode(
    agent: HybridRNNAgent,
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


def export_to_csv(result: Dict, filepath: str, level: int):
    """
    导出结果为CSV格式（类似Result.xlsx）

    格式: [日期, 区域, 剩余资金, 剩余水量, 剩余食物量, 操作]
    """
    import csv

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

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
            action_str = format_action(record["action"], record["position"])
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


def export_to_xlsx(result: Dict, filepath: str, level: int):
    """
    导出结果为Excel格式
    """
    try:
        from openpyxl import Workbook
    except ImportError:
        print("警告: 未安装openpyxl，使用CSV格式导出")
        export_to_csv(result, filepath.replace(".xlsx", ".csv"), level)
        return

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "第三关" if level == 3 else "第四关"

    # 表头
    headers = ["日期", "区域", "剩余资金(元)", "剩余水量(箱)", "剩余食物量(箱)", "操作"]
    ws.append(headers)

    # 数据
    for record in result["records"]:
        action_str = format_action(record["action"], record["position"])
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


def format_action(action: Dict, position: int) -> str:
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
        parts.append("挖矿")

    # 购买
    buy_water = action.get("buy_water", 0)
    buy_food = action.get("buy_food", 0)
    if buy_water > 0 or buy_food > 0:
        parts.append(f"购买(水{buy_water}食{buy_food})")

    return "+".join(parts)


def analyze_strategy(result: Dict, env):
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


def evaluate_multiple_runs(agent_path: str, level: int, num_runs: int = 100, device: str = None):
    """
    多次运行评估，统计性能
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载智能体
    agent = HybridRNNAgent.load(agent_path, device=device)
    agent.eval()

    # 创建环境
    env = make_env(level=level, seed=None)

    print(f"\n多轮评估: {num_runs} 次")
    print(f"模型: {agent_path}")
    print(f"关卡: {level}")

    results = []

    for i in range(num_runs):
        result = run_episode(agent, env, seed=i, deterministic=True)
        results.append(result)

        if (i + 1) % 20 == 0:
            print(f"  已完成 {i + 1}/{num_runs}")

    # 统计
    reached = [r["reached"] for r in results]
    final_moneys = [r["final_money"] for r in results]
    lengths = [r["length"] for r in results]
    returns = [r["return"] for r in results]

    print("\n统计结果:")
    print(f"  到达率: {np.mean(reached) * 100:.1f}% ({sum(reached)}/{num_runs})")
    print(f"  平均最终资金: {np.mean(final_moneys):.2f} ± {np.std(final_moneys):.2f}")
    print(f"  平均回合长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  平均回报: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

    # 找出最佳和最差
    best_idx = np.argmax(final_moneys)
    worst_idx = np.argmin(final_moneys)

    print(f"\n最佳表现 (Run {best_idx}):")
    print(f"  最终资金: {results[best_idx]['final_money']:.2f}")
    print(f"  路径: {' -> '.join(map(str, results[best_idx]['path'][:20]))}...")

    return results


def generate_result_excel(
    agent_path: str, level: int, output_dir: str = str(RESULTS_DIR), device: str = None
):
    """
    生成最终结果文件（选择最佳表现的运行）
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载智能体
    agent = HybridRNNAgent.load(agent_path, device=device)
    agent.eval()

    # 创建环境
    env = make_env(level=level, seed=None)

    # 运行多次，选择最佳
    print("\n寻找最佳策略...")
    best_result = None
    best_money = float("-inf")

    for seed in range(50):
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

    # 导出
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if level == 3:
        filepath = output_dir / "Result_第三关.xlsx"
    else:
        filepath = output_dir / "Result_第四关.xlsx"

    export_to_xlsx(best_result, str(filepath), level)
    analyze_strategy(best_result, env)

    return best_result


def main():
    parser = argparse.ArgumentParser(description="评估沙漠穿越智能体")
    parser.add_argument("agent_path", type=str, help="智能体模型路径")
    parser.add_argument("--level", type=int, default=3, choices=[3, 4], help="关卡（3或4）")
    parser.add_argument(
        "--runs", type=int, default=1, help="运行次数（1表示单轮详细输出，>1表示统计）"
    )
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR), help="输出目录")
    parser.add_argument("--device", type=str, default=None, help="计算设备（cuda/cpu）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（单轮模式）")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    if args.runs == 1:
        # 单轮详细评估
        if args.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device

        agent = HybridRNNAgent.load(args.agent_path, device=device)
        agent.eval()
        env = make_env(level=args.level, seed=None)

        result = run_episode(agent, env, seed=args.seed, deterministic=True, verbose=args.verbose)

        analyze_strategy(result, env)

        # 导出结果
        export_to_xlsx(result, f"{args.output}/level{args.level}_result.xlsx", args.level)

    else:
        # 多轮统计
        evaluate_multiple_runs(args.agent_path, args.level, args.runs, args.device)


if __name__ == "__main__":
    main()
