import math

from src.pipeline.level4pipeline import (
    ExitCriteria,
    PipelineConfig,
    StageSpec,
    _build_edges,
    _compute_epsilon,
    _merge_warmup_summaries,
    _node3_to_mine_choice,
    _pass_criteria,
    _should_relax_stage2_warmup,
    _stage2_warmup_filter,
    _stage_episode_epsilon,
    _stage_metrics,
    _stage_override,
    _warn_level35_trap,
    build_stage_specs,
)


def test_build_stage_specs_fallback_start_node():
    cfg = PipelineConfig(stage1_start_node_one_based=99)
    stages = build_stage_specs(cfg)
    assert len(stages) == 3
    assert stages[0].start_node_one_based == 18


def test_build_edges_removes_target_edges():
    edges = _build_edges([(3, 4), (18, 19)])
    assert (3, 4) not in edges
    assert (18, 19) not in edges


def test_stage_override_applies_core_fields():
    stage = StageSpec(
        stage_id=1,
        name="mine_survival",
        init_money=3000,
        num_days=30,
        start_node_one_based=18,
        villages_one_based=[],
        removed_edges_one_based=[(3, 4)],
        weather_mode="balanced",
        warmup_episodes=10,
        warmup_oracle_interval=2,
        max_rl_episodes=10,
        min_rl_episodes=5,
        eval_interval=5,
        eval_window=5,
        exit_criteria=ExitCriteria(),
        fallback_note="fallback",
    )

    override = _stage_override(stage)
    assert override["INIT_MONEY"] == 3000
    assert override["START"] == 17
    assert override["MINES"] == [17]
    assert (3, 4) not in override["EDGES"]


def test_compute_epsilon_schedule():
    cfg = PipelineConfig(epsilon_start=0.3, epsilon_end=0.05, epsilon_decay_ratio=0.5)
    max_episodes = 100
    assert _compute_epsilon(0, max_episodes, cfg) == 0.3
    assert _compute_epsilon(60, max_episodes, cfg) == 0.05
    mid = _compute_epsilon(25, max_episodes, cfg)
    assert 0.05 < mid < 0.3


def test_stage_episode_epsilon_reset_for_new_stage():
    cfg = PipelineConfig(epsilon_start=0.3, epsilon_end=0.05, epsilon_decay_ratio=0.5)
    assert _stage_episode_epsilon(0, 2, 100, cfg) == 0.3
    assert _stage_episode_epsilon(1, 2, 100, cfg) < 0.3


def test_node3_to_mine_choice_cases():
    assert _node3_to_mine_choice([0, 2, 7, 8]) == 1.0
    assert _node3_to_mine_choice([0, 2, 3, 8]) == 0.0
    assert _node3_to_mine_choice([0, 1, 4]) is None


def test_stage_metrics_and_criteria():
    records = [
        {
            "success": 1.0,
            "steps": 12.0,
            "mine_days": 2.0,
            "net_profit": 100.0,
            "reached_mine": 1.0,
            "node3_choice_to_mine": 1.0,
        },
        {
            "success": 1.0,
            "steps": 14.0,
            "mine_days": 3.0,
            "net_profit": 50.0,
            "reached_mine": 1.0,
            "node3_choice_to_mine": float("nan"),
        },
    ]
    metrics = _stage_metrics(records)
    assert metrics["success_rate"] == 1.0
    assert metrics["avg_steps"] == 13.0
    assert metrics["avg_mine_days"] == 2.5
    assert math.isclose(metrics["node3_choice_prob"], 1.0)

    stage = StageSpec(
        stage_id=3,
        name="global_optimization",
        init_money=10000,
        num_days=30,
        start_node_one_based=1,
        villages_one_based=[7],
        removed_edges_one_based=[],
        weather_mode="balanced",
        warmup_episodes=0,
        warmup_oracle_interval=1,
        max_rl_episodes=10,
        min_rl_episodes=5,
        eval_interval=5,
        eval_window=5,
        exit_criteria=ExitCriteria(
            success_rate_min=0.75,
            avg_net_profit_min=0.0,
            avg_mine_days_range=(1.5, 3.0),
        ),
        fallback_note="",
    )
    assert _pass_criteria(stage, metrics)


def test_warn_level35_trap():
    records = [{"steps": 6.0, "mine_days": 0.0} for _ in range(50)]
    assert _warn_level35_trap(records, shortest_len=5)

    records_ok = [{"steps": 10.0, "mine_days": 1.0} for _ in range(50)]
    assert not _warn_level35_trap(records_ok, shortest_len=5)


def test_stage2_warmup_filter():
    assert _stage2_warmup_filter({"path_history": [0, 5, 17, 12]})
    assert not _stage2_warmup_filter({"path_history": [0, 1, 2]})
    assert not _stage2_warmup_filter({"path_history": "invalid"})


def test_should_relax_stage2_warmup():
    stage2 = StageSpec(
        stage_id=2,
        name="path_to_mine",
        init_money=5000,
        num_days=30,
        start_node_one_based=1,
        villages_one_based=[],
        removed_edges_one_based=[],
        weather_mode="balanced",
        warmup_episodes=10,
        warmup_oracle_interval=1,
        max_rl_episodes=10,
        min_rl_episodes=5,
        eval_interval=5,
        eval_window=5,
        exit_criteria=ExitCriteria(),
        fallback_note="",
    )
    assert _should_relax_stage2_warmup(stage2, {"accepted_episodes": 10.0}, 50)
    assert not _should_relax_stage2_warmup(stage2, {"accepted_episodes": 60.0}, 50)


def test_merge_warmup_summaries():
    primary = {
        "episodes": 10.0,
        "accepted_episodes": 4.0,
        "filtered_episodes": 6.0,
        "solved": 10.0,
        "avg_steps": 8.0,
        "avg_mine_per_episode": 2.0,
        "avg_buy_water": 10.0,
        "avg_buy_food": 12.0,
        "avg_value_loss": 20.0,
        "match_rate": 0.5,
        "success_rate": 0.6,
        "final_value_loss": 12.0,
        "value_loss_trend": -1.0,
    }
    extra = {
        "episodes": 20.0,
        "accepted_episodes": 10.0,
        "filtered_episodes": 10.0,
        "solved": 20.0,
        "avg_steps": 10.0,
        "avg_mine_per_episode": 3.0,
        "avg_buy_water": 20.0,
        "avg_buy_food": 22.0,
        "avg_value_loss": 10.0,
        "match_rate": 0.8,
        "success_rate": 0.9,
        "final_value_loss": 5.0,
        "value_loss_trend": -0.5,
    }

    merged = _merge_warmup_summaries(primary, extra)
    assert merged["episodes"] == 30.0
    assert merged["accepted_episodes"] == 14.0
    assert merged["filtered_episodes"] == 16.0
    assert merged["final_value_loss"] == 5.0
    assert merged["success_rate"] > 0.6
