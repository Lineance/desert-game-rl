from src.pipeline.curriculum import (
    CurriculumState,
    build_episode_stages,
    resolve_curriculum_modes,
    stage_for_episode,
    stage_progress,
)


def test_resolve_curriculum_modes_default_level35():
    modes = resolve_curriculum_modes(35, None)
    assert modes == ["train_easy", "train_medium", "eval"]


def test_resolve_curriculum_modes_with_invalid_entries_fallback_to_valid():
    modes = resolve_curriculum_modes(35, "bad_mode,train_easy,train_medium")
    assert modes == ["train_easy", "train_medium"]


def test_build_episode_stages_even_split_and_lookup():
    stages = build_episode_stages(["train_easy", "train_medium", "eval"], num_episodes=10)
    assert len(stages) == 3
    assert stages[0].start_episode == 0
    assert stages[0].end_episode == 3
    assert stages[1].start_episode == 4
    assert stages[1].end_episode == 6
    assert stages[2].start_episode == 7
    assert stages[2].end_episode == 9

    assert stage_for_episode(stages, 0).weather_mode == "train_easy"
    assert stage_for_episode(stages, 5).weather_mode == "train_medium"
    assert stage_for_episode(stages, 9).weather_mode == "eval"


def test_stage_progress_clamped_to_stage_bounds():
    stages = build_episode_stages(["balanced", "unpredictable"], num_episodes=4)
    first_stage = stages[0]
    last_stage = stages[-1]

    assert stage_progress(first_stage, -10) == 0.5
    assert stage_progress(first_stage, 0) == 0.5
    assert stage_progress(first_stage, 1) == 1.0
    assert stage_progress(last_stage, 10) == 1.0


def test_curriculum_state_roundtrip():
    state = CurriculumState(
        enabled=True, stage_modes=["train_easy", "train_medium"], current_stage_id=1
    )
    payload = state.to_dict()
    restored = CurriculumState.from_dict(payload)

    assert restored is not None
    assert restored.enabled is True
    assert restored.stage_modes == ["train_easy", "train_medium"]
    assert restored.current_stage_id == 1
