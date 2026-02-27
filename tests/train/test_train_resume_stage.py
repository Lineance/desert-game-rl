import src.pipeline.train as train


def test_resolve_stage_runtime_continue_uses_checkpoint_context():
    checkpoint = {
        "stage_context": {
            "stage_name": "stage3",
            "stage2_episodes_done": 42,
            "stage3_episodes_done": 7,
            "stage3_start_episode": 120,
        }
    }

    state = train._resolve_stage_runtime(True, "continue", checkpoint)

    assert state.stage_name == "stage3"
    assert state.stage2_episodes_done == 42
    assert state.stage3_episodes_done == 7
    assert state.stage3_start_episode == 120


def test_resolve_stage_runtime_recompute_resets_to_stage2():
    checkpoint = {
        "stage_context": {
            "stage_name": "stage3",
            "stage2_episodes_done": 42,
            "stage3_episodes_done": 7,
            "stage3_start_episode": 120,
        }
    }

    state = train._resolve_stage_runtime(True, "recompute", checkpoint)

    assert state.stage_name == "stage2"
    assert state.stage2_episodes_done == 0
    assert state.stage3_episodes_done == 0
    assert state.stage3_start_episode is None
