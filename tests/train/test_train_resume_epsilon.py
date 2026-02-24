import src.pipeline.train as train


def test_resume_epsilon_anchor_and_decay_continuation():
    eps_start = 0.30
    eps_end = 0.05
    eps_decay_episodes = 70
    start_episode = 40
    saved_epsilon = 0.19

    first = train._compute_resume_epsilon(
        start_episode,
        start_episode,
        saved_epsilon,
        eps_start,
        eps_end,
        eps_decay_episodes,
    )
    later = train._compute_resume_epsilon(
        start_episode + 10,
        start_episode,
        saved_epsilon,
        eps_start,
        eps_end,
        eps_decay_episodes,
    )
    tail = train._compute_resume_epsilon(
        eps_decay_episodes + 20,
        start_episode,
        saved_epsilon,
        eps_start,
        eps_end,
        eps_decay_episodes,
    )

    assert first == saved_epsilon
    assert eps_end <= later < saved_epsilon
    assert abs(tail - eps_end) < 1e-9


def test_resume_epsilon_without_saved_value_falls_back_to_original_schedule():
    eps_start = 0.30
    eps_end = 0.05
    eps_decay_episodes = 70

    expected = train._compute_epsilon(12, eps_start, eps_end, eps_decay_episodes)
    actual = train._compute_resume_epsilon(
        12,
        start_episode=10,
        resume_epsilon=None,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_episodes=eps_decay_episodes,
    )

    assert abs(actual - expected) < 1e-12
