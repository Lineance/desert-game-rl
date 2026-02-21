from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.env.config import Level3Config, Level4Config, Level35Config


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: int
    weather_mode: str
    start_episode: int
    end_episode: int


@dataclass
class CurriculumState:
    enabled: bool
    stage_modes: List[str]
    current_stage_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "stage_modes": list(self.stage_modes),
            "current_stage_id": int(self.current_stage_id),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["CurriculumState"]:
        if not isinstance(payload, dict):
            return None
        stage_modes = payload.get("stage_modes", [])
        if not isinstance(stage_modes, list):
            stage_modes = []
        parsed_modes = [str(mode) for mode in stage_modes if str(mode).strip()]
        return cls(
            enabled=bool(payload.get("enabled", False)),
            stage_modes=parsed_modes,
            current_stage_id=max(0, int(payload.get("current_stage_id", 0))),
        )


def get_level_weather_modes(level: int) -> Mapping[str, Sequence[float]]:
    if level == 3:
        return Level3Config.WEATHER_MODES
    if level == 35:
        return Level35Config.WEATHER_MODES
    if level == 4:
        return Level4Config.WEATHER_MODES
    raise ValueError(f"Unknown level: {level}")


def resolve_curriculum_modes(
    level: int,
    requested_modes: Optional[str],
) -> List[str]:
    available_modes = get_level_weather_modes(level)

    if requested_modes is not None and requested_modes.strip():
        candidate_modes = [mode.strip() for mode in requested_modes.split(",") if mode.strip()]
        valid_modes = [mode for mode in candidate_modes if mode in available_modes]
        if valid_modes:
            return valid_modes

    if level == 35:
        default_order = ["train_easy", "train_medium", "eval"]
    elif level == 3:
        default_order = ["no_sandstorm", "sunny_bias", "hot_bias"]
    elif level == 4:
        default_order = ["balanced", "unpredictable"]
    else:
        default_order = list(available_modes.keys())

    selected = [mode for mode in default_order if mode in available_modes]
    if selected:
        return selected

    return list(available_modes.keys())


def build_episode_stages(modes: Sequence[str], num_episodes: int) -> List[CurriculumStage]:
    if num_episodes <= 0:
        return []
    cleaned_modes = [str(mode).strip() for mode in modes if str(mode).strip()]
    if not cleaned_modes:
        raise ValueError("Curriculum requires at least one weather mode")

    stage_count = len(cleaned_modes)
    base_len = num_episodes // stage_count
    remainder = num_episodes % stage_count

    stages: List[CurriculumStage] = []
    cursor = 0
    for idx, mode in enumerate(cleaned_modes):
        length = base_len + (1 if idx < remainder else 0)
        if length <= 0:
            continue
        start = cursor
        end = cursor + length - 1
        stages.append(
            CurriculumStage(
                stage_id=idx,
                weather_mode=mode,
                start_episode=start,
                end_episode=end,
            )
        )
        cursor = end + 1

    if not stages:
        first_mode = cleaned_modes[0]
        stages.append(CurriculumStage(0, first_mode, 0, num_episodes - 1))

    return stages


def stage_for_episode(stages: Sequence[CurriculumStage], episode: int) -> CurriculumStage:
    if not stages:
        raise ValueError("No curriculum stage available")
    if episode <= stages[0].start_episode:
        return stages[0]
    for stage in stages:
        if stage.start_episode <= episode <= stage.end_episode:
            return stage
    return stages[-1]


def stage_progress(stage: CurriculumStage, episode: int) -> float:
    total = max(1, stage.end_episode - stage.start_episode + 1)
    current = min(max(episode, stage.start_episode), stage.end_episode)
    offset = current - stage.start_episode
    return float(offset + 1) / float(total)


def next_stage_start_episode(stages: Sequence[CurriculumStage], stage_id: int) -> Optional[int]:
    for stage in stages:
        if stage.stage_id == stage_id + 1:
            return stage.start_episode
    return None


def build_curriculum_state(enabled: bool, modes: Sequence[str], stage_id: int) -> CurriculumState:
    return CurriculumState(
        enabled=enabled, stage_modes=[str(mode) for mode in modes], current_stage_id=stage_id
    )
