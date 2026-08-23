from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PreparedSample:
    step: int
    exploration_targets: list
    trajectory_target: Optional[dict]
    action_code: int
    floor_id: Optional[int]
    floor_height: Optional[float]
    floor_transition: bool
    expert_candidate_index: Optional[int]
    trajectory_goal_distance: Optional[float]
    trajectory_is_endpoint: bool
    rgb: Any
    topdown: Any


@dataclass
class EpisodeGenerationResult:
    task_data: Optional[Dict[str, Any]]
    termination_reason: str
    candidate_stats: Dict[str, int]
    floor_switches: int = 0
    transition_frames: int = 0
