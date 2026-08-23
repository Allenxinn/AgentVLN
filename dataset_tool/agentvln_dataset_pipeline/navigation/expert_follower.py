from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.dataset_contract import build_geodesic_path, strict_geodesic_distance


@dataclass(frozen=True)
class ExpertDecision:
    action: Optional[int]
    should_stop: bool
    termination_reason: Optional[str]
    goal_geodesic_distance: float
    remaining_geodesic_path: List[np.ndarray]


class ExpertTrajectoryFollower:
    def __init__(
        self,
        sim,
        waypoints,
        goal_position,
        waypoint_radius = 0.5,
        goal_radius = 1.0,
        sample_interval = 0.25,
    ):
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        self.sim = sim
        self.waypoints = [np.asarray(point, dtype=float) for point in waypoints]
        self.goal_position = np.asarray(goal_position, dtype=float)
        self.waypoint_radius = float(waypoint_radius)
        self.goal_radius = float(goal_radius)
        self.sample_interval = float(sample_interval)
        self.current_waypoint_idx = 0
        self._follower = ShortestPathFollower(
            sim, goal_radius=self.waypoint_radius, return_one_hot=False
        )

    def _position(self) -> np.ndarray:
        return np.asarray(self.sim.get_agent_state().position, dtype=float)

    def _goal_distance(self) -> float:
        return strict_geodesic_distance(
            self.sim, self._position(), self.goal_position
        )

    def _advance_reached_waypoints(self) -> None:
        position = self._position()
        while self.current_waypoint_idx < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_idx]
            distance = strict_geodesic_distance(self.sim, position, waypoint)
            if not np.isfinite(distance) or distance > self.waypoint_radius:
                break
            self.current_waypoint_idx += 1

    def _remaining_targets(self) -> List[np.ndarray]:
        targets = list(self.waypoints[self.current_waypoint_idx :])
        if not targets or np.linalg.norm(targets[-1] - self.goal_position) > 1e-4:
            targets.append(self.goal_position)
        return targets

    def _build_remaining_path(self) -> Optional[List[np.ndarray]]:
        return build_geodesic_path(
            self.sim,
            self._position(),
            self._remaining_targets(),
            self.sample_interval,
        )

    def decide(self) -> ExpertDecision:
        goal_distance = self._goal_distance()
        if not np.isfinite(goal_distance):
            return ExpertDecision(
                None, False, "follower_failure", goal_distance, []
            )
        if goal_distance <= self.goal_radius:
            path = self._build_remaining_path() or [self._position()]
            return ExpertDecision(None, True, "reached_goal", goal_distance, path)

        self._advance_reached_waypoints()
        path = self._build_remaining_path()
        if not path:
            return ExpertDecision(
                None, False, "follower_failure", goal_distance, []
            )
        while True:
            target = (
                self.waypoints[self.current_waypoint_idx]
                if self.current_waypoint_idx < len(self.waypoints)
                else self.goal_position
            )
            action = self._follower.get_next_action(target)
            if action is not None and int(action) != 0:
                return ExpertDecision(
                    int(action), False, None, goal_distance, path
                )
            if self.current_waypoint_idx >= len(self.waypoints):
                return ExpertDecision(
                    None, False, "follower_failure", goal_distance, path
                )
            self.current_waypoint_idx += 1
            path = self._build_remaining_path()
            if not path:
                return ExpertDecision(
                    None, False, "follower_failure", goal_distance, []
                )
