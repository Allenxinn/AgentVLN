from typing import List, Union, Optional, Tuple

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import DistanceToGoal, Success


def euclidean_distance(
    pos_a, pos_b
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


@registry.register_measure
class PathLength(Measure):
    cls_uuid: str = "path_length"

    def __init__(self, sim, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args, **kwargs):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position


@registry.register_measure
class OracleNavigationError(Measure):
    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args, task, **kwargs):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    cls_uuid: str = "oracle_success"

    def __init__(self, *args, config, **kwargs):
        print(f"in oracle success init: args = {args}, kwargs = {kwargs}")
        self._config = config
        super().__init__()

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args, task, **kwargs):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        # self._metric = float(self._metric or d < self._config["success_distance"])
        self._metric = float(self._metric or d < 3.0)


@registry.register_measure
class OracleSPL(Measure):
    cls_uuid: str = "oracle_spl"

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(self.uuid, ["spl"])
        self._metric = 0.0

    def update_metric(self, *args, task, **kwargs):
        spl = task.measurements.measures["spl"].get_metric()
        self._metric = max(self._metric, spl)

@registry.register_measure
class PL(Measure):
    def __init__(
        self, sim, config, *args, **kwargs
    ):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args, **kwargs) -> str:
        return "pl"

    def reset_metric(self, episode, task, *args, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task, *args, **kwargs
    ):
        # ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

@registry.register_measure
class StepsTaken(Measure):
    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._metric = 0.0

    def update_metric(self, *args, **kwargs):
        self._metric += 1.0

