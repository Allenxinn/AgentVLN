"""Per-floor top-down map state and floor transition detection."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from habitat.utils.visualizations import maps

from .topdown_map import TopDownMapBuilder


@dataclass
class FloorState:
    floor_id: int
    anchor_height: float
    builder: TopDownMapBuilder


@dataclass
class FloorUpdate:
    floor_id: Optional[int]
    floor_height: Optional[float]
    in_transition: bool
    switched: bool = False
    created: bool = False


class FloorMapManager:
    def __init__(self, sim, map_config, initial_height = None):
        self.sim = sim
        self.map_config = map_config
        cfg = map_config.get("floor_management", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.floor_match_tolerance = float(cfg.get("floor_match_tolerance", 0.5))
        self.floor_departure_threshold = float(
            cfg.get("floor_departure_threshold", 0.75)
        )
        self.min_floor_separation = float(cfg.get("min_floor_separation", 1.5))
        self.stable_window_steps = max(2, int(cfg.get("stable_window_steps", 4)))
        self.stable_height_tolerance = float(
            cfg.get("stable_height_tolerance", 0.10)
        )
        self.min_component_area_m2 = float(
            cfg.get("min_navigable_component_area_m2", 4.0)
        )

        height = float(
            sim.get_agent_state().position[1]
            if initial_height is None
            else initial_height
        )
        initial = FloorState(
            floor_id=0,
            anchor_height=height,
            builder=TopDownMapBuilder(sim, map_config, map_height=height),
        )
        self.floors: Dict[int, FloorState] = {0: initial}
        self.active_floor_id = 0
        self.in_transition = False
        self._transition_builder: Optional[TopDownMapBuilder] = None
        self._height_window = deque(maxlen=self.stable_window_steps)

    @property
    def active_state(self) -> FloorState:
        return self.floors[self.active_floor_id]

    @property
    def active_builder(self) -> TopDownMapBuilder:
        if self.in_transition and self._transition_builder is not None:
            return self._transition_builder
        return self.active_state.builder

    @property
    def full_map(self):
        return self.active_builder.full_map

    def _matching_floor(self, height) -> Optional[int]:
        matches = [
            (abs(state.anchor_height - height), floor_id)
            for floor_id, state in self.floors.items()
            if abs(state.anchor_height - height) <= self.floor_match_tolerance
        ]
        return min(matches)[1] if matches else None

    def _builder_contains_position(self, builder, position) -> bool:
        if (
            abs(float(position[1]) - float(builder.map_height))
            > self.floor_departure_threshold
        ):
            return False
        row, col = maps.to_grid(
            position[2],
            position[0],
            builder.full_map.shape[:2],
            sim=self.sim,
        )
        row, col = int(row), int(col)
        return (
            0 <= row < builder.full_map.shape[0]
            and 0 <= col < builder.full_map.shape[1]
            and builder.full_map[row, col] > 0
        )

    def _ensure_transition_builder(self, height, agent_position) -> None:
        if (
            self._transition_builder is not None
            and self._builder_contains_position(
                self._transition_builder, agent_position
            )
        ):
            return
        self._transition_builder = TopDownMapBuilder(
            self.sim, self.map_config, map_height=float(height)
        )

    def _build_valid_floor(
        self, height, agent_position
    ) -> Optional[TopDownMapBuilder]:
        candidate_builder = TopDownMapBuilder(
            self.sim, self.map_config, map_height=float(height)
        )
        candidate_map = candidate_builder.full_map
        row, col = maps.to_grid(
            agent_position[2],
            agent_position[0],
            candidate_map.shape[:2],
            sim=self.sim,
        )
        row, col = int(row), int(col)
        if not (0 <= row < candidate_map.shape[0] and 0 <= col < candidate_map.shape[1]):
            return None
        if candidate_map[row, col] == 0:
            return None

        count, labels = cv2.connectedComponents((candidate_map > 0).astype(np.uint8))
        if count <= 1:
            return None
        component = labels[row, col]
        if component == 0:
            return None
        area_pixels = int(np.count_nonzero(labels == component))
        meters_per_pixel = maps.calculate_meters_per_pixel(
            int(self.map_config.get("resolution", 512)), sim=self.sim
        )
        if (
            area_pixels * meters_per_pixel * meters_per_pixel
            < self.min_component_area_m2
        ):
            return None
        return candidate_builder

    def update(self, agent_state) -> FloorUpdate:
        height = float(agent_state.position[1])
        previous_floor = self.active_floor_id

        if not self.enabled:
            self._transition_builder = None
            return FloorUpdate(
                previous_floor, self.active_state.anchor_height, False
            )

        matching = self._matching_floor(height)
        if matching is not None:
            self.active_floor_id = matching
            self.in_transition = False
            self._transition_builder = None
            self._height_window.clear()
            state = self.active_state
            return FloorUpdate(
                state.floor_id,
                state.anchor_height,
                False,
                switched=matching != previous_floor,
            )

        active_height = self.active_state.anchor_height
        if abs(height - active_height) <= self.floor_departure_threshold:
            self.in_transition = False
            self._transition_builder = None
            self._height_window.clear()
            return FloorUpdate(previous_floor, active_height, False)

        self.in_transition = True
        self._ensure_transition_builder(height, agent_state.position)
        self._height_window.append(height)
        if len(self._height_window) < self.stable_window_steps:
            return FloorUpdate(None, None, True)
        if max(self._height_window) - min(self._height_window) > self.stable_height_tolerance:
            return FloorUpdate(None, None, True)

        stable_height = float(np.median(np.asarray(self._height_window)))
        nearest_separation = min(
            abs(stable_height - state.anchor_height) for state in self.floors.values()
        )
        if nearest_separation < self.min_floor_separation:
            return FloorUpdate(None, None, True)
        candidate_builder = self._build_valid_floor(
            stable_height, agent_state.position
        )
        if candidate_builder is None:
            return FloorUpdate(None, None, True)

        floor_id = max(self.floors) + 1
        self.floors[floor_id] = FloorState(
            floor_id=floor_id,
            anchor_height=stable_height,
            builder=candidate_builder,
        )
        self.active_floor_id = floor_id
        self.in_transition = False
        self._transition_builder = None
        self._height_window.clear()
        return FloorUpdate(
            floor_id, stable_height, False, switched=True, created=True
        )

    def update_visibility(self, agent_state, fov = 110.0):
        return self.active_builder.update_visibility(agent_state, fov=fov)

    def get_floor_metadata(self) -> List[dict]:
        return [
            {"floor_id": state.floor_id, "height": state.anchor_height}
            for _, state in sorted(self.floors.items())
        ]
