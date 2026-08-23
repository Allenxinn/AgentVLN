import numpy as np
import cv2
from habitat.utils.visualizations import fog_of_war, maps


INVALID_BACKGROUND_COLOR = (255, 255, 255)
OBSTACLE_COLOR = (255, 255, 255)
UNEXPLORED_FREE_COLOR = (60, 60, 60)
EXPLORED_FREE_COLOR = (127, 127, 127)


def convert_meters_to_pixel(meters, map_resolution, sim) -> int:
    return int(meters / maps.calculate_meters_per_pixel(map_resolution, sim=sim))


def convert_square_meters_to_pixel_area(
    area_m2, map_resolution, sim
) -> float:
    from ..core.dataset_contract import square_meters_to_pixel_area

    meters_per_pixel = maps.calculate_meters_per_pixel(map_resolution, sim=sim)
    return square_meters_to_pixel_area(area_m2, meters_per_pixel)


def wrap_heading(heading):
    return (heading + np.pi) % (2 * np.pi) - np.pi


class TopDownMapBuilder:
    
    def __init__(self, sim, config, map_height = None):
        self.sim = sim
        self.map_resolution = config.get('resolution', 512)
        self.visible_radius = config.get('visible_radius', 10.0)
        floor_config = config.get('floor_management', {})
        self.height_tolerance = max(
            0.0, float(floor_config.get('floor_match_tolerance', 0.5))
        )
        self.map_height = (
            float(map_height)
            if map_height is not None
            else float(sim.get_agent_state().position[1])
        )

        self.sampled_heights = self._height_samples()
        self.full_map = self._build_floor_map()
        self.floor_footprint = self._build_floor_footprint()
        
        self.fog_of_war_mask = np.zeros_like(self.full_map, dtype=np.uint8)
        
        self.visibility_dist_pixels = convert_meters_to_pixel(
            self.visible_radius, self.map_resolution, sim
        )
        self._use_habitat_fog = callable(
            getattr(fog_of_war, 'reveal_fog_of_war', None)
        )

    def _height_samples(self) -> list:
        if self.height_tolerance <= 0.0:
            return [self.map_height]
        return [
            self.map_height - self.height_tolerance,
            self.map_height,
            self.map_height + self.height_tolerance,
        ]

    def _build_floor_map(self) -> np.ndarray:
        floor_map = None
        for height in self.sampled_heights:
            height_map = maps.get_topdown_map(
                self.sim.pathfinder,
                float(height),
                map_resolution=self.map_resolution,
                draw_border=False,
            )
            height_map = (np.asarray(height_map) > 0).astype(np.uint8)
            if floor_map is None:
                floor_map = height_map
            elif height_map.shape != floor_map.shape:
                raise ValueError(
                    "Top-down navmesh slices have inconsistent shapes: "
                    f"{floor_map.shape} vs {height_map.shape}"
                )
            else:
                floor_map = cv2.bitwise_or(floor_map, height_map)

        if floor_map is None or floor_map.size == 0:
            raise ValueError(
                f"Unable to build top-down map at height {self.map_height:.3f}"
            )
        return floor_map

    def _build_floor_footprint(self) -> np.ndarray:
        contours, _ = cv2.findContours(
            self.full_map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        footprint = np.zeros_like(self.full_map, dtype=np.uint8)
        if contours:
            cv2.drawContours(footprint, contours, -1, 1, thickness=-1)
        return footprint
    
    def get_agent_map_position(self, agent_position) -> np.ndarray:
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self.full_map.shape[0], self.full_map.shape[1]),
            sim=self.sim,
        )
        return np.array([a_x, a_y])
    
    def get_polar_angle(self, agent_rotation) -> float:
        from habitat.tasks.utils import cartesian_to_polar
        from habitat.utils.geometry_utils import quaternion_rotate_vector
        
        heading_vector = quaternion_rotate_vector(
            agent_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return float(phi + np.pi)

    def _reveal_visibility_with_rays(
        self,
        current_point,
        current_angle,
        fov,
    ) -> np.ndarray:
        row, col = (int(current_point[0]), int(current_point[1]))
        center_angle = -float(current_angle) + np.pi / 2.0
        half_fov = np.deg2rad(float(fov)) / 2.0
        ray_count = max(2, int(np.ceil(float(fov))) + 1)
        ray_angles = np.linspace(
            center_angle - half_fov,
            center_angle + half_fov,
            ray_count,
        )
        distances = np.arange(
            int(self.visibility_dist_pixels) + 1, dtype=np.float64
        )
        endpoints = [(col, row)]

        for angle in ray_angles:
            cols = np.rint(col + distances * np.cos(angle)).astype(np.int32)
            rows = np.rint(row + distances * np.sin(angle)).astype(np.int32)
            in_bounds = (
                (rows >= 0)
                & (rows < self.full_map.shape[0])
                & (cols >= 0)
                & (cols < self.full_map.shape[1])
            )
            rows, cols = rows[in_bounds], cols[in_bounds]
            if rows.size == 0:
                continue

            keep = np.ones(rows.shape[0], dtype=bool)
            if rows.shape[0] > 1:
                keep[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
            rows, cols = rows[keep], cols[keep]

            blocked = np.flatnonzero(self.full_map[rows, cols] == 0)
            stop = int(blocked[0]) if blocked.size else rows.shape[0]
            last_visible = max(0, stop - 1)
            endpoints.append((int(cols[last_visible]), int(rows[last_visible])))

        visible = np.zeros_like(self.full_map, dtype=np.uint8)
        hull = cv2.convexHull(np.asarray(endpoints, dtype=np.int32))
        cv2.fillConvexPoly(visible, hull, 1)
        visible = cv2.bitwise_and(visible, self.full_map)
        return cv2.bitwise_or(self.fog_of_war_mask, visible)

    def _reveal_visibility(
        self,
        current_point,
        current_angle,
        fov,
    ) -> np.ndarray:
        if self._use_habitat_fog:
            try:
                topdown = np.ascontiguousarray(self.full_map, dtype=np.uint8)
                previous = np.ascontiguousarray(
                    self.fog_of_war_mask, dtype=np.uint8
                )
                revealed = fog_of_war.reveal_fog_of_war(
                    topdown,
                    previous,
                    np.asarray(current_point, dtype=np.int64),
                    float(current_angle),
                    fov=float(fov),
                    max_line_len=float(self.visibility_dist_pixels),
                )
                revealed = np.asarray(revealed, dtype=np.uint8)
                if revealed.shape != self.full_map.shape:
                    raise ValueError(
                        "Habitat fog-of-war returned shape "
                        f"{revealed.shape}, expected {self.full_map.shape}"
                    )
                return np.ascontiguousarray(revealed)
            except Exception as exc:
                self._use_habitat_fog = False
                print(
                    "[TopDownMap Warning] Habitat reveal_fog_of_war failed "
                    f"({type(exc).__name__}: {exc}); using local ray casting."
                )

        return self._reveal_visibility_with_rays(
            current_point, current_angle, fov
        )
    
    def _get_fov_cone_mask(
        self, 
        current_point, 
        current_angle, 
        fov
    ) -> np.ndarray:
        curr_pt_cv2 = current_point[::-1].astype(int)
        angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2))
        
        cone_mask = cv2.ellipse(
            np.zeros_like(self.full_map),
            tuple(curr_pt_cv2),
            (int(self.visibility_dist_pixels), int(self.visibility_dist_pixels)),
            0,
            angle_cv2 - fov / 2,
            angle_cv2 + fov / 2,
            1,
            -1,
        )
        return cone_mask
    
    def update_visibility(
        self, 
        agent_state, 
        fov = 110.0
    ) -> np.ndarray:
        current_point = self.get_agent_map_position(agent_state.position)
        current_angle = self.get_polar_angle(agent_state.rotation)
        
        row, col = current_point.astype(int)
        if not (
            0 <= row < self.full_map.shape[0]
            and 0 <= col < self.full_map.shape[1]
        ):
            raise ValueError(
                f"Agent maps outside top-down bounds at ({row}, {col})"
            )
        if self.full_map[row, col] == 0:
            print(
                "[TopDownMap Warning] Agent is outside the navigable floor "
                f"slice at height {self.map_height:.3f}: ({row}, {col})"
            )
            return self.fog_of_war_mask

        self.fog_of_war_mask = self._reveal_visibility(
            current_point,
            current_angle,
            fov,
        )
        return self.fog_of_war_mask
    
    def get_explored_map(self) -> np.ndarray:
        return cv2.bitwise_and(self.full_map, self.fog_of_war_mask)
    
    def get_visible_obstacles_map(self, agent_state, fov = 110.0) -> np.ndarray:
        current_point = self.get_agent_map_position(agent_state.position)
        current_angle = self.get_polar_angle(agent_state.rotation)
        
        cone_mask = self._get_fov_cone_mask(current_point, current_angle, fov)
        
        obstacles = (1 - self.full_map).astype(np.uint8)
        
        visible_obstacles = cv2.bitwise_and(obstacles, cone_mask)
        
        return visible_obstacles
    
    def build_topdown_frame(
        self, 
        agent_state, 
        fov = 110.0,
        show_fov = True
    ) -> np.ndarray:
        vis_map = np.full(
            (*self.full_map.shape, 3), INVALID_BACKGROUND_COLOR, dtype=np.uint8
        )

        obstacles = (self.floor_footprint > 0) & (self.full_map == 0)
        vis_map[obstacles] = OBSTACLE_COLOR

        vis_map[self.full_map > 0] = UNEXPLORED_FREE_COLOR
        
        explored = self.get_explored_map()
        vis_map[explored > 0] = EXPLORED_FREE_COLOR
        
        if show_fov:
            current_point = self.get_agent_map_position(agent_state.position)
            current_angle = self.get_polar_angle(agent_state.rotation)
            
            overlay = vis_map.copy()
            curr_pt_cv2 = tuple(current_point[::-1].astype(int))
            angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2))

            fov_mask = cv2.bitwise_and(
                self._get_fov_cone_mask(current_point, current_angle, fov),
                self.floor_footprint,
            )
            overlay[fov_mask > 0] = (255, 255, 255)
            vis_map = cv2.addWeighted(overlay, 0.3, vis_map, 0.7, 0)
            
            cv2.circle(vis_map, curr_pt_cv2, 8, (0, 0, 255), -1)
            
            arrow_len = 15
            end_pt = (
                int(curr_pt_cv2[0] + arrow_len * np.cos(np.deg2rad(angle_cv2))),
                int(curr_pt_cv2[1] + arrow_len * np.sin(np.deg2rad(angle_cv2)))
            )
            cv2.arrowedLine(vis_map, curr_pt_cv2, end_pt, (0, 0, 255), 2)
        
        return vis_map
    
    def reset(self):
        self.fog_of_war_mask = np.zeros_like(self.full_map, dtype=np.uint8)
