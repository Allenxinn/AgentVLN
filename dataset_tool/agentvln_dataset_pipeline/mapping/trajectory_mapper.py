"""
Trajectory mapper module.
Maps navigation trajectory points to RGB pixel coordinates with distance thresholds.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .coordinate_transformer import CoordinateTransformer, VisibilityStatus
from .candidates import is_depth_visible
from habitat.utils.visualizations import maps


class TrajectoryMapper:
    def __init__(self, config, coord_transformer):
        self.max_distance = config.get('max_distance', 5.0)
        self.min_distance = config.get('min_distance', 1.0)
        self.sample_interval = config.get('sample_interval', 0.25)
        self.occlusion_tolerance = config.get('occlusion_tolerance', 0.2)
        self.depth_max = config.get('depth_max', 10.0)
        self.floor_match_tolerance = config.get('floor_match_tolerance', 0.5)
        self.coord_transformer = coord_transformer
        
        self.meters_per_pixel = None
    
    def set_map_scale(self, meters_per_pixel):
        self.meters_per_pixel = meters_per_pixel
    
    def _compute_distance_to_agent(
        self, 
        point, 
        agent_position
    ) -> float:
        return np.sqrt(
            (point[0] - agent_position[0])**2 + 
            (point[2] - agent_position[2])**2
        )
    
    def map_trajectory_to_topdown(
        self, 
        trajectory_points, 
        top_down_map
    ) -> List[np.ndarray]:
        map_coords = []
        for point in trajectory_points:
            map_pixel = self.coord_transformer.world_to_map(point, top_down_map)
            map_coords.append(map_pixel)
        return map_coords
    
    def get_farthest_visible_point(
        self,
        path_points,
        agent_state,
        top_down_map,
        current_step,
        depth_image = None,
        occlusion_tolerance = 0.0,
        sample_interval = None,
        floor_height = None,
        floor_match_tolerance = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], VisibilityStatus, Optional[Dict[str, Any]]]:
        if not path_points or depth_image is None:
            return None, None, VisibilityStatus.BEHIND_CAMERA, None

        agent_pos = np.asarray(agent_state.position, dtype=float)
        path_points = [np.asarray(point, dtype=float) for point in path_points]
        tolerance = (
            self.floor_match_tolerance
            if floor_match_tolerance is None
            else floor_match_tolerance
        )
        active_height = float(agent_pos[1] if floor_height is None else floor_height)

        candidates = []
        cumulative = 0.0
        previous = agent_pos
        for point in path_points:
            point = np.asarray(point, dtype=float)
            cumulative += float(np.linalg.norm(point - previous))
            previous = point
            if cumulative < self.min_distance or cumulative > self.max_distance:
                continue
            if abs(float(point[1]) - active_height) > tolerance:
                continue
            candidates.append((point, cumulative))

        for point, dist in reversed(candidates):
            px, st, history = self.coord_transformer.world_to_pixel(
                point, agent_state, current_step
            )
            if st != VisibilityStatus.VISIBLE:
                continue
            if not is_depth_visible(
                point,
                px,
                agent_state,
                self.coord_transformer,
                depth_image,
                max_depth=self.depth_max,
                tolerance=(
                    self.occlusion_tolerance
                    if occlusion_tolerance == 0.0
                    else occlusion_tolerance
                ),
            ):
                continue
            return px, point, VisibilityStatus.VISIBLE, None

        return None, None, VisibilityStatus.BEHIND_CAMERA, None
    
    def _check_occlusion(
        self,
        world_point,
        agent_state,
        pixel_coords,
        depth_image,
        tolerance = 0.1,
        max_depth = 10,
    ) -> bool:
        return not is_depth_visible(
            world_point,
            pixel_coords,
            agent_state,
            self.coord_transformer,
            depth_image,
            max_depth=max_depth,
            tolerance=tolerance,
        )
    
    def get_trajectory_target(
        self,
        trajectory_points,
        agent_state,
        top_down_map,
        current_step,
        depth_image = None,
        floor_height = None,
        floor_match_tolerance = None,
        in_transition = False,
    ) -> Dict[str, Any]:
        if in_transition:
            return {
                'pixel_coords': None,
                'world_coords': None,
                'visibility_status': None,
                'history_info': None,
            }
        pixel, world_pt, status, history = self.get_farthest_visible_point(
            trajectory_points,
            agent_state,
            top_down_map,
            current_step,
            depth_image,
            floor_height=floor_height,
            floor_match_tolerance=floor_match_tolerance,
        )
        
        result = {
            'pixel_coords': pixel.tolist() if pixel is not None else None,
            'world_coords': world_pt.tolist() if world_pt is not None else None,
            'visibility_status': int(status),
            'history_info': history
        }
        
        return result
