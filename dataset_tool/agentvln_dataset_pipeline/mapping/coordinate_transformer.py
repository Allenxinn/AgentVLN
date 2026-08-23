import numpy as np
import magnum as mn
from typing import Tuple, Optional, Dict, List, Any
from enum import IntEnum
from habitat.utils.visualizations import maps


class VisibilityStatus(IntEnum):
    VISIBLE = 0
    BEHIND_CAMERA = 1
    LEFT_OF_VIEW = 2
    RIGHT_OF_VIEW = 3
    ABOVE_VIEW = 4
    BELOW_VIEW = 5


class CoordinateTransformer:
    def __init__(self, camera_config, sim):
        self.sim = sim
        self.width = camera_config.get('width', 720)
        self.height = camera_config.get('height', 640)
        self.hfov = camera_config.get('hfov', 110)
        self.camera_height = camera_config.get('camera_height', 1.25)
        
        # Compute camera intrinsics
        self.hfov_rad = np.deg2rad(self.hfov)
        self.vfov_rad = 2 * np.arctan(np.tan(self.hfov_rad / 2) * self.height / self.width)
        
        self.fx = self.width / (2 * np.tan(self.hfov_rad / 2))
        self.fy = self.height / (2 * np.tan(self.vfov_rad / 2))
        # Principal point: for 0-indexed pixels, center is at (width-1)/2, (height-1)/2
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0
        
        # Historical visibility tracking: {point_key: [(step_id, pixel_coords), ...]}
        self.visibility_history: Dict[str, List[Tuple[int, Tuple[int, int]]]] = {}
    
    def _point_to_key(self, world_point) -> str:
        rounded = np.round(world_point, 2)
        return f"{rounded[0]:.2f},{rounded[1]:.2f},{rounded[2]:.2f}"
    
    def world_to_map(
        self, 
        world_point, 
        top_down_map
    ) -> np.ndarray:
        a_x, a_y = maps.to_grid(
            world_point[2],
            world_point[0],
            (top_down_map.shape[0], top_down_map.shape[1]),
            sim=self.sim,
        )
        return np.array([a_x, a_y])
    
    def map_to_world(
        self, 
        map_pixel, 
        agent_position,
        top_down_map,
        point_height = None
    ) -> np.ndarray:
        x, y = map_pixel[0], map_pixel[1]
        realworld_x, realworld_y = maps.from_grid(
            x, y,
            (top_down_map.shape[0], top_down_map.shape[1]),
            self.sim,
        )
        
        # Snap to navmesh to get valid point
        target_point = [realworld_y, agent_position[1], realworld_x]
        world_point = self.sim.pathfinder.snap_point(
            mn.Vector3(target_point)
        )
        world_point = np.array(world_point)
        
        # Override height if specified
        if point_height is not None:
            world_point[1] = point_height
        
        return world_point
    
    def world_to_camera(
        self, 
        world_point, 
        agent_state
    ) -> np.ndarray:
        from habitat.utils.geometry_utils import quaternion_rotate_vector
        
        # Get relative position
        relative_pos = world_point - agent_state.position
        
        # Rotate to agent's local frame
        local_pos = quaternion_rotate_vector(
            agent_state.rotation.inverse(), 
            relative_pos
        )
        
        # Convert to camera frame (Habitat uses -z as forward)
        # Camera: x-right, y-down, z-forward
        camera_pos = np.array([
            local_pos[0],   # x (right)
            -local_pos[1],  # y (down) 
            -local_pos[2]   # z (forward)
        ])
        
        return camera_pos
    
    def camera_to_pixel(
        self, 
        camera_point
    ) -> Tuple[np.ndarray, VisibilityStatus]:
        x, y, z = camera_point
        
        # Check if behind camera
        if z <= 0:
            return np.array([-1, -1]), VisibilityStatus.BEHIND_CAMERA
        
        # Project to image plane
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        
        pixel = np.array([int(u), int(v)])
        
        # Determine visibility status
        if 0 <= u < self.width and 0 <= v < self.height:
            return pixel, VisibilityStatus.VISIBLE
        elif u < 0:
            return pixel, VisibilityStatus.LEFT_OF_VIEW
        elif u >= self.width:
            return pixel, VisibilityStatus.RIGHT_OF_VIEW
        elif v < 0:
            return pixel, VisibilityStatus.ABOVE_VIEW
        else:
            return pixel, VisibilityStatus.BELOW_VIEW
    
    def world_to_pixel(
        self, 
        world_point, 
        agent_state,
        current_step,
        point_height = None  # None means use floor level (agent's y position)
    ) -> Tuple[np.ndarray, VisibilityStatus, Optional[Dict[str, Any]]]:
        project_point = np.array(world_point, dtype=float)
        
        if point_height is None:
            # Default: use agent's floor level (y-coordinate)
            project_point[1] = agent_state.position[1] - self.camera_height # offset by camera height
        else:
            project_point[1] = point_height
        
        camera_point = self.world_to_camera(project_point, agent_state)
        pixel, status = self.camera_to_pixel(camera_point)
        
        point_key = self._point_to_key(world_point)
        
        if status == VisibilityStatus.VISIBLE:
            if point_key not in self.visibility_history:
                self.visibility_history[point_key] = []
            self.visibility_history[point_key].append((current_step, tuple(pixel)))
            return pixel, status, None
        else:
            history_info = self.get_historical_visibility(world_point)
            return pixel, status, history_info
    
    def update_visibility_history(
        self, 
        world_point, 
        step_id, 
        pixel_coords
    ):
        point_key = self._point_to_key(world_point)
        if point_key not in self.visibility_history:
            self.visibility_history[point_key] = []
        self.visibility_history[point_key].append((step_id, pixel_coords))
    
    def get_historical_visibility(
        self, 
        world_point
    ) -> Optional[Dict[str, Any]]:
        point_key = self._point_to_key(world_point)
        
        if point_key in self.visibility_history and self.visibility_history[point_key]:
            # Return most recent visibility
            step_id, pixel = self.visibility_history[point_key][-1]
            return {"step": step_id, "pixel": list(pixel)}
        
        return None
    
    def is_point_visible(
        self, 
        world_point, 
        agent_state
    ) -> bool:
        camera_point = self.world_to_camera(world_point, agent_state)
        _, status = self.camera_to_pixel(camera_point)
        return status == VisibilityStatus.VISIBLE
    
    def project_to_image_boundary(
        self, 
        camera_point
    ) -> Tuple[np.ndarray, VisibilityStatus]:
        x, y, z = camera_point
        
        if z <= 0:
            return np.array([-1, -1]), VisibilityStatus.BEHIND_CAMERA
        
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        
        # Clamp to image boundaries
        u_clamped = np.clip(u, 0, self.width - 1)
        v_clamped = np.clip(v, 0, self.height - 1)
        
        # Determine original status
        if u < 0:
            status = VisibilityStatus.LEFT_OF_VIEW
        elif u >= self.width:
            status = VisibilityStatus.RIGHT_OF_VIEW
        elif v < 0:
            status = VisibilityStatus.ABOVE_VIEW
        elif v >= self.height:
            status = VisibilityStatus.BELOW_VIEW
        else:
            status = VisibilityStatus.VISIBLE
        
        return np.array([int(u_clamped), int(v_clamped)]), status
    
    def reset_history(self):
        self.visibility_history.clear()
    
    @staticmethod
    def status_to_string(status) -> str:
        return {
            VisibilityStatus.VISIBLE: "VISIBLE",
            VisibilityStatus.BEHIND_CAMERA: "BEHIND",
            VisibilityStatus.LEFT_OF_VIEW: "LEFT",
            VisibilityStatus.RIGHT_OF_VIEW: "RIGHT",
            VisibilityStatus.ABOVE_VIEW: "ABOVE",
            VisibilityStatus.BELOW_VIEW: "BELOW",
        }.get(status, "UNKNOWN")
