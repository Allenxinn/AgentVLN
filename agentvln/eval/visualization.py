import numpy as np
import cv2
from typing import List, Tuple, Optional
from agentvln.eval.habitat_utils.coordinate_transformer import VisibilityStatus


# Color definitions (BGR format)
COLORS = {
    'target': (0, 255, 0),           # Green - exploration targets
    'target_invisible': (0, 165, 255),  # Orange - invisible targets
    'trajectory': (255, 0, 0),       # Blue - trajectory target
    'trajectory_path': (255, 200, 0), # Light blue - expert trajectory path
    'agent': (0, 0, 255),            # Red - agent position
    'start': (0, 255, 0),            # Green - start point
    'end': (0, 0, 255),              # Red - end point  
    'path': (255, 255, 0),           # Cyan - traveled path
    'frontier': None,                 # Rainbow colors per frontier
    'fov': (255, 255, 255),          # White - FOV area
}


class Visualizer: 
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.target_radius = self.config.get('target_radius', 8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 2
        
        self.traveled_path = []
    
    def reset_path(self):
        self.traveled_path = []
    
    def add_path_point(self, map_pos: Tuple[int, int]):
        self.traveled_path.append(map_pos)
    
    def draw_targets_on_rgb(
        self,
        rgb_image: np.ndarray,
        pixel_coords: List[Tuple[int, int]],
        visibility_status: List[int] = None,
        labels: List[str] = None
    ) -> np.ndarray:
        vis_img = rgb_image.copy()
        
        if visibility_status is None:
            visibility_status = [VisibilityStatus.VISIBLE] * len(pixel_coords)
        
        for i, (coord, status) in enumerate(zip(pixel_coords, visibility_status)):
            if coord is None:
                continue
            
            u, v = int(coord[0]), int(coord[1])
            
            if status == VisibilityStatus.VISIBLE:
                color = COLORS['target']
            else:
                color = COLORS['target_invisible']
            
            if 0 <= u < vis_img.shape[1] and 0 <= v < vis_img.shape[0]:
                cv2.circle(vis_img, (u, v), self.target_radius, color, -1)
                cv2.circle(vis_img, (u, v), self.target_radius, (0, 0, 0), 2)
                
                if labels and i < len(labels):
                    cv2.putText(
                        vis_img, labels[i], 
                        (u + 10, v - 10),
                        self.font, self.font_scale, color, self.font_thickness
                    )
            else:
                self._draw_direction_indicator(vis_img, status, i)
        
        return vis_img
    
    def _draw_direction_indicator(
        self,
        image: np.ndarray,
        status: int,
        index: int
    ):
        h, w = image.shape[:2]
        color = COLORS['target_invisible']
        arrow_size = 30
        margin = 20
        
        if status == VisibilityStatus.LEFT_OF_VIEW:
            start = (margin + arrow_size, h // 2 + index * 40)
            end = (margin, h // 2 + index * 40)
        elif status == VisibilityStatus.RIGHT_OF_VIEW:
            start = (w - margin - arrow_size, h // 2 + index * 40)
            end = (w - margin, h // 2 + index * 40)
        elif status == VisibilityStatus.ABOVE_VIEW:
            start = (w // 2 + index * 40, margin + arrow_size)
            end = (w // 2 + index * 40, margin)
        elif status == VisibilityStatus.BELOW_VIEW:
            start = (w // 2 + index * 40, h - margin - arrow_size)
            end = (w // 2 + index * 40, h - margin)
        else:
            return
        
        cv2.arrowedLine(image, start, end, color, 3, tipLength=0.4)
    
    def draw_targets_on_topdown(
        self,
        topdown_map: np.ndarray,
        map_coords: List[Tuple[int, int]],
        colors: List[Tuple[int, int, int]] = None,
        labels: List[str] = None
    ) -> np.ndarray:
        vis_map = topdown_map.copy() if len(topdown_map.shape) == 3 else \
                  cv2.cvtColor(topdown_map, cv2.COLOR_GRAY2BGR)
        
        for i, coord in enumerate(map_coords):
            if coord is None:
                continue
            
            row, col = int(coord[0]), int(coord[1])
            color = colors[i] if colors and i < len(colors) else COLORS['target']
            
            if 0 <= row < vis_map.shape[0] and 0 <= col < vis_map.shape[1]:
                cv2.circle(vis_map, (col, row), self.target_radius, color, -1)
                cv2.circle(vis_map, (col, row), self.target_radius, (0, 0, 0), 2)
                
                if labels and i < len(labels):
                    cv2.putText(
                        vis_map, labels[i],
                        (col + 10, row - 10),
                        self.font, self.font_scale, color, self.font_thickness
                    )
        
        return vis_map
    
    def draw_start_end_path(
        self,
        topdown_map: np.ndarray,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        trajectory_map_coords: List[Tuple[int, int]] = None,
        traveled_path: List[Tuple[int, int]] = None
    ) -> np.ndarray:
        vis_map = topdown_map.copy() if len(topdown_map.shape) == 3 else \
                  cv2.cvtColor(topdown_map, cv2.COLOR_GRAY2BGR)
        
        if trajectory_map_coords and len(trajectory_map_coords) > 1:
            for i in range(len(trajectory_map_coords) - 1):
                pt1 = trajectory_map_coords[i]
                pt2 = trajectory_map_coords[i + 1]
                if pt1 is not None and pt2 is not None:
                    cv2_pt1 = (int(pt1[1]), int(pt1[0]))
                    cv2_pt2 = (int(pt2[1]), int(pt2[0]))
                    self._draw_dashed_line(vis_map, cv2_pt1, cv2_pt2, COLORS['trajectory_path'], 2, gap=8)
            
            for pt in trajectory_map_coords:
                if pt is not None:
                    cv2.circle(vis_map, (int(pt[1]), int(pt[0])), 4, COLORS['trajectory_path'], -1)
        
        if traveled_path and len(traveled_path) > 1:
            for i in range(len(traveled_path) - 1):
                pt1 = traveled_path[i]
                pt2 = traveled_path[i + 1]
                if pt1 is not None and pt2 is not None:
                    cv2_pt1 = (int(pt1[1]), int(pt1[0]))
                    cv2_pt2 = (int(pt2[1]), int(pt2[0]))
                    cv2.line(vis_map, cv2_pt1, cv2_pt2, COLORS['path'], 3)
        
        if start_pos is not None:
            row, col = int(start_pos[0]), int(start_pos[1])
            if 0 <= row < vis_map.shape[0] and 0 <= col < vis_map.shape[1]:
                size = 10
                cv2.rectangle(vis_map, (col - size, row - size), (col + size, row + size), COLORS['start'], -1)
                cv2.rectangle(vis_map, (col - size, row - size), (col + size, row + size), (0, 0, 0), 2)
                cv2.putText(vis_map, "S", (col - 5, row + 5), self.font, 0.5, (0, 0, 0), 2)
        
        if end_pos is not None:
            row, col = int(end_pos[0]), int(end_pos[1])
            if 0 <= row < vis_map.shape[0] and 0 <= col < vis_map.shape[1]:
                size = 12
                pts = np.array([
                    [col, row - size],
                    [col + size, row],
                    [col, row + size],
                    [col - size, row]
                ], dtype=np.int32)
                cv2.fillPoly(vis_map, [pts], COLORS['end'])
                cv2.polylines(vis_map, [pts], True, (0, 0, 0), 2)
                cv2.putText(vis_map, "E", (col - 5, row + 5), self.font, 0.5, (255, 255, 255), 2)
        
        return vis_map
    
    def _draw_dashed_line(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        gap: int = 10
    ):
        dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
        if dist == 0:
            return
        
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int(pt1[0] * (1 - r) + pt2[0] * r)
            y = int(pt1[1] * (1 - r) + pt2[1] * r)
            pts.append((x, y))
        
        for i in range(0, len(pts) - 1, 2):
            if i + 1 < len(pts):
                cv2.line(img, pts[i], pts[i + 1], color, thickness)
    
    def draw_agent_on_topdown(
        self,
        topdown_map: np.ndarray,
        agent_map_pos: Tuple[int, int],
        agent_angle: float = None,
        fov: float = 110,
        visibility_dist: int = 50
    ) -> np.ndarray:
        vis_map = topdown_map.copy()
        row, col = int(agent_map_pos[0]), int(agent_map_pos[1])
        
        if agent_angle is not None:
            overlay = vis_map.copy()
            angle_deg = np.rad2deg(-agent_angle + np.pi / 2)
            
            cv2.ellipse(
                overlay, (col, row),
                (visibility_dist, visibility_dist),
                0, angle_deg - fov / 2, angle_deg + fov / 2,
                COLORS['fov'], -1
            )
            vis_map = cv2.addWeighted(overlay, 0.3, vis_map, 0.7, 0)
            
            arrow_len = 25
            end_col = int(col + arrow_len * np.cos(np.deg2rad(angle_deg)))
            end_row = int(row + arrow_len * np.sin(np.deg2rad(angle_deg)))
            cv2.arrowedLine(vis_map, (col, row), (end_col, end_row), COLORS['agent'], 2)
        
        # Draw agent position
        cv2.circle(vis_map, (col, row), 10, COLORS['agent'], -1)
        
        return vis_map
    
    def draw_frontiers(
        self,
        topdown_map: np.ndarray,
        frontiers: List[np.ndarray]
    ) -> np.ndarray:
        vis_map = topdown_map.copy() if len(topdown_map.shape) == 3 else \
                  cv2.cvtColor(topdown_map, cv2.COLOR_GRAY2BGR)
        
        for i, frontier in enumerate(frontiers):
            if len(frontiers) > 0:
                color = cv2.applyColorMap(
                    np.uint8([255 * (i + 1) / len(frontiers)]), 
                    cv2.COLORMAP_RAINBOW
                )[0][0]
                color = tuple(int(c) for c in color)
            else:
                color = (0, 255, 0)
            
            points = frontier.reshape(-1, 2)
            for j in range(len(points) - 1):
                pt1 = tuple(points[j].astype(int))
                pt2 = tuple(points[j + 1].astype(int))
                cv2.line(vis_map, pt1, pt2, color, 2)
        
        return vis_map
    
    def draw_trajectory_target(
        self,
        image: np.ndarray,
        pixel_coord: Tuple[int, int],
        visibility_status: int,
        is_rgb: bool = True
    ) -> np.ndarray:
        vis_img = image.copy()
        
        if pixel_coord is None:
            if not is_rgb:
                return vis_img
            self._draw_direction_indicator(vis_img, visibility_status, 0)
            return vis_img
        
        u, v = int(pixel_coord[0]), int(pixel_coord[1])
        color = COLORS['trajectory']
        
        if 0 <= u < vis_img.shape[1] and 0 <= v < vis_img.shape[0]:
            size = 12
            pts = np.array([
                [u, v - size],
                [u + size, v],
                [u, v + size],
                [u - size, v]
            ], dtype=np.int32)
            cv2.fillPoly(vis_img, [pts], color)
            cv2.polylines(vis_img, [pts], True, (0, 0, 0), 2)
        
        return vis_img
    
    def create_combined_visualization(
        self,
        rgb_vis: np.ndarray,
        topdown_vis: np.ndarray,
        target_height: int = None
    ) -> np.ndarray:
        if target_height is None:
            target_height = max(rgb_vis.shape[0], topdown_vis.shape[0])
        
        rgb_scale = target_height / rgb_vis.shape[0]
        rgb_resized = cv2.resize(
            rgb_vis, 
            (int(rgb_vis.shape[1] * rgb_scale), target_height)
        )
        
        topdown_scale = target_height / topdown_vis.shape[0]
        topdown_resized = cv2.resize(
            topdown_vis,
            (int(topdown_vis.shape[1] * topdown_scale), target_height)
        )
        
        combined = np.concatenate([rgb_resized, topdown_resized], axis=1)
        
        return combined
    
    def add_info_text(
        self,
        image: np.ndarray,
        text_lines: List[str],
        position: str = 'top_left'
    ) -> np.ndarray:
        vis_img = image.copy()
        line_height = 25
        margin = 10
        
        for i, line in enumerate(text_lines):
            if position.startswith('top'):
                y = margin + (i + 1) * line_height
            else:
                y = vis_img.shape[0] - margin - (len(text_lines) - i) * line_height
            
            if position.endswith('left'):
                x = margin
            else:
                text_size = cv2.getTextSize(line, self.font, self.font_scale, self.font_thickness)[0]
                x = vis_img.shape[1] - margin - text_size[0]
            
            text_size = cv2.getTextSize(line, self.font, self.font_scale, self.font_thickness)[0]
            cv2.rectangle(
                vis_img,
                (x - 5, y - text_size[1] - 5),
                (x + text_size[0] + 5, y + 5),
                (0, 0, 0), -1
            )
            
            cv2.putText(
                vis_img, line, (x, y),
                self.font, self.font_scale, (255, 255, 255), self.font_thickness
            )
        
        return vis_img
    
    def add_legend(
        self,
        image: np.ndarray,
        position: str = 'bottom_right'
    ) -> np.ndarray:
        vis_img = image.copy()
        
        legend_items = [
            ("Start", COLORS['start']),
            ("End", COLORS['end']),
            ("Expert Path", COLORS['trajectory_path']),
            ("Traveled", COLORS['path']),
            ("Targets", COLORS['target']),
            ("Nav Target", COLORS['trajectory']),
        ]
        
        margin = 10
        item_height = 20
        legend_width = 120
        legend_height = len(legend_items) * item_height + margin * 2
        
        if position == 'bottom_right':
            x_start = vis_img.shape[1] - legend_width - margin
            y_start = vis_img.shape[0] - legend_height - margin
        else:
            x_start = margin
            y_start = margin
        
        cv2.rectangle(
            vis_img,
            (x_start, y_start),
            (x_start + legend_width, y_start + legend_height),
            (0, 0, 0), -1
        )
        cv2.rectangle(
            vis_img,
            (x_start, y_start),
            (x_start + legend_width, y_start + legend_height),
            (255, 255, 255), 1
        )
        
        for i, (label, color) in enumerate(legend_items):
            y = y_start + margin + i * item_height + 15
            cv2.circle(vis_img, (x_start + 15, y - 5), 5, color, -1)
            cv2.putText(vis_img, label, (x_start + 30, y), self.font, 0.4, (255, 255, 255), 1)
        
        return vis_img
