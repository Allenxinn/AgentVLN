import numpy as np
import cv2
from typing import List, Optional, Tuple
from numba import njit


@njit
def contour_to_frontiers(contour, unexplored_mask):
    bad_inds = []
    num_contour_points = len(contour)
    for idx in range(num_contour_points):
        x, y = contour[idx][0]
        if unexplored_mask[y, x] == 0:
            bad_inds.append(idx)
    frontiers = np.split(contour, bad_inds)
    
    filtered_frontiers = []
    front_last_split = (
        0 not in bad_inds
        and len(bad_inds) > 0
        and max(bad_inds) < num_contour_points - 2
    )
    for idx, f in enumerate(frontiers):
        if len(f) > 2 or (idx == 0 and front_last_split):
            if idx == 0:
                filtered_frontiers.append(f)
            else:
                filtered_frontiers.append(f[1:])
    
    if len(filtered_frontiers) > 1 and front_last_split:
        last_frontier = filtered_frontiers.pop()
        filtered_frontiers[0] = np.concatenate((last_frontier, filtered_frontiers[0]))
    
    return filtered_frontiers


class ExplorationTargetGenerator:
    def __init__(self, config):
        self.obstacle_distance_threshold = config.get('obstacle_distance_threshold', 0.5)
        self.min_target_spacing = config.get('min_target_spacing', 1.0)
        self.max_targets = config.get('max_targets', 5)
        
        # These will be set based on map resolution
        self.obstacle_dist_pixels = None
        self.min_spacing_pixels = None
    
    def set_pixel_scale(self, meters_per_pixel):
        self.obstacle_dist_pixels = int(self.obstacle_distance_threshold / meters_per_pixel)
        self.min_spacing_pixels = int(self.min_target_spacing / meters_per_pixel)
    
    def _interpolate_contour(self, contour) -> np.ndarray:
        if len(contour) < 2:
            return contour
        
        interpolated = [contour[0]]
        for i in range(len(contour) - 1):
            p1, p2 = contour[i][0], contour[i + 1][0]
            dist = np.linalg.norm(p2 - p1)
            if dist > 1:
                num_points = int(dist)
                for j in range(1, num_points):
                    t = j / num_points
                    new_point = (p1 * (1 - t) + p2 * t).astype(int)
                    interpolated.append(new_point.reshape(1, 2))
            interpolated.append(contour[i + 1])
        
        return np.array(interpolated).reshape(-1, 1, 2)
    
    def detect_frontiers(
        self, 
        explored_map, 
        full_map,
        area_thresh = -1.0
    ) -> List[np.ndarray]:
        if area_thresh > 0:
            unexplored = full_map.copy()
            unexplored[explored_map > 0] = 0
            
            contours, _ = cv2.findContours(
                unexplored, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Fill small unexplored areas
            for cnt in contours:
                if cv2.contourArea(cnt) < area_thresh:
                    cv2.drawContours(explored_map, [cnt], 0, 1, -1)
        
        # Find contours of explored area
        contours, _ = cv2.findContours(
            explored_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create unexplored mask with blur for tolerance
        unexplored_mask = np.where(explored_map > 0, 0, full_map)
        unexplored_mask = cv2.blur(
            np.where(unexplored_mask > 0, 255, unexplored_mask).astype(np.uint8), 
            (3, 3)
        )
        
        frontiers = []
        for contour in contours:
            interpolated = self._interpolate_contour(contour)
            frontier_segments = contour_to_frontiers(interpolated, unexplored_mask)
            frontiers.extend(frontier_segments)
        
        return frontiers
    
    def _get_frontier_midpoint(self, frontier) -> np.ndarray:
        if len(frontier) < 2:
            return frontier[0][0].astype(float)
        
        points = frontier.reshape(-1, 2)
        dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cum_dist = np.cumsum(dists)
        total_length = cum_dist[-1] if len(cum_dist) > 0 else 0
        
        if total_length == 0:
            return points[0].astype(float)
        
        half_length = total_length / 2
        idx = np.searchsorted(cum_dist, half_length)
        
        if idx == 0:
            return points[0].astype(float)
        
        prev_dist = cum_dist[idx - 1] if idx > 0 else 0
        segment_dist = cum_dist[idx] - prev_dist
        t = (half_length - prev_dist) / segment_dist if segment_dist > 0 else 0
        
        midpoint = points[idx] * t + points[idx + 1] * (1 - t) if idx < len(points) - 1 else points[idx]
        return midpoint.astype(float)
    
    def filter_by_obstacle_distance(
        self, 
        candidates, 
        obstacle_map
    ) -> np.ndarray:
        if len(candidates) == 0 or self.obstacle_dist_pixels is None:
            return candidates
        
        # Compute distance transform from obstacles
        dist_transform = cv2.distanceTransform(
            (1 - obstacle_map).astype(np.uint8), 
            cv2.DIST_L2, 
            5
        )
        
        # Filter candidates by distance
        filtered = []
        for point in candidates:
            x, y = int(point[0]), int(point[1])
            if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
                if dist_transform[y, x] >= self.obstacle_dist_pixels:
                    filtered.append(point)
        
        return np.array(filtered) if filtered else np.array([])
    
    def sample_targets_with_spacing(
        self, 
        candidates
    ) -> np.ndarray:
        if len(candidates) == 0:
            return np.array([])
        
        if len(candidates) <= self.max_targets:
            if self.min_spacing_pixels is None or self.min_spacing_pixels <= 0:
                return candidates
        
        selected = []
        remaining = list(range(len(candidates)))
        
        while len(selected) < self.max_targets and remaining:
            idx = remaining.pop(0)
            selected.append(candidates[idx])
            
            if self.min_spacing_pixels and self.min_spacing_pixels > 0:
                new_remaining = []
                for i in remaining:
                    dist = np.linalg.norm(candidates[i] - candidates[idx])
                    if dist >= self.min_spacing_pixels:
                        new_remaining.append(i)
                remaining = new_remaining
        
        return np.array(selected)

    def generate_candidate_metadata(
        self,
        explored_map,
        full_map,
        obstacle_map,
        area_thresh = -1.0,
    ) -> Tuple[List[dict], List[np.ndarray]]:
        frontiers = self.detect_frontiers(
            explored_map.copy(), full_map, area_thresh
        )
        if not frontiers:
            return [], []

        midpoints = np.asarray(
            [self._get_frontier_midpoint(frontier) for frontier in frontiers]
        )
        safe_midpoints = self.filter_by_obstacle_distance(midpoints, obstacle_map)
        if len(safe_midpoints) == 0:
            return [], frontiers

        metadata = []
        for point in safe_midpoints:
            distances = np.linalg.norm(midpoints - point, axis=1)
            frontier_idx = int(np.argmin(distances))
            frontier_points = frontiers[frontier_idx].reshape(-1, 2)
            if len(frontier_points) > 1:
                frontier_length = float(
                    np.linalg.norm(np.diff(frontier_points, axis=0), axis=1).sum()
                )
            else:
                frontier_length = 0.0
            metadata.append({
                "map_xy": np.asarray(point, dtype=float),
                "frontier_length": frontier_length,
                "frontier_index": frontier_idx,
            })
        return metadata, frontiers

    def select_metadata_with_spacing(self, candidates) -> List[dict]:
        selected = []
        for candidate in candidates:
            point = np.asarray(candidate["map_xy"], dtype=float)
            if self.min_spacing_pixels and self.min_spacing_pixels > 0:
                if any(
                    np.linalg.norm(
                        point - np.asarray(existing["map_xy"], dtype=float)
                    ) < self.min_spacing_pixels
                    for existing in selected
                ):
                    continue
            selected.append(candidate)
            if len(selected) >= self.max_targets:
                break
        return selected
    
    def generate_targets(
        self, 
        explored_map,
        full_map,
        obstacle_map,
        area_thresh = -1.0
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        metadata, frontiers = self.generate_candidate_metadata(
            explored_map, full_map, obstacle_map, area_thresh
        )
        if not metadata:
            return np.array([]), frontiers
        targets = self.sample_targets_with_spacing(
            np.asarray([item["map_xy"] for item in metadata])
        )
        return targets, frontiers
