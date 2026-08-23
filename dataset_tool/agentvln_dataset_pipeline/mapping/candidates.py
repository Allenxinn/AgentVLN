from typing import Dict, List, Optional, Tuple

import numpy as np

from .coordinate_transformer import VisibilityStatus


def is_depth_visible(
    world_point,
    pixel,
    agent_state,
    coord_transformer,
    depth_image,
    max_depth = 10.0,
    tolerance = 0.2,
    sample_radius = 3,
) -> bool:
    if depth_image is None:
        return False
    depth = np.asarray(depth_image)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.size == 0:
        return False

    height, width = depth.shape
    px, py = int(pixel[0]), int(pixel[1])
    if not (0 <= px < width and 0 <= py < height):
        return False

    x0, x1 = max(0, px - sample_radius), min(width, px + sample_radius + 1)
    y0, y1 = max(0, py - sample_radius), min(height, py + sample_radius + 1)
    patch = depth[y0:y1, x0:x1].astype(float)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return False

    # world_to_pixel projects navigation points onto the current floor plane.
    projected_point = np.asarray(world_point, dtype=float).copy()
    projected_point[1] = (
        float(agent_state.position[1]) - float(coord_transformer.camera_height)
    )
    expected_depth = float(
        coord_transformer.world_to_camera(projected_point, agent_state)[2]
    )
    if not np.isfinite(expected_depth) or expected_depth <= 0:
        return False

    actual_depth = float(np.median(valid))
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size > 0 and float(np.max(finite_depth)) <= 1.0 + 1e-6:
        actual_depth *= float(max_depth)
    return actual_depth + float(tolerance) >= expected_depth


def build_visible_exploration_targets(
    target_generator,
    coord_transformer,
    map_builder,
    agent_state,
    current_step,
    depth_image,
    area_thresh,
    floor_height,
    floor_match_tolerance = 0.5,
    depth_max = 10.0,
    occlusion_tolerance = 0.2,
) -> Tuple[List[Dict], Dict[str, int]]:
    explored_map = map_builder.get_explored_map()
    obstacle_map = (1 - map_builder.full_map).astype(np.uint8)
    raw, frontiers = target_generator.generate_candidate_metadata(
        explored_map.copy(), map_builder.full_map, obstacle_map, area_thresh
    )
    stats = {
        "raw": len(raw),
        "wrong_floor": 0,
        "out_of_view": 0,
        "occluded_or_invalid_depth": 0,
        "kept": 0,
    }
    visible = []
    agent_pos = np.asarray(agent_state.position, dtype=float)
    image_width = int(coord_transformer.width)
    image_height = int(coord_transformer.height)

    for item in raw:
        col, row = [int(value) for value in item["map_xy"]]
        world_point = coord_transformer.map_to_world(
            np.asarray([row, col]), agent_state.position, map_builder.full_map
        )
        world_point = np.asarray(world_point, dtype=float)
        if (
            world_point.shape != (3,)
            or not np.all(np.isfinite(world_point))
            or abs(float(world_point[1]) - float(floor_height))
            > float(floor_match_tolerance)
        ):
            stats["wrong_floor"] += 1
            continue

        pixel, status, _ = coord_transformer.world_to_pixel(
            world_point, agent_state, current_step
        )
        if (
            pixel is None
            or status != VisibilityStatus.VISIBLE
            or not (0 <= int(pixel[0]) < image_width)
            or not (0 <= int(pixel[1]) < image_height)
        ):
            stats["out_of_view"] += 1
            continue
        if not is_depth_visible(
            world_point,
            pixel,
            agent_state,
            coord_transformer,
            depth_image,
            max_depth=depth_max,
            tolerance=occlusion_tolerance,
        ):
            stats["occluded_or_invalid_depth"] += 1
            continue

        item = dict(item)
        item["horizontal_distance"] = float(
            np.linalg.norm(world_point[[0, 2]] - agent_pos[[0, 2]])
        )
        item["target"] = {
            "topdown_coords": [row, col],
            "pixel_coords": [int(pixel[0]), int(pixel[1])],
            "world_coords": world_point.tolist(),
            "visibility_status": int(VisibilityStatus.VISIBLE),
            "history_info": None,
        }
        visible.append(item)

    visible.sort(
        key=lambda item: (
            -float(item["frontier_length"]),
            float(item["horizontal_distance"]),
            int(item["target"]["topdown_coords"][0]),
            int(item["target"]["topdown_coords"][1]),
        )
    )
    selected = target_generator.select_metadata_with_spacing(visible)
    targets = [item["target"] for item in selected]
    stats["kept"] = len(targets)
    return targets, stats


def match_expert_candidate(
    sim,
    exploration_targets,
    trajectory_target,
    floor_height,
    floor_match_tolerance = 0.5,
    distance_threshold = 2.0,
) -> Optional[int]:
    if not exploration_targets or not trajectory_target:
        return None
    trajectory_world = trajectory_target.get("world_coords")
    if trajectory_world is None or floor_height is None:
        return None
    trajectory_world = np.asarray(trajectory_world, dtype=float)
    if abs(float(trajectory_world[1]) - float(floor_height)) > floor_match_tolerance:
        return None

    scored = []
    for index, target in enumerate(exploration_targets):
        candidate = np.asarray(target.get("world_coords"), dtype=float)
        if candidate.shape != (3,):
            continue
        if abs(float(candidate[1]) - float(floor_height)) > floor_match_tolerance:
            continue
        geodesic = float("inf")
        try:
            geodesic = float(sim.geodesic_distance(candidate, trajectory_world))
        except Exception:
            pass
        if np.isfinite(geodesic):
            scored.append((geodesic, index))
    if not scored:
        return None
    distance, index = min(scored)
    return int(index) if distance <= float(distance_threshold) else None
