from typing import Any, Dict, Iterable, List, Optional

import numpy as np


SCHEMA_VERSION = 3
TERMINATION_REASONS = {
    "reached_goal",
    "max_steps",
    "environment_over",
    "follower_failure",
    "missing_reference_path",
    "action_execution_failure",
}
GENERATION_CONFIG_KEYS = {
    "waypoint_radius_m",
    "stop_radius_m",
    "endpoint_target_radius_m",
    "min_unexplored_area_m2",
    "expert_candidate_match_distance_m",
    "trajectory_sample_interval_m",
    "update_interval",
}
ACTUATION_NOISE_DISTRIBUTION = "symmetric_uniform_relative_v1"
ACTUATION_NOISE_CONFIG_KEYS = {
    "actuation_noise_factor",
    "actuation_noise_seed",
    "actuation_noise_distribution",
    "actuation_noise_verbose",
}


def get_actuation_noise_settings(config) -> Dict[str, Any]:
    noise = config.get("generation", {}).get("actuation_noise", {})
    factor = float(noise.get("factor", 0.0))
    seed = noise.get("seed", 42)
    verbose = noise.get("verbose", False)
    if not np.isfinite(factor) or not 0.0 <= factor < 1.0:
        raise ValueError("generation.actuation_noise.factor must satisfy 0 <= factor < 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("generation.actuation_noise.seed must be an integer")
    if int(seed) < 0 or int(seed) >= (1 << 63):
        raise ValueError("generation.actuation_noise.seed must be in [0, 2^63)")
    if not isinstance(verbose, bool):
        raise ValueError("generation.actuation_noise.verbose must be boolean")
    return {
        "factor": factor,
        "seed": int(seed),
        "distribution": ACTUATION_NOISE_DISTRIBUTION,
        "verbose": verbose,
    }


def validate_generation_settings(config) -> None:
    exploration = config.get("exploration", {})
    area_m2 = float(exploration.get("min_unexplored_area_m2", 2.0))
    if area_m2 < 0.0:
        raise ValueError("exploration.min_unexplored_area_m2 must be >= 0")
    match_distance = float(
        exploration.get("expert_candidate_match_distance_m", 2.0)
    )
    if match_distance <= 0.0:
        raise ValueError(
            "exploration.expert_candidate_match_distance_m must be > 0"
        )
    trajectory = config.get("trajectory", {})
    if float(trajectory.get("sample_interval", 0.25)) <= 0.0:
        raise ValueError("trajectory.sample_interval must be > 0")
    if int(config.get("output", {}).get("update_interval", 5)) <= 0:
        raise ValueError("output.update_interval must be > 0")
    get_actuation_noise_settings(config)


def build_generation_config(config) -> Dict[str, Any]:
    validate_generation_settings(config)
    exploration = config.get("exploration", {})
    trajectory = config.get("trajectory", {})
    output = config.get("output", {})
    noise = get_actuation_noise_settings(config)
    return {
        "waypoint_radius_m": 0.5,
        "stop_radius_m": 1.0,
        "endpoint_target_radius_m": 1.0,
        "min_unexplored_area_m2": float(
            exploration.get("min_unexplored_area_m2", 2.0)
        ),
        "expert_candidate_match_distance_m": float(
            exploration.get("expert_candidate_match_distance_m", 2.0)
        ),
        "trajectory_sample_interval_m": float(
            trajectory.get("sample_interval", 0.25)
        ),
        "update_interval": int(output.get("update_interval", 5)),
        "actuation_noise_factor": noise["factor"],
        "actuation_noise_seed": noise["seed"],
        "actuation_noise_distribution": noise["distribution"],
        "actuation_noise_verbose": noise["verbose"],
    }


def dataset_actuation_noise_settings(generation_config):
    present = ACTUATION_NOISE_CONFIG_KEYS.intersection(generation_config)
    if present and present != ACTUATION_NOISE_CONFIG_KEYS:
        missing = ACTUATION_NOISE_CONFIG_KEYS - present
        raise ValueError(
            "Incomplete schema-v3 actuation-noise config: "
            + ", ".join(sorted(missing))
        )
    if not present:
        return {
            "factor": 0.0,
            "seed": 42,
            "distribution": ACTUATION_NOISE_DISTRIBUTION,
            "verbose": False,
        }
    factor = float(generation_config["actuation_noise_factor"])
    seed = generation_config["actuation_noise_seed"]
    distribution = generation_config["actuation_noise_distribution"]
    verbose = generation_config["actuation_noise_verbose"]
    if not np.isfinite(factor) or not 0.0 <= factor < 1.0:
        raise ValueError("Invalid schema-v3 actuation_noise_factor")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 63):
        raise ValueError("Invalid schema-v3 actuation_noise_seed")
    if distribution != ACTUATION_NOISE_DISTRIBUTION:
        raise ValueError("Unsupported schema-v3 actuation noise distribution")
    if not isinstance(verbose, bool):
        raise ValueError("Invalid schema-v3 actuation_noise_verbose")
    return {
        "factor": factor,
        "seed": seed,
        "distribution": distribution,
        "verbose": verbose,
    }


def square_meters_to_pixel_area(
    area_m2, meters_per_pixel
) -> float:
    area_m2 = float(area_m2)
    meters_per_pixel = float(meters_per_pixel)
    if area_m2 < 0.0:
        raise ValueError("area_m2 must be >= 0")
    if meters_per_pixel <= 0.0 or not np.isfinite(meters_per_pixel):
        raise ValueError("meters_per_pixel must be finite and > 0")
    if area_m2 == 0.0:
        return 0.0
    return area_m2 / (meters_per_pixel ** 2)


def strict_geodesic_distance(sim, start, end) -> float:
    if start is None or end is None:
        return float("inf")
    try:
        distance = float(sim.geodesic_distance(start, end))
    except Exception:
        return float("inf")
    return distance if np.isfinite(distance) and distance >= 0.0 else float("inf")


def densify_polyline(
    points, interval = 0.25
) -> List[np.ndarray]:
    path = [np.asarray(point, dtype=float) for point in points]
    if not path:
        return []
    if len(path) == 1:
        return path
    dense = [path[0]]
    for end in path[1:]:
        start = dense[-1]
        distance = float(np.linalg.norm(end - start))
        if distance <= 1e-8:
            continue
        steps = max(1, int(np.ceil(distance / float(interval))))
        for index in range(1, steps + 1):
            dense.append(start + (end - start) * (index / steps))
    return dense


def find_geodesic_segment(sim, start, end) -> Optional[List[np.ndarray]]:
    try:
        import habitat_sim

        shortest_path_type = getattr(habitat_sim, "ShortestPath", None)
        if shortest_path_type is None:
            shortest_path_type = habitat_sim.nav.ShortestPath
        request = shortest_path_type()
        request.requested_start = np.asarray(start, dtype=float)
        request.requested_end = np.asarray(end, dtype=float)
        found = sim.pathfinder.find_path(request)
        points = [np.asarray(point, dtype=float) for point in request.points]
        if (
            not found
            or not np.isfinite(float(request.geodesic_distance))
            or len(points) < 1
        ):
            return None
        return points
    except Exception:
        return None


def build_geodesic_path(
    sim,
    start,
    ordered_targets,
    interval = 0.25,
) -> Optional[List[np.ndarray]]:
    current = np.asarray(start, dtype=float)
    joined: List[np.ndarray] = [current]
    for target in ordered_targets:
        target = np.asarray(target, dtype=float)
        if np.linalg.norm(target - current) <= 1e-6:
            continue
        segment = find_geodesic_segment(sim, current, target)
        if not segment:
            return None
        if np.linalg.norm(segment[0] - joined[-1]) <= 1e-4:
            segment = segment[1:]
        joined.extend(segment)
        current = target
    return densify_polyline(joined, interval)


def validate_dataset_contract(dataset, expected_config=None) -> None:
    if int(dataset.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version={SCHEMA_VERSION}, got "
            f"{dataset.get('schema_version')!r}; use a new output directory"
        )
    generation_config = dataset.get("generation_config")
    if not isinstance(generation_config, dict):
        raise ValueError("Schema-v3 dataset is missing generation_config")
    missing_config = GENERATION_CONFIG_KEYS - set(generation_config)
    if missing_config:
        raise ValueError(
            "Schema-v3 generation_config is missing: "
            + ", ".join(sorted(missing_config))
        )
    if expected_config is not None and generation_config != expected_config:
        raise ValueError(
            "Generation settings do not match the existing shard; "
            "use a new output directory"
        )
    noise_settings = dataset_actuation_noise_settings(generation_config)

    stop_radius = float(generation_config["stop_radius_m"])
    endpoint_radius = float(generation_config["endpoint_target_radius_m"])
    columns = (
        "topdown_coords", "pixel_coords", "world_coords",
        "visibility_status", "history_info", "trajectory_pixel",
        "trajectory_world", "trajectory_status",
        "trajectory_goal_distances", "trajectory_is_endpoint", "actions",
        "floor_ids", "floor_heights", "floor_transition",
        "expert_candidate_indices",
    )
    for task_index, task in enumerate(dataset.get("tasks", [])):
        prefix = f"Schema-v3 task {task_index}"
        for key in (
            "goal_world", "termination_reason", "success",
            "final_geodesic_distance",
        ):
            if key not in task:
                raise ValueError(f"{prefix} is missing {key}")
        goal = task.get("goal_world")
        if not isinstance(goal, list) or len(goal) != 3:
            raise ValueError(f"{prefix} has invalid goal_world")
        reason = task.get("termination_reason")
        if reason not in TERMINATION_REASONS:
            raise ValueError(f"{prefix} has invalid termination_reason={reason!r}")
        actions = task.get("actions")
        if not isinstance(actions, list):
            raise ValueError(f"{prefix} is missing actions")
        length = len(actions)
        if length == 0:
            raise ValueError(f"{prefix} has no samples")
        for key in columns:
            value = task.get(key)
            if not isinstance(value, list) or len(value) != length:
                raise ValueError(
                    f"{prefix} has invalid {key} length; expected {length}"
                )
        action_debug = task.get("action_debug")
        if noise_settings["verbose"] and action_debug is None:
            raise ValueError(f"{prefix} is missing verbose action_debug")
        if action_debug is not None:
            if not noise_settings["verbose"]:
                raise ValueError(f"{prefix} has action_debug while verbose is disabled")
            if not isinstance(action_debug, list) or len(action_debug) != length:
                raise ValueError(f"{prefix} has invalid action_debug length")
        stop_steps = []
        for step, action in enumerate(actions):
            if action is None:
                if action_debug is not None and action_debug[step] is not None:
                    raise ValueError(
                        f"{prefix} has debug data for an empty step {step}"
                    )
                continue
            if int(action) not in (0, 1, 2, 3):
                raise ValueError(f"{prefix} has invalid action at step {step}")
            if int(action) == 0:
                stop_steps.append(step)
            if action_debug is not None:
                debug = action_debug[step]
                if (
                    not isinstance(debug, dict)
                    or int(debug.get("expert_action", -1)) != int(action)
                ):
                    raise ValueError(
                        f"{prefix} action_debug disagrees at step {step}"
                    )
                if int(action) == 0:
                    if bool(debug.get("executed")):
                        raise ValueError(f"{prefix} executed STOP at step {step}")
                else:
                    if not bool(debug.get("executed")):
                        raise ValueError(
                            f"{prefix} did not execute saved action at step {step}"
                        )
                    scale = float(debug.get("scale"))
                    lower = 1.0 - noise_settings["factor"]
                    upper = 1.0 + noise_settings["factor"]
                    if not lower - 1e-12 <= scale <= upper + 1e-12:
                        raise ValueError(
                            f"{prefix} action_debug scale is out of bounds at step {step}"
                        )
                    nominal = float(debug.get("nominal_amount"))
                    sampled = float(debug.get("sampled_amount"))
                    if not np.isclose(sampled, nominal * scale):
                        raise ValueError(
                            f"{prefix} action_debug amount disagrees at step {step}"
                        )
            if bool(task["floor_transition"][step]):
                if (
                    task["pixel_coords"][step]
                    or task["world_coords"][step]
                    or task["trajectory_pixel"][step] is not None
                    or task["trajectory_world"][step] is not None
                    or task["expert_candidate_indices"][step] is not None
                    or bool(task["trajectory_is_endpoint"][step])
                ):
                    raise ValueError(
                        f"{prefix} transition step {step} has coordinate labels"
                    )
            candidate = task["expert_candidate_indices"][step]
            if candidate is not None and not (
                0 <= int(candidate) < len(task["pixel_coords"][step])
            ):
                raise ValueError(
                    f"{prefix} has invalid expert candidate at step {step}"
                )
            if bool(task["trajectory_is_endpoint"][step]):
                distance = task["trajectory_goal_distances"][step]
                if (
                    distance is None
                    or not np.isfinite(float(distance))
                    or float(distance) > endpoint_radius
                ):
                    raise ValueError(
                        f"{prefix} has invalid endpoint at step {step}"
                    )
        success = bool(task.get("success"))
        if success != (reason == "reached_goal"):
            raise ValueError(f"{prefix} success disagrees with termination_reason")
        final_distance = task.get("final_geodesic_distance")
        if success:
            if (
                stop_steps != [length - 1]
                or final_distance is None
                or not np.isfinite(float(final_distance))
                or float(final_distance) > stop_radius
            ):
                raise ValueError(f"{prefix} has an unverifiable STOP")
        elif stop_steps:
            raise ValueError(f"{prefix} failure trajectory contains STOP")
