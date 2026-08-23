import numpy as np

from ..core.models import EpisodeGenerationResult, PreparedSample
from ..mapping.candidates import build_visible_exploration_targets, match_expert_candidate
from ..mapping.coordinate_transformer import CoordinateTransformer
from ..mapping.exploration_targets import ExplorationTargetGenerator
from ..mapping.floor_manager import FloorMapManager
from ..mapping.topdown_map import convert_square_meters_to_pixel_area
from ..mapping.trajectory_mapper import TrajectoryMapper
from ..navigation.action_executor import (
    ActionExecutionError,
    PerturbedActionExecutor,
    action_debug_record,
)
from ..navigation.expert_follower import ExpertTrajectoryFollower
from ..core.dataset_contract import strict_geodesic_distance


def _empty_stats():
    return {
        "raw": 0,
        "wrong_floor": 0,
        "out_of_view": 0,
        "occluded_or_invalid_depth": 0,
        "kept": 0,
    }


def _commit_prepared_sample(prepared, writer, task_data, scene_id, task_id,
                            action_debug):
    writer.commit_sample(task_data, scene_id, task_id, prepared, action_debug)


def generate_current_episode(
    env,
    obs,
    config,
    writer,
) -> EpisodeGenerationResult:
    from habitat.utils.visualizations import maps

    episode = env.current_episode
    sim = env._sim
    scene_id = episode.scene_id.split("/")[-1].replace(".glb", "")
    task_id = str(episode.episode_id)
    instruction = (
        episode.instruction.instruction_text
        if hasattr(episode.instruction, "instruction_text")
        else str(episode.instruction)
    )

    if not (hasattr(episode, "reference_path") and episode.reference_path):
        return EpisodeGenerationResult(None, "missing_reference_path", _empty_stats())
    if not (hasattr(episode, "goals") and episode.goals):
        return EpisodeGenerationResult(None, "missing_reference_path", _empty_stats())
    waypoints = [np.asarray(point, dtype=float) for point in episode.reference_path]
    goal = np.asarray(episode.goals[0].position, dtype=float)

    generation_config = writer.dataset["generation_config"]
    start = np.asarray(sim.get_agent_state().position, dtype=float)
    floor_manager = FloorMapManager(sim, config["map"], initial_height=float(start[1]))
    target_generator = ExplorationTargetGenerator(config["exploration"])
    transformer = CoordinateTransformer(config["camera"], sim)
    trajectory_mapper = TrajectoryMapper(config["trajectory"], transformer)
    follower = ExpertTrajectoryFollower(
        sim,
        waypoints,
        goal,
        waypoint_radius=generation_config["waypoint_radius_m"],
        goal_radius=generation_config["stop_radius_m"],
        sample_interval=generation_config["trajectory_sample_interval_m"],
    )
    action_executor = PerturbedActionExecutor(
        sim,
        factor=generation_config.get("actuation_noise_factor", 0.0),
        seed=generation_config.get("actuation_noise_seed", 42),
        verbose=generation_config.get("actuation_noise_verbose", False),
    )
    action_debug_enabled = bool(
        generation_config.get("actuation_noise_verbose", False)
    )

    meters_per_pixel = maps.calculate_meters_per_pixel(
        config["map"]["resolution"], sim=sim
    )
    target_generator.set_pixel_scale(meters_per_pixel)
    trajectory_mapper.set_map_scale(meters_per_pixel)
    area_threshold = convert_square_meters_to_pixel_area(
        generation_config["min_unexplored_area_m2"],
        config["map"]["resolution"],
        sim,
    )

    task_data = writer.create_task_data(
        scene_id, task_id, instruction, 0, goal_world=goal
    )

    update_interval = int(generation_config["update_interval"])
    max_steps = int(config.get("generation", {}).get("max_steps", 500))
    endpoint_radius = float(generation_config["endpoint_target_radius_m"])
    match_radius = float(
        generation_config["expert_candidate_match_distance_m"]
    )
    candidate_stats = _empty_stats()
    floor_switches = 0
    transition_frames = 0
    termination_reason = None
    final_distance = float("inf")
    step = 0

    while True:
        if step >= max_steps:
            termination_reason = "max_steps"
            break
        if env.episode_over:
            termination_reason = "environment_over"
            break

        agent_state = sim.get_agent_state()
        floor_update = floor_manager.update(agent_state)
        floor_switches += int(floor_update.switched)
        transition_frames += int(floor_update.in_transition)
        map_builder = floor_manager.active_builder
        floor_manager.update_visibility(
            agent_state, fov=config["camera"].get("hfov", 110)
        )
        decision = follower.decide()
        final_distance = decision.goal_geodesic_distance
        if decision.termination_reason == "follower_failure":
            termination_reason = "follower_failure"
            break
        if not decision.should_stop and (
            decision.action is None or int(decision.action) not in (1, 2, 3)
        ):
            termination_reason = "follower_failure"
            break

        action_code = 0 if decision.should_stop else int(decision.action)
        execution_plan = action_executor.plan(
            action_code, scene_id, task_id, step
        )
        should_save = decision.should_stop or step % update_interval == 0
        prepared_sample = None
        if should_save:
            rgb = obs.get("rgb", obs.get("front", None))
            if rgb is None:
                termination_reason = "follower_failure"
                break
            else:
                depth = obs.get("depth", None)
                processing_height = float(agent_state.position[1])
                if floor_update.in_transition:
                    exploration_targets = []
                    trajectory_target = None
                    expert_candidate_index = None
                    trajectory_goal_distance = None
                    trajectory_is_endpoint = False
                else:
                    exploration_targets, stats = (
                        build_visible_exploration_targets(
                            target_generator,
                            transformer,
                            map_builder,
                            agent_state,
                            step,
                            depth,
                            area_threshold,
                            processing_height,
                            floor_manager.floor_match_tolerance,
                            config.get("camera", {}).get("depth_max", 10.0),
                            config.get("exploration", {}).get(
                                "occlusion_tolerance", 0.2
                            ),
                        )
                    )
                    for key, value in stats.items():
                        candidate_stats[key] += value
                    trajectory_target = trajectory_mapper.get_trajectory_target(
                        decision.remaining_geodesic_path,
                        agent_state,
                        map_builder.full_map,
                        step,
                        depth,
                        floor_height=processing_height,
                        in_transition=False,
                    )
                    trajectory_world = (
                        trajectory_target.get("world_coords")
                        if trajectory_target else None
                    )
                    trajectory_goal_distance = strict_geodesic_distance(
                        sim, trajectory_world, goal
                    )
                    if not np.isfinite(trajectory_goal_distance):
                        trajectory_goal_distance = None
                    trajectory_is_endpoint = (
                        trajectory_goal_distance is not None
                        and trajectory_goal_distance <= endpoint_radius
                    )
                    expert_candidate_index = match_expert_candidate(
                        sim,
                        exploration_targets,
                        trajectory_target,
                        processing_height,
                        floor_manager.floor_match_tolerance,
                        distance_threshold=match_radius,
                    )

                topdown = map_builder.build_topdown_frame(
                    agent_state, fov=config["camera"].get("hfov", 110)
                )
                prepared_sample = PreparedSample(
                    step=step,
                    exploration_targets=exploration_targets,
                    trajectory_target=trajectory_target,
                    action_code=action_code,
                    floor_id=floor_update.floor_id,
                    floor_height=floor_update.floor_height,
                    floor_transition=floor_update.in_transition,
                    expert_candidate_index=expert_candidate_index,
                    trajectory_goal_distance=trajectory_goal_distance,
                    trajectory_is_endpoint=trajectory_is_endpoint,
                    rgb=rgb,
                    topdown=topdown,
                )

        if decision.should_stop:
            _commit_prepared_sample(
                prepared_sample,
                writer,
                task_data,
                scene_id,
                task_id,
                (
                    action_debug_record(execution_plan)
                    if action_debug_enabled else None
                ),
            )
            termination_reason = "reached_goal"
            break
        try:
            execution_result = action_executor.execute(env, execution_plan)
        except ActionExecutionError as error:
            if prepared_sample is not None and error.executed:
                _commit_prepared_sample(
                    prepared_sample,
                    writer,
                    task_data,
                    scene_id,
                    task_id,
                    (
                        action_debug_record(execution_plan, error.result)
                        if action_debug_enabled else None
                    ),
                )
            termination_reason = "action_execution_failure"
            break
        if prepared_sample is not None:
            _commit_prepared_sample(
                prepared_sample,
                writer,
                task_data,
                scene_id,
                task_id,
                (
                    action_debug_record(execution_plan, execution_result)
                    if action_debug_enabled else None
                ),
            )
        obs = execution_result.observation
        step += 1

    try:
        terminal_position = np.asarray(
            sim.get_agent_state().position, dtype=float
        )
    except Exception:
        terminal_position = None
    terminal_distance = strict_geodesic_distance(
        sim, terminal_position, goal
    )
    if np.isfinite(terminal_distance):
        final_distance = terminal_distance
    elif termination_reason != "reached_goal":
        final_distance = float("inf")
    writer.update_task_total_steps(task_data, len(task_data["actions"]))
    writer.set_task_floors(task_data, floor_manager.get_floor_metadata())
    writer.finalize_task(task_data, termination_reason, final_distance)
    if not any(action is not None for action in task_data["actions"]):
        task_data = None
    return EpisodeGenerationResult(
        task_data,
        termination_reason,
        candidate_stats,
        floor_switches,
        transition_frames,
    )


class EpisodeGenerator:
    def __init__(self, config, writer):
        self.config = config
        self.writer = writer

    def generate(self, env, obs) -> EpisodeGenerationResult:
        return generate_current_episode(env, obs, self.config, self.writer)
