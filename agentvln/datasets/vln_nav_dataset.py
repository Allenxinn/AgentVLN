import os
import json
import gzip
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        def __getitem__(self, idx): raise NotImplementedError
        def __len__(self): raise NotImplementedError

from .nav_dataset_config import NavDatasetConfig


VISIBLE = 0
BEHIND = 1
LEFT = 2
RIGHT = 3


class VLNNavDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        data_root: str,
        config: Optional[NavDatasetConfig] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = data_root
        self.config = config or NavDatasetConfig()
        self.rng = random.Random(seed)
        self._lmdb_envs = {}  
        
        self.data = self._load_json(json_path)
        self.schema_version = int(self.data.get("schema_version", 1))
        self._validate_schema()
        
        self.samples: List[Tuple[int, int]] = []
        
        self.task_valid_steps: Dict[int, List[int]] = {}
        self._build_sample_index()
        
        print(f"[VLNNavDataset] Loaded {len(self.data.get('tasks', []))} tasks, "
              f"{len(self.samples)} valid samples.")

    def _validate_schema(self):
        tasks = self.data.get("tasks", [])
        if self.schema_version < 3:
            skipped_stops = sum(
                1 for task in tasks for action in task.get("actions", [])
                if action is not None and int(action) == 0
            )
            skipped_targets = sum(
                1
                for task in tasks
                for target in task.get("trajectory_pixel", [])
                if target is not None
            )
            abnormal_tasks = 0
            for task in tasks:
                action_count = len(task.get("actions", []))
                lengths = [
                    len(task.get(key, []))
                    for key in (
                        "pixel_coords", "world_coords", "visibility_status",
                        "trajectory_pixel", "trajectory_world",
                        "trajectory_status",
                    )
                    if key in task
                ]
                abnormal_tasks += int(
                    not task.get("actions")
                    or any(length != action_count for length in lengths)
                )
            self.legacy_downgrade_summary = {
                "skipped_stops": skipped_stops,
                "disabled_target_frames": skipped_targets,
                "abnormal_tasks": abnormal_tasks,
            }
            print(
                "[VLNNavDataset] Legacy schema detected: STOP and target "
                "samples are disabled; "
                f"skipped_stop={skipped_stops}, "
                f"disabled_target_frames={skipped_targets}, "
                f"abnormal_tasks={abnormal_tasks}."
            )
            return
        if self.schema_version != 3:
            raise ValueError(f"Unsupported schema_version={self.schema_version}")
        generation_config = self.data.get("generation_config")
        if not isinstance(generation_config, dict):
            raise ValueError("Schema v3 requires top-level generation_config")
        required_config = {
            "waypoint_radius_m", "stop_radius_m", "endpoint_target_radius_m",
            "min_unexplored_area_m2", "expert_candidate_match_distance_m",
            "trajectory_sample_interval_m", "update_interval",
        }
        missing_config = required_config - set(generation_config)
        if missing_config:
            raise ValueError(
                "Schema v3 generation_config is missing: "
                + ", ".join(sorted(missing_config))
            )
        required = (
            "goal_world", "termination_reason", "success",
            "final_geodesic_distance", "trajectory_goal_distances",
            "trajectory_is_endpoint", "expert_candidate_indices",
        )
        for index, task in enumerate(tasks):
            missing = [key for key in required if key not in task]
            if missing:
                raise ValueError(
                    f"Schema-v3 task {index} is missing: {', '.join(missing)}"
                )
            if task["goal_world"] is None or task["termination_reason"] is None:
                raise ValueError(
                    f"Schema-v3 task {index} has invalid goal/termination metadata"
                )
            action_count = len(task.get("actions", []))
            if action_count == 0:
                raise ValueError(f"Schema-v3 task {index} has no samples")
            for key in (
                "topdown_coords", "pixel_coords", "world_coords",
                "visibility_status", "history_info", "trajectory_pixel",
                "trajectory_world", "trajectory_status",
                "trajectory_goal_distances", "trajectory_is_endpoint",
                "actions", "floor_ids", "floor_heights", "floor_transition",
                "expert_candidate_indices",
            ):
                if key not in task:
                    raise ValueError(f"Schema-v3 task {index} is missing {key}")
                if len(task[key]) != action_count:
                    raise ValueError(
                        f"Schema-v3 task {index} has {key} length "
                        f"{len(task[key])}, expected {action_count}"
                    )
            stop_steps = [
                step for step, action in enumerate(task.get("actions", []))
                if action is not None and int(action) == 0
            ]
            reached_goal = task["termination_reason"] == "reached_goal"
            if bool(task["success"]) != reached_goal:
                raise ValueError(
                    f"Schema-v3 task {index} success disagrees with termination"
                )
            final_distance = task.get("final_geodesic_distance")
            stop_radius = float(generation_config["stop_radius_m"])
            if reached_goal and (
                stop_steps != [action_count - 1]
                or final_distance is None
                or not np.isfinite(float(final_distance))
                or float(final_distance) > stop_radius
            ):
                raise ValueError(
                    f"Schema-v3 task {index} has an unverifiable STOP"
                )
            if not reached_goal and stop_steps:
                raise ValueError(
                    f"Schema-v3 failed task {index} contains STOP"
                )
            endpoint_radius = float(
                generation_config["endpoint_target_radius_m"]
            )
            for step in range(action_count):
                if bool(task["floor_transition"][step]) and (
                    task["pixel_coords"][step]
                    or task["trajectory_pixel"][step] is not None
                    or task["expert_candidate_indices"][step] is not None
                    or bool(task["trajectory_is_endpoint"][step])
                ):
                    raise ValueError(
                        f"Schema-v3 task {index} transition step {step} "
                        "contains coordinate supervision"
                    )
                if bool(task["trajectory_is_endpoint"][step]):
                    distance = task["trajectory_goal_distances"][step]
                    if (
                        distance is None
                        or not np.isfinite(float(distance))
                        or float(distance) > endpoint_radius
                    ):
                        raise ValueError(
                            f"Schema-v3 task {index} has invalid endpoint "
                            f"at step {step}"
                        )
    
    @staticmethod
    def _load_json(filepath: str) -> Dict[str, Any]:
        """Load JSON or gzipped JSON file."""
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_sample_index(self):
        """
        Build flat (task_idx, step_idx) sample list from the dataset.
        
        Action sequence deduplication:
        When a step produces a merged action sequence of N identical actions,
        the following N-1 steps with the same action are SKIPPED as samples
        (they would produce redundant shorter sequences). However, all valid
        steps remain in task_valid_steps for history frame collection.
        """
        tasks = self.data.get('tasks', [])
        for task_idx, task in enumerate(tasks):
            actions = task.get('actions', [])
            valid_steps = [
                step_idx
                for step_idx, action in enumerate(actions)
                if action is not None
                and (getattr(self, "schema_version", 1) >= 3 or int(action) != 0)
            ]
            self.task_valid_steps[task_idx] = valid_steps

        for task_idx, task in enumerate(tasks):
            valid_steps = self.task_valid_steps[task_idx]
            actions = task.get('actions', [])
            pos = 0
            while pos < len(valid_steps):
                step_idx = valid_steps[pos]
                self.samples.append((task_idx, step_idx))

                mode, _ = self._determine_output_mode(
                    self._safe_get(task, 'pixel_coords', step_idx, []),
                    self._safe_get(task, 'world_coords', step_idx, []),
                    self._safe_get(task, 'visibility_status', step_idx, []),
                    self._safe_get(task, 'trajectory_pixel', step_idx, None),
                    self._safe_get(task, 'trajectory_status', step_idx, None),
                    self._safe_get(task, 'trajectory_world', step_idx, None),
                    actions[step_idx], task, task_idx, step_idx,
                    bool(self._safe_get(task, 'floor_transition', step_idx, False)),
                    self._safe_get(
                        task, 'expert_candidate_indices', step_idx, None
                    ),
                    bool(self._safe_get(
                        task, 'trajectory_is_endpoint', step_idx, False
                    )),
                )
                if mode != "action":
                    pos += 1
                    continue

                current_action = int(actions[step_idx])
                seq_len = 1
                for j in range(
                    pos + 1,
                    min(
                        pos + 1 + self.config.max_action_lookahead,
                        len(valid_steps),
                    ),
                ):
                    next_step = valid_steps[j]
                    next_action = actions[next_step]
                    next_mode, _ = self._determine_output_mode(
                        self._safe_get(task, 'pixel_coords', next_step, []),
                        self._safe_get(task, 'world_coords', next_step, []),
                        self._safe_get(task, 'visibility_status', next_step, []),
                        self._safe_get(task, 'trajectory_pixel', next_step, None),
                        self._safe_get(task, 'trajectory_status', next_step, None),
                        self._safe_get(task, 'trajectory_world', next_step, None),
                        next_action, task, task_idx, next_step,
                        bool(self._safe_get(
                            task, 'floor_transition', next_step, False
                        )),
                        self._safe_get(
                            task, 'expert_candidate_indices', next_step, None
                        ),
                        bool(self._safe_get(
                            task, 'trajectory_is_endpoint', next_step, False
                        )),
                    )
                    if (
                        next_mode == "action"
                        and next_action is not None
                        and int(next_action) == current_action
                    ):
                        seq_len += 1
                    else:
                        break
                pos += seq_len
    

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        max_retries = 20
        current_idx = idx
        
        for attempt in range(max_retries):
            try:
                return self._get_item_inner(current_idx)
            except (IOError, ValueError) as e:
                print(f"[VLNNavDataset] Warning: Sample {current_idx} failed: {e}. Retrying with random sample...")
                current_idx = self.rng.randint(0, len(self) - 1)
        
        raise RuntimeError(f"Failed to load valid sample after {max_retries} attempts.")

    def _get_item_inner(self, idx: int) -> Dict[str, Any]:
        task_idx, step_idx = self.samples[idx]
        task = self.data['tasks'][task_idx]
        
        candidates_pixel = self._safe_get(task, 'pixel_coords', step_idx, [])
        candidates_world = self._safe_get(task, 'world_coords', step_idx, [])
        candidates_vis = self._safe_get(task, 'visibility_status', step_idx, [])
        
        traj_pixel = self._safe_get(task, 'trajectory_pixel', step_idx, None)
        traj_world = self._safe_get(task, 'trajectory_world', step_idx, None)
        traj_status = self._safe_get(task, 'trajectory_status', step_idx, None)
        floor_transition = bool(
            self._safe_get(task, 'floor_transition', step_idx, False)
        )
        expert_candidate_index = self._safe_get(
            task, 'expert_candidate_indices', step_idx, None
        )
        trajectory_is_endpoint = bool(self._safe_get(
            task, 'trajectory_is_endpoint', step_idx, False
        ))
        
        action = task['actions'][step_idx]
        instruction = task.get('instruction', '')
        scene_id = task['scene_id']
        task_id = task['task_id']
        
        output_mode, label_data = self._determine_output_mode(
            candidates_pixel, candidates_world, candidates_vis,
            traj_pixel, traj_status, traj_world,
            action, task, task_idx, step_idx, floor_transition,
            expert_candidate_index, trajectory_is_endpoint,
        )
        
        visible_frontiers = (
            [] if floor_transition else
            self._get_visible_frontiers(candidates_pixel, candidates_vis)
        )
        history_paths = self._get_history_image_paths(scene_id, task_id, task_idx, step_idx)
        current_image_path = self._get_image_path(scene_id, task_id, step_idx)
        prompt = self._build_prompt(instruction, visible_frontiers, history_paths)
        response, trainable_ranges = self._build_response(output_mode, label_data)
        all_image_paths = history_paths + [current_image_path]
        if self.is_lmdb:
            all_images = self._get_image_data(all_image_paths)
        else:
            all_images = all_image_paths
        
        label_coords = None
        action_sequence = None
        if output_mode in ("frontier", "target"):
            label_coords = label_data
        elif output_mode == "action":
            action_sequence = label_data
        
        return {
            "prompt": prompt,
            "response": response,
            "images": all_images,
            "output_mode": output_mode,
            "label_coords": label_coords,
            "action_sequence": action_sequence,
            "trainable_ranges": trainable_ranges,
            "all_candidates_pixel": candidates_pixel,
            "all_candidates_vis": candidates_vis,
            "traj_pixel": traj_pixel,
            "metadata": {
                "scene_id": scene_id,
                "task_id": task_id,
                "step": step_idx,
                "task_idx": task_idx,
                "instruction": instruction,
                "floor_id": self._safe_get(task, 'floor_ids', step_idx, None),
                "floor_height": self._safe_get(
                    task, 'floor_heights', step_idx, None
                ),
                "floor_transition": floor_transition,
            }
        }

    @property
    def is_lmdb(self) -> bool:
        if self._is_lmdb_path(self.data_root):
            return True
        if os.path.isdir(self.data_root):
            try:
                for name in os.listdir(self.data_root):
                    if name.endswith('.lmdb') and os.path.isdir(os.path.join(self.data_root, name)):
                        return True
            except OSError:
                pass
        return False

    def _get_lmdb_env(self, scene_id: Optional[str] = None):
        import lmdb
        
        # Case 1: Monolithic LMDB at data_root
        if self._is_lmdb_path(self.data_root):
            if 'monolithic' not in self._lmdb_envs:
                self._lmdb_envs['monolithic'] = lmdb.open(
                    self.data_root, readonly=True, lock=False, readahead=False, meminit=False
                )
            return self._lmdb_envs['monolithic']

        if scene_id:
            if scene_id not in self._lmdb_envs:
                lmdb_path = os.path.join(self.data_root, f"{scene_id}.lmdb")
                if self._is_lmdb_path(lmdb_path):
                    self._lmdb_envs[scene_id] = lmdb.open(
                        lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
                    )
                else:
                    return None
            return self._lmdb_envs[scene_id]
            
        return None

    def _is_lmdb_path(self, path: str) -> bool:
        if os.path.isdir(path):
            return os.path.exists(os.path.join(path, "data.mdb"))
        return os.path.isfile(path) and path.endswith(".lmdb")

    def reset_lmdb(self):
        for env in self._lmdb_envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._lmdb_envs = {}

    def _get_image_data(self, paths: List[str]) -> List[Any]:
        if not self.is_lmdb:
            return paths
        
        images = []
        import io
        from PIL import Image
        
        for path in paths:
            rel_path = os.path.relpath(path, self.data_root).replace("\\", "/")
            parts = rel_path.split("/")
            
            img_bytes = None
            
            if len(parts) >= 2:
                scene_id = parts[0]
                
                env = self._get_lmdb_env(scene_id)
                
                if env:
                    with env.begin(write=False) as txn:
                        task_rel_path = "/".join(parts[1:])
                        key_no_ext = os.path.splitext(task_rel_path)[0]
                        img_bytes = txn.get(key_no_ext.encode("utf-8"))
                        
                        if img_bytes is None:
                             full_key_no_ext = os.path.splitext(rel_path)[0]
                             img_bytes = txn.get(full_key_no_ext.encode("utf-8"))

                        if img_bytes is None:
                            img_bytes = txn.get(rel_path.encode("utf-8"))
            
            if img_bytes is None:
                env = self._get_lmdb_env(None) 
                if env:
                    with env.begin(write=False) as txn:
                        # Try keys
                        if len(parts) >= 2:
                             task_rel_path = "/".join(parts[1:])
                             key_no_ext = os.path.splitext(task_rel_path)[0]
                             img_bytes = txn.get(key_no_ext.encode("utf-8"))
                        
                        if img_bytes is None:
                             full_key_no_ext = os.path.splitext(rel_path)[0]
                             img_bytes = txn.get(full_key_no_ext.encode("utf-8"))

                        if img_bytes is None:
                            img_bytes = txn.get(rel_path.encode("utf-8"))

            if img_bytes is not None:
                try:
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                except Exception as e:
                    raise IOError(f"Failed to decode image from LMDB: {rel_path}. Error: {e}")
            else:
                raise IOError(f"Image not found in LMDB: {rel_path}")
                
        return images
    
    
    def _determine_output_mode(
        self,
        candidates_pixel: List, candidates_world: List, candidates_vis: List,
        traj_pixel, traj_status, traj_world,
        action: int, task: Dict, task_idx: int, step_idx: int,
        floor_transition: bool = False,
        expert_candidate_index: int = None,
        trajectory_is_endpoint: bool = False,
    ) -> Tuple[str, Any]:
        if action == 0:
            return "stop", None
        if floor_transition:
            return "action", self._get_action_sequence(task, task_idx, step_idx)
        
        is_endpoint = (
            bool(trajectory_is_endpoint)
            if getattr(self, "schema_version", 1) >= 3
            else False
        )
        
        traj_is_visible = (traj_status is not None and int(traj_status) == VISIBLE)
        
        if is_endpoint and traj_is_visible and traj_pixel is not None:
            return "target", list(traj_pixel)
        
        
        if (
            expert_candidate_index is not None
            and 0 <= int(expert_candidate_index) < len(candidates_pixel)
        ):
            closest_idx, closest_dist = int(expert_candidate_index), 0.0
        elif getattr(self, "schema_version", 1) < 3:
            closest_idx, closest_dist = self._find_closest_candidate(
                candidates_world, candidates_pixel, traj_world, traj_pixel
            )
        else:
            closest_idx, closest_dist = None, float('inf')
        
        has_close_candidate = (
            closest_idx is not None
            if getattr(self, "schema_version", 1) >= 3
            else closest_idx is not None
            and closest_dist < self._get_distance_threshold(
                traj_world is not None
            )
        )
        
        if has_close_candidate:
            cand_vis = (candidates_vis[closest_idx]
                        if closest_idx < len(candidates_vis) else None)
            if cand_vis is not None and int(cand_vis) == VISIBLE:
                return "frontier", list(candidates_pixel[closest_idx])
        
        # Rule 3: action sequence (no suitable candidate)
        action_seq = self._get_action_sequence(task, task_idx, step_idx)
        return "action", action_seq
    
    def _is_target_endpoint(
        self, task_idx: int, step_idx: int, traj_world
    ) -> bool:
        if getattr(self, "schema_version", 1) >= 3:
            return bool(self._safe_get(
                self.data['tasks'][task_idx],
                'trajectory_is_endpoint', step_idx, False
            ))
        return False
    
    def _find_closest_candidate(
        self,
        candidates_world: List, candidates_pixel: List,
        traj_world, traj_pixel
    ) -> Tuple[Optional[int], float]:
        closest_idx = None
        closest_dist = float('inf')
        
        if traj_world is not None and candidates_world:
            traj_pos = np.array(traj_world, dtype=float)
            for i, cw in enumerate(candidates_world):
                if cw is not None:
                    dist = np.linalg.norm(np.array(cw, dtype=float) - traj_pos)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = i
        elif traj_pixel is not None and candidates_pixel:
            traj_px = np.array(traj_pixel, dtype=float)
            for i, cp in enumerate(candidates_pixel):
                if cp is not None:
                    dist = np.linalg.norm(np.array(cp, dtype=float) - traj_px)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = i
        
        return closest_idx, closest_dist
    
    def _get_distance_threshold(self, using_world: bool) -> float:
        if using_world:
            return self.config.distance_threshold  # meters
        return self.config.pixel_distance_threshold  # pixels
    
    def _get_action_sequence(
        self, task: Dict, task_idx: int, step_idx: int
    ) -> List[int]:
        valid_steps = self.task_valid_steps.get(task_idx, [])
        actions = task.get('actions', [])
        
        try:
            current_pos = valid_steps.index(step_idx)
        except ValueError:
            return [int(actions[step_idx])] if step_idx < len(actions) else [1]
        
        current_action = int(actions[step_idx])
        if current_action == 0: 
            return [0]
        
        sequence = [current_action]
        max_look = self.config.max_action_lookahead
        current_transition = bool(
            self._safe_get(task, 'floor_transition', step_idx, False)
        )
        
        for i in range(current_pos + 1,
                       min(current_pos + 1 + max_look, len(valid_steps))):
            next_step = valid_steps[i]
            next_action = actions[next_step]
            next_transition = bool(
                self._safe_get(task, 'floor_transition', next_step, False)
            )
            if next_transition != current_transition:
                break
            if self._step_has_coordinate_supervision(task, task_idx, next_step):
                break
            if next_action is not None and int(next_action) == current_action:
                sequence.append(current_action)
            else:
                break
        
        return sequence

    def _step_has_coordinate_supervision(
        self, task: Dict, task_idx: int, step_idx: int
    ) -> bool:
        action = self._safe_get(task, 'actions', step_idx, None)
        if action is None or int(action) == 0:
            return False
        if bool(self._safe_get(task, 'floor_transition', step_idx, False)):
            return False

        traj_pixel = self._safe_get(task, 'trajectory_pixel', step_idx, None)
        traj_world = self._safe_get(task, 'trajectory_world', step_idx, None)
        traj_status = self._safe_get(task, 'trajectory_status', step_idx, None)
        if (
            traj_pixel is not None
            and traj_status is not None
            and int(traj_status) == VISIBLE
            and getattr(self, "schema_version", 1) >= 3
            and bool(self._safe_get(
                task, 'trajectory_is_endpoint', step_idx, False
            ))
        ):
            return True

        pixels = self._safe_get(task, 'pixel_coords', step_idx, [])
        visibility = self._safe_get(task, 'visibility_status', step_idx, [])
        index = self._safe_get(
            task, 'expert_candidate_indices', step_idx, None
        )
        if index is None and getattr(self, "schema_version", 1) < 3:
            worlds = self._safe_get(task, 'world_coords', step_idx, [])
            index, distance = self._find_closest_candidate(
                worlds, pixels, traj_world, traj_pixel
            )
            if index is None or distance >= self._get_distance_threshold(
                traj_world is not None
            ):
                return False
        if index is None:
            return False
        index = int(index)
        return (
            0 <= index < len(pixels)
            and index < len(visibility)
            and visibility[index] is not None
            and int(visibility[index]) == VISIBLE
        )
    
    def _get_visible_frontiers(
        self, candidates_pixel: List, candidates_vis: List
    ) -> List[List[int]]:
        visible = []
        for i, pixel in enumerate(candidates_pixel):
            if pixel is not None:
                vis = candidates_vis[i] if i < len(candidates_vis) else None
                if vis is not None and int(vis) == VISIBLE:
                    visible.append(list(pixel))
        return visible
    
    def _get_history_image_paths(
        self, scene_id: str, task_id: str, task_idx: int, step_idx: int
    ) -> List[str]:
        valid_steps = self.task_valid_steps.get(task_idx, [])
        try:
            current_pos = valid_steps.index(step_idx)
        except ValueError:
            return []
        
        if current_pos == 0:
            return []  
        
        all_history_steps = valid_steps[:current_pos]
        max_hist = self.config.max_history_frames
        
        if len(all_history_steps) <= max_hist:
            selected_steps = all_history_steps
        else:
            indices = np.linspace(0, len(all_history_steps) - 1, max_hist, dtype=int)
            selected_steps = [all_history_steps[i] for i in indices]
        
        return [self._get_image_path(scene_id, task_id, s) for s in selected_steps]
    
    def _get_image_path(self, scene_id: str, task_id: str, step: int) -> str:
        ext = self.config.image_ext
        return os.path.join(
            self.data_root, scene_id, task_id, "rgb", f"step_{step:04d}.{ext}"
        )
    
    def _build_prompt(
        self,
        instruction: str,
        visible_frontiers: List[List[int]],
        history_paths: List[str],
    ) -> str:
        img_token = self.config.image_token
        
        # Select prompt template based on whether visible frontiers exist
        if visible_frontiers:
            prompt = self.config.prompt_template
            frontier_str = self.config.format_coord_list(visible_frontiers)
            prompt = prompt.replace("<frontiers_coord>", frontier_str)
        else:
            prompt = self.config.prompt_template_no_frontiers
        
        # Handle history
        if history_paths:
            history_str = "".join([img_token] * len(history_paths))
            prompt = prompt.replace("<history>", history_str)
        else:
            prompt = prompt.replace(
                "These are your historical observations: <history>. ", ""
            )
            prompt = prompt.replace("<history>", "")
        
        conjunction = self.rng.choice(self.config.conjunctions)
        
        prompt = prompt.replace("<instruction>", instruction)
        prompt = prompt.replace("<conjunction>", conjunction)
        prompt = prompt.replace("<image>", img_token)
        
        return prompt
    
    def _build_response(
        self, output_mode: str, label_data: Any
    ) -> Tuple[str, List[Tuple[int, int]]]:
        cfg = self.config
        
        if output_mode == "stop":
            response = "STOP"
            trainable_ranges = [(0, len(response))]
            return response, trainable_ranges
        
        if output_mode == "frontier":
            prefix = cfg.frontier_prefix
            coord = label_data  # [u, v]
            content = " " + cfg.format_coord(coord[0], coord[1])
        
        elif output_mode == "target":
            prefix = cfg.target_prefix
            coord = label_data  # [u, v]
            content = " " + cfg.format_coord(coord[0], coord[1])
        
        elif output_mode == "action":
            prefix = cfg.action_prefix
            action_strs = [cfg.get_action_display(a) for a in label_data]
            content = " " + " ".join(action_strs)
        
        else:
            prefix = ""
            content = "STOP"
        
        response = prefix + content
        
        trainable_ranges = []
        if cfg.prefix_trainable:
            trainable_ranges = [(0, len(response))]
        else:
            trainable_ranges = [(len(prefix), len(response))]
        
        return response, trainable_ranges
    
    
    @staticmethod
    def _safe_get(task: Dict, key: str, index: int, default=None):
        arr = task.get(key, [])
        if index < len(arr):
            return arr[index] if arr[index] is not None else default
        return default
    
    def get_output_mode_stats(self) -> Dict[str, int]:
        stats = {"frontier": 0, "target": 0, "action": 0, "stop": 0}
        for idx in range(len(self)):
            sample = self[idx]
            mode = sample['output_mode']
            stats[mode] = stats.get(mode, 0) + 1
        return stats


def vln_worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if isinstance(dataset, VLNNavDataset) and hasattr(dataset, 'reset_lmdb'):
            dataset.reset_lmdb()
