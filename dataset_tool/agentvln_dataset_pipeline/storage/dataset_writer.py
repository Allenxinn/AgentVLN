import os
import json
import gzip
import numpy as np
from typing import Dict, List, Any, Optional

from .lmdb_image_store import LMDBImageStore


class NumpyEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DatasetWriter:
    VISIBILITY_MAP = {
        "VISIBLE": 0,
        "BEHIND": 1,
        "LEFT": 2,
        "RIGHT": 3,
        "ABOVE": 4,
        "BELOW": 5
    }
    
    def __init__(self, output_base_path,
                 generation_config = None,
                 image_formats=None, jpg_quality = 95,
                 lmdb_map_size=1024 * 1024 * 1024 * 100):
        self.output_base_path = output_base_path
        os.makedirs(output_base_path, exist_ok=True)
        self.image_store = LMDBImageStore(
            output_base_path,
            map_size=lmdb_map_size,
            image_formats=image_formats,
            jpg_quality=jpg_quality,
        )
        
        # Data accumulator for current dataset
        self.generation_config = dict(generation_config or {})
        self.verbose_action_debug = bool(
            self.generation_config.get("actuation_noise_verbose", False)
        )
        self.dataset = {
            "schema_version": 3,
            "generation_config": self.generation_config,
            "tasks": [],
            "visibility_map": self.VISIBILITY_MAP
        }
    
    def commit_sample(self, task_data, scene_id, task_id, sample, action_debug):
        self.add_step_data(
            task_data,
            sample.step,
            exploration_targets=sample.exploration_targets,
            trajectory_target=sample.trajectory_target,
            action=sample.action_code,
            floor_id=sample.floor_id,
            floor_height=sample.floor_height,
            floor_transition=sample.floor_transition,
            expert_candidate_index=sample.expert_candidate_index,
            trajectory_goal_distance=sample.trajectory_goal_distance,
            trajectory_is_endpoint=sample.trajectory_is_endpoint,
            action_debug=action_debug,
        )
        self.image_store.save_rgb(sample.rgb, scene_id, task_id, sample.step)
        self.image_store.save_topdown(sample.topdown, scene_id, task_id, sample.step)
    
    def create_task_data(
        self,
        scene_id,
        task_id,
        instruction,
        total_steps,
        goal_world=None,
    ) -> Dict[str, Any]:
        task_data = {
            "scene_id": scene_id,
            "task_id": task_id,
            "instruction": instruction,
            "total_steps": total_steps,
            "goal_world": (
                np.asarray(goal_world, dtype=float).tolist()
                if goal_world is not None else None
            ),
            "termination_reason": None,
            "success": False,
            "final_geodesic_distance": None,
            "topdown_coords": [],      # List[List[List[int]]] - per step, per target
            "pixel_coords": [],        # List[List[List[int]]]
            "world_coords": [],        # List[List[List[float]]]
            "visibility_status": [],   # List[List[int]]
            "history_info": [],        # List[List[Optional[Dict]]]
            "trajectory_pixel": [],    # List[Optional[List[int]]]
            "trajectory_world": [],    # List[Optional[List[float]]] - trajectory target world coords
            "trajectory_status": [],   # List[Optional[int]]
            "trajectory_goal_distances": [],
            "trajectory_is_endpoint": [],
            "actions": [],             # List[int] - 0:STOP, 1:FORWARD, 2:LEFT, 3:RIGHT
            "floor_ids": [],           # List[Optional[int]]
            "floor_heights": [],       # List[Optional[float]]
            "floor_transition": [],    # List[bool]
            "expert_candidate_indices": [],  # List[Optional[int]]
            "floors": [],              # Discovered floor metadata
        }
        if self.verbose_action_debug:
            task_data["action_debug"] = []
        return task_data
    
    def add_step_data(
        self,
        task_data,
        step,
        exploration_targets = None,
        trajectory_target = None,
        action = None,
        floor_id = None,
        floor_height = None,
        floor_transition = False,
        expert_candidate_index = None,
        trajectory_goal_distance = None,
        trajectory_is_endpoint = False,
        action_debug = None,
    ):
        if step < 0:
            raise ValueError("step must be >= 0")
        if len(task_data["actions"]) > step:
            raise ValueError(f"step {step} has already been saved")
        if action is None or int(action) not in (0, 1, 2, 3):
            raise ValueError("action must be one of 0/1/2/3")
        if floor_transition:
            exploration_targets = []
            trajectory_target = None
            expert_candidate_index = None
            trajectory_goal_distance = None
            trajectory_is_endpoint = False

        # Ensure arrays are long enough (pad with empty if needed)
        while len(task_data["topdown_coords"]) < step:
            task_data["topdown_coords"].append([])
            task_data["pixel_coords"].append([])
            task_data["world_coords"].append([])
            task_data["visibility_status"].append([])
            task_data["history_info"].append([])
            task_data["trajectory_pixel"].append(None)
            task_data["trajectory_world"].append(None)
            task_data["trajectory_status"].append(None)
            task_data["trajectory_goal_distances"].append(None)
            task_data["trajectory_is_endpoint"].append(False)
            task_data["actions"].append(None)
            task_data["floor_ids"].append(None)
            task_data["floor_heights"].append(None)
            task_data["floor_transition"].append(False)
            task_data["expert_candidate_indices"].append(None)
            if "action_debug" in task_data:
                task_data["action_debug"].append(None)
        
        if exploration_targets:
            step_topdown = []
            step_pixel = []
            step_world = []
            step_status = []
            step_history = []
            
            for target in exploration_targets:
                step_topdown.append(target.get("topdown_coords"))
                step_pixel.append(target.get("pixel_coords"))
                step_world.append(target.get("world_coords"))
                step_status.append(target.get("visibility_status", 0))
                step_history.append(target.get("history_info"))
            
            task_data["topdown_coords"].append(step_topdown)
            task_data["pixel_coords"].append(step_pixel)
            task_data["world_coords"].append(step_world)
            task_data["visibility_status"].append(step_status)
            task_data["history_info"].append(step_history)
        else:
            task_data["topdown_coords"].append([])
            task_data["pixel_coords"].append([])
            task_data["world_coords"].append([])
            task_data["visibility_status"].append([])
            task_data["history_info"].append([])
        
        if trajectory_target:
            task_data["trajectory_pixel"].append(trajectory_target.get("pixel_coords"))
            task_data["trajectory_world"].append(trajectory_target.get("world_coords"))
            task_data["trajectory_status"].append(trajectory_target.get("visibility_status"))
        else:
            task_data["trajectory_pixel"].append(None)
            task_data["trajectory_world"].append(None)
            task_data["trajectory_status"].append(None)
        task_data["trajectory_goal_distances"].append(
            None if trajectory_goal_distance is None
            else float(trajectory_goal_distance)
        )
        task_data["trajectory_is_endpoint"].append(
            bool(trajectory_is_endpoint)
        )
        
        task_data["actions"].append(int(action))
        task_data["floor_ids"].append(
            None if floor_transition or floor_id is None else int(floor_id)
        )
        task_data["floor_heights"].append(
            None
            if floor_transition or floor_height is None
            else float(floor_height)
        )
        task_data["floor_transition"].append(bool(floor_transition))
        task_data["expert_candidate_indices"].append(
            None
            if floor_transition or expert_candidate_index is None
            else int(expert_candidate_index)
        )
        if "action_debug" in task_data:
            if action_debug is None:
                raise ValueError("verbose schema-v3 samples require action_debug")
            task_data["action_debug"].append(dict(action_debug))

    def set_task_floors(self, task_data, floors):
        task_data["floors"] = [
            {"floor_id": int(item["floor_id"]), "height": float(item["height"])}
            for item in floors
        ]

    def finalize_task(
        self,
        task_data,
        termination_reason,
        final_geodesic_distance,
    ):
        task_data["termination_reason"] = str(termination_reason)
        task_data["success"] = termination_reason == "reached_goal"
        task_data["final_geodesic_distance"] = (
            float(final_geodesic_distance)
            if final_geodesic_distance is not None
            and np.isfinite(final_geodesic_distance)
            else None
        )
    
    def add_task(self, task_data):
        """Add completed task data to dataset."""
        self.dataset["tasks"].append(task_data)
    
    def save_dataset(self, filename = "exploration_data.json", compress = False):
        from ..core.dataset_contract import validate_dataset_contract

        validate_dataset_contract(self.dataset, self.generation_config)
        filepath = os.path.join(self.output_base_path, filename)
        
        if compress:
            filepath += ".gz"
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(self.dataset, f, separators=(',', ':'), cls=NumpyEncoder)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, separators=(',', ':'), cls=NumpyEncoder)
        
        print(f"Saved dataset to {filepath}")
        return filepath
    
    def load_dataset(self, filepath) -> Dict[str, Any]:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def update_task_total_steps(self, task_data, total_steps):
        task_data["total_steps"] = total_steps
    
    def clear(self):
        self.dataset = {
            "schema_version": 3,
            "generation_config": self.generation_config,
            "tasks": [],
            "visibility_map": self.VISIBILITY_MAP
        }

    def close(self):
        self.image_store.close()
