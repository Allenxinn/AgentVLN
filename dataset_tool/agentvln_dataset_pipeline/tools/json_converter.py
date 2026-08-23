import json
import gzip
import argparse
import os
from typing import Dict, List, Any, Optional


# Reverse visibility map
VISIBILITY_NAMES = {
    0: "VISIBLE",
    1: "BEHIND",
    2: "LEFT",
    3: "RIGHT",
    4: "ABOVE",
    5: "BELOW"
}

# Action names map
ACTION_NAMES = {
    0: "STOP",
    1: "FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT"
}


class JsonConverter:
    """
    Converts between compact columnar JSON and human-readable format.
    """
    
    @staticmethod
    def load_compact(filepath) -> Dict[str, Any]:
        """Load compact JSON file."""
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    @staticmethod
    def save_readable(data, filepath, indent = 2):
        """Save data as human-readable JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        print(f"Saved readable JSON to {filepath}")
    
    @staticmethod
    def convert_task_to_readable(task_data) -> Dict[str, Any]:
        """
        Convert a single task from compact columnar format to readable nested format.
        
        Args:
            task_data: Task data in compact format
            
        Returns:
            Task data in readable format
        """
        readable = {
            "schema_version": 3 if "goal_world" in task_data else (
                2 if "floor_ids" in task_data else 1
            ),
            "scene_id": task_data["scene_id"],
            "task_id": task_data["task_id"],
            "instruction": task_data.get("instruction", ""),
            "total_steps": task_data.get("total_steps", 0),
            "floors": task_data.get("floors", []),
            "goal_world": task_data.get("goal_world"),
            "termination_reason": task_data.get("termination_reason"),
            "success": task_data.get("success"),
            "final_geodesic_distance": task_data.get(
                "final_geodesic_distance"
            ),
            "steps": []
        }
        
        # Get arrays
        topdown_coords = task_data.get("topdown_coords", [])
        pixel_coords = task_data.get("pixel_coords", [])
        world_coords = task_data.get("world_coords", [])
        visibility_status = task_data.get("visibility_status", [])
        history_info = task_data.get("history_info", [])
        trajectory_pixel = task_data.get("trajectory_pixel", [])
        trajectory_status = task_data.get("trajectory_status", [])
        actions = task_data.get("actions", [])
        floor_ids = task_data.get("floor_ids", [])
        floor_heights = task_data.get("floor_heights", [])
        floor_transition = task_data.get("floor_transition", [])
        expert_candidate_indices = task_data.get("expert_candidate_indices", [])
        trajectory_goal_distances = task_data.get(
            "trajectory_goal_distances", []
        )
        trajectory_is_endpoint = task_data.get("trajectory_is_endpoint", [])
        action_debug = task_data.get("action_debug")
        
        # Determine number of steps from data
        num_steps = max(
            len(topdown_coords),
            len(pixel_coords),
            len(trajectory_pixel),
            task_data.get("total_steps", 0)
        )
        
        for step in range(num_steps):
            step_data = {"step": step}
            step_data["floor_id"] = (
                floor_ids[step] if step < len(floor_ids) else None
            )
            step_data["floor_height"] = (
                floor_heights[step] if step < len(floor_heights) else None
            )
            step_data["floor_transition"] = bool(
                floor_transition[step]
                if step < len(floor_transition)
                else False
            )
            step_data["expert_candidate_index"] = (
                expert_candidate_indices[step]
                if step < len(expert_candidate_indices)
                else None
            )
            step_data["trajectory_goal_distance"] = (
                trajectory_goal_distances[step]
                if step < len(trajectory_goal_distances) else None
            )
            step_data["trajectory_is_endpoint"] = bool(
                trajectory_is_endpoint[step]
                if step < len(trajectory_is_endpoint) else False
            )
            
            # Build exploration targets
            exploration_targets = []
            if step < len(topdown_coords):
                step_topdown = topdown_coords[step] if topdown_coords[step] else []
                step_pixel = pixel_coords[step] if step < len(pixel_coords) and pixel_coords[step] else []
                step_world = world_coords[step] if step < len(world_coords) and world_coords[step] else []
                step_status = visibility_status[step] if step < len(visibility_status) and visibility_status[step] else []
                step_history = history_info[step] if step < len(history_info) and history_info[step] else []
                
                num_targets = len(step_topdown)
                for i in range(num_targets):
                    target = {
                        "topdown_coords": step_topdown[i] if i < len(step_topdown) else None,
                        "pixel_coords": step_pixel[i] if i < len(step_pixel) else None,
                        "world_coords": step_world[i] if i < len(step_world) else None,
                        "visibility_status": VISIBILITY_NAMES.get(
                            step_status[i] if i < len(step_status) else 0, 
                            "UNKNOWN"
                        ),
                        "history_info": step_history[i] if i < len(step_history) else None
                    }
                    exploration_targets.append(target)
            
            step_data["exploration_targets"] = exploration_targets
            
            # Build trajectory target
            traj_pixel = trajectory_pixel[step] if step < len(trajectory_pixel) else None
            trajectory_world = task_data.get("trajectory_world", [])
            traj_world = (
                trajectory_world[step] if step < len(trajectory_world) else None
            )
            traj_status = trajectory_status[step] if step < len(trajectory_status) else None
            
            if traj_pixel is not None or traj_world is not None or traj_status is not None:
                step_data["trajectory_target"] = {
                    "pixel_coords": traj_pixel,
                    "world_coords": traj_world,
                    "visibility_status": VISIBILITY_NAMES.get(traj_status, "UNKNOWN") if traj_status is not None else None
                }
            else:
                step_data["trajectory_target"] = None
            
            # Add action
            if step < len(actions) and actions[step] is not None:
                step_data["action"] = {
                    "code": actions[step],
                    "name": ACTION_NAMES.get(actions[step], "UNKNOWN")
                }
            else:
                step_data["action"] = None
            if action_debug is not None:
                step_data["action_debug"] = (
                    action_debug[step] if step < len(action_debug) else None
                )
            
            readable["steps"].append(step_data)
        
        return readable
    
    @classmethod
    def extract_by_scene(
        cls, 
        compact_data, 
        scene_id
    ) -> List[Dict[str, Any]]:
        """
        Extract all tasks for a given scene and convert to readable format.
        
        Args:
            compact_data: Full compact dataset
            scene_id: Scene ID to filter by
            
        Returns:
            List of readable task dictionaries
        """
        results = []
        for task in compact_data.get("tasks", []):
            if task.get("scene_id") == scene_id:
                results.append(cls.convert_task_to_readable(task))
        return results
    
    @classmethod
    def extract_by_task(
        cls, 
        compact_data, 
        task_id
    ) -> Optional[Dict[str, Any]]:
        """
        Extract a specific task and convert to readable format.
        
        Args:
            compact_data: Full compact dataset
            task_id: Task ID to find
            
        Returns:
            Readable task dictionary or None if not found
        """
        for task in compact_data.get("tasks", []):
            if task.get("task_id") == task_id:
                return cls.convert_task_to_readable(task)
        return None
    
    @classmethod
    def convert_full_dataset(
        cls, 
        compact_data
    ) -> Dict[str, Any]:
        """
        Convert entire dataset to readable format.
        
        Args:
            compact_data: Full compact dataset
            
        Returns:
            Full readable dataset
        """
        readable = {
            "schema_version": compact_data.get("schema_version", 1),
            "generation_config": compact_data.get("generation_config"),
            "tasks": [],
            "visibility_map": compact_data.get("visibility_map", {})
        }
        
        for task in compact_data.get("tasks", []):
            readable["tasks"].append(cls.convert_task_to_readable(task))
        
        return readable
    
    @classmethod
    def list_scenes(cls, compact_data) -> List[str]:
        """List all unique scene IDs in the dataset."""
        scenes = set()
        for task in compact_data.get("tasks", []):
            scenes.add(task.get("scene_id", ""))
        return sorted(scenes)
    
    @classmethod
    def list_tasks(cls, compact_data, scene_id = None) -> List[str]:
        """List all task IDs, optionally filtered by scene."""
        tasks = []
        for task in compact_data.get("tasks", []):
            if scene_id is None or task.get("scene_id") == scene_id:
                tasks.append(task.get("task_id", ""))
        return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Convert compact JSON to human-readable format"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Input compact JSON file path"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output readable JSON file path"
    )
    parser.add_argument(
        "--scene",
        help="Filter by scene ID (extracts all tasks for the scene)"
    )
    parser.add_argument(
        "--task",
        help="Filter by task ID (extracts single task)"
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List all scene IDs in the dataset"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all task IDs (use with --scene to filter)"
    )
    
    args = parser.parse_args()
    
    # Load compact data
    converter = JsonConverter()
    compact_data = converter.load_compact(args.input)
    
    # List scenes
    if args.list_scenes:
        scenes = converter.list_scenes(compact_data)
        print(f"Found {len(scenes)} scenes:")
        for scene in scenes:
            print(f"  {scene}")
        return
    
    # List tasks
    if args.list_tasks:
        tasks = converter.list_tasks(compact_data, args.scene)
        print(f"Found {len(tasks)} tasks:")
        for task in tasks:
            print(f"  {task}")
        return
    
    # Extract and convert
    if args.task:
        result = converter.extract_by_task(compact_data, args.task)
        if result is None:
            print(f"Task '{args.task}' not found")
            return
    elif args.scene:
        result = converter.extract_by_scene(compact_data, args.scene)
        if not result:
            print(f"No tasks found for scene '{args.scene}'")
            return
        result = {"scene_id": args.scene, "tasks": result}
    else:
        result = converter.convert_full_dataset(compact_data)
    
    # Save or print
    if args.output:
        converter.save_readable(result, args.output)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
