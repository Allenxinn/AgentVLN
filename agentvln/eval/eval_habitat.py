import os
import sys
import yaml
import json
import time
import argparse
import logging
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agentvln.eval.navigation_metrics import (
    EpisodeResult,
    NavigationMetrics,
    compute_ndtw,
    is_rxr_validation,
    load_rxr_ground_truth,
    validate_episode_ground_truth,
)

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    from agentvln.eval.habitat_utils.generation_utils import (
        get_actuation_noise_settings,
        validate_generation_settings,
    )

    validation_config = dict(config)
    validation_config.setdefault("trajectory", {"sample_interval": 0.25})
    validation_config.setdefault("output", {"update_interval": 1})
    validation_config["trajectory"].setdefault("sample_interval", 0.25)
    validation_config["output"].setdefault("update_interval", 1)
    validate_generation_settings(validation_config)
    get_actuation_noise_settings({
        "generation": {
            "actuation_noise": config.get("evaluation", {}).get(
                "actuation_noise", {}
            )
        }
    })
    return config



class ResponseParser:
    COORD_PATTERN = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
    TEXT_ACTIONS = {"FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3, "STOP": 0}
    SYMBOL_ACTIONS = {"↑": 1, "←": 2, "→": 3, "⏹": 0}

    ACTION_MAP = None 

    def __init__(self, nav_config):
        self.frontier_prefix = nav_config.get('frontier_prefix', '<frontiers_coord>')
        self.target_prefix = nav_config.get('target_prefix', '<target>')
        self.action_prefix = nav_config.get('action_prefix', '<action>')
        self.use_symbols = nav_config.get('use_action_symbols', False)

    def parse(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        result = {
            "task_type": "unknown",
            "coordinate": None,
            "action_sequence": None,
            "raw_text": text,
            "parse_success": False,
        }

        if text.upper() == "STOP" or text.strip() == "⏹":
            result["task_type"] = "stop"
            result["parse_success"] = True
            return result

        if text.startswith(self.frontier_prefix):
            result["task_type"] = "frontier"
            remainder = text[len(self.frontier_prefix):]
            coord = self._extract_first_coord(remainder)
            if coord is not None:
                result["coordinate"] = coord
                result["parse_success"] = True

        elif text.startswith(self.target_prefix):
            result["task_type"] = "target"
            remainder = text[len(self.target_prefix):]
            coord = self._extract_first_coord(remainder)
            if coord is not None:
                result["coordinate"] = coord
                result["parse_success"] = True

        elif text.startswith(self.action_prefix):
            result["task_type"] = "action"
            remainder = text[len(self.action_prefix):].strip()
            actions = self._extract_actions(remainder)
            if actions:
                result["action_sequence"] = actions
                result["parse_success"] = True

        else:
            result = self._try_auto_detect(text, result)

        return result

    def _extract_first_coord(self, text: str) -> Optional[List[float]]:
        match = self.COORD_PATTERN.search(text)
        if match:
            return [float(match.group(1)), float(match.group(2))]
        return None

    def _extract_actions(self, text: str) -> List[int]:
        actions = []
        action_map = self.SYMBOL_ACTIONS if self.use_symbols else self.TEXT_ACTIONS
        tokens = text.split()
        for token in tokens:
            token_clean = token.strip().rstrip(",").rstrip(";")
            if token_clean in action_map:
                actions.append(action_map[token_clean])
        return actions

    def _try_auto_detect(self, text: str, result: Dict) -> Dict:
        coord = self._extract_first_coord(text)
        if coord is not None:
            result["coordinate"] = coord
            result["parse_success"] = True
            return result

        actions = self._extract_actions(text)
        if actions:
            result["task_type"] = "action"
            result["action_sequence"] = actions
            result["parse_success"] = True
            return result

        if "STOP" in text.upper() or "⏹" in text:
            result["task_type"] = "stop"
            result["parse_success"] = True
            return result

        return result


class PromptBuilder:
    def __init__(self, nav_config: Dict):
        self.nav_config = nav_config
        self.image_token = "<image>"

        self.prompt_template = (
            "You are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Where should you go next to stay on track? "
            "Select the most suitable waypoint coordinate from the following <frontiers_coord>. "
            "If no suitable coordinate is found, output the action command. "
            "If the destination is visible, output the coordinates of the target point directly. "
            "Please output STOP when you have successfully completed the task. "
            "These are your historical observations: <history>. "
            "<conjunction><image>."
        )

        self.prompt_template_no_frontiers = (
            "You are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Where should you go next to stay on track? "
            "Enter the action command directly. "
            "If you can see the endpoint, enter the coordinates of the target point. "
            "Please output STOP when you have successfully completed the task. "
            "These are your historical observations: <history>. "
            "<conjunction><image>."
        )

        self.conjunctions = [
            "Now observe: ",
            "Current view: ",
            "Looking ahead: ",
            "What you see now: ",
            "Your current observation: ",
        ]

        import random
        self.rng = random.Random(42)

    def build_prompt(
        self,
        instruction: str,
        visible_frontiers: List[List[int]],
        history_images: List,  
    ) -> str:
        img_token = self.image_token

        if visible_frontiers:
            prompt = self.prompt_template
            frontier_str = self._format_coord_list(visible_frontiers)
            prompt = prompt.replace("<frontiers_coord>", frontier_str)
        else:
            prompt = self.prompt_template_no_frontiers

        if history_images:
            history_str = "".join([img_token] * len(history_images))
            prompt = prompt.replace("<history>", history_str)
        else:
            prompt = prompt.replace(
                "These are your historical observations: <history>. ", ""
            )
            prompt = prompt.replace("<history>", "")

        conjunction = self.rng.choice(self.conjunctions)

        prompt = prompt.replace("<instruction>", instruction)
        prompt = prompt.replace("<conjunction>", conjunction)
        prompt = prompt.replace("<image>", img_token)

        return prompt

    def _format_coord(self, u: int, v: int) -> str:
        return f"({u}, {v})"

    def _format_coord_list(self, coords: List[List[int]]) -> str:
        formatted = [self._format_coord(c[0], c[1]) for c in coords if c is not None]
        return "[" + ", ".join(formatted) + "]"


class ModelInference:
    def __init__(self, config: Dict):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_config = config['model']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(model_config.get('torch_dtype', 'bfloat16'), torch.bfloat16)

        logger.info(f"Loading model: {model_config['model_name_or_path']}")
        self.processor = AutoProcessor.from_pretrained(
            model_config['model_name_or_path'],
            trust_remote_code=model_config.get('trust_remote_code', True),
            min_pixels=model_config.get('image_min_pixels', 256 * 28 * 28),
            max_pixels=model_config.get('image_max_pixels', 1280 * 28 * 28),
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_config['model_name_or_path'],
            torch_dtype=torch_dtype,
            trust_remote_code=model_config.get('trust_remote_code', True),
            attn_implementation=model_config.get('attn_implementation', 'flash_attention_2'),
        ).to(self.device)
        self.model.eval()

        self.max_new_tokens = model_config.get('max_new_tokens', 128)
        logger.info(f"Model loaded on {self.device}")

    def generate(self, prompt: str, images: List) -> str:
        import torch
        from PIL import Image
        content_parts = []
        image_token = "<image>"
        img_idx = 0
        segments = prompt.split(image_token)
        for i, segment in enumerate(segments):
            if segment:
                content_parts.append({"type": "text", "text": segment})
            if i < len(segments) - 1 and images and img_idx < len(images):
                content_parts.append({"type": "image", "image": images[img_idx]})
                img_idx += 1

        messages = [{"role": "user", "content": content_parts}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=False,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        decoded = self.processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )

        return decoded[0].strip() if decoded else ""


def setup_habitat_env(
    config: Dict[str, Any], config_source: Optional[str] = None
):
    from agentvln.habitat_extensions import measures as _measures  # noqa: F401
    from habitat import Env

    from habitat_baselines.config.default import get_config
    from agentvln.eval.habitat_utils.config_utils import (
        apply_exploration_overrides,
        print_override_report,
    )

    dataset_type = str(config['dataset']['type'])
    is_rxr = dataset_type.casefold() in {"rxr", "rxr-vln-ce-v1"}
    if is_rxr:
        try:
            __import__("habitat_extensions.task")
        except ImportError as import_error:
            try:
                from habitat.core.registry import registry
                rxr_dataset = registry.get_dataset("RxR-VLN-CE-v1")
            except Exception:
                rxr_dataset = None
            if rxr_dataset is None:
                raise RuntimeError(
                    "RxR evaluation requires the VLN-CE dataset registry entry "
                    "'RxR-VLN-CE-v1' (normally provided by "
                    "habitat_extensions.task)"
                ) from import_error

    default_habitat_config = os.path.join(
        _REPO_ROOT, "agentvln", "configs", "vln_r2r.yaml"
    )
    exp_config = config.get('habitat_config_path', default_habitat_config)
    if not os.path.isabs(exp_config):
        config_dir = (
            os.path.dirname(os.path.abspath(config_source))
            if config_source else os.getcwd()
        )
        exp_config = os.path.abspath(os.path.join(config_dir, exp_config))

    effective_config = dict(config)
    effective_dataset = dict(config.get("dataset", {}))
    config_dir = (
        os.path.dirname(os.path.abspath(config_source))
        if config_source else os.getcwd()
    )
    for key in ("data_path", "scenes_dir"):
        value = effective_dataset.get(key)
        if isinstance(value, str) and not os.path.isabs(value):
            effective_dataset[key] = os.path.abspath(os.path.join(config_dir, value))
    effective_config["dataset"] = effective_dataset

    habitat_config = get_config(exp_config)
    overrides = apply_exploration_overrides(habitat_config, effective_config)
    print(f"[Config] Habitat base config: {exp_config}")
    print_override_report(
        overrides,
        os.path.basename(config_source) if config_source else "evaluation config",
    )
    try:
        env = Env(habitat_config)
    except Exception as exc:
        if is_rxr:
            raise RuntimeError(
                "Failed to create the RxR Habitat environment. Verify that "
                "'RxR-VLN-CE-v1' is registered and dataset paths/roles/languages "
                "match the installed VLN-CE extension."
            ) from exc
        raise
    return env, habitat_config


def world_to_map_coords(world_pos: np.ndarray, sim, map_shape: tuple) -> tuple:
    from habitat.utils.visualizations import maps
    row, col = maps.to_grid(
        world_pos[2],
        world_pos[0],
        map_shape[:2],
        sim=sim,
    )
    return (row, col)


def world_path_for_floor(world_points, floor_height, tolerance, sim, map_shape):
    if floor_height is None:
        return []
    return [
        world_to_map_coords(point, sim, map_shape)
        for point in world_points
        if abs(float(point[1]) - float(floor_height)) <= float(tolerance)
    ]


def _try_inverse_projection(
    pixel_u: float,
    pixel_v: float,
    agent_state,
    coord_transformer,
    map_builder,
) -> Optional[np.ndarray]:
    CAMERA_HEIGHT_OFFSET = 1.25
    # Maximum acceptable horizontal distance when snapping to navmesh
    SNAP_TOLERANCE = 3.0

    try:
        import magnum as mn

        if not hasattr(coord_transformer, 'fx'):
            return None

        fx = coord_transformer.fx
        fy = coord_transformer.fy
        cx = coord_transformer.cx
        cy = coord_transformer.cy

        cam_x = (pixel_u - cx) / fx
        cam_y = (pixel_v - cy) / fy
        cam_z = 1.0

        local_dir = np.array([cam_x, -cam_y, -cam_z])
        local_dir = local_dir / np.linalg.norm(local_dir)

        from habitat.utils.geometry_utils import quaternion_rotate_vector
        ray_world = quaternion_rotate_vector(agent_state.rotation, local_dir)

        pos = np.array(agent_state.position)

        if abs(ray_world[1]) > 1e-6:
            t = -CAMERA_HEIGHT_OFFSET / ray_world[1]
            if t <= 0:
                t = 5.0
            else:
                t = min(t, 20.0)  
        else:
            t = 5.0

        floor_point = pos + t * ray_world

        nav_point = map_builder.sim.pathfinder.snap_point(
            mn.Vector3(floor_point)
        )
        if not np.isnan(nav_point).any():
            horiz_dist = np.sqrt(
                (nav_point[0] - floor_point[0]) ** 2
                + (nav_point[2] - floor_point[2]) ** 2
            )
            if horiz_dist < SNAP_TOLERANCE:
                return np.array(nav_point)

        for t_fb in [2.0, 4.0, 6.0, 8.0]:
            candidate = pos + t_fb * ray_world
            nav_point = map_builder.sim.pathfinder.snap_point(
                mn.Vector3(candidate)
            )
            if not np.isnan(nav_point).any():
                horiz_dist = np.sqrt(
                    (nav_point[0] - candidate[0]) ** 2
                    + (nav_point[2] - candidate[2]) ** 2
                )
                if horiz_dist < SNAP_TOLERANCE:
                    return np.array(nav_point)

    except Exception as e:
        logger.debug(
            f"Inverse projection failed for pixel ({pixel_u}, {pixel_v}): {e}"
        )

    return None


def pixel_to_world_via_map(
    pixel_coord: List[float],
    agent_state,
    coord_transformer,
    map_builder,
    exploration_targets: List[Dict],
) -> Optional[np.ndarray]:
    pred = np.array(pixel_coord, dtype=float)
    PIXEL_MATCH_THRESHOLD = 10.0
    NEIGHBOR_SEARCH_RADIUS = 3

    best_dist = float('inf')
    best_world = None
    for target in exploration_targets:
        if target['pixel_coords'] is None:
            continue
        if target.get('visibility_status', -1) != 0:  # Not visible
            continue
        tgt_px = np.array(target['pixel_coords'], dtype=float)
        dist = np.linalg.norm(pred - tgt_px)
        if dist < best_dist:
            best_dist = dist
            best_world = np.array(target['world_coords'])

    if best_world is not None and best_dist < PIXEL_MATCH_THRESHOLD:
        return best_world

    result = _try_inverse_projection(
        pixel_coord[0], pixel_coord[1],
        agent_state, coord_transformer, map_builder,
    )
    if result is not None:
        return result

    neighbor_offsets = []
    for du in range(-NEIGHBOR_SEARCH_RADIUS, NEIGHBOR_SEARCH_RADIUS + 1):
        for dv in range(-NEIGHBOR_SEARCH_RADIUS, NEIGHBOR_SEARCH_RADIUS + 1):
            if du == 0 and dv == 0:
                continue 
            r = np.sqrt(du * du + dv * dv)
            if r <= NEIGHBOR_SEARCH_RADIUS:
                neighbor_offsets.append((r, du, dv))
    neighbor_offsets.sort(key=lambda x: x[0])  

    for _, du, dv in neighbor_offsets:
        neighbor_u = pixel_coord[0] + du
        neighbor_v = pixel_coord[1] + dv
        result = _try_inverse_projection(
            neighbor_u, neighbor_v,
            agent_state, coord_transformer, map_builder,
        )
        if result is not None:
            logger.info(
                f"Pixel ({pixel_coord[0]}, {pixel_coord[1]}) failed, "
                f"but neighbor ({neighbor_u}, {neighbor_v}) succeeded "
                f"(offset=({du}, {dv}))"
            )
            return result

    if best_world is not None:
        return best_world

    return None


def plan_path_to_target(sim, target_world: np.ndarray, goal_radius: float = 0.5) -> List[int]:
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    follower = ShortestPathFollower(sim, goal_radius=goal_radius, return_one_hot=False)

    actions = []
    max_actions = 50  

    for _ in range(max_actions):
        action = follower.get_next_action(target_world)
        if action is None or action == 0:  # STOP or no action
            break
        actions.append(int(action))

    return actions


def plan_full_path_to_target(sim, target_world: np.ndarray, goal_radius: float = 0.5) -> List[int]:
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    follower = ShortestPathFollower(sim, goal_radius=goal_radius, return_one_hot=False)
    actions = []
    max_actions = 50

    for _ in range(max_actions):
        action = follower.get_next_action(target_world)
        if action is None or action == 0:
            break
        actions.append(int(action))
        break

    return actions


class HistoryManager:
    def __init__(self, max_frames: int = 8):
        self.max_frames = max_frames
        self.history = []  # List of PIL images

    def add_frame(self, rgb_image):
        from PIL import Image
        if isinstance(rgb_image, np.ndarray):
            img = Image.fromarray(rgb_image)
        else:
            img = rgb_image
        self.history.append(img)

    def get_history_images(self) -> List:
        if len(self.history) <= self.max_frames:
            return list(self.history)
        else:
            indices = np.linspace(0, len(self.history) - 1,
                                  self.max_frames, dtype=int)
            return [self.history[i] for i in indices]

    def reset(self):
        self.history = []


class StopRule(ABC):
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, parsed: Dict[str, Any], world_coord: Optional[np.ndarray]) -> Optional[np.ndarray]:
        pass


class ConsecutiveTargetStopRule(StopRule):
    def __init__(
        self,
        enabled: bool = True,
        consecutive_count: int = 3,
        distance_threshold: float = 1.0,
    ):
        super().__init__(name="consecutive_target_stop", enabled=enabled)
        self.consecutive_count = consecutive_count
        self.distance_threshold = distance_threshold
        self._history: List[np.ndarray] = []  

    def reset(self):
        self._history = []

    def update(self, parsed: Dict[str, Any], world_coord: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self.enabled:
            return None

        if parsed.get("task_type") != "target":
            self._history = []
            return None

        if world_coord is None:
            self._history = []
            return None

        self._history.append(world_coord.copy())

        if len(self._history) < self.consecutive_count:
            return None

        self._history = self._history[-self.consecutive_count:]

        for i in range(len(self._history) - 1):
            dist = np.linalg.norm(self._history[i + 1] - self._history[i])
            if dist > self.distance_threshold:
                return None

        target = self._history[-1].copy()
        logger.info(
            f"ConsecutiveTargetStopRule triggered: {self.consecutive_count} "
            f"consecutive <target> predictions within {self.distance_threshold}m. "
            f"Stop target: {target}"
        )
        return target


class StopRuleManager:
    def __init__(self, stop_rules_config: Dict[str, Any]):
        self.rules: List[StopRule] = []
        self._build_rules(stop_rules_config)

    def _build_rules(self, cfg: Dict[str, Any]):
        ct_cfg = cfg.get("consecutive_target_stop", {})
        self.rules.append(ConsecutiveTargetStopRule(
            enabled=ct_cfg.get("enabled", True),
            consecutive_count=ct_cfg.get("consecutive_count", 3),
            distance_threshold=ct_cfg.get("distance_threshold", 1.0),
        ))

    def reset(self):
        for rule in self.rules:
            rule.reset()

    def update(
        self, parsed: Dict[str, Any], world_coord: Optional[np.ndarray]
    ) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        for rule in self.rules:
            result = rule.update(parsed, world_coord)
            if result is not None:
                return True, result, rule.name
        return False, None, None

    def get_active_rules(self) -> List[str]:
        return [r.name for r in self.rules if r.enabled]


def run_habitat_eval(args):
    import cv2
    from PIL import Image
    from habitat.utils.visualizations import maps
    from agentvln.eval.visualization import Visualizer

    from agentvln.eval.habitat_utils.topdown_map_builder import (
        convert_square_meters_to_pixel_area,
    )
    from agentvln.eval.habitat_utils.floor_map_manager import FloorMapManager
    from agentvln.eval.habitat_utils.candidate_utils import (
        build_visible_exploration_targets,
    )
    from agentvln.eval.habitat_utils.exploration_target_generator import (
        ExplorationTargetGenerator,
    )
    from agentvln.eval.habitat_utils.coordinate_transformer import (
        CoordinateTransformer,
        VisibilityStatus,
    )
    from agentvln.eval.habitat_utils.action_perturbation import (
        ActionExecutionError,
        PerturbedActionExecutor,
    )
    from agentvln.eval.habitat_utils.generation_utils import (
        get_actuation_noise_settings,
    )

    config = load_config(args.config)

    if args.debug:
        config['debug']['enabled'] = True
    if args.debug_episodes:
        config['debug']['max_episodes'] = args.debug_episodes
    if args.no_vis:
        config['debug']['visualization'] = False

    plan_interval = config['evaluation'].get('plan_interval', 4)
    max_steps = config['evaluation'].get('max_steps', 500)
    success_distance = config['evaluation'].get('success_distance', 3.0)
    nav_arrival_distance = config['evaluation'].get(
        'nav_arrival_distance', 0.25)  # step_length: arrival threshold
    enable_vis = config['debug'].get('visualization', True)
    output_base = config['output']['base_path']
    os.makedirs(output_base, exist_ok=True)

    print("Setting up Habitat environment...")
    env, habitat_config = setup_habitat_env(config, args.config)
    eval_noise = get_actuation_noise_settings({
        "generation": {
            "actuation_noise": config.get("evaluation", {}).get(
                "actuation_noise", {}
            )
        }
    })
    action_executor = PerturbedActionExecutor(
        env._sim,
        factor=eval_noise["factor"],
        seed=eval_noise["seed"],
        verbose=False,
    )

    episodes = env.episodes
    if config['debug']['enabled']:
        max_eps = config['debug'].get('max_episodes', 5)
        episodes = episodes[:max_eps]
        print(f"Debug mode: processing {len(episodes)} episodes")

    dataset_config = config.get("dataset", {})
    ndtw_enabled = is_rxr_validation(dataset_config)
    rxr_ground_truth = {}
    if ndtw_enabled:
        rxr_ground_truth = load_rxr_ground_truth(dataset_config, args.config)
        validate_episode_ground_truth(
            [episode.episode_id for episode in episodes], rxr_ground_truth
        )
        print(
            "RxR nDTW enabled: "
            f"loaded {len(rxr_ground_truth)} GT trajectories"
        )

    print("Loading model...")
    model_inference = ModelInference(config)

    nav_config = config.get('nav_config', {})
    prompt_builder = PromptBuilder(nav_config)
    response_parser = ResponseParser(nav_config)

    max_history = nav_config.get('max_history_frames', 8)
    history_manager = HistoryManager(max_frames=max_history)

    stop_rules_config = config.get('stop_rules', {})
    stop_rule_manager = StopRuleManager(stop_rules_config)
    active_rules = stop_rule_manager.get_active_rules()
    if active_rules:
        print(f"Active extra stop rules: {active_rules}")

    nav_metrics = NavigationMetrics(
        success_distance=success_distance,
        include_ndtw=ndtw_enabled,
    )

    visualizer = Visualizer()

    print(f"Processing {len(episodes)} episodes...")
    print(f"Plan interval: {plan_interval} steps")
    print(f"Success distance: {success_distance}m")
    print(f"Max steps: {max_steps}")
    print(
        "Evaluation actuation noise: "
        f"factor={eval_noise['factor']}, seed={eval_noise['seed']}"
    )

    all_episode_results = []

    for ep_idx, episode in enumerate(episodes):
        obs = env.reset()
        current_episode = env.current_episode

        episode_id = str(current_episode.episode_id)
        scene_id = current_episode.scene_id.split('/')[-1].replace('.glb', '')
        instruction = (
            current_episode.instruction.instruction_text
            if hasattr(current_episode.instruction, 'instruction_text')
            else str(current_episode.instruction)
        )

        print(f"\n[{ep_idx + 1}/{len(episodes)}] Episode: {episode_id}, Scene: {scene_id}")
        print(f"  Instruction: {instruction[:100]}...")

        goal_position = None
        if hasattr(current_episode, 'goals') and current_episode.goals:
            goal_position = np.array(current_episode.goals[0].position)
        elif hasattr(current_episode, 'reference_path') and current_episode.reference_path:
            goal_position = np.array(current_episode.reference_path[-1])
        else:
            print("  WARNING: No goal found, skipping episode")
            continue

        reference_path = []
        if hasattr(current_episode, 'reference_path') and current_episode.reference_path:
            reference_path = [np.array(p) for p in current_episode.reference_path]
            
        optimal_path_length = 0.0
        if reference_path:
            for i in range(len(reference_path) - 1):
                optimal_path_length += np.linalg.norm(
                    reference_path[i + 1] - reference_path[i]
                )
        else:
            start_pos = np.array(current_episode.start_position)
            optimal_path_length = env._sim.geodesic_distance(start_pos, goal_position)
            if optimal_path_length == float('inf') or np.isnan(optimal_path_length):
                optimal_path_length = np.linalg.norm(start_pos - goal_position)

        sim = env._sim
        floor_manager = FloorMapManager(
            sim,
            config['map'],
            initial_height=float(sim.get_agent_state().position[1]),
        )
        map_builder = floor_manager.active_builder
        target_generator = ExplorationTargetGenerator(config['exploration'])
        coord_transformer = CoordinateTransformer(config['camera'], sim)

        meters_per_pixel = maps.calculate_meters_per_pixel(
            config['map']['resolution'], sim=sim
        )
        target_generator.set_pixel_scale(meters_per_pixel)

        visualizer.reset_path()
        history_manager.reset()
        coord_transformer.reset_history()
        stop_rule_manager.reset()

        start_world = np.array(current_episode.start_position)
        start_map = world_to_map_coords(start_world, sim, map_builder.full_map.shape)
        end_map = world_to_map_coords(
            goal_position, sim, map_builder.full_map.shape
        ) if goal_position is not None else None

        episode_result = EpisodeResult(
            episode_id=episode_id,
            scene_id=scene_id,
            instruction=instruction,
            optimal_path_length=optimal_path_length,
        )

        step = 0
        path_length = 0.0
        prev_position = np.array(sim.get_agent_state().position)
        agent_trajectory = [prev_position.copy()]
        min_dist_to_goal = float('inf')
        traveled_paths = {0: []} 
        pending_actions = []  
        agent_stopped = False
        current_nav_target = None  
        stop_rule_target = None  
        stop_rule_triggered_name = None  
        last_model_step = -plan_interval  
        was_floor_transition = False

        ep_output_dir = os.path.join(output_base, scene_id, episode_id)
        if enable_vis:
            os.makedirs(ep_output_dir, exist_ok=True)

        initial_dist = np.linalg.norm(
            np.array(sim.get_agent_state().position) - goal_position
        )
        print(f"  Start distance to goal: {initial_dist:.2f}m")

        while step < max_steps and not env.episode_over and not agent_stopped:
            agent_state = sim.get_agent_state()
            current_pos = np.array(agent_state.position)

            step_dist = np.linalg.norm(current_pos - prev_position)
            path_length += step_dist
            prev_position = current_pos.copy()

            # Track min distance to goal (for Oracle SR)
            dist_to_goal = np.linalg.norm(current_pos - goal_position)
            min_dist_to_goal = min(min_dist_to_goal, dist_to_goal)

            # Track position for visualization
            floor_update = floor_manager.update(agent_state)
            map_builder = floor_manager.active_builder
            if floor_update.in_transition and not was_floor_transition:
                # A 2D coordinate target becomes ambiguous as soon as vertical
                # traversal begins; request action supervision/inference instead.
                current_nav_target = None
                pending_actions = []
            was_floor_transition = floor_update.in_transition
            agent_map_pos = world_to_map_coords(
                current_pos, sim, map_builder.full_map.shape
            )
            traveled_path = traveled_paths.setdefault(
                floor_manager.active_floor_id, []
            )
            if not floor_update.in_transition:
                traveled_path.append(agent_map_pos)
            visible_ref_map_coords = world_path_for_floor(
                reference_path,
                floor_update.floor_height,
                floor_manager.floor_match_tolerance,
                sim,
                map_builder.full_map.shape,
            )
            visible_start_map = (
                start_map
                if floor_update.floor_height is not None
                and abs(start_world[1] - floor_update.floor_height)
                <= floor_manager.floor_match_tolerance
                else None
            )
            visible_end_map = (
                end_map
                if floor_update.floor_height is not None
                and abs(goal_position[1] - floor_update.floor_height)
                <= floor_manager.floor_match_tolerance
                else None
            )

            # Update map visibility
            floor_manager.update_visibility(
                agent_state, fov=config['camera'].get('hfov', 110)
            )

            if stop_rule_target is not None:
                actions = plan_path_to_target(
                    sim, stop_rule_target, goal_radius=0.5
                )
                if actions:
                    pending_actions = actions
                else:
                    agent_stopped = True
                    if config['debug']['enabled']:
                        print(
                            f"  Step {step}: Agent arrived at stop rule "
                            f"'{stop_rule_triggered_name}' target, "
                            f"executing STOP"
                        )
                    if enable_vis:
                        rgb_now = obs.get('rgb', obs.get('front', None))
                        if rgb_now is not None:
                            _save_step_visualization(
                                step=step,
                                rgb=rgb_now,
                                map_builder=map_builder,
                                visualizer=visualizer,
                                agent_state=agent_state,
                                exploration_targets=[],
                                frontiers=[],
                                start_map=visible_start_map,
                                end_map=visible_end_map,
                                ref_map_coords=visible_ref_map_coords,
                                traveled_path=traveled_path,
                                parsed={'task_type': 'stop', 'raw_text': f'STOP (rule: {stop_rule_triggered_name})'},
                                dist_to_goal=dist_to_goal,
                                ep_output_dir=ep_output_dir,
                                sim=sim,
                                config=config,
                                vis_type='stop',
                                nav_target_world=stop_rule_target,
                                goal_position=(
                                    goal_position
                                    if visible_end_map is not None else None
                                ),
                                coord_transformer=coord_transformer,
                            )
                    break

                if enable_vis and (step - last_model_step) % plan_interval == 0:
                    rgb_now = obs.get('rgb', obs.get('front', None))
                    if rgb_now is not None:
                        _save_step_visualization(
                            step=step,
                            rgb=rgb_now,
                            map_builder=map_builder,
                            visualizer=visualizer,
                            agent_state=agent_state,
                            exploration_targets=[],
                            frontiers=[],
                            start_map=visible_start_map,
                            end_map=visible_end_map,
                            ref_map_coords=visible_ref_map_coords,
                            traveled_path=traveled_path,
                            parsed={'task_type': 'nav', 'raw_text': f'Navigating to {stop_rule_triggered_name} target'},
                            dist_to_goal=dist_to_goal,
                            ep_output_dir=ep_output_dir,
                            sim=sim,
                            config=config,
                            vis_type='nav',
                            nav_target_world=stop_rule_target,
                            goal_position=(
                                goal_position
                                if visible_end_map is not None else None
                            ),
                            coord_transformer=coord_transformer,
                        )

            if current_nav_target is not None:
                dist_to_nav = np.linalg.norm(current_pos - current_nav_target)
                if dist_to_nav <= nav_arrival_distance:
                    # Arrived at nav target → clear it so model is invoked
                    current_nav_target = None
                    pending_actions = []
                    
            task_complete = (
                len(pending_actions) == 0 and current_nav_target is None
            )
            plan_interval_exceeded = (
                step - last_model_step >= plan_interval
            )

            should_invoke_model = (
                stop_rule_target is None and  # Not in stop-rule navigation
                (task_complete or plan_interval_exceeded)
            )
            
            if not should_invoke_model and len(pending_actions) == 0:
                if current_nav_target is not None:
                    actions = plan_path_to_target(
                        sim, current_nav_target, goal_radius=0.5
                    )
                    if actions:
                        pending_actions = actions
                    else:
                        current_nav_target = None

            if should_invoke_model:
                rgb = obs.get('rgb', obs.get('front', None))
                if rgb is None:
                    print(f"  WARNING: No RGB at step {step}")
                    pending_actions = [1]  # Default: FORWARD
                else:
                    area_thresh = convert_square_meters_to_pixel_area(
                        config.get('exploration', {}).get(
                            'min_unexplored_area_m2', 2.0
                        ),
                        config['map']['resolution'], sim,
                    )
                    if floor_update.in_transition:
                        exploration_targets, frontiers = [], []
                    else:
                        exploration_targets, frontiers, _ = (
                            build_visible_exploration_targets(
                                target_generator,
                                coord_transformer,
                                map_builder,
                                agent_state,
                                step,
                                obs.get('depth', None),
                                area_thresh,
                                floor_update.floor_height,
                                floor_manager.floor_match_tolerance,
                                config.get('camera', {}).get(
                                    'depth_max', 10.0
                                ),
                                config.get('exploration', {}).get(
                                    'occlusion_tolerance', 0.2
                                ),
                            )
                        )

                    visible_frontiers = []
                    for et in exploration_targets:
                        if (et['pixel_coords'] is not None and
                                et['visibility_status'] == 0):
                            visible_frontiers.append(
                                [int(et['pixel_coords'][0]),
                                 int(et['pixel_coords'][1])]
                            )

                    history_images = history_manager.get_history_images()
                    prompt = prompt_builder.build_prompt(
                        instruction, visible_frontiers, history_images
                    )

                    current_img = Image.fromarray(rgb)
                    all_images = history_images + [current_img]

                    model_output = model_inference.generate(prompt, all_images)
                    episode_result.num_model_calls += 1

                    parsed = response_parser.parse(model_output)

                    if config['debug']['enabled']:
                        print(f"  Step {step}: Model output: {model_output[:80]}")
                        print(f"          Parsed: type={parsed['task_type']}, "
                              f"coord={parsed.get('coordinate')}, "
                              f"actions={parsed.get('action_sequence')}")

                    # Record prediction
                    episode_result.predictions.append({
                        'step': step,
                        'model_output': model_output,
                        'parsed': parsed,
                        'dist_to_goal': float(dist_to_goal),
                    })

                    current_nav_target = None 
                    last_model_step = step  

                    if parsed['task_type'] == 'stop':
                        agent_stopped = True
                        pending_actions = []
                        stop_rule_manager.update(parsed, None)

                    elif (
                        parsed.get('coordinate') is not None
                        and not floor_update.in_transition
                    ):
                        pixel_pred = parsed['coordinate']
                        target_world = pixel_to_world_via_map(
                            pixel_pred, agent_state, coord_transformer,
                            map_builder, exploration_targets,
                        )

                        rule_fired, rule_target, rule_name = (
                            stop_rule_manager.update(parsed, target_world)
                        )
                        if rule_fired:
                            stop_rule_target = rule_target
                            stop_rule_triggered_name = rule_name
                            if config['debug']['enabled']:
                                print(
                                    f"  Step {step}: Stop rule "
                                    f"'{rule_name}' triggered, "
                                    f"target={rule_target}"
                                )
                            actions = plan_path_to_target(
                                sim, rule_target, goal_radius=0.5
                            )
                            pending_actions = actions if actions else []

                        elif target_world is not None:
                            current_nav_target = target_world
                            actions = plan_path_to_target(
                                sim, target_world, goal_radius=0.5
                            )
                            if actions:
                                pending_actions = actions
                            else:
                                pending_actions = [1]  # Default forward
                        else:
                            logger.warning(
                                f"  Step {step}: Failed to map pixel "
                                f"{pixel_pred} to world"
                            )
                            pending_actions = [1]  # Default forward

                    elif parsed.get('action_sequence') is not None:
                        action_seq = parsed['action_sequence']
                        pending_actions = [a for a in action_seq if a != 0]
                        if not pending_actions:
                            agent_stopped = True
                        stop_rule_manager.update(parsed, None)

                    else:
                        pending_actions = [1]
                        stop_rule_manager.update(parsed, None)

                    history_manager.add_frame(rgb)

                    if enable_vis and rgb is not None:
                        _effective_nav_target = None
                        if current_nav_target is not None:
                            _effective_nav_target = current_nav_target
                        elif stop_rule_target is not None:
                            _effective_nav_target = stop_rule_target
                        elif parsed.get('coordinate') is not None and 'target_world' in locals():
                            _effective_nav_target = target_world

                        _save_step_visualization(
                            step=step,
                            rgb=rgb,
                            map_builder=map_builder,
                            visualizer=visualizer,
                            agent_state=agent_state,
                            exploration_targets=exploration_targets,
                            frontiers=frontiers,
                            start_map=visible_start_map,
                            end_map=visible_end_map,
                            ref_map_coords=visible_ref_map_coords,
                            traveled_path=traveled_path,
                            parsed=parsed,
                            dist_to_goal=dist_to_goal,
                            ep_output_dir=ep_output_dir,
                            sim=sim,
                            config=config,
                            vis_type='plan',
                            nav_target_world=_effective_nav_target,
                            goal_position=(
                                goal_position
                                if visible_end_map is not None else None
                            ),
                            coord_transformer=coord_transformer,
                        )

            # Execute action
            if agent_stopped:
                break

            if pending_actions:
                action = pending_actions.pop(0)
            else:
                # Fallback: move forward
                action = 1

            try:
                execution_plan = action_executor.plan(
                    action, scene_id, episode_id, step
                )
                obs = action_executor.execute(env, execution_plan).observation
            except ActionExecutionError as e:
                logger.warning(f"  Step {step}: Action {action} failed: {e}")
                break

            executed_position = np.array(sim.get_agent_state().position)
            if not np.array_equal(executed_position, agent_trajectory[-1]):
                agent_trajectory.append(executed_position.copy())
            step += 1

        # Episode finished - compute metrics
        final_pos = np.array(sim.get_agent_state().position)
        final_dist = np.linalg.norm(final_pos - goal_position)
        min_dist_to_goal = min(min_dist_to_goal, final_dist)
        if not np.array_equal(final_pos, agent_trajectory[-1]):
            agent_trajectory.append(final_pos.copy())

        # Try geodesic distance for more accurate NE
        try:
            geodesic_dist = sim.geodesic_distance(final_pos, goal_position)
            if not np.isnan(geodesic_dist) and geodesic_dist != float('inf'):
                final_dist = geodesic_dist
        except Exception:
            pass

        episode_result.distance_to_goal = final_dist
        episode_result.min_distance_to_goal = min_dist_to_goal
        episode_result.success = (final_dist <= success_distance and agent_stopped)
        episode_result.oracle_success = (min_dist_to_goal <= success_distance)
        episode_result.path_length = path_length
        episode_result.num_steps = step
        episode_result.stop_called = agent_stopped

        # SPL = S * L_opt / max(L_opt, L_actual)
        if episode_result.success and optimal_path_length > 0:
            episode_result.spl = (
                optimal_path_length /
                max(optimal_path_length, path_length)
            )
        else:
            episode_result.spl = 0.0

        if ndtw_enabled:
            episode_result.ndtw = compute_ndtw(
                agent_trajectory,
                rxr_ground_truth[episode_id],
                success_distance,
            )

        nav_metrics.add_episode(episode_result)

        metric_suffix = (
            f", ndtw={episode_result.ndtw:.4f}"
            if episode_result.ndtw is not None else ""
        )
        print(
            f"  Finished: {step} steps, dist={final_dist:.2f}m, "
            f"success={episode_result.success}, "
            f"spl={episode_result.spl:.4f}{metric_suffix}, "
            f"model_calls={episode_result.num_model_calls}"
        )

        # Save per-episode result
        all_episode_results.append({
            'episode_id': episode_result.episode_id,
            'scene_id': episode_result.scene_id,
            'instruction': episode_result.instruction[:200],
            'success': episode_result.success,
            'spl': episode_result.spl,
            **(
                {'ndtw': episode_result.ndtw}
                if episode_result.ndtw is not None else {}
            ),
            'distance_to_goal': episode_result.distance_to_goal,
            'oracle_success': episode_result.oracle_success,
            'path_length': episode_result.path_length,
            'optimal_path_length': episode_result.optimal_path_length,
            'num_steps': episode_result.num_steps,
            'num_model_calls': episode_result.num_model_calls,
            'stop_called': episode_result.stop_called,
            'predictions': episode_result.predictions,
        })

    # Compute and print final metrics
    final_metrics = nav_metrics.print_report()

    # Save results
    results_path = os.path.join(output_base, "eval_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': final_metrics,
            'config': {
                'plan_interval': plan_interval,
                'max_steps': max_steps,
                'success_distance': success_distance,
                'dataset_type': dataset_config.get('type', ''),
                'dataset_split': dataset_config.get('split', ''),
                'model': config.get('model', {}).get('model_name_or_path', ''),
            },
            'episodes': all_episode_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {results_path}")

    env.close()
    print("Done!")


def _save_step_visualization(
    step, rgb, map_builder, visualizer, agent_state,
    exploration_targets, frontiers,
    start_map, end_map, ref_map_coords, traveled_path,
    parsed, dist_to_goal, ep_output_dir, sim, config,
    vis_type='plan',
    nav_target_world=None,
    goal_position=None,
    coord_transformer=None,
):
    import cv2
    from habitat.utils.visualizations import maps

    rgb_vis = rgb.copy()
    pixel_coords = [t['pixel_coords'] for t in exploration_targets]
    vis_status = [t['visibility_status'] for t in exploration_targets]
    rgb_vis = visualizer.draw_targets_on_rgb(rgb_vis, pixel_coords, vis_status)

    if parsed.get('coordinate') is not None:
        pred_u, pred_v = int(parsed['coordinate'][0]), int(parsed['coordinate'][1])
        H, W = rgb_vis.shape[:2]
        if 0 <= pred_u < W and 0 <= pred_v < H:
            cv2.circle(rgb_vis, (pred_u, pred_v), 12, (0, 0, 255), 3)  # Red circle
            cv2.putText(
                rgb_vis, "PRED", (pred_u - 20, pred_v - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

    if goal_position is not None and coord_transformer is not None:
        try:
            goal_pixel, goal_vis_status, _ = coord_transformer.world_to_pixel(
                goal_position, agent_state, current_step=step
            )
            if goal_vis_status == 0:  
                gu, gv = int(goal_pixel[0]), int(goal_pixel[1])
                H, W = rgb_vis.shape[:2]
                if 0 <= gu < W and 0 <= gv < H:
                    diamond_size = 14
                    pts = np.array([
                        [gu, gv - diamond_size],
                        [gu + diamond_size, gv],
                        [gu, gv + diamond_size],
                        [gu - diamond_size, gv]
                    ], dtype=np.int32)
                    cv2.fillPoly(rgb_vis, [pts], (255, 0, 255))  # Magenta fill
                    cv2.polylines(rgb_vis, [pts], True, (0, 0, 0), 2)
                    cv2.putText(
                        rgb_vis, "GOAL", (gu - 20, gv - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
                    )
        except Exception as e:
            logger.debug(f"Goal projection to RGB failed: {e}")

    topdown_vis = map_builder.get_visualization_map(agent_state)
    map_vis = topdown_vis.copy()

    map_vis = visualizer.draw_start_end_path(
        map_vis,
        start_pos=start_map,
        end_pos=end_map,
        trajectory_map_coords=ref_map_coords,
        traveled_path=traveled_path,
    )

    map_coords = [t['topdown_coords'] for t in exploration_targets]
    map_vis = visualizer.draw_targets_on_topdown(map_vis, map_coords)

    # Draw frontiers
    map_vis = visualizer.draw_frontiers(map_vis, frontiers)

    # Draw navigation target prediction on top-down map as a star marker
    if nav_target_world is not None:
        try:
            nav_map_pos = world_to_map_coords(
                nav_target_world, sim, map_builder.full_map.shape
            )
            nav_row, nav_col = int(nav_map_pos[0]), int(nav_map_pos[1])
            if 0 <= nav_row < map_vis.shape[0] and 0 <= nav_col < map_vis.shape[1]:
                # Draw a star marker for the predicted nav target
                _draw_star_marker(map_vis, nav_col, nav_row, size=12, color=(0, 165, 255))  # Orange star
                cv2.putText(
                    map_vis, "NAV",
                    (nav_col + 14, nav_row - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2
                )
        except Exception as e:
            logger.debug(f"Nav target topdown drawing failed: {e}")

    # Add legend
    map_vis = visualizer.add_legend(map_vis)

    # Combine
    combined = visualizer.create_combined_visualization(rgb_vis, map_vis)

    # Add info text
    pred_text = parsed.get('raw_text', '')[:50]
    type_label = {'plan': 'Plan', 'nav': 'Nav→Target', 'stop': 'STOP'}.get(vis_type, vis_type)
    combined = visualizer.add_info_text(
        combined,
        [
            f"Step: {step}  [{type_label}]",
            f"Type: {parsed.get('task_type', '?')}",
            f"Dist: {dist_to_goal:.1f}m",
            f"Pred: {pred_text}",
        ]
    )

    # Save with naming: step_xxxx_type.jpg
    vis_path = os.path.join(ep_output_dir, f"step_{step:04d}_{parsed.get('task_type', '?')}.jpg")
    cv2.imwrite(vis_path, combined)


def _draw_star_marker(image, cx, cy, size=12, color=(0, 165, 255), thickness=-1):
    import cv2
    inner_size = size * 0.4
    points = []
    for i in range(5):
        # Outer point
        angle_outer = np.deg2rad(-90 + i * 72)
        points.append([
            int(cx + size * np.cos(angle_outer)),
            int(cy + size * np.sin(angle_outer)),
        ])
        # Inner point
        angle_inner = np.deg2rad(-90 + i * 72 + 36)
        points.append([
            int(cx + inner_size * np.cos(angle_inner)),
            int(cy + inner_size * np.sin(angle_inner)),
        ])
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(image, [pts], color)
    cv2.polylines(image, [pts], True, (0, 0, 0), 1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLN model in Habitat environment"
    )
    parser.add_argument(
        "--config", "-c",
        default=os.path.join(_REPO_ROOT, "agentvln", "configs", "habitat_eval_config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode (limits episodes)",
    )
    parser.add_argument(
        "--debug-episodes",
        type=int,
        help="Number of episodes to process in debug mode",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable visualization output",
    )
    parser.add_argument(
        "--plan-interval",
        type=int,
        default=None,
        help="Override model invocation interval (steps)",
    )

    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.plan_interval is not None:
        config = load_config(args.config)
        config['evaluation']['plan_interval'] = args.plan_interval
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            yaml.dump(config, f)
            args.config = f.name

    run_habitat_eval(args)


if __name__ == "__main__":
    main()
