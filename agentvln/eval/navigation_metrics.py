import gzip
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


RXR_DATASET_TYPES = {"rxr", "rxr-vln-ce-v1"}
RXR_VALIDATION_SPLITS = {"val_seen", "val_unseen"}
RXR_ANNOTATION_ROLES = ("guide", "follower")


def is_rxr_validation(dataset_config: Mapping[str, Any]) -> bool:
    dataset_type = str(dataset_config.get("type", "")).casefold()
    split = str(dataset_config.get("split", "")).casefold()
    return dataset_type in RXR_DATASET_TYPES and split in RXR_VALIDATION_SPLITS


def _as_trajectory(points: Sequence[Sequence[float]], name: str) -> np.ndarray:
    try:
        trajectory = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric 3D coordinates") from exc

    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] != 3:
        raise ValueError(f"{name} must be a non-empty N x 3 trajectory")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return trajectory


def _drop_consecutive_duplicates(trajectory: np.ndarray) -> np.ndarray:
    if len(trajectory) <= 1:
        return trajectory
    keep = np.ones(len(trajectory), dtype=bool)
    keep[1:] = np.any(trajectory[1:] != trajectory[:-1], axis=1)
    return trajectory[keep]


def compute_ndtw(
    agent_locations: Sequence[Sequence[float]],
    reference_locations: Sequence[Sequence[float]],
    success_distance: float,
) -> float:
    if not math.isfinite(success_distance) or success_distance <= 0:
        raise ValueError("success_distance must be a positive finite number")

    agent = _drop_consecutive_duplicates(
        _as_trajectory(agent_locations, "agent trajectory")
    )
    reference = _as_trajectory(reference_locations, "reference trajectory")

    previous = np.full(len(reference) + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for agent_point in agent:
        current = np.full(len(reference) + 1, np.inf, dtype=np.float64)
        for ref_index, reference_point in enumerate(reference, start=1):
            point_distance = np.linalg.norm(agent_point - reference_point)
            current[ref_index] = point_distance + min(
                previous[ref_index],
                current[ref_index - 1],
                previous[ref_index - 1],
            )
        previous = current

    dtw_distance = float(previous[-1])
    return float(np.exp(-dtw_distance / (len(reference) * success_distance)))


def _resolve_config_path(path: str, config_source: Optional[str]) -> str:
    expanded = os.path.expanduser(os.path.expandvars(path))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    base_dir = (
        os.path.dirname(os.path.abspath(config_source))
        if config_source
        else os.getcwd()
    )
    return os.path.normpath(os.path.join(base_dir, expanded))


def _configured_roles(dataset_config: Mapping[str, Any]) -> List[str]:
    roles = dataset_config.get("roles", ["guide"])
    if isinstance(roles, str):
        roles = [roles]
    roles = [str(role).casefold() for role in roles]
    if roles == ["*"]:
        return list(RXR_ANNOTATION_ROLES)
    if not roles or not set(roles).issubset(RXR_ANNOTATION_ROLES):
        raise ValueError(
            "dataset.roles must contain 'guide', 'follower', or the single value '*'"
        )
    return roles


def load_rxr_ground_truth(
    dataset_config: Mapping[str, Any],
    config_source: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    gt_template = dataset_config.get("gt_path")
    if not isinstance(gt_template, str) or not gt_template.strip():
        raise ValueError(
            "RxR validation requires dataset.gt_path pointing to "
            "the official *_gt.json.gz file"
        )

    split = str(dataset_config.get("split", ""))
    roles = _configured_roles(dataset_config)
    paths = []
    if "{role}" in gt_template:
        for role in roles:
            paths.append((role, gt_template.format(split=split, role=role)))
    else:
        paths.append((None, gt_template.format(split=split, role=roles[0])))

    ground_truth: Dict[str, np.ndarray] = {}
    for role, configured_path in paths:
        path = _resolve_config_path(configured_path, config_source)
        if not os.path.isfile(path):
            role_suffix = f" for role {role!r}" if role else ""
            raise FileNotFoundError(f"RxR GT file not found{role_suffix}: {path}")

        try:
            with gzip.open(path, "rt", encoding="utf-8") as gt_file:
                payload = json.load(gt_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed to read RxR GT file {path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"RxR GT file must contain an episode mapping: {path}")

        for raw_episode_id, record in payload.items():
            episode_id = str(raw_episode_id)
            if not isinstance(record, dict) or "locations" not in record:
                raise ValueError(
                    f"RxR GT episode {episode_id!r} in {path} has no locations"
                )
            locations = _as_trajectory(
                record["locations"],
                f"RxR GT episode {episode_id!r} locations",
            )
            if episode_id in ground_truth:
                raise ValueError(
                    f"Duplicate RxR GT episode id {episode_id!r} across GT files"
                )
            ground_truth[episode_id] = locations

    if not ground_truth:
        raise ValueError("RxR GT files contain no episodes")
    return ground_truth


def validate_episode_ground_truth(
    episode_ids: Sequence[Any], ground_truth: Mapping[str, np.ndarray]
) -> None:
    missing = [str(ep_id) for ep_id in episode_ids if str(ep_id) not in ground_truth]
    if missing:
        preview = ", ".join(repr(ep_id) for ep_id in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            f"RxR GT is missing {len(missing)} evaluated episode(s): {preview}{suffix}"
        )


@dataclass
class EpisodeResult:
    episode_id: str
    scene_id: str
    instruction: str
    success: bool = False
    spl: float = 0.0
    ndtw: Optional[float] = None
    distance_to_goal: float = float("inf")
    oracle_success: bool = False
    min_distance_to_goal: float = float("inf")
    path_length: float = 0.0
    optimal_path_length: float = 0.0
    num_steps: int = 0
    num_model_calls: int = 0
    stop_called: bool = False
    predictions: List[Dict] = field(default_factory=list)


class NavigationMetrics:
    def __init__(self, success_distance: float = 3.0, include_ndtw: bool = False):
        self.success_distance = success_distance
        self.include_ndtw = include_ndtw
        self.episodes: List[EpisodeResult] = []

    def add_episode(self, result: EpisodeResult):
        self.episodes.append(result)

    def compute(self) -> Dict[str, Any]:
        if not self.episodes:
            return {}

        n = len(self.episodes)
        sr = sum(1 for ep in self.episodes if ep.success) / n
        spl = sum(ep.spl for ep in self.episodes) / n
        ne = sum(ep.distance_to_goal for ep in self.episodes) / n
        osr = sum(1 for ep in self.episodes if ep.oracle_success) / n
        avg_steps = sum(ep.num_steps for ep in self.episodes) / n
        avg_model_calls = sum(ep.num_model_calls for ep in self.episodes) / n
        avg_path_length = sum(ep.path_length for ep in self.episodes) / n
        stop_rate = sum(1 for ep in self.episodes if ep.stop_called) / n

        metrics = {
            "num_episodes": n,
            "SR": round(sr, 4),
            "SPL": round(spl, 4),
            "NE": round(ne, 2),
            "Oracle_SR": round(osr, 4),
            "Avg_Steps": round(avg_steps, 1),
            "Avg_Model_Calls": round(avg_model_calls, 1),
            "Avg_Path_Length": round(avg_path_length, 2),
            "Stop_Rate": round(stop_rate, 4),
        }
        if self.include_ndtw:
            missing = [ep.episode_id for ep in self.episodes if ep.ndtw is None]
            if missing:
                raise ValueError(
                    "nDTW is enabled but missing for episode(s): "
                    + ", ".join(missing[:5])
                )
            metrics["nDTW"] = round(
                sum(float(ep.ndtw) for ep in self.episodes) / n, 4
            )
        return metrics

    def print_report(self):
        metrics = self.compute()
        if not metrics:
            print("No episodes evaluated.")
            return {}

        print("\n" + "=" * 60)
        print("VLN NAVIGATION EVALUATION REPORT")
        print("=" * 60)
        print(f"  Episodes:         {metrics['num_episodes']}")
        print(f"  SR (↑):           {metrics['SR']:.4f}")
        print(f"  SPL (↑):          {metrics['SPL']:.4f}")
        if "nDTW" in metrics:
            print(f"  nDTW (↑):         {metrics['nDTW']:.4f}")
        print(f"  NE (↓):           {metrics['NE']:.2f}m")
        print(f"  Oracle SR (↑):    {metrics['Oracle_SR']:.4f}")
        print(f"  Avg Steps:        {metrics['Avg_Steps']:.1f}")
        print(f"  Avg Model Calls:  {metrics['Avg_Model_Calls']:.1f}")
        print(f"  Avg Path Length:  {metrics['Avg_Path_Length']:.2f}m")
        print(f"  Stop Rate:        {metrics['Stop_Rate']:.4f}")
        print("=" * 60)
        return metrics
