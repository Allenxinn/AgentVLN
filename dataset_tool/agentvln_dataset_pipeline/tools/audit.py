import argparse
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict

import lmdb

from ..core.dataset_contract import (
    dataset_actuation_noise_settings,
    validate_dataset_contract,
)


def _load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def _label(task, step):
    action = task["actions"][step]
    if action == 0:
        return ("stop",)
    if (
        task["trajectory_is_endpoint"][step]
        and task["trajectory_pixel"][step] is not None
        and task["trajectory_status"][step] == 0
    ):
        return ("target", tuple(task["trajectory_pixel"][step]))
    index = task["expert_candidate_indices"][step]
    if (
        index is not None
        and 0 <= int(index) < len(task["pixel_coords"][step])
        and int(index) < len(task["visibility_status"][step])
        and task["visibility_status"][step][int(index)] == 0
    ):
        return ("frontier", tuple(task["pixel_coords"][step][int(index)]))
    return ("action", int(action))


def _image_bytes(root, task, step, environments):
    scene_id = str(task["scene_id"])
    path = os.path.join(root, f"{scene_id}.lmdb")
    if not os.path.exists(path):
        return None
    if scene_id not in environments:
        environments[scene_id] = lmdb.open(path, readonly=True, lock=False)
    key = f"{task['task_id']}/rgb/step_{step:04d}".encode("utf-8")
    with environments[scene_id].begin() as transaction:
        return transaction.get(key)


def audit(dataset, data_root=None):
    validate_dataset_contract(dataset)
    config = dataset["generation_config"]
    noise_settings = dataset_actuation_noise_settings(config)
    counts = Counter()
    labels_by_hash = defaultdict(set)
    missing_images = 0
    sampled_scales = []
    environments = {}

    try:
        for task in dataset.get("tasks", []):
            counts[f"termination:{task['termination_reason']}"] += 1
            for step, action in enumerate(task["actions"]):
                if action is None:
                    continue
                label = _label(task, step)
                counts[f"label:{label[0]}"] += 1
                counts["samples"] += 1
                if task["floor_transition"][step]:
                    counts["transition_frames"] += 1
                if task["expert_candidate_indices"][step] is not None:
                    counts["expert_candidate_matches"] += 1
                if task["trajectory_world"][step] is not None:
                    counts["trajectory_targets"] += 1
                action_debug = task.get("action_debug")
                if action_debug is not None and action_debug[step] is not None:
                    debug = action_debug[step]
                    counts["verbose_action_records"] += 1
                    if int(action) != 0:
                        scale = float(debug["scale"])
                        sampled_scales.append(scale)
                        if abs(scale - 1.0) > 1e-12:
                            counts["perturbed_actions"] += 1
                        if debug.get("collision") is True:
                            counts["collisions"] += 1
                if data_root:
                    image = _image_bytes(data_root, task, step, environments)
                    if image is None:
                        missing_images += 1
                    else:
                        digest = hashlib.sha256(image).hexdigest()
                        labels_by_hash[digest].add(label)
    finally:
        for environment in environments.values():
            environment.close()

    conflicts = {
        digest: sorted(repr(label) for label in labels)
        for digest, labels in labels_by_hash.items()
        if len(labels) > 1
    }
    summary = {
        "schema_version": dataset["schema_version"],
        "generation_config": config,
        "actuation_noise": noise_settings,
        "tasks": len(dataset.get("tasks", [])),
        "counts": dict(sorted(counts.items())),
        "sampled_scale": {
            "count": len(sampled_scales),
            "min": min(sampled_scales) if sampled_scales else None,
            "max": max(sampled_scales) if sampled_scales else None,
            "mean": (
                sum(sampled_scales) / len(sampled_scales)
                if sampled_scales else None
            ),
        },
        "candidate_recall_at_5": (
            counts["expert_candidate_matches"] / counts["trajectory_targets"]
            if counts["trajectory_targets"] else None
        ),
        "rgb_conflict_count": len(conflicts),
        "rgb_conflict_rate": (
            len(conflicts) / len(labels_by_hash) if labels_by_hash else 0.0
        ),
        "missing_image_count": missing_images,
        "rgb_conflicts": conflicts,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Schema-v3 JSON or JSON.GZ")
    parser.add_argument(
        "--data-root",
        help="Optional root containing per-scene LMDB databases",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    summary = audit(_load(args.dataset), args.data_root)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered + "\n")
    print(rendered)
    if summary["rgb_conflict_count"] or summary["missing_image_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
