from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


@dataclass(frozen=True)
class ConfigOverride:
    source_path: str
    target_path: str
    old_value: Any
    new_value: Any

    @property
    def is_conflict(self) -> bool:
        return self.old_value != self.new_value


def _get_path(root, path) -> Any:
    value = root
    for key in path:
        if isinstance(value, Mapping):
            value = value[key]
        else:
            value = getattr(value, key)
    return value


def _set_path(root, path, value) -> None:
    parent = root
    for key in path[:-1]:
        if isinstance(parent, Mapping):
            parent = parent[key]
        else:
            parent = getattr(parent, key)

    key = path[-1]
    if isinstance(parent, MutableMapping):
        parent[key] = value
    else:
        setattr(parent, key, value)


def _has_path(root, path) -> bool:
    try:
        _get_path(root, path)
        return True
    except (AttributeError, KeyError, TypeError):
        return False


@contextmanager
def mutable_config(config):
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None

    try:
        from omegaconf import read_write
    except ImportError:
        read_write = None

    try:
        from omegaconf import open_dict
    except ImportError:
        open_dict = None

    if OmegaConf is not None and OmegaConf.is_config(config):
        if read_write is not None:
            with read_write(config):
                if open_dict is not None:
                    with open_dict(config):
                        yield config
                else:
                    yield config
        else:
            was_readonly = OmegaConf.is_readonly(config)
            OmegaConf.set_readonly(config, False)
            try:
                yield config
            finally:
                OmegaConf.set_readonly(config, was_readonly)
        return

    mutable_update = getattr(config, "mutable_update", None)
    if callable(mutable_update):
        with mutable_update():
            yield config
        return

    defrost = getattr(config, "defrost", None)
    freeze = getattr(config, "freeze", None)
    if callable(defrost) and callable(freeze):
        is_frozen = getattr(config, "is_frozen", None)
        was_frozen = bool(is_frozen()) if callable(is_frozen) else True
        defrost()
        try:
            yield config
        finally:
            if was_frozen:
                freeze()
        return

    yield config


def _normalise_dataset_type(dataset_type) -> Any:
    if isinstance(dataset_type, str) and dataset_type.casefold() == "r2r":
        return "R2RVLN-v1"
    if isinstance(dataset_type, str) and dataset_type.casefold() == "rxr":
        return "RxR-VLN-CE-v1"
    return dataset_type


def _override_specs(config) -> Iterable[Tuple[str, Tuple[str, ...], Any]]:
    dataset = config.get("dataset", {})
    if "type" in dataset:
        yield (
            "dataset.type",
            ("habitat", "dataset", "type"),
            _normalise_dataset_type(dataset["type"]),
        )
    for key in ("split", "data_path", "scenes_dir"):
        if key in dataset:
            yield (
                f"dataset.{key}",
                ("habitat", "dataset", key),
                dataset[key],
            )
    normalized_type = _normalise_dataset_type(dataset.get("type"))
    if (
        isinstance(normalized_type, str)
        and normalized_type.casefold() == "rxr-vln-ce-v1"
    ):
        for key in ("roles", "languages"):
            if key in dataset:
                yield (
                    f"dataset.{key}",
                    ("habitat", "dataset", key),
                    dataset[key],
                )

    camera = config.get("camera", {})
    for sensor_name in ("rgb_sensor", "depth_sensor"):
        sensor_path = (
            "habitat",
            "simulator",
            "agents",
            "main_agent",
            "sim_sensors",
            sensor_name,
        )
        for key in ("width", "height", "hfov"):
            if key in camera:
                yield f"camera.{key}", sensor_path + (key,), camera[key]
        if "camera_height" in camera:
            yield (
                "camera.camera_height",
                sensor_path + ("position",),
                [0.0, camera["camera_height"], 0.0],
            )

    if "depth_max" in camera:
        yield (
            "camera.depth_max",
            (
                "habitat",
                "simulator",
                "agents",
                "main_agent",
                "sim_sensors",
                "depth_sensor",
                "max_depth",
            ),
            camera["depth_max"],
        )


def apply_exploration_overrides(
    habitat_config,
    exploration_config,
) -> List[ConfigOverride]:
    specs = list(_override_specs(exploration_config))
    addable_dataset_fields = {"dataset.roles", "dataset.languages"}
    missing = [
        ".".join(path)
        for source, path, _ in specs
        if not _has_path(habitat_config, path)
        and source not in addable_dataset_fields
    ]
    if missing:
        raise KeyError(
            "Habitat config is missing fields required by exploration_config: "
            + ", ".join(missing)
        )

    overrides = [
        ConfigOverride(
            source,
            ".".join(path),
            _get_path(habitat_config, path)
            if _has_path(habitat_config, path) else None,
            value,
        )
        for source, path, value in specs
    ]

    with mutable_config(habitat_config):
        for (_, path, value), override in zip(specs, overrides):
            if override.old_value != value:
                _set_path(habitat_config, path, value)

    return overrides


def print_override_report(
    overrides,
    source_name = "exploration_config.yaml",
) -> None:
    conflicts = [item for item in overrides if item.is_conflict]
    print(
        f"[Config] Applied {len(overrides)} Habitat overrides from {source_name}; "
        f"{source_name} has higher priority."
    )
    if not conflicts:
        print("[Config] No conflicting values found in overlapping fields.")
        return

    print(
        f"[Config Override] Found {len(conflicts)} conflict(s); "
        f"using values from {source_name}:"
    )
    for item in conflicts:
        print(
            f"  - {item.source_path} -> {item.target_path}: "
            f"{item.old_value!r} -> {item.new_value!r}"
        )
