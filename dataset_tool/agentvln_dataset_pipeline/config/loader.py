from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_EXPLORATION_CONFIG = CONFIG_DIR / "exploration_config.yaml"
DEFAULT_HABITAT_CONFIG = CONFIG_DIR / "vln_r2r.yaml"


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
