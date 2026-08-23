import os
import sys

from ...config.loader import DEFAULT_HABITAT_CONFIG
from ...config.overrides import apply_exploration_overrides, print_override_report


def create_environment(config, config_source=None):
    from habitat import Env

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    external_root = os.path.abspath(os.path.join(project_root, "..", "infer_r2r_rxr"))
    if external_root not in sys.path:
        sys.path.insert(0, external_root)
    from habitat_baselines.config.default import get_config

    habitat_path = config.get("habitat_config_path", str(DEFAULT_HABITAT_CONFIG))
    if not os.path.isabs(habitat_path):
        base = (
            os.path.dirname(os.path.abspath(config_source))
            if config_source else str(DEFAULT_HABITAT_CONFIG.parent)
        )
        habitat_path = os.path.abspath(os.path.join(base, habitat_path))
    print(f"[Config] Habitat base config: {habitat_path}")
    habitat_config = get_config(habitat_path)
    overrides = apply_exploration_overrides(habitat_config, config)
    print_override_report(
        overrides,
        os.path.basename(config_source) if config_source else "exploration_config.yaml",
    )
    return Env(habitat_config)
