from ..config.loader import load_config
from ..core.dataset_contract import build_generation_config
from ..integrations.habitat.environment import create_environment
from ..storage.dataset_writer import DatasetWriter
from .episode_generator import EpisodeGenerator


def run_exploration(args):
    config = load_config(args.config)
    if args.debug:
        config.setdefault("debug", {})["enabled"] = True
    if args.debug_episodes is not None:
        config.setdefault("debug", {})["max_episodes"] = args.debug_episodes

    generation_config = build_generation_config(config)
    env = create_environment(config, args.config)
    writer = DatasetWriter(
        config["output"]["base_path"],
        generation_config=generation_config,
        image_formats=config.get("output", {}).get("lmdb_formats"),
        jpg_quality=config.get("output", {}).get("lmdb_jpg_quality", 95),
    )
    generator = EpisodeGenerator(config, writer)

    episode_count = len(env.episodes)
    if config.get("debug", {}).get("enabled", False):
        episode_count = min(
            episode_count, int(config.get("debug", {}).get("max_episodes", 3))
        )
    print(f"Processing {episode_count} episodes with schema v3...")
    try:
        for episode_index in range(episode_count):
            obs = env.reset()
            episode = env.current_episode
            result = generator.generate(env, obs)
            if result.task_data is not None:
                writer.add_task(result.task_data)
            samples = len(result.task_data["actions"]) if result.task_data else 0
            print(
                f"[{episode_index + 1}/{episode_count}] ep={episode.episode_id} "
                f"termination={result.termination_reason} samples={samples}"
            )
        writer.save_dataset("exploration_data.json")
    finally:
        writer.close()
        env.close()
