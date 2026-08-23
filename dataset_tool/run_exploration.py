import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = (
        Path(__file__).resolve().parent
        / "agentvln_dataset_pipeline"
        / "config"
        / "exploration_config.yaml"
    )
    parser.add_argument("--config", "-c", default=str(default_config))
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--debug-episodes", type=int)
    args = parser.parse_args()

    from agentvln_dataset_pipeline.application.runner import run_exploration

    run_exploration(args)


if __name__ == "__main__":
    main()
