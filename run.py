"""Single entry point for the Spatial-Behavioural Analytics pipeline.

Usage examples:

    python run.py
    python run.py --config config/settings.yaml
    python run.py --stage features
    python run.py --stage apriori
    python run.py --stage clustering
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_pipeline


_STAGES = ["all", "features", "apriori", "clustering"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial-Behavioural Analytics pipeline for elderly food vulnerability"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to the YAML configuration file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=_STAGES,
        help="Run only a specific stage of the pipeline (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(config_path=args.config, stage=args.stage)


if __name__ == "__main__":
    main()
