"""CLI for final experimental AI recovery of concrete OSM surface values.

Example:

    python -m bike_router.tools.surface_ai_experiment --mode all --max-edges 5000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate AI models for concrete OSM surface recovery."
    )
    parser.add_argument("--mode", choices=("all",), default="all")
    parser.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke runs; default uses all edges.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model candidates override.")
    parser.add_argument("--min-class-count", type=int, default=None)
    parser.add_argument("--sample-step-m", type=float, default=None)
    parser.add_argument("--pixel-window", type=int, default=None)
    parser.add_argument("--spatial-grid-m", type=float, default=None)
    parser.add_argument("--test-share", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--no-satellite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _override_config(config, args):
    from dataclasses import replace

    updates = {}
    if args.models:
        updates["model_candidates"] = tuple(x.strip() for x in args.models.split(",") if x.strip())
    if args.min_class_count is not None:
        updates["min_class_count"] = int(args.min_class_count)
    if args.sample_step_m is not None:
        updates["sample_step_m"] = float(args.sample_step_m)
    if args.pixel_window is not None:
        updates["pixel_window"] = int(args.pixel_window)
    if args.spatial_grid_m is not None:
        updates["spatial_grid_m"] = float(args.spatial_grid_m)
    if args.test_share is not None:
        updates["test_share"] = float(args.test_share)
    if args.random_state is not None:
        updates["random_state"] = int(args.random_state)
    if args.n_estimators is not None:
        updates["rf_n_estimators"] = int(args.n_estimators)
    if args.no_satellite:
        updates["use_satellite_features"] = False
    return replace(config, **updates) if updates else config


def main(argv: list[str] | None = None) -> None:
    _ensure_pkg_path()
    from bike_router.config import Settings
    from bike_router.services.surface_ai import (
        SurfaceAIConfig,
        run_surface_ai_experiment,
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = _override_config(SurfaceAIConfig.from_env(), args)
    artifacts = run_surface_ai_experiment(
        settings=Settings(),
        config=config,
        max_edges=args.max_edges,
        output_dir=args.output_dir,
    )
    print(str(artifacts.output_dir))


if __name__ == "__main__":
    main()

