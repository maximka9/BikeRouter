"""CLI for the experimental OSM surface ML recovery module.

Run from the repository parent:

    python -m bike_router.tools.surface_ml_experiment
    python -m bike_router.tools.surface_ml_experiment --max-edges 500   # smoke run

Artifacts: ``<BIKE_ROUTER_BASE_DIR or bike_router/data>/experiments/surface_ml_YYYYMMDD_HHMMSS/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run an experimental ML model for missing OSM surface tags."
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="Optional cap on edges for smoke runs; default is no limit (all edges in area).",
    )
    parser.add_argument("--sample-step-m", type=float, default=7.0)
    parser.add_argument("--pixel-window", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--spatial-grid-m", type=float, default=300.0)
    parser.add_argument("--test-share", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-known-edges", type=int, default=100)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    _ensure_pkg_path()
    from bike_router.config import Settings
    from bike_router.services.surface_ml import (
        SurfaceMLConfig,
        run_surface_ml_experiment,
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = SurfaceMLConfig(
        sample_step_m=float(args.sample_step_m),
        pixel_window=int(args.pixel_window),
        min_confidence=float(args.min_confidence),
        spatial_grid_m=float(args.spatial_grid_m),
        test_share=float(args.test_share),
        random_state=int(args.random_state),
        min_known_edges=int(args.min_known_edges),
        n_estimators=int(args.n_estimators),
    )
    artifacts = run_surface_ml_experiment(
        settings=Settings(),
        config=config,
        max_edges=args.max_edges,
    )
    print(str(artifacts.output_dir))


if __name__ == "__main__":
    main()

