"""Сравнение одного маршрута с SURFACE_AI_RUNTIME off vs on.

Пример::

    python -m bike_router.tools.surface_runtime_route_experiment --modes off,on
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def run_modes(modes_csv: str) -> List[Dict[str, Any]]:
    """Для каждого режима (off|on) — warmup + один ``compute_route``; возвращает строки-отчёты."""
    _ensure_pkg_path()
    from bike_router.app import Application
    from bike_router.config import Settings
    from bike_router.engine import RouteEngine

    modes = [m.strip().lower() for m in modes_csv.split(",") if m.strip()]
    out: List[Dict[str, Any]] = []
    s = Settings()
    start = s.start_coords
    end = s.end_coords
    for m in modes:
        en = "true" if m == "on" else "false"
        os.environ["SURFACE_AI_RUNTIME_ENABLED"] = en
        app = Application(Settings())
        eng = RouteEngine(app)
        eng.warmup()
        rr = eng.compute_route(start, end, "cyclist", "full")
        sr = rr.surface_resolution or {}
        out.append(
            {
                "surface_ai_runtime": m,
                "length_m": rr.length_m,
                "cost": rr.cost,
                "surface_resolution": sr,
            }
        )
    return out


def main() -> None:
    _ensure_pkg_path()
    parser = argparse.ArgumentParser(
        description="Сравнение маршрута (START_* / END_* из .env) с ML runtime off vs on.",
    )
    parser.add_argument(
        "--modes",
        default="off,on",
        help="Через запятую: off, on или оба.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    rows = run_modes(args.modes)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
