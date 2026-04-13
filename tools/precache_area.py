#!/usr/bin/env python3
"""CLI: предсборка area_precache (OSM + веса + опционально зелень) по PRECACHE_AREA_POLYGON_WKT.

Запуск из каталога с настроенным ``BIKE_ROUTER_BASE_DIR`` / ``.env``::

    python -m bike_router.tools.precache_area

В Docker (тот же ``.env``, каталог ``./data`` на хосте)::

    docker compose run --rm bike-router python -m bike_router.tools.precache_area
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Пакет при запуске как скрипт
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("precache_area")


def main() -> int:
    from bike_router.app import Application
    from bike_router.config import Settings
    from bike_router.services.area_graph_cache import (
        build_area_precache,
        precache_area_dir,
    )

    s = Settings()
    if not s.has_precache_area_polygon:
        logger.error(
            "Задайте PRECACHE_AREA_POLYGON_WKT в окружении или .env "
            "(и при необходимости BIKE_ROUTER_BASE_DIR)."
        )
        return 1
    app = Application(s)
    out = build_area_precache(app)
    logger.info("Готово: %s", out)
    logger.info("Каталог предкэша: %s", precache_area_dir(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
