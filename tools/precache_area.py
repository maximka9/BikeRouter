#!/usr/bin/env python3
"""CLI: предсборка area_precache (OSM + веса + опционально зелень) по PRECACHE_AREA_POLYGON_WKT.

Запуск из каталога с настроенным ``BIKE_ROUTER_BASE_DIR`` / ``.env``::

    python -m bike_router.tools.precache_area

В Docker (тот же ``.env``, каталог ``./data`` на хосте)::

    docker compose run --rm bike-router python -m bike_router.tools.precache_area

Первые секунды после команды могут **молчать**: идёт импорт тяжёлых библиотек.
При подозрении на зависание: ``python -u -m bike_router.tools.precache_area`` (без буфера вывода)
или смотрите строки ``precache_area: …`` в stderr. Забитый системный диск сильно замедляет старт.
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


def _phase(msg: str) -> None:
    """Ранний вывод до тяжёлых импортов (geopandas/osmnx могут молчать минутами)."""
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    _phase("precache_area: загрузка модулей (geopandas, osmnx — может занять 1–3 мин при холодном старте)…")
    from bike_router.app import Application
    from bike_router.config import Settings
    from bike_router.services.area_graph_cache import (
        ensure_area_precache,
        precache_area_dir,
    )

    _phase("precache_area: чтение Settings из .env…")
    s = Settings()
    _phase(
        "precache_area: этапы — (1) graph_base + phase1, "
        "(2) area_green_edges / спутник, (3) graph_green — см. логи area_precache…"
    )
    if not s.has_precache_area_polygon:
        logger.error(
            "Задайте PRECACHE_AREA_POLYGON_WKT в окружении или .env "
            "(и при необходимости BIKE_ROUTER_BASE_DIR)."
        )
        return 1
    _phase(
        f"precache_area: BIKE_ROUTER_BASE_DIR={s.base_dir!r} — создание Application (OSMnx)…"
    )
    app = Application(s)
    _phase("precache_area: запуск ensure_area_precache…")
    out = ensure_area_precache(app)
    logger.info("Готово: %s", out)
    logger.info("Каталог предкэша: %s", precache_area_dir(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
