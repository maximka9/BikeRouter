#!/usr/bin/env python3
"""Точка входа для Docker entrypoint: идемпотентная подготовка area_precache до запуска API.

См. также: ``python -m bike_router.tools.precache_area`` (то же поведение через ``ensure_area_precache``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ensure_precache")


def _phase(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    _phase("ensure_precache: загрузка модулей…")
    from bike_router.app import Application
    from bike_router.config import Settings
    from bike_router.services.area_graph_cache import ensure_area_precache

    _phase("ensure_precache: Settings…")
    s = Settings()
    if not s.precache_area_enabled:
        logger.info(
            "PRECACHE_AREA_ENABLED=false — пропуск ensure (задайте true для проверки арены)"
        )
        return 0
    if not s.has_precache_area_polygon:
        logger.error("PRECACHE_AREA_ENABLED=true, но нет PRECACHE_AREA_POLYGON_WKT")
        return 1
    try:
        _phase("ensure_precache: Application + ensure_area_precache…")
        ensure_area_precache(Application(s))
    except Exception as exc:
        logger.exception("ensure_area_precache: ошибка: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
