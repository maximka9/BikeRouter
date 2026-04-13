"""Счётчики Prometheus (опционально). Подключение: pip install prometheus_client."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_OVERPASS_SUBRETRY: Any = None
_ROUTE_DISK_HIT: Any = None
_ROUTE_DISK_MISS: Any = None
_CORRIDOR_DISK_HIT: Any = None
_GREEN_JOB_FAILED: Any = None
_TILE_FAIL: Any = None


def _c(name: str, doc: str) -> Optional[Any]:
    try:
        from prometheus_client import Counter

        return Counter(name, doc, [])
    except ImportError:
        return None


_initialized = False


def _lazy() -> None:
    global _initialized
    global _OVERPASS_SUBRETRY, _ROUTE_DISK_HIT, _ROUTE_DISK_MISS
    global _CORRIDOR_DISK_HIT, _GREEN_JOB_FAILED, _TILE_FAIL
    if _initialized:
        return
    _initialized = True
    _OVERPASS_SUBRETRY = _c(
        "bike_router_overpass_subretries_total",
        "Повторные попытки на одном mirror Overpass",
    )
    _ROUTE_DISK_HIT = _c(
        "bike_router_route_disk_cache_hit_total",
        "Попадание в дисковый кэш POST /alternatives",
    )
    _ROUTE_DISK_MISS = _c(
        "bike_router_route_disk_cache_miss_total",
        "Промах дискового кэша маршрутов",
    )
    _CORRIDOR_DISK_HIT = _c(
        "bike_router_corridor_graph_disk_cache_hit_total",
        "Граф коридора загружен с диска (GraphML)",
    )
    _GREEN_JOB_FAILED = _c(
        "bike_router_green_background_failed_total",
        "Фоновый зелёный маршрут завершился с ошибкой",
    )
    _TILE_FAIL = _c(
        "bike_router_tile_download_fail_total",
        "Неудачная загрузка одного спутникового тайла",
    )


def inc_overpass_subretry() -> None:
    _lazy()
    if _OVERPASS_SUBRETRY is not None:
        _OVERPASS_SUBRETRY.inc()


def inc_route_disk_hit() -> None:
    _lazy()
    if _ROUTE_DISK_HIT is not None:
        _ROUTE_DISK_HIT.inc()


def inc_route_disk_miss() -> None:
    _lazy()
    if _ROUTE_DISK_MISS is not None:
        _ROUTE_DISK_MISS.inc()


def inc_corridor_disk_hit() -> None:
    _lazy()
    if _CORRIDOR_DISK_HIT is not None:
        _CORRIDOR_DISK_HIT.inc()


def inc_green_job_failed() -> None:
    _lazy()
    if _GREEN_JOB_FAILED is not None:
        _GREEN_JOB_FAILED.inc()


def inc_tile_fail(n: int = 1) -> None:
    _lazy()
    if _TILE_FAIL is not None and n > 0:
        _TILE_FAIL.inc(n)


def metrics_text() -> bytes:
    try:
        from prometheus_client import generate_latest

        _lazy()
        return generate_latest()
    except ImportError:
        return b"# prometheus_client not installed\n"
