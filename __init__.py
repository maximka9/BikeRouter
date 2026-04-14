"""bike_router — оптимизация маршрутов для велосипедистов и пешеходов.

Тяжёлые подмодули (``engine``, ``geopandas``) не импортируются при
``import bike_router`` — только по обращению к имени (PEP 562), чтобы CLI
(``precache_area`` и т.п.) не «висели» без вывода.
"""

from __future__ import annotations

import importlib
from typing import Any, List

__version__ = "3.0.0"
__all__ = [
    "Application",
    "RouteEngine",
    "Settings",
    "CYCLIST",
    "PEDESTRIAN",
    "PROFILES",
]

_LAZY_ATTR: dict[str, tuple[str, str]] = {
    "Application": (".app", "Application"),
    "RouteEngine": (".engine", "RouteEngine"),
    "Settings": (".config", "Settings"),
    "CYCLIST": (".config", "CYCLIST"),
    "PEDESTRIAN": (".config", "PEDESTRIAN"),
    "PROFILES": (".config", "PROFILES"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTR:
        mod_name, attr = _LAZY_ATTR[name]
        mod = importlib.import_module(mod_name, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(__all__) + ["__version__"])
