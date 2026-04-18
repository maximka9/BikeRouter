"""Диагностика каталога area_precache для текущего .env.

Движок ищет ``meta.json`` только в **одном** каталоге:
``cache/area_precache/<sha256>/``, где хэш зависит от WKT арены, имени и
**статического fingerprint** (OSM-фильтр, спутник, zoom, буферы, зелень;
без профилей маршрутизации и тепло/стресса). Соседние папки не подхватываются.

Запуск (в Docker)::

    docker compose exec bike-router python -m bike_router.tools.precache_status

или локально из корня репозитория::

    python -m bike_router.tools.precache_status
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    _ensure_pkg_path()
    from bike_router.config import Settings
    from bike_router.services.area_graph_cache import (
        area_green_edges_bundle_is_valid,
        area_green_edges_bundle_meta_path,
        area_green_edges_content_fingerprint,
        area_green_edges_pkl_path,
        load_area_green_edges_bundle_meta,
        area_precache_directory_id,
        area_static_content_fingerprint,
        meta_path,
        precache_area_dir,
        precache_area_root,
        graph_base_path,
        graph_green_path,
    )

    s = Settings()
    print("BIKE_ROUTER_BASE_DIR:", s.base_dir)
    print("cache_dir:", s.cache_dir)
    print("GRAPH_CORRIDOR_MODE:", s.graph_corridor_mode)
    print("PRECACHE_AREA_ENABLED:", s.precache_area_enabled)
    print("has_precache_polygon:", s.has_precache_area_polygon)
    print("use_dynamic_corridor_graph:", s.use_dynamic_corridor_graph)
    print()

    if not s.has_precache_area_polygon:
        print("PRECACHE_AREA_POLYGON_WKT не задан — предкэш арены не используется.")
        return

    fp = area_static_content_fingerprint(s)
    wkt = s.precache_area_polygon_wkt_stripped
    aid = area_precache_directory_id(wkt, s.precache_area_name, fp)
    d = precache_area_dir(s)

    print("Статический fingerprint арены (SHA-256), без scoring/тепла/погоды:")
    print(" ", fp)
    print()
    print("Ожидаемый идентификатор каталога (SHA-256 от wkt+name+fp):")
    print(" ", aid)
    print()
    print("Ожидаемый путь предкэша (именно сюда смотрит load_meta):")
    print(" ", d)
    print()
    meta = meta_path(s)
    print("meta.json:", "да" if meta.is_file() else "НЕТ", f"({meta})")
    gb = graph_base_path(s)
    gg = graph_green_path(s)
    print("graph_base.graphml:", "да" if gb.is_file() else "нет", f"({gb.name})")
    print("graph_green.graphml:", "да" if gg.is_file() else "нет", f"({gg.name})")
    agm = area_green_edges_bundle_meta_path(s)
    agp = area_green_edges_pkl_path(s)
    ag_fp = area_green_edges_content_fingerprint(s)
    ag_side = load_area_green_edges_bundle_meta(s)
    ec_ag = int(ag_side.get("edge_count", -1)) if ag_side else -1
    ag_ok = (
        area_green_edges_bundle_is_valid(s, ec_ag)
        if ec_ag >= 0
        else (agm.is_file() and agp.is_file())
    )
    print()
    print("Статический кэш зелени арены (area_green_edges), fingerprint пайплайна:")
    print(" ", ag_fp)
    print("area_green_edges meta.json:", "да" if agm.is_file() else "нет", f"({agm.name})")
    print("area_green_edges green_edges.pkl:", "да" if agp.is_file() else "нет", f"({agp.name})")
    print(
        "валидность (размер/edge_count/fingerprint):",
        "да" if ag_ok else "нет или неполно",
    )
    print()

    root = precache_area_root(s)
    if not root.is_dir():
        print("Корень area_precache отсутствует:", root)
        return

    subs = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"Все каталоги под {root} ({len(subs)} шт.):")
    for p in subs:
        m = p / "meta.json"
        tag = "meta.json+" if m.is_file() else "без meta.json"
        gbf = (p / "graph_base.graphml").is_file()
        ggf = (p / "graph_green.graphml").is_file()
        mark = "  <-- ТЕКУЩИЙ ОЖИДАЕМЫЙ" if p.name == aid else ""
        print(
            f"  {p.name[:16]}…  {tag}  base={gbf!s:5} green={ggf!s:5}{mark}"
        )

    print()
    print(
        "Если ожидаемый каталог без meta.json, а соседние с meta — "
        "пересоберите precache при текущем .env или выровняйте версии/фингерпринт."
    )


if __name__ == "__main__":
    main()
