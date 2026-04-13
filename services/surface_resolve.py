"""Восстановление покрытия рёбер: OSM → эвристика highway/tracktype → unknown."""

from __future__ import annotations

from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

_TRACKTYPE_SURFACE = {
    "grade1": "paved",
    "grade2": "paved",
    "grade3": "compacted",
    "grade4": "unpaved",
    "grade5": "unpaved",
}

# Типы из OSM_HIGHWAY_FILTER — консервативно для велосипеда
_HIGHWAY_SURFACE = {
    "construction": "compacted",
    "cycleway": "paved",
    "footway": "paved",
    "living_street": "paved",
    "path": "compacted",
    "pedestrian": "paved",
    "platform": "paved",
    "primary": "paved",
    "primary_link": "paved",
    "residential": "paved",
    "secondary": "paved",
    "secondary_link": "paved",
    "service": "paved",
    "steps": "paved",
    "tertiary": "paved",
    "tertiary_link": "paved",
    "track": "unpaved",
    "unclassified": "compacted",
}


def _first_value(val: Any) -> Any:
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _norm_tag(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, list):
        val = val[0] if val else None
    s = str(val).strip().lower()
    return s


def infer_surface_from_tracktype_highway(
    highway: Any, tracktype: Any
) -> Optional[str]:
    """Эвристика, если в OSM нет тега ``surface``."""
    tt = _norm_tag(tracktype)
    if tt in _TRACKTYPE_SURFACE:
        return _TRACKTYPE_SURFACE[tt]
    hw = _norm_tag(highway)
    if not hw:
        return None
    return _HIGHWAY_SURFACE.get(hw)


def resolve_surface_effective(
    surface_osm: str,
    highway: Any,
    tracktype: Any,
) -> str:
    """Итоговая метка для коэффициента profile.surface (нижний регистр)."""
    s = (surface_osm or "").strip().lower()
    if s:
        return s

    inf = infer_surface_from_tracktype_highway(highway, tracktype)
    if inf:
        return inf

    return "unknown"


def build_surface_effective_column(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Добавить ``surface_osm`` (нормализованный тег OSM или пусто)."""
    gdf = edges_gdf.copy()
    if "surface" in gdf.columns:
        gdf["surface_osm"] = gdf["surface"].map(
            lambda x: _norm_tag(_first_value(x))
        )
    else:
        gdf["surface_osm"] = ""

    return gdf


def apply_surface_resolution(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Колонка ``surface_effective`` для весов (без внешних файлов)."""
    gdf = build_surface_effective_column(edges_gdf)

    def _row(r: pd.Series) -> str:
        return resolve_surface_effective(
            str(r.get("surface_osm") or ""),
            r.get("highway"),
            r.get("tracktype"),
        )

    gdf["surface_effective"] = gdf.apply(_row, axis=1)
    return gdf
