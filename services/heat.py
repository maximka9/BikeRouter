"""Тепловая стоимость ребра: ориентация, открытость, тень от зелени/«каньона»."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from ..config import HEAT_COST_SCALE, TimeSlotDef, TIME_SLOTS

# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------


def bearing_deg_from_lonlat(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """Азимут направления сегмента (° от севера по часовой)."""
    dlon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(
        lat2r
    ) * math.cos(dlon)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def edge_bearing_deg_from_geom(geom: Any) -> float:
    """Первый сегмент геометрии ребра (LineString / MultiLineString)."""
    if geom is None:
        return 0.0
    gt = getattr(geom, "geom_type", "") or ""
    coords = []
    try:
        if gt in ("LineString", "LinearRing"):
            coords = list(geom.coords)
        elif gt == "MultiLineString":
            g0 = geom.geoms[0]
            coords = list(g0.coords)
        elif gt == "GeometryCollection":
            for g in geom.geoms:
                if getattr(g, "geom_type", "") == "LineString" and len(g.coords) >= 2:
                    coords = list(g.coords)
                    break
    except Exception:
        coords = []
    if len(coords) < 2:
        return 0.0
    lon1, lat1 = coords[0][0], coords[0][1]
    lon2, lat2 = coords[-1][0], coords[-1][1]
    return bearing_deg_from_lonlat(lon1, lat1, lon2, lat2)


def angle_diff_deg(a: float, b: float) -> float:
    """Минимальная разница углов [0..180]."""
    d = abs((a - b) % 360.0)
    if d > 180.0:
        d = 360.0 - d
    return d


# ---------------------------------------------------------------------------
# Тепловая модель (упрощённая)
# ---------------------------------------------------------------------------


def open_sky_fraction(trees_pct: float, grass_pct: float) -> float:
    """Доля «открытого неба» вдоль коридора (не равно зелёный комфорт)."""
    t = max(0.0, min(100.0, float(trees_pct or 0.0)))
    g = max(0.0, min(100.0, float(grass_pct or 0.0)))
    # Крона сильнее закрывает небо, чем трава
    blocked = min(0.92, (t / 100.0) * 0.72 + (g / 100.0) * 0.18)
    return max(0.08, 1.0 - blocked)


def tree_shadow_mitigation(trees_pct: float) -> float:
    """Ослабление солнечного штрафа за счёт тени от деревьев [0..1]."""
    t = max(0.0, min(100.0, float(trees_pct or 0.0)))
    return min(0.85, (t / 100.0) * 0.9)


def building_canyon_mitigation(tags: Dict[str, Any]) -> float:
    """Прокси тени зданий: узкая улица → больше тени с боков."""
    w = tags.get("width") or tags.get("est:width")
    if w is None:
        return 0.25  # неизвестно — лёгкая «городская» тень
    try:
        wm = float(str(w).replace("m", "").strip().split()[0])
    except (ValueError, IndexError):
        return 0.25
    if wm <= 0:
        return 0.25
    # < 12 м — заметный каньон
    return float(min(0.75, max(0.1, 1.0 - wm / 35.0)))


def solar_exposure_unit(
    bearing_deg: float,
    slot: TimeSlotDef,
    open_frac: float,
    tree_mit: float,
    build_mit: float,
) -> float:
    """Безразмерная экспозиция [0..1] для ребра."""
    sun = slot.sun_azimuth_deg
    ins = max(0.0, min(1.0, slot.insolation_scale))
    # Горизонтальная проекция: |cos(diff)| — фасады/стены упрощённо
    diff = angle_diff_deg(bearing_deg + 90.0, sun)
    direct = abs(math.cos(math.radians(diff)))
    shade = min(0.95, tree_mit * 0.85 + build_mit * 0.55)
    exposed = direct * open_frac * (1.0 - shade)
    return max(0.0, min(1.0, exposed * ins))


def heat_cost_for_edge(
    length_m: float,
    exposure_unit: float,
    *,
    scale: float = HEAT_COST_SCALE,
) -> float:
    """Интегральный тепловой штраф на ребре (масштаб как у physical)."""
    if length_m <= 0 or not math.isfinite(length_m):
        return 0.0
    return float(scale * length_m * exposure_unit)


def exposure_units_for_all_slots(
    bearing_deg: float,
    trees_pct: float,
    grass_pct: float,
    tags: Dict[str, Any],
) -> Dict[str, float]:
    """Безразмерная экспозиция [0..1] по каждому слоту."""
    open_frac = open_sky_fraction(trees_pct, grass_pct)
    tree_m = tree_shadow_mitigation(trees_pct)
    bld_m = building_canyon_mitigation(tags)
    return {
        slot.key: solar_exposure_unit(
            bearing_deg, slot, open_frac, tree_m, bld_m
        )
        for slot in TIME_SLOTS
    }


def heat_costs_for_all_slots(
    length_m: float,
    bearing_deg: float,
    trees_pct: float,
    grass_pct: float,
    tags: Dict[str, Any],
) -> Dict[str, float]:
    """Тепловая стоимость по каждому именованному слоту TIME_SLOTS."""
    exps = exposure_units_for_all_slots(bearing_deg, trees_pct, grass_pct, tags)
    return {k: heat_cost_for_edge(length_m, v) for k, v in exps.items()}


def heat_metrics_for_route(
    lengths: list[float],
    heat_values: list[float],
    exposure_units: list[float],
    *,
    exposure_threshold: float = 0.45,
) -> Dict[str, float]:
    """Суммарный тепловой штраф и длина сильной экспозиции."""
    if not lengths:
        return {
            "total_heat_cost": 0.0,
            "exposed_high_length_m": 0.0,
            "avg_exposure_unit": 0.0,
        }
    total_m = sum(lengths)
    th = sum(heat_values)
    hi_len = sum(
        ln
        for ln, eu in zip(lengths, exposure_units)
        if eu >= exposure_threshold
    )
    w_exp = (
        sum(ln * eu for ln, eu in zip(lengths, exposure_units)) / total_m
        if total_m > 0
        else 0.0
    )
    return {
        "total_heat_cost": float(th),
        "exposed_high_length_m": float(hi_len),
        "avg_exposure_unit": float(w_exp),
    }
