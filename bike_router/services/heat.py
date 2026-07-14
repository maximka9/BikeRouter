"""Тепловая модель: разложение на тень растительности, зданий и открытый неблагоприятный участок."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..config import HEAT_COST_SCALE, TimeSlotDef, TIME_SLOTS
from .policy_data import load_heat_season_policy

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
    coords: List[Tuple[float, float]] = []
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
# Доли тени / открытости (признаки ребра)
# ---------------------------------------------------------------------------


def open_sky_fraction(trees_pct: float, grass_pct: float) -> float:
    """Доля «открытого неба» вдоль коридора."""
    t = max(0.0, min(100.0, float(trees_pct or 0.0)))
    g = max(0.0, min(100.0, float(grass_pct or 0.0)))
    blocked = min(0.92, (t / 100.0) * 0.72 + (g / 100.0) * 0.18)
    return max(0.08, 1.0 - blocked)


def vegetation_shade_share(trees_pct: float, grass_pct: float) -> float:
    """Доля «тени от растительности» [0..1] (крона сильнее травы)."""
    t = max(0.0, min(100.0, float(trees_pct or 0.0)))
    g = max(0.0, min(100.0, float(grass_pct or 0.0)))
    return float(min(0.95, (t / 100.0) * 0.92 + (g / 100.0) * 0.25))


def covered_share(tags: Dict[str, Any]) -> float:
    """Доля укрытия / навеса / прохода [0..1] по OSM (covered, tunnel, indoor)."""
    if not tags:
        return 0.0
    indoor = tags.get("indoor")
    if indoor in ("yes", "room"):
        return 0.92
    tun = tags.get("tunnel")
    if tun == "building_passage":
        return 0.95
    if tun in ("yes", "true", "1"):
        return 0.82
    cov = tags.get("covered")
    if cov in ("yes", "roof"):
        return 0.80
    if cov in ("arcade", "colonnade"):
        return 0.88
    if cov:
        return 0.52
    return 0.0


def _parse_float_tag(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(str(raw).replace("m", "").replace(",", ".").strip().split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def street_width_proxy_m(tags: Dict[str, Any]) -> float:
    """Прокси ширины улицы по OSM width/lanes/highway, если нет building footprints."""
    if not tags:
        return 18.0
    w = _parse_float_tag(tags.get("width") or tags.get("est:width") or tags.get("est_width"))
    if w is not None and w > 0:
        return float(max(2.0, min(60.0, w)))
    lanes = _parse_float_tag(tags.get("lanes"))
    if lanes is not None and lanes > 0:
        return float(max(3.0, min(60.0, lanes * 3.25 + 3.0)))
    hw = str(tags.get("highway") or "").strip().lower()
    defaults = {
        "footway": 4.0,
        "path": 3.0,
        "cycleway": 4.2,
        "pedestrian": 9.0,
        "living_street": 8.5,
        "residential": 12.0,
        "service": 9.0,
        "unclassified": 13.0,
        "tertiary": 16.0,
        "secondary": 20.0,
        "primary": 24.0,
        "steps": 3.0,
        "track": 4.5,
    }
    return float(defaults.get(hw, 18.0))


def urban_canyon_score(tags: Dict[str, Any]) -> float:
    """0..1: узкая улица/пешеходный коридор как прокси городского каньона."""
    w = street_width_proxy_m(tags)
    width_score = max(0.0, min(1.0, (28.0 - w) / 24.0))
    hw = str((tags or {}).get("highway") or "").strip().lower()
    hw_boost = 0.12 if hw in ("footway", "pedestrian", "living_street", "service") else 0.0
    return float(max(0.0, min(1.0, width_score + hw_boost)))


def building_shade_share(tags: Dict[str, Any]) -> float:
    """Прокси тени зданий по ширине проезда [0..1]."""
    canyon = urban_canyon_score(tags)
    return float(min(0.82, max(0.05, 0.12 + 0.70 * canyon)))


def direct_sun_factor(bearing_deg: float, slot: TimeSlotDef) -> float:
    """Относительная прямая инсоляция фасада/полотна [0..1]."""
    sun = slot.sun_azimuth_deg
    ins = max(0.0, min(1.0, slot.insolation_scale))
    diff = angle_diff_deg(bearing_deg + 90.0, sun)
    direct = abs(math.cos(math.radians(diff)))
    return max(0.0, min(1.0, direct * ins))


def open_unfavorable_unit(
    O: float,
    V: float,
    B: float,
    D: float,
    *,
    veg_weight: float = 0.88,
    bld_weight: float = 0.62,
) -> float:
    """Безразмерная «открытая теплонеблагоприятная» экспозиция [0..1].

    Не сводится к одной зелени: учитываются независимые ослабления от V и B.
    """
    Om = max(0.0, min(1.0, O))
    Vm = max(0.0, min(1.0, V))
    Bm = max(0.0, min(1.0, B))
    Dm = max(0.0, min(1.0, D))
    # совместное ослабление прямого солнца на открытом участке
    shade_combined = 1.0 - (1.0 - Vm * veg_weight) * (1.0 - Bm * bld_weight)
    return max(0.0, min(1.0, Dm * Om * (1.0 - shade_combined)))


def solar_shade_potential(V: float, B: float, C: float) -> float:
    """0..1: совместный потенциал защиты от солнца растительностью/зданиями/укрытиями."""
    Vm = max(0.0, min(1.0, float(V)))
    Bm = max(0.0, min(1.0, float(B)))
    Cm = max(0.0, min(1.0, float(C)))
    return float(1.0 - (1.0 - 0.88 * Vm) * (1.0 - 0.62 * Bm) * (1.0 - 0.90 * Cm))


def legacy_exposure_unit(
    bearing_deg: float,
    slot: TimeSlotDef,
    open_frac: float,
    tree_mit: float,
    build_mit: float,
) -> float:
    """Быстрый прокси (fallback), совместимый с первой версией."""
    sun = slot.sun_azimuth_deg
    ins = max(0.0, min(1.0, slot.insolation_scale))
    diff = angle_diff_deg(bearing_deg + 90.0, sun)
    direct = abs(math.cos(math.radians(diff)))
    shade = min(0.95, tree_mit * 0.85 + build_mit * 0.55)
    exposed = direct * open_frac * (1.0 - shade)
    return max(0.0, min(1.0, exposed * ins))


def wet_surface_edge_slip_factor(surface_effective: Any) -> float:
    """[0..1] риск дискомфорта мокрого покрытия (доп. поправка к heat, не дублируя physical)."""
    s = str(surface_effective or "unknown").strip().lower()
    if s in ("asphalt", "concrete", "paved"):
        return 0.10
    if s in ("sett", "cobblestone", "unhewn_cobblestone"):
        return 0.42
    if s in (
        "gravel",
        "fine_gravel",
        "compacted",
        "ground",
        "earth",
        "grass",
        "unpaved",
        "dirt",
    ):
        return 0.78
    if s in ("wood", "metal"):
        return 0.50
    if s in ("unknown", "", "na"):
        return 0.32
    return 0.38


def heat_cost_for_edge(
    length_m: float,
    exposure_unit: float,
    *,
    scale: float = HEAT_COST_SCALE,
) -> float:
    if length_m <= 0 or not math.isfinite(length_m):
        return 0.0
    return float(scale * length_m * exposure_unit)


def thermal_edge_features(
    bearing_deg: float,
    trees_pct: float,
    grass_pct: float,
    tags: Dict[str, Any],
) -> Tuple[float, float, float, float, float]:
    """O, V, B, C (укрытие), tree_mit — для тепловой маршрутизации."""
    O = open_sky_fraction(trees_pct, grass_pct)
    V = vegetation_shade_share(trees_pct, grass_pct)
    B = building_shade_share(tags)
    C = covered_share(tags)
    tree_mit = min(0.85, (max(0.0, min(100.0, trees_pct)) / 100.0) * 0.9)
    return O, V, B, C, tree_mit


def exposure_units_detailed_all_slots(
    bearing_deg: float,
    trees_pct: float,
    grass_pct: float,
    tags: Dict[str, Any],
    *,
    use_fallback: bool,
) -> Dict[str, float]:
    """Безразмерная тепловая нагрузка по слотам (детально или прокси)."""
    O, V, B, _c, _tm = thermal_edge_features(
        bearing_deg, trees_pct, grass_pct, tags
    )
    out: Dict[str, float] = {}
    for slot in TIME_SLOTS:
        D = direct_sun_factor(bearing_deg, slot)
        if use_fallback:
            open_frac = O
            tree_mit = min(0.85, (float(trees_pct or 0) / 100.0) * 0.9)
            b_mit = B
            out[slot.key] = legacy_exposure_unit(
                bearing_deg, slot, open_frac, tree_mit, b_mit
            )
        else:
            out[slot.key] = open_unfavorable_unit(O, V, B, D)
    return out


def heat_context_multiplier(
    season: str,
    time_slot_key: str,
    air_temperature_c: Optional[float],
    *,
    policy: Optional[Dict[str, Any]] = None,
) -> float:
    """Множитель к уже посчитанному heat_cost на ребре (сезон + опционально T воздуха)."""
    pol = policy if policy is not None else load_heat_season_policy()
    if not pol:
        return 1.0
    season = (season or "summer").strip().lower()
    if season in ("spring", "autumn"):
        season = "spring_autumn"
    slots_map = (pol.get("slot_temperature_factors") or {}).get(season)
    if not isinstance(slots_map, dict):
        slots_map = (pol.get("slot_temperature_factors") or {}).get("summer", {})
    m = float(slots_map.get(time_slot_key, 1.0)) if isinstance(slots_map, dict) else 1.0
    if air_temperature_c is None:
        return max(0.05, min(2.0, m))
    at = pol.get("air_temperature") or {}
    ref = float(at.get("reference_c", 22.0))
    per = float(at.get("per_degree_above", 0.045))
    mx = float(at.get("max_mult", 1.55))
    extra = max(0.0, float(air_temperature_c) - ref)
    tmul = 1.0 + per * extra
    tmul = min(mx, tmul)
    return max(0.05, min(2.5, m * tmul))


def heat_costs_for_all_slots(
    length_m: float,
    bearing_deg: float,
    trees_pct: float,
    grass_pct: float,
    tags: Dict[str, Any],
    *,
    use_fallback: bool = False,
) -> Dict[str, float]:
    exps = exposure_units_detailed_all_slots(
        bearing_deg, trees_pct, grass_pct, tags, use_fallback=use_fallback
    )
    return {k: heat_cost_for_edge(length_m, v) for k, v in exps.items()}


def heat_metrics_for_route(
    lengths: list[float],
    heat_values: list[float],
    exposure_units: list[float],
    *,
    exposure_threshold: float = 0.45,
) -> Dict[str, float]:
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
