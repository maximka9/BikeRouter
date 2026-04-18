"""Транспортный стресс и воспринимаемая безопасность по тегам OSM (LTS-подобная шкала)."""

from __future__ import annotations

import math
import re
from typing import Any, Dict

from ..config import STRESS_COST_SCALE

# ---------------------------------------------------------------------------
# Парсинг числовых тегов OSM
# ---------------------------------------------------------------------------


def _parse_lanes(val: Any) -> float:
    if val is None:
        return 1.0
    s = str(val).strip()
    if not s or s.lower() in ("none", "no", "n/a"):
        return 1.0
    parts = re.split(r"[|;]", s)
    nums: list[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)", p)
        if m:
            try:
                nums.append(float(m.group(1)))
            except ValueError:
                continue
    if not nums:
        return 1.0
    return float(max(nums))


def _parse_maxspeed_kmh(val: Any) -> float:
    if val is None:
        return 30.0
    s = str(val).strip().lower()
    if not s or s in ("none", "walk", "signals"):
        return 30.0
    if "mph" in s:
        m = re.search(r"(\d+)", s)
        if m:
            return float(m.group(1)) * 1.60934
        return 30.0
    m = re.search(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return 30.0


def _has_protected_bike(tags: Dict[str, Any]) -> bool:
    for k in (
        "cycleway",
        "cycleway:left",
        "cycleway:right",
        "cycleway:both",
        "bicycle_road",
    ):
        v = tags.get(k)
        if v is None:
            continue
        sv = str(v).lower()
        if sv in ("", "no", "none"):
            continue
        if "track" in sv or "lane" in sv or sv in ("yes", "designated", "separate"):
            return True
    if str(tags.get("segregated", "")).lower() == "yes":
        return True
    return False


def _highway_base_stress(hw: str) -> float:
    """Базовый уровень 1–4 (непрерывная шкала) по highway."""
    h = (hw or "unclassified").lower()
    calm = {
        "cycleway": 1.0,
        "path": 1.2,
        "footway": 1.4,
        "pedestrian": 1.3,
        "living_street": 1.2,
        "track": 1.5,
        "steps": 3.5,
    }
    if h in calm:
        return calm[h]
    mid = {
        "residential": 1.6,
        "service": 1.7,
        "unclassified": 1.8,
        "tertiary": 2.0,
        "tertiary_link": 2.0,
    }
    if h in mid:
        return mid[h]
    busy = {
        "secondary": 2.6,
        "secondary_link": 2.6,
        "primary": 3.2,
        "primary_link": 3.2,
        "trunk": 3.6,
        "trunk_link": 3.6,
        "motorway": 4.0,
        "motorway_link": 4.0,
        "construction": 3.0,
    }
    if h in busy:
        return busy[h]
    return 2.0


def lts_from_osm_tags(tags: Dict[str, Any]) -> float:
    """Непрерывная оценка стресса ~1.0 (низкий) … 4.0 (высокий).

    Использует highway, lanes, maxspeed, велоинфраструктуру, перекрёстки.
    """
    hw = str(tags.get("highway") or "unclassified").lower()
    if hw == "steps":
        return 3.5

    base = _highway_base_stress(hw)
    lanes = _parse_lanes(tags.get("lanes"))
    ms = _parse_maxspeed_kmh(tags.get("maxspeed"))

    # Полосы и скорость повышают стресс
    lane_pen = 0.0
    if lanes >= 3:
        lane_pen += 0.35
    elif lanes >= 2:
        lane_pen += 0.15

    speed_pen = 0.0
    if ms >= 70:
        speed_pen += 0.55
    elif ms >= 50:
        speed_pen += 0.35
    elif ms >= 40:
        speed_pen += 0.2

    protected = _has_protected_bike(tags)
    if protected:
        base = max(1.0, base - 0.9)
        lane_pen *= 0.5
        speed_pen *= 0.6

    j = str(tags.get("junction") or "").lower()
    if j in ("roundabout", "circular"):
        base += 0.25
    if tags.get("crossing"):
        base += 0.08
    if str(tags.get("traffic_signals") or "").lower() in ("yes", "signal"):
        base += 0.05

    # Пешеходный тротуар без велосипеда — конфликт
    sw = str(tags.get("sidewalk") or tags.get("sidewalk:both") or "").lower()
    if sw and sw not in ("no", "none") and hw in ("footway", "path") and not protected:
        base += 0.1

    raw = base + lane_pen + speed_pen
    return float(max(1.0, min(4.0, raw)))


def stress_cost_for_edge(
    length_m: float,
    lts: float,
    *,
    scale: float = STRESS_COST_SCALE,
) -> float:
    """Стоимость стресса на ребре (сопоставима с порядком physical weight)."""
    if length_m <= 0 or not math.isfinite(length_m):
        return 0.0
    # (lts-1) даёт 0..3; умножаем на длину
    return float(scale * length_m * ((lts - 1.0) / 3.0))


def stress_metrics_for_route(
    lengths: list[float],
    lts_values: list[float],
    *,
    high_threshold: float = 2.5,
) -> Dict[str, float]:
    """Агрегаты по маршруту: средний/макс LTS, доля «высокого» стресса."""
    if not lengths:
        return {
            "avg_lts": 1.0,
            "max_lts": 1.0,
            "high_stress_length_m": 0.0,
            "high_stress_fraction": 0.0,
            "total_stress_cost": 0.0,
        }
    total_m = sum(lengths)
    tw = sum(l * lv for l, lv in zip(lengths, lts_values))
    high_m = sum(
        l for l, lv in zip(lengths, lts_values) if lv >= high_threshold
    )
    costs = [stress_cost_for_edge(l, lv) for l, lv in zip(lengths, lts_values)]
    return {
        "avg_lts": float(tw / total_m) if total_m > 0 else 1.0,
        "max_lts": float(max(lts_values) if lts_values else 1.0),
        "high_stress_length_m": float(high_m),
        "high_stress_fraction": float(high_m / total_m) if total_m > 0 else 0.0,
        "total_stress_cost": float(sum(costs)),
    }
